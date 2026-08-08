#!/usr/bin/env python3
"""
Roadblock acceptance test v2.

Changes from v1:
1) Start menu: run all or only a selected test.
2) Position tests ask for the ACTUAL placement coordinates before sampling.
3) TTL is timed from raw YOLO roadblock disappearance, not from Enter.
4) Odom test first verifies that /odom actually changed; if the car was lifted or
   odom did not move, the result is INVALID instead of a false FAIL.
5) Never publishes /cmd_vel and never changes runtime parameters.
"""

import csv
import math
import os
import signal
import subprocess
import threading
import time
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

import rclpy
from ai_msgs.msg import PerceptionTargets
from nav_msgs.msg import Odometry
from roadblock_interfaces.msg import RoadblockArray
from rclpy.node import Node

RB_TOPIC = "/roadblock_ground_array"
ODOM_TOPIC = "/odom"
DET_TOPIC = "/hobot_dnn_detection"
LOG_ROOT = Path("/root/intelligent_car_ws/test_logs/roadblock_acceptance")

POS_PASS = 0.05
POS_WARN = 0.07
PROP_PASS = 0.05
PROP_WARN = 0.08

def yaw(q):
    return math.atan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y*q.y + q.z*q.z))

def cmean(v):
    return math.atan2(mean([math.sin(x) for x in v]), mean([math.cos(x) for x in v]))

def grade(e, p, w):
    return "PASS" if e <= p else "WARN" if e <= w else "FAIL"

def ask_float(label, default):
    s = input(f"{label} [默认 {default:+.3f}]: ").strip()
    if not s:
        return float(default)
    return float(s)

class N(rclpy.node.Node):
    def __init__(self, log_dir):
        super().__init__("roadblock_acceptance_test")
        self.lock = threading.Lock()
        self.test = "BACKGROUND"
        self.rb = deque(maxlen=30000)
        self.od = deque(maxlen=30000)
        self.det = deque(maxlen=30000)

        self.fs = open(log_dir/"samples.csv","w",newline="",buffering=1)
        self.fe = open(log_dir/"events.csv","w",newline="",buffering=1)
        self.fo = open(log_dir/"odom.csv","w",newline="",buffering=1)
        self.fr = open(log_dir/"roadblock.csv","w",newline="",buffering=1)
        self.fd = open(log_dir/"detections.csv","w",newline="",buffering=1)

        self.ws = csv.writer(self.fs); self.ws.writerow(["test_name","timestamp","id","x","y"])
        self.we = csv.writer(self.fe); self.we.writerow(["timestamp","event"])
        self.wo = csv.writer(self.fo); self.wo.writerow(["timestamp","x","y","yaw"])
        self.wr = csv.writer(self.fr); self.wr.writerow(["timestamp","frame_id","id","x","y"])
        self.wd = csv.writer(self.fd); self.wd.writerow(["timestamp","roadblock_count"])

        self.create_subscription(RoadblockArray,RB_TOPIC,self.rb_cb,10)
        self.create_subscription(Odometry,ODOM_TOPIC,self.od_cb,50)
        self.create_subscription(PerceptionTargets,DET_TOPIC,self.det_cb,10)

    def rb_cb(self,m):
        t=time.time()
        obs=[(int(o.id),float(o.x),float(o.y)) for o in m.obstacles]
        with self.lock:
            self.rb.append((t,obs))
            tn=self.test
            if obs:
                for i,x,y in obs:
                    self.wr.writerow([f"{t:.6f}",m.header.frame_id,i,f"{x:.6f}",f"{y:.6f}"])
                    if tn!="BACKGROUND":
                        self.ws.writerow([tn,f"{t:.6f}",i,f"{x:.6f}",f"{y:.6f}"])
            else:
                self.wr.writerow([f"{t:.6f}",m.header.frame_id,"","",""])

    def od_cb(self,m):
        t=time.time()
        p=m.pose.pose.position
        z=yaw(m.pose.pose.orientation)
        with self.lock:
            self.od.append((t,float(p.x),float(p.y),z))
            self.wo.writerow([f"{t:.6f}",f"{p.x:.6f}",f"{p.y:.6f}",f"{z:.6f}"])

    def det_cb(self,m):
        t=time.time()
        count=0
        try:
            for target in m.targets:
                if getattr(target,"type","")=="roadblock":
                    count += 1
        except Exception:
            count=0
        with self.lock:
            self.det.append((t,count))
            self.wd.writerow([f"{t:.6f}",count])

    def event(self,s):
        with self.lock:
            self.we.writerow([f"{time.time():.6f}",s])
        print("[EVENT]",s)

    def pubs(self,topic):
        return len(self.get_publishers_info_by_topic(topic))

    def frames(self,a,b):
        with self.lock:
            return [(t,list(o)) for t,o in self.rb if a<=t<=b]

    def odoms(self,a,b):
        with self.lock:
            return [x for x in self.od if a<=x[0]<=b]

    def dets(self,a,b):
        with self.lock:
            return [x for x in self.det if a<=x[0]<=b]

    def close(self):
        for f in [self.fs,self.fe,self.fo,self.fr,self.fd]:
            try: f.close()
            except: pass

class Runner:
    def __init__(self,n,d):
        self.n,self.d=n,d
        self.res=[]
        self.lp=None
        self.lf=None

    def add(self,name,status,detail):
        self.res.append((name,status,detail))
        print(f"[{status}] {name}: {detail}")

    def ask(self,text,skip=True):
        print("\n"+"="*72)
        print(text.strip())
        print("\nEnter=继续"+("；s=跳过" if skip else "")+"；q=结束")
        print("="*72)
        a=input("> ").strip().lower()
        if a=="q":
            raise KeyboardInterrupt
        return "skip" if skip and a=="s" else "go"

    def cmd(self,args):
        try:
            p=subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=6)
            return p.returncode,p.stdout.strip()
        except Exception as e:
            return 999,str(e)

    def topic_type(self,t):
        rc,o=self.cmd(["ros2","topic","type",t])
        return o if rc==0 else "unknown"

    def start_localizer(self):
        if self.n.pubs(RB_TOPIC):
            return
        self.lf=open(self.d/"localizer.log","a",buffering=1)
        self.lp=subprocess.Popen(
            ["ros2","launch","roadblock_localization","roadblock_localization.launch.py"],
            stdout=self.lf,stderr=subprocess.STDOUT,start_new_session=True,text=True
        )
        self.n.event(f"LOCALIZER_STARTED pid={self.lp.pid}")
        time.sleep(2)

    def stop_localizer(self):
        if self.lp and self.lp.poll() is None:
            try:
                os.killpg(self.lp.pid,signal.SIGINT); self.lp.wait(timeout=5)
            except Exception:
                try: os.killpg(self.lp.pid,signal.SIGTERM)
                except Exception: pass
        if self.lf:
            try:self.lf.close()
            except:pass

    def test0(self):
        self.start_localizer()
        while True:
            a,b,c=self.n.pubs(ODOM_TOPIC),self.n.pubs(DET_TOPIC),self.n.pubs(RB_TOPIC)
            print(f"{ODOM_TOPIC}: {a} pub, {self.topic_type(ODOM_TOPIC)}")
            print(f"{DET_TOPIC}: {b} pub, {self.topic_type(DET_TOPIC)}")
            print(f"{RB_TOPIC}: {c} pub, {self.topic_type(RB_TOPIC)}")
            rc,o=self.cmd(["ros2","interface","show","roadblock_interfaces/msg/Roadblock"])
            iface=rc==0 and all(s in o for s in ["int32 id","float32 x","float32 y"])
            if a and b and c and iface:
                self.add("TEST0 ROS chain","PASS","输入/输出话题与Roadblock接口正常")
                return True
            if input("启动缺失节点后 Enter 重试；q退出：").strip().lower()=="q":
                self.add("TEST0 ROS chain","FAIL","ROS链路未就绪")
                return False

    def collect(self,name,sec):
        self.n.test=name
        self.n.event(name+"_START")
        a=time.time()
        time.sleep(sec)
        b=time.time()
        self.n.event(name+"_END")
        self.n.test="BACKGROUND"
        return self.n.frames(a,b),self.n.odoms(a,b)

    @staticmethod
    def byid(frames):
        d=defaultdict(list)
        for t,obs in frames:
            for i,x,y in obs:
                d[i].append((t,x,y))
        return d

    @staticmethod
    def st(samples):
        xs=[x[1] for x in samples]
        ys=[x[2] for x in samples]
        return mean(xs),mean(ys),pstdev(xs) if len(xs)>1 else 0.0,pstdev(ys) if len(ys)>1 else 0.0

    def custom_position(self,name,default_x,default_y,edge=False):
        print("\n请先按你实际摆放的位置输入真实坐标。")
        xt=ask_float("真实 X / m",default_x)
        yt=ask_float("真实 Y / m",default_y)
        if self.ask(f"""
{name}

请将单个锥桶底座中心实际放在：
  X = {xt:.3f} m
  Y = {yt:+.3f} m

脚本将以你刚输入的实际位置作为真值，不再假定固定坐标。
""")=="skip":
            self.add(name,"SKIP","user skipped")
            return
        time.sleep(1)
        fr,_=self.collect(name,3)
        d=self.byid(fr)
        if not d:
            self.add(name,"WARN" if edge else "FAIL","NO_RELIABLE_POSITION")
            return
        cand=[]
        for i,s in d.items():
            mx,my,sx,sy=self.st(s)
            e=math.hypot(mx-xt,my-yt)
            cand.append((e,i,len(s),mx,my,sx,sy))
        e,i,n,mx,my,sx,sy=min(cand)
        status=grade(e,POS_PASS,POS_WARN)
        if len(d)>1 and status=="PASS":
            status="WARN"
        if yt>0 and my<=0:
            status="FAIL"
        if yt<0 and my>=0:
            status="FAIL"
        self.add(name,status,
                 f"truth=({xt:.3f},{yt:.3f}), id={i}, n={n}, "
                 f"mean=({mx:.3f},{my:.3f})m, std=({sx:.3f},{sy:.3f})m, "
                 f"error={e*100:.1f}cm, ids={sorted(d)}")

    def test1(self):
        self.custom_position("TEST1 Center",0.80,0.00)

    def test2(self):
        self.custom_position("TEST2A Left",0.85,+0.35,edge=True)
        self.custom_position("TEST2B Right",0.85,-0.35,edge=True)

    def test3(self):
        if self.ask("""
TEST3 Stable ID

同时摆两个锥桶：
  A≈(0.70,+0.20)
  B≈(0.95,-0.20)

保持10秒不动。
""")=="skip":
            self.add("TEST3 Stable ID","SKIP","user skipped")
            return
        fr,_=self.collect("TEST3 Stable ID",10)
        lc,rc,ids=Counter(),Counter(),set()
        for _,obs in fr:
            for i,x,y in obs:
                ids.add(i)
                if y>0: lc[i]+=1
                elif y<0: rc[i]+=1
        if not lc or not rc:
            self.add("TEST3 Stable ID","FAIL",f"左右目标不完整 left={dict(lc)} right={dict(rc)}")
            return
        li,ln=lc.most_common(1)[0]
        ri,rn=rc.most_common(1)[0]
        ld=ln/sum(lc.values())
        rd=rn/sum(rc.values())
        dm=min(ld,rd)
        status="PASS" if li!=ri and dm>=.90 else "WARN" if li!=ri and dm>=.70 else "FAIL"
        if len(ids)>4:
            status="FAIL"
        elif len(ids)>2 and status=="PASS":
            status="WARN"
        self.add("TEST3 Stable ID",status,
                 f"left id={li} dom={ld:.1%}, right id={ri} dom={rd:.1%}, ids={sorted(ids)}")

    def wait_raw_loss(self,start_t,timeout=15.0,stable_absence=0.30):
        """Wait until raw YOLO roadblock detections have been absent continuously."""
        deadline=time.time()+timeout
        absent_since=None
        while time.time()<deadline:
            now=time.time()
            ds=self.n.dets(max(start_t,now-0.25),now)
            seen=any(c>0 for _,c in ds)
            if seen:
                absent_since=None
            else:
                # Require that detection messages are actually arriving.
                if ds:
                    if absent_since is None:
                        absent_since=now
                    if now-absent_since>=stable_absence:
                        return absent_since
            time.sleep(0.03)
        return None

    def test4(self):
        if self.ask("""
TEST4 TTL

只保留一个锥桶在画面中央并稳定识别。

这版不再要求你 0.5 秒内从电脑跑到锥桶。
建议你先站到锥桶附近准备好，再回车开始。
脚本会自动观察原始 YOLO：
只有检测到 roadblock 真正消失后，TTL 才开始计时。
""")=="skip":
            self.add("TEST4 TTL","SKIP","user skipped")
            return

        time.sleep(1)
        fr,_=self.collect("TEST4 PRE",2)
        d=self.byid(fr)
        if not d:
            self.add("TEST4 TTL","FAIL","预采样没有Track")
            return
        tid=max(d,key=lambda i:len(d[i]))

        self.ask(f"""
当前目标 id={tid}。

你现在可以慢慢走到锥桶旁边准备。
准备好后按 Enter，然后把锥桶移出相机画面/完全遮挡。

计时起点不是 Enter，而是 /hobot_dnn_detection 中 roadblock
真正连续消失约0.3秒的时刻。
""",skip=False)

        enter_t=time.time()
        self.n.event("TEST4_WAIT_RAW_LOSS")
        loss_t=self.wait_raw_loss(enter_t,timeout=15.0,stable_absence=0.30)
        if loss_t is None:
            self.add("TEST4 TTL","INVALID","15秒内没有观察到原始YOLO roadblock稳定消失；可能仍有roadblock误检")
            return

        self.n.event("TEST4_RAW_LOSS")
        print("[INFO] 已检测到原始 YOLO roadblock 消失，开始测 TTL...")
        end_t=loss_t+5.0
        while time.time()<end_t:
            time.sleep(0.05)

        frames=self.n.frames(loss_t,end_t)
        appearances=[]
        pos=[]
        for t,obs in frames:
            for i,x,y in obs:
                if i==tid:
                    appearances.append(t)
                    pos.append((x,y))

        ttl=(max(appearances)-loss_t) if appearances else 0.0
        jump=max([math.hypot(pos[k][0]-pos[k-1][0],pos[k][1]-pos[k-1][1])
                  for k in range(1,len(pos))] or [0.0])

        if 1.0<=ttl<=3.2:
            status="PASS"
        elif 0.3<=ttl<1.0 or 3.2<ttl<=4.0:
            status="WARN"
        elif ttl>4.0:
            status="FAIL"
        else:
            status="WARN"
        if jump>0.30:
            status="FAIL"

        self.add("TEST4 TTL",status,
                 f"id={tid}, measured_from_raw_yolo_loss={ttl:.2f}s, max_jump={jump:.3f}m")

    @staticmethod
    def pose(od):
        if not od:
            return None
        return mean([x[1] for x in od]),mean([x[2] for x in od]),cmean([x[3] for x in od])

    @staticmethod
    def b2o(x,y,p):
        rx,ry,a=p
        c,s=math.cos(a),math.sin(a)
        return rx+c*x-s*y,ry+s*x+c*y

    @staticmethod
    def o2b(x,y,p):
        rx,ry,a=p
        dx,dy=x-rx,y-ry
        c,s=math.cos(a),math.sin(a)
        return c*dx+s*dy,-s*dx+c*dy

    def test5(self):
        if self.ask("""
TEST5 Odom consistency

只保留一个固定锥桶在前方约 X=0.80m,Y=0。
车辆和锥桶先静止。

重要：
必须让车轮实际驱动车辆运动。
不能抬起小车搬动，否则轮式 odom 不会记录真实位移。

本脚本不会发布 /cmd_vel。
""")=="skip":
            self.add("TEST5 Odom","SKIP","user skipped")
            return

        time.sleep(1)
        fr,od=self.collect("TEST5 BEFORE",2)
        d=self.byid(fr)
        p0=self.pose(od)
        if not d or not p0:
            self.add("TEST5 Odom","FAIL","移动前缺Track或odom")
            return

        tid=max(d,key=lambda i:len(d[i]))
        mx,my,_,_=self.st(d[tid])
        xo,yo=self.b2o(mx,my,p0)

        self.ask(f"""
移动前：
  id={tid}
  obstacle=({mx:.3f},{my:.3f})

现在请用底盘/车轮实际驱动车辆缓慢前进约0.10m。
锥桶保持不动。
停稳后按 Enter。
""",skip=False)

        time.sleep(.5)
        fr2,od2=self.collect("TEST5 AFTER",2)
        d2=self.byid(fr2)
        p1=self.pose(od2)
        if not d2 or not p1:
            self.add("TEST5 Odom","FAIL","移动后缺Track或odom")
            return

        dx=p1[0]-p0[0]
        dy=p1[1]-p0[1]
        trans=math.hypot(dx,dy)
        dyaw=math.atan2(math.sin(p1[2]-p0[2]),math.cos(p1[2]-p0[2]))

        # Prevent the previous false FAIL when the user physically lifted the car.
        if trans<0.05 and abs(dyaw)<math.radians(3.0):
            self.add("TEST5 Odom","INVALID",
                     f"/odom only changed {trans*100:.1f}cm, yaw={math.degrees(dyaw):.1f}deg；"
                     "没有形成有效底盘运动，不能判断odom传播")
            return

        px,py=self.o2b(xo,yo,p1)

        if tid in d2:
            aid=tid
            st=self.st(d2[tid])
            changed=False
        else:
            candidates=[]
            for i,s in d2.items():
                z=self.st(s)
                candidates.append((math.hypot(z[0]-px,z[1]-py),i,z))
            if not candidates:
                self.add("TEST5 Odom","FAIL","移动后没有Track")
                return
            _,aid,st=min(candidates)
            changed=True

        ax,ay=st[0],st[1]
        e=math.hypot(ax-px,ay-py)
        status=grade(e,PROP_PASS,PROP_WARN)
        if changed:
            status="FAIL"

        self.add("TEST5 Odom",status,
                 f"odom_move={trans*100:.1f}cm, yaw={math.degrees(dyaw):.1f}deg, "
                 f"id {tid}->{aid}, predicted=({px:.3f},{py:.3f}), "
                 f"actual=({ax:.3f},{ay:.3f}), error={e*100:.1f}cm")

    def menu(self):
        print("""
================ Roadblock 验收测试 v2 ================

1  全部测试
2  只测定位（中心 + 左右）
3  只测 Stable ID
4  只测 TTL
5  只测 Odom
6  单点自定义定位
q  退出
========================================================
""")
        return input("选择：").strip().lower()

    def summary(self):
        statuses=[s for _,s,_ in self.res]
        if any(s=="FAIL" for s in statuses):
            final="FAIL"
        elif any(s in ("INVALID","SKIP") for s in statuses):
            final="INCOMPLETE"
        elif any(s=="WARN" for s in statuses):
            final="PASS_WITH_WARNINGS"
        else:
            final="PASS"

        lines=["="*72,"ROADBLOCK ACCEPTANCE TEST V2","="*72]
        for n,s,d in self.res:
            lines += [f"{n:<30} {s}",f"  {d}"]
        lines += ["-"*72,f"FINAL RESULT: {final}",f"Log dir: {self.d}","="*72]
        text="\n".join(lines)
        print("\n"+text)
        (self.d/"summary.txt").write_text(text+"\n",encoding="utf-8")

    def run(self):
        if not self.test0():
            self.summary()
            return
        choice=self.menu()
        if choice=="q":
            return
        if choice=="1":
            self.test1(); self.test2(); self.test3(); self.test4(); self.test5()
        elif choice=="2":
            self.test1(); self.test2()
        elif choice=="3":
            self.test3()
        elif choice=="4":
            self.test4()
        elif choice=="5":
            self.test5()
        elif choice=="6":
            self.custom_position("CUSTOM POSITION",0.80,0.00)
        else:
            self.add("MENU","INVALID",f"未知选项: {choice}")
        self.summary()

def main():
    d=LOG_ROOT/datetime.now().strftime("%Y%m%d_%H%M%S")
    d.mkdir(parents=True,exist_ok=True)
    (d/"localizer.log").touch()
    print("[INFO] 日志目录:",d)
    print("[INFO] 本脚本只观测，不发布 /cmd_vel，不修改参数。")

    rclpy.init()
    n=N(d)
    r=Runner(n,d)
    th=threading.Thread(target=rclpy.spin,args=(n,),daemon=True)
    th.start()

    try:
        r.run()
    except KeyboardInterrupt:
        n.event("USER_ABORT")
        r.add("RUN","SKIP","用户中止")
        r.summary()
    finally:
        r.stop_localizer()
        n.close()
        try:
            n.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass
        th.join(timeout=1)

if __name__=="__main__":
    main()
