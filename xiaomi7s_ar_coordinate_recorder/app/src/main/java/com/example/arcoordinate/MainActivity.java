package com.example.arcoordinate;

import android.Manifest;
import android.app.Activity;
import android.content.ContentValues;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.net.Uri;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.os.Environment;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import com.google.ar.core.ArCoreApk;
import com.google.ar.core.Camera;
import com.google.ar.core.Config;
import com.google.ar.core.Frame;
import com.google.ar.core.Pose;
import com.google.ar.core.Session;
import com.google.ar.core.TrackingFailureReason;
import com.google.ar.core.TrackingState;
import com.google.ar.core.exceptions.CameraNotAvailableException;
import com.google.ar.core.exceptions.UnavailableException;

import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicBoolean;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

public class MainActivity extends Activity implements GLSurfaceView.Renderer {
    private static final int CAMERA_PERMISSION_REQUEST = 1001;
    private static final long CONTINUOUS_PERIOD_NS = 100_000_000L; // 10 Hz

    private GLSurfaceView glView;
    private TextView statusText;
    private TextView xText;
    private TextView yText;
    private TextView zText;
    private TextView infoText;
    private Button continuousButton;

    private Session session;
    private boolean installRequested = false;
    private int cameraTextureId = -1;
    private boolean cameraTextureBound = false;

    private volatile Pose originPose;
    private final AtomicBoolean resetOriginRequested = new AtomicBoolean(false);
    private volatile boolean tracking = false;
    private volatile float currentX = 0f;
    private volatile float currentY = 0f;
    private volatile float currentZ = 0f;
    private volatile long currentFrameTimestampNs = 0L;

    private final List<CoordinateRecord> records = new ArrayList<>();
    private volatile boolean continuousRecording = false;
    private long lastContinuousRecordNs = 0L;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        buildUi();

        if (!hasCameraPermission()) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST);
        }
    }

    private void buildUi() {
        FrameLayout root = new FrameLayout(this);
        root.setBackgroundColor(Color.rgb(245, 245, 245));

        glView = new GLSurfaceView(this);
        glView.setEGLContextClientVersion(2);
        glView.setPreserveEGLContextOnPause(true);
        glView.setRenderer(this);
        glView.setRenderMode(GLSurfaceView.RENDERMODE_CONTINUOUSLY);
        glView.setVisibility(View.VISIBLE);
        FrameLayout.LayoutParams glParams = new FrameLayout.LayoutParams(2, 2);
        glParams.gravity = Gravity.TOP | Gravity.START;
        root.addView(glView, glParams);

        ScrollView scrollView = new ScrollView(this);
        FrameLayout.LayoutParams scrollParams = new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT);
        root.addView(scrollView, scrollParams);

        LinearLayout panel = new LinearLayout(this);
        panel.setOrientation(LinearLayout.VERTICAL);
        panel.setPadding(dp(24), dp(24), dp(24), dp(24));
        scrollView.addView(panel, new ScrollView.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT));

        TextView title = new TextView(this);
        title.setText("AR 坐标记录器");
        title.setTextSize(28f);
        title.setTextColor(Color.BLACK);
        title.setGravity(Gravity.CENTER_HORIZONTAL);
        panel.addView(title);

        TextView subtitle = new TextView(this);
        subtitle.setText("启动后首个稳定跟踪点 = (0, 0, 0)，单位：米");
        subtitle.setTextSize(16f);
        subtitle.setTextColor(Color.DKGRAY);
        subtitle.setPadding(0, dp(8), 0, dp(18));
        subtitle.setGravity(Gravity.CENTER_HORIZONTAL);
        panel.addView(subtitle);

        statusText = makeText("状态：等待 ARCore", 18f, Color.rgb(180, 80, 0));
        panel.addView(statusText);

        xText = makeCoordinateText("X = +0.000 m");
        yText = makeCoordinateText("Y = +0.000 m");
        zText = makeCoordinateText("Z = +0.000 m");
        panel.addView(xText);
        panel.addView(yText);
        panel.addView(zText);

        TextView axes = makeText("坐标轴：+X 向初始相机右侧，+Y 向上，+Z 向初始相机后方（相机朝向为 -Z）", 14f, Color.DKGRAY);
        axes.setPadding(0, dp(8), 0, dp(16));
        panel.addView(axes);

        Button recordButton = makeButton("记录当前位置");
        recordButton.setOnClickListener(v -> recordCurrentPosition("manual"));
        panel.addView(recordButton);

        continuousButton = makeButton("开始连续记录（10 Hz）");
        continuousButton.setOnClickListener(v -> {
            continuousRecording = !continuousRecording;
            continuousButton.setText(continuousRecording ? "停止连续记录" : "开始连续记录（10 Hz）");
            Toast.makeText(this, continuousRecording ? "连续记录已开始" : "连续记录已停止", Toast.LENGTH_SHORT).show();
        });
        panel.addView(continuousButton);

        Button resetButton = makeButton("当前位置设为新原点");
        resetButton.setOnClickListener(v -> {
            resetOriginRequested.set(true);
            Toast.makeText(this, "将在下一个稳定 AR 帧重置原点", Toast.LENGTH_SHORT).show();
        });
        panel.addView(resetButton);

        Button exportButton = makeButton("导出 CSV 到 Downloads");
        exportButton.setOnClickListener(v -> exportCsv());
        panel.addView(exportButton);

        infoText = makeText("已记录：0 点", 15f, Color.DKGRAY);
        infoText.setPadding(0, dp(12), 0, 0);
        panel.addView(infoText);

        TextView note = makeText(
                "说明：这里是 ARCore 局部相对坐标，不是 GPS 经纬度。退出应用或重新建立 Session 后世界坐标不会自动保持一致。",
                14f,
                Color.DKGRAY);
        note.setPadding(0, dp(18), 0, 0);
        panel.addView(note);

        setContentView(root);
    }

    private TextView makeText(String text, float sp, int color) {
        TextView tv = new TextView(this);
        tv.setText(text);
        tv.setTextSize(sp);
        tv.setTextColor(color);
        tv.setPadding(0, dp(6), 0, dp(6));
        return tv;
    }

    private TextView makeCoordinateText(String text) {
        TextView tv = makeText(text, 32f, Color.BLACK);
        tv.setGravity(Gravity.CENTER_HORIZONTAL);
        tv.setPadding(0, dp(10), 0, dp(10));
        return tv;
    }

    private Button makeButton(String text) {
        Button button = new Button(this);
        button.setText(text);
        button.setTextSize(17f);
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT);
        params.topMargin = dp(10);
        button.setLayoutParams(params);
        return button;
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private boolean hasCameraPermission() {
        return checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
    }

    @Override
    protected void onResume() {
        super.onResume();
        resumeArIfReady();
    }

    private void resumeArIfReady() {
        if (!hasCameraPermission()) {
            return;
        }

        try {
            if (session == null) {
                ArCoreApk.InstallStatus installStatus = ArCoreApk.getInstance().requestInstall(this, !installRequested);
                if (installStatus == ArCoreApk.InstallStatus.INSTALL_REQUESTED) {
                    installRequested = true;
                    setStatus("状态：等待安装/更新 Google Play Services for AR", false);
                    return;
                }

                session = new Session(this);
                Config config = new Config(session);
                config.setFocusMode(Config.FocusMode.AUTO);
                config.setUpdateMode(Config.UpdateMode.LATEST_CAMERA_IMAGE);
                session.configure(config);
                cameraTextureBound = false;
            }

            session.resume();
            glView.onResume();
            setStatus("状态：正在寻找稳定视觉跟踪，请移动平板", false);
        } catch (UnavailableException e) {
            setStatus("ARCore 不可用：" + e.getClass().getSimpleName() + " - " + safeMessage(e), true);
        } catch (CameraNotAvailableException e) {
            setStatus("相机不可用：" + safeMessage(e), true);
        } catch (Exception e) {
            setStatus("启动失败：" + e.getClass().getSimpleName() + " - " + safeMessage(e), true);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (glView != null) {
            glView.onPause();
        }
        if (session != null) {
            session.pause();
        }
    }

    @Override
    protected void onDestroy() {
        if (session != null) {
            session.close();
            session = null;
        }
        super.onDestroy();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION_REQUEST) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                resumeArIfReady();
            } else {
                setStatus("需要相机权限才能进行 ARCore 定位", true);
            }
        }
    }

    @Override
    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        GLES20.glClearColor(0f, 0f, 0f, 1f);
        cameraTextureId = createExternalTexture();
        cameraTextureBound = false;
    }

    @Override
    public void onSurfaceChanged(GL10 gl, int width, int height) {
        GLES20.glViewport(0, 0, width, height);
        if (session != null) {
            session.setDisplayGeometry(getWindowManager().getDefaultDisplay().getRotation(), width, height);
        }
    }

    @Override
    public void onDrawFrame(GL10 gl) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT | GLES20.GL_DEPTH_BUFFER_BIT);
        Session localSession = session;
        if (localSession == null || cameraTextureId == -1) {
            return;
        }

        try {
            if (!cameraTextureBound) {
                localSession.setCameraTextureName(cameraTextureId);
                cameraTextureBound = true;
            }

            Frame frame = localSession.update();
            currentFrameTimestampNs = frame.getTimestamp();
            Camera camera = frame.getCamera();

            if (camera.getTrackingState() == TrackingState.TRACKING) {
                Pose cameraPose = camera.getPose();
                if (originPose == null || resetOriginRequested.getAndSet(false)) {
                    originPose = cameraPose;
                }

                Pose relativePose = originPose.inverse().compose(cameraPose);
                currentX = relativePose.tx();
                currentY = relativePose.ty();
                currentZ = relativePose.tz();
                tracking = true;

                if (continuousRecording) {
                    long nowNs = SystemClock.elapsedRealtimeNanos();
                    if (nowNs - lastContinuousRecordNs >= CONTINUOUS_PERIOD_NS) {
                        lastContinuousRecordNs = nowNs;
                        addRecordFromRenderThread("continuous", nowNs, currentFrameTimestampNs, currentX, currentY, currentZ);
                    }
                }

                runOnUiThread(() -> {
                    xText.setText(String.format(Locale.US, "X = %+.3f m", currentX));
                    yText.setText(String.format(Locale.US, "Y = %+.3f m", currentY));
                    zText.setText(String.format(Locale.US, "Z = %+.3f m", currentZ));
                    statusText.setText("状态：TRACKING");
                    statusText.setTextColor(Color.rgb(0, 130, 60));
                });
            } else {
                tracking = false;
                TrackingFailureReason reason = camera.getTrackingFailureReason();
                String reasonText = reason == TrackingFailureReason.NONE ? "等待跟踪" : reason.toString();
                runOnUiThread(() -> setStatus("状态：" + camera.getTrackingState() + " / " + reasonText, false));
            }
        } catch (CameraNotAvailableException e) {
            tracking = false;
            runOnUiThread(() -> setStatus("相机断开：" + safeMessage(e), true));
        } catch (Exception e) {
            tracking = false;
            runOnUiThread(() -> setStatus("AR 帧错误：" + e.getClass().getSimpleName() + " - " + safeMessage(e), true));
        }
    }

    private int createExternalTexture() {
        int[] textures = new int[1];
        GLES20.glGenTextures(1, textures, 0);
        int textureId = textures[0];
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
        return textureId;
    }

    private void recordCurrentPosition(String mode) {
        if (!tracking) {
            Toast.makeText(this, "当前不是 TRACKING，暂不能记录", Toast.LENGTH_SHORT).show();
            return;
        }
        long nowNs = SystemClock.elapsedRealtimeNanos();
        synchronized (records) {
            records.add(new CoordinateRecord(System.currentTimeMillis(), nowNs, currentFrameTimestampNs, currentX, currentY, currentZ, mode));
            infoText.setText("已记录：" + records.size() + " 点");
        }
        Toast.makeText(this, String.format(Locale.US, "已记录 (%.3f, %.3f, %.3f) m", currentX, currentY, currentZ), Toast.LENGTH_SHORT).show();
    }

    private void addRecordFromRenderThread(String mode, long androidTimeNs, long frameTimestampNs, float x, float y, float z) {
        int count;
        synchronized (records) {
            records.add(new CoordinateRecord(System.currentTimeMillis(), androidTimeNs, frameTimestampNs, x, y, z, mode));
            count = records.size();
        }
        int finalCount = count;
        runOnUiThread(() -> infoText.setText("已记录：" + finalCount + " 点"));
    }

    private void exportCsv() {
        List<CoordinateRecord> snapshot;
        synchronized (records) {
            snapshot = new ArrayList<>(records);
        }
        if (snapshot.isEmpty()) {
            Toast.makeText(this, "还没有记录任何坐标", Toast.LENGTH_SHORT).show();
            return;
        }

        StringBuilder csv = new StringBuilder();
        csv.append("index,wall_time_iso,android_elapsed_ns,ar_frame_timestamp_ns,mode,x_m,y_m,z_m\n");
        SimpleDateFormat iso = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSZ", Locale.US);
        for (int i = 0; i < snapshot.size(); i++) {
            CoordinateRecord r = snapshot.get(i);
            csv.append(i + 1).append(',')
                    .append(iso.format(new Date(r.wallTimeMs))).append(',')
                    .append(r.androidElapsedNs).append(',')
                    .append(r.arFrameTimestampNs).append(',')
                    .append(r.mode).append(',')
                    .append(String.format(Locale.US, "%.6f", r.x)).append(',')
                    .append(String.format(Locale.US, "%.6f", r.y)).append(',')
                    .append(String.format(Locale.US, "%.6f", r.z)).append('\n');
        }

        String fileName = "ar_coordinates_" + new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date()) + ".csv";
        ContentValues values = new ContentValues();
        values.put(MediaStore.MediaColumns.DISPLAY_NAME, fileName);
        values.put(MediaStore.MediaColumns.MIME_TYPE, "text/csv");
        values.put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS + "/ARCoordinateRecorder");

        Uri uri = null;
        try {
            uri = getContentResolver().insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values);
            if (uri == null) {
                throw new IllegalStateException("MediaStore insert returned null");
            }
            try (OutputStream out = getContentResolver().openOutputStream(uri)) {
                if (out == null) {
                    throw new IllegalStateException("openOutputStream returned null");
                }
                out.write(csv.toString().getBytes(StandardCharsets.UTF_8));
            }
            Toast.makeText(this, "已导出：Downloads/ARCoordinateRecorder/" + fileName, Toast.LENGTH_LONG).show();
        } catch (Exception e) {
            if (uri != null) {
                getContentResolver().delete(uri, null, null);
            }
            Toast.makeText(this, "导出失败：" + safeMessage(e), Toast.LENGTH_LONG).show();
        }
    }

    private void setStatus(String text, boolean error) {
        if (statusText == null) return;
        statusText.setText(text);
        statusText.setTextColor(error ? Color.rgb(180, 0, 0) : Color.rgb(180, 80, 0));
    }

    private String safeMessage(Throwable t) {
        return t.getMessage() == null ? "无详细信息" : t.getMessage();
    }

    private static class CoordinateRecord {
        final long wallTimeMs;
        final long androidElapsedNs;
        final long arFrameTimestampNs;
        final float x;
        final float y;
        final float z;
        final String mode;

        CoordinateRecord(long wallTimeMs, long androidElapsedNs, long arFrameTimestampNs,
                         float x, float y, float z, String mode) {
            this.wallTimeMs = wallTimeMs;
            this.androidElapsedNs = androidElapsedNs;
            this.arFrameTimestampNs = arFrameTimestampNs;
            this.x = x;
            this.y = y;
            this.z = z;
            this.mode = mode;
        }
    }
}
