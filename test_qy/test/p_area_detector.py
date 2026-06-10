import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera open failed")
    exit()

# =========================
# 逆透视四点
# =========================
src = np.float32([
    [196, 377],   # 左下
    [320, 373],   # 右下
    [307, 294],   # 右上
    [231, 298]    # 左上
])

selected_idx = -1
drag_radius = 20


# =========================
# 鼠标拖动四点
# =========================
def on_mouse(event, x, y, flags, param):
    global selected_idx

    if event == cv2.EVENT_LBUTTONDOWN:

        for i, p in enumerate(src):

            if np.hypot(p[0]-x, p[1]-y) < drag_radius:
                selected_idx = i
                break

    elif event == cv2.EVENT_MOUSEMOVE:

        if selected_idx != -1:
            src[selected_idx] = [x, y]

    elif event == cv2.EVENT_LBUTTONUP:

        selected_idx = -1

        print("当前四点坐标:")
        print(src.tolist())


cv2.namedWindow("Original")
cv2.setMouseCallback("Original", on_mouse)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    h, w = frame.shape[:2]

    # =========================
    # Bird View
    # =========================

    dst = np.float32([
        [0, h],
        [w, h],
        [w, 0],
        [0, 0]
    ])

    M = cv2.getPerspectiveTransform(src, dst)

    bird = cv2.warpPerspective(frame, M, (w, h))

    # =========================
    # 提取米色区域
    # =========================

    hsv = cv2.cvtColor(bird, cv2.COLOR_BGR2HSV)

    lower_beige = np.array([5, 20, 150])
    upper_beige = np.array([25, 120, 255])

    mask = cv2.inRange(
        hsv,
        lower_beige,
        upper_beige
    )

    # 去噪

    kernel = np.ones((5,5),np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # =========================
    # 找最大轮廓
    # =========================

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) > 0:

        c = max(contours, key=cv2.contourArea)

        area = cv2.contourArea(c)

        if area > 1000:

            rect = cv2.minAreaRect(c)

            (cx, cy) = rect[0]

            box = cv2.boxPoints(rect)
            box = np.int32(box)

            cv2.drawContours(
                bird,
                [box],
                0,
                (0,255,0),
                2
            )

            cv2.circle(
                bird,
                (int(cx), int(cy)),
                6,
                (0,0,255),
                -1
            )

            cv2.putText(
                bird,
                f"Center: ({int(cx)}, {int(cy)})",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,0,255),
                2
            )

            print(
                f"P区域中心: ({int(cx)}, {int(cy)})"
            )

    # =========================
    # 原图显示
    # =========================

    for p in src:

        cv2.circle(
            frame,
            (int(p[0]), int(p[1])),
            8,
            (0,0,255),
            -1
        )

    cv2.imshow("Original", frame)
    cv2.imshow("Bird View", bird)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()