import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera open failed")
    exit()

src = np.float32([
    [200, 400],
    [440, 400],
    [360, 260],
    [280, 260]
])

selected_idx = -1
drag_radius = 20


def draw_dashed_line(img, pt1, pt2, color, thickness=2, dash_length=20, gap_length=10):
    pt1 = np.array(pt1, dtype=np.float32)
    pt2 = np.array(pt2, dtype=np.float32)
    line_len = np.linalg.norm(pt2 - pt1)
    if line_len < 1:
        return
    direction = (pt2 - pt1) / line_len
    d = 0.0
    while d < line_len:
        start = (pt1 + direction * d).astype(int)
        end = (pt1 + direction * min(d + dash_length, line_len)).astype(int)
        cv2.line(img, tuple(start), tuple(end), color, thickness)
        d += dash_length + gap_length


def on_mouse(event, x, y, flags, param):
    del flags
    global selected_idx
    pts = param
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, p in enumerate(pts):
            if np.hypot(p[0] - x, p[1] - y) <= drag_radius:
                selected_idx = i
                break
    elif event == cv2.EVENT_MOUSEMOVE and selected_idx != -1:
        pts[selected_idx] = [x, y]
    elif event == cv2.EVENT_LBUTTONUP:
        selected_idx = -1
        print("Current points:", src.tolist())


cv2.namedWindow("Original")
cv2.setMouseCallback("Original", on_mouse, src)
print("Drag the points to align ROI, press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    dst = np.float32([
        [0, h],
        [w, h],
        [w, 0],
        [0, 0]
    ])

    matrix = cv2.getPerspectiveTransform(src, dst)
    bird = cv2.warpPerspective(frame, matrix, (w, h))

    for i, p in enumerate(src):
        cv2.circle(frame, (int(p[0]), int(p[1])), 8, (0, 0, 255), -1)
        cv2.putText(
            frame,
            f"{i}:{int(p[0])},{int(p[1])}",
            (int(p[0]) + 10, int(p[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1
        )

    for i in range(len(src)):
        pt1 = tuple(src[i].astype(int))
        pt2 = tuple(src[(i + 1) % len(src)].astype(int))
        draw_dashed_line(frame, pt1, pt2, (0, 255, 0), thickness=2, dash_length=15, gap_length=10)

    cv2.imshow("Original", frame)
    cv2.imshow("Bird View", bird)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
