import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera open failed")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    src = np.float32([
        [264, 427],
        [335, 422],
        [315, 325],
        [271, 329]
    ])
    dst = np.float32([
        [0, h],
        [w, h],
        [w, 0],
        [0, 0]
    ])

    matrix = cv2.getPerspectiveTransform(src, dst)
    bird = cv2.warpPerspective(frame, matrix, (w, h))

    for p in src:
        cv2.circle(frame, (int(p[0]), int(p[1])), 8, (0, 0, 255), -1)

    cv2.imshow("Original", frame)
    cv2.imshow("Bird View", bird)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
