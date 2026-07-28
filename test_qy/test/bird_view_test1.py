#python纯代码版,无节点

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

    # =========================
    # 原图上的4个点
    # =========================
    src = np.float32([
        [257, 391],   # 左下
        [315, 395],   # 右下
        [307, 277],   # 右上
        [278, 275]    # 左上
    ])

#当前四点坐标: [[257.0, 391.0], [315.0, 395.0], [307.0, 277.0], [278.0, 275.0]]


    # =========================
    # 目标俯视图点
    # =========================
    dst = np.float32([
        [0, h],
        [w, h],
        [w, 0],
        [0, 0]
    ])

    # =========================
    # 计算透视矩阵
    # =========================
    M = cv2.getPerspectiveTransform(src, dst)

    # =========================
    # 逆透视变换
    # =========================
    bird = cv2.warpPerspective(frame, M, (w, h))

    # =========================
    # 在原图画点
    # =========================
    for p in src:
        cv2.circle(frame, (int(p[0]), int(p[1])), 8, (0, 0, 255), -1)

    # =========================
    # 显示
    # =========================
    cv2.imshow("Original", frame)
    cv2.imshow("Bird View", bird)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()