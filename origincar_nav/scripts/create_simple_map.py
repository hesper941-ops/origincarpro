#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from PIL import Image, ImageDraw

# =========================
# 地图参数
# =========================
MAP_LENGTH_M = 6.0        # x方向长度，单位 m
MAP_WIDTH_M = 8.0         # y方向宽度，单位 m
RESOLUTION = 0.05         # m/pixel
BORDER_THICKNESS_M = 0.20 # 黑色边界厚度，单位 m

OUTPUT_DIR = os.path.expanduser(
    "~/intelligent_car_ws/src/origincar_nav/maps"
)

MAP_NAME = "map"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    width_px = int(MAP_LENGTH_M / RESOLUTION)
    height_px = int(MAP_WIDTH_M / RESOLUTION)
    border_px = max(1, int(BORDER_THICKNESS_M / RESOLUTION))

    print("========== 开始生成地图 ==========")
    print(f"地图实际范围: x=0~{MAP_LENGTH_M} m, y=0~{MAP_WIDTH_M} m")
    print(f"地图像素尺寸: {width_px} x {height_px}")
    print(f"地图分辨率: {RESOLUTION} m/pixel")
    print(f"边界厚度: {BORDER_THICKNESS_M} m = {border_px} px")

    # 白色 255 = 可通行
    # 黑色 0 = 障碍物
    img = Image.new("L", (width_px, height_px), 255)
    draw = ImageDraw.Draw(img)

    # 黑色边界
    draw.rectangle(
        [0, 0, width_px - 1, height_px - 1],
        outline=0,
        width=border_px
    )

    pgm_path = os.path.join(OUTPUT_DIR, MAP_NAME + ".pgm")
    yaml_path = os.path.join(OUTPUT_DIR, MAP_NAME + ".yaml")

    img.save(pgm_path)

    # 左下角作为 map 原点
    origin_x = -1.0
    origin_y = -1.0

    yaml_content = f"""image: {MAP_NAME}.pgm
mode: trinary
resolution: {RESOLUTION}
origin: [{origin_x:.3f}, {origin_y:.3f}, 0.0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.25
"""

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print("========== 地图生成完成 ==========")
    print(f"PGM:  {pgm_path}")
    print(f"YAML: {yaml_path}")
    print("")
    print(yaml_content)


if __name__ == "__main__":
    main()