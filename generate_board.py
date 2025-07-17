#!/usr/bin/env python3
# generate_board.py - 生成ChArUco标定板图像
import cv2
import numpy as np
from config import BOARD, CHARUCO_ROWS, CHARUCO_COLS, SQUARE_LENGTH_M, MARKER_LENGTH_M, DICT_ID

def generate_charuco_board():
    """生成ChArUco标定板图像用于对比"""
    
    # 设置输出图像大小 (像素)
    # 假设每个方格在图像中是50像素
    pixels_per_square = 50
    img_width = CHARUCO_COLS * pixels_per_square
    img_height = CHARUCO_ROWS * pixels_per_square
    
    print(f"生成ChArUco标定板:")
    print(f"  - 尺寸: {CHARUCO_COLS} x {CHARUCO_ROWS} (列 x 行)")
    print(f"  - 方格边长: {SQUARE_LENGTH_M*1000:.1f} mm")
    print(f"  - ArUco标记边长: {MARKER_LENGTH_M*1000:.1f} mm") 
    print(f"  - ArUco字典: {DICT_ID}")
    print(f"  - 输出图像尺寸: {img_width} x {img_height} 像素")
    
    # 生成标定板图像
    img = BOARD.generateImage((img_width, img_height))
    
    # 添加信息文本
    info_img = img.copy()
    if len(info_img.shape) == 2:  # 灰度图转RGB
        info_img = cv2.cvtColor(info_img, cv2.COLOR_GRAY2BGR)
    
    # 在图像底部添加配置信息
    text_y = img_height - 60
    cv2.putText(info_img, f"ChArUco Board: {CHARUCO_COLS}x{CHARUCO_ROWS}", 
                (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(info_img, f"Square: {SQUARE_LENGTH_M*1000:.1f}mm, Marker: {MARKER_LENGTH_M*1000:.1f}mm", 
                (10, text_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(info_img, f"Dict: {DICT_ID}", 
                (10, text_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # 保存图像
    cv2.imwrite("charuco_board_reference.png", img)
    cv2.imwrite("charuco_board_with_info.png", info_img)
    
    print(f"✓ 标定板图像已保存:")
    print(f"  - charuco_board_reference.png (纯净版)")
    print(f"  - charuco_board_with_info.png (带配置信息)")
    
    # 打印一些ArUco ID信息
    print(f"\n标定板中的ArUco标记ID范围:")
    print(f"  - 总共有 {(CHARUCO_COLS-1)*(CHARUCO_ROWS-1)} 个ArUco标记")
    print(f"  - ID从 0 到 {(CHARUCO_COLS-1)*(CHARUCO_ROWS-1)-1}")
    
    # 检查字典容量
    dict_size = cv2.aruco.getPredefinedDictionary(DICT_ID).bytesList.shape[0]
    print(f"  - 使用的字典包含 {dict_size} 个标记")
    
    if (CHARUCO_COLS-1)*(CHARUCO_ROWS-1) > dict_size:
        print(f"  ⚠️  警告: 需要的标记数量超过了字典容量!")
    else:
        print(f"  ✓ 字典容量足够")

if __name__ == "__main__":
    generate_charuco_board()
