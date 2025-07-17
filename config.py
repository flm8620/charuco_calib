# config.py
import cv2
import numpy as np

# ──────── ChArUco 板参数 ────────
CHARUCO_ROWS      = 9              # 棋盘行数
CHARUCO_COLS      = 12               # 棋盘列数
SQUARE_LENGTH_M   = 0.015           # 单格边长 15 mm
MARKER_LENGTH_M   = 0.01125         # ArUco 方码边长 11.25 mm

# 5×5 字典，前 200 个码用于 9×12 ChArUco‑200
DICT_ID = cv2.aruco.DICT_5X5_250
DICT    = cv2.aruco.getPredefinedDictionary(DICT_ID)

# 构建 ChArUco Board 对象
BOARD = cv2.aruco.CharucoBoard(
        size          = (CHARUCO_COLS, CHARUCO_ROWS),
        squareLength  = SQUARE_LENGTH_M,
        markerLength  = MARKER_LENGTH_M,
        dictionary    = DICT
)
