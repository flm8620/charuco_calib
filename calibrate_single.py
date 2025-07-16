# calibrate_single.py
import cv2, yaml, argparse, os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from config import BOARD, DICT          # ← 来自上节
MIN_CORNERS      = 20                   # 每帧最少角点阈值
FRAME_STEP       = 3                    # 每隔 N 帧取 1 帧加速
MAX_REPROJ_ERR   = 1.5                  # 过滤高误差(像素)

def detect_charuco(img, camera_matrix=None, dist=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 1) ArUco 检测
    corners, ids, _ = cv2.aruco.detectMarkers(gray, DICT)
    if ids is None or len(ids) == 0:
        return None
    # 2) 可选 refine，提高鲁棒性
    if camera_matrix is not None:
        cv2.aruco.refineDetectedMarkers(
            image         = gray,
            board         = BOARD,
            detectedCorners= corners,
            detectedIds   = ids,
            rejectedCorners = [],
            cameraMatrix  = camera_matrix,
            distCoeffs    = dist
        )
    # 3) ChArUco 亚像素内插
    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        markerCorners = corners,
        markerIds     = ids,
        image         = gray,
        board         = BOARD
    )
    if retval is None or retval < MIN_CORNERS:
        return None
    return corners, ids, charuco_corners, charuco_ids

def calibrate_from_video(video_path, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "frames"; vis_dir.mkdir(exist_ok=True)

    all_corners, all_ids, img_size = [], [], None
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for idx in tqdm(range(total), desc="Scanning video"):
        ret, frame = cap.read();  # 到文件尾 ret==False
        if not ret: break
        if idx % FRAME_STEP: continue

        res = detect_charuco(frame)
        if res is None: continue
        corners, ids, charuco_corners, charuco_ids = res

        # 保存可视化
        vis = frame.copy()
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)
        cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids, (0,255,0))
        cv2.imwrite(str(vis_dir / f"raw_{idx:05d}.png"), vis)

        all_corners.append(charuco_corners)
        all_ids.append(charuco_ids)
        img_size = frame.shape[:2][::-1]  # (w,h)

    cap.release()
    if len(all_corners) < 5:
        raise RuntimeError("可用帧太少，无法完成标定")

    # 初标定
    flags = (cv2.CALIB_RATIONAL_MODEL |
             cv2.CALIB_FIX_ASPECT_RATIO)
    ret, K, dist, rvecs, tvecs, per_view_err = cv2.aruco.calibrateCameraCharucoExtended(
        charucoCorners  = all_corners,
        charucoIds      = all_ids,
        board           = BOARD,
        imageSize       = img_size,
        cameraMatrix    = None,
        distCoeffs      = None,
        flags           = flags
    )
    print(f"Initial RMS = {ret:.4f}px")

    # 过滤重投影误差过大的帧
    keep = [i for i,e in enumerate(per_view_err) if e < MAX_REPROJ_ERR]
    if len(keep) < len(per_view_err):
        all_corners = [all_corners[i] for i in keep]
        all_ids     = [all_ids[i] for i in keep]
        ret, K, dist, rvecs, tvecs, _ = cv2.aruco.calibrateCameraCharucoExtended(
            charucoCorners  = all_corners,
            charucoIds      = all_ids,
            board           = BOARD,
            imageSize       = img_size,
            cameraMatrix    = None,
            distCoeffs      = None,
            flags           = flags
        )
        print(f"Refined RMS = {ret:.4f}px")

    # 保存内参
    with open(out_dir / "intrinsic.yaml", "w") as f:
        yaml.dump(dict(
            image_width  = img_size[0],
            image_height = img_size[1],
            camera_matrix= K.tolist(),
            dist_coeffs  = dist.ravel().tolist(),
            rms          = float(ret)
        ), f)
    print(f"[√] 内参已写入 {out_dir/'intrinsic.yaml'}")

    return K, dist
# CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out",   default="single_out")
    args = ap.parse_args()
    calibrate_from_video(args.video, args.out)
