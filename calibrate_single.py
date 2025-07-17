# calibrate_single.py
import cv2, yaml, argparse, os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from config import BOARD, DICT          # ← 来自上节
MIN_CORNERS      = 15                   # 每帧最少角点阈值
FRAME_STEP       = 50                    # 每隔 N 帧取 1 帧加速
MAX_REPROJ_ERR   = 1.5                  # 过滤高误差(像素)

def detect_charuco_debug(img, frame_idx, debug_dir):
    """调试版本，保存检测过程"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    debug_vis = img.copy()
    
    # 1) ArUco 检测
    corners, ids, _ = cv2.aruco.detectMarkers(gray, DICT)
    
    if ids is None or len(ids) == 0:
        cv2.putText(debug_vis, "NO ARUCO MARKERS", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)  # 橙色
        cv2.imwrite(str(debug_dir / f"frame_{frame_idx:05d}_no_aruco.png"), debug_vis)
        print(f"帧 {frame_idx}: 未检测到ArUco标记")
        return None
    
    # 绘制ArUco标记
    cv2.aruco.drawDetectedMarkers(debug_vis, corners, ids)
    
    # 2) ChArUco 亚像素内插
    print(f"  - 检测到 {len(ids)} 个ArUco标记，IDs: {ids.flatten()}")
    print(f"  - Board参数: {BOARD.getChessboardSize()}, 字典: {BOARD.getDictionary().bytesList.shape[0]}")
    
    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        markerCorners = corners,
        markerIds     = ids,
        image         = gray,
        board         = BOARD
    )
    
    print(f"  - interpolateCornersCharuco返回: retval={retval}, corners={charuco_corners is not None}, ids={charuco_ids is not None}")
    
    if retval is None or retval < MIN_CORNERS:
        cv2.putText(debug_vis, f"ArUco: {len(ids)}, ChArUco: {retval if retval else 0} < {MIN_CORNERS}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)  # 橙色
        cv2.putText(debug_vis, "INSUFFICIENT CORNERS", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)  # 橙色
        cv2.imwrite(str(debug_dir / f"frame_{frame_idx:05d}_insufficient.png"), debug_vis)
        print(f"帧 {frame_idx}: ArUco={len(ids)}, ChArUco={retval if retval else 0} < {MIN_CORNERS}")
        return None
    
    # 绘制ChArUco角点
    cv2.aruco.drawDetectedCornersCharuco(debug_vis, charuco_corners, charuco_ids, (0,255,0))
    cv2.putText(debug_vis, f"ArUco: {len(ids)}, ChArUco: {len(charuco_corners)}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)  # 橙色
    cv2.putText(debug_vis, "SUCCESS", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)  # 橙色
    cv2.imwrite(str(debug_dir / f"frame_{frame_idx:05d}_success.png"), debug_vis)
    print(f"帧 {frame_idx}: 成功检测 ArUco={len(ids)}, ChArUco={len(charuco_corners)}")
    
    return charuco_corners, charuco_ids

def calibrate_from_video(video_path, out_dir):
    out_dir = Path(out_dir)
    # remove previous output if exists
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
        
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir / "debug"
    debug_dir.mkdir(exist_ok=True)

    all_corners, all_ids, img_size = [], [], None
    success_count = 0
    
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频总帧数: {total}")
    print(f"帧采样间隔: {FRAME_STEP} (每{FRAME_STEP}帧取1帧)")
    print(f"最少角点要求: {MIN_CORNERS}")

    for idx in tqdm(range(total), desc="Scanning video"):
        ret, frame = cap.read()
        if not ret: break
        if idx % FRAME_STEP: continue

        # 使用调试版本的检测函数
        result = detect_charuco_debug(frame, idx, debug_dir)
        
        if result is not None:
            charuco_corners, charuco_ids = result
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)
            img_size = frame.shape[:2][::-1]  # (w,h)
            success_count += 1

    cap.release()
    
    print(f"\n=== 检测统计 ===")
    print(f"成功检测帧数: {success_count}")
    print(f"调试图像已保存到: {debug_dir}")
    
    if len(all_corners) < 5:
        print(f"\n❌ 错误: 有效帧数 ({success_count}) 少于最低要求 (5帧)")
        print("建议:")
        print("1. 检查debug目录中的图像，确认ChArUco板是否清晰可见")
        print("2. 尝试降低MIN_CORNERS参数")
        print("3. 尝试减小FRAME_STEP参数以获取更多帧")
        raise RuntimeError("可用帧太少，无法完成标定")

    # ...existing calibration code...

    # 初标定
    flags = (cv2.CALIB_RATIONAL_MODEL |
             cv2.CALIB_FIX_ASPECT_RATIO)
    
    # OpenCV 4.10.0 API
    ret, K, dist, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, per_view_err = cv2.aruco.calibrateCameraCharucoExtended(
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
        
        # 重新标定
        ret, K, dist, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, _ = cv2.aruco.calibrateCameraCharucoExtended(
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
