# calibrate_stereo_images.py
import cv2, yaml, argparse, os
import numpy as np
from pathlib import Path
from config import BOARD, DICT
from calibrate_single import detect_charuco_debug, MIN_CORNERS

def load_intrinsic(yaml_path):
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return (np.asarray(data["camera_matrix"]),
            np.asarray(data["dist_coeffs"]))

def stereo_calibrate_images(l_image, r_image, l_yaml, r_yaml, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir / "debug"
    debug_dir.mkdir(exist_ok=True)
    
    K1, D1 = load_intrinsic(l_yaml)
    K2, D2 = load_intrinsic(r_yaml)

    # 读取左右图像
    frameL = cv2.imread(str(l_image))
    frameR = cv2.imread(str(r_image))
    
    if frameL is None or frameR is None:
        raise ValueError("无法读取图像文件")
    
    print(f"左图像尺寸: {frameL.shape[:2]}")
    print(f"右图像尺寸: {frameR.shape[:2]}")

    # 检测ChArUco角点
    print("检测左图像ChArUco角点...")
    resL = detect_charuco_debug(frameL, 0, debug_dir)
    print("检测右图像ChArUco角点...")
    resR = detect_charuco_debug(frameR, 1, debug_dir)
    
    if resL is None or resR is None:
        raise RuntimeError("其中一张图像未能检测到足够的ChArUco角点")

    chL, idL = resL
    chR, idR = resR
    
    # 取左右公共 id
    common = np.intersect1d(idL.flatten(), idR.flatten())
    print(f"公共角点ID数量: {len(common)}")
    
    if len(common) < MIN_CORNERS:
        raise RuntimeError(f"公共角点数量 ({len(common)}) 少于最低要求 ({MIN_CORNERS})")

    # 依次匹配对应 corner
    obj, ptsL, ptsR = [], [], []
    for c_id in common:
        idxL = np.where(idL == c_id)[0][0]
        idxR = np.where(idR == c_id)[0][0]
        
        # 获取ChArUco角点的3D坐标
        board_corners = BOARD.getChessboardCorners()
        obj.append(board_corners[c_id])
        ptsL.append(chL[idxL][0])  # ChArUco角点是[[x,y]]格式
        ptsR.append(chR[idxR][0])
    
    obj_pts = [np.array(obj, dtype=np.float32)]
    imgL = [np.array(ptsL, dtype=np.float32)]
    imgR = [np.array(ptsR, dtype=np.float32)]

    # 可视化匹配结果
    # 处理不同尺寸的图像 - 将图像调整到相同高度
    h1, w1 = frameL.shape[:2]
    h2, w2 = frameR.shape[:2]
    
    # 选择较小的高度作为目标高度
    target_height = min(h1, h2)
    
    # 按比例缩放图像
    scale1 = target_height / h1
    scale2 = target_height / h2
    
    frameL_resized = cv2.resize(frameL, (int(w1 * scale1), target_height))
    frameR_resized = cv2.resize(frameR, (int(w2 * scale2), target_height))
    
    print(f"调整后左图尺寸: {frameL_resized.shape[:2]}")
    print(f"调整后右图尺寸: {frameR_resized.shape[:2]}")
    
    vis = np.hstack([frameL_resized, frameR_resized])
    
    # 在拼接图像上绘制匹配的角点
    for i, c_id in enumerate(common):
        # 左图角点 (需要按比例缩放)
        pt_l = tuple(map(int, [ptsL[i][0] * scale1, ptsL[i][1] * scale1]))
        cv2.circle(vis, pt_l, 5, (0, 255, 0), -1)
        cv2.putText(vis, str(c_id), (pt_l[0]+10, pt_l[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 右图角点 (需要偏移frameL_resized的宽度并按比例缩放)
        pt_r = tuple(map(int, [ptsR[i][0] * scale2 + frameL_resized.shape[1], ptsR[i][1] * scale2]))
        cv2.circle(vis, pt_r, 5, (0, 255, 0), -1)
        cv2.putText(vis, str(c_id), (pt_r[0]+10, pt_r[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 连线显示匹配关系
        cv2.line(vis, pt_l, pt_r, (255, 0, 0), 1)
    
    cv2.imwrite(str(out_dir/"stereo_matches.png"), vis)
    print(f"匹配可视化保存到: {out_dir/'stereo_matches.png'}")

    print(f"用于标定的角点对数: {len(common)}")
    
    # 双目标定
    flags = (cv2.CALIB_FIX_INTRINSIC |
             cv2.CALIB_USE_INTRINSIC_GUESS |
             cv2.CALIB_RATIONAL_MODEL)
    
    # 注意：使用原始左图像的尺寸进行标定
    image_size = (frameL.shape[1], frameL.shape[0])
    print(f"标定使用的图像尺寸: {image_size}")
    
    rms, K1_out, D1_out, K2_out, D2_out, R, T, E, F = cv2.stereoCalibrate(
        objectPoints   = obj_pts,
        imagePoints1   = imgL,
        imagePoints2   = imgR,
        cameraMatrix1  = K1,
        distCoeffs1    = D1,
        cameraMatrix2  = K2,
        distCoeffs2    = D2,
        imageSize      = image_size,
        R              = None,
        T              = None,
        flags          = flags,
        criteria       = (cv2.TERM_CRITERIA_MAX_ITER +
                          cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    )
    print(f"Stereo RMS = {rms:.4f}px")

    # 保存外参
    with open(out_dir/"stereo.yaml", "w") as f:
        yaml.dump(dict(
            # 变换关系: P_right = R * P_left + T
            R = R.tolist(),  # 从左相机到右相机的旋转 (3x3)
            T = T.ravel().tolist(),  # 从左相机到右相机的平移 (3x1)
            rms = float(rms),
            essential = E.tolist(),
            fundamental = F.tolist(),
            left_image_size = [frameL.shape[1], frameL.shape[0]],
            right_image_size = [frameR.shape[1], frameR.shape[0]],
            # 相机在世界坐标系(左相机坐标系)中的位置和姿态
            right_camera_position_in_left = (-R.T @ T).ravel().tolist(),
            right_camera_rotation_in_left = R.T.tolist()
        ), f)
    print(f"[√] R、T 已写入 {out_dir/'stereo.yaml'}")
    
    # 打印变换关系说明
    print("\n=== 双目标定结果说明 ===")
    print("变换关系: P_right = R * P_left + T")
    print(f"R (左→右旋转):\n{R}")
    print(f"T (左→右平移): {T.ravel()}")
    print(f"右相机在左相机坐标系中的位置: {(-R.T @ T).ravel()}")
    print(f"基线距离: {np.linalg.norm(T):.4f} 米")

    # 生成PLY可视化文件
    left_size = (frameL.shape[1], frameL.shape[0])
    right_size = (frameR.shape[1], frameR.shape[0])
    generate_camera_ply(K1, K2, R, T, left_size, right_size, out_dir)
    
    return R, T

def generate_camera_ply(K1, K2, R, T, left_image_size, right_image_size, out_dir):
    """
    生成相机frustum的PLY文件
    
    数学关系说明：
    - R, T 是从左相机坐标系到右相机坐标系的变换
    - P_right = R * P_left + T
    - 左相机作为世界坐标系原点 (0,0,0)
    - 右相机在世界坐标系中的位置: -R^T * T
    - 右相机在世界坐标系中的姿态: R^T
    
    Args:
        K1: 左相机内参矩阵
        K2: 右相机内参矩阵
        R: 从左相机到右相机的旋转矩阵
        T: 从左相机到右相机的平移向量
        left_image_size: 左图像尺寸 (width, height)
        right_image_size: 右图像尺寸 (width, height)
        out_dir: 输出目录
    """
    # 定义frustum的深度
    frustum_depth = 0.1  # 30cm深度
    
    def camera_frustum_points(K, image_size, R_cam=None, t_cam=None):
        """生成单个相机的frustum角点"""
        width, height = image_size
        
        # 图像角点在像素坐标系
        corners_2d = np.array([
            [0, 0],
            [width, 0], 
            [width, height],
            [0, height]
        ], dtype=np.float32)
        
        # 转换到归一化相机坐标系
        corners_2d_hom = np.hstack([corners_2d, np.ones((4, 1))])
        K_inv = np.linalg.inv(K)
        corners_norm = (K_inv @ corners_2d_hom.T).T
        
        # 生成3D frustum点
        points = []
        
        # 相机中心 (原点)
        points.append([0, 0, 0])
        
        # frustum远端的四个角点
        for corner in corners_norm:
            x, y, z = corner
            # 投影到深度frustum_depth的平面
            points.append([x * frustum_depth, y * frustum_depth, frustum_depth])
        
        points = np.array(points)
        
        # 应用相机的旋转和平移
        if R_cam is not None:
            points = (R_cam @ points.T).T
        if t_cam is not None:
            points += t_cam.reshape(1, 3)
            
        return points
    
    # 相机1 (左相机，作为世界坐标系原点)
    # 位置: (0, 0, 0), 姿态: 单位矩阵
    cam1_points = camera_frustum_points(K1, left_image_size)
    
    # 相机2 (右相机)
    # 从左相机坐标系到右相机坐标系: P_right = R * P_left + T
    # 因此右相机在世界坐标系(左相机坐标系)中的变换:
    # - 位置: -R^T * T (右相机中心在左相机坐标系中的位置)
    # - 姿态: R^T (右相机坐标轴在左相机坐标系中的方向)
    cam2_position = -R.T @ T
    cam2_orientation = R.T
    cam2_points = camera_frustum_points(K2, right_image_size, cam2_orientation, cam2_position)
    
    # 合并所有点
    all_points = np.vstack([cam1_points, cam2_points])
    
    # 定义面 (四个三角形组成一个四棱锥)
    def camera_faces(offset=0):
        """生成单个相机frustum的面"""
        faces = []
        center = offset + 0  # 相机中心点索引
        corners = [offset + 1, offset + 2, offset + 3, offset + 4]  # 四个角点
        
        # 四个三角形面
        faces.append([center, corners[0], corners[1]])  # 上面
        faces.append([center, corners[1], corners[2]])  # 右面  
        faces.append([center, corners[2], corners[3]])  # 下面
        faces.append([center, corners[3], corners[0]])  # 左面
        
        return faces
    
    cam1_faces = camera_faces(0)
    cam2_faces = camera_faces(5)  # 相机2的点从索引5开始
    all_faces = cam1_faces + cam2_faces
    
    # 写入PLY文件
    ply_file = out_dir / "cameras.ply"
    with open(ply_file, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(all_points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {len(all_faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        
        # 写入顶点 (前5个点是相机1，后5个是相机2)
        for i, point in enumerate(all_points):
            if i < 5:  # 相机1 - 红色
                color = "255 0 0"
            else:  # 相机2 - 蓝色
                color = "0 0 255"
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {color}\n")
        
        # 写入面
        for face in all_faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    
    print(f"[√] 相机可视化PLY文件已保存到: {ply_file}")
    print("可以使用MeshLab、CloudCompare或Blender打开查看")

# CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="从两张图片进行双目标定")
    ap.add_argument("--left_image",  required=True, help="左相机图片路径")
    ap.add_argument("--right_image", required=True, help="右相机图片路径")
    ap.add_argument("--left_yaml",   required=True, help="左相机内参文件")
    ap.add_argument("--right_yaml",  required=True, help="右相机内参文件")
    ap.add_argument("--out",         default="stereo_images_out", help="输出目录")
    args = ap.parse_args()
    
    stereo_calibrate_images(args.left_image, args.right_image,
                           args.left_yaml, args.right_yaml, args.out)
