# calibrate_stereo.py
import cv2, yaml, argparse, os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from config import BOARD, DICT
from calibrate_single import detect_charuco, MIN_CORNERS, FRAME_STEP

def load_intrinsic(yaml_path):
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return (np.asarray(data["camera_matrix"]),
            np.asarray(data["dist_coeffs"]))

def stereo_calibrate(l_video, r_video, l_yaml, r_yaml, out_dir):
    out_dir = Path(out_dir); (out_dir/"pairs").mkdir(parents=True, exist_ok=True)
    K1, D1 = load_intrinsic(l_yaml)
    K2, D2 = load_intrinsic(r_yaml)

    capL, capR = cv2.VideoCapture(l_video), cv2.VideoCapture(r_video)
    total = int(min(capL.get(cv2.CAP_PROP_FRAME_COUNT),
                    capR.get(cv2.CAP_PROP_FRAME_COUNT)))

    obj_pts, imgL, imgR = [], [], []
    for idx in tqdm(range(total), desc="Stereo scan"):
        retL, frameL = capL.read(); retR, frameR = capR.read()
        if not (retL and retR): break
        if idx % FRAME_STEP: continue

        resL = detect_charuco(frameL, K1, D1)
        resR = detect_charuco(frameR, K2, D2)
        if resL is None or resR is None: continue

        _, _, chL, idL = resL
        _, _, chR, idR = resR
        # 取左右公共 id
        common = np.intersect1d(idL.flatten(), idR.flatten())
        if len(common) < MIN_CORNERS: continue

        # 依次匹配对应 corner
        obj, ptsL, ptsR = [], [], []
        for c_id in common:
            idxL = np.where(idL == c_id)[0][0]
            idxR = np.where(idR == c_id)[0][0]
            obj.append(BOARD.chessboardCorners[c_id])
            ptsL.append(chL[idxL])
            ptsR.append(chR[idxR])
        obj_pts.append(np.array(obj, dtype=np.float32))
        imgL.append(np.array(ptsL, dtype=np.float32))
        imgR.append(np.array(ptsR, dtype=np.float32))

        # 可视化
        vis = np.hstack([frameL, frameR])
        cv2.imwrite(str(out_dir/"pairs"/f"pair_{idx:05d}.png"), vis)

    print(f"有效帧对: {len(obj_pts)}")
    flags = (cv2.CALIB_FIX_INTRINSIC |
             cv2.CALIB_USE_INTRINSIC_GUESS |
             cv2.CALIB_RATIONAL_MODEL)
    rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        objectPoints   = obj_pts,
        imagePoints1   = imgL,
        imagePoints2   = imgR,
        cameraMatrix1  = K1,
        distCoeffs1    = D1,
        cameraMatrix2  = K2,
        distCoeffs2    = D2,
        imageSize      = (frameL.shape[1], frameL.shape[0]),
        R              = None,
        T              = None,
        flags          = flags,
        criteria       = (cv2.TERM_CRITERIA_MAX_ITER +
                          cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    )
    print(f"Stereo RMS = {rms:.4f}px")

    with open(out_dir/"stereo.yaml", "w") as f:
        yaml.dump(dict(
            R = R.tolist(),
            T = T.ravel().tolist(),
            rms = float(rms),
            essential = E.tolist(),
            fundamental = F.tolist()
        ), f)
    print(f"[√] R、T 已写入 {out_dir/'stereo.yaml'}")
    return R, T

# CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--left_video",  required=True)
    ap.add_argument("--right_video", required=True)
    ap.add_argument("--left_yaml",   required=True)
    ap.add_argument("--right_yaml",  required=True)
    ap.add_argument("--out",         default="stereo_out")
    args = ap.parse_args()
    stereo_calibrate(args.left_video, args.right_video,
                     args.left_yaml, args.right_yaml, args.out)
