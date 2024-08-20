import cv2
import numpy as np
import os
import json
import glob
from tkinter import Tk
from tkinter.filedialog import askdirectory, askopenfilename
from datetime import datetime

# GUIで画像フォルダとパラメータファイルを選択
Tk().withdraw()
image_folder_left = askdirectory(title="左カメラのチェスボード画像が保存されているフォルダを選択してください")
param_file_left = askopenfilename(title="左カメラのパラメータファイルを選択してください", filetypes=[("JSON files", "*.json")])
image_folder_right = askdirectory(title="右カメラのチェスボード画像が保存されているフォルダを選択してください")
param_file_right = askopenfilename(title="右カメラのパラメータファイルを選択してください", filetypes=[("JSON files", "*.json")])

# カメラパラメータを読み込む
with open(param_file_left, 'r') as f:
    camera_params_left = json.load(f)
camera_matrix_left = np.array(camera_params_left['camera_matrix'])
dist_coeffs_left = np.array(camera_params_left['dist_coeffs'])

with open(param_file_right, 'r') as f:
    camera_params_right = json.load(f)
camera_matrix_right = np.array(camera_params_right['camera_matrix'])
dist_coeffs_right = np.array(camera_params_right['dist_coeffs'])

# 画像のパスを取得
image_paths_left = sorted(glob.glob(os.path.join(image_folder_left, '*.jpg')))
image_paths_right = sorted(glob.glob(os.path.join(image_folder_right, '*.jpg')))

# チェスボードのサイズ
chessboard_size = (6, 9)
square_size = 0.04

# 3Dポイントの準備
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size  # 実際のサイズにスケール

# 3Dポイントと2Dポイントのリスト
objpoints = []
imgpoints_left = []
imgpoints_right = []

# 画像ごとの処理
for img_path_left, img_path_right in zip(image_paths_left, image_paths_right):
    img_left = cv2.imread(img_path_left)
    img_right = cv2.imread(img_path_right)
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size, None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size, None)

    if ret_left and ret_right:
        # サブピクセル精度でコーナーを調整
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_left_refined = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
        corners_right_refined = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints_left.append(corners_left_refined)
        imgpoints_right.append(corners_right_refined)

# ステレオキャリブレーション
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
flags = cv2.CALIB_FIX_INTRINSIC

retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right, camera_matrix_left, dist_coeffs_left,
    camera_matrix_right, dist_coeffs_right, gray_left.shape[::-1], criteria, flags
)

# 結果を表示
print(f"Retval: {retval}")
print(f"CameraMatrix1: \n{cameraMatrix1}")
print(f"DistCoeffs1: \n{distCoeffs1}")
print(f"CameraMatrix2: \n{cameraMatrix2}")
print(f"DistCoeffs2: \n{distCoeffs2}")
print(f"R: \n{R}")
print(f"T: \n{T}")
print(f"E: \n{E}")
print(f"F: \n{F}")

# 結果をJSONファイルに保存
result = {
    'retval': retval,
    'cameraMatrix1': cameraMatrix1.tolist(),
    'distCoeffs1': distCoeffs1.tolist(),
    'cameraMatrix2': cameraMatrix2.tolist(),
    'distCoeffs2': distCoeffs2.tolist(),
    'R': R.tolist(),
    'T': T.tolist(),
    'E': E.tolist(),
    'F': F.tolist()
}

# 平行化処理
rectified_image_folder_left = os.path.join(image_folder_left, 'rectified_image')
rectified_image_folder_right = os.path.join(image_folder_right, 'rectified_image')
os.makedirs(rectified_image_folder_left, exist_ok=True)
os.makedirs(rectified_image_folder_right, exist_ok=True)

R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, gray_left.shape[::-1], R, T
)

map1x, map1y = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, gray_left.shape[::-1], cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, gray_right.shape[::-1], cv2.CV_32FC1)

# 平行化された画像とマップの保存
for idx, (img_path_left, img_path_right) in enumerate(zip(image_paths_left, image_paths_right)):
    img_left = cv2.imread(img_path_left)
    img_right = cv2.imread(img_path_right)
    img_rect_left = cv2.remap(img_left, map1x, map1y, cv2.INTER_LINEAR)
    img_rect_right = cv2.remap(img_right, map2x, map2y, cv2.INTER_LINEAR)
    
    cv2.imwrite(os.path.join(rectified_image_folder_left, f'rectified_left_{os.path.basename(img_path_left)}.jpg'), img_rect_left)
    cv2.imwrite(os.path.join(rectified_image_folder_right, f'rectified_right_{os.path.basename(img_path_right)}.jpg'), img_rect_right)

# 平行化マップの保存
rectification_maps = {
    'map1x': map1x.tolist(),
    'map1y': map1y.tolist(),
    'map2x': map2x.tolist(),
    'map2y': map2y.tolist()
}

# 現在時刻を取得
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# 結果を保存するフォルダを作成
stereo_calib_result_folder = 'stereo_calib_result'
os.makedirs(stereo_calib_result_folder, exist_ok=True)

# JSONファイルの保存
result_filename = f'stereo_calibration_result_{current_time}.json'
rectification_maps_filename = f'rectification_maps_{current_time}.json'

with open(os.path.join(stereo_calib_result_folder, result_filename), 'w') as f:
    json.dump(result, f, indent=4)

with open(os.path.join(stereo_calib_result_folder, rectification_maps_filename), 'w') as f:
    json.dump(rectification_maps, f, indent=4)