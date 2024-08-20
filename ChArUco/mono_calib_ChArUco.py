import cv2
import numpy as np
import os
from cv2 import aruco
import glob
from datetime import datetime
from tkinter import Tk, filedialog
import json

# initファイルを読み込む関数
def read_init_file(file_path):
    params = {}
    with open(file_path, 'r') as file:
        for line in file:
            # 空行またはコメント行をスキップ
            if line.strip() == '' or line.strip().startswith('#'):
                continue
            
            # コロンでキーとバリューを分割
            try:
                key, value = line.split(':', 1)  # 1回だけ分割
                params[key.strip()] = value.strip()
            except ValueError:
                # コロンが含まれていない行はスキップ
                continue

    return params

# GUIでフォルダを選択
Tk().withdraw()  # Tkinterのウィンドウを隠す
folder_path = filedialog.askdirectory(title="Select the folder containing chessboard images")

# init.txtファイルのパスを指定
init_file_path = os.path.join('./', 'init_calib.txt')

# init.txtファイルからパラメータを読み込む
params = read_init_file(init_file_path)
chessboard_size = tuple(map(int, params['chessboard_size'].split(',')))
square_size = float(params['square_size'])
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

flag_init_cam_param = params['flag_init_cam_param']
init_fx_guess = float(params['init_fx_guess'])
init_fy_guess = float(params['init_fy_guess'])
init_cx_guess = float(params['init_cx_guess'])
init_cy_guess = float(params['init_cy_guess'])
# 初期カメラ行列の設定（例: 既知の値や以前のキャリブレーションからの推定値）
init_camera_matrix = np.array([[init_fx_guess, 0, init_cx_guess],
                         [0, init_fy_guess, init_cy_guess],
                         [0, 0, 1]], dtype=np.float64)

# チェスボードのコーナー座標を保存するリスト
obj_points = []
img_points = []

# チェスボードの物理的な座標
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size  # 実際のサイズにスケール

# 画像ファイルのリストを取得
images = glob.glob(os.path.join(folder_path, '*.jpg')) + glob.glob(os.path.join(folder_path, '*.png'))

# corner_detected_imageフォルダの作成
corner_detected_folder = os.path.join(folder_path, 'corner_detected_image')
os.makedirs(corner_detected_folder, exist_ok=True)

# カメラキャリブレーションko-
detected_results = []

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # charucoのコーナーを見つける
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)

    if len(corners)>0:
        # サブピクセル精度でコーナーを調整
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        obj_points.append(objp)
        img_points.append(corners_refined)

        detected_results.append((os.path.basename(fname), True))
    else:
        detected_results.append((os.path.basename(fname), False))

# キャリブレーション
if flag_init_cam_param:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], init_camera_matrix, None, [],[],flags=cv2.CALIB_USE_INTRINSIC_GUESS)
else:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

# カメラパラメータをまとめて一つのファイルに保存
camera_params = {
    'ret': ret,
    'camera_matrix': camera_matrix.tolist(),
    'dist_coeffs': dist_coeffs.tolist(),
    'rvecs': [rvec.tolist() for rvec in rvecs],
    'tvecs': [tvec.tolist() for tvec in tvecs]
}

# キャリブレーション結果(カメラ内部パラメータ)のコンソール表示
print(f"Re-projection Error (RMSE): {ret}")
print(f"Camera Matrix:\n{camera_matrix}")
print(f"Distortion Coefficients:\n{dist_coeffs}")
print(f"Rotation Vectors:\n{rvecs}")
print(f"Translation Vectors:\n{tvecs}")

# 再投影誤差を計算して描画
errors = []
detected_img_idx = 0
for i, fname in enumerate(images):
    if detected_results[i][1]:  # コーナーが検出された画像のみ処理
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # オブジェクトポイントから再投影
        img_points1 = img_points[detected_img_idx]
        img_points2, _ = cv2.projectPoints(obj_points[detected_img_idx], rvecs[detected_img_idx], tvecs[detected_img_idx], camera_matrix, dist_coeffs)

        # 再投影誤差を計算
        per_point_sqr_errors = abs(img_points1.reshape(-1,2)-img_points2.reshape(-1,2))**2
        sum_error = np.sum(per_point_sqr_errors)
        error = np.sqrt(sum_error/len(img_points2))
        errors.append(error)
        
        # コーナーを画像に描画する
        cv2.drawChessboardCorners(img, chessboard_size, img_points[detected_img_idx], True)
        # 座標軸を描画する
        axis = np.float32([[2.5*square_size, 0, 0], [0, 2.5*square_size, 0], [0, 0, 2.5*square_size]]).reshape(-1, 3)
        imgpts, _ = cv2.projectPoints(axis, rvecs[detected_img_idx], tvecs[detected_img_idx], camera_matrix, dist_coeffs)

        # img = cv2.line(img, tuple(map(int,img_points2[0].ravel())), tuple(map(int,imgpts[0].ravel())), (0, 0, 255), 5) # X軸 赤
        # img = cv2.line(img, tuple(map(int,img_points2[0].ravel())), tuple(map(int,imgpts[1].ravel())), (0, 255, 0), 5) # Y軸 緑
        # img = cv2.line(img, tuple(map(int,img_points2[0].ravel())), tuple(map(int,imgpts[2].ravel())), (255, 0, 0), 5) # Z軸 青
        cv2.putText(img, f"RMSE: {error:.5f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # 再投影誤差を画像に描画
        cv2.imwrite(os.path.join(corner_detected_folder, os.path.basename(fname)), img)
        detected_img_idx += 1
        
# 現在時刻を取得
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# カメラパラメータを保存するディレクトリの作成
camera_param_folder = os.path.join(folder_path, 'camera_param')
os.makedirs(camera_param_folder, exist_ok=True)

# JSONファイルの保存
camera_param_filename = f'camera_params_{current_time}.json'
calibration_log_filename = f'calibration_log_{current_time}.csv'

params_file_path = os.path.join(camera_param_folder, camera_param_filename)

with open(params_file_path, 'w') as f:
    json.dump(camera_params, f, indent=4)

# キャリブレーションの結果をログとして.csvに保存
log_file_path = os.path.join(camera_param_folder, calibration_log_filename)

detected_img_idx = 0
with open(log_file_path, 'w') as f:
    # ヘッダー情報
    f.write(f"******start haeder******\n")
    f.write(f"Code execution time:,,, {datetime.now()}\n")
    f.write(f"flag_init_cam_param:,,, {flag_init_cam_param}\n")
    f.write(f"init camera Matrix:,,, {init_camera_matrix.tolist()}\n")
    f.write(f"******end haeder******\n\n")
    
    f.write(f"<<calibration result>>\n")
    f.write(f"Re-projection Error (RMSE):,,, {ret}\n")
    f.write(f"Camera Matrix:,,, {camera_matrix.tolist()}\n")
    f.write(f"Distortion Coefficients:,,, {dist_coeffs.tolist()}\n")
    
    f.write("Image File, Corners Detected, RMSE, rvec, ,,tvec\n")
    # 各画像の結果を記載
    for i, (fname, detected) in enumerate(detected_results):
        if detected:
            rmse = errors[detected_img_idx]
            rvec = rvecs[detected_img_idx].flatten()
            tvec = tvecs[detected_img_idx].flatten()
            f.write(f"{fname}, {detected}, {rmse:.5f}, {rvec.tolist()}, {tvec.tolist()}\n")
            detected_img_idx += 1
        else:
            f.write(f"{fname}, {detected}, N/A, N/A, ,,N/A\n")
