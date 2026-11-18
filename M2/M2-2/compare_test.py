import cv2
import numpy as np
from pathlib import Path

# 读取标定参数
storage = cv2.FileStorage("../M2-1/camera_calibration.yaml",cv2.FILE_STORAGE_READ)

if not storage.isOpened():
    raise SystemExit("无法打开标定文件")

camera_matrix = storage.getNode("camera_matrix").mat()
dist_coeffs = storage.getNode("dist_coeffs").mat()

storage.release()

# 全零表示计算时忽略畸变
zero_dist = np.zeros_like(dist_coeffs)

# 使用做过标定的摄像头
cap = cv2.VideoCapture(2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise SystemExit("摄像头打开失败")

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

marker_length = 0.064
half = marker_length / 2

object_points = np.array([
    [-half,  half, 0],
    [ half,  half, 0],
    [ half, -half, 0],
    [-half, -half, 0]
], dtype=np.float32)

save_dir = Path(__file__).parent / "tests"
save_dir.mkdir(parents=True, exist_ok=True)
save_count = 1

while True:
    ret, image = cap.read()

    if not ret:
        print("摄像头读取失败")
        break

    if image.shape[:2] != (480, 640):
        print("摄像头实际画面不是 640×480，与标定分辨率不一致")
        break

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 每帧只检测一次，两次计算共用这些角点
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary)

    # 同一帧复制两份，用于分别画结果
    left_image = image.copy()
    right_image = image.copy()

    for i in range(2):
        if i == 0:
            current_dist = dist_coeffs
            result_image = left_image
            title = "WITH distortion"
        else:
            current_dist = zero_dist
            result_image = right_image
            title = "WITHOUT distortion"

        cv2.putText(result_image, title, (20, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if ids is None:
            cv2.putText(result_image, "No marker", (20, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            continue

        pts = corners[0][0]

        success, rvec, tvec = cv2.solvePnP(object_points,pts,camera_matrix,current_dist)

        if not success:
            cv2.putText(result_image, "Pose failed", (20, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            continue

        cv2.aruco.drawDetectedMarkers(result_image, corners, ids)

        cv2.drawFrameAxes(result_image,camera_matrix,current_dist,rvec,tvec,marker_length / 2)

        x = tvec[0][0]
        y = tvec[1][0]
        z = tvec[2][0]

        distance = np.sqrt(x*x + y*y + z*z)

        rotation_matrix, _ = cv2.Rodrigues(rvec)

        sy = np.sqrt(rotation_matrix[0][0] ** 2+ rotation_matrix[1][0] ** 2)

        roll = np.arctan2(rotation_matrix[2][1],rotation_matrix[2][2])
        pitch = np.arctan2(-rotation_matrix[2][0], sy)
        yaw = np.arctan2(rotation_matrix[1][0],rotation_matrix[0][0])

        roll = np.degrees(roll)
        pitch = np.degrees(pitch)
        yaw = np.degrees(yaw)

        cv2.putText(result_image, f"X: {x:.4f} m", (20, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(result_image, f"Y: {y:.4f} m", (20, 90),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(result_image, f"Z: {z:.4f} m", (20, 120),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(result_image, f"Distance: {distance:.4f} m", (20, 150),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(result_image, f"Roll: {roll:.2f} deg", (20, 180),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(result_image, f"Pitch: {pitch:.2f} deg", (20, 210),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(result_image, f"Yaw: {yaw:.2f} deg", (20, 240),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 把左右两张结果拼在一起
    comparison = np.hstack([left_image, right_image])

    cv2.imshow("Comparison", comparison)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        save_path = save_dir / f"compare_{save_count}.png"

        while save_path.exists():
            save_count = save_count + 1
            save_path = save_dir / f"compare_{save_count}.png"

        success = cv2.imwrite(str(save_path), comparison)

        if success:
            print("图片已保存到：", save_path)
            save_count = save_count + 1
        else:
            print("图片保存失败")

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()