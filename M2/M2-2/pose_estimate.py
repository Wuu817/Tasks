import cv2
import numpy as np
import time
from pathlib import Path
import argparse

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description="ArUco 位姿估计")

# 图片、视频、摄像头必须选一种，而且只能选一种
source = parser.add_mutually_exclusive_group(required=True)

source.add_argument("--image", help="图片路径")
source.add_argument("--video", help="视频路径")
source.add_argument("--camera", type=int, help="摄像头编号")

# 是否显示结果窗口
parser.add_argument("--show", action="store_true")

# 默认使用 M2-1 的标定文件
default_calib = (Path(__file__).resolve().parent.parent/ "M2-1"/ "camera_calibration.yaml")

parser.add_argument("--calib",default=str(default_calib),help="相机标定文件路径")

# 读取终端命令中的参数
args = parser.parse_args()

cap = None

if args.camera is not None:
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

elif args.video is not None:
    cap = cv2.VideoCapture(args.video)

# 图片模式不需要打开摄像头或视频
if cap is not None:
    if not cap.isOpened():
        parser.exit(1, "无法打开摄像头或视频\n")

def load_calibration():
    storage = cv2.FileStorage(args.calib, cv2.FILE_STORAGE_READ)

    if not storage.isOpened():
        parser.exit(1, "无法打开标定文件：" + args.calib + "\n")

    camera_matrix = storage.getNode("camera_matrix").mat()
    dist_coeffs = storage.getNode("dist_coeffs").mat()

    storage.release()

    if camera_matrix is None or dist_coeffs is None:
        parser.exit(1,"标定文件缺少 camera_matrix 或 dist_coeffs\n")

    return camera_matrix, dist_coeffs

camera_matrix, dist_coeffs = load_calibration()

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

marker_length = 0.064
half = marker_length / 2
object_points = np.array([
    [-half,  half, 0],
    [ half,  half, 0],
    [ half, -half, 0],
    [-half, -half, 0]
], dtype=np.float32)

save_dir = Path(__file__).parent / "results"
save_dir.mkdir(parents=True, exist_ok=True)

save_count = 1

# 记录统计起点、已处理帧数、显示的帧率
fps_start = time.perf_counter()
frame_count = 0
fps = 0.0

while True:
    if args.image is not None:
        # 图片模式：从文件读取一张图片
        image = cv2.imread(args.image)

        if image is None:
            print("图片读取失败，请检查路径：", args.image)
            break

    else:
        # 摄像头或视频模式：读取下一帧
        ret, image = cap.read()

        if not ret:
            if args.video is not None:
                print("视频读取结束或无法继续读取")
            else:
                print("摄像头读取失败")
            break

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)

        pts = corners[0][0]

        success, rvec, tvec = cv2.solvePnP(object_points, pts, camera_matrix, dist_coeffs)

        if not success:
            print("位姿计算失败")
            if args.image is not None:
                break
            continue

        # 从标记中心画出三色坐标轴，轴长为 3.2 cm
        cv2.drawFrameAxes(image,camera_matrix,dist_coeffs,rvec,tvec,marker_length / 2)


        x = tvec[0][0]
        y = tvec[1][0]
        z = tvec[2][0]

        distance = np.sqrt(x*x + y*y + z*z)

        rotation_matrix, _ = cv2.Rodrigues(rvec)

        sy = np.sqrt(rotation_matrix[0][0] ** 2 + rotation_matrix[1][0] ** 2)

        roll = np.arctan2(rotation_matrix[2][1], rotation_matrix[2][2])
        pitch = np.arctan2(-rotation_matrix[2][0], sy)
        yaw = np.arctan2(rotation_matrix[1][0], rotation_matrix[0][0])

        roll = np.degrees(roll)
        pitch = np.degrees(pitch)
        yaw = np.degrees(yaw)

        cv2.putText(image, f"Distance: {distance:.3f} m", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(image, f"X: {x:.3f} m", (20, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 255, 0),2)

        cv2.putText(image, f"Y: {y:.3f} m", (20, 90),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 255, 0),2)

        cv2.putText(image, f"Z: {z:.3f} m", (20, 120),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 255, 0),2)

        cv2.putText(image, f"Roll: {roll:.2f} deg", (20, 150),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 255, 0),2)

        cv2.putText(image, f"Pitch: {pitch:.2f} deg", (20, 180),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 255, 0),2)

        cv2.putText(image, f"Yaw: {yaw:.2f} deg", (20, 210),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        names = ["TL", "TR", "BR", "BL"]

        for i in range(4):
            corner_x = int(pts[i][0])
            corner_y = int(pts[i][1])

            cv2.circle(image, (corner_x, corner_y),8, (0, 0, 255), -1)
            cv2.putText(image, names[i], (corner_x + 10, corner_y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 无论有没有检测到标记，都统计帧率
    frame_count = frame_count + 1
    current_time = time.perf_counter()
    elapsed = current_time - fps_start

    if elapsed >= 1.0:
        fps = frame_count / elapsed
        frame_count = 0
        fps_start = current_time

    cv2.putText(image, f"FPS: {fps:.1f}", (20, 240),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 图片模式自动保存；视频和摄像头默认不保存
    should_save = args.image is not None

    # 默认没有按键
    key = -1

    if args.show:
        cv2.imshow("Pose", image)

        if args.image is not None:
            # 图片停在窗口里，按任意键后继续保存并退出
            key = cv2.waitKey(0) & 0xFF
        else:
            # 视频和摄像头继续刷新画面
            key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        should_save = True

    if should_save:
        save_path = save_dir / f"axis_{save_count}.png"
        
        # 跳过已有的文件，重启程序后也不会覆盖
        while save_path.exists():
            save_count = save_count + 1
            save_path = save_dir / f"axis_{save_count}.png"

        success = cv2.imwrite(str(save_path), image)

        if success:
            print("图片已保存到：", save_path)
            save_count = save_count + 1
        else:
            print("图片保存失败")

    if key == ord("q"):
        break
    # 单张图片处理一次就结束，避免反复读取同一张图片
    if args.image is not None:
        break

if cap is not None:
    cap.release()

if args.show:
    cv2.destroyAllWindows()