from pathlib import Path
import cv2
import numpy as np

# 找到存放照片的文件夹
project_dir = Path(__file__).parent
image_dir = project_dir / "calib_imgs"

# 保存标出角点的图片，方便我们检查
result_dir = project_dir / "results" / "corners"
result_dir.mkdir(parents=True, exist_ok=True)
image_paths = sorted(image_dir.glob("calib_*.png"))

# 生成真实棋盘上54个内部交点的坐标，单位是毫米
object_points = []

for row in range(6):
    for col in range(9):
        x = col * 11.5
        y = row * 11.5
        z = 0

        object_points.append([x, y, z])

# 转成OpenCV计算需要的数组格式，使用32位浮点数
object_points = np.array(object_points, dtype=np.float32)

# 分别保存每张有效照片对应的真实坐标和照片坐标
all_object_points = []
all_image_points = []        

valid_count = 0

for image_path in image_paths:
    image = cv2.imread(str(image_path))
    if image is None:
        print(image_path.name, "图片读取失败")
        continue

    # 转成灰度图，用来寻找黑白格交点
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_size = (gray.shape[1], gray.shape[0])
    # 9列、6行内部角点，不是方格数量
    found, corners = cv2.findChessboardCorners(gray,(9, 6))

    if found:
        # 最多调整30次；位置变化小于0.001像素时也可以停止
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,30,0.001)

        # 进一步调整角点坐标
        corners = cv2.cornerSubPix(gray,corners,(11, 11),(-1, -1),criteria)

        valid_count = valid_count + 1

        all_object_points.append(object_points.copy())
        all_image_points.append(corners)
        print(image_path.name, "找到全部54个角点")

        cv2.drawChessboardCorners(image,(9, 6),corners,found)

        output_path = result_dir / image_path.name
        cv2.imwrite(str(output_path), image)
    else:
        print(image_path.name, "没有找齐角点")

print("照片总数：", len(image_paths))
print("找到全部角点的照片数：", valid_count)

if valid_count < 15:
    print("有效照片不足15张，请补拍后再标定")
else:
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(all_object_points,all_image_points,image_size,None,None)

    print("标定误差 RMS：", rms)
    print("摄像头内参矩阵：")
    print(camera_matrix)
    print("畸变系数：")
    print(dist_coeffs)

    # 保存参数文件
    calibration_path = project_dir / "camera_calibration.yaml"

    storage = cv2.FileStorage(str(calibration_path),cv2.FILE_STORAGE_WRITE)

    if not storage.isOpened():
        raise OSError("无法创建标定参数文件")

    storage.write("image_width", image_size[0])
    storage.write("image_height", image_size[1])
    storage.write("board_columns", 9)
    storage.write("board_rows", 6)
    storage.write("square_size_mm", 11.5)
    storage.write("valid_images", valid_count)
    storage.write("rms", rms)
    storage.write("camera_matrix", camera_matrix)
    storage.write("dist_coeffs", dist_coeffs)

    storage.release()

    print("标定参数已保存到：", calibration_path)