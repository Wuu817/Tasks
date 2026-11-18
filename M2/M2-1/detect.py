import argparse
from pathlib import Path

import cv2


def load_calibration(calibration_path):
    """从 YAML 文件读取相机内参和畸变系数。"""
    storage = cv2.FileStorage(str(calibration_path), cv2.FILE_STORAGE_READ)

    if not storage.isOpened():
        raise OSError("无法读取标定文件：" + str(calibration_path))

    camera_matrix = storage.getNode("camera_matrix").mat()
    dist_coeffs = storage.getNode("dist_coeffs").mat()

    storage.release()

    return camera_matrix, dist_coeffs


def detect(image):
    """识别一张画面中的红黄绿灯，并在图像上标记结果。"""

    # 1. 去噪
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # 2. 转 HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 3. 颜色分割
    red1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 100, 100), (179, 255, 255))
    red_mask = cv2.bitwise_or(red1, red2)

    yellow_mask = cv2.inRange(hsv, (20, 100, 100), (35, 255, 255))
    green_mask = cv2.inRange(hsv, (40, 100, 100), (85, 255, 255))

    # 4. 形态学处理使用的 3×3 矩形核
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # 保存目前找到的最佳候选
    best_color = None
    best_contour = None
    best_area = 0

    # 依次检查三种颜色
    for color in ["RED", "YELLOW", "GREEN"]:

        if color == "RED":
            mask = red_mask
        elif color == "YELLOW":
            mask = yellow_mask
        else:
            mask = green_mask

        # 5. 开运算：去除 mask 中的小白色噪点
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 6. 查找轮廓
        contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < 100:
                continue

            perimeter = cv2.arcLength(contour, True)

            if perimeter == 0:
                continue

            # 圆度越接近 1，越接近圆形
            circularity = 4 * 3.141592653589793 * area / (perimeter * perimeter)

            if circularity < 0.7:
                continue

            # 多个候选同时存在时，选择面积最大的
            if area > best_area:
                best_area = area
                best_color = color
                best_contour = contour

    results = []

    # 7. 状态判定
    if best_contour is not None:
        center, radius = cv2.minEnclosingCircle(best_contour)
        x, y = center

        x = int(x)
        y = int(y)
        radius = int(radius)

        # 8. 画最小外接圆
        cv2.circle(image, (x, y), radius, (0, 255, 0), 2)

        text = f"{best_color} x={x} y={y} area={best_area:.0f} radius={radius}"
        results.append(text)

        cv2.putText(image,text,(10, 25),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),1)

    else:
        cv2.putText(image,"UNKNOWN",(10, 25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255, 255, 255),2)

        results.append("UNKNOWN")

    return image, results


def main():
    parser = argparse.ArgumentParser(description="识别图片、视频或摄像头中的红黄绿灯")

    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--image", help="要识别的图片路径")
    sources.add_argument("--video", help="要识别的视频路径")
    sources.add_argument("--camera", type=int, help="摄像头编号，例如 0")

    parser.add_argument("--calib",default=str(Path(__file__).parent / "camera_calibration.yaml"),help="相机标定 YAML 文件路径")

    parser.add_argument("--show", action="store_true", help="显示检测窗口")
    parser.add_argument("--output", help="图片检测结果保存路径")

    args = parser.parse_args()

    # 读取相机标定参数
    camera_matrix, dist_coeffs = load_calibration(Path(args.calib))

    if args.output is not None and args.image is None:
        parser.error("--output 目前只用于保存图片检测结果")

    if args.camera is not None and args.camera < 0:
        parser.error("摄像头编号不能小于 0")

    # ==================== 图片模式 ====================

    if args.image is not None:
        image = cv2.imread(args.image)

        if image is None:
            parser.exit(1, "图片读取失败，请检查路径：" + args.image + "\n")

        image = cv2.undistort(image, camera_matrix, dist_coeffs)
        image, results = detect(image)

        for text in results:
            print(text)

        if args.output is None:
            output_path = Path(__file__).parent / "results" / (Path(args.image).stem + "_detected.png")
        else:
            output_path = Path(args.output)

        if output_path.resolve() == Path(args.image).resolve():
            parser.error("输出路径不能与原始图片相同")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        saved = cv2.imwrite(str(output_path), image)

        if not saved:
            parser.exit(1, "检测结果保存失败\n")

        print("已保存：", output_path)

        if args.show:
            cv2.imshow("Traffic Light", image)
            print("点击图片窗口，按任意键关闭")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return

    # ==================== 视频 / 摄像头模式 ====================

    if args.video is not None:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        cap.release()
        parser.exit(1, "无法打开视频或摄像头，请检查路径或编号\n")

    if args.show:
        print("点击检测窗口后：按 s 保存当前画面，按 q 退出")
    else:
        print("不显示窗口；按 Ctrl+C 停止")

    frame_count = 0
    save_count = 1

    try:
        while True:
            success, image = cap.read()

            if not success:
                if args.video is not None:
                    if frame_count == 0:
                        parser.exit(1, "视频未读取到有效画面\n")

                    print("视频读取结束，共处理", frame_count, "帧")
                else:
                    parser.exit(1, "没有读取到摄像头画面\n")

                break

            frame_count = frame_count + 1

            # 去畸变
            image = cv2.undistort(image, camera_matrix, dist_coeffs)

            # 检测当前帧
            image, results = detect(image)

            if args.show:
                cv2.imshow("Traffic Light", image)
                key = cv2.waitKey(1) & 0xFF

                # 按 s 保存当前检测结果
                if key == ord("s"):
                    save_dir = Path(__file__).parent / "results" / "normal"
                    save_dir.mkdir(parents=True, exist_ok=True)

                    save_path = save_dir / f"{save_count}.png"

                    # 已有同名图片就继续找下一个编号
                    while save_path.exists():
                        save_count = save_count + 1
                        save_path = save_dir / f"{save_count}.png"

                    saved = cv2.imwrite(str(save_path), image)

                    if saved:
                        print("已保存：", save_path)
                        save_count = save_count + 1
                    else:
                        print("图片保存失败")

                # 按 q 退出
                if key == ord("q"):
                    break

            else:
                print("第", frame_count, "帧：", "; ".join(results))

    except KeyboardInterrupt:
        print("\n已停止识别")

    finally:
        cap.release()

        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()