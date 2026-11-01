import argparse
from pathlib import Path

import cv2


def detect(image):
    """识别一张画面，在图上标记，并返回文字结果。"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 筛选红色
    red1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 100, 100), (179, 255, 255))
    red_mask = cv2.bitwise_or(red1, red2)

    # 筛选黄色和绿色
    yellow_mask = cv2.inRange(hsv, (20, 100, 100), (35, 255, 255))
    green_mask = cv2.inRange(hsv, (40, 100, 100), (85, 255, 255))

    # 每张新画面都重新记录检测结果
    text_y = 25
    found = False
    results = []

    for color in ["RED", "YELLOW", "GREEN"]:
        if color == "RED":
            mask = red_mask
        elif color == "YELLOW":
            mask = yellow_mask
        else:
            mask = green_mask

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < 100:
                continue

            perimeter = cv2.arcLength(contour, True)

            if perimeter == 0:
                continue

            circularity = 4 * 3.141592653589793 * area / (perimeter * perimeter)

            if circularity < 0.7:
                continue

            found = True

            center, radius = cv2.minEnclosingCircle(contour)
            x, y = center

            x = int(x)
            y = int(y)
            radius = int(radius)

            cv2.circle(image, (x, y), radius, (0, 255, 0), 2)

            text = f"{color} x={x} y={y} area={area:.0f}"
            results.append(text)

            cv2.putText(image,text,(10, text_y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),1)

            text_y = text_y + 25

    if not found:
        cv2.putText(image,"UNKNOWN",(10, 25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255, 255, 255),2)


    if not found:
        results.append("UNKNOWN")

    return image, results


def main():
    parser = argparse.ArgumentParser(description="识别图片、视频或摄像头中的红黄绿灯")
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--image", help="要识别的图片路径")
    sources.add_argument("--video", help="要识别的视频路径")
    sources.add_argument("--camera", type=int, help="摄像头编号，例如 0")
    parser.add_argument("--show", action="store_true", help="显示检测窗口，按 q 退出")
    parser.add_argument("--output", help="图片检测结果的保存路径；默认 results/图片名_detected.png")
    args = parser.parse_args()

    if args.output is not None and args.image is None:
        parser.error("--output 目前只用于保存图片检测结果")
    if args.camera is not None and args.camera < 0:
        parser.error("摄像头编号不能小于 0")

    # 图片只处理一次，保存带标记的结果。
    if args.image is not None:
        image = cv2.imread(args.image)
        if image is None:
            parser.exit(1, "图片读取失败，请检查路径：" + args.image + "\n")

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
        if not cv2.imwrite(str(output_path), image):
            parser.exit(1, "检测结果保存失败\n")
        print("已保存：", output_path)

        if args.show:
            cv2.imshow("Traffic Light", image)
            print("点击图片窗口，按任意键关闭")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # 视频和摄像头都需要逐帧读取。
    if args.video is not None:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        cap.release()
        parser.exit(1, "无法打开视频或摄像头，请检查路径或编号\n")

    if args.show:
        print("点击检测窗口，按 q 退出")
    else:
        print("不显示窗口；按 Ctrl+C 停止")

    frame_count = 0
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
            image, results = detect(image)
            if args.show:
                cv2.imshow("Traffic Light", image)
                key = cv2.waitKey(1) & 0xFF
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
