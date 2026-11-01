from pathlib import Path

import cv2

# 照片保存在 capture.py 旁边的 light_imgs 文件夹
save_dir = Path(__file__).parent / "light_imgs"
save_dir.mkdir(exist_ok=True)

cap = cv2.VideoCapture(2)
number = 1

if not cap.isOpened():
    print("摄像头打开失败")
else:
    print("点击摄像头窗口：按 s 保存照片，按 q 退出")

    while True:
        success, frame = cap.read()

        if not success:
            print("没有读取到摄像头画面")
            break

        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            # 跳过已有文件，避免覆盖之前的照片
            path = save_dir / f"calib_{number:03d}.png"

            while path.exists():
                number = number + 1
                path = save_dir / f"calib_{number:03d}.png"

            if cv2.imwrite(str(path), frame):
                print("已保存：", path.name)
                number = number + 1
            else:
                print("照片保存失败")

        elif key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()