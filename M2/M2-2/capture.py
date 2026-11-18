import cv2

cap = cv2.VideoCapture(2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    success, frame = cap.read()

    if not success:
        print("摄像头读取失败")
        break

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        cv2.imwrite("marker/camera_test.png", frame)
        print("图片已保存到 marker/camera_test.png")

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()