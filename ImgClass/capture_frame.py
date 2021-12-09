import cv2
import uuid

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("Capture Frame", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s'):
        img_name = f"opencv_frame_{str(uuid.uuid1())}.jpeg"
        cv2.imwrite(img_name, frame)
        print(f"{img_name} written!")

cam.release()

cv2.destroyAllWindows()