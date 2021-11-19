import cv2
vid = cv2.VideoCapture(1)
  
while(True):
    ret, frame = vid.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# # vid.release()

# # cv2.destroyAllWindows()
# import cv2

# def rotate(img, rot_angle=0, width=640, height=480):
#     dimensions = (width, height)
#     rot_point = (width//2, height//2)
#     rot_mat = cv2.getRotationMatrix2D(rot_point, rot_angle, 1.0)
#     return cv2.warpAffine(img, rot_mat, dimensions)

# cam = cv2.VideoCapture(1)

# cv2.namedWindow("test")

# img_counter = 0

# while True:
#     ret, frame = cam.read()
#     if not ret:
#         print("failed to grab frame")
#         break
#     # frame = rotate(frame, 180)
#     cv2.imshow("test", frame)

#     k = cv2.waitKey(1)
#     if k%256 == 27:
#         # ESC pressed
#         print("Escape hit, closing...")
#         break
#     elif k%256 == 32:
#         # SPACE pressed
#         img_name = "opencv_frame_{}.png".format(img_counter)
#         cv2.imwrite(img_name, frame)
#         print("{} written!".format(img_name))
#         img_counter += 1

# cam.release()

# cv2.destroyAllWindows()