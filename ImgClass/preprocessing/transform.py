import cv2 as cv
import matplotlib.pyplot as plt
import uuid
import os
import glob
import numpy as np

def edge_det(img):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
    mor = cv.morphologyEx(gray_img, cv.MORPH_DILATE, kernel)
    out_gray = cv.divide(gray_img, mor, scale=255)
    thresh = cv.threshold(out_gray, 0, 255, cv.THRESH_OTSU)[1]
    return thresh

def rotate(img, rot_angle=0, width=299, height=299):
    dimensions = (width, height)
    rot_point = (width//2, height//2)
    rot_mat = cv.getRotationMatrix2D(rot_point, rot_angle, 1.0)
    return cv.warpAffine(img, rot_mat, dimensions)

def flip(img, flipping=0): #0 vertical, 1 horizontal, -1 both axis
    img_flip = cv.flip(img, flipping)
    return img_flip

def translation(img, x, y):
    trans_mat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, trans_mat, dimensions)

def transformation(img_list):
    img = [cv.resize(img, (299, 299)) for img in img_list]
    # img = [edge_det(i) for i in img]
    img_rot1 = [rotate(i, 90) for i in img]
    img_rot2 = [rotate(i, 180) for i in img]
    img_rot3 = [rotate(i, 270) for i in img]
    flip_ver = [flip(i, 0) for i in img]
    flip_hor = [flip(i, 1) for i in img]
    flip_both = [flip(i, -1) for i in img]
    # img_trans = [translation(i, 10, 10) for i in img]
    img_final = [*img, *img_rot1, *img_rot2, *img_rot3, *flip_ver, *flip_hor, *flip_both]
    return img_final

def write_img():
    label = ['ngfl', 'ngsc', 'ngsh', 'ok']
    train_path = os.path.join('workspace', 'images')

    for imgnum in range(len(img_ngfl)):
        img = img_ngfl[imgnum]
        imgname = os.path.join(train_path, label[0], 'ngfl-' + '{}.jpg'.format(str(uuid.uuid1())))
        cv.imwrite(imgname, img)

    for imgnum in range(len(img_ngsc)):
        img = img_ngsc[imgnum]
        imgname = os.path.join(train_path, label[1], 'ngsc-' + '{}.jpeg'.format(str(uuid.uuid1())))
        cv.imwrite(imgname, img)
    
    for imgnum in range(len(img_ngsh)):
        img = img_ngsh[imgnum]
        imgname = os.path.join(train_path, label[2], 'ngsh-' + '{}.jpeg'.format(str(uuid.uuid1())))
        cv.imwrite(imgname, img)

    for imgnum in range(len(img_ok)):
        img = img_ok[imgnum]
        imgname = os.path.join(train_path, label[3], 'ok-' + '{}.jpeg'.format(str(uuid.uuid1())))
        cv.imwrite(imgname, img)

img_ngfl_raw = [cv.imread(file) for file in glob.glob('workspace\\images\\raw\\NGFL\\*.jpg')]
img_ngfl = transformation(img_ngfl_raw)
img_ngsc_raw = [cv.imread(file) for file in glob.glob('workspace\\images\\raw\\NGSC\\*.jpg')]
img_ngsc = transformation(img_ngsc_raw)
img_ngsh_raw = [cv.imread(file) for file in glob.glob('workspace\\images\\raw\\NGSH\\*.jpg')]
img_ngsh = transformation(img_ngsh_raw)
img_ok_raw = [cv.imread(file) for file in glob.glob('workspace\\images\\raw\\OK\\*.jpg')]
img_ok = transformation(img_ok_raw)

write_img()




# gray_hist = cv.calcHist([gray_img], [0], None, [256], [0, 255])
# plt.figure()
# plt.title('Gray Histogram')
# plt.xlabel('Range')
# plt.ylabel('Number of Pixel')
# plt.plot(gray_hist)
# plt.xlim([0,255])
# plt.show()

# cv.imshow('blur', blur)
# cv.imshow('thresh', thresh)
# cv.imshow('canny', canny)
# cv.imshow('asaa', img_ng[0])
# cv.waitKey(0)