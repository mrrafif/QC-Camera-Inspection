import cv2 as cv
import matplotlib.pyplot as plt
import uuid
import os
import glob
import numpy as np

def edge_det(img):
    # img = cv.resize(img, (150, 150))
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (9,9))
    mor = cv.morphologyEx(gray_img, cv.MORPH_OPEN, kernel)
    blur = cv.GaussianBlur(mor, (5,5), cv.BORDER_DEFAULT)
    thresh = cv.threshold(blur, 140, 255, cv.THRESH_BINARY)[1]
    canny = cv.Canny(thresh, 125, 255)
    # cv.imshow('blur', blur)
    # cv.imshow('thresh', thresh)
    # cv.imshow('canny', canny)
    # cv.waitKey(0)
    return canny

def rotate(img, rot_angle=0, width=150, height=150):
    dimensions = (width, height)
    rot_point = (width//2, height//2)
    rot_mat = cv.getRotationMatrix2D(rot_point, rot_angle, 1.0)
    return cv.warpAffine(img, rot_mat, dimensions)

def translation(img, x, y):
    trans_mat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, trans_mat, dimensions)

def transformation(img_list):
    img = [cv.resize(img, (150, 150)) for img in img_list]
    img = [edge_det(i) for i in img]
    img_rot = [rotate(i, 180) for i in img]
    img_trans = [translation(i, 20, 20) for i in img]
    img_final = [*img, *img_rot, *img_trans]
    return img_final

def write_img():
    label = ['ng', 'ok']
    train_path = os.path.join('workspace', 'images', 'train-dai')

    for imgnum in range(len(img_ng)):
        img = img_ng[imgnum]
        imgname = os.path.join(train_path, label[0], 'ng-' + '{}.jpeg'.format(str(uuid.uuid1())))
        cv.imwrite(imgname, img)

    for imgnum in range(len(img_ok)):
        img = img_ok[imgnum]
        imgname = os.path.join(train_path, label[1], 'ok-' + '{}.jpeg'.format(str(uuid.uuid1())))
        cv.imwrite(imgname, img)

img_ng_raw = [cv.imread(file) for file in glob.glob('workspace\\images\\raw\\ng*.jpeg')]
img_ng = transformation(img_ng_raw)
img_ok_raw = [cv.imread(file) for file in glob.glob('workspace\\images\\raw\\ok*.jpeg')]
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