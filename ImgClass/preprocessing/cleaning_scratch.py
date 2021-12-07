import cv2 as cv
import matplotlib.pyplot as plt

i = cv.imread('WhatsApp Unknown 2021-11-19 at 13.40.46\\test (2).jpeg')
img = cv.resize(i, (600,600))
blur = cv.GaussianBlur(img, (3,3), cv.BORDER_DEFAULT)
# cv.imshow('awal', img)

gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
mor = cv.morphologyEx(gray_img, cv.MORPH_DILATE, kernel)

out_gray = cv.divide(gray_img, mor, scale=255)
thresh = cv.threshold(out_gray, 0, 255, cv.THRESH_OTSU)[1]
# thresh = cv.threshold(out_gray, 200, 255, cv.THRESH_BINARY)[1]
contours = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[0]
# cv.drawContours(img, contours, -1, (0,0,255), 1)

cv.imshow('img', img)
cv.imshow('gray', out_gray)
cv.imshow('thresh', thresh)

# gray_hist = cv.calcHist([out_gray], [0], None, [256], [0, 255])
# plt.figure()
# plt.title('Gray Histogram')
# plt.xlabel('Range')
# plt.ylabel('Number of Pixel')
# plt.plot(gray_hist)
# plt.xlim([0,255])
# plt.show()

cv.waitKey(0)