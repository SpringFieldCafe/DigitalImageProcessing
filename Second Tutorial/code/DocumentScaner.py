import cv2
import numpy as np
from Modal_Stack import stackImages

###############
widthImg = 640
heightImg = 480
###############

cap = cv2.VideoCapture(0)
cap.set(3, widthImg)
cap.set(4, heightImg)
cap.set(10, 150)


def preProcessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgGray = cv2.equalizeHist(imgGray)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    v = np.median(imgBlur)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))

    imgCanny = cv2.Canny(imgBlur, lower, upper)
    
    kernel = np.ones((5, 5), np.uint8)

    imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
    imgThres = cv2.erode(imgDial, kernel, iterations=1)

    return imgThres


def reorder(myPoints):
    myPoints=myPoints.reshape((4,2))
    myPointsNew=np.zeros((4,1,2),np.int32)
    add=myPoints.sum(1)
    myPointsNew[0]=myPoints[np.argmin(add)]
    myPointsNew[3]=myPoints[np.argmax(add)]
    diff=np.diff(myPoints,axis=1)
    myPointsNew[1]=myPoints[np.argmin(diff)]
    myPointsNew[2]=myPoints[np.argmax(diff)]
    return myPointsNew


def getContours(img, imgContour):
    biggest = np.array([])
    maxArea = 0

    contours_info = cv2.findContours(
        img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # 兼容 OpenCV 3 和 OpenCV 4
    if len(contours_info) == 3:
        _, contours, hierarchy = contours_info
    else:
        contours, hierarchy = contours_info

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 5000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area

    if biggest.size != 0:
        cv2.drawContours(imgContour, [biggest], -1, (200, 190, 180), 20)

    return biggest


def getWarp(img, biggest):
    biggest = reorder(biggest)

    pts1 = np.float32(biggest)
    pts2 = np.float32([
        [0, 0],
        [widthImg, 0],
        [0, heightImg],
        [widthImg, heightImg]
    ])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    return imgOutput


while True:
    success, img = cap.read()

    if not success:
        print("摄像头读取失败")
        break

    img = cv2.resize(img, (widthImg, heightImg))
    imgContour = img.copy()

    imgThres = preProcessing(img)

    biggest = getContours(imgThres, imgContour)

    cv2.imshow('Res', imgContour)
    cv2.imshow('Threshold', imgThres)

    # 只有检测到四边形时，才进行透视变换
    if biggest.size != 0:
        imgWarped = getWarp(img, biggest)
        cv2.imshow('Warped', imgWarped)
    else:
        print("未检测到合适的四边形轮廓")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()