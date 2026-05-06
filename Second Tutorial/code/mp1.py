import cv2
import numpy as np
import multiprocessing as mp
import queue


framewidth = 640
frameheight = 480

myColors = [
    [65, 89, 72, 86, 202, 217]
]

myColorValues = [
    [0, 204, 0]
]


def getContours(img):
    contours_info = cv2.findContours(
        img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours_info) == 3:
        _, contours, hierarchy = contours_info
    else:
        contours, hierarchy = contours_info

    x, y, w, h = 0, 0, 0, 0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 500:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)

    return x + w // 2, y


def findColor(img, myColors):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    newPoints = []

    for colorId, color in enumerate(myColors):
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])

        mask = cv2.inRange(imgHSV, lower, upper)
        x, y = getContours(mask)

        if x != 0 and y != 0:
            newPoints.append([x, y, colorId])

    return newPoints


def drawOnCanvas(imgResult, myPoints, myColorValues):
    for point in myPoints:
        x = point[0]
        y = point[1]
        colorId = point[2]

        cv2.circle(
            imgResult,
            (x, y),
            10,
            myColorValues[colorId],
            cv2.FILLED
        )


def capture_process(data_queue, stop_event):
    cap = cv2.VideoCapture(0)
    cap.set(3, framewidth)
    cap.set(4, frameheight)
    cap.set(10, 150)

    while not stop_event.is_set():
        success, img = cap.read()

        if not success:
            continue

        newPoints = findColor(img, myColors)

        # 队列满了就丢掉旧帧，避免延迟越来越大
        if data_queue.full():
            try:
                data_queue.get_nowait()
            except queue.Empty:
                pass

        data_queue.put_nowait((img, newPoints))

    cap.release()


if __name__ == "__main__":
    data_queue = mp.Queue(maxsize=2)
    stop_event = mp.Event()

    p = mp.Process(target=capture_process, args=(data_queue, stop_event))
    p.start()

    myPoints = []

    while True:
        try:
            img, newPoints = data_queue.get(timeout=1)
        except queue.Empty:
            continue

        imgResult = img.copy()

        if len(newPoints) != 0:
            myPoints.extend(newPoints)

        if len(myPoints) != 0:
            drawOnCanvas(imgResult, myPoints, myColorValues)

        cv2.imshow("Result", imgResult)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            break

    p.join(timeout=1)

    if p.is_alive():
        p.terminate()
        p.join()

    cv2.destroyAllWindows()
