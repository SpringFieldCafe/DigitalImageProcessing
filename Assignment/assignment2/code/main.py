import numpy as np
import cv2
from pathlib import Path
import os

cwd=Path.cwd()

def stackImages(scale,imgArray):
    rowsAvailable=isinstance(imgArray[0],list)
    if rowsAvailable:
        width=imgArray[0][0].shape[1]
        height=imgArray[0][0].shape[0]
        rows=len(imgArray)
        cols= max(len(row) for row in imgArray)
        imgBlack = np.zeros_like(imgArray[0][0])
        for x in range(0,rows):
            while len(imgArray[x]) < cols:
                imgArray[x].append(imgBlack.copy())
            for y in range(0,cols):
                if imgArray[x][y].shape[:2]==imgArray[0][0].shape[:2]:
                    imgArray[x][y]=cv2.resize(imgArray[x][y],(0,0),None,scale,scale)
                else:
                    imgArray[x][y]=cv2.resize(imgArray[x][y],(imgArray[0][0].shape[1],imgArray[0][0].shape[0]),None,scale,scale)
                if len(imgArray[x][y].shape)==2:
                    imgArray[x][y]=cv2.cvtColor(imgArray[x][y],cv2.COLOR_GRAY2BGR)
        imgBlack=np.zeros((height,width,3),np.uint8)
        hor=[imgBlack]*rows
        for x in range(0,rows):
            hor[x]=np.hstack(imgArray[x])
        ver=np.vstack(hor)
    else:
        width=imgArray[0].shape[1]
        height=imgArray[0].shape[0]
        cols=len(imgArray)

        for x in range(0,cols):
            if imgArray[x].shape[:2]==imgArray[0].shape[:2]:
                imgArray[x]=cv2.resize(imgArray[x],(0,0),None,scale,scale)
            else:
                imgArray[x]=cv2.resize(imgArray[x],(imgArray[0].shape[1],imgArray[0].shape[0]),None,scale,scale)  
            if len(imgArray[x].shape)==2:
                imgArray[x]=cv2.cvtColor(imgArray[x],cv2.COLOR_GRAY2BGR)
        hor=np.hstack(imgArray)
        ver=hor
    return ver

