'''
Document Scanner - Reads video feed to scan documents and save the scanned document
Author : Disha Sarawgi
Date : 4 October, 2020
Reference : https://www.youtube.com/watch?v=WQeoO7MI0Bs&ab_channel=Murtaza%27sWorkshop-RoboticsandAI
'''
import cv2
import numpy as np
import datetime

frameWidth = 480
frameHeight = 640
cap = cv2.VideoCapture(1) # capture video feed from external webcam
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

def getContours(img, imgContour):
    '''
    Detect contours, approximate by polynomials, draw the largest 4 sided polynomial.
    :param img: original image taken by webcam
    :param imgContour: image on which the contours are drawn
    :return: array of 4 [x y] points of largest rectangle identified
    '''
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    maxArea = 0
    largest = np.array([])
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            # cv2.drawContours(imgContour,cnt,-1,(255,0,0),2)
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            if area > maxArea and len(approx) == 4:
                largest = approx
                maxArea = area
        cv2.drawContours(imgContour, largest, -1, (255, 0, 0), 20)
    return largest

def reorder(myPoints):
    '''
    reorder 4 corner points to get the right warp matrix
    :param myPoints: array of 4 corner contour points
    :return: reordered points
    '''
    myPoints = myPoints.reshape((4,2))
    myPointsRes = np.zeros((4,1,2), np.int32)
    add = myPoints.sum(1)
    subt = np.diff(myPoints, axis=1)

    myPointsRes[0] = myPoints[np.argmin(add)]
    myPointsRes[3] = myPoints[np.argmax(add)]
    myPointsRes[1] = myPoints[np.argmin(subt)]
    myPointsRes[2] = myPoints[np.argmax(subt)]

    return myPointsRes

def getWarp(img, biggest):
    '''
    To return the warped image from the identified contour points
    :param img: the original image
    :param biggest: the identified contour points
    :return: the warped and cropped result of the scanned document
    '''
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0],[frameWidth,0],[0,frameHeight],[frameWidth,frameHeight]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgRes = cv2.warpPerspective(img, matrix, (frameWidth,frameHeight))

    imgCrop = imgRes[10:imgRes.shape[0]-10, 10:imgRes.shape[1]-10] # to remove noise from image edges to get

    return imgCrop


def preProcessing(img):
    '''
    Preprocess the original image to obtain the edges
    :param img: original image
    :return: processed image (with highlighted edges)
    '''
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,200,200)
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,kernel,iterations=2)
    imgThres = cv2.erode(imgDial, kernel, iterations=1)

    return imgThres

def main():

    while True:
        success, img = cap.read()

        if success == True:
            imgContour = img.copy()
            imgThres = preProcessing(img)
            biggest = getContours(imgThres, imgContour)
            if len(biggest.shape) > 2:
                imgWarped = getWarp(img, biggest)
                cv2.imshow("Warped", imgWarped)

            imgThres_3 = cv2.cvtColor(imgThres, cv2.COLOR_GRAY2BGR)
            imgStack = np.hstack((imgThres_3, imgContour))
            cv2.imshow("Result", imgStack)

        ## to save an image
        if (cv2.waitKey(1) & 0xFF == ord('s')):
            cv2.imwrite("Scan_" + datetime.datetime.now().strftime("%H_%M_%S") + ".jpg", imgWarped)
            print("Saving image!")

        elif (cv2.waitKey(1) & 0xFF == ord('q')) or success == False:
            break
    cap.release()

if __name__ == '__main__':
    main()
