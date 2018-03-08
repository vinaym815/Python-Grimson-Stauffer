#!/usr/bin/python3

from scipy import stats
from multiprocessing import Pool
import numpy as np
import cv2 as cv
import time

####################################################################################################################
alpha = 0.001           # learning rate
initWt = 0.001          # weight of a new gaussian
initVar = 255           # variance of a new gaussian
maxG = 5                # maximum number of gaussians
wtsThreshold = 0.9      # weight threshold for background
numProcess = 4          # number of processes

####################################################################################################################
def work(inputVec):
    bigDataIn = inputVec.reshape(4+5*maxG)
    pixData = bigDataIn[0:3]
    frameStrut = bigDataIn[3:]
    flagModelFit = False
    for k in range(int(frameStrut[-1])):
        dis = pixData - frameStrut[2*maxG+3*k:2*maxG+3*(k+1)]

        # checking if any of the gaussians fit
        if np.dot(dis, dis) < 6.25*frameStrut[maxG+k]:
            flagModelFit = True

            # updating parameters
            rho = alpha*stats.multivariate_normal.pdf(
                pixData, frameStrut[2*maxG+3*k:2*maxG+3*(k+1)], frameStrut[maxG+k]*np.eye(3))
            frameStrut[k] = (1-alpha)*frameStrut[k] + alpha
            frameStrut[maxG+k] = (1-rho)*frameStrut[maxG+k] + rho*np.dot(dis,dis)
            frameStrut[2*maxG+3*k:2*maxG+3*(k+1)] = (1-rho)*frameStrut[2*maxG+3*k:2*maxG+3*(k+1)] + \
                                                          rho*pixData
            break
        else:
            frameStrut[k] = (1-alpha)*frameStrut[k]

    # In case none of the gaussians matched, adding new gaussian
    if not flagModelFit:
        if frameStrut[-1] < maxG:
            ind = int(frameStrut[-1])
            frameStrut[-1] += 1
        else:
            ind = np.argmin(np.divide(frameStrut[0:maxG], frameStrut[maxG:2*maxG]))

        frameStrut[ind] = initWt
        frameStrut[maxG+ind] = initVar
        frameStrut[2*maxG+3*ind:2*maxG+3*(ind+1)] = pixData

    # normalizing the weights
    frameStrut[0:maxG] = np.divide(frameStrut[0:maxG], np.sum(frameStrut[0:maxG]))

    # sorting based on the ratio of weight/variance
    indexing = np.argsort(np.divide(frameStrut[0:int(frameStrut[-1])], frameStrut[maxG:maxG+int(frameStrut[-1])]))
    wtSum = 0
    for k in indexing[::-1]:
        dis = pixData - frameStrut[2*maxG+3*k:2*maxG+3*(k+1)]
        var = frameStrut[maxG+k]
        if np.dot(dis, dis) < 6.25*var:
            pixData[:] = 255*np.ones(3)

        wtSum += frameStrut[k]
        if wtSum > wtsThreshold:
            break
    return np.concatenate((pixData, frameStrut))


####################################################################################################################
cap = cv.VideoCapture('umcp.mpg')
wd = int(cap.get(3))                  # width of frame
ht = int(cap.get(4))                  # height of frame
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi', fourcc, 20.0, (wd, ht))

# bigDataMat x,y axis : pixel coordinates; 
# z axis : 3 RGB + n*Wts + n*variances + 3*n*means + numGauss
bigDataMat = np.zeros((ht, wd, 4+5*maxG))
tStart = time.time()
p = Pool(processes=numProcess)
while True:
    ret, frame = cap.read()
    if ret:
        tFrameStart = time.time()
        bigDataMat[:,:,0:3] = frame
        inputData = np.split(bigDataMat.reshape(wd*ht,4+5*maxG), wd*ht)

        # mapping the work on pool of processes
        data = p.map(work, inputData)
        bigDataMat = np.reshape(np.concatenate(data,axis=0),(ht,wd,4+5*maxG))

        # converting the image to uint8 for displaying
        editFrame = np.uint8(bigDataMat[:,:,0:3])

        print('Last frame took : {:3.2f} secs'.format(time.time() - tFrameStart))
        cv.imshow('Frame', frame)
        cv.imshow('Editedframe', editFrame)
        out.write(editFrame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv.destroyAllWindows()
print('Total processing time was {0:0.2f} sec'.format(time.time() - tStart))
