import numpy as np
import cv2


leftGaussianPyr=[]
leftLaplacianPyr=[]
rightGaussianPyr = []
rightLaplacianPyr = []


def blend(d,n,left,right):

    ll=left.shape[0]%(2**n)
    rr=left.shape[1]%(2**n)
    left=cv2.resize(left,(left.shape[1]-rr,left.shape[0]-ll))
    right=cv2.resize(right,(right.shape[1]-rr,right.shape[0]-ll))


    createPyramids(left, n, right)
    res=collapsePyramids(d)
    return res

def collapsePyramids(d):
    lt = leftGaussianPyr.pop()
    rt = rightGaussianPyr.pop()
    mask=np.ones((lt.shape[0],lt.shape[1],3))
    mask[:,lt.shape[1]//2:,:]=0
    mask=cv2.GaussianBlur(mask,(d,d),0)
    res = lt*mask + rt*(1-mask)
    res=cv2.resize(res,(res.shape[1]*2,res.shape[0]*2))
    leftLaplacianPyr.pop()
    rightLaplacianPyr.pop()

    while(len(leftLaplacianPyr)>0):
        lt=leftLaplacianPyr.pop()
        rt=rightLaplacianPyr.pop()
        mask = np.ones((lt.shape[0], lt.shape[1],3))
        mask[:, lt.shape[1] // 2:,:] = 0
        mask = cv2.GaussianBlur(mask, (d, d),0)

        res=res+lt*mask+rt*(1-mask)

        res = cv2.resize(res, (res.shape[1] * 2, res.shape[0] * 2))

    return res

def createPyramids(left, n, right):
    right = np.float64(right)
    left = np.float64(left)
    tl=cv2.GaussianBlur(left, (15, 15), 0)
    tr=cv2.GaussianBlur(right, (15, 15), 0)
    leftGaussianPyr.append(tl)
    rightGaussianPyr.append(tr)
    leftLaplacianPyr.append(left-tl)
    rightLaplacianPyr.append(right-tr)
    left=tl
    right=tr
    for i in range(n):
        # creating gaussian and laplacian pyramid for left
        tempL = cv2.resize(left, (left.shape[1] // 2 , left.shape[0] // 2 ))

        tempGaussianL = cv2.GaussianBlur(tempL, (5, 5), 0)
        tempLaplacianL = tempL - tempGaussianL
        left=tempGaussianL

        leftGaussianPyr.append(tempGaussianL)
        leftLaplacianPyr.append(tempLaplacianL)

        # creating gaussian and laplacian pyramid for right

        tempR = cv2.resize(right, (right.shape[1] // 2, right.shape[0] // 2))

        tempGaussianR = cv2.GaussianBlur(tempR, (5, 5), 0)
        tempLaplacianR = tempR - tempGaussianR
        right = tempGaussianR

        rightGaussianPyr.append(tempGaussianR)
        rightLaplacianPyr.append(tempLaplacianR)

        # cv2.imwrite('Left_laplacian_' + str(i) + '.jpg',tempLaplacianL)
        # cv2.imwrite('Right_laplacian_' + str(i) + '.jpg',
        #             cv2.normalize(tempLaplacianR, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        # cv2.imwrite('Left_Gaussian_' + str(i) + '.jpg', tempGaussianL)
        # cv2.imwrite('Right_Gaussian_' + str(i) + '.jpg', tempGaussianR)


l=cv2.imread('3.source.jpg')
r=cv2.imread('4.target.jpg')

res=blend(11,3,l,r)
cv2.imwrite('res2.jpg',cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U))