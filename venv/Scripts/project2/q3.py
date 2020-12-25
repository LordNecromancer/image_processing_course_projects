import cv2
import numpy as np
import math

original=cv2.imread('books.jpg')
topInit=(318,104)
middleInit=(155,429)
bottomInit=(421,668)


topPosition=np.array([[666-topInit[0],208-topInit[1]],[601-topInit[0],394-topInit[1]],[318-topInit[0],287-topInit[1]],[382-topInit[0],104-topInit[1]]])
middlePosition=np.array([[364-middleInit[0],741-middleInit[1]],[155-middleInit[0],707-middleInit[1]],[205-middleInit[0],429-middleInit[1]],[411-middleInit[0],469-middleInit[1]]])
bottomPosition=np.array([[813-bottomInit[0],969-bottomInit[1]],[610-bottomInit[0],1100-bottomInit[1]],[421-bottomInit[0],796-bottomInit[1]],[623-bottomInit[0],668-bottomInit[1]]])


def calculateBookDimension(v):
    small=int(np.sqrt((v[1]-v[0])[0]**2+(v[1]-v[0])[1]**2))+1
    big=int(np.sqrt((v[1]-v[2])[0]**2+(v[1]-v[2])[1]**2))+1

    return (small,big)

topDim=calculateBookDimension(topPosition)
middleDim=calculateBookDimension(middlePosition)
bottomDim=calculateBookDimension(bottomPosition)

topDst=np.array([[0,0],[topDim[0],0],[topDim[0],topDim[1]],[0,topDim[1]]])
middleDst=np.array([[0,0],[middleDim[0],0],[middleDim[0],middleDim[1]],[0,middleDim[1]]])
bottomDst=np.array([[0,0],[bottomDim[0],0],[bottomDim[0],bottomDim[1]],[0,bottomDim[1]]])


topHInv=np.array(np.linalg.inv(cv2.findHomography(topPosition,topDst)[0]))
middleHInv=np.array(np.linalg.inv(cv2.findHomography(middlePosition,middleDst)[0]))
bottomHInv=np.array(np.linalg.inv(cv2.findHomography(bottomPosition,bottomDst)[0]))


top=original[104:394,318:666,:]
middle=original[429:741,155:411,:]
bottom=original[668:1100,421:813,:]



def findMapValue(image,i,j,invH):
    m=(np.matmul(invH,np.matrix.transpose(np.array([i,j,1]))))
    m[0]=m[0]/m[2]
    m[1]=m[1]/m[2]
    a=m[1]-int(m[1])
    b=m[0]-int(m[0])

    m[0]=int(m[0])
    m[1]=int(m[1])

    if((m[1]>=image.shape[0]-1 or m[0]>=image.shape[1]-1 or m[0]<0 or m[1]<0)):
        return (0,0,0)

    result=[]
    for k in range(3):


        channel=image[:,:,k]
        temp=np.matmul(np.array([1-a,a]),np.array(
        [
            [channel[int(m[1]),int(m[0])],channel[int(m[1]+1),int(m[0])]],
            [channel[int(m[1]),int(m[0]+1)],channel[int(m[1]+1),int(m[0]+1)]]

        ]))

        temp=np.matmul(temp,np.array([[1-b],[b]]))
        result.append(temp)


    return result


def project(image,invH,dim):
    result=np.empty([dim[1],dim[0],3],dtype=int)
    for i in range(result.shape[1]):
        for j in range(result.shape[0]):

          value=findMapValue(image,i,j,invH)
          result[j,i,:]=value

    return result


finalTop=np.uint8(project(top,topHInv,topDim))
finalMiddle=np.uint8(project(middle,middleHInv,middleDim))
finalBottom=np.uint8(project(bottom,bottomHInv,bottomDim))





cv2.imwrite("res04.jpg",finalTop)
cv2.imwrite("res05.jpg",finalMiddle)
cv2.imwrite("res06.jpg",finalBottom)




