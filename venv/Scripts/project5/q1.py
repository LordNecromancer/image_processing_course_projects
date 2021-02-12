import numpy as np
from scipy.sparse import lil_matrix as lil_matrix,csr_matrix
from scipy.sparse import linalg as linalg
import cv2


indicesDict={}
maskY=[]
maskx=[]
def blend():
    global indicesDict


    res=targetArea.copy()
    indicesDict=positionIndicesMap()

    A=createMatrixA()




    for k in range (3):

        temp=src[:,:,k]

        b = createMatrixb(temp,targetArea[:,:,k])
        x = linalg.spsolve(A, b)
        x[x<0]=0
        x[x>255]=255
        #x=x.reshape(temp.shape[0],temp.shape[1])



        for ind,elem in enumerate(indicesDict):
            res[elem[0],elem[1],k]=int(x[ind])







    return res


def positionIndicesMap():
    global maskY,maskx
    maskY,maskx=np.where(mask>0)
    #print(len(maskx))
    temp={}
    l=np.dstack((maskY,maskx))[0]
    for ind,elem in enumerate(l) :
        temp[elem[0],elem[1]]=ind
    return temp

def createMatrixA():
    temp=np.dstack((maskY,maskx))[0]

    A = lil_matrix((len(maskx), len(maskx)),dtype=float)

    for j,i in temp:
            counter = indicesDict[j, i]

            try:
                up=indicesDict[j-1,i]
                down=indicesDict[j+1,i]
                right=indicesDict[j,i+1]
                left=indicesDict[j,i-1]
                A[counter,up]=-1
                A[counter,down]=-1
                A[counter,counter]=4
                A[counter,right]=-1
                A[counter, left] = -1
            except:
                A[counter,counter]=1


    return A

def createMatrixb(source,tgtArea):
    temp = np.dstack((maskY, maskx))[0]

    b = lil_matrix((len(maskx), 1),dtype=float)
    for j,i in indicesDict:
        counter = indicesDict[j, i]

        try:
            up = indicesDict[j - 1, i]
            down = indicesDict[j + 1, i]
            right = indicesDict[j, i + 1]
            left = indicesDict[j, i - 1]

            b[counter,0]=4*source[j,i]-source[j,i-1]-source[j,i+1]-source[j+1,i]-source[j-1,i]

        except:
            b[counter,0]=tgtArea[j,i]

    return b


src=cv2.imread('1.source.jpg')
target=cv2.imread('2.target.jpg')
mask=cv2.imread('mask.jpg')[:,:,0]
#src[src<80]=80
mask[mask<130]=0
mask[mask>130]=255
kernel = np.ones((3,3),np.uint8)
#mask=cv2.dilate(mask,kernel,iterations=1)
src=cv2.resize(src,(src.shape[1]//2,src.shape[0]//2))
mask=cv2.resize(mask,(mask.shape[1]//2,mask.shape[0]//2))
target=cv2.resize(target,(target.shape[1]//1,target.shape[0]//1))

targetArea=target[450:450+src.shape[0],815:815+src.shape[1],:]



res=blend()

result=target.copy()
result[450:450+src.shape[0],815:815+src.shape[1],:]=res

result=cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imwrite('res1.jpg',result)
