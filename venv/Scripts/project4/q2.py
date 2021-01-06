from imutils import face_utils
import dlib
import cv2
import numpy as np
import skimage.color as color
from PIL import Image, ImageDraw


triangleDict={}
triangleByIndex=[]
images=[]
interval=0.01
itr=0
distanceVector=[]
initialPos=[]
targetPos=[]

def morph(initial,target):
    global distanceVector,initialPos,targetPos

    trainedPredictor = "shape_predictor_68_face_landmarks.dat"
    detect = dlib.get_frontal_face_detector()
    predict = dlib.shape_predictor(trainedPredictor)

    initialFace=detect(initial,1)[0]
    initialPos=predict(initial,initialFace)
    targetFace=detect(target,1)[0]
    targetPos=predict(target,targetFace)
    initialPos=face_utils.shape_to_np(initialPos)
    targetPos=face_utils.shape_to_np(targetPos)
    initialPos=np.concatenate((initialPos,initialBoundPoints))
    targetPos=np.concatenate((targetPos,targetBoundPoints))
    #print(initialPos,targetPos)
    distanceVector=targetPos-initialPos
    initialTriangle=cv2.Subdiv2D((0,0,initial.shape[1],initial.shape[0]))

    for ind,(i,j) in enumerate(initialPos):
        initialTriangle.insert((i,j))
        triangleDict[(i,j)]=ind


    makeTriangleByIndex(initialTriangle)
    #drawTriangles()
    startMorphing()

def startMorphing():
    global interval,itr

    while( itr<=1):
        print(itr)
        itr+=interval
        landmarks=makeIntermediateLandmarks()
        initialTransform=findAffinTransform(landmarks,type="initial")
        targetTransform=findAffinTransform(landmarks,type="target")
        initialImage=makeIntermediateImage(initial,landmarks,initialTransform,type='initial')
        targetImage=makeIntermediateImage(target,landmarks,targetTransform,type='target')
        image=(1-itr)*initialImage+itr*targetImage
        image=cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        images.append(image)




def makeIntermediateImage(image,landmarks,transform,type):
    height,width=image.shape[0],image.shape[1]
    newImage=np.empty([height,width,3],dtype=int)
    dest = initialPos if type == "initial" else targetPos

    for ind,t in enumerate(triangleByIndex):
        destTriangle=[[dest[t[0]][0],dest[t[0]][1]],[dest[t[1]][0],dest[t[1]][1]],[dest[t[2]][0],dest[t[2]][1]]]

        targetTriangle=[[landmarks[t[0]][0],landmarks[t[0]][1]],[landmarks[t[1]][0],landmarks[t[1]][1]],[landmarks[t[2]][0],landmarks[t[2]][1]]]
        destTriangle=tuple(map(tuple,destTriangle))
        temp = Image.new('L', (width, height), 0)
        ImageDraw.Draw(temp).polygon(destTriangle, outline=1, fill=1)

        targetTriangle = tuple(map(tuple, targetTriangle))
        tempT = Image.new('L', (width, height), 0)
        ImageDraw.Draw(tempT).polygon(targetTriangle, outline=1, fill=1)

        mask1 = np.array(temp)

        mask2 = np.array(tempT)
        tempImage=np.empty([height,width,3],dtype=int)

        for k in range(3):
            channel=tempImage[:,:,k]
            channel=cv2.warpAffine(image[:,:,k],transform[ind],(width, height))
            tempImage[:,:,k]=channel
        tempImage=cv2.normalize(tempImage, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        #cv2.imwrite('tempImage.jpg', tempImage)

        newImage[mask2>0]=tempImage[mask2>0]
        #cv2.imwrite('newImage.jpg', newImage)

        # indices=np.nonzero(mask)
        # indices=np.dstack((indices[1],indices[0]))[0]
        # for (i,j) in indices:
        #     newImage[j,i,:]=findMapValue(image,i,j,transform[ind])

    return newImage






def findAffinTransform(landmarks,type):
    affinTransforms=[]
    for t in triangleByIndex:
        dest=initialPos if type=="initial" else targetPos
        srcPoints=np.array([[landmarks[t[0]][0],landmarks[t[0]][1]],[landmarks[t[1]][0],landmarks[t[1]][1]],[landmarks[t[2]][0],landmarks[t[2]][1]]],dtype=np.float32)
        destPoints=np.array([[dest[t[0]][0],dest[t[0]][1]],[dest[t[1]][0],dest[t[1]][1]],[dest[t[2]][0],dest[t[2]][1]]],dtype=np.float32)
        affin=cv2.getAffineTransform(destPoints,srcPoints)
        affinTransforms.append(affin)
    return affinTransforms

def findMapValue(image,i,j,affinTransform):
    m=(np.matmul(affinTransform,np.matrix.transpose(np.array([i,j,1]))))
    # m[0]=m[0]/m[2]
    # m[1]=m[1]/m[2]
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
def makeIntermediateLandmarks():
    intermediateLandmarks=[]
    for ind,p in enumerate(initialPos):
        intermediateLandmarks.append(p+distanceVector[ind]*itr)
    return intermediateLandmarks



def makeTriangleByIndex(initialTriangle):
    for p in initialTriangle.getTriangleList():
        pt1 = (p[0], p[1])
        indexpt1 = triangleDict[pt1]

        pt2 = (p[2], p[3])
        indexpt2 = triangleDict[pt2]

        pt3 = (p[4], p[5])
        indexpt3 = triangleDict[pt3]

        triangleByIndex.append([indexpt1, indexpt2, indexpt3])


#print(triangleDict)
def drawTriangles():
    for h in triangleByIndex:
        initialpt1 = initialPos[h[0]]
        targetpt1 = targetPos[h[0]]
        initialpt2 = initialPos[h[1]]
        targetpt2 = targetPos[h[1]]
        initialpt3 = initialPos[h[2]]
        targetpt3 = targetPos[h[2]]

        cv2.line(initial, (initialpt1[0], initialpt1[1]), (initialpt2[0], initialpt2[1]), (0, 0, 255), 1)
        cv2.line(initial, (initialpt2[0], initialpt2[1]), (initialpt3[0], initialpt3[1]), (0, 0, 255), 1)
        cv2.line(initial, (initialpt3[0], initialpt3[1]), (initialpt1[0], initialpt1[1]), (0, 0, 255), 1)

        cv2.line(target, (targetpt1[0], targetpt1[1]), (targetpt2[0], targetpt2[1]), (0, 0, 255), 1)
        cv2.line(target, (targetpt2[0], targetpt2[1]), (targetpt3[0], targetpt3[1]), (0, 0, 255), 1)
        cv2.line(target, (targetpt3[0], targetpt3[1]), (targetpt1[0], targetpt1[1]), (0, 0, 255), 1)


#if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):




initial=cv2.imread('me.jpg')
target=cv2.imread("me2.jpg")
minW=min(initial.shape[1],target.shape[1])
minH=min(initial.shape[0],target.shape[0])

initial=cv2.resize(initial,(minW//2,minH//2))
target=cv2.resize(target,(minW//2,minH//2))

cv2.imwrite('src.jpg',initial)
cv2.imwrite('trgt.jpg',target)


initialBoundPoints=np.array([[0,0],[initial.shape[1]//2,0],[initial.shape[1]-1,0],[0,initial.shape[0]//2],[initial.shape[1]-1,initial.shape[0]//2],[0,initial.shape[0]-1],[initial.shape[1]//2,initial.shape[0]-1],[initial.shape[1]-1,initial.shape[0]-1]])
targetBoundPoints=np.array([[0,0],[target.shape[1]//2,0],[target.shape[1]-1,0],[0,target.shape[0]//2],[target.shape[1]-1,target.shape[0]//2],[0,target.shape[0]-1],[target.shape[1]//2,target.shape[0]-1],[target.shape[1]-1,target.shape[0]-1]])



morph(initial,target)
cv2.imwrite('initial_face_landmarks.jpg',initial)
cv2.imwrite('target_face_landmarks.jpg',target)
video=cv2.VideoWriter("res2.mp4",cv2.VideoWriter_fourcc(*'mp4v'),15,(target.shape[1],target.shape[0]))
print(len(images))
for i in range(len(images)):

    video.write(images[i])

video.release()
