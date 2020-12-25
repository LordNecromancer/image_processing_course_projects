import numpy as np
import cv2


alpha=0.004
gamma=180
delta=250
vertices=[]
im=cv2.imread("tasbih.jpg")
images=[]
postions=[]

edges = cv2.Canny(im,100,170,L2gradient=True)
center=[0,0]
dbar=0


def getInternalEnergy(curr,prev):

    return alpha*(((curr[0]-prev[0])**2+(curr[1]-prev[1])**2 - (dbar))**2)

def getExternalEnergy(curr):
    return -gamma*(edges[curr[0],curr[1]]**2)


def getRadialEnergy(curr):
    return delta*((curr[0]-center[1])**2+(curr[1]-center[0])**2)

def calculateDbar():
    dist=0
    for ind,v in enumerate(vertices):
        p=vertices[((ind-1)%len(vertices))]
        dist+=((v[0] - p[0]) ** 2 + (v[1] - p[1]) ** 2)
    return dist/len(vertices)


def drawLines(temp):
    for ind,v in enumerate(vertices):
        cv2.line(temp,(v[1],v[0]),(vertices[(ind+1)%len(vertices)][1],vertices[(ind+1)%len(vertices)][0]),(0,255,0),2)


def initializeContour(num,radius):
  global center
  global dbar
  temp = im.copy()
  print(len(vertices))

  if(len(vertices)==0):

    center=(int(im.shape[0]/2)-20,int(im.shape[1]/2)-130)
    cv2.circle(im,center,5,(255,0,0))
    for i in range(num):
        p=(int(np.sin(i*2*np.pi/num)*radius)+center[1],int(np.cos(i*2*np.pi/num)*radius)+center[0])
        cv2.circle(temp,(p[1],p[0]),5,(0,0,255))
        vertices.append(p)
  else:
      calculateCenter()
      for v in vertices:
          cv2.circle(temp, (v[1], v[0]), 5, (0, 0, 255))
  drawLines(temp)
  dbar=calculateDbar()
  images.append(temp)


def calculateCenter():
    global center
    sumX = 0
    sumY = 0
    for v in vertices:

        sumX += v[1]
        sumY += v[0]
    center = (int(sumY / len(vertices)), int(sumX / len(vertices)))


def moveContour(m):

    global postions
    postions = getPositions(m)

    for i in range(400):
        paths,cost=findPaths(m)
        s=findBestStart(m,paths,cost)
        moveVertices(m,s,paths)

def getPositions(m):
    p = []

    for j in range(int(-m / 2), int(m / 2)+1):
        for i in range(int(-m / 2), int(m / 2)+1):
            p.append([j, i])
    return p


def findPaths(m):
    paths = np.empty((m ** 2, len(vertices))).astype(np.int8)
    currentCost = np.zeros(m ** 2)




    for ind, v in enumerate(vertices):

        p = vertices[ind-1 ]
        tempCurrentCost=currentCost.copy()


        for ind1,i in enumerate(postions):
            tempEnergy = np.inf

            tempMin = 0
            currIndex=(v[0]+i[0],v[1]+i[1])
            external = getExternalEnergy(currIndex)
            radial=getRadialEnergy(currIndex)


            for ind2,j in enumerate(postions):
                pIndex=(p[0]+j[0],p[1]+j[1])
                energy = external +radial+ getInternalEnergy(currIndex, pIndex) + currentCost[ind2]
                if (energy < tempEnergy):
                    tempEnergy = energy
                    tempMin = ind2

            paths[ind1, ind] = tempMin
            tempCurrentCost[ind1] =  tempEnergy
        currentCost=tempCurrentCost
    return (paths,currentCost)

def findBestStart(m,paths,cost):

    for ind,i in enumerate(postions):
        p = vertices[len(vertices) - 1]
        s=vertices[0]
        sInd = ind
        n=len(vertices)-1
        while n>0:
            sInd=paths[sInd,n]
            n-=1



        startIndex=(s[0]+postions[sInd][0],s[1]+postions[sInd][1])

        pIndex = (p[0]+i[0],p[1]+i[1])
        energy = getExternalEnergy(startIndex) + getInternalEnergy(startIndex, pIndex)+ getRadialEnergy(pIndex)


        cost[ind]+=energy

    cost=list(cost)
    m=cost.index(min(cost))
    return m

def moveVertices(m,s,paths):
    global dbar
    temp=im.copy()
    sInd = s
    dist=0
    for n in range (len(vertices) - 1,0,-1):

        newInd=(vertices[n][0] +postions[sInd][0], vertices[n][1] +postions[sInd][1])
        vertices[n]=newInd
        cv2.circle(temp,(newInd[1],newInd[0]),5,(0,0,255))
        if(n!=len(vertices) - 1):
            dist+=((vertices[n+1][0] - vertices[n][0]) ** 2 + ((vertices[n+1][1] - (vertices[n][1])) ** 2))


        sInd = paths[sInd,n]
        n -= 1
    # for the first vertix
    newInd = (vertices[0][0] + postions[sInd][0], vertices[0][1] + postions[sInd][1])
    vertices[0] = newInd
    cv2.circle(temp, (newInd[1], newInd[0]), 5, (0, 0, 255))
    dist += ((vertices[0][0] - vertices[len(vertices) - 1][0]) ** 2 + ((vertices[0][1] - (vertices[len(vertices) - 1][1])) ** 2))
    dbar=dist/len(vertices)

    drawLines(temp)
    calculateCenter()

    images.append(temp)

# def drawCircle(e,x,y,flag,parameters):
#     if e == cv2.EVENT_LBUTTONDOWN:
#
#         cv2.circle(im, (x, y), 5, (0, 0, 255))


def onClick(e, x, y,flag,parameters):
    if e == cv2.EVENT_LBUTTONDOWN:
        vertices.append((y,x))


cv2.imshow("window", im)
cv2.namedWindow('window')
#cv2.setMouseCallback('window', drawCircle)

cv2.setMouseCallback('window', onClick)

cv2.waitKey(0)
cv2.destroyAllWindows()
input("press key to continue")


initializeContour(100,250)
moveContour(3)
print(len(images))
video=cv2.VideoWriter("contour.mp4",cv2.VideoWriter_fourcc(*'mp4v'),40,(im.shape[1],im.shape[0]))

for i in range(len(images)):

    video.write(images[i])

video.release()
cv2.imwrite("edges.jpg",edges)
cv2.imwrite("res10.jpg",images.pop())
