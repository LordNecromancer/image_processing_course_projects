import numpy as np
import cv2
import time
import skimage.segmentation


class Cluster:
    def __init__(self,centerWidth,centerHeight,lab,ind):
        self.cw=centerWidth
        self.ch=centerHeight
        self.clab=lab
        self.pixels=[]
        self.index=ind


    def updateCenter(self,center):
        self.cw=center[0][0]
        self.ch=center[0][1]
        self.clab=center[1]



class Slic:
    def __init__(self,image,k,maxIter,alpha):

       self.clusters = []
       self.image=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
       self.threshold=10**5
       self.labels = np.empty((image.shape[0], image.shape[1]),dtype=np.int)
       self.minDist = np.full((image.shape[0], image.shape[1]),fill_value=np.inf,dtype=np.float64)
       self.k=k
       self.alpha=alpha
       self.errorTol=10
       self.maxIter=maxIter

    def slic(self):

       self.positions=self.calculatePositions()
       self.initializeClusters(self.image,self.k)
       print(len(self.clusters))
       self.adjustCenters(self.image)
       self.calculateSuperPixels()
       self.iterateClusters()
       self.image=cv2.cvtColor(self.image,cv2.COLOR_LAB2BGR)
       self.findEdges()

       return self.image

    def calculatePositions(self):
        pos=np.indices((self.image.shape[0],self.image.shape[1]),dtype=np.int)
        pos=np.dstack((pos[1],pos[0]))

        return pos
    def initializeClusters(self,image,k):

        w=self.image.shape[1]
        h=self.image.shape[0]
        if(np.sqrt(k)==int(np.sqrt(k))):
            numW=np.sqrt(k)
            numH=np.sqrt(k)
        else:
            t=k/2
            numW = np.sqrt(t)*2
            numH = np.sqrt(t)

        sw=np.uint64(w/numW)
        sh=np.uint64(h/numH)
        self.s=np.int(min(sh,sw))
        n=0
        for i in range(int(sw/2),w,sw):
            for j in range(int(sh/2),h,sh):
                c=Cluster(i,j,image[j,i,:],n)
                self.clusters.append(c)
                n+=1


    def adjustCenters(self,image):

        for cluster in self.clusters:
            min=np.inf
            minIndex=(cluster.cw,cluster.ch)
            for i in range(cluster.cw-2,cluster.cw+2):
                for j in range(cluster.ch - 2, cluster.ch + 2):

                   try:

                        gradient=np.sum((image[j,i+1,:]-image[j,i-1,:])**2)+np.sum((image[j+1,i,:]-image[j-1,i,:])**2)

                        if gradient<min:
                            minIndex=(i,j)
                            min=gradient
                   except:
                       continue
            cluster.updateCenter((minIndex,image[minIndex[1],minIndex[0],:]))



    def calculateSuperPixels(self):
        for cluster in self.clusters:
            widthLowerBound=cluster.cw-self.s if cluster.cw-self.s>=0 else 0
            widthHigherBound=cluster.cw+self.s if cluster.cw+self.s < self.image.shape[1] else self.image.shape[1]
            heightLowerBound=cluster.ch - self.s if cluster.ch-self.s>=0 else 0
            heightHigherBound=cluster.ch + self.s if cluster.ch+self.s < self.image.shape[0] else self.image.shape[0]

            temp=self.image[heightLowerBound: heightHigherBound,widthLowerBound:widthHigherBound,:]
            tempPos=self.positions[heightLowerBound: heightHigherBound,widthLowerBound:widthHigherBound]

            dist = self.calculateDist((tempPos, temp), ((cluster.cw, cluster.ch), cluster.clab))
            mask=np.zeros((tempPos.shape[0],tempPos.shape[1]))
            mask[dist<self.minDist[heightLowerBound: heightHigherBound,widthLowerBound:widthHigherBound]]=1
            self.minDist[heightLowerBound: heightHigherBound, widthLowerBound:widthHigherBound][mask>0]=dist[mask>0]





            self.labels[heightLowerBound: heightHigherBound, widthLowerBound:widthHigherBound][mask>0]=cluster.index


    def calculateDist(self,pixels,center):

        pSpace=pixels[0]
        plab=pixels[1]
        cSpace=center[0]
        clab=center[1]

        centerPos=np.full(pSpace.shape,fill_value=cSpace)
        centerLab=np.full(plab.shape,fill_value=clab)



        alpha=self.alpha

        dlab=np.sum((plab-centerLab)**2,axis=2)
        dxy=np.sum((pSpace-centerPos)**2,axis=2)
        #print(centerPos,pSpace)

        dist=dlab+alpha*dxy


        return dist


    def iterateClusters(self):
      iteration=0
      error=np.inf
      while(self.errorTol<error and iteration<self.maxIter)  :
        print(error)
        iteration+=1
        error=0
        print(iteration)
        s=0
        for cluster in self.clusters:
            widthLowerBound = cluster.cw - self.s if cluster.cw - self.s >= 0 else 0
            widthHigherBound = cluster.cw + self.s if cluster.cw + self.s < self.image.shape[1] else self.image.shape[1]-1
            heightLowerBound = cluster.ch - self.s if cluster.ch - self.s >= 0 else 0
            heightHigherBound = cluster.ch + self.s if cluster.ch + self.s < self.image.shape[0] else self.image.shape[0]-1


            ind=cluster.index

            p= self.positions[heightLowerBound: heightHigherBound, widthLowerBound:widthHigherBound][self.labels[heightLowerBound: heightHigherBound, widthLowerBound:widthHigherBound]==ind]
            meanxy=np.mean(p,axis=0)
            meanlab=self.image[int(meanxy[1]),int(meanxy[0]),:]
            error+=(cluster.cw-int(meanxy[0]))**2+(cluster.ch-int(meanxy[1]))**2
            cluster.updateCenter(((int(meanxy[0]),int(meanxy[1])),meanlab))
        self.calculateSuperPixels()


    def findEdges(self):

        self.image=skimage.segmentation.mark_boundaries(self.image,self.labels,(0,0,0))
        self.image = cv2.normalize(self.image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # temp=np.zeros((self.image.shape[0],self.image.shape[1]))
          #
          # for j in range(1,self.labels.shape[0]-1):
          #
          #     for i in range(1,self.labels.shape[1]-1):
          #         cur=self.labels[j,i]
          #
          #         up=self.labels[j-1,i]
          #         down=self.labels[j+1,i]
          #         right=self.labels[j,i+1]
          #         left=self.labels[j,i-1]
          #
          #         neighbours=[left,right,down,up]
          #         directions=[(-1,0),(1,0),(0,1),(0,-1)]
          #         corners=[(-1,-1),(1,-1),(-1,1),(1,1)]
          #
          #         shouldDraw=False
          #
          #
          #
          #         for ind,n in enumerate(neighbours):
          #                if(n!=cur and temp[j+directions[ind][1],i+directions[ind][0]]==0):
          #                        shouldDraw=True
          #                        break

                  # for c in corners:
                  #     if (temp[c[0]+j, c[1]+i] ==1):
                  #                        shouldDraw = False
                  #                        break
                  # if(shouldDraw):
                  #     temp[j, i] = 1


                     # self.image[j,i,:]=(0,-128,-128)
          #temp=cv2.morphologyEx( temp,cv2.MORPH_CLOSE,np.ones((9,9)))
          #temp=cv2.dilate(temp,np.ones((7,7)))
          #temp=cv2.GaussianBlur(np.uint8(temp),(5,3),0)
          # j,i=np.where(temp==1)
          # self.image[j,i,:]=(0,-128,-128)



image=cv2.imread('slic.jpg')
res05=Slic(image,64,30,0.0085).slic()
res06=Slic(image,256,30,0.025).slic()
res07=Slic(image,1024,30,0.082).slic()
res08=Slic(image,2048,30,0.2).slic()

cv2.imwrite("res05.jpg",res05)
cv2.imwrite("res06.jpg",res06)
cv2.imwrite("res07.jpg",res07)
cv2.imwrite("res08.jpg",res08)


