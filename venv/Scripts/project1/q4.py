import cv2
import numpy as np
import matplotlib.pyplot as plt


#import both source and refrence images and resize them in a way that their width*height is equal
src=cv2.imread("Dark.jpg")
ref=cv2.imread("Pink.jpg")
src=cv2.resize(src,(4000,3000))
ref=cv2.resize(ref,(3000,4000))

#changing the color space to hsv
srcHSV= cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
refHSV= cv2.cvtColor(ref, cv2.COLOR_BGR2HSV)

#new lists to store the CD function. the index of the list is the intensity and the value at that index is the
#value of CD function at that index. v is for value , h for hue and s for saturation.
hSrcCD=[]
hRefCD=[]
sSrcCD=[]
sRefCD=[]
vSrcCD=[]
vRefCD=[]


#making 3 new dictionaries to store the mapping between differen r and s .these dictionaries save a lot of redundant computations
hInvRefCD=dict()
sInvRefCD=dict()
vInvRefCD=dict()

#splitting the channels
h,s,v=cv2.split(srcHSV)
h1,s1,v1=cv2.split(refHSV)

#calculating the histograms of h,s,v for both source and refrence
vSrcHist=cv2.calcHist([v],[0], None, [256], [0, 256])
vRefHist=cv2.calcHist([v1],[0], None, [256], [0, 256])

hSrcHist=cv2.calcHist([h],[0], None, [180], [0, 180])
hRefHist=cv2.calcHist([h1],[0], None, [180], [0, 180])

sSrcHist=cv2.calcHist([s],[0], None, [256], [0, 256])
sRefHist=cv2.calcHist([s1],[0], None, [256], [0, 256])


#function to calculate CD . it takes in a histogram and a cd . it calculates the cd of that histogram at all
#intensities and stores the result in the cd array.
def calculateCD(hist,cd):
 for index,current in enumerate(hist):
    if(index==0):
        cd.append(current )
    else:
     cd.append(current+cd[index-1])
#using the calculateCD function to fill the CD arrays. for example to find the CD of value channel of source
#and storing it in vSrcCD we use calculateCD(vSrcHist,vSrcCD)
calculateCD(vSrcHist,vSrcCD)
calculateCD(vRefHist,vRefCD)
calculateCD(hSrcHist,hSrcCD)
calculateCD(hRefHist,hRefCD)
calculateCD(sSrcHist,sSrcCD)
calculateCD(sRefHist,sRefCD)

#this function takes three parameters . intensity which is r (intensity of the source image) it then uses srcCD to
#finds the CD of that intensity . and finds the s for which the absolute value of srcCD(intensity)-refCD(s) is minimum
# the s is returned
def findClosest(intensity,srcCD,refCD):
    m = refCD[len(refCD) - 1]
    intensityCD=srcCD[intensity][0]
    for index,current in enumerate(refCD) :

        if(abs(current-intensityCD)<=m):

            m = abs(current - intensityCD)
        else:
            return index-1
    return 255

#this function is used to fill the dictionaries storing the mapping.r is the key and s which is returned by
#findClosest is the value
def inverseRefCD(srcCD,refCD,invRefCD):
 for t in range(len(srcCD)):

  invRefCD[t]=findClosest(t,srcCD,refCD)
#filling the dictionaries
inverseRefCD(hSrcCD,hRefCD,hInvRefCD)
inverseRefCD(sSrcCD,sRefCD,sInvRefCD)
inverseRefCD(vSrcCD,vRefCD,vInvRefCD)

#creating 3 empty new arrays with the source dimensions to store the mapped values
nv=np.empty([v.shape[0],v.shape[1]],dtype=int)
ns=np.empty([v.shape[0],v.shape[1]],dtype=int)
nh=np.empty([v.shape[0],v.shape[1]],dtype=int)

#filling the empty arrays with the new values. note that the size of vSrcCD is equal to sSrcCD and I could use either
#of them in the loop
for i in  range(len(vSrcCD)):
    nv[v==i]=vInvRefCD[i]
    ns[s==i]=sInvRefCD[i]
#since hSrcCD has a different size(0 to 179) I use a different loop
for i in  range(len(hSrcCD)):
    nh[h==i]=hInvRefCD[i]

#merging the new arrays and making sure they are 8 bit.then changing the color space
nh=np.uint8(nh)
nv=np.uint8(nv)
ns=np.uint8(ns)
result=cv2.merge((nh,ns,nv))
result=cv2.cvtColor(result, cv2.COLOR_HSV2BGR)



#calculating the histogram of each channel of result
hResHist=cv2.calcHist([nh],[0], None, [180], [0, 180])
vResHist=cv2.calcHist([nv],[0], None, [256], [0, 256])
sResHist=cv2.calcHist([ns],[0], None, [256], [0, 256])

#saving  the histogram of result and the result itself

cv2.imwrite('res06.jpg',result)



plot, axis = plt.subplots(2, 2)
plot.tight_layout()
axis[0, 0].plot(hResHist)
axis[0, 0].set_title('hue')
axis[0, 1].plot(vResHist)
axis[0, 1].set_title('value')
axis[1, 0].plot(sResHist)
axis[1, 0].set_title('saturation')


plt.savefig("res05.jpg")


