import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

#to show the result image in a window that fits the screen , I create a normal sized window and display the image in it

cv2.namedWindow("window", cv2.WINDOW_NORMAL)

#-1 keeps the 16bit format and grayscale

original=cv2.imread("melons.tif",-1)

#getting the height of the original image and divide it to three images vertically
height=original.shape[0]
blue=original[:int(height/3),:]
green=original[int(height/3 )+1: int(2*height/3),:]
red=original[int(2*height/3) +1: ,:]


#since the original height might not be a multiple of 3, the height of red,blue and green might be
#different. these lines ensure they have the same height.the reason for float64 is explained in the documents.
h1=red.shape[0]
h2=green.shape[0]
h3=blue.shape[0]
h=min(h1,h2,h3)

blue=np.float64(blue[:h,:])
green=np.float64(green[:h,:])
red=np.float64(red[:h,:])


#in the following lines I create a pyramid of images with 1/2 , 1/4 , 1/8 and 1/16 size of the original image.
#h and w are the height and width of the main red,green,blue . since they all have the same width and height I used
#red to get the width and height of all three. I could also use blue or green
h=red.shape[0]
w=red.shape[1]

blueHalf=(cv2.resize(blue,(int(w/2),int(h/2))))
greenHalf=(cv2.resize(green,(int(w/2),int(h/2))))
redHalf=(cv2.resize(red,(int(w/2),int(h/2))))

bluequarter=(cv2.resize(blue,(int(w/4),int(h/4))))
greenquarter=(cv2.resize(green,(int(w/4),int(h/4))))
redquarter=(cv2.resize(red,(int(w/4),int(h/4))))

blueEighth=(cv2.resize(blue,(int(w/8),int(h/8))))
greenEighth=cv2.resize(green,(int(w/8),int(h/8)))
redEighth=(cv2.resize(red,(int(w/8),int(h/8))))


blueSixteen=(cv2.resize(blue,(int(w/16),int(h/16))))
greenSixteen=(cv2.resize(green,(int(w/16),int(h/16))))
redSixteen=(cv2.resize(red,(int(w/16),int(h/16))))
#blue is fixed


#this function is only used for the top of the pyramid i.e. the smallest size.It takes in the following parameters:
#b,g,r : the blue , green and red image (the smallest size)
#wb , hb : the width and height bound to move the red or green until the best translation is found.the actual bound is
#from -w/2 to w/2 for width and -h/2 to h/2 for height
#note that blue is fixed and only red and green are moved
def findTheBestTranslationBase(b,g,r,wb,hb):
#I want to find a translation which gives the minimum SSD ,So at first I set the ssds to infinity , so that anything
#lower than that would replace infinity.this function finds the best translation for both red and green
    rMinSSD=math.inf
    rMinIndex=(0,0)
    gMinSSD = math.inf
    gMinIndex = (0, 0)
    for ty in range(-int(hb/2),int(hb/2)):
     for tx in range(-int(wb/2),int(wb/2)):
#making a transormation matrix to use in warpAffine
         translationMatrix=np.float32([[1,0,tx],[0,1,ty]])
#translating red and green by (tx,ty) and stoing it in rtemp and g temp
         rtemp = cv2.warpAffine(r, translationMatrix,(b.shape[1],b.shape[0]))
         gtemp=cv2.warpAffine(g, translationMatrix,(b.shape[1],b.shape[0]))
#filling the empty spaces after the translation with the maximum intensity .(explained in doc)
         rtemp[rtemp==0]=65535
         gtemp[gtemp==0]=65535

#finding the SSD of green and red image based on blue after the translation
         rssd=np.sum(((b-rtemp)**2))
         gssd=np.sum(((b-gtemp)**2))

#if the SSD for each of the images (red and green ) is lower than their corrosponding current lowest SSD , we update
#the minimum SSD and store the tx , ty in which minimum occurs
         if(rssd<=rMinSSD):
             rMinSSD=rssd
             rMinIndex=(tx,ty)

         if (gssd <= gMinSSD):
            gMinSSD = gssd
            gMinIndex = (tx, ty)
#after we have checked inside the bound , the tx,ty for which the minimum ssd occurs , is returned
    return  (rMinIndex,gMinIndex)


#this function is  used for all levels of pyramid except the top .It takes in the following parameters:
#b:blue
# target:  green or red image for which we wish to find the best translation
#wb , hb :two times the  tx and ty of the level above (foe example if we are trying to find the best tx,ty of the 1/8 image
# wb=2*(the best translation of 1/16 in x directio )
#s is a constant integer used to find the lower and higher bounds like this: wb-s, wb+s for width and hb-s ,hb+s for height
#note that blue is fixed and only red and green are moved
def findTheBestTranslation(b,target,wb,hb,s):

    MinSSD=math.inf
    MinIndex=(0,0)

    for ty in range(hb-s,hb+s):
     for tx in range(wb-s,wb+s):
         translationMatrix=np.float32([[1,0,tx],[0,1,ty]])
         temp = cv2.warpAffine(target, translationMatrix,(target.shape[1],target.shape[0]))

         temp[temp==0]=65535
         ssd=np.sum((b-temp)**2)


         if(ssd<=MinSSD):
             MinSSD=ssd
             MinIndex=(tx,ty)

    return  (MinIndex)


#find the tx and ty (best translation) at 1/16 image size . (findTheBestTranslationBase finds tx and ty of both red and green
#and returns ((red tx,redty) , (green tx , green ty )
rgsixteen=findTheBestTranslationBase(blueSixteen,greenSixteen,redSixteen,redSixteen.shape[1]/2,redSixteen.shape[0]/2)

#find the tx and ty (best translation) at 1/8 image size for red and green seperatly.the bounds for each red and green are determined by
#tx and ty determined at the previous step and s which is a constant
reigth=findTheBestTranslation(blueEighth,redEighth,2*rgsixteen[0][0],2*rgsixteen[0][1],2)
geigth=findTheBestTranslation(blueEighth,greenEighth,2*rgsixteen[1][0],2*rgsixteen[1][1],2)

#find the tx and ty (best translation) at 1/4 (with the same explanations as above )
rquarter=findTheBestTranslation(bluequarter,redquarter,2*reigth[0],2*reigth[1],2)
gquarter=findTheBestTranslation(bluequarter,greenquarter,2*geigth[0],2*geigth[1],2)

#find the tx and ty (best translation) at 1/2
rhalf=findTheBestTranslation(blueHalf,redHalf,2*rquarter[0],2*rquarter[1],2)
ghalf=findTheBestTranslation(blueHalf,greenHalf,2*gquarter[0],2*gquarter[1],2)

#find the tx and ty (best translation) for the original red and green
finalRed=findTheBestTranslation(blue,red,2*rhalf[0],2*rhalf[1],2)
finalGreen=findTheBestTranslation(blue,green,2*ghalf[0],2*ghalf[1],2)


#create the translation matrix based on tx and ty received at the final step and translate the images.
rTranslationMatrix=np.float32([[1,0,finalRed[0]],[0,1,finalRed[1]]])
rTemp = cv2.warpAffine(red, rTranslationMatrix,(blue.shape[1],blue.shape[0]))

gTranslationMatrix=np.float32([[1,0,finalGreen[0]],[0,1,finalGreen[1]]])
gTemp = cv2.warpAffine(green, gTranslationMatrix,(blue.shape[1],blue.shape[0]))

#merge the channels to get the final result
result=cv2.merge((blue,gTemp,rTemp))
#normalize the 16 bit image to get the 8bit version.
#cv2.Norm_MINMAX maps the highest intensity present in the image to 255 and lowest to 0 .
result=cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

#displaying the final image in the window created at the beggining.also saving the result
cv2.imshow("window",result)
cv2.imwrite('res04.jpg',result)

cv2.waitKey(0)
cv2.destroyAllWindows();

