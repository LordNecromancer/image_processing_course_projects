import cv2
import numpy as np
import math


n=np.float32(cv2.imread("q4_01_near.jpg"))
f=np.float32(cv2.imread("q4_02_far.jpg"))

#(widths,heights)


npoints=np.array([[611,823],[386,378],[1,1]])


fpoints=np.array([[303,420],[392,392],[1,1]])

hf,wf,c=f.shape
hn,wn,c=n.shape


def rotate(image,name,points):
    for theta in range(-20,30):
        if(points[1][0]==points[1][1]):
            cv2.imwrite(name, image)
            return image
        rotationMatrix=cv2.getRotationMatrix2D((points[0][0],points[1][0]),theta,1)
        newY=np.matmul(rotationMatrix,points)[1][1]

        if(np.abs(newY-points[1][0])<=2):
            newImage=cv2.warpAffine(image,rotationMatrix,(image.shape[1],image.shape[0]))

            cv2.imwrite(name,newImage)
            return newImage
    newImage = cv2.warpAffine(image, rotationMatrix, (image.shape[1], image.shape[0]))

    cv2.imwrite(name, image)
    return image

def scale(near,far):
    global fpoints,npoints
    ndist = npoints[0][1] - npoints[0][0]
    fdist = fpoints[0][1] - fpoints[0][0]


    if(ndist>fdist):
        scale = ndist / fdist

        far = cv2.resize(far, (int(scale * far.shape[1]), int(scale * far.shape[0])))
        fpoints=fpoints*scale
        cv2.imwrite("q4_04_far.jpg",far)
        return (near,far)

    elif (ndist < fdist):
        scale = fdist/ndist
        near = cv2.resize(near, (int(scale * near.shape[1]), int(scale * near.shape[0])))
        npoints=npoints*scale

        cv2.imwrite("q4_03_near.jpg", near)
        return (near,far)


def translate(near,far):


    hf, wf, c = far.shape
    hn, wn, c = near.shape
    baseX=far if wf>=wn else near

    dx = npoints[0][0] - fpoints[0][0]
    dy = npoints[1][0] - fpoints[1][0]

    if(baseX is near):
        dx=-dx

    xTranslationMatrix = np.float32([[1, 0, dx], [0, 1, 0]])
    temp = cv2.warpAffine(baseX, xTranslationMatrix, (min(wn,wf), baseX.shape[0]))
    if(baseX is far) :
        cv2.imwrite("q4_04_far.jpg",temp)
        far=temp
        hf, wf, c = far.shape
    else:
        cv2.imwrite("q4_03_near.jpg", temp)
        near = temp
        hn, wn, c = near.shape

    baseY=far if hf>=hn else near
    if (baseY is near):
        dy = -dy

    yTranslationMatrix = np.float32([[1, 0, 0], [0, 1, dy]])
    temp1 = cv2.warpAffine(baseY, yTranslationMatrix, (min(wn, wf), min(hn, hf)))
    if (baseY is far):
        cv2.imwrite("q4_04_far.jpg", temp1)
    else:
        cv2.imwrite("q4_03_near.jpg", temp1)


n=rotate(n,"q4_03_near.jpg",npoints)
f=rotate(f,"q4_04_far.jpg",fpoints)
n,f=scale(n,f)
translate(n,f)

near=cv2.imread("near.jpg")[100:-100,200:-100,:]
far=cv2.imread("far.jpg")[100:-100,200:-100,:]


resultN=[]
resultF=[]
for k in range(3):
 fnear=np.fft.fftshift(np.fft.fft2(near[:,:,k]))
 ffar=np.fft.fftshift(np.fft.fft2(far[:,:,k]))

 logMagNear=np.log10(np.abs(fnear)+np.ones(ffar.shape))
 logMagFar=np.log10(np.abs(ffar)+np.ones(ffar.shape))
 logMagNear=cv2.normalize(logMagNear, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
 logMagFar=cv2.normalize(logMagFar, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
 resultN.append(logMagNear)
 resultF.append(logMagFar)

logMagNear=cv2.merge(resultN)
logMagFar=cv2.merge(resultF)


def getGaussianFilter(sigma,image,isLowPass,cutoff):

    filter=np.empty([image.shape[0],image.shape[1]])
    cutoffFilter=np.empty([image.shape[0],image.shape[1]])

    ux=int(image.shape[1]/2)
    uy=int(image.shape[0]/2)
    for i in range(image.shape[1]):
        for j in range(image.shape[0]):
            if(isLowPass):
                filter[j, i] = math.exp(-1.0 * ((i - ux) ** 2 + (j - uy) ** 2) / (2 * sigma ** 2))

                if(np.sqrt((i-ux)**2+(j-uy)** 2)<cutoff):
                    cutoffFilter[j, i] = math.exp(-1.0 * ((i - ux) ** 2 + (j - uy) ** 2) / (2 * sigma ** 2))
            else:
                filter[j, i] = 1-(math.exp(-1.0 * ((i - ux) ** 2 + (j - uy) ** 2) / (2 * sigma ** 2)))

                if (np.sqrt((i-ux)**2+(j-uy)** 2) > cutoff):
                    cutoffFilter[j, i] = 1-(math.exp(-1.0 * ((i - ux) ** 2 + (j - uy) ** 2) / (2 * sigma ** 2)))




    return (filter,cutoffFilter)





lowPassFilter,cutoffL=getGaussianFilter(20,far[:,:,0],True,20)
highPassFilter,cutoffH=getGaussianFilter(40,near[:,:,0],False,10)

normalizedLowPassFilter=cv2.normalize(lowPassFilter, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
normalizedHighPassFilter=cv2.normalize((highPassFilter), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

def applyFilter(filter,image):
    res=np.empty([filter.shape[0],filter.shape[1],3],dtype='complex_')
    for k in range(3):
        x = np.fft.fftshift(np.fft.fft2(image[:, :, k]))

        res[:,:,k]=(filter * x)
    return res

lowFiltered=applyFilter(cutoffL,far)
highFiltered=applyFilter(cutoffH,near)

lowFiltered[highFiltered!=0]=1*lowFiltered[highFiltered!=0]/2
highFiltered[lowFiltered!=0]=1*highFiltered[lowFiltered!=0]/2
hybridFrequency =highFiltered+lowFiltered
hybridNear=np.log10(np.abs(highFiltered)+np.ones(highFiltered.shape))
hybridFar=np.log10(np.abs(lowFiltered)+np.ones(lowFiltered.shape))



spatial=[]
for k in range(3):
 hybridSpatial=np.real(np.fft.ifft2(np.fft.ifftshift(hybridFrequency[:,:,k])))

 spatial.append((hybridSpatial))

hybridSpatial=cv2.merge(spatial)
resizedHybridSpatial=cv2.resize(hybridSpatial,(int(hybridSpatial.shape[1]/6),int(hybridSpatial.shape[0]/6)))

hybridFrequency=np.log10(np.abs(hybridFrequency))
hybridNear=cv2.normalize(hybridNear, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
hybridFar=cv2.normalize(hybridFar, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
hybridFrequency=cv2.normalize(hybridFrequency, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
hybridSpatial=cv2.normalize(hybridSpatial, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
resizedHybridSpatial=cv2.normalize(resizedHybridSpatial, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

highpassCutoff=cv2.normalize(np.log10(cutoffH+np.ones(highPassFilter.shape)), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
lowpassCutoff=cv2.normalize(np.log10(cutoffL+np.ones(lowPassFilter.shape)), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


cv2.imwrite("Q4_11_highpassed.jpg",hybridNear)
cv2.imwrite("Q4_12_lowpassed.jpg",hybridFar)
cv2.imwrite("Q4_13_hybrid_frequency.jpg",hybridFrequency)
cv2.imwrite("Q4_14_hybrid_near.jpg",hybridSpatial)
cv2.imwrite("Q4_15_hybrid_far.jpg",resizedHybridSpatial)
cv2.imwrite("Q4_09_highpass_cutoff.jpg",highpassCutoff)
cv2.imwrite("Q4_010_lowpass_cutoff.jpg",lowpassCutoff)

cv2.imwrite("Q4_07_highpass_40.jpg",normalizedHighPassFilter)
cv2.imwrite("Q4_08_lowpass_20.jpg",normalizedLowPassFilter)
cv2.imwrite("q4_05_dft_near.jpg",logMagNear)
cv2.imwrite("q4_06_dft_far.jpg",logMagFar)