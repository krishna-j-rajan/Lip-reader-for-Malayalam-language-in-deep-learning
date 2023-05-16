import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageFilter
import os
import pandas as pd
import sys
import os
import dlib
import glob
import cv2
import re
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
from imutils import face_utils
import numpy as np
import mahotas
import mahotas.demos
from mahotas.thresholding import soft_threshold
from pylab import imshow, show
from os import path
import pickle

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('Project_Extra/shape_predictor_68_face_landmarks.dat')


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)',text) ]


def hog1(img):
    fd, hog_image = hog(img, orientations=6, pixels_per_cell=(2, 2), cells_per_block=(3, 3), visualize=True, multichannel=True)
    return hog_image

def surf1(img):
    surf = cv2.xfeatures2d.SURF_create(400)
    kp, des = surf.detectAndCompute(img,None)
    
def haar(img):
    h = mahotas.haar(img)
    return h


def crop_mouth(img):
    # img = cv2.imread(f)
    img1=img
    img2=img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    

    dets = detector(img, 1)
   
    for k, d in enumerate(dets):
        
        dt=abs(d.left()-d.right())+abs(d.top()-d.bottom())
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        shape1 = face_utils.shape_to_np(shape)

        for (x, y) in shape1[48:68]:
            cv2.circle(img2, (x, y), 1, (0, 0, 255), -1)
     
       
        xmouthpoints = [shape.part(x).x for x in range(48,67)]
        ymouthpoints = [shape.part(x).y for x in range(48,67)]
        maxx = max(xmouthpoints)
        minx = min(xmouthpoints)
        maxy = max(ymouthpoints)
        miny = min(ymouthpoints) 

        # to show the mouth properly pad both sides
        pad = 1000//dt
      
       

        crop_image = img1[miny-pad:maxy+pad,minx-pad:maxx+pad]
        crop_image=cv2.resize(crop_image,(40,40))
        c=crop_image

        crop_image1 = img2[miny-pad:maxy+pad,minx-pad:maxx+pad]
        crop_image1=cv2.resize(crop_image1,(40,40))

        crop_image2 = img[miny-pad:maxy+pad,minx-pad:maxx+pad]
        crop_image2=cv2.resize(crop_image2,(40,40))

        hr=haar(crop_image2)

        crop_image=hog1(crop_image)
       
        cv2.imwrite('eh1.jpg',c)
        return crop_image,crop_image1,hr
        


def process(path):
    img=cv2.imread(path)
    kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
    img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    dst=cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    cv2.imwrite('eh.jpg',dst)
    dst,dst1,dst2=crop_mouth(dst)
    return dst,dst1,dst2
    


def read_from_folder(path):
    print("==================================================")
    print("Scanning Folders....")
    data1=[]
    data2=[]
    data3=[]
    labels=[]
    subfolders= os.listdir(path)
    # print(subfolders)
    for sf in subfolders:
        videos=os.listdir(path+"/"+sf)
        # print(videos)
        c=0
        for vid in videos:
            imgs=os.listdir(path+"/"+sf+"/"+vid)
            imgs.sort(key=natural_keys)
            # print(imgs)
            dl1=[]
            dl2=[]
            dl3=[]
            for j in imgs:

                path1=path+"/"+sf+"/"+vid+"/"+j
                print(path1)

                d1,d2,d3=process(path1)
                # print("\nSHAPES\n")
                # print(d1.shape)
                # print(d2.shape)
                # print(d3.shape)

                dl1.append(d1)
                dl2.append(d2)
                dl3.append(d3)
  
            dl1=np.array(dl1)
            dl2=np.array(dl2)
            dl3=np.array(dl3)

            # print(dl1.shape,'=======')

            data1.append(dl1)  

            data2.append(dl2) 

            data3.append(dl3) 

            print("sf : ",sf)
            labels.append(sf)
 
        c+=1    
                   

    return np.array(data1),np.array(data2),np.array(data3),np.array(labels)


if __name__=="__main__":
    data1,data2,data3,labels=read_from_folder('Video_Frames')
    print("\nMy shapes\n")
    print(data1.shape)
    print(data2.shape)
    print(data3.shape)
    print(labels.shape)

    pickle.dump(data1,open('data1.pkl','wb'))
    pickle.dump(data2,open('data2.pkl','wb'))
    pickle.dump(data3,open('data3.pkl','wb'))
    pickle.dump(labels,open('labels.pkl','wb'))


