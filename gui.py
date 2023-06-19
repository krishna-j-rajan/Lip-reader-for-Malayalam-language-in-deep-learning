from tkinter import *
from tkinter import messagebox
from PIL import ImageTk,Image
from tkinter.filedialog import askopenfilename
import cv2 as cv
import cv2
import numpy as np
import time
import dlib
from imutils import face_utils
import mahotas
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import mahotas.demos
from mahotas.thresholding import soft_threshold
import os
from tensorflow.keras.models import load_model
from rotate import test
a=Tk()
a.title("Lip Reader : Malayalam")
a.geometry("800x650")
a.minsize(800,650)
a.maxsize(800,650)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('Project_Extra/shape_predictor_68_face_landmarks.dat')

loaded_model=load_model("Trained_Model/new_model_final.h5",compile=False)
height=32
width=32


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
    


def frame_conversion(video_path):
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()

    totalframecount= int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("The total number of frames in this video is ", totalframecount)

    count = 1
    while(vidcap.isOpened()):
      ret, frame = vidcap.read()
      if ret == False:
          break

      if count==(totalframecount-1):
        while totalframecount<115:
          cv2.imwrite("Out_Frames/frame%d.jpg" % (totalframecount), frame)
          print(totalframecount)
          totalframecount+=1
      cv2.imwrite("Out_Frames/frame%d.jpg" % (count), frame)
      count=count+1


    vidcap.release()
    cv2.destroyAllWindows()




def prediction():
    
    list_box.insert(1,"Loading Video")
    list_box.insert(2,"")
    list_box.insert(3,"Preprocessing")
    list_box.insert(4,"")
    list_box.insert(5,"Loading Model")
    list_box.insert(6,"")
    list_box.insert(7,"Prediction")

    import cv2
    import pandas as pd

    #to delete images in 'check_frames' folder
    filelist1 = [ f1 for f1 in os.listdir('Out_Frames') if f1.endswith(".jpg") ]
    for f1 in filelist1:
        os.remove(os.path.join('Out_Frames', f1))


    print("\nPath : ")
    print(path)

    frame_conversion(path)

    ####### test()

    data1=[]
    data2=[]
    data3=[]


    path1="Out_Frames"#"CHECK"#"Out_Frames"

    imgs=os.listdir(path1)
    imgs.sort(key=natural_keys)
    print(imgs)
    dl1=[]
    dl2=[]
    dl3=[]
    for img in imgs:
        d1,d2,d3=process(path1+"/"+img)

        dl1.append(d1)
        dl2.append(d2)
        dl3.append(d3)


    dl1=np.array(dl1)
    dl2=np.array(dl2)
    dl3=np.array(dl3)

    data1.append(dl1)  
    data2.append(dl2) 
    data3.append(dl3) 

    data1=np.array(data1)
    data2=np.array(data2)
    data3=np.array(data3)

    data1=data1/255
    data2=data2/255
    data3=data3/255

    data1=np.stack(data1)
    data2=np.stack(data2)
    data3=np.stack(data3)


    print(data1.shape[0])
    print(data2.shape[0])
    print(data3.shape[0])

    f1=np.reshape(data1,(1,114,40,40,1))
    f2=np.reshape(data2,(1,114,40,40,3))
    f3=np.reshape(data3,(1,114,40,40,1))

    pred1=loaded_model.predict(f1)[0]
    
    print("pred1 : ",pred1)
    my_pred=np.argmax(pred1)
    print("argmax : ",my_pred)

    # s=u'അമ്മ'
    # b=s.encode('utf-8').decode('utf-8')
    # print(b)
     
    print("\n RESULT")
    if my_pred==0:
        a=u'അമ്മ'
        b=a.encode('utf-8').decode('utf-8')
        print(b)
    elif my_pred==1:
        a=u'തുണി'
        b=a.encode('utf-8').decode('utf-8')
        print(b)
    elif my_pred==2:
        a=u'വെള്ളം'
        b=a.encode('utf-8').decode('utf-8')
        print(b)
    elif my_pred==3:
        a=u'കാലം'
        b=a.encode('utf-8').decode('utf-8')
        print(b)
    elif my_pred==4:
        a=u'പശു'
        b=a.encode('utf-8').decode('utf-8')
        print(b)
    elif my_pred==5:
        a=u'നമസ്കാരം'
        b=a.encode('utf-8').decode('utf-8')
        print(b)
    elif my_pred==6:
        a=u'അപ്പൂപ്പന്‍'
        b=a.encode('utf-8').decode('utf-8')
        print(b)
    elif my_pred==7:
        a=u'രാജാവ്'
        b=a.encode('utf-8').decode('utf-8')
        print(b)
    elif my_pred==8:
        a=u'താമര'
        b=a.encode('utf-8').decode('utf-8')
        print(b)
    elif my_pred==9:
        a=u'വന്നു'
        b=a.encode('utf-8').decode('utf-8')
        print(b)
    elif my_pred==10:
        a=u'മുഖം'
        b=a.encode('utf-8').decode('utf-8')
        print(b)
    elif my_pred==11:
        a=u'കടുവ'
        b=a.encode('utf-8').decode('utf-8')
        print(b)
    elif my_pred==12:
        a=u'ബലം'
        b=a.encode('utf-8').decode('utf-8')
        print(b)
    elif my_pred==13:
        a=u'ഉപ്പ്'
        b=a.encode('utf-8').decode('utf-8')
        print(b)
    elif my_pred==14:
        a=u'ചെവി'
        b=a.encode('utf-8').decode('utf-8')
        print(b)
    elif my_pred==15:
        a=u'ഇല'
        b=a.encode('utf-8').decode('utf-8')
        print(b)
    else:
        print("No words")

  
    out_label.config(text=b)


def Check():
    global f
    f.pack_forget()

    f=Frame(a,bg="white")
    f.pack(side="top",fill="both",expand=True)


    
    global f1
    f1=Frame(f,bg="white")
    f1.place(x=0,y=0,width=560,height=340)
    f1.config()
                   
    upload_pic_button=Button(f1,text="Upload Video",command=Upload,bg="pink")
    upload_pic_button.pack(side="top",pady=50)
    
    global my_label
    my_label=Label(f1,bg="Lavender")
    global label
    label=Label(f1,text="NO FILES UPLOADED",foreground="black",font="arial 10",bg="white")
    label.pack()

    
    predict_button=Button(f1,text="Predict",command=prediction,bg="deepskyblue")
    predict_button.pack(side="bottom",pady=50)

    
    global f2
    f2=Frame(f,bg="white")
    f2.place(x=0,y=340,width=560,height=320)
    f2.config(pady=20)
    
    result_label=Label(f2,text="RESULT",font="arial 16",bg="white")
    result_label.pack(padx=0,pady=0)

    global out_label
    out_label=Label(f2,text="",bg="white",font="arial 16")
    out_label.pack(pady=40)
    

    f3=Frame(f,bg="grey")
    f3.place(x=560,y=0,width=240,height=690)
    f3.config()

    name_label=Label(f3,text="STEPS UNDERTAKEN",font="arial 14",bg="grey")
    name_label.pack(pady=20)

    global list_box
    list_box=Listbox(f3,height=12,width=31)
    list_box.pack()


def Upload():
    global path
    label.config(text='No files uploaded',foreground="red",font="arial 10")
    my_label.config(text='')
    list_box.delete(0,END)
    out_label.config(text='')
    path=askopenfilename(title='Open a file',
                         initialdir='Test',
                         filetypes=[("MP4","*.mp4")])
    print("<<<<<<<<<<<<<",path)
    if(path==''):
        label.config(text="No files uploaded",foreground="red",font="arial 10")
    else:
        global my_file_name
        my_file_name=os.path.basename(path)

        my_label.config(text=my_file_name)
        my_label.pack(pady=20)
        label.config(text="Video Uploaded Successfully",font="arial 14", foreground="green")
    


def Home():
    global f
    f.pack_forget()
    
    f=Frame(a,bg="cornflower blue")
    f.pack(side="top",fill="both",expand=True)
    front_image = Image.open("Project_Extra/home.jpg")
    front_photo = ImageTk.PhotoImage(front_image.resize((a.winfo_width(), a.winfo_height()), Image.ANTIALIAS))
    front_label = Label(f, image=front_photo)
    front_label.image = front_photo
    front_label.pack()

    home_label=Label(f,text="Lip Reader : Malayalam",font="arial 35",bg="cornflower blue")
    home_label.place(x=150,y=250)




f=Frame(a,bg="cornflower blue")
f.pack(side="top",fill="both",expand=True)
front_image1 = Image.open("Project_Extra/home.jpg")
front_photo1 = ImageTk.PhotoImage(front_image1.resize((800,650), Image.ANTIALIAS))
front_label1 = Label(f, image=front_photo1)
front_label1.image = front_photo1
front_label1.pack()

home_label=Label(f,text="Lip Reader : Malayalam",font="arial 35",bg="cornflower blue")
home_label.place(x=150,y=250)


m=Menu(a)
m.add_command(label="Home",command=Home)
checkmenu=Menu(m)
m.add_command(label="Check",command=Check)
a.config(menu=m)




a.mainloop()
