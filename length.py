import cv2
import os

path="Project_Dataset"

folders=os.listdir(path)
print(folders)


for i in folders:
  print(i)
  inside_path=path+"/"+i
  videos=os.listdir(inside_path)
  print(videos)
  for video in videos:
    video_path=inside_path+"/"+video
    print(video_path)

    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()

    totalframecount= int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("The total number of frames in this video is ", totalframecount)



print("\n[INFO] : All videos converted to Frames\n")
print("[Alert] : Frames are stored to Video_Frames Folder\n")


