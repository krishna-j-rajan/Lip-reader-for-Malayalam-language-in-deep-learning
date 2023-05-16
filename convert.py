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


    if not os.path.exists("Video_Frames/%s/%s" % (i,video)):
        os.makedirs("Video_Frames/%s/%s" % (i,video))

    count = 1
    while(vidcap.isOpened()):
      ret, frame = vidcap.read()
      if ret == False:
          break

      if count==(totalframecount-1):
        print("hai")
        while totalframecount<115:
          cv2.imwrite("Video_Frames/%s/%s/frame%d.jpg" % (i,video,totalframecount), frame)
          print(totalframecount)
          totalframecount+=1
      cv2.imwrite("Video_Frames/%s/%s/frame%d.jpg" % (i,video,count), frame)
      count=count+1


    vidcap.release()
    cv2.destroyAllWindows()


print("\n[INFO] : All videos converted to Frames\n")
print("[Alert] : Frames are stored to Video_Frames Folder\n")


