import cv2
import os
import pickle

#one folder containing inverted images (so converting that inverted images to normal)

def test():
	path="Out_Frames"

	images=os.listdir(path)


	for i in images:

		src = cv2.imread(path+"/"+i)

		#VID_20230214_145918.mp4

		image = cv2.rotate(src, cv2.ROTATE_180)

		cv2.imwrite("CHECK/%s" % (i), image)
		cv2.waitKey(0)


def main():
	path="Video_Frames/1/one"

	images=os.listdir(path)


	for i in images:

		src = cv2.imread(path+"/"+i)

		#VID_20230214_145918.mp4

		image = cv2.rotate(src, cv2.ROTATE_180)

		cv2.imwrite("Video_Frames/1/VID_20230214_145918.mp4/%s" % (i), image)
		cv2.waitKey(0)



def pickle_loader():
	pickle1,pickle2,pickle3=pickle.load(open('data1.pkl','rb')),pickle.load(open('data2.pkl','rb')),pickle.load(open('labels.pkl','rb'))
	return pickle1,pickle2,pickle3

#For rotating
# main()