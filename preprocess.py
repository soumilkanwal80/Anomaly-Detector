# This script extracts reshapes each video into 171x128 and stores them as a folder of frames, and creates our train-val split.

import os, cv2, threading, shutil
import numpy as np
DATASET_PATH = "./dataset/"
FEATURE_PATH = "./train_an/"
VAL_PATH = "./val_an/"
num_threads = 64
num_process = 16
CATEGORIES = os.listdir(DATASET_PATH)

def processVideo(vid_path, name):
    if len(os.listdir(FEATURE_PATH + name)) > 0:
        return
    videoStream = cv2.VideoCapture(vid_path)
    print(name)
    frameCount = int(videoStream.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = 0
    while True:
        ret, frame = videoStream.read()
        if ret == False:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (171, 128))
        cv2.imwrite(FEATURE_PATH + name + '/' + str(count) + '.png', frame)
        count = count + 1
        
def getPaths():
    vid_paths = []
    feature_paths = []
    for category in CATEGORIES:
        video_path = DATASET_PATH + category + '/'
        if os.path.isdir(video_path) is True:
            videos = os.listdir(video_path)
            videos.sort()
            idx = 0
            for video in videos:
                vid_paths.append(video_path + video)
                feature_paths.append(category + '/' + str(idx))
                idx = idx + 1
    return vid_paths, feature_paths


def threadFunction(l1, l2):
    for i in range(len(l1)):
        processVideo(l1[i], l2[i])


def init(del_normal = True):
	

	for i in CATEGORIES:
	    if os.path.isdir(FEATURE_PATH + i)== False and os.path.isdir(DATASET_PATH + i) == True:
	        os.makedirs(FEATURE_PATH + i)


	video_paths, feature_paths = getPaths()

	for i in feature_paths:
	    if os.path.isdir(FEATURE_PATH + i) == False:
	        os.makedirs(FEATURE_PATH + i)


	for num in range(num_threads-1):
	    low, up = int(2077*num/num_threads), int(2077*(num+1)/num_threads)
	    threading.Thread(target = threadFunction, args = (video_paths[low:up], feature_paths[low:up])).start()

	for i in range(2077):
	    processVideo(video_paths[i], feature_paths[i])

	for i in CATEGORIES:
	    os.makedirs(VAL_PATH + i)


	if del_normal == False:    
		for i in CATEGORIES:
		    if i == "normal":
		        for j in range(100):
		            shutil.move(FEATURE_PATH + i + '/' + str(j), VAL_PATH + i + '/' + str(j))
		    else:
		        for j in range(4):
		            shutil.move(FEATURE_PATH + i + '/' + str(j), VAL_PATH + i + '/' + str(j))
	
	else:
		for i in CATEGORIES:
		    if i == "normal":
		    	for j in range(len(os.listdir(FEATURE_PATH + i))):
		            shutil.rmtree(FEATURE_PATH + i + '/' + str(j))
		    else:
		        for j in range(4):
		            shutil.move(FEATURE_PATH + i + '/' + str(j), VAL_PATH + i + '/' + str(j))
		os.rmdir(FEATURE_PATH + "normal")
		os.rmdir(VAL_PATH + "normal")





init(del_normal = True)
FEATURE_PATH = "./train/"
VAL_PATH = './val/'

init(del_normal = False)



