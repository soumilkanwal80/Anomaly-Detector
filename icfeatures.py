from ic_code.demo.demo_feature_extraction import calculateImageFeatures
from tqdm import tqdm
import cv2, os
from matplotlib import pyplot as plt
import skvideo.io as skv
from tqdm import tqdm
import numpy as np

ic_path = './train_2'
ic_val_path = './val_2'
src_path = './dataset'

def extractICFeatures(category, file_name):
	vid = skv.vread(os.path.join(src_path, category, file_name))
	features = []
	for i in tqdm(range(vid.shape[0])):
		frame_features = calculateImageFeatures(vid[i])
		frame_features = frame_features.mean(axis = 0)
		frame_features[np.isnan(frame_features)] = 0
		features.append(frame_features)

	features = np.array(features)
	if os.path.isdir(ic_path) == False:
		os.makedirs(ic_path)
	
	for j in category:
		if os.path.isdir(os.path.join(ic_path, category)) == False:
			os.makedirs(os.path.join(ic_path, category))
	
	np.save(os.path.join(ic_path, category, file_name.replace('mp4', '.npy')), features)

category = os.listdir(src_path)
if os.path.isdir(ic_val_path) == False:
	os.makedirs(ic_val_path)
for j in category:
	if os.path.isdir(os.path.join(ic_val_path, j)) == False:
		os.makedirs(os.path.join(ic_val_path, j))

for i in category:
	for j in os.listdir(os.path.join(src_path, i)):
		extractICFeatures(i, j)

for i in CATEGORIES:
	if i == "normal":
		for j in range(100):
			shutil.move(os.path.join(ic_path, i, str(j) + '.npy'), os.path.join(ic_val_path, i, str(j) + '.npy'))
	else:
		for j in range(4):
			shutil.move(os.path.join(ic_path, i, str(j) + '.npy'), os.path.join(ic_val_path, i, str(j) + '.npy'))
