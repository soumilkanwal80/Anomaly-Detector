import numpy as np
import torch
import torchvision.models as tv
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.optim as optim
import math
import cv2, os
from matplotlib import pyplot as plt
import skvideo.io as skv
from tqdm import tqdm
from random import randint, seed, shuffle
from torchvision import transforms
from models import *

res = tv.resnet101(pretrained=True)
res = nn.Sequential(*list(res.children())[:-2])
res = res.eval()

res3d = tv.video.r2plus1d_18(pretrained = True)
res3d = nn.Sequential(*list(res3d.children())[:-1])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
    mean = np.array([0.485, 0.456, 0.406]),
    std = np.array([0.229, 0.224, 0.225])
    )
])

transform3d = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
    mean = np.array([0.43216, 0.394666, 0.37645]),
    std = np.array([0.22803, 0.22145, 0.216989])
    )
])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
decoder = decoder = SERLOC_B_Anomaly(res, res3d).to(device)

category = os.listdir('./train_an')
cat2idx = {}
idx2cat = {}
for i in range(len(category)):
    cat2idx[category[i]] = i
    idx2cat[i] = category[i]

def get_paths(mode = 'train'):
    IMG_PATH = './' + mode +  '_an'
    FEATURE_PATH = './' + mode+'_features'
    X = []
    Y = []
    Z = []
    CATEGORIES = os.listdir(IMG_PATH)
    for i in range(len(CATEGORIES)):
        tmp_path = os.path.join(IMG_PATH, CATEGORIES[i])
        for j in os.listdir(tmp_path):
            feature_path=os.path.join(FEATURE_PATH,CATEGORIES[i] ,j) + '.npy'
            if os.path.isfile(feature_path):
                X.append(j)
                Y.append(i)
                Z.append(1)
                X.append(j)
                Y.append(i)
                Z.append(0)
            else:
                print("feature not found.")
                print(feature_path)
    
    return X, Y, Z    

X_train, Y_train, Z_train = get_paths('train')
X_val, Y_val, Z_val = get_paths('val')
train_steps_per_epoch = int(len(X_train)/16)
val_steps_per_epoch = int(len(X_val)/16)

class VideoLoader(Dataset):
    def __init__(self, folder_name, label, Z, batch_size = 8, clip_len = 16, max_objects=None, mode = 'train',obj_mode='mean'):
        super()
        self.folder_name = folder_name
        self.label = label
        self.batch_size = batch_size
        self.mode = mode
        self.max_objects=max_objects
        self.clip_len = clip_len
        self.max = self.__len__()
        self.iter = 0
        self.Z = Z
        self.obj_mode=obj_mode
        
        if self.obj_mode=='mean':
            self.FEATURE_PATH = './' + self.mode + '_2'
            
    def __len__(self):
        return math.floor(len(self.folder_name)/self.batch_size)
    
    def __getitem__(self, idx):
        batch_folder_names = self.folder_name[idx*self.batch_size:(idx + 1)*self.batch_size]
        batch_label = self.label[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_Z = self.Z[idx*self.batch_size:(idx+1)*self.batch_size]
        
        IMG_PATH = './' + self.mode + '_an'
        
        #FLOW_PATH = './' + self.mode + '_flow_n'
        
        X = []
        Y = []
        C = []
        F = []
        X3D = []
        
        obj_count=[]
        for j in range(self.batch_size):
            folder_path = os.path.join(IMG_PATH, idx2cat[batch_label[j]], batch_folder_names[j])
            feature_path = os.path.join(self.FEATURE_PATH, idx2cat[batch_label[j]], batch_folder_names[j]) + '.npy'
            #flow_path = os.path.join(FLOW_PATH, idx2cat[batch_label[j]], batch_folder_names[j])
            
            
            caption_features = np.load(feature_path, allow_pickle = True)
            
            
            imgs = os.listdir(folder_path)
            num = len(imgs)
            start_idx = randint(0, num - self.clip_len)
            
            
            x = []
            c = []
            x3d = []
            
            #Temporal loop
            for i in range(self.clip_len):
                im = cv2.imread(folder_path + '/' + str(start_idx + i) + '.png')
                imo = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im3d = transform3d(imo[8:120, 30:142, :].astype('float32')/255).numpy()
                if batch_Z[j] == 0:
                    im3d = cv2.flip(im3d, 1)
                x3d.append(im3d)
    
            x3d_np=np.array(x3d)#[T,W,H,3]
            frame_num=int(self.clip_len/2)
            x_np=x3d_np[frame_num,:]#[W,H,3]
            
            c = caption_features[start_idx + frame_num]#[Feature_dim] ot [num_obj, feature_dim]
            c[np.isnan(c)] = 0
                            
            X.append(x_np)
            C.append(c)
            X3D.append(x3d_np)
            #print(j, len(batch_label), batch_label)
            Y.append(batch_label[j])
        
        return np.array(X), np.array(Y), np.array(C), np.array(X3D)

    def on_epoch_end(self):
        mapIndexPosition = list(zip(self.folder_name, self.label, self.Z))
        shuffle(mapIndexPosition)
        self.folder_name, self.label, self.Z = zip(*mapIndexPosition)
    
    def __next__(self):
        #print(self.iter)
        if self.iter >= self.max:
            self.iter = 0
            self.on_epoch_end()
        res = self.__getitem__(self.iter)
        self.iter = self.iter + 1
        return res
    
train_gen = VideoLoader(X_train, Y_train, Z_train, 16, mode = 'train')
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(decoder.parameters(), lr = 1e-5)


for epoch in range(15):
    running_loss = 0
    for steps in tqdm(range(train_steps_per_epoch)):
        optimizer.zero_grad()
        
        x, y_true, c, x3d = next(train_gen)
        x3d = torch.from_numpy(x3d).float().permute((0, 2, 1, 3, 4)).to(device)
        x = torch.from_numpy(x).float().to(device) 
        y_true = torch.from_numpy(y_true).to(device)
        y_true = [F.one_hot(i, 12) for i in y_true]
        y_true = torch.stack(y_true).float()
        c = torch.from_numpy(c).float().to(device)
        
        
        
        y_pred = decoder(x, x3d, c)
        
        loss = criterion(y_pred, y_true)
        
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        
    print("epoch ",epoch," loss")
    print(running_loss/train_steps_per_epoch)

torch.save(decoder.state_dict(), './weights/serlocB_Anomaly.pt')