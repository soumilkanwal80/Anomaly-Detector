#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import io

import detectron2

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# import some common libraries
import numpy as np
import cv2
import torch

# Show the image in ipynb
from IPython.display import clear_output, Image, display
import PIL.Image
def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = io.BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))


# In[2]:
# Load VG Classes
data_path = os.path.join('./ic_code/demo', 'data/genome/1600-400-20')


vg_classes = []
with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
    for object in f.readlines():
        vg_classes.append(object.split(',')[0].lower().strip())

MetadataCatalog.get("vg").thing_classes = vg_classes


# In[3]:


cfg = get_cfg()
cfg.merge_from_file("./ic_code/configs/VG-Detection/faster_rcnn_R_101_C4_caffe.yaml")
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
# VG Weight
cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe.pkl"
predictor = DefaultPredictor(cfg)


# In[23]:


NUM_OBJECTS = 36

from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, fast_rcnn_inference_single_image

def calculateImageFeatures(raw_image):
    with torch.no_grad():
        raw_height, raw_width = raw_image.shape[:2]
        #print("Original image size: ", (raw_height, raw_width))
        
        # Preprocessing
        image = predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
        #print("Transformed image size: ", image.shape[:2])
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": raw_height, "width": raw_width}]
        images = predictor.model.preprocess_image(inputs)
        
        # Run Backbone Res1-Res4
        features = predictor.model.backbone(images.tensor)
        
        # Generate proposals with RPN
        proposals, _ = predictor.model.proposal_generator(images, features, None)
        proposal = proposals[0]
        #print('Proposal Boxes size:', proposal.proposal_boxes.tensor.shape)
        
        # Run RoI head for each proposal (RoI Pooling + Res5)
        proposal_boxes = [x.proposal_boxes for x in proposals]
        features = [features[f] for f in predictor.model.roi_heads.in_features]
        box_features = predictor.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        #print('Pooled features size:', feature_pooled.shape)
        
        # Predict classes and boxes for each proposal.
        pred_class_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(feature_pooled)
        outputs = FastRCNNOutputs(
            predictor.model.roi_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            predictor.model.roi_heads.smooth_l1_beta,
        )
        probs = outputs.predict_probs()[0]
        boxes = outputs.predict_boxes()[0]
        
        # Note: BUTD uses raw RoI predictions,
        #       we use the predicted boxes instead.
        # boxes = proposal_boxes[0].tensor    
        
        # NMS
        for nms_thresh in np.arange(0.5, 1.0, 0.1):
            instances, ids = fast_rcnn_inference_single_image(
                boxes, probs, image.shape[1:], 
                score_thresh=0.2, nms_thresh=nms_thresh, topk_per_image=NUM_OBJECTS
            )
            if len(ids) == NUM_OBJECTS:
                break
                
        instances = detector_postprocess(instances, raw_height, raw_width)
        roi_features = feature_pooled[ids].detach()
        #print(instances)
        
        return roi_features.to('cpu').numpy()


# In[62]:


# from tqdm import tqdm



# # In[39]:


# DATA_PATH = '../../val'
# CATEGORIES = os.listdir(DATA_PATH)
# FEATURE_PATH = './val_features'
# for i in CATEGORIES:
#     #print(i)
#     FOLDER_PATH = os.path.join(DATA_PATH, i)
#     for j in os.listdir(FOLDER_PATH):
#         if os.path.isdir(os.path.join(FEATURE_PATH, i)) == False:
#             os.makedirs(os.path.join(FEATURE_PATH, i))
            
# feat_list = []
# im_list = []


# # In[40]:


# for i in CATEGORIES:
#     #print(i)
#     FOLDER_PATH = os.path.join(DATA_PATH, i)
#     for j in os.listdir(FOLDER_PATH):
#         FEATURE_FILE = os.path.join(FEATURE_PATH, i, str(j) + '.npy')
#         IMG_DIR = os.path.join(FOLDER_PATH, j)
#         feat_list.append(FEATURE_FILE)
#         im_list.append(IMG_DIR)


# # In[63]:


# def feature_save(args):
#     IMG_DIR, FEATURE_FILE = args
#     if os.path.isfile(FEATURE_FILE) == True:
#         print("hi")
#         return
#     print(FEATURE_FILE)
#     feat = []
#     for k in range(len(os.listdir(IMG_DIR))):
#         im = cv2.imread(os.path.join(IMG_DIR, str(k) + '.png'))
#         feat.append(doit(im))
#     np.save(FEATURE_FILE, feat)


# # In[58]:



# args = ('../../val/normal/56', './56.npy')
# feature_save(args)


# idx = []
# for i in range(4 + 1):
#     idx.append(int(1929*i/4))


# # In[60]:

# #print(len(im_list))
# #comb= zip(im_list, feat_list)
# #for i in range(4):
# #    comb.append(zip(im_list[idx[i]:idx[i+1]], feat_list[idx[i]:idx[i+1]]))


# # In[64]:


# def processfn(comb_it):
    
#     for i in tqdm(list(comb_it)):

#         #it = next(comb_it)
#         feature_save(i)


# # In[72]:
# #processfn(comb[0])

# #processfn(comb[1])
# #processfn(comb)
# #processfn(comb[3])

