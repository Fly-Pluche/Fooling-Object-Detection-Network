import torch
import json
from models import MaskRCNN
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

ann_path = '/home/ray/data/deepfashion2/train/train.txt'
root = '/home/ray/data/deepfashion2/train/'
with open(ann_path, 'r') as f:
    annos = f.readlines()
images = [os.path.join(root, 'image', item.strip()[:-5] + '.jpg') for item in annos]
annos = [os.path.join(root, 'annos', item.strip()) for item in annos]

model = MaskRCNN()

# print(annos)
for ann, image in tqdm(zip(annos, images)):
    f = open(ann, 'r')
    ann_ = json.loads(f.read())
    f.close()
    if ann_.get('people_boxes') is None:
        im = np.asarray(Image.open(image))
        outputs = model.default_predictor(im)["instances"].to('cpu')
        boxes = outputs.pred_boxes
        classes = outputs.pred_classes
        boxes = boxes.clone().tensor
        boxes = boxes[classes == 0].numpy().tolist()
        ann_['people_boxes'] = boxes
        ann_ = json.dumps(ann_)
        with open(ann, 'w') as f2:
            f2.write(ann_)

