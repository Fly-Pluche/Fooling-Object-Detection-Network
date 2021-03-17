import torch
import json
from models import MaskRCNN,FasterRCNN
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

model = FasterRCNN()

# print(annos)
for ann, image in tqdm(zip(annos, images)):
    f = open(ann, 'r')
    ann_ = json.loads(f.read())
    f.close()
    if ann_.get('people_boxes') is None:
        im = np.asarray(Image.open(image))
        outputs = model.default_predictor(im)["instances"].to('cpu')
        scores = outputs.scores
        boxes = outputs.pred_boxes
        classes = outputs.pred_classes
        boxes = boxes.clone().tensor
        classes = classes[scores > 0.95]
        boxes = boxes[scores > 0.95]
        boxes = boxes[classes == 0]
        boxes = boxes.numpy().tolist()
        ann_['people_boxes'] = boxes
        ann_ = json.dumps(ann_)
        with open(ann, 'w') as f2:
            f2.write(ann_)

    else:
        if len(ann_['people_boxes']) >= 3 or len(ann_['people_boxes']) == 0:
            print(ann)
            # print(ann_['people_boxes'])
            # os.system('rm ' + ann)
