# import torch
# from torchvision.transforms import functional
# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np
# import torch.nn as nn

# with open('/home/ray/data/deepfashion2/train/train.txt', 'r') as f:
#     a.json = f.readlines()
#
# import os
# from tqdm import tqdm
# for item in tqdm(a.json):
#     file_path = os.path.join('/home/ray/data/deepfashion2/train/annos', item.strip())
#     b = os.path.join('/home/ray/data/deepfooling/annos', item.strip())
#     os.system('sudo cp %s %s' % (file_path, b))


import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

annos_path = '/home/ray/data/deepfooling/annos/045404.json'
images_path = '/home/ray/data/deepfooling/image/045404.jpg'

with open(annos_path, 'r') as f:
    anno = json.loads(f.read())
import copy

people_box = np.array(anno['people_boxes'][0], dtype=np.int32)
a = np.array(cv2.imread(images_path), dtype=np.uint8)
for item in anno:
    if 'item' in item:
        anno_item = anno[item]
        if anno_item['category_name'] != 'short sleeve top':
            continue
        landmarks = torch.tensor(anno[item]['landmarks'])
        landmarks = landmarks.view(-1, 3)
        # landmarks =
        temp1 = copy.deepcopy(landmarks[6])
        temp2 = copy.deepcopy(landmarks[7])
        temp3 = copy.deepcopy(landmarks[8])
        temp4 = copy.deepcopy(landmarks[9])
        temp5 = copy.deepcopy(landmarks[10])
        temp6 = copy.deepcopy(landmarks[11])
        temp7 = copy.deepcopy(landmarks[12])
        temp8 = copy.deepcopy(landmarks[13])
        temp9 = copy.deepcopy(landmarks[14])
        landmarks[6] = landmarks[24]
        landmarks[7] = landmarks[23]
        landmarks[8] = landmarks[22]
        landmarks[9] = landmarks[21]
        landmarks[10] = landmarks[20]
        landmarks[11] = landmarks[19]
        landmarks[12] = landmarks[18]
        landmarks[13] = landmarks[17]
        landmarks[14] = landmarks[16]
        landmarks[24] = temp1
        landmarks[23] = temp2
        landmarks[22] = temp3
        landmarks[21] = temp4
        landmarks[20] = temp5
        landmarks[19] = temp6
        landmarks[18] = temp7
        landmarks[17] = temp8
        landmarks[16] = temp9
        landmarks = landmarks[1:, :2]
        # landmarks = landmarks.unsqueeze(0)
        landmarks = np.array(landmarks, dtype=np.int32)
        zeros = np.zeros((a.shape), dtype=np.uint8)
        mask = cv2.fillPoly(zeros, [landmarks], color=(0, 164, 120))
        a = cv2.addWeighted(a, 0.8, mask, 0.5, 0)
        cv2.polylines(a, [landmarks], isClosed=True, thickness=5, color=(144, 238, 144))
        cv2.rectangle(a, (people_box[0], people_box[1]), (people_box[2], people_box[3]), (255, 227, 132), 2)
        for i in landmarks:
            cv2.circle(a, tuple(i), 2, (185, 130, 66), 4)
        plt.imshow(a)
        plt.show()
        cv2.imwrite('images/5.jpg', a)
print(anno)

"""
masks_batch = []
landmarks_scaled = np.copy(landmarks)
landmarks_scaled = landmarks_scaled * self.configs.img_size_big[0]
polygons_batch = landmarks_scaled[:, :, 0: 2]  # [batch size, max_landmarks, )]
bg_mask = np.zeros(self.configs.img_size_big, dtype=np.uint8)
for polygons in polygons_batch:
    polygons = np.array(polygons, np.int32)
    if np.sum(polygons) != 0:
        masks = cv2.fillPoly(bg_mask, [polygons], color=1)
        masks_batch.append(np.asarray(Image.fromarray(masks)))
    else:
        masks_batch.append(bg_mask)
masks_batch = np.array(masks_batch, np.float)
return masks_batch
"""

# import cv2
#
# a.json = cv2.imread('images/1.jpg')
# b = cv2.imread('images/2.jpg')
# c = cv2.imread('images/3.jpg')
#
# a.json = cv2.cvtColor(a.json, cv2.COLOR_BGR2RGB)
# b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
# c = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
#
# cv2.imwrite('images/1.jpg', a.json)
# cv2.imwrite('images/2.jpg', b)
# cv2.imwrite('images/3.jpg', c)

# import cv2
# a.json = cv2.imread('/home/corona/attack/Fooling-Object-Detection-Network/patches/MaskRCNN.jpg')
# print(a.json.shape)



import pycocotools










