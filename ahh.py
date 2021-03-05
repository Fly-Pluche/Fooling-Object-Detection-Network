import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import json
import os
from tqdm import tqdm
from utils.parse_annotations import ParseTools

# image = np.array(Image.open('/home/ray/data/deepfashion2/validation/image/011867.jpg'))
# plt.imshow(image)
# plt.show()
#
# json_files = os.listdir('/home/ray/data/deepfashion2/validation/annos')
# json_files = [os.path.join('/home/ray/data/deepfashion2/validation/annos', item) for item in json_files]
#
# tools = ParseTools()
# needed_points = [12, 20, 13, 19, 14, 18, 15, 17]
#
# anno = tools.parse_anno_file('/home/ray/data/deepfashion2/validation/annos/024574.json', need_classes=[1])
# # print(anno)
# # exit()
# f = open('/home/ray/data/deepfashion2/validation/train.txt', 'w')
# for item in json_files:
#     flag = 0
#     anno = tools.parse_anno_file(item, need_classes=[1])
#     if len(anno['viewpoints']) == 0:
#         flag = 1
#     if 1 in anno['viewpoints'] or 3 in anno['viewpoints']:
#         flag = 1
#
#     for l in anno['landmarks']:
#
#         if flag == 1:
#             break
#         for point in needed_points:
#             if flag == 1:
#                 break
#             if l[point * 3 - 1] == 0:
#                 flag = 1
#     # print(item)
#     if flag == 0:
#         f.writelines(item + '\n')
#         print(item)
# f.close()

from utils.parse_annotations import ParseTools
from tqdm import tqdm
needed = ''
tools = ParseTools()
with open('/home/corona/datasets/WiderPerson/train/train.txt','r') as f:
    a = f.readlines()

for item in tqdm(a):
    item = item.strip()
    info = tools.load_image(item)
    print(len(info['labels']))
    if 1 < len(info['labels']) <= 20:
        needed += item
        needed+='\n'
    else:
        print('...')

with open('/home/corona/datasets/WiderPerson/train/train2.txt','w') as f:
    f.write(needed)


