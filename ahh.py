import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import json
import os
from tqdm import tqdm
from utils.parse_annotations import ParseTools
import json

# image = np.array(Image.open('/home/ray/data/deepfashion2/validation/image/025781.jpg'))
# plt.imshow(image)
# plt.show()
root = '/Users/keter/Documents/Fooling-Object-Detection-Network/train'
json_files = os.listdir(root + '/annos')
json_files = [os.path.join(root + '/annos', item) for item in json_files]

tools = ParseTools()
needed_points = [12, 20, 13, 19, 14, 18, 15, 17, 16]

# anno = tools.parse_anno_file(root + '/annos/027581.json', need_classes=[1])
# print(anno)

f = open(root + '/train2.txt', 'w')
for item in tqdm(json_files):
    try:
        flag = 0
        anno = tools.parse_anno_file(item, need_classes=[1, 2])
        if len(anno['landmarks']) == 0:
            flag = 1
            continue
        landmarks = np.array(anno['landmarks'])
        y = landmarks[..., 1]
        if len(y) == 0:
            continue
        y = y.reshape(np.prod(y.shape))
        if sum(y) == 0:
            continue

        y = np.min(y[y != 0])

        if len(anno['viewpoints']) == 0:
            flag = 1
        if 1 in anno['viewpoints'] or 3 in anno['viewpoints']:
            flag = 1

        for l in anno['landmarks']:
            if flag == 1:
                break
            for point in needed_points:
                if flag == 1:
                    break
                if l[point - 1, 2] == 0:
                    flag = 1
        # print(item)
        if flag == 0:
            f.writelines(item + '\n')
    except:
        continue
f.close()

# from utils.parse_annotations import ParseTools
# from tqdm import tqdm
# needed = ''
# tools = ParseTools()
# with open('/home/corona/datasets/WiderPerson/train/train.txt','r') as f:
#     a = f.readlines()
#
# for item in tqdm(a):
#     item = item.strip()
#     info = tools.load_image(item)
#     print(len(info['labels']))
#     if 1 < len(info['labels']) <= 20:
#         needed += item
#         needed+='\n'
#     else:
#         print('...')
#
with open(root + '/train.txt', 'r') as f:
    anns = f.readlines()
    print(len(anns))
    # for ann in anns:
    #     with open(ann.strip(), 'r') as f:
    #         a = f.read()
    #         a = json.loads(a)
    #     print(a)
