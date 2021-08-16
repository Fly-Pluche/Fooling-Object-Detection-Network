import time

import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from evaluator import MaxExtractor, TotalVariation, UnionDetector
from evaluator import PatchEvaluator
from load_data import ListDatasetAnn
from models import *
import matplotlib.pyplot as plt
from patch import PatchTransformerPro, PatchApplierPro
from patch_config import *
from utils.transforms import CMYK2RGB
from utils.utils import imshow
import warnings
import cv2

warnings.filterwarnings('ignore')

# set random seed
torch.manual_seed(2233)
torch.cuda.manual_seed(2233)
np.random.seed(2233)

# train
torch.autograd.set_detect_anomaly(True)
yolov3 = Yolov3('./config/yolov3.cfg', './weights/yolov3.weights')
points = torch.rand(2, 1000)
points[points < 0] = points[points < 0] * -1
points = points * 415
points.requires_grad_(True)
optimizer = torch.optim.Adam([points], lr=0.05)
for i in tqdm(range(200000000)):
    with torch.autograd.detect_anomaly():
        optimizer.zero_grad()
        points_cuda = points.cuda()
        points_cuda = points_cuda.type(torch.long)
        image = torch.cuda.FloatTensor(3, 416, 416).fill_(1)
        image[:, points_cuda[0], points_cuda[1]] = 0
        result = yolov3(image)
        classes = result['instances'].pred_classes
        boxes = result['instances'].pred_boxes.tensor
        boxes[:, [1, 3]] = 416 - boxes[:, [1, 3]]
        print(classes, end='   ')
        loss_classes = torch.mean(torch.sum(torch.abs(classes - 20)))
        #     boxes = -boxes
        #     boxes[boxes<0] = 0
        loss_boxes = torch.mean(torch.sum(boxes.clone(), dim=1))
        # loss_boxes.backward()
        loss = loss_classes + loss_boxes
        print(loss_classes)
        print(loss_boxes)
        loss.backward()
        optimizer.step()
        points = torch.clip(points, 0, 416)
        if i % 5000 == 0:
            img = np.asarray(functional.to_pil_image(image.cpu()))
            cv2.imwrite(f"./linear/{i}.jpg", img)
        del points_cuda, image, result, loss


