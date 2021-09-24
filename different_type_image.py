"""
测试不同类型的图片对于攻击效果的影响
"""

from utils.utils import *
from evaluator import calculate_asr
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

torch.manual_seed(2233)
torch.cuda.manual_seed(2233)
np.random.seed(2233)
# load image from raw
# img_path_raw = './logs/20210911-171933_base_YOLO_with_coco_datasets2/86.4_asr.raw'
img_path_raw = './logs/20210913-192713_base_YOLO_with_coco_datasets2/87.9_asr.raw'
img_tensor = raw2torch(img_path_raw, np.array([3, 950, 950]))
adv = img_tensor.cuda()
torchvision.utils.save_image(adv, 'a.png')
torchvision.utils.save_image(adv, 'a.jpg')
result = calculate_asr(adv, 4, use_config=False)
print(result)
# (0.8613342647572476, 0.7897435897435897, 0.44038929440389296, 0.6304985337243402)


image_png = read_image_torch('a.png').cuda()
image_jpg = read_image_torch('a.jpg').cuda()

result = calculate_asr(image_png, 4, use_config=False)
print(result)
#  (0.8644778204680406, 0.8041237113402062, 0.4119106699751861, 0.630718954248366)
result = calculate_asr(image_jpg, 4, use_config=False)
print(result)
#  (0.7174292699965071, 0.8346153846153846, 0.4298245614035088, 0.44106463878326996)
