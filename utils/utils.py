import torch
import matplotlib.pyplot as plt
from torchvision.transforms import functional
import numpy as np
from PIL import Image
import torchvision

def boxes_scale(boxes, img_size, target_img_size):
    """
    Scale the box according to the size of the picture
    boxes: [N,x_min,y_min,x_max,y_max]
    img_size([w,h]) ==> target_img_size([w,h])
    """
    boxes[:, 0] = target_img_size[0] * boxes[:, 0] / img_size[0]
    boxes[:, 1] = target_img_size[1] * boxes[:, 1] / img_size[1]
    boxes[:, 2] = target_img_size[0] * boxes[:, 2] / img_size[0]
    boxes[:, 3] = target_img_size[1] * boxes[:, 3] / img_size[1]
    return boxes


def pytorch_imshow(img):
    """
    fake cv2 pytorch_imshow
    Args:
        img: a pytorch tensor
    """
    img = np.asarray(functional.to_pil_image(img))
    plt.imshow(img)
    plt.show()


def save_raw(np_image, file_path):
    """
    save row type image
    Args:
        np_image: a numpy array
        file_path: file name to save
    """
    np_image = np_image.astype(np.float32)
    np_image.tofile(file_path)


def torch2raw(torch_image, file_path):
    """
    torch to image
    Args:
        torch_image: a torch array
        file_path: the path to save row file
    """
    img = np.asarray(torch_image)
    save_raw(img, file_path)


def raw2torch(file_path, image_size):
    """
    row image to torch tensor
    Args:
        file_path: file path to read
    """
    raw_img = np.fromfile(file_path, dtype=np.float32)
    raw_img = np.resize(raw_img, image_size)
    img = torch.tensor(raw_img)
    return img


def read_image_torch(img_path):
    img = Image.open(img_path)
    img = functional.pil_to_tensor(img) / 255.0
    return img
