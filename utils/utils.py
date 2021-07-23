import torch


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


def imshow(img):
    pass





