"""
some image tools to produce attack
"""

import cv2
import matplotlib.pyplot as plt
from .parse_annotations import ParseTools
import numpy as np
import random
from skimage import transform as stf
import random


class ImageTools:
    def __init__(self):
        self.parse_tools = ParseTools()
        self.label2name = None
        self.label2color = None
        self.classes = {0: 'person'}

    # add patch on the people in the images
    # Rendering function in the paper
    def add_patch_on_image(self, frame, patch, boxes):
        for box in boxes:
            patch, mask = self.produce_patch(patch)
            frame = self.add_patch_in_box(frame, patch, box, mask)
        return frame

    # add patch on the box
    # box: [x_min,y_min,x_max,y_max]
    # mask: a mask has the same size with the patch
    def add_patch_in_box(self, frame, patch, box, mask):
        box_height = box[3] - box[1]
        box_weight = box[2] - box[0]
        scale = box_height / 2.5 / patch.shape[0]  # calculate the scale is used for resizing the mask and patch
        patch = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        patch_height = patch.shape[0]
        patch_weight = patch.shape[1]
        # let patch at the middle of the box
        box_middle_x_min = box[0] + box_weight / 2 - patch_weight / 2 + random.randint(
            -int((box_weight / 2 - patch_weight / 2) / 3), int((box_weight / 2 - patch_weight / 2) / 3)
        )
        box_middle_y_min = box[1] + box_height / 2 - patch_height / 2 + random.randint(
            -int((box_height / 2 - patch_height / 2) / 3), int((box_height / 2 - patch_height / 2) / 3))
        # create a mask and add mask to the right place
        mask_, patch_mask = self.__create_patch_masks(box_middle_x_min, box_middle_y_min, frame, mask, patch,
                                                      patch_height, patch_weight)
        frame[mask_ != 0] = patch_mask[mask_ != 0]
        return frame

    # create two masks of patch; one is colorful mask other only has black and white
    def __create_patch_masks(self, box_middle_x_min, box_middle_y_min, frame, mask, patch, patch_height, patch_weight):
        mask_ = np.zeros(frame.shape, dtype=np.uint8)
        patch_mask = np.copy(mask_)
        # add mask to the right place
        patch_mask[int(box_middle_y_min):int(box_middle_y_min + patch_height),
        int(box_middle_x_min):int(box_middle_x_min + patch_weight)] = patch
        mask_[int(box_middle_y_min):int(box_middle_y_min + patch_height),
        int(box_middle_x_min):int(box_middle_x_min + patch_weight)] = mask
        return mask_, patch_mask

    def cv2_imshow(self, frame, transform=True):
        if transform:
            # bgr to rgb
            b, g, r = cv2.split(frame)
            frame = cv2.merge([r, g, b])
        plt.imshow(frame)
        plt.xticks([])  # remove x label
        plt.yticks([])  # remove y label
        plt.show()

    # draw boxes on the image
    # boxes: [(x_min, y_min, x_max, y_max),.....]
    # labels: a list has the same size with boxes. the label of the boxes.
    def visual(self, frame, boxes, labels, color=(80, 190, 232)):
        for box, label in zip(boxes, labels):
            self.draw_box(frame, box, label, color)
        return frame

    # draw a beautiful box
    def draw_box(self, frame, box, label, color):
        font = cv2.FAST_FEATURE_DETECTOR_THRESHOLD
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness=2)
        cv2.rectangle(frame, (x_min, y_min),
                      (min((frame.shape[1] - 1, x_min + 9 * len(self.classes[int(label)]))),
                       min((frame.shape[0] - 1, y_min + 15))),
                      color, thickness=-1)
        cv2.putText(frame, self.classes[int(label)], (x_min, min(frame.shape[0] - 1, y_min + 10)), font, 0.5,
                    (255, 255, 255), thickness=2)

        return frame

    def produce_random_patch(self):
        imarray = np.random.rand(250, 150, 3) * 255
        im = imarray.astype(np.uint8)
        return im

    def visual_from_file_path(self, file_path):
        frame_info = self.parse_tools.load_image(file_path, is_xywh2xyxy=True)
        print(frame_info['boxes'])
        frame = self.visual(frame_info['image'], frame_info['boxes'], frame_info['labels'])
        return frame

    def random_change_brightness_and_contrast(self, frame):
        # randomly change image's brightness and contrast
        brightness = random.randint(200, 260)
        contrast = random.randint(100, 200)
        effect = self.controller(frame, brightness, contrast)
        return effect

    def controller(self, img, brightness=255, contrast=127):
        brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))
        contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                max = 255
            else:
                shadow = 0
                max = 255 + brightness
            al_pha = (max - shadow) / 255
            ga_mma = shadow
            # the weighted sum of two arrays
            cal = cv2.addWeighted(img, al_pha,
                                  img, 0, ga_mma)
        else:
            cal = img
        if contrast != 0:
            Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
            Gamma = 127 * (1 - Alpha)
            # the weighted sum of two arrays
            cal = cv2.addWeighted(cal, Alpha,
                                  cal, 0, Gamma)
        return cal

    def random_change_rotate(self, image, mask=None):
        angle = random.randint(-10, 10)
        image = self.rotate_image(image, angle)
        if mask is not None:
            mask = self.rotate_image(mask, angle)
            return patch, mask
        return image

    # angle: int
    @staticmethod
    def rotate_image(image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    # shearing change
    def random_shearing(self, frame, mask=None):
        # Create Afine transform
        afine_tf = stf.AffineTransform(shear=random.randint(-10, 10) / 100)
        # Apply transform to image data
        modified = stf.warp(frame, inverse_map=afine_tf) * 255
        modified = modified.astype(np.uint8)
        if mask is not None:  # make mask
            mask_modified = stf.warp(mask, inverse_map=afine_tf) * 255
            mask_modified = mask_modified.astype(np.uint8)
            return modified, mask_modified
        return modified

    # produce patch: change brightness contrast rotate and use shearing change
    def produce_patch(self, patch):
        mask = np.zeros(patch.shape, dtype=np.uint8) + 255
        patch = self.random_change_brightness_and_contrast(patch)
        patch, mask = self.random_change_rotate(patch, mask)
        patch, mask = self.random_shearing(patch, mask)
        return patch, mask


def random_choose(txt_path) -> str:
    with open(txt_path, 'r') as f:
        images = f.readlines()
    image = images[random.randint(0, len(images))]
    image = image.replace('\n', '')
    return image


if __name__ == '__main__':
    tools = ImageTools()
    # img_path = random_choose('/home/corona/datasets/flir_yolo/train/train.txt')
    img_path = "/home/corona/datasets/flir_yolo/train/images/01659.jpeg"
    # img = cv2.imread(img_path)
    img_info = tools.parse_tools.load_image(img_path, is_xyxy2xywh=True)
    patch = tools.produce_random_patch()
    # patch = tools.random_change_rotate(patch)
    image = tools.visual_from_file_path(img_path)
    print(image.shape)
    tools.cv2_imshow(image)
