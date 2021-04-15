import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from models import *
from torch.utils.data import DataLoader
from load_data import ListDatasetAnn
from patch_config import patch_configs
from patch import PatchTransformerPro, PatchApplierPro
from torchvision.transforms import functional
from torchvision import transforms


def predict_one_image(model, image, threshold=0.8):
    """
    Predict one image
    Args:
        model: A default mode from models.py
        image: a rgb np array
        threshold: boxes under this value will not be shown in the result.
    """
    output = model.default_predictor(image)
    result = model.visual_instance_predictions(image, output, mode='pil', threshold=threshold)
    return result


def predict_one_video(model, video_path, output_path, threshold=0.5):
    """
    Predict one video
    Args:
        model: A default mode from models.py
        video_path: the path of the vedio you want to input
        output_path: the pathc of the vedio you want to output
        threshold: boxes under this value will not be shown in the result
    """
    video = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    total_frames = int(video.get(7))
    fps = 25
    _, first_image = video.read()
    video_width, video_height = first_image.shape[1], first_image.shape[0]

    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (int(video_width * 0.6), int(video_height * 0.6)))
    for i in tqdm(range(total_frames - 1)):
        _, frame = video.read()
        # BGR to RGB
        frame = frame[..., ::-1]
        frame = cv2.resize(frame, (int(video_width * 0.6), int(video_height * 0.6)))
        result = predict_one_image(model, frame, threshold)
        video_writer.write(result[..., ::-1])
    video_writer.release()


def get_data_loader():
    # load datasets
    config = patch_configs['base']()
    datasets = ListDatasetAnn(config.deepfashion_txt)
    data_loader = DataLoader(
        datasets,
        batch_size=1,
        num_workers=1,
        shuffle=True,
    )
    return data_loader


def generate_patch(config, load_from_file=None, is_random=False):
    """
    generate patch by loading from file or producing random patch
    Args:
        config: the config set about patch
        load_from_file: the path of the patch file
        is_random: choose to produce a random patch or not
    """
    # load a image from local patch
    if load_from_file is not None:
        patch = Image.open(load_from_file)
        patch = patch.resize((config.patch_size, config.patch_size))
        patch = transforms.PILToTensor()(patch) / 255.
        return patch
    if is_random:
        return torch.rand((3, config.patch_size, config.patch_size))
    return torch.full((3, config.patch_size, config.patch_size), 0.5)


def generate_attacked_results(adv_patch_cpu, config, data_loader, model):
    patch_transformer = PatchTransformerPro().cuda()
    patch_applier = PatchApplierPro().cuda()
    adv_patch = adv_patch_cpu.cuda()
    i = 0
    for image_batch, clothes_boxes_batch, people_boxes_batch, labels_batch, landmarks_batch, segmentations_batch in tqdm(
            data_loader):
        image_batch = image_batch.cuda()
        labels_batch = labels_batch.cuda()
        people_boxes_batch = people_boxes_batch.cuda()
        adv_batch_t, adv_batch_mask_t = patch_transformer(adv_patch,
                                                          clothes_boxes_batch,
                                                          segmentations_batch,
                                                          landmarks_batch,
                                                          image_batch)
        p_img_batch = patch_applier(image_batch, adv_batch_t, adv_batch_mask_t)
        p_img_batch = F.interpolate(p_img_batch, (config.img_size[1], config.img_size[0]))
        # show apply affection
        # plt.imshow(np.array(functional.to_pil_image(image_batch[0])))
        # plt.show()
        p_img = p_img_batch[0]
        img = image_batch[0]
        with torch.no_grad():
            p_img_output = model(p_img)
            img_output = model(img)
        # p_scores = p_img_output['instances'].scores
        # scores = img_output['instances'].scores
        # p_labels = p_img_output['instances'].pred_classes
        # labels = p_img_output['instances'].pred_classes

        p_img = model.visual_instance_predictions(p_img, p_img_output)
        img = model.visual_instance_predictions(img, img_output)

        cv2.imwrite('outputs/' + str(i) + '_0.jpg', p_img[:, :, ::-1])
        cv2.imwrite('outputs/' + str(i) + '_1.jpg', img[:, :, ::-1])
        i += 1
        # plt.imshow(p_img)
        # # plt.imshow(np.array(functional.to_pil_image(p_img)))
        # plt.show()


if __name__ == '__main__':
    model = FasterRCNN()
    data_loader = get_data_loader()
    config = patch_configs['base']()
    adv_patch_cpu = generate_patch(config, load_from_file='patches/retinanet.jpg')
    generate_attacked_results(adv_patch_cpu, config, data_loader, model)
    # predict_one_video(model, 'vedios/VID_20210408_081759.mp4', 'vedios/output.mp4')
    # image = np.array(Image.open('images/IMG_20210407_114942.jpg').resize((1066, 800)))
    # print(image.shape)
    # # image = np.resize(image, (300, 400, 3))
    # # print(image.shape)
    # output = model.default_predictor(image)
    # a = model.visual_instance_predictions(image, output, mode='pil', threshold=0.5)
    # import matplotlib.pyplot as plt
    #
    # plt.imshow(a)
    # plt.show()
