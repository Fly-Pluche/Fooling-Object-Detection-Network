import cv2
from PIL import Image
from tqdm import tqdm
from models import *


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


if __name__ == '__main__':
    model = MaskRCNN()
    predict_one_video(model, 'vedios/VID_20210408_081759.mp4', 'vedios/output.mp4')
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
