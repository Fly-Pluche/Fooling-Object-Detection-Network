import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

# Model
torch.hub.set_dir('./')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=True)
model = model.autoshape()  # add autoshape wrapper IMPORTANT

# Images
torch.hub.download_url_to_file(
    'https://raw.githubusercontent.com/ultralytics/yolov5/master/inference/images/zidane.jpg', 'zidane.jpg')
img1 = Image.open('../yolov3/weights/zidane.jpg')  # PIL
# img2 = cv2.imread('zidane.jpg')[:, :, ::-1]  # opencv (BGR to RGB)
# img3 = np.zeros((640, 1280, 3))  # numpy
# imgs = [img1, img2, img3]  # batched inference

# # Inference
prediction = model([img1], size=640)  # includes NMS
print(prediction.xyxy)

#
# # Plot
# for i, img in enumerate(imgs):
#     print('\nImage %g/%g: %s ' % (i + 1, len(imgs), img.shape), end='')
#     img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img  # from np
#     if prediction[i] is not None:  # is not None
#         for *box, conf, cls in prediction[i]:  # [xy1, xy2], confidence, class
#             print('class %g %.2f, ' % (cls, conf), end='')  # label
#             ImageDraw.Draw(img).rectangle(box, width=3)  # plot
#     img.save('results%g.jpg' % i)  # save


# Camera
# img1 = Image.open('zidane.jpg')  # PIL
# img2 = cv2.imread('zidane.jpg')[:, :, ::-1]  # opencv (BGR to RGB)
# img3 = np.zeros((640, 1280, 3))  # numpy
# imgs = [img1, img2, img3]  # batched inference

# Inference
# prediction = model(imgs, size=640)  # includes NMS

# # Plot
# for i, img in enumerate(imgs):
#     print('\nImage %g/%g: %s ' % (i + 1, len(imgs), img.shape), end='')
#     img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img  # from np
#     if prediction[i] is not None:  # is not None
#         for *box, conf, cls in prediction[i]:  # [xy1, xy2], confidence, class
#             print('class %g %.2f, ' % (cls, conf), end='')  # label
#             ImageDraw.Draw(img).rectangle(box, width=3)  # plot
#     img.save('results%g.jpg' % i)  # save

# cap = cv2.VideoCapture(0)
#
# while (True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     print('frame', frame)
#     prediction = model(frame, size=640)
#     cv2.imshow('aaa', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()
