from torchvision import transforms


class Compose_(transforms.Compose):
    def __init__(self, transforms):
        super(Compose_, self).__init__(transforms)

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img
