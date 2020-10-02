from PIL import Image
import torchvision.models as models
from torchvision import transforms
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from cam.base_cam import GradCAMpp

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class ImageCropper:
    def __init__(self, crop_width, crop_height, interpolation_mode=Image.BICUBIC):
        self.width = crop_width
        self.height = crop_height
        self.interpolation_mode = interpolation_mode
        self.gradcam_model = self._load_vgg_gradcam()

    def _load_vgg_gradcam(self):
        vgg = models.vgg16(pretrained=True).eval()
        vgg_model_dict = dict(
            type="vgg16",
            arch=vgg,
            layer_name="feature_29",
            input_size=(224, 224)
        )
        return GradCAMpp(vgg_model_dict)

    def crop(self, img):
        img_h, img_w = img.height, img.width
        crop_h = min(self.height, img_h)
        crop_w = min(self.width, img_w)

        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0)
        scorecam_map, _ = self.gradcam_model(input_batch)
        scorecam_scores = scorecam_map[0][0]
        # uint8 にするために 255 を掛ける
        np_scorecam_map = scorecam_scores.numpy() * 255
        im_scores = np.asarray(
            Image.fromarray(np_scorecam_map.astype(np.uint8)).resize(
                (img_w, img_h), self.interpolation_mode))

        max_score = 0
        maxh, maxw = 0, 0
        # 最大スコアの部分を探索
        for i in range(img_h - crop_h + 1):  # 縦
            for j in range(img_w - crop_w + 1):  # 横
                crop_score = im_scores[i: i + crop_h, j: j + crop_w].sum()
                if crop_score > max_score:
                    max_score = crop_score
                    maxh, maxw = i, j

        return img.crop((maxw, maxh, maxw + crop_w, maxh + crop_h))
