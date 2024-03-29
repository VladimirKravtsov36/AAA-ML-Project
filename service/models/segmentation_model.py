import numpy as np
import torch
from torchvision import transforms as T
import segmentation_models_pytorch as smp
import cv2
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


class SegmentationModel:
    """
    Модель для сегментации авто
    change_background возвращает обрезаное авто на белом фоне
    
    """
    def __init__(self, weights: str):
        "Инициализация модели, загрузка весов и переход в eval mode"
        
        self.model = smp.DeepLabV3Plus("resnet18", classes=1,
                                       encoder_weights=None)

        self.model.load_state_dict(torch.load(weights,
                                              map_location=torch.device('cpu')))
        self.model.eval()

    def preprocess(self, image: np.array) -> torch.Tensor:

        transforms = T.Compose([
            T.ToPILImage(),
            T.Resize(size=(224, 224)),
            T.ToTensor(),
            T.Normalize(MEAN, STD)
        ])

        return transforms(image)

    def predict(self, image: np.array) -> torch.Tensor:

        self.src_image_size = image.shape[:2]

        prep_image = self.preprocess(image)
        prediction = self.model(prep_image[None, :, :, :])

        logits = prediction.sigmoid().detach().numpy()[0][0]

        return (logits > 0.5)*255

    def change_background(self, image: np.array) -> np.array:
        """
        Возвращает обрезаное авто на белом фоне
        """
        mask = self.predict(image)
        mask = cv2.resize(mask.astype('uint8'), self.src_image_size[::-1])

        white_background = Image.new('RGB', self.src_image_size[::-1], (255, 255, 255))
        pil_mask = Image.fromarray(mask)
        pil_image = Image.fromarray(image)

        white_background.paste(pil_image, mask=pil_mask)

        return np.array(white_background)
