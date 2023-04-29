import numpy as np
import torch
import torchvision
from torchvision import transforms as T

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


class AngleModel:

    def __init__(self, weights: str):

        self.model = torchvision.models.efficientnet_b2()
        self.model.classifier[1] = torch.nn.Linear(1408, 7)
        self.model.classifier.append(torch.nn.Softmax())

        self.model.load_state_dict(torch.load(weights,
                                              map_location=torch.device('cpu')))
        self.model.eval()

    def preprocess(self, image: np.array) -> torch.Tensor:

        transforms = T.Compose([
            T.ToPILImage(),
            T.Resize(size=(288, 288),
                     interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(MEAN, STD)
        ])

        return transforms(image)

    def predict(self, image: np.array) -> int:
        prep_image = self.preprocess(image)
        prediction = self.model(prep_image[None, :, :, :])[0].argmax()

        return int(prediction)

    def is_good_angle(self, image: np.array) -> bool:

        prediction = self.predict(image)

        return prediction in [0, 2, 3, 4]
