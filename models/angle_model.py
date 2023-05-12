import numpy as np
import torch
import torchvision
from torchvision import transforms as T

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


class AngleModel:
    """
    Модель для предсказания угла обзора фото
    Рассматриваются следующие типы (подробнее в соответствующей инструкции):
    
    0. Стандартный вид (видно перед и заднее колесо)
    1. Открытые двери (или багажник)
    2. Вид сбоку
    3. Вид сзади 
    4. Вид спереди
    5. Нет машины
    6. Другое
    """
    def __init__(self, weights: str):

        self.model = torchvision.models.efficientnet_b2()#efficientnet_v2_s() or efficientnet_b2()
        self.model.classifier[1] = torch.nn.Linear(1408, 7)#(1280, 7) for efficientnet_v2_s, (1408, 7), for efficientnet_b2()
        self.model.classifier.append(torch.nn.Softmax())

        self.model.load_state_dict(torch.load(weights,
                                              map_location=torch.device('cpu')))
        self.model.eval()

    def preprocess(self, image: np.array) -> torch.Tensor:

        transforms = T.Compose([
            T.ToPILImage(),
            T.Resize(size=(288, 288), #(384, 384) for efficientnet_v2_s() or (288, 288) for efficientnet_b2()
                     interpolation=T.InterpolationMode.BICUBIC),#BILINEAR for efficientnet_v2_s() or BICUBIC for efficientnet_b2()
            T.ToTensor(),
            T.Normalize(MEAN, STD)
        ])

        return transforms(image)

    def predict(self, image: np.array) -> int:
        """
        Возвращает метку класса от 0 до 6, 
        подробнее в описании класса и инструкции
        """
        prep_image = self.preprocess(image)
        prediction = self.model(prep_image[None, :, :, :])[0].argmax()

        return int(prediction)

    def is_good_angle(self, image: np.array) -> bool:
        """
        Функция для отсеивания авто по углу обзора
        Оставляются только следующие типы:
        
        1. Стандартный вид (видно перед и заднее колесо)
        2. Вид сбоку
        3. Вид сзади 
        4. Вид спереди
        """
        prediction = self.predict(image)

        return prediction in [0, 2, 3, 4]
