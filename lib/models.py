from models.angle_model import AngleModel
from models.segmentation_model import SegmentationModel


ANGLE_WEIGHTS = "./models/weights/mobilenet_best_loss_21_0.8651.pt"
SEGMENTATION_WEIGHTS = "./models/weights/0.0487_24_epochs_resnet18.pt"
# models/weights/angle_model_8_0.8223.pt - efficientnet_v2_s


def angle_model():
    return AngleModel(ANGLE_WEIGHTS)


def segmentation_model():
    return SegmentationModel(SEGMENTATION_WEIGHTS)
