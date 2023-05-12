from models.angle_model import AngleModel
from models.segmentation_model import SegmentationModel


ANGLE_WEIGHTS = "./models/weights/angle_model_b2_0.8021.pt"
SEGMENTATION_WEIGHTS = "./models/weights/0.0487_24_epochs_resnet18.pt"
# models/weights/angle_model_8_0.8223.pt - efficientnet_v2_s


def angle_model():
    return AngleModel(ANGLE_WEIGHTS)


def segmentation_model():
    return SegmentationModel(SEGMENTATION_WEIGHTS)
