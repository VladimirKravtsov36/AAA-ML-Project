from models.angle_model import AngleModel
from models.segmentation_model import SegmentationModel


ANGLE_WEIGHTS = './models/weights/angle_classification.pt'
SEGMENTATION_WEIGHTS = './models/weights/model_best_loss_resnet_18.pt'


def angle_model():
    return AngleModel(ANGLE_WEIGHTS)


def segmentation_model():
    return SegmentationModel(SEGMENTATION_WEIGHTS)
