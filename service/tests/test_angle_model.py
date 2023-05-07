from unittest.mock import patch, MagicMock
import json
import pytest
from PIL import Image
import os
import numpy as np
from models.angle_model import AngleModel 

ERR_STR = "No car on photo or bad angle, cannot change background"
ANGLE_WEIGHTS = './weights/angle_classification.pt'
BAD_ANGLE_IMAGES_LIST = ["opened_doors_image.jpg","bad_example.jpg"]
GOOD_ANGLE_IMAGES_LIST = ["front_view.jpg","back_view.jpg","standard.jpg","side_view.jpg","truck_standard.jpg"]

PATH_TO_SERVICE = r"G:\Vlad\avito\car_segmentation_project\AAA-ML-Project\service\test_images"

url = 'http://localhost:8080/change_background'


@pytest.mark.parametrize("test_img_name", BAD_ANGLE_IMAGES_LIST)
def test_check_bad_examples(test_img_name):
    
    img_path = os.path.join(PATH_TO_SERVICE,test_img_name)
    angle_model = AngleModel(ANGLE_WEIGHTS)
    image = Image.open(img_path)
    image_array = np.array(image)
    angle_is_good = angle_model.is_good_angle(image_array)
    
    assert angle_is_good == False, f'Expected False, got {angle_model.predict(image_array)} class' 

@pytest.mark.parametrize("test_img_name", GOOD_ANGLE_IMAGES_LIST)
def test_check_good_examples(test_img_name):
    
    img_path = os.path.join(PATH_TO_SERVICE,test_img_name)
    angle_model = AngleModel(ANGLE_WEIGHTS)
    image = Image.open(img_path)
    image_array = np.array(image)
    angle_is_good = angle_model.is_good_angle(image_array)
    
    assert angle_is_good == True, f'Expected angle_is_good == False, got {angle_model.predict(image_array)} class' 
