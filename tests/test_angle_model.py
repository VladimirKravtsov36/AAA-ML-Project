from unittest.mock import patch, MagicMock
import json
import pytest
from PIL import Image
import os
import numpy as np
from models.angle_model import AngleModel 
#Укажи полный путь к проекту в .env файле
from dotenv import load_dotenv
dotenv_path = '.env'
load_dotenv(dotenv_path)

PATH_TO_SERVICE = os.getenv('PATH_TO_PROJ')

ERR_STR = "No car on photo or bad angle, cannot change background"
ANGLE_WEIGHTS = PATH_TO_SERVICE + '/' + "models/weights/angle_classification.pt"
BAD_ANGLE_IMAGES_LIST = ["opened_doors_image.jpg","bad_example.jpg"]

PATH_TO_TESTS = PATH_TO_SERVICE + '/' + 'tests/test_images'
url = 'http://localhost:8080/change_background'


@pytest.mark.parametrize("test_img_name", BAD_ANGLE_IMAGES_LIST)
def test_check_bad_examples(test_img_name):
    
    img_path = os.path.join(PATH_TO_TESTS,test_img_name)
    angle_model = AngleModel(ANGLE_WEIGHTS)
    image = Image.open(img_path)
    image_array = np.array(image)
    angle_is_good = angle_model.is_good_angle(image_array)
    
    assert angle_is_good == False, f"Expected is_good_angle=False, got {angle_model.predict(image_array)} class" 

