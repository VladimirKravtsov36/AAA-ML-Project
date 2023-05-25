from unittest.mock import patch, MagicMock
import json
import pytest
from PIL import Image
import os
import numpy as np
from models.angle_model import AngleModel
import glob

# Укажи полный путь к проекту в .env файле
from dotenv import load_dotenv

dotenv_path = ".env"
load_dotenv(dotenv_path)

PATH_TO_SERVICE = os.getenv("PATH_TO_PROJ")

ERR_STR = "No car on photo or bad angle, cannot change background"
##"models/weights/angle_model_8_0.8223.pt" для efficientnet_v2_s()
##"models/weights/angle_classification.pt" для effnet_b2
ANGLE_WEIGHTS = PATH_TO_SERVICE + "/" + "./models/weights/mobilenet_best_loss_21_0.8651.pt"
BAD_ANGLE_IMAGES_LIST = ["opened_doors_image1.jpg","opened_doors_image2.jpg"]
BAD_ANGLE_IMAGES_LIST_ALL = glob.glob("tests/test_images/bad_images/*.jpg")
GOOD_ANGLE_IMAGES_LIST_ALL = glob.glob("tests/test_images/good_images/*.jpg")


PATH_TO_TESTS = PATH_TO_SERVICE + "/" + "tests/test_images"
url = "http://localhost:8080/change_background"

@pytest.mark.xfail(reason="opeded doors class detection not implemented by classification model (too few samples)")
@pytest.mark.parametrize("test_img_name", BAD_ANGLE_IMAGES_LIST)
def test_check_bad__failed_examples(test_img_name):

    img_path = os.path.join(PATH_TO_TESTS, test_img_name)
    angle_model = AngleModel(ANGLE_WEIGHTS)
    image = Image.open(img_path)
    image_array = np.array(image)
    angle_is_good = angle_model.is_good_angle(image_array)

    assert (
        angle_is_good == False
    ), f"Expected is_good_angle=False, got {angle_model.predict(image_array)} class"


def test_check_bad_examples_stats():

    counter_all = len(BAD_ANGLE_IMAGES_LIST_ALL)
    correct = 0
    angle_model = AngleModel(ANGLE_WEIGHTS)
    for curr_img_path in BAD_ANGLE_IMAGES_LIST_ALL:
        img_path = os.path.join(PATH_TO_SERVICE, curr_img_path)
        image = Image.open(img_path)
        image_array = np.array(image)
        angle_is_good = angle_model.is_good_angle(image_array)
        if not angle_is_good:
            correct += 1
    assert (
        correct / counter_all >= 0.9
    ), f"Expected 90% or more correct answers, got {correct / counter_all}"


def test_check_good_examples_stats():

    counter_all = len(GOOD_ANGLE_IMAGES_LIST_ALL)
    correct = 0
    angle_model = AngleModel(ANGLE_WEIGHTS)
    for curr_img_path in GOOD_ANGLE_IMAGES_LIST_ALL:
        img_path = os.path.join(PATH_TO_SERVICE, curr_img_path)
        image = Image.open(img_path)
        image_array = np.array(image)
        angle_is_good = angle_model.is_good_angle(image_array)
        correct += angle_is_good
    assert (
        correct / counter_all >= 0.85
    ), f"Expected 85% or more correct answers, got {correct / counter_all}"
