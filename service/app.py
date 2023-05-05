from flask import Flask, request, Response
from PIL import Image
import numpy as np
from io import BytesIO

from models.angle_model import AngleModel
from models.segmentation_model import SegmentationModel

app = Flask(__name__)

ANGLE_WEIGHTS = './weights/angle_classification.pt'
SEGMENTATION_WEIGHTS = './weights/model_best_loss_resnet_18.pt'

angle_model = AngleModel(ANGLE_WEIGHTS)
segmentation_model = SegmentationModel(SEGMENTATION_WEIGHTS)


@app.route('/change_background', methods=['POST'])
def change_backgound():
    """
    Замена фона на белый
    Работает только с .jpg или .jpeg файлами!
    """
    image_file = request.files['image']
    image = Image.open(image_file.stream)
    image_array = np.array(image)

    good_angle = angle_model.is_good_angle(image_array)

    if good_angle:
        new_background = segmentation_model.change_background(image_array)
        new_background_pil = Image.fromarray(new_background)
        output_file = BytesIO()
        new_background_pil.save(output_file, format=image.format)
        output_file.seek(0)
        return Response(output_file.read(), mimetype='image/' + image.format.lower())
    else:
        return {'Message': 'No car on photo or bad angle, cannot change background'}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
