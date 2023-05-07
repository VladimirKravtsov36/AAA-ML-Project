from aiohttp.web import Response
from aiohttp.web import View
from aiohttp_jinja2 import render_template
import numpy as np
from lib.image import image_to_img_src
from lib.image import open_image
from PIL import Image
from io import BytesIO


class IndexView(View):
    async def get(self) -> Response:
        return render_template("index.html", self.request, {})

    async def post(self) -> Response:
        try:
            form = await self.request.post()
            image = open_image(form["image"].file)
            image_array = np.array(image)

            angle_check = self.request.app["angle_model"]

            good_angle = angle_check.is_good_angle(image_array)

            if good_angle:
                segmentation_model = self.request.app["segmentation_model"]
                new_background = segmentation_model.change_background(image_array)
                new_background_pil = Image.fromarray(new_background)
                output_file = image_to_img_src(new_background_pil)
                ctx = {"image": output_file}
                return render_template("index.html", self.request, ctx)
            else:
                return {'Message': 'No car on photo or bad angle, cannot change background'}

        except Exception as err:
            ctx = {"error": str(err)}
            return render_template("index.html", self.request, ctx)
