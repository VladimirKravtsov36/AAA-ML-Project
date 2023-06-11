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
            err = ""
            form = await self.request.post()
            if(not form["image"]):
                err = AttributeError("Отправлен пустой запрос")
            else:     
                image = open_image(form["image"].file)
                image_array = np.array(image)
                
                if(image_array.shape[-1] != 3):
                    err = NotImplementedError(
                    "Загрузите другое фото, поддерживаются только .jpg и .jpeg изображения"
                    )       
                else:    
                    angle_check = self.request.app["angle_model"]
                    good_angle = angle_check.is_good_angle(image_array)

                    if good_angle:
                        segmentation_model = self.request.app["segmentation_model"]
                        new_background = segmentation_model.change_background(image_array)
                        new_background_pil = Image.fromarray(new_background)
                        output_file = image_to_img_src(new_background_pil)
                        ctx = {"image": output_file}
                    else:
                        err = ValueError(
                        """Отсутствует автомобиль на фото или 
                        плохой ракурс, загрузите другое фото"""
                    )       
            if (err):
                ctx = {"error": str(err)}
            return render_template("index.html", self.request, ctx)

        except Exception as err:
            ctx = {"error": str(err)}
            return render_template("index.html", self.request, ctx)
