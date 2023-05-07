from pathlib import Path

import aiohttp_jinja2
import jinja2
from aiohttp.web import Application

from lib import views
from lib.models import angle_model, segmentation_model


lib = Path("lib")


def create_app() -> Application:
    app = Application()
    # setup routes
    app.router.add_static("/static/", lib / "static")
    app.router.add_view("/", views.IndexView, name="index")
    # setup templates
    aiohttp_jinja2.setup(
        app=app,
        loader=jinja2.FileSystemLoader(lib / "templates"),
    )
    app["angle_model"] = angle_model()
    app["segmentation_model"] = segmentation_model()
    return app


async def async_create_app() -> Application:
    return create_app()
