from aiohttp import FormData
from http import HTTPStatus

from bs4 import BeautifulSoup
import pytest


async def test_index_page_contain_valid_multipart_form(client):
    response = await client.get('/')
    text = await response.text()
    assert response.status == HTTPStatus.OK, text
    html = BeautifulSoup(text, 'html.parser')
    form = html.find('form')
    assert form is not None
    assert form.attrs['method'].lower() == 'post'
    assert form.attrs['enctype'].lower() == 'multipart/form-data'
    submit = form.find('button', {'type': 'submit'})
    assert submit is not None


async def test_if_sent_faulty_image_then_error_appear_in_response(client):
    faulty_image_path = 'tests/test_images/bad_example.jpg'
    form = FormData()
    form.add_field(
        name='image',
        value=open(faulty_image_path, 'rb'),
        content_type='image/jpeg',
        filename='test_image.jpg',
    )
    response = await client.post('/', data=form)
    text = (await response.text()).lower()
    assert response.status == HTTPStatus.OK, text
    error = 'no car on photo or bad angle, cannot change background'
    assert error in text, text
