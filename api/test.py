from fastapi import FastAPI, Request, File, UploadFile
from pydantic import BaseModel
from pydantic.fields import Field
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import cv2
import io
from typing import List
from PIL import Image
import numpy as np
import sys

import imagecrop

templates = Jinja2Templates(directory="templates")
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

crop_width, crop_height = 200, 200
crop_model = imagecrop.ImageCropper(crop_width, crop_height)


class Item(BaseModel):
    name: str = Field(None, min_length=2, max_length=5)


"""
uvicorn test:app --reload
"""


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/test/{name}")
async def root(name: str):
    return {"message": "Hello " + name}


@app.post("/items/")
async def create_item(item: Item):
    # curl -X POST -d'{"name": "kai"}' http://127.0.0.1:8000/items/
    data = {
        "name": f"your name : {item.name}",
        "res": "OK"
    }
    return data


@app.get("/template")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/img_post")
async def img_post(request: Request):
    return templates.TemplateResponse("img_post.html", {"request": request})


def read_image(bin_data):
    """画像を読み込む

    Arguments:
        bin_data {bytes} -- 画像のバイナリデータ

    Returns:
        numpy.array -- 画像
    """
    file_bytes = np.asarray(bytearray(bin_data.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return img


@app.post("/api/image_recognition")
async def image_recognition(files: List[UploadFile] = File(...)):
    """画像認識API

    Keyword Arguments:
        files {List[UploadFile]} -- アップロードされたファイル情報 (default: {File(...)})

    Returns:
        dict -- 推論結果
    """
    bin_data = io.BytesIO(files[0].file.read())
    img = read_image(bin_data)
    cropped_img = crop_model.crop(img)
    cropped_img.save("./cropped.jpg")
    return {"response": "OK"}
