from fastapi import FastAPI, Request, File, UploadFile
from pydantic import BaseModel
from pydantic.fields import Field
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import cv2
import io
from typing import List

import numpy as np
templates = Jinja2Templates(directory="templates")
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


class Item(BaseModel):
    name: str = Field(None, min_length=2, max_length=5)


"""
uvicorn test:app --reload
"""


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/test/{name}")
async def root(name):
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


def read_image(bin_data, size=(224, 224)):
    """画像を読み込む

    Arguments:
        bin_data {bytes} -- 画像のバイナリデータ

    Keyword Arguments:
        size {tuple} -- リサイズしたい画像サイズ (default: {(224, 224)})

    Returns:
        numpy.array -- 画像
    """
    file_bytes = np.asarray(bytearray(bin_data.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
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
    return {"response": "OK"}
