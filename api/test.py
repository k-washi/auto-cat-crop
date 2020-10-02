from fastapi import FastAPI, Request, File, UploadFile
from pydantic import BaseModel
from pydantic.fields import Field
from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="templates")


class Item(BaseModel):
    name: str = Field(None, min_length=2, max_length=5)


app = FastAPI()

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
