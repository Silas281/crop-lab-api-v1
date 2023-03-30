import util_functions
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI()


@app.get("/")
def homePage(request: Request):
    return {"request": request}


@app.post("/diagnose")
async def predict(file: UploadFile = File(...)):
    return util_functions.predict_raw_image(file)
