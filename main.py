from fastapi import FastAPI, File
from segmentation import get_yolov5, get_image_from_bytes, detect_and_return_ocr
from starlette.responses import Response
import io
from PIL import Image
import os
import shutil
import json
from fastapi.middleware.cors import CORSMiddleware

MESSAGE = f""" Automatically detect number plate and return the text in the image."""
model = get_yolov5()

app = FastAPI(
    title="Custom YOLOV5 API",
    description=MESSAGE,
    version="0.0.1",
)

origins = ["http://localhost", "http://localhost:8000", "*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def index():
    return {
        "Classes trained": MESSAGE,
    }


@app.get("/notify/v1/health")
def get_health():
    """
    Usage on K8S
    readinessProbe:
        httpGet:
            path: /notify/v1/health
            port: 80
    livenessProbe:
        httpGet:
            path: /notify/v1/health
            port: 80
    :return:
        dict(msg='OK')
    """
    return dict(msg="OK")


@app.post("/object-to-json")
async def detect_Number_plate_return_json_result(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    result = detect_and_return_ocr(input_image)
    output_file_dir = os.path.join(os.getcwd(), "out_put")
    shutil.rmtree(output_file_dir)
    return {"result": result.to_dict()}


@app.post("/object-to-img")
async def detect_Number_plate_return_base64_img(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    result = detect_and_return_ocr(input_image)
    output_file_dir = os.path.join(os.getcwd(), "out_put")
    file_path = os.path.join(output_file_dir, os.listdir(output_file_dir)[0])
    bytes_io = io.BytesIO()
    img_base64 = Image.open(file_path).convert("RGB")
    img_base64.save(bytes_io, format="jpeg")
    shutil.rmtree(output_file_dir)
    return Response(content=bytes_io.getvalue() , media_type="image/jpeg") 
