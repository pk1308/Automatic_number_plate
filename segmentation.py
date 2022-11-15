import torch
from PIL import Image, ImageDraw, ImageFont
import io
import easyocr


def get_yolov5():
    model = torch.hub.load('./yolov5', 'custom', path='./model/best.pt', source='local')
    model.conf = 0.8
    return model


def get_image_from_bytes(binary_image, max_size=1024):
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    width, height = input_image.size
    resize_factor = min(max_size / width, max_size / height)
    resized_image = input_image.resize(
        (
            int(input_image.width * resize_factor),
            int(input_image.height * resize_factor),
        )
    )
    return resized_image


def detect_and_return_ocr(result, image_path):
    result = result.pandas().xyxy[0]
    if len(result) != 0:
        image = Image.open(image_path)
        cropped_image = image.crop((result.iloc[0].tolist()[:4]))
        cropped_image.save("cropped_image.jpg")
        reader = easyocr.Reader(['en'])
        result_text = reader.readtext("cropped_image.jpg")
        draw = ImageDraw.Draw(image)
        myFont = ImageFont.truetype('FreeMono.ttf', 65)
        draw.text((0, 0), result_text[0][1], (255, 0, 0) , font=myFont)
        image.save(image_path)
        return result_text[0][1]
    else:
        return "No Number Plate Found"
