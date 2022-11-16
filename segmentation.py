import torch
from PIL import Image, ImageDraw, ImageFont
import io
import easyocr
import os 


def get_yolov5():
    model = torch.hub.load('./yolov5', 'custom', path='./yolov5/runs/train/yolov5s_results/weights/best.pt' ,source='local') 
    model.conf = 0.7
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


def detect_and_return_ocr(input_image):
    model = get_yolov5()

    results = model(input_image)
    result_df = results.pandas().xyxy[0]
    results.render() 
    output_file_dir = os.path.join(os.getcwd(), "out_put")
    results.save(save_dir=output_file_dir)
    output_file_path = os.path.join(output_file_dir, os.listdir(output_file_dir)[0])
    if not result_df.empty:
        image = Image.open(output_file_path)
        cropped_image = image.crop((result_df.iloc[0].tolist()[:4]))
        cropped_image.save("cropped_image.jpg")
        reader = easyocr.Reader(['en'])
        result_text = reader.readtext("cropped_image.jpg")
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("./arial.ttf", 80)
        draw.text((0, 0), result_text[0][1], (255, 0, 0) , font=font , align="center")
        image.save(output_file_path)
        result_df["text"] = result_text[0][1]
        return result_df
    else:
        return "No Number Plate Found"
