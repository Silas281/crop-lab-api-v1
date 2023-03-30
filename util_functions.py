import io
import base64
import datetime
from fastapi import File, UploadFile
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from torchvision.datasets import ImageFolder
import json


def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


device = get_device()
# model
model = loaded_model = torch.load('./crop_labv1.pth', map_location='cpu')


# imagenet classes
class_names = ['Background_without_leaves', 'Cassava_Bacterial_Blight_(CBB)', 'Cassava_Brown_Streak_Disease_(CBSD)', 'Cassava_Green_Mottle_(CGM)', 'Cassava_Healthy', 'Cassava_Mosaic_Disease_(CMD)', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
               'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Rice_Brown_Spot', 'Rice_Healthy', 'Rice_Hispa', 'Rice_Leaf_Blast', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']


def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def predict_raw_image(single_img_path):
    """Predict disease"""
    imgByte = single_img_path.file.read()
    img = transform_image(imgByte)

    x = img
    output = model(x)
    _, preds = torch.max(output, dim=1)

    result = {
        "predictions": {
            "class_name": class_names[preds[0]]
        }
    }

    return result
