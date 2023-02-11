import json
import logging
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# defining model and loading weights to it
def model_fn(model_dir):
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 133))
    
    with open(os.path.join(model_dir, "model.pth"), 'rb') as f:
        model.load_state_dict(torch.load(f))
        
    model.eval()
    
    return model


# data preprocessing
def input_fn(request_body, request_content_type):
    assert request_content_type == "image/jpeg"
    data = Image.open(io.BytesIO(request_body))
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]) 
    data = transform(data)

    return data

# inference
def predict_fn(input_object, model):  
    with torch.no_grad():
        prediction = model(input_object.unsqueeze(0))
        
    return prediction

# postprocess
def output_fn(predictions, content_type):
    assert content_type == "application/json"
    
    res = predictions.cpu().numpy().tolist()
    return json.dumps(res)