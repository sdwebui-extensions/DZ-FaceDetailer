import requests
import os, sys
import subprocess
from tqdm import tqdm
from pip._internal import main as pip_main
from pathlib import Path
from folder_paths import models_dir

try:
    import mediapipe
except:
    print('FaceDetailer: Installing requirements')
    my_path = os.path.dirname(__file__)
    subprocess.check_call([sys.executable, "-s", "-m", "pip", "install", "-r", os.path.join(my_path, "requirements.txt")])

model_url = "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt"

save_loc = os.path.join(models_dir, "dz_facedetailer", "yolo")

if not os.path.exists(save_loc):
    print('FaceDetailer: Creating models directory')
    os.makedirs(save_loc, exist_ok=True)
    download_model()
else:
    print('FaceDetailer: Model directory already exists')
    download_model()

from .DZFaceDetailer import FaceDetailer

NODE_CLASS_MAPPINGS = {
    "DZ_Face_Detailer": FaceDetailer,
}


