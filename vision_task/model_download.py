import os
import subprocess

def setup_yolov10():
    # Install YOLOv10 from GitHub
    subprocess.run(['pip', 'install', '-q', 'git+https://github.com/THU-MIG/yolov10.git'], check=True)

    # Ensure the models/weights directory exists
    model_dir = './models/weights'
    os.makedirs(model_dir, exist_ok=True)

    # Download the YOLOv10 model weights
    url = 'https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt'
    subprocess.run(['wget', '-P', model_dir, '-q', url], check=True)

    print("Setup complete. YOLOv10 is installed and model weights are downloaded.")

if __name__ == "__main__":
    setup_yolov10()
