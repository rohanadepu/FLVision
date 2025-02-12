
# Federated Learning with Flower and YOLO on the VisDrone Dataset

This project demonstrates how to set up a federated learning environment using the Flower framework with a YOLOv10 object detection model trained on a custom dataset. The example includes running a Flower server with optional secure aggregation and a YOLOv10 client training on the VisDrone dataset.

## Resources

- **YOLOv10 Training Notebook:** [Train YOLOv10 on Custom Dataset](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov10-object-detection-on-custom-dataset.ipynb#scrollTo=SaKTSzSWnG7s)
- **Ultralytics YOLOv8 Tutorial:** [Ultralytics YOLOv8 Tutorial Notebook](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb)
- **VisDrone Dataset Details:** [VisDrone Dataset YAML](https://docs.ultralytics.com/datasets/detect/visdrone/#dataset-yaml)
- **Weights mismatch problem:** [Weights mismatch yolov8-flower](https://github.com/ultralytics/ultralytics/issues/9554) 

## Installation

Before running the server and client scripts, ensure you have the necessary dependencies:

```bash
pip install flwr
pip install torch
pip install ultralytics
```


Initialize the weights for one epoch on the server to avoid weights mismatch during aggregation.

```bash
python yolo_init.py 
```


## Running the Server

To start the Flower server with secure aggregation enabled, run the following command in your terminal:

```bash
python server.py 
```

## Running the Client



To start the YOLOv8 client  use the following command:


```bash
python client.py 
```
