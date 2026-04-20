from IPython.lib.display import exists
import os
from ultralytics import YOLO
import torch

RUN_DIR = '../results'
LOG_FILE = '../results/training_log.log'

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def main():
  print(f'CUDA available: {torch.cuda.is_available()}')
  print('Start training on CUDA')

  model = YOLO('yolov8n.pt')

  model.train(
      data='../configs/data.yaml',
      epochs=40,
      imgsz=640,
      batch=16,
      device=0,
      cache=True,
      workers=2,
      close_mosaic=10,
      project='runs/detect',
      name='adas_yolov8n_det',
      amp=True,
      patience=10
  )
  print("Training finished")


if __name__ ==  "__main__":
  main()