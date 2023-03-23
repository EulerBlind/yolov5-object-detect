conda activate deeplearing
cd /home/euler/Github/yolov5
python detect.py --source /opt/euler_datasets/football_dataset/predict/images --weights /home/euler/Github/yolov5/runs/train/exp3/weights/best.pt --conf 0.4 --img 640