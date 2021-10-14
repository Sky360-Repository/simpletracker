# inpiration: https://keras.io/examples/keras_recipes/creating_tfrecords/

import cv2
import os

dataset_dir='data/annotations/ds0'
videos_dir=os.path.join(dataset_dir, "video")
ann_dir=os.path.join(dataset_dir, "ann")

for filename in os.listdir(videos_dir):
    video_filename=os.path.join(videos_dir,filename)
    vid = cv2.VideoCapture(video_filename)

    annotation_filename=os.path.join(videos_dir,filename,".json")
