# inpiration: https://keras.io/examples/keras_recipes/creating_tfrecords/

import cv2
import os
import json
import sys

dataset_dir='data/ds0'
videos_dir=os.path.join(dataset_dir, "video")
ann_dir=os.path.join(dataset_dir, "ann")

#import supervisely_lib as sly
#from supervisely_lib.project.project import Dataset

#project = sly.Project('./data', sly.OpenMode.READ)

#ds = sly.project.project.Dataset(dataset_dir, sly.OpenMode.READ)
#print(ds)

for filename in os.listdir(videos_dir):
    video_filename=os.path.join(videos_dir,filename)

    video = cv2.VideoCapture(video_filename)

    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()
    
#    count = 0
#    while ok:
#        cv2.imwrite("frame%4d.jpg" % count, image)     # save frame as JPEG file      
#        ok,image = vidcap.read()
#        print('Read a new frame: ', ok)
#        count += 1
    annotation_filename=os.path.join(ann_dir,filename+".json")
    with open(annotation_filename) as f:
      annotations = json.load(f)
      print(annotations)
      for frame in annotations['frames']:
          print(frame['index'])

    