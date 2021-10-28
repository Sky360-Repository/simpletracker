import argparse
import os
import json
import csv

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--stf_path', help='Path to STF directory')
    parser = parser.parse_args(args)

    if parser.stf_path is None:
            raise ValueError('Must provide --stf_path when training on STF,')
    converter=STF2CSV(parser.stf_path)
    converter.process()



class STF2CSV():
    """STF dataset."""
    def __init__(self, stf_dir):
        self.classes = []
        self.stf_dir = stf_dir

    def process(self):
        with open('annotations.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            
            for name in os.listdir(self.stf_dir):
                path=os.path.join(self.stf_dir, name)
                if os.path.isdir(path):
                    annotations_filename=os.path.join(path,'annotations.json')
                    with open(annotations_filename, 'r') as file:
                        annotations_file=json.load(file)
                
                    image_dir=os.path.join(path,'images')
                    for frame in annotations_file['frames']:
                        frame_number = frame['frame']
                        image_filename = os.path.join(image_dir,f"{frame_number:06}.jpg")
                        for ann_dict in frame['annotations']:
                            bbox=ann_dict['bbox']
                            track_id=ann_dict['track_id']
                            label=annotations_file['track_labels'][str(track_id)]
                            self._add_label_if_necessary(label)
                            csvwriter.writerow([image_filename,bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3],label])
        with open('classes.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            i=0
            for cls in self.classes:
                csvwriter.writerow([cls,i])
                i+=1
        


    def _add_label_if_necessary(self,label):
        if label not in self.classes:
            self.classes.append(label)



if __name__ == '__main__':
    main()
