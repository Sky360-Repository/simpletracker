import argparse
import os
import json
import csv
import shutil

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--stf_path', help='Path to STF directory')
    parser.add_argument('--out_path', help='Path to output directory')
    parser = parser.parse_args(args)

    if parser.stf_path is None:
            raise ValueError('Must provide --stf_path when training on STF,')
    if parser.out_path is None:
            raise ValueError('Must provide --out_path when training on STF,')
    converter=STF2CSV(parser.stf_path,parser.out_path)
    converter.process()



class STF2CSV():
    """STF dataset."""
    def __init__(self, stf_dir, out_dir):
        self.classes = []
        self.stf_dir = stf_dir
        self.out_dir = out_dir
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

    def process(self):
        ann_file=os.path.join(self.out_dir,'annotations.csv')
        with open(ann_file, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            
            for name in os.listdir(self.stf_dir):
                path=os.path.join(self.stf_dir, name)
                out_segment_dir=os.path.join(self.out_dir,name)
                os.mkdir(out_segment_dir)
                if os.path.isdir(path):
                    annotations_filename=os.path.join(path,'annotations.json')
                    with open(annotations_filename, 'r') as file:
                        annotations_file=json.load(file)
                
                    image_dir=os.path.join(path,'images')
                    for frame in annotations_file['frames']:
                        frame_number = frame['frame']
                        file=f"{frame_number:06}.png"
                        image_filename = os.path.join(image_dir,file)
                        target_filename=os.path.join(out_segment_dir,file)
                        if os.path.exists(image_filename):
                            shutil.copy(image_filename,target_filename)
                        for ann_dict in frame['annotations']:
                            bbox=ann_dict['bbox']
                            track_id=ann_dict['track_id']
                            label=annotations_file['track_labels'][str(track_id)]
                            self._add_label_if_necessary(label)
                            csvwriter.writerow([target_filename,bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3],label])
        classes_file=os.path.join(self.out_dir,'classes.csv')
        with open(classes_file, 'w', newline='') as csvfile:
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
