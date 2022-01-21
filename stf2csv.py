# Original work Copyright (c) 2022 Sky360
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

import argparse
import os
import json
import csv
import shutil
import random

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


    def _write_annotations(self,set_name, dirs):
        set_dir=os.path.join(self.out_dir, set_name)
        os.mkdir(set_dir)
        ann_file=os.path.join(set_dir,'annotations.csv')
        with open(ann_file, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            for name in dirs:
                path=os.path.join(self.stf_dir, name)
                out_segment_dir=os.path.join(self.out_dir,set_name,name)
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

    def process(self):
            
        dirs=[]

        for name in os.listdir(self.stf_dir):
            fullname=os.path.join(self.stf_dir,name)
            if os.path.isdir(fullname):
                dirs.append(name)
        random.shuffle(dirs)
        test_size=max(2,int(0.1 * len(dirs)))
        test_set=dirs[:test_size]
        dirs=dirs[test_size:]
        print(f"test size:({test_size}), set: {test_set}")
        self._write_annotations('test', test_set)

        validation_size=max(2,int(0.2 * len(dirs)))
        validation_set=dirs[:validation_size]
        dirs=dirs[validation_size:]
        print(f"validation set:{validation_set}")
        self._write_annotations('validation', validation_set)

        print(f"train set:{dirs}")
        self._write_annotations('train', dirs)

            
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
