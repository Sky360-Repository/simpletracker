
import argparse
import os
import json
import csv
import shutil
import random
import xml.etree.ElementTree as ET  # elementpath
from xml.dom import minidom


def main(args=None):
    parser = argparse.ArgumentParser(description='Conver STF format to VOC')

    parser.add_argument('--stf_path', help='Path to STF directory')
    parser.add_argument('--out_path', help='Path to output directory')
    parser = parser.parse_args(args)

    if parser.stf_path is None:
        raise ValueError('Must provide --stf_path')
    if parser.out_path is None:
        raise ValueError('Must provide --out_path')
    converter = STF2VOC(parser.stf_path, parser.out_path)
    converter.process()


def get_or_create(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path


class STF2VOC():
    """STF dataset."""

    def __init__(self, stf_dir, out_dir):
        self.classes = []
        self.stf_dir = stf_dir
        self.out_dir = out_dir
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

    def _write_annotations(self, set_name, dirs):
        # mkdir <out>/[train|test]
        set_dir = os.path.join(self.out_dir, set_name)
        os.mkdir(set_dir)

        out_annotations_dir = get_or_create(
            os.path.join(set_dir, 'annotations'))
        out_images_dir = get_or_create(os.path.join(set_dir, 'images'))
        out_optical_flow_dir = get_or_create(
            os.path.join(set_dir, 'optical_flow'))
        image_count = 100000
        # Loop over segment dirs
        for name in dirs:
            in_segment_path = os.path.join(self.stf_dir, name)
            if os.path.isdir(in_segment_path):
                in_annotations_filename = os.path.join(
                    in_segment_path, 'annotations.json')
                with open(in_annotations_filename, 'r') as file:
                    in_annotations = json.load(file)

                in_image_path = os.path.join(in_segment_path, 'images')
                for frame in in_annotations['frames']:
                    frame_number = frame['frame']
                    in_file_base = f"{frame_number:06}"
                    out_file_base = f"{image_count:06}"

                    root = ET.Element('annotation')
                    for ann_dict in frame['annotations']:
                        bbox = ann_dict['bbox']
                        track_id = ann_dict['track_id']
                        label = in_annotations['track_labels'][str(track_id)]
                        if label == 'motion':
                            xmin, ymin, xmax, ymax = [
                                bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
                            object = ET.SubElement(root, 'object')
                            name = ET.SubElement(object, 'name')
                            name.text = 'motion'
                            bndbox = ET.SubElement(object, 'bndbox')
                            xmin_el = ET.SubElement(bndbox, 'xmin')
                            xmin_el.text = f"{xmin}"
                            ymin_el = ET.SubElement(bndbox, 'ymin')
                            ymin_el.text = f"{ymin}"
                            xmax_el = ET.SubElement(bndbox, 'xmax')
                            xmax_el.text = f"{xmax}"
                            ymax_el = ET.SubElement(bndbox, 'ymax')
                            ymax_el.text = f"{ymax}"

                    if len(root) > 0:
                        image_abosolute_filename = os.path.join(
                            in_image_path, f"{in_file_base}.original.jpg")
                        target_image_filename = os.path.join(
                            out_images_dir, f"{out_file_base}.jpg")
                        shutil.copy(image_abosolute_filename,
                                    target_image_filename)

                        optical_flow_abosolute_filename = os.path.join(
                            in_image_path, f"{in_file_base}.optical_flow.jpg")
                        target_optical_flow_filename = os.path.join(
                            out_optical_flow_dir, f"{out_file_base}.jpg")
                        shutil.copy(optical_flow_abosolute_filename,
                                    target_optical_flow_filename)

                        out_xml_file = os.path.join(
                            out_annotations_dir, f"{out_file_base}.xml")
                        xmlstr = minidom.parseString(
                            ET.tostring(root)).toprettyxml(indent="   ")
                        with open(out_xml_file, "w") as f:
                            f.write(xmlstr)

                        image_count += 1

    def process(self):

        dirs = []

        for name in os.listdir(self.stf_dir):
            fullname = os.path.join(self.stf_dir, name)
            if os.path.isdir(fullname):
                dirs.append(name)
        random.shuffle(dirs)
        test_size = max(1, int(0.1 * len(dirs)))
        test_set = dirs[:test_size]
        dirs = dirs[test_size:]
        print(f"test size:({test_size}), set: {test_set}")
        self._write_annotations('test', test_set)

        #validation_size = max(2, int(0.2 * len(dirs)))
        #validation_set = dirs[:validation_size]
        #dirs = dirs[validation_size:]
        #print(f"validation set:{validation_set}")
        #self._write_annotations('validation', validation_set)

        print(f"train set:{dirs}")
        self._write_annotations('train', dirs)

        classes_file = os.path.join(self.out_dir, 'classes.csv')
        with open(classes_file, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            i = 0
            for cls in self.classes:
                csvwriter.writerow([cls, i])
                i += 1

    def _add_label_if_necessary(self, label):
        if label not in self.classes:
            self.classes.append(label)


if __name__ == '__main__':
    main()
