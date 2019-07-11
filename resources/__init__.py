import os
import importlib

FID_PATH = os.path.join(os.path.dirname(__file__), 'fid.py')
DETECTOR_PATH = os.path.join(os.path.dirname(__file__), 'ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb')

tf_lpips_pkg = importlib.import_module('resources.lpips-tensorflow.lpips_tf')

def get_coco_imagenet_categories():
    with open(os.path.join(os.path.dirname(__file__), 'coco_imagenet_intersection.txt')) as f:
        category_list = [line.strip() for line in f]
    return category_list


