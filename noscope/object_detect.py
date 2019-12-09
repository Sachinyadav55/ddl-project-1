#!/usr/bin/env python
import numpy as np
import os
import sys
import tensorflow as tf
import pandas as pd
import time
import shutil
import math
import logging
import glob
from utils import label_map_util
from utils import visualization_utils as vis_util

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm
import argparse




# Supress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)


# This is needed since the notebook is stored in the object_detection folder.

sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')


root_dir = '' # select the root directory, it should have both label map as well the as the frozen checkpoint in it
PATH_TO_FROZEN_GRAPH = os.path.join(root_dir, '/frozen_inference_graph.pb')

PATH_TO_LABELS = os.path.join(root_dir, 'label_map.pbtxt')



parser = argparse.ArgumentParser(description='This script is for inferencing videos to create dataset for training shallow neural nets')

parser.add_argument('-M', action='store', dest='model',default = PATH_TO_FROZEN_GRAPH)
parser.add_argument('-L', action='store', dest = 'parts_label',default = PATH_TO_LABELS)
parser.add_argument('-V' ,action='store', dest = 'video')
parser.add_argument('-A',action ='store_true',dest= 'analyze_all', default = False )
parser.add_argument('-GPU_to_use', action = 'store',dest = 'gpus_2_use', default = "0")

results = parser.parse_args()

print("Using GPU:" + results.gpus_2_use)


video = results.video

# Script to run the inference, reodered the nesting to avoid repeated allocation
# of the same network on to the GPU which saves at least 3 seconds
def run_inference_for_single_image(image, graph):

    # Input:  takes the image (np.ndarray) and the tf_graph object containing the
    #         neural network graph
    # Output: Returns a dictionary with bouding boxes, their scores and rankings.
    #         also give the number of predictions above the threshold
  with graph.as_default():
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[1], image.shape[2])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.9), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: image})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.int64)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


# Reading the serializing the models

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(results.model, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# Setting which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = results.gpus_2_use
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
category_index = label_map_util.create_category_index_from_labelmap(results.parts_label, use_display_name=True)

# loading the model on to the GPU
with tf.Session(graph=detection_graph,config=config) as sess:
    print("Processing " + video + " now.\n")

    # Reading the video from the directory
    cap = cv2.VideoCapture(video)
    nframes = int(cap.get(7))

    # getting ready to pick 10,000 random frames
    np.random.seed(0)
    ind = list(np.random.permutation(np.arange(nframes))[:10000])
    np.random.seed(0)
    ind_rem = list(np.random.permutation(np.arange(nframes))[10000:])


    # Creating directories for the test and validation datasets
    if os.path.isdir(video.split(".")[0]):
        shutil.rmtree(video.split(".")[0])
    os.mkdir(video.split(".")[0])

    if os.path.isdir(video.split(".")[0] + "/temp"):
        shutil.rmtree(video.split(".")[0] + "/temp")
    os.mkdir(video.split(".")[0] + "/temp")
    os.mkdir(video.split(".")[0] + "/temp/0")
    os.mkdir(video.split(".")[0] + "/temp/1")

    counter = {'frame': [],
              'count': [],
              'boxes': []}
    # in case if you want to run inference on the whole video
    if results.analyze_all:
        looper = ind_rem
    else:
        looper = ind

    # let the inference begin!
    for i in tqdm(ind):
        cap.set(1,i)

        ret,image_np = cap.read()

        if ret:
            image_np_expanded = np.expand_dims(image_np, axis=0)


          # Actual detection.
            output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
            f_num=len(output_dict['detection_scores'][output_dict['detection_scores'] > 0.9])

            # making dataset for binary classification
            if f_num == 0:
                cv2.imwrite(video.split(".")[0] + "/temp/0/"+str(i)+".png",image_np )
            else:
                cv2.imwrite(video.split(".")[0] + "/temp/1/"+str(i)+".png",image_np )


            counter['frame'].append(i)
            counter['count'].append(f_num)

    p = pd.DataFrame.from_dict(counter).sort_values(by='frame')

    p.to_hdf(video.split(".")[0]+"h5", key = 'p',mode='w')
