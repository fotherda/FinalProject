'''
Created on 28 Jul 2017

@author: david
'''

import numpy as np
import os, cv2, sys
import argparse
import matplotlib as mp
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import math
import os.path as osp


from tensorflow.python.platform import gfile
from model.config import cfg
from resnet_v1_sep import resnetv1_sep
from model.test import _get_blobs
from model.test import im_detect
from matplotlib.ticker import MaxNLocator
from layer_name import LayerName
from compression_stats import CompressionStats, CompressedNetDescription
from tensorflow.python.framework import ops
from collections import OrderedDict
from copy_elements import copy_variable_to_graph
from copy_elements import copy_op_to_graph


PARAM_OPTIONS = {
    'max_depth': 10000,
    'min_bytes': 0,  # Only >=1
    'min_micros': 0,  # Only >=1
    'min_params': 1,
    'min_float_ops': 0,
    'device_regexes': ['.*'],
    'order_by': 'params',
    'account_type_regexes': ['.*'],
    'start_name_regexes': ['.*'],
    'trim_name_regexes': [],
    'show_name_regexes': ['.*'],
    'hide_name_regexes': [],
    'account_displayed_op_only': True,
    'select': ['bytes','params','op_types','tensor_value'],
    'viz': False,
    'dump_to_file': ''
}
PERF_OPTIONS = {
    'max_depth': 10000,
    'min_bytes': 0,  # Only >=1
    'min_micros': 0,  # Only >=1
    'min_params': 0,
    'min_float_ops': 1,
    'device_regexes': ['.*'],
    'order_by': 'micros',
    'account_type_regexes': ['.*'],
    'start_name_regexes': ['.*'],
    'trim_name_regexes': [],
    'show_name_regexes': ['.*'],
    'hide_name_regexes': [],
    'account_displayed_op_only': True,
    'select': ['float_ops','micros','bytes','op_types','tensor_value'],
    'viz': False,
    'dump_to_file': ''
}

class ProfileStats(object):
  '''
  holds profiling information for a network
  '''

  def __init__(self, net_desc, run_metadata):
    self._net_desc = net_desc
    self._run_metadata = run_metadata
    
    
  def extract_data(self):
#     tf.contrib.tfprof.model_analyzer.print_model_analysis(
#         tf.get_default_graph(),
#         run_meta=self._run_metadata,
#         tfprof_options=tf.contrib.tfprof.model_analyzer.PRINT_ALL_TIMING_MEMORY)
        
    stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        run_meta=self._run_metadata,
        tfprof_options=PARAM_OPTIONS)
    
    stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        run_meta=self._run_metadata,
        tfprof_options=PERF_OPTIONS)
    
  #   res = list(stats.DESCRIPTOR.fields_by_name.keys())
    fields = ['name', 'tensor_value', 'exec_micros', 'requested_bytes', 'parameters', 
     'float_ops', 'inputs', 'device', 'total_exec_micros', 'total_requested_bytes', 
     'total_parameters', 'total_float_ops', 'total_inputs', 'shapes', 'children']
  
    
#     for k, v in stats.ListFields():
#       print(k.camelcase_name + ': ' + str(v))
#   #     value = stats.name
#   #     value = stats.total_parameters
#   #     value = stats.float_ops
#   
#     for child in stats.children:
#       print('\n')
#       for k, v in child.ListFields():
#         print(k.camelcase_name + ': ' + str(v))
#            
  
