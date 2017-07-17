'''
Created on 2 Jul 2017

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

from model.config import cfg
from resnet_v1_sep import resnetv1_sep
from model.test import _get_blobs
from model.test import im_detect
from matplotlib.ticker import MaxNLocator
from layer_name import LayerName
from compression_stats import CompressionStats
from tensorflow.python.framework import ops
from collections import OrderedDict
from copy_elements import copy_variable_to_graph
from copy_elements import copy_op_to_graph
from operator import itemgetter

sys.path.append('/home/david/Project/tf-faster-rcnn_27/tools')
from demo import *

figure_path = '/home/david/host/figures'

def show_all_variables(show, *args):
  total_count = 0
  for idx, op in enumerate(tf.global_variables()):
    for name_filter in args:
      if name_filter not in op.name:
        continue
      shape = op.get_shape()
      count = np.prod(shape)
      if show:
        print("[%2d] %s %s = %s" % (idx, op.name, shape, count))
      total_count += int(count)
  if show:
    print("[Total] variable size: %s" % "{:,}".format(total_count))
  return total_count

def plot_filters(plots_dict, n):
  fig = plt.figure(figsize=(15,8))
  n_columns = n + 1
  n_rows = len(plots_dict) + 1
  
  a = plt.subplot(n_rows, n_columns, 1)
  a.text(0.75, 0.5, 'Base Model\nFilter', fontsize=16, horizontalalignment='center', verticalalignment='center')
  imgs = plots_dict[ list(plots_dict.keys())[0] ]
  for i in range(n):
    plt.subplot(n_rows, n_columns, i+2)
    plt.title('channel ' + str(i+1), fontsize=16)
    plt.imshow(imgs[1,i,:,:], interpolation="nearest", cmap="gray")
  
  for j, (k, imgs) in enumerate(plots_dict.iteritems()):
    a = plt.subplot(n_rows, n_columns, (j+1)*n_columns + 1)
    a.text(0.75, 0.5, 'Low-rank\nFilter\nK=' + str(k), fontsize=16,horizontalalignment='center', verticalalignment='center')
    for i in range(n):
      plt.subplot(n_rows, n_columns, (j+1)*n_columns+i+2)
      plt.imshow(imgs[0,i,:,:], interpolation="nearest", cmap="gray")

  axes = fig.get_axes()
  for ax in axes:
    ax.axis('off')
  plt.tight_layout()
  plt.show()

def check_reconstruction_error(V, H, W, K):
  s = W.shape
  assert s[0] == s[1]
  C = s[2]
  d = s[0]
  N = s[3]
  
  f_norm_sum = 0
  plots = np.zeros((2,N,d,d))
  for n in range(N):
    for c in range(C):
      
      sum_prod = np.zeros((d,d))
      for k in range(K):
        p = np.outer(V[c,k,:], H[n,k,:])
        sum_prod += p
      diff = np.subtract(W[:,:,c,n], sum_prod)
      f_norm = np.linalg.norm(diff, ord='fro')
      f_norm_sum += f_norm
      plots[0,n,:,:] = sum_prod
      plots[1,n,:,:] = W[:,:,c,n]
  
  return f_norm_sum, plots
  
class SeparableNet(object):
  
  def __init__(self, scope_idx, base_net, sess, saved_model_path, comp_weights_dict, 
               K_by_layer_dict, var_count_dict, base_variables):

    self._base_net = base_net
    self._comp_weights_dict = comp_weights_dict
    self._K_by_layer_dict = K_by_layer_dict
    self._sess = sess
    self._saved_model_path = saved_model_path
    self._base_variables = base_variables

    self._net_sep = resnetv1_sep(scope_idx, batch_size=1, num_layers=101, 
                        comp_weights_dict=comp_weights_dict, K_by_layer_dict=K_by_layer_dict)
#     show_all_variables(True, self._net_sep.get_scope())
    self._net_sep.create_architecture(self._sess, "TEST", 21,
                          tag='default', anchor_scales=[8, 16, 32])
#     init = tf.global_variables_initializer()

    self._reduced_var_count = self.calc_variable_count(var_count_dict)
    self.assign_trained_weights_to_unchanged_layers()
    self.assign_trained_weights_to_separable_layers()  
    
  def get_reduced_var_count(self):
    return self._reduced_var_count 
  
  def calc_variable_count(self, var_count_dict, print_details = False):
    base_total = 0
    sep_total = 0
    
    for layer_name in self._comp_weights_dict:
#       name = layer_name.net_layer_weights(self._base_net.get_scope())
#       base_count = show_all_variables(print_details, name)
      base_count = var_count_dict[layer_name]

      name = layer_name.net_layer_weights(self._net_sep.get_scope())
      name_sep = layer_name.sep_version().net_layer_weights(self._net_sep.get_scope())
      sep_count = show_all_variables(print_details, name, name_sep)
      base_total += base_count
      sep_total += sep_count
      if print_details:
        print layer_name + ': '+str(base_count)+'-'+ str(sep_count)+' = '+str(base_count - sep_count)
      
    if print_details:
      print 'separable net has ' + str(base_total - sep_total) + ' fewer variables'
    
    return base_total - sep_total

    
  def assign_trained_weights_to_unchanged_layers(self):
    with tf.variable_scope(self._net_sep.get_scope(), reuse=True):
      restore_var_dict = {}

      for v in self._base_variables:
        name_full = v.op.name
#         if self._base_net.get_scope() not in name_full:
#           continue   #ignore resnet_v1_sep_101
        layer_name = LayerName(name_full, 'net_layer_weights')
        if layer_name in self._comp_weights_dict.keys():
          continue
                
        restore_var_dict[name_full] = tf.get_variable(layer_name.layer_weights()) 

    saver = tf.train.Saver(restore_var_dict)
    saver.restore(self._sess, self._saved_model_path)
    
  
  def assign_trained_weights_to_separable_layers(self):
  
    all_ops = []
    with tf.variable_scope(self._net_sep.get_scope(), reuse=True):
      for layer_name, source_weights in self._comp_weights_dict.iteritems():
        layer1_name = LayerName(layer_name.replace('conv', 'convsep')+'/weights','layer_weights')
        dest_weights_1 = tf.get_variable(layer1_name.layer_weights())
        dest_weights_2 = tf.get_variable(layer_name.layer_weights())
        K = self._K_by_layer_dict[layer_name]
        ops = self.get_assign_ops(source_weights, dest_weights_1, dest_weights_2, K)
        all_ops.extend(ops)
          
      self._sess.run(all_ops)

  def get_assign_ops(self, source_weights, dest_weights_1, dest_weights_2, K, plot=False):
    
    if plot :
      f_norms = np.empty(K+1)
      plots_dict = OrderedDict()
      Ks = [21,5,2,1]
      for k in Ks:
        V, H = self.get_low_rank_filters(source_weights, k)
        f_norms[k], plots = check_reconstruction_error(V, H, source_weights, k)
        plots_dict[k] = plots
      plot_filters(plots_dict, 4)
    else:
      V, H = self.get_low_rank_filters(source_weights, K)
        
    V = np.moveaxis(V, source=2, destination=0)
  #   V = np.expand_dims(V, axis=0)
    V = np.expand_dims(V, axis=1)
    assign_op_1 = tf.assign(dest_weights_1, V)
  
    H = np.swapaxes(H, axis1=2, axis2=0)
  #   H = np.expand_dims(H, axis=1)
    H = np.expand_dims(H, axis=0)
    assign_op_2 = tf.assign(dest_weights_2, H)
    
    return [assign_op_1, assign_op_2]


  def compare_outputs(self, blobs, sess, base_outputs, compressed_layers=[], 
                      plot=False):
  #     final_output = LayerName('block4/unit_3/bottleneck_v1/conv3')
    final_output = LayerName('block4/unit_3/bottleneck_v1/conv3/BatchNorm')

    compressed_layers.append(final_output)

    if base_outputs is None:
      base_outputs = self._base_net.get_outputs(blobs, compressed_layers, sess)
    sep_outputs = self._net_sep.get_outputs(blobs, compressed_layers, sess)

    for name in compressed_layers:
#     for name in base_outputs.keys():
      base_output = base_outputs[name]
      sep_output = sep_outputs[name]
    
      if plot:
        self.plot_outputs(base_output, sep_output, name)
      diff = np.subtract(base_output, sep_output)
      
      base_output_mean = np.mean(np.absolute(base_output)) 
      diff_mean_abs = np.mean(np.absolute(diff))
      diff_stdev_abs = np.std(np.absolute(diff))
      diff_max_abs = np.max(np.absolute(diff))
      
#       print('base mean=', base_output_mean, ' diff mean=', diff_mean_abs, 
#             ' stdev=', diff_stdev_abs, ' max=', diff_max_abs)
    
    return base_output_mean, diff_mean_abs, diff_stdev_abs, diff_max_abs
  
  def run_inference(self, blobs, base_outputs, compressed_layers):
  
    outputs = self._net_sep.test_image_2(self._sess, blobs['data'], blobs['im_info'], 
                                         compressed_layers)
#     conv1_1_sep, conv1_2_sep = self._net_sep.test_image_2(self._sess, blobs['data'], 
#                                                           blobs['im_info'], compressed_layers)
    
    for layer_name in compressed_layers:
#     for key, output in outputs.iteritems():
#       name = name.replace('/weights','')
      base_output = base_outputs[layer_name.net_layer(self._base_net.get_scope())]
#       name = name.replace('resnet_v1_101','resnet_v1_sep_101')
      sep_output = outputs[layer_name.net_layer(self._net_sep.get_scope())]
      
      self.plot_outputs(base_output, sep_output)
      diff = np.subtract(base_output, sep_output)
      
      base_output_mean = np.mean(np.absolute(base_output)), 
      diff_mean_abs = np.mean(np.absolute(diff))
      diff_stdev_abs = np.std(np.absolute(diff)),
      diff_max_abs = np.max(np.absolute(diff))
      
      print('conv1 mean=', base_output_mean, ' mean=', diff_mean_abs, 
            ' stdev=', diff_stdev_abs, ' max=', diff_max_abs)
    
    return base_output_mean, diff_mean_abs, diff_stdev_abs, diff_max_abs
    
  def tensor_to_matrix(self, W_arr):
    # convert 4-D tensor to 2-D using bijection from Tai et al 2016
    s = W_arr.shape
    assert s[0] == s[1]
    C = s[2]
    d = s[0]
    N = s[3]
    W = np.empty([C*d, d*N])
    
    for i1 in range(1,C+1):
      for i2 in range(1,d+1):
        for i3 in range(1,d+1):
          for i4 in range(1,N+1):
            j1 = (i1-1)*d + i2
            j2 = (i4-1)*d + i3
            W[j1-1, j2-1] = W_arr[i2-1,i3-1,i1-1,i4-1] #subtract 1 to adjust for zero based arrays
  
    return C,d,N,W  
    
  def get_low_rank_filters(self, weights, K):
    C,d,N,W = self.tensor_to_matrix(weights) # W=Cd x Nd
    U,D,Qt = np.linalg.svd(W) # U=Cd x Cd, Q=Nd x Nd, D=Cd, 
    Q = np.transpose(Qt)
    
    V = np.empty([C,K,d])
    H = np.empty([N,K,d])
    
    for k in range(K):
      for j in range(d):
        for c in range(C):
          V[c,k,j] = U[c*d + j, k] * np.sqrt(D[k])
        for n in range(N):
          H[n,k,j] = Q[n*d + j, k] * np.sqrt(D[k])
  
    return V, H 

  def plot_outputs(self, units, units_sep, name):
      filters = 4
      fig = plt.figure(figsize=(15,8))
      n_columns = filters+1
      n_rows = 1
      
      if name in self._K_by_layer_dict.keys():
        K = self._K_by_layer_dict[name]
      else:
        K = 0
      
#       a = plt.subplot(n_rows, n_columns, 1)
#       a.text(0.75, 0.5, 'Base\nModel', fontsize=16, horizontalalignment='center', verticalalignment='center')
      a = plt.subplot(n_rows, n_columns, 1)
#       a = plt.subplot(n_rows, n_columns, n_columns + 1)
      a.text(0.75, 0.5, 'Compressed\nModel\nK=' + str(K), fontsize=16,horizontalalignment='center', verticalalignment='center')
#       a = plt.subplot(n_rows, n_columns, n_columns*2 + 1)
#       a.text(0.5, 0.5, 'Reconstruction\nError', fontsize=16,horizontalalignment='center', verticalalignment='center')
      
      for i in range(filters):
          combined_data = np.array([units[0,:,:,i],units_sep[0,:,:,i]])
          _min, _max = np.amin(combined_data), np.amax(combined_data)

#           a = plt.subplot(n_rows, n_columns, i+2)
#           plt.title('Channel ' + str(i+1), fontsize=16)
#           plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray",vmin=_min, vmax=_max)
          
          a = plt.subplot(n_rows, n_columns, i+2)
#           a = plt.subplot(n_rows, n_columns, n_columns + i+2)
          plt.imshow(units_sep[0,:,:,i], interpolation="nearest", cmap="gray",vmin=_min, vmax=_max)
    
#           a = plt.subplot(n_rows, n_columns, n_columns*2 + i+2)
#           diff = np.subtract(units[0,:,:,i], units_sep[0,:,:,i])
#           plt.imshow(diff, interpolation="nearest", cmap="gray",vmin=_min, vmax=_max)
    
      axes = fig.get_axes()
      for ax in axes:
        ax.axis('off')
#       fig.suptitle('Layer: ' + name, fontsize=18, x=0.5, y=0.02, horizontalalignment='center', verticalalignment='center')
      plt.tight_layout()
      plt.show()
#       plt.savefig(figure_path+'/output_imgs_K'+str(self._K)+'.png')
  
def get_blobs():
    im_names = ['000456.jpg', '000542.jpg', '001150.jpg', '001763.jpg', '004545.jpg']
    im_file = os.path.join(cfg.DATA_DIR, 'demo', im_names[0])
    im = cv2.imread(im_file)
    blobs, im_scales = _get_blobs(im)
    assert len(im_scales) == 1, "Only single-image batch implemented"
    im_blob = blobs['data']
    blobs['im_info'] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
    return blobs
        
def get_layer_names():    
    
    layers = []
    layers.append( LayerName('conv1/weights') )
    
    d = {'1':3, '2':4, '3':23, '4':3}
    block_layer_dict = OrderedDict(sorted(d.items(), key=lambda t: t[0]))

    for block_num, num_layers in block_layer_dict.iteritems():
      for unit_num in range(1,num_layers+1):
        layers.append( LayerName('block'+block_num+'/unit_'+str(unit_num)+
                                 '/bottleneck_v1/conv2/weights') )
      
    return layers
        
def calc_reconstruction_errors(base_net, sess, saved_model_path, tfconfig):
#     show_all_variables(True, 'resnet_v1_101/')

    compressed_layers = [
        LayerName('conv1/weights'),
        LayerName('block1/unit_1/bottleneck_v1/conv2/weights'),
        LayerName('block1/unit_2/bottleneck_v1/conv2/weights'),
        LayerName('block1/unit_3/bottleneck_v1/conv2/weights'),
        LayerName('block2/unit_1/bottleneck_v1/conv2/weights'),
        LayerName('block2/unit_2/bottleneck_v1/conv2/weights'),
        LayerName('block2/unit_3/bottleneck_v1/conv2/weights'),
        LayerName('block2/unit_4/bottleneck_v1/conv2/weights'),
        LayerName('block3/unit_1/bottleneck_v1/conv2/weights'),
                        ]

    layers_names = get_layer_names()

    K_by_layer = [1,
                  21,
                  21,
                  21]
    
    blobs = get_blobs()
    final_output = LayerName('block4/unit_3/bottleneck_v1/conv3/BatchNorm')

    base_outputs = base_net.get_outputs(blobs, [final_output], sess)
    
    base_variables = tf.global_variables()
    default_graph = tf.get_default_graph()
    
    var_count_dict = OrderedDict()
    for layer_name in layers_names:
      name = layer_name.net_layer_weights(base_net.get_scope())
      var_count_dict[layer_name] = show_all_variables(False, name)
  
    all_comp_weights_dict = {}
    K_by_layer_dict = {}
    with sess.as_default():
      with tf.variable_scope(base_net._resnet_scope, reuse=True):
        for layer_name in layers_names:
          weights = tf.get_variable(layer_name.layer_weights())
          all_comp_weights_dict[layer_name] = weights.eval()

#     base_net_graph = tf.get_default_graph() # Save the old graph for later iteration
#     sep_net_graph = tf.Graph() # Create an empty graph

    Ks = range(1,11)
#     Ks = [400]
    Ks = [1, 250, 768]
#     Ks = [1, 10, 96, 150, 192]
#     Ks = [1, 5, 15, 21]
#     stats = CompressionStats(filename='CompressionStats.pi')
#     stats.plot()
    layer_idxs = [10,20,31,33]
#     K_by_layer_dict = {}

    stats = CompressionStats(['base_mean','diff_mean','diff_stdev','diff_max','var_redux'],
                             itemgetter(*layer_idxs)(layers_names), Ks, filename_suffix='31,33')

#     with sep_net_graph.as_default():  
    sess.close()
    scope_idx=1
    with default_graph.as_default():
      for l, layer_name in enumerate(layers_names):
        if l not in layer_idxs:
          continue
        sess = tf.Session(config=tfconfig)
        for k in Ks:

          comp_weights_dict = { layer_name: all_comp_weights_dict[layer_name] }
          K_by_layer_dict[layer_name] = k
          
          sep_net = SeparableNet(scope_idx, base_net, sess, saved_model_path, comp_weights_dict,\
                                 K_by_layer_dict, var_count_dict, base_variables)
          
          base_mean, diff_mean, diff_stdev, diff_max = \
            sep_net.compare_outputs(blobs, sess, base_outputs)
    
          stats.set('base_mean', layer_name, k, base_mean)
          stats.set('diff_mean', layer_name, k, diff_mean)
          stats.set('diff_stdev', layer_name, k, diff_stdev)
          stats.set('diff_max', layer_name, k, diff_max)
          stats.set('var_redux', layer_name, k, sep_net.get_reduced_var_count())
          stats.save()
          print layer_name + ' K=' + str(k) + ' complete'
  
  #         show_all_variables(True, sep_net._net_sep.get_scope())
          scope_idx += 1
#         tf.reset_default_graph()
        sess.close()
  #         show_all_variables(True, sep_net._net_sep.get_scope())
  
  
    exit()

    #do the plotting      
#     fig, ax = plt.subplots()
#     ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#     plt.plot(range(1,Kmax+1),diff_means,'ro-')
#     plt.title('Reconstruction Error - conv1')
#     plt.ylabel('mean abs error')
#     plt.xlabel('K - rank of approximation')
#     plt.show()  
    

def view(base_net, sess, saved_model_path, tfconfig):
  
    calc_reconstruction_errors(base_net, sess, saved_model_path, tfconfig)
    exit()
#     show_all_variables('resnet_v1_101/conv1')

