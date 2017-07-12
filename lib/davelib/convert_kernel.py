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
from tensorflow.python.framework import ops


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

def plot_filters(imgs, n):
    plt.figure(1, figsize=(15,10))
    n_columns = n
    n_rows = 2
    for i in range(n):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('H(V) ' + str(i))
        plt.imshow(imgs[0,i,:,:], interpolation="nearest", cmap="gray")

        plt.subplot(n_rows, n_columns, n_columns + i+1)
        plt.title('W ' + str(i))
        plt.imshow(imgs[1,i,:,:], interpolation="nearest", cmap="gray")

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
  
  def __init__(self, K, base_net, sess, saved_model_path, comp_weights_dict, K_by_layer_dict):
    self._K = K
    self._base_net = base_net
    self._comp_weights_dict = comp_weights_dict
    self._K_by_layer_dict = K_by_layer_dict
    self._sess = sess
    self._saved_model_path = saved_model_path

    self._net_sep = resnetv1_sep(self._K, batch_size=1, num_layers=101, 
                        comp_weights_dict=comp_weights_dict, K_by_layer_dict=K_by_layer_dict)
    self._net_sep.create_architecture(self._sess, "TEST", 21,
                          tag='default', anchor_scales=[8, 16, 32])
#     show_all_variables(self._net_sep.get_scope())

    self.assign_trained_weights_to_unchanged_layers()
    self.assign_trained_weights_to_separable_layers()  

    
  def assign_trained_weights_to_unchanged_layers(self):
    with tf.variable_scope(self._net_sep.get_scope(), reuse=True):
      restore_var_dict = {}

      for v in tf.global_variables():
        name_full = v.op.name
        if self._base_net.get_scope() not in name_full:
          continue   #ignore resnet_v1_sep_101
        layer_name = LayerName(name_full, 'net_layer_weights')
        if layer_name in self._comp_weights_dict.keys():
          continue
                
        restore_var_dict[name_full] = tf.get_variable(layer_name.layer_weights()) 

    saver = tf.train.Saver(restore_var_dict)
    saver.restore(self._sess, self._saved_model_path)
    
  
  def assign_trained_weights_to_separable_layers(self):
  
    all_ops = []
    with tf.variable_scope(self._net_sep.get_scope(), reuse=True) as sc:
      for layer_name, source_weights in self._comp_weights_dict.iteritems():
        layer1_name = LayerName(layer_name.replace('conv', 'convsep')+'/weights','layer_weights')
        dest_weights_1 = tf.get_variable(layer1_name.layer_weights())
        dest_weights_2 = tf.get_variable(layer_name.layer_weights())
        K = self._K_by_layer_dict[layer_name]
        ops = self.get_assign_ops(source_weights, dest_weights_1, dest_weights_2, K)
        all_ops.extend(ops)
          
      self._sess.run(all_ops)

  def get_assign_ops(self, source_weights, dest_weights_1, dest_weights_2, K):
    f_norms = np.empty(self._K+1)
    V, H = self.get_low_rank_filters(source_weights, K)
    f_norms[K], plots = check_reconstruction_error(V, H, source_weights, K)
    
#     for k in range(self._K,self._K+1):
#       V, H = self.get_low_rank_filters(source_weights, k)
#       f_norms[k], plots = check_reconstruction_error(V, H, source_weights, k)
  #     plot_filters(plots, 8)
        
    V = np.moveaxis(V, source=2, destination=0)
  #   V = np.expand_dims(V, axis=0)
    V = np.expand_dims(V, axis=1)
    assign_op_1 = tf.assign(dest_weights_1, V)
  
    H = np.swapaxes(H, axis1=2, axis2=0)
  #   H = np.expand_dims(H, axis=1)
    H = np.expand_dims(H, axis=0)
    assign_op_2 = tf.assign(dest_weights_2, H)
    
    return [assign_op_1, assign_op_2]


  def compare_outputs(self, blobs, sess, compressed_layers):
    base_outputs = self._base_net.get_outputs(blobs, compressed_layers, sess)
    sep_outputs = self._net_sep.get_outputs(blobs, compressed_layers, sess)

    for name in base_outputs.keys():
      base_output = base_outputs[name]
      sep_output = sep_outputs[name]
    
      self.plot_outputs(base_output, sep_output)
      diff = np.subtract(base_output, sep_output)
      
      base_output_mean = np.mean(np.absolute(base_output)), 
      diff_mean_abs = np.mean(np.absolute(diff))
      diff_stdev_abs = np.std(np.absolute(diff)),
      diff_max_abs = np.max(np.absolute(diff))
      
      print('conv1 mean=', base_output_mean, ' mean=', diff_mean_abs, 
            ' stdev=', diff_stdev_abs, ' max=', diff_max_abs)
    
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

  def plot_outputs(self, units, units_sep):
      filters = 4
      fig = plt.figure(figsize=(15,8))
      n_columns = filters+1
      n_rows = 3
      
      a = plt.subplot(n_rows, n_columns, 1)
      a.text(0.5, 0.5, 'Base Model', fontsize=16, horizontalalignment='center', verticalalignment='center')
      a = plt.subplot(n_rows, n_columns, n_columns + 1)
      a.text(0.5, 0.5, 'Compressed Model\nK=' + str(self._K), fontsize=16,horizontalalignment='center', verticalalignment='center')
      a = plt.subplot(n_rows, n_columns, n_columns*2 + 1)
      a.text(0.5, 0.5, 'Reconstruction\nError', fontsize=16,horizontalalignment='center', verticalalignment='center')
      
      for i in range(filters):
          combined_data = np.array([units[0,:,:,i],units_sep[0,:,:,i]])
          _min, _max = np.amin(combined_data), np.amax(combined_data)

          a = plt.subplot(n_rows, n_columns, i+2)
          plt.title('Channel ' + str(i+1), fontsize=16)
          plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray",vmin=_min, vmax=_max)
          
          a = plt.subplot(n_rows, n_columns, n_columns + i+2)
          plt.imshow(units_sep[0,:,:,i], interpolation="nearest", cmap="gray",vmin=_min, vmax=_max)
    
          a = plt.subplot(n_rows, n_columns, n_columns*2 + i+2)
          diff = np.subtract(units[0,:,:,i], units_sep[0,:,:,i])
          plt.imshow(diff, interpolation="nearest", cmap="gray",vmin=_min, vmax=_max)
    
      axes = fig.get_axes()
      for ax in axes:
        ax.axis('off')
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
  
def compare_variable_count(base_net, net_sep, compressed_layers):
  
    base_total = 0
    sep_total = 0
    print_details = True
    for layer_name in compressed_layers:
      name = layer_name.net_layer_weights(base_net.get_scope())
      base_count = show_all_variables(print_details, name)

      name = layer_name.net_layer_weights(net_sep.get_scope())
      name_sep = layer_name.sep_version().net_layer_weights(net_sep.get_scope())
      sep_count = show_all_variables(print_details, name, name_sep)
      base_total += base_count
      sep_total += sep_count
      
      print layer_name + ': '+str(base_count)+'-'+ str(sep_count)+' = '+str(base_count - sep_count)
      
    
#     total_count = 0
#     for idx, op in enumerate(tf.global_variables()):
#       for name_filter in args:
#         if name_filter not in op.name:
#           continue
#         shape = op.get_shape()
#         count = np.prod(shape)
#         print("[%2d] %s %s = %s" % (idx, op.name, shape, count))
#         total_count += int(count)
#     print("[Total] variable size: %s" % "{:,}".format(total_count))
#     return total_count

    print 'separable net has ' + str(base_total - sep_total) + ' fewer variables'
    
def calc_reconstruction_errors(base_net, sess, saved_model_path):
#     show_all_variables('resnet_v1_101/')

    compressed_layers = [
        LayerName('conv1/weights'),
        LayerName('block1/unit_1/bottleneck_v1/conv2/weights'),
        LayerName('block1/unit_2/bottleneck_v1/conv2/weights', 'layer_weights'),
        LayerName('block1/unit_3/bottleneck_v1/conv2/weights', 'layer_weights'),
                        ]

    K_by_layer = [2,
                  2,
                  2,
                  2]

    comp_weights_dict = {}
    K_by_layer_dict = {}
    with sess.as_default():
      with tf.variable_scope(base_net._resnet_scope, reuse=True):
        for i, layer_name in enumerate(compressed_layers):
          weights = tf.get_variable(layer_name.layer_weights())
          comp_weights_dict[layer_name] = weights.eval()
          K_by_layer_dict[layer_name] = K_by_layer[i]
      
#     base_outputs = base_net.test_image_2(sess, blobs['data'], blobs['im_info'], compressed_layers)

    Kmax = 21 # C x d is max possible value
    diff_means = np.empty(Kmax)
#     Ks = [1,5,10,15,21]
    Ks = [21]
    for k in Ks:
#     for k in range(1,Kmax+1):
      sep_net = SeparableNet(k, base_net, sess, saved_model_path, comp_weights_dict=comp_weights_dict,
                             K_by_layer_dict=K_by_layer_dict)
      
      compare_variable_count(base_net, sep_net._net_sep, compressed_layers)
      base_output_mean, diff_mean_abs, diff_stdev_abs, diff_max_abs = \
        sep_net.compare_outputs(get_blobs(), sess, compressed_layers)

#       base_output_mean, diff_mean_abs, diff_stdev_abs, diff_max_abs = \
#               sep_net.run_inference(blobs, base_outputs, compressed_layers)
      diff_means[k-1] = diff_mean_abs

    exit()

    #do the plotting      
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(range(1,Kmax+1),diff_means,'ro-')
    plt.title('Reconstruction Error - conv1')
    plt.ylabel('mean abs error')
    plt.xlabel('K - rank of approximation')
    plt.show()  
    

def view(base_net, sess, saved_model_path):
  
    calc_reconstruction_errors(base_net, sess, saved_model_path)
    exit()
#     show_all_variables('resnet_v1_101/conv1')

