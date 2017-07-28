'''
Created on 2 Jul 2017

@author: david
'''

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.layers.python.layers import utils

from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib import layers as layers_lib
from tensorflow.python.ops import array_ops
# from tensorflow.contrib.slim import arg_scope
from nets.network import Network
from tensorflow.python.framework import ops
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers
from model.config import cfg
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.layers.python.layers import utils

from nets.resnet_v1 import resnetv1
from nets.resnet_v1 import resnet_arg_scope
from layer_name import LayerName


def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
  """Strided 2-D convolution with 'SAME' padding.

  When stride > 1, then we do explicit zero-padding, followed by conv2d with
  'VALID' padding.

  Note that

     net = conv2d_same(inputs, num_outputs, 3, stride=stride)

  is equivalent to

     net = tf.contrib.layers.conv2d(inputs, num_outputs, 3, stride=1,
     padding='SAME')
     net = subsample(net, factor=stride)

  whereas

     net = tf.contrib.layers.conv2d(inputs, num_outputs, 3, stride=stride,
     padding='SAME')

  is different when the input's height or width is even, which is why we add the
  current function. For more details, see ResnetUtilsTest.testConv2DSameEven().

  Args:
    inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
    num_outputs: An integer, the number of output filters.
    kernel_size: An int with the kernel_size of the filters.
    stride: An integer, the output stride.
    rate: An integer, rate for atrous convolution.
    scope: Scope.

  Returns:
    output: A 4-D tensor of size [batch, height_out, width_out, channels] with
      the convolution output.
  """
  if stride == 1:
    return layers_lib.conv2d(
        inputs,
        num_outputs,
        kernel_size,
        stride=1,
        rate=rate,
        padding='SAME',
        scope=scope)
  else:
    pad_beg = range(2)
    pad_end = range(2)
    for i in range(2):
      kernel_size_effective = kernel_size[i] + (kernel_size[i] - 1) * (rate - 1)
      pad_total = kernel_size_effective - 1
      pad_beg[i] = pad_total // 2
      pad_end[i] = pad_total - pad_beg[i]
    inputs = array_ops.pad(
        inputs, [[0, 0], [pad_beg[0], pad_end[0]], [pad_beg[1], pad_end[1]], [0, 0]])
    return layers_lib.conv2d(
        inputs,
        num_outputs,
        kernel_size,
        stride=stride,
        rate=rate,
        padding='VALID',
        scope=scope)


class resnetv1_sep(resnetv1):
    
  def __init__(self, scope_idx, batch_size, num_layers, comp_weights_dict, K_by_layer_dict):
    resnetv1.__init__(self, batch_size, num_layers)
    self._resnet_scope = 'resnet_v1_sep%d_%d' % (scope_idx, num_layers)  
    self._comp_weights_dict = comp_weights_dict
    self._K_by_layer_dict = K_by_layer_dict
    self.bottleneck_func = self.bottleneck
    self._end_points_collection = self._resnet_scope + '_end_points'



#   @add_arg_scope
  def bottleneck(self,
                 inputs,
                 depth,
                 depth_bottleneck,
                 stride,
                 rate=1,
                 outputs_collections=None,
                 scope=None):
    """Bottleneck residual unit variant with BN after convolutions.
   
    This is the original residual unit proposed in [1]. See Fig. 1(a) of [2] for
    its definition. Note that we use here the bottleneck variant which has an
    extra bottleneck layer.
   
    When putting together two consecutive ResNet blocks that use this unit, one
    should use stride = 2 in the last unit of the first block.
   
    Args:
      inputs: A tensor of size [batch, height, width, channels].
      depth: The depth of the ResNet unit output.
      depth_bottleneck: The depth of the bottleneck layers.
      stride: The ResNet unit's stride. Determines the amount of downsampling of
        the units output compared to its input.
      rate: An integer, rate for atrous convolution.
      outputs_collections: Collection to add the ResNet unit output.
      scope: Optional variable_scope.
   
    Returns:
      The ResNet unit's output.
    """
    with variable_scope.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
      depth_in = utils.last_dimension(inputs.get_shape(), min_rank=4)
      if depth == depth_in:
        shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
      else:
        shortcut = layers.conv2d(
            inputs,
            depth, [1, 1],
            stride=stride,
            activation_fn=None,
            scope='shortcut')
   
      residual = layers.conv2d(
          inputs, depth_bottleneck, [1, 1], stride=1, scope='conv1')
       
      layer_name = LayerName(sc.name + '/conv2', 'net_layer')
#       name = sc.name.replace(self._resnet_scope,'') + '/conv2'
      if layer_name in self._comp_weights_dict.keys():
        residual = self.separate_conv_layer(residual, depth_bottleneck, 3, stride, 
                                            rate=rate, layer_name='conv2', full_layer_name=layer_name)
      else:
        residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride, 
                                            rate=rate, scope='conv2')
      residual = layers.conv2d(
          residual, depth, [1, 1], stride=1, activation_fn=None, scope='conv3')
   
      output = nn_ops.relu(shortcut + residual)
   
      return utils.collect_named_outputs(outputs_collections, sc.name, output)

  def separate_conv_layer(self, inputs, num_output_channels, kernel_size, stride, rate,
                          layer_name, full_layer_name):
    with arg_scope(
      [slim.conv2d],
      weights_regularizer=None,
      weights_initializer=None,
      trainable=False,
      activation_fn=None,
      normalizer_fn=None,
      normalizer_params=None,
      biases_initializer=None): #make first layer clean, no BN no biases no activation func

      K = self._K_by_layer_dict[full_layer_name]
      layer1_name = LayerName(layer_name.replace('conv', 'convsep'))
      net = conv2d_same(inputs, K, kernel_size=(kernel_size,1), stride=[stride,1],
                         scope=layer1_name)
    
    with slim.arg_scope(resnet_arg_scope(is_training=False)):
      net = conv2d_same(net, num_output_channels, kernel_size=(1,kernel_size), 
                        stride=[1,stride], scope=layer_name)
      
    return net
    
  def rpn_convolution(self, net_conv4, is_training, initializer):
    layer_name = 'rpn_conv/3x3'

    if layer_name not in self._comp_weights_dict.keys():
      return slim.conv2d(net_conv4, 512, [3, 3], trainable=is_training, weights_initializer=initializer,
                        scope=layer_name)

    K = self._K_by_layer_dict[layer_name]
    layer1_name = LayerName(layer_name.replace('conv', 'convsep'))
    with arg_scope(
      [slim.conv2d],
      trainable=False,
      normalizer_fn=None,
      normalizer_params=None,
      biases_initializer=None,
      biases_regularizer=None): #make first layer clean, no BN no biases no activation func
     
      net = slim.conv2d(net_conv4, K, [3, 1], trainable=is_training, weights_initializer=initializer,
                        scope=layer1_name)

    with slim.arg_scope(resnet_arg_scope(is_training=False)):
      with arg_scope(
        [slim.conv2d],
        trainable=False,
        normalizer_fn=None,
        normalizer_params=None): #make second layer no BN but with biases
        net = slim.conv2d(net, 512, [1, 3], trainable=is_training, weights_initializer=initializer,
                          scope=layer_name)
    return net

  
  def build_base(self):
    layer_name = LayerName('conv1')
    if layer_name in self._comp_weights_dict.keys():
      with tf.variable_scope(self._resnet_scope, self._resnet_scope) as sc:
        N = self._comp_weights_dict[layer_name].shape[3]
  
        net = self.separate_conv_layer(self._image, N, 7, 2, rate=None, layer_name=layer_name,
                                       full_layer_name=layer_name)
  
        end_points_collection = self._resnet_scope + '_end_points'
        utils.collect_named_outputs(end_points_collection, self._resnet_scope+'/conv1', net)        
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
        net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool1')
    else:
      net = super(resnetv1_sep, self).build_base()
    return net
