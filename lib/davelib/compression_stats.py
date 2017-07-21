'''
Created on 13 Jul 2017

@author: david
'''
import numpy as np
import pickle as pi
import matplotlib as mp
import matplotlib.pyplot as plt

from collections import OrderedDict,defaultdict
from matplotlib.ticker import MaxNLocator


class CompressedNetDescription(dict):
   
  def __init__(self, compressed_layers, Ks):
    items = []
    for layer, K in zip(compressed_layers,Ks):
      self[layer] = K
      items.extend((layer, K))
    self._key = tuple(items)
      
  def __key(self):
    return self._key
  
  def __eq__(self, other):
    if isinstance(other, self.__class__):
        return self.__key() == other.__key()
    return False
  

  def __hash__(self):
    return hash(self.__key())

class CompressionStats(object):

  def __init__(self, filename=None, filename_suffix=''):
    
    self._filename_suffix = filename_suffix
    if filename: #load from pickle file
      self.load_from_file(filename)
    else:  
      self._stats = defaultdict( defaultdict )
  
  def load_from_file(self, filename):  
    self._stats = pi.load( open( filename, "rb" ) ) 
#     print self._stats   

  def set(self, K_by_layer_dict, type_label, value):
    self._stats[K_by_layer_dict][type_label] = value
#     for keys in self._stats.keys():
#       print keys
    
  def save(self):
    pi.dump( self._stats, open( 'CompressionStats_%s.pi'%self._filename_suffix, "wb" ) )

  def build_label_layer_K_dict(self):
    new_dict = defaultdict( lambda: defaultdict (lambda: defaultdict(float)) )
    for K_by_layer_dict, d in self._stats.iteritems():
      if len(K_by_layer_dict) != 1:
        continue
      layer = list(K_by_layer_dict.keys())[0]
      K = K_by_layer_dict[layer]
      for type_label, value in d.iteritems():
        new_dict[type_label][layer][K] = value
    return new_dict

  def plot(self, plot_type_label=None):
    fig, ax = plt.subplots()
    plt.title('Reconstruction Error')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    legend_labels = []
    n_rows = len(plot_type_label)
    n_columns = 1
    plt_idx = 1
     
    data_dict = self.build_label_layer_K_dict()
     
    for i, (type_label, d) in enumerate(sorted(data_dict.iteritems())):
      if plot_type_label and type_label not in plot_type_label:
        continue
      num_layers = len(d)
      num_K = len( list(d.values())[0] )
      plot_data = np.zeros( (num_K, num_layers) )
      a = plt.subplot(n_rows, n_columns, plt_idx)
      plt_idx += 1
 
      layer_idxs = []
      for j, (layer, d2) in enumerate(sorted(d.iteritems())):
#         if j not in [31,33]:
#           continue
        Ks = []
        layer_idxs.append(j)
        for k, (K, val) in enumerate(sorted(d2.iteritems())):
          plot_data[k,j] = val
          Ks.append(K)
#           plot_data[k,j] = self._stats[type_label][layer][K]

      for k, (K, val) in enumerate(sorted(d2.iteritems())):
        plt.plot(layer_idxs, plot_data[k,:],'o-')
        legend_labels.append('K=%d'%K)
#         plt.plot(Ks, plot_data[:,j],'o-')
#       legend_labels.append(layer)
#       legend_labels.append(type_label)
#       plt.plot(range(1,num_layers+1), plot_data[0,:],'ro-')
      plt.ylabel(type_label)
 
    plt.xlabel('K')
#       plt.ylabel(type_label)
    plt.legend(legend_labels)
#       plt.xlabel('layer index')
    plt.show()  

     
  def plot_by_Kfracs(self, Kfracs, plot_type_label=None):
    base_total = 47567455 # total no. parameters in base net
    plot_data = []
     
    for ii, (net_desc, data_dict) in enumerate(sorted(self._stats.iteritems())):
      for i, (type_label, value) in enumerate(sorted(data_dict.iteritems())):
        if plot_type_label and type_label not in plot_type_label:
          continue
        if type_label =='var_redux':
          value = int( (base_total - value) / 1000000 )
        plot_data.append(value)
        
    plt.ticklabel_format(style='plain')
    plt.plot(Kfracs, plot_data,'o-')
    plt.ylabel('No. parameters in net $x10^6$')
#     plt.ylabel('mAP')
    plt.xlabel(r'fraction of $K_{max}$')
    plt.show()  
     
       
  def plot_K_by_layer(self, layers_names, Kfracs, plot_type_label=None):
    plt.title('Profile of layer compressions',fontsize=16)
    legend_labels = []
     
    for ii, (net_desc, data_dict) in enumerate(sorted(self._stats . iteritems())):
      for i, (type_label, value) in enumerate(sorted(data_dict.iteritems())):
        if plot_type_label and type_label not in plot_type_label:
          continue
        num_layers = len(net_desc)
        plot_data = []
        x = range(0,num_layers)
        for layer in layers_names:
          K = net_desc[layer]
          plot_data.append(K)
        plt.plot(x, plot_data,'o-')
        legend_labels.append('%.2f'%Kfracs[ii])
    
    plt.ylabel('K', fontsize=16)
    plt.xlabel('layer', fontsize=16)
    
    layers_names = [name.replace('/bottleneck_v1/conv2','') for name in layers_names]
    plt.xticks(x, layers_names, rotation='vertical')
    plt.legend(legend_labels, title=r'fraction of $K_{max}$')
    plt.show()  
     
       
