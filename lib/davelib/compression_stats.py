'''
Created on 13 Jul 2017

@author: david
'''
import numpy as np
import pickle as pi
import matplotlib as mp
import matplotlib.pyplot as plt
import re

from collections import OrderedDict,defaultdict
from matplotlib.ticker import MaxNLocator


class CompressedNetDescription(dict):
   
  def __init__(self, compressed_layers, Ks):
    items = []
    self._Ks = []
    for layer, K in zip(compressed_layers,Ks):
      self[layer] = K
      self._Ks.append(K)
      items.extend((layer, K))
    self._key = tuple(items)
      
  def __key(self):
    return self['conv1']
  
#   def __key(self):
#     return self._key
#   
  def __eq__(self, other):
    if isinstance(other, self.__class__):
        return self.__key() == other.__key()
    return False
  
  def __hash__(self):
    return hash(self.__key())
  
  def __str__(self):
    return '\n'.join(map(str, sorted(self.iteritems())))

  def get_Kfrac(self):
    for layer, K in self.iteritems():
      if 'block4' in layer:
        Kfrac = K / 768.0
        break
      elif 'block3' in layer:
        Kfrac = K / 384.0
        break
        
    return Kfrac


class CompressionStats(object):

  def __init__(self, filename_suffix=''):
    self._filename_suffix = filename_suffix
    if filename_suffix: #load from pickle file
      self.load_from_file('CompressionStats_'+filename_suffix+'.pi')
    else:  
      self._stats = defaultdict( defaultdict )
      
  def __str__(self):
    str_list = []
    for K_by_layer_dict, d in self._stats.iteritems():
      str_list.append(str(K_by_layer_dict))
      str_list.append('\n')
      
      for type_label, value in d.iteritems():
        str_list.append(type_label + '\t' + str(value) + '\n')
    return ''.join(str_list)
  
  
  def print_Kfracs(self):
    str_list = []
    str_list.append('Kfracs:\n')
    for net_desc, d in self._stats.iteritems():
      str_list.append('%.2f'%net_desc.get_Kfrac())
      str_list.append('\n')

    str_list.append('Data Types:\n')
    for type_label in d:
        str_list.append(type_label + '\n')
    
    print( ''.join(str_list) )
  
  
  def load_from_file(self, filename):  
    self._stats = pi.load( open( filename, "rb" ) ) 
#     print self._stats   

  def set(self, K_by_layer_dict, type_label, value):
    self._stats[K_by_layer_dict][type_label] = value
#     for keys in self._stats.keys():
#       print keys
    
  def save(self, suffix=None):
    if not suffix:
      suffix = self._filename_suffix
    pi.dump( self._stats, open( 'CompressionStats_%s.pi'%suffix, "wb" ) )

  def add_data_type(self, type_label, values): #values must be in order to match sorted net descriptions
    for i, (K_by_layer_dict, d) in enumerate(sorted(self._stats.iteritems())):
      print(str(K_by_layer_dict['conv1']))
      d[type_label] = values[i]
    
  def merge(self, filename):
    other_stats = pi.load( open( filename, "rb" ) ) 
    
    for net_desc, data_dict in sorted(other_stats.iteritems()):
      if net_desc in self._stats:
        print('net_desc already exists')
        for type_label, d in sorted(data_dict.iteritems()):
          if type_label not in self._stats[net_desc]:
            self._stats[net_desc][type_label] = d
            print('added ' + type_label) 
      else:
        self._stats[net_desc] = data_dict


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

     
  def plot_by_Kfracs(self, plot_type_label=None):
    base_total = 47567455 # total no. parameters in base net
    plot_data = []
    calced_Kfracs = [] 
     
    for net_desc, data_dict in sorted(self._stats.iteritems()):
      for type_label, value in sorted(data_dict.iteritems()):
        if plot_type_label and plot_type_label not in type_label:
          continue
        if type_label =='var_redux':
          value = int( (base_total - value) / 1000000 )
        plot_data.append(value)
        calced_Kfracs.append( net_desc.get_Kfrac() )
        
    plt.ticklabel_format(style='plain')
    plt.plot(calced_Kfracs, plot_data,'o-')
#     plt.plot(Kfracs, plot_data,'o-')
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
     
  def plot_single_layers(self, layers_names, Kfracs, plot_type_label=None, ylabel=None):
    plt.title('Effects of single layer compressions',fontsize=16)
    legend_labels = []
    num_Kfracs = len(Kfracs)
    xs = range(1,len(layers_names)+1)
    plot_data = np.empty((len(layers_names), num_Kfracs))
     
    for ii, (net_desc, data_dict) in enumerate(sorted(self._stats.iteritems())):
      for i, (type_label, value) in enumerate(sorted(data_dict.iteritems())):
        if plot_type_label and type_label not in plot_type_label:
          continue
        layer_idx = layers_names.index(list(net_desc.keys())[0])
        Kfrac_idx = ii % num_Kfracs
        plot_data[layer_idx, Kfrac_idx] = value        
    
    for Kfrac_idx, Kfrac in enumerate(Kfracs):
      plt.plot(xs, plot_data[:,Kfrac_idx],'o-')
      legend_labels.append('%.2f'%Kfrac)
    legend_labels[0] = 'K=1'  
      
    plt.ylabel(ylabel, fontsize=16)
    plt.xlabel('layer', fontsize=16)
    
#     layers_names = [name.replace('/bottleneck_v1/conv2','') for name in layers_names]
#     plt.xticks(x, layers_names, rotation='vertical')
    plt.legend(legend_labels, title=r'fraction of $K_{max}$')
    plt.show()  
     
       
  def plot_correlation(self, type_labels, labels=None):
#     legend_labels = []
    if not labels:
      labels=[]

    for type_label in type_labels:
      xs = []
      ys = []
  
      for net_desc, data_dict in sorted(self._stats . iteritems()):
#         net_desc
        mAP = [data_dict[key] for key in data_dict.keys() if re.match('mAP', key)]
        if type_label in data_dict:
          diff_mean = data_dict[type_label]
          xs.append(mAP[0])
          ys.append(diff_mean)
          print( net_desc )
          labels.append('%.2f'%net_desc.get_Kfrac())
  
      plt.plot(xs, ys,'.')
  #     np.corcoef()
      
      plt.xlabel('mAP', fontsize=16)
      plt.ylabel('mean reconstruction error', fontsize=16)
  #     x1,x2,y1,y2 = plt.axis()
  #     plt.axis((0.7,0.8,y1,y2))
  
      if labels:
        for label, x, y in zip(labels, xs, ys):
          plt.annotate(
              str(label),
              xy=(x, y), xytext=(20, -10),
              textcoords='offset points', ha='right', va='bottom',
    #           bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
    #           bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
    #           arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0')
              )

    plt.legend(['block4 output','block3 output'])
    plt.show()  
     
     
  def data_dict_from_Kfrac(self, Kfrac):
    for net_desc, data_dict in sorted(self._stats.iteritems()):
      if abs(net_desc.get_Kfrac() - Kfrac) < 1.0e-9:
        return data_dict
    return None
        
       
  def get_key_value_match(self, d, match_key_pattern):     
    for key, value in d.iteritems():
      if re.match(match_key_pattern, key):
        return value
    return None
      
      
  def plot_correlation_btw_stats(self, other_stats, type_label, labels=None):
#     legend_labels = []
    if not labels:
      labels=[]

    xs = []
    ys = []

    for net_desc, data_dict in sorted(self._stats . iteritems()):
      value = self.get_key_value_match(data_dict, type_label)
      if value:
        other_data_dict = other_stats.data_dict_from_Kfrac(net_desc.get_Kfrac())
        if other_data_dict:
          other_value = self.get_key_value_match(other_data_dict, type_label)
          ys.append(value)
          xs.append(other_value)
#           print( net_desc )
          labels.append('%.2f'%net_desc.get_Kfrac())

      plt.plot(xs, ys,'.')
  
      
      
  #     x1,x2,y1,y2 = plt.axis()
  #     plt.axis((0.7,0.8,y1,y2))
  
      if labels:
        for label, x, y in zip(labels, xs, ys):
          if label == '0.40':
            xoffset = -20
          else:
            xoffset = 30
            
          if label == '0.60':
            yoffset = 10
          elif label == '0.50':
            yoffset = -20
          else:
            yoffset = -10
            
          plt.annotate(
              str(label),
              xy=(x, y), xytext=(xoffset, yoffset),
              textcoords='offset points', ha='right', va='bottom',
    #           bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
    #           bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
              arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0')
              )

    plt.ylabel(type_label+' #images 200', fontsize=16)
    plt.xlabel(type_label+' #images 4952', fontsize=16)
    nxs = np.vstack((xs,ys))
    corr_coeff = np.corrcoef(nxs)
    plt.text(0.2, 0.65, 'corr coeff = %.3f'%corr_coeff[0,1], fontsize=16, horizontalalignment='center', verticalalignment='center')
    plt.show()  
     
       
