'''
Created on 13 Jul 2017

@author: david
'''
import numpy as np
import pickle as pi
import matplotlib as mp
import matplotlib.pyplot as plt

from collections import OrderedDict
from matplotlib.ticker import MaxNLocator



class CompressionStats(object):

  def __init__(self, type_labels=None, compressed_layers=None, Ks=None, filename=None, 
               filename_suffix=''):
    
    if filename: #load from pickle file
      self.load_from_file(filename)
    else:  
      self._filename_suffix = filename_suffix
      self._stats = OrderedDict()
      for type_label in type_labels:
        self._stats[type_label] = OrderedDict()
  
        if compressed_layers is not None:
          for name in compressed_layers:
            self._stats[type_label][name] = OrderedDict()
            if Ks is not None:
              for k in Ks:
                self._stats[type_label][name][k] = 0
  
  def load_from_file(self, filename):  
    self._stats = pi.load( open( filename, "rb" ) ) 
#     print self._stats   

  def set(self, type_label, layer, K, value):
    if type_label not in self._stats:
      self._stats[type_label] = OrderedDict()
      
    if layer not in self._stats[type_label]:
      self._stats[type_label][layer] = OrderedDict()
      
    self._stats[type_label][layer][K] = value
    
  def save(self):
    pi.dump( self._stats, open( 'CompressionStats_%s.pi'%self._filename_suffix, "wb" ) )

  def plot(self, plot_type_label=None):
    fig, ax = plt.subplots()
    plt.title('Reconstruction Error')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    legend_labels = []
    n_rows = len(plot_type_label)
    n_columns = 1
    plt_idx = 1
    
    for i, (type_label, d) in enumerate(self._stats.iteritems()):
      if plot_type_label and type_label not in plot_type_label:
        continue
      num_layers = len(d)
      num_K = len( list(d.values())[0] )
      plot_data = np.zeros( (num_K, num_layers) )
      a = plt.subplot(n_rows, n_columns, plt_idx)
      plt_idx += 1

      for j, (layer, d2) in enumerate(d.iteritems()):
#         if j not in [31,33]:
#           continue
        Ks = []
        for k, (K, val) in enumerate(d2.iteritems()):
          plot_data[k,j] = val
          Ks.append(K)
#           plot_data[k,j] = self._stats[type_label][layer][K]
    
        plt.plot(Ks, plot_data[:,j],'o-')
#         plt.plot(Ks, plot_data[:,j],'ro-')
        legend_labels.append(layer)
#       legend_labels.append(type_label)
#       plt.plot(range(1,num_layers+1), plot_data[0,:],'ro-')
      plt.ylabel(type_label)

    plt.xlabel('K')
#       plt.ylabel(type_label)
    plt.legend(legend_labels)
#       plt.xlabel('layer index')
    plt.show()  
    
      
