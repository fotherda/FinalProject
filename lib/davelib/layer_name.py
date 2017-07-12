'''
Created on 10 Jul 2017

@author: david
'''

class LayerName(str):

  def __new__(cls, value, flag=None):
    if flag in ['net_layer','net_layer_weights']:
      idx = value.index('/')
      value = value[idx+1:]
#     if flag in ['layer_weights','net_layer_weights']:
#         value = value.replace('/weights','')
    value = value.replace('/weights','')
    # explicitly only pass value to the str constructor
    return super(LayerName, cls).__new__(cls, value)
  
  def __init__(self, value, flag=None):
        # ... and don't even call the str initializer 
      self.flag = flag
      if 'weights' in value:
        self._has_weights = True
      else:
        self._has_weights = False

  def net_layer_weights(self, net):
    return net + '/' + self.layer_weights()
      
  def net_layer(self, net):
    return net + '/' + self
      
  def layer_weights(self):
    if self._has_weights:
      return self + '/weights'
    else:
      return self 
    
  def sep_version(self):
    return LayerName(self.layer_weights().replace('conv', 'convsep'), self.flag)
    
