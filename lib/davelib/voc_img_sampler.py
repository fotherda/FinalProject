'''
Created on 16 Jul 2017

@author: david
'''
import pickle as pi
import numpy as np
import collections
import copy
import sys, os
# cachefile = '/home/david/Project/tf-faster-rcnn_27/data/VOCdevkit2007/annotations_cache/annots.pkl'
cachefile = 'data/VOCdevkit2007/annotations_cache/annots.pkl'
sample_file = 'sample_imgs.pi'


class VOCImgSampler:
  
  """ a completely non-optimized class to do balanced subsampling of the test images where
      we try to ensure the relative numbers or images of each class are the same in the 
      subsample as they are in the full data set.
  """

  def __init__(self):
    with open(cachefile, 'rb') as f:
      try:
        self._recs = pi.load(f)
      except:
        self._recs = pi.load(f, encoding='bytes')

    self._class_img_dict = collections.defaultdict(list)
    self._img_class_dict = collections.defaultdict(set)
    
    for imagename, annos in self._recs.iteritems():
      for anno in annos:
        classname = anno['name']
        difficult = anno['difficult']
        if not difficult: #the evaluation ignores difficult cases so no point in using those images
          self._class_img_dict[classname].append(imagename) 
          self._img_class_dict[imagename].add(classname) 
          
        
  def get_ratios(self, class_img_dict):      
    sum_num_imgs = 0
    for classname, img_list in class_img_dict.iteritems():
#       print(classname + ': ' + str(len(img_list)))
      sum_num_imgs += len(img_list)

    ratios = {}
    for classname, img_list in class_img_dict.iteritems():
      ratios[classname] = float(len(img_list)) / sum_num_imgs
    
    return ratios
        

  def get_imgs(self, num_imgs): 
    if os.path.isfile(sample_file):
      with open(sample_file, 'rb') as f:
        sample_names_dict = pi.load(f)
        if num_imgs not in sample_names_dict:
          sample_names = []
          sample_names_dict[num_imgs] = sample_names
        else:
          return sample_names_dict[num_imgs]
    else:
      sample_names = []
      sample_names_dict[num_imgs] = sample_names
          
    target_ratios = self.get_ratios(self._class_img_dict)  
    sample_class_img_dict = {}
    for key in target_ratios.keys():
      sample_class_img_dict[key] = []
    
    act_ratios = dict.fromkeys(target_ratios.keys(), 0)
    
    while len(sample_names) < num_imgs:
      max_ = 0
      next_class=None
      for classname, target_ratio in target_ratios.iteritems():
        act_ratio = act_ratios[classname]
        deficit = target_ratio - act_ratio
        if deficit > max_:
          next_class = classname
          max_ = deficit
          
      poss_imgs = set(self._class_img_dict[next_class]) #get all imgs containing nextclass object
      poss_imgs = list(poss_imgs) #remove duplicates
      
      min_deficit_img = None
      min_deficit = sys.float_info.max
      for poss_img in poss_imgs:
        if poss_img in sample_names:
          continue
        # calc the diff/deficit between target and act if we add this img
        copy_dict = copy.deepcopy(sample_class_img_dict)
        for c in self._img_class_dict[poss_img]:
          copy_dict[c].append(poss_img)
      
        act_ratios = self.get_ratios(copy_dict)  
        sum_deficit = self.get_sum_deficit(target_ratios, act_ratios)
        if sum_deficit < min_deficit:
          min_deficit = sum_deficit
          min_deficit_img = poss_img
        
      if min_deficit_img:
        sample_names.append(min_deficit_img)
        for c in self._img_class_dict[min_deficit_img]:
          sample_class_img_dict[c].append(min_deficit_img)

      act_ratios = self.get_ratios(sample_class_img_dict)  


    if False: #for comparison just use random sample
      sample_names = np.random.choice(self._recs.keys(), size=num_imgs, replace=False)
      for name in sample_names:
        for c in self._img_class_dict[name]:
          sample_class_img_dict[c].append(name)
      act_ratios = self.get_ratios(sample_class_img_dict)  

    #print results    
    sum_deficit = 0      
    for classname, target_ratio in target_ratios.iteritems():
      act_ratio = act_ratios[classname]
      deficit = target_ratio - act_ratio
      sum_deficit += abs(deficit)
      print(str(deficit)+' = '+str(target_ratio)+' - '+str(act_ratio) + '  ' + classname)
    
    print 'sum_deficit='+str(sum_deficit)    

    pi.dump( sample_names_dict, open( sample_file, "wb" ) )
      
    return sample_names
  
  def get_sum_deficit(self, target_ratios, act_ratios):
    sum_deficit = 0      
    for classname, target_ratio in target_ratios.iteritems():
      act_ratio = act_ratios[classname]
      deficit = target_ratio - act_ratio
      sum_deficit += abs(deficit)
    return sum_deficit
