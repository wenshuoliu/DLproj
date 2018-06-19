import numpy as np
import os
from scipy.ndimage.interpolation import zoom
    
path = '/nfs/turbo/umms-awaljee/wsliu/Data/CT_scans/'

target_size = (224, 224, 128)

file_lst = os.listdir(path+'ndarray/')
for f in file_lst:
    raw_x = np.load(path+'ndarray/'+f)
    ratios = (target_size[0]/raw_x.shape[0], target_size[1]/raw_x.shape[1], target_size[2]/raw_x.shape[2])
    x = zoom(raw_x, zoom=ratios)
    np.save(path+'low_resolution/'+f, x)