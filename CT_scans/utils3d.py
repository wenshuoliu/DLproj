import dicom
import os
import numpy as np
import scipy.io as sio
from scipy.ndimage.interpolation import zoom

def dcm_to_ndarray(data_path, target_size):
    """Convert a stack of dicom files into 3d numpy array
    
    # Arguments
        data_path: str of the path where the dicom files are stored
        target_size: tuple of the dimention of the output ndarray (x, y, z)
        
    # Return
        a 3d ndarray
    """
    #get the list of the names of the dcm files:
    file_lst = []  
    for dirName, subdirList, fileList in os.walk(data_path):
        fileList.sort()
        for filename in fileList:
            if ".dcm" in filename.lower():  
                file_lst.append(os.path.join(dirName,filename))
                
    dcm1 = dicom.read_file(file_lst[0])
    dim3d = (int(dcm1.Rows), int(dcm1.Columns), len(file_lst))
    
    data_array = np.zeros(dim3d, dtype=dcm1.pixel_array.dtype)
    # loop through all the DICOM files
    for f in file_lst:
        # read the file
        dcm = dicom.read_file(f)
        # store the raw image data
        data_array[:, :, file_lst.index(f)] = dcm.pixel_array
        
    ratios = (target_size[0]/data_array.shape[0], target_size[1]/data_array.shape[1], target_size[2]/data_array.shape[2])
    rescaled = zoom(data_array, zoom=ratios)
    return rescaled

def get_metadata(data_path):
    """Get the meta data of the dcm files 
    """
    #get the list of the names of the dcm files:
    file_lst = []  
    for dirName, subdirList, fileList in os.walk(data_path):
        for filename in fileList:
            if ".dcm" in filename.lower():  
                file_lst.append(os.path.join(dirName,filename))
                
    dcm1 = dicom.read_file(file_lst[0])
    dim3d = (int(dcm1.Rows), int(dcm1.Columns), len(file_lst))
    meta = dict(x_spacing=float(dcm1.PixelSpacing[0]), 
                y_spacing=float(dcm1.PixelSpacing[1]), 
                z_spacing=float(dcm1.SliceThickness),
                x_dim=int(dcm1.Rows), 
                y_dim=int(dcm1.Columns), 
                z_dim=len(file_lst))
    return meta
    
    