import pydicom as dicom
import os
import numpy as np
import scipy.io as sio
from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt

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
        for filename in fileList:
            if ".dcm" in filename.lower():  
                file_lst.append(os.path.join(dirName,filename))

    #some scans contain dcm file with abnormal dimensions, remove them first:
    for f in file_lst[:]:
        dcm = dicom.read_file(f)
        if dcm.Rows>640 or dcm.Rows<400 or dcm.Columns>640 or dcm.Columns<400:
            file_lst.remove(f)
    file_lst.sort()

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

def mat_to_ndarray(data_path, target_size):
    """Convert a matlab .mat file into 3d numpy array
    
    # Arguments
        data_path: str of the path where the dicom files are stored
        target_size: tuple of the dimention of the output ndarray (x, y, z)
        
    # Return
        a 3d ndarray
    """
    dirName, subdirlst, file_lst = next(os.walk(data_path))
    if 'VOX.mat' not in file_lst:
        print("No VOX.mat file under data_path!")
        return None
    mat = sio.loadmat(os.path.join(data_path, 'VOX.mat'))
    data_array = mat['V']
    ratios = (target_size[0]/data_array.shape[0], target_size[1]/data_array.shape[1], target_size[2]/data_array.shape[2])
    rescaled = zoom(data_array, zoom=ratios)
    return rescaled
    

def get_metadata(data_path, source_type='dcm'):
    """Get the meta data from data_path, of either dcm or mat files 
    """
    if source_type == 'dcm':
        #get the list of the names of the dcm files:
        file_lst = []  
        for dirName, subdirList, fileList in os.walk(data_path):
            for filename in fileList:
                if ".dcm" in filename.lower():  
                    file_lst.append(os.path.join(dirName,filename))

        extra_dim = False
        for f in file_lst[:]:
            dcm = dicom.read_file(f)
            if dcm.Rows>640 or dcm.Rows<400 or dcm.Columns>640 or dcm.Columns<400:
                extra_dim = True
                file_lst.remove(f)
    
        dcm1 = dicom.read_file(file_lst[0])
        dim3d = (int(dcm1.Rows), int(dcm1.Columns), len(file_lst))
        spacings = (float(dcm1.PixelSpacing[0]), float(dcm1.PixelSpacing[1]), float(dcm1.SliceThickness))
    else:
        dirName, subdirlst, file_lst = next(os.walk(data_path))
        if 'VOX.mat' not in file_lst:
            print("No VOX.mat file under data_path!")
            return None
        mat = sio.loadmat(os.path.join(data_path, 'VOX.mat'))
        x = mat['xVec'].flatten()
        y = mat['yVec'].flatten()
        z = mat['zVec'].flatten()
        spacings = (x[1] - x[0], y[1]-y[0], z[1] - z[0])
        dim3d = (len(x), len(y), len(z))
        
        extra_dim = False
        for f in file_lst:
            if '.dcm' in f.lower():
                extra_dim = True
    
    meta = dict(x_spacing=spacings[0], 
                y_spacing=spacings[1], 
                z_spacing=spacings[2],
                x_dim=dim3d[0], 
                y_dim=dim3d[1], 
                z_dim=dim3d[2],
                extra_dim=extra_dim)
    return meta

def plot_ndarray(ndarray, meta = None, figsize = (16, 4)):
    dims = ndarray.shape
    if meta:
        x = np.linspace(0, (meta['x_dim']+1)*meta['x_spacing'], num = dims[0])
        y = np.linspace(0, (meta['y_dim']+1)*meta['y_spacing'], num = dims[1])
        z = np.linspace(0, (meta['z_dim']+1)*meta['z_spacing'], num = dims[2])
    else:
        x = np.linspace(0, 512*0.7, num = dims[0])
        y = np.linspace(0, 512*0.7, num = dims[1])
        z = np.linspace(0, 192*2.5, num = dims[2])
        
    plt.figure(figsize = figsize)
    
    plt.subplot(1, 4, 1)
    #sp.axes().set_aspect('equal', 'datalim')
    plt.set_cmap(plt.gray())
    plt.pcolormesh(x, y, np.transpose(ndarray[:, :, int(dims[2]/4)]))
    
    plt.subplot(1, 4, 2)
    #sp.axes().set_aspect('equal', 'datalim')
    plt.set_cmap(plt.gray())
    plt.pcolormesh(x, y, np.transpose(ndarray[:, :, int(dims[2]/2)]))
    
    plt.subplot(1, 4, 3)
    #sp.axes().set_aspect('equal', 'datalim')
    plt.set_cmap(plt.gray())
    plt.pcolormesh(x, y, np.transpose(ndarray[:, :, int(dims[2]*3/4)]))
    
    plt.subplot(1, 4, 4)
    #sp.axes().set_aspect('equal', 'datalim')
    plt.set_cmap(plt.gray())
    plt.pcolormesh(x, z, np.transpose(ndarray[:, int(dims[1]/2), :]))
        
    
