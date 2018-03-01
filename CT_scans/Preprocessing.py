from utils3d import *
import pandas as pd
    
path = '/nfs/turbo/intmed-bnallamo-turbo/wsliu/Data/CT_scans/'
raw_folder, scan_lst, etc_files = next(os.walk(path+'raw'))
scan_lst.sort()

x_spacing = []
y_spacing = []
z_spacing = []
x_dim = []
y_dim = []
z_dim = []

for scan in scan_lst:
    scan_path = os.path.join(raw_folder, scan)
    ndarray = dcm_to_ndarray(scan_path, target_size=(256, 256, 192))
    np.save(path+'ndarray/'+scan+'.npy', ndarray)
    meta = get_metadata(scan_path)
    x_spacing.append(meta['x_spacing'])
    y_spacing.append(meta['y_spacing'])
    z_spacing.append(meta['z_spacing'])
    x_dim.append(meta['x_dim'])
    y_dim.append(meta['y_dim'])
    z_dim.append(meta['z_dim'])
    
meta_df = pd.DataFrame(dict(ID=scan_lst, x_spacing=x_spacing, y_spacing=y_spacing, z_spacing=z_spacing, x_dim=x_dim, 
                            y_dim=y_dim, z_dim=z_dim))

meta_df.to_csv('output/meta_data.csv')