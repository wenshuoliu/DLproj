from utils3d import *
import pandas as pd
    
path = '/nfs/turbo/intmed-bnallamo-turbo/wsliu/Data/CT_scans/'
raw_folder, scan_lst, etc_files = next(os.walk(path+'raw2'))

scan_lst.sort()
dcm_lst = []
mat_lst = []
for f in scan_lst:
    if not f[:2] == 's9':
        mat_lst.append(f)
    else:
        dcm_lst.append(f)

mat_lst.remove('s89457')
mat_lst.remove('s11418')
dcm_lst.append('s89457')
scan_lst.remove('s94214')
scan_lst.remove('s95108')

def read_meta(scan_lst, output_path, data_type, output_size=50):
    ID = []
    x_spacing = []
    y_spacing = []
    z_spacing = []
    x_dim = []
    y_dim = []
    z_dim = []
    extra_dim = []

    for ind, scan in enumerate(scan_lst):
        scan_path = os.path.join(raw_folder, scan)
        try:
            meta = get_metadata(scan_path, data_type)
            ID.append(scan)
            x_spacing.append(meta['x_spacing'])
            y_spacing.append(meta['y_spacing'])
            z_spacing.append(meta['z_spacing'])
            x_dim.append(meta['x_dim'])
            y_dim.append(meta['y_dim'])
            z_dim.append(meta['z_dim'])
            extra_dim.append(meta['extra_dim'])
        except IndexError:
            print(scan, 'has IndexError! \n')
        if (ind+1)%output_size == 0 or ind == len(scan_lst)-1:
            meta_df = pd.DataFrame(dict(ID=ID, x_spacing=x_spacing, y_spacing=y_spacing, z_spacing=z_spacing, x_dim=x_dim, 
                            y_dim=y_dim, z_dim=z_dim, extra_dim=extra_dim))
            meta_df.to_csv(output_path+'meta_data'+ data_type + str(ind // output_size)+'.csv')
            
read_meta(mat_lst, './output/', 'mat')
read_meta(dcm_lst, './output/', 'dcm')
