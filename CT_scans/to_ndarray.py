from utils3d import *
import pandas as pd
    
path = '/nfs/turbo/intmed-bnallamo-turbo/wsliu/Data/CT_scans/'
raw_folder, scan_lst, etc_files = next(os.walk(path+'raw2'))

for f in scan_lst[:]:
    if not f[:2] == 's9':
        scan_lst.remove(f)

scan_lst.append('s89457')
scan_lst.remove('s94214')
scan_lst.remove('s95108')
scan_lst.sort()

ind = scan_lst.index('s91219')
scan_lst = scan_lst[(ind+1):]

for scan in scan_lst:
    scan_path = os.path.join(raw_folder, scan)
    try:
        ndarray = dcm_to_ndarray(scan_path, target_size=(256, 256, 192))
        np.save(path+'ndarray2/'+scan+'.npy', ndarray)
    except IndexError:
        print(scan, ' has IndexError!\n')

