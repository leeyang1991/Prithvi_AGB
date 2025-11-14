
import platform
import os
from os.path import join

computer_name = platform.node()
centimeter_factor = 1 / 2.54
if 'wheat' in computer_name:
    # Wheat
    this_root = '/data/home/wenzhang/Yang/Prithvi_AGB/'
    global_device = 'cuda'
    print('Platform: Wheat')
elif 'yangli-ubt' in computer_name:
    # Dell
    this_root = '/home/yangli/HDD/Projects_Data/Prithvi_AGB/'
    global_device = 'cuda'
    print('Platform: Dell')
elif 'Yang-M4Pro.local' in computer_name:
    # MacBook
    import matplotlib
    print('Platform: MacOS')
    global_device = 'mps'
    # this_root = '/Volumes/HDD/GPP_ML/'
    this_root = '/Volumes/NVME4T/Prithvi_AGB/'
    matplotlib.use('TkAgg')
# elif 'yangligeo' in computer_name:
#     # yangligeo VPS
#     print('Platform: yangligeo')
#     this_root = '/root/GPP_ML/'
else:
    print('computer_name:',computer_name)
    raise ValueError('computer_name not recognized')
if not os.path.isdir(this_root):
    raise ValueError(f'working directory not found: {this_root}')

print('this_root:', this_root)
data_root = join(this_root, 'data')
results_root = join(this_root, 'results')
temp_root = join(this_root, 'temp')
conf_root = join(this_root, 'conf')