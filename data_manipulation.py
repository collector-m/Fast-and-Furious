import numpy as np 
from argoverse.utils import ply_loader, se3, mayavi_wrapper
from argoverse.data_loading.pose_loader import get_city_SE3_egovehicle_at_sensor_t
from argoverse.visualization import mayavi_utils
import point_cloud_projection_utils
import matplotlib.pyplot as plt
import os
import pickle

LOG_ID = 'c6911883-1843-3727-8eaa-41dc8cda8993'
DATASET_DIR = 'argoverse-tracking/sample'

def load_all_clouds(dataset=DATASET_DIR, log_id=LOG_ID):
    cloud_dict = {}
    file_path = dataset + '/' + log_id + '/lidar'
    for file in os.listdir(dataset + '/' + log_id + '/lidar'):
        point_cloud = ply_loader.load_ply(file_path + '/' + file)
        timestamp = int(file[3:-4])
        cloud_dict[timestamp] = point_cloud
    return cloud_dict

def perform_SE3(cloud_dict, dataset=DATASET_DIR, log_id=LOG_ID):
    for k, v in cloud_dict.items():
        SE3 = get_city_SE3_egovehicle_at_sensor_t(str(k), dataset, log_id)
        cloud_dict[k] = SE3.transform_point_cloud(v)
    return cloud_dict

def create_occupancy_grids(cloud_dict, resolution=0.2, Xsize=144, Ysize=80, bins = 29, minHeight = -2, maxHeight = 3.5):
    regionX = int(Xsize/resolution)
    regionY = int(Ysize/resolution)
    deleted_keys = []
    for k, v in cloud_dict.items():
        #try:
        print(k)
        cloud_dict[k] = point_cloud_projection_utils.form_voxel_slice_img(regionX, regionY, v, num_slices=bins, zmin=minHeight, zmax=maxHeight)
        #except:
            #deleted_keys.append(k)
    for k in deleted_keys:
        del cloud_dict[k]
    return cloud_dict

def save_occupancy_grids(cloud_dict):
    with open('occupancies.pickle', 'wb') as handle:
        pickle.dump(cloud_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_occupancy_grids():
    with open('occupancies.pickle', 'rb') as handle:
        return pickle.load(handle)

def visualize_point_cloud(cloud_dict):
    for k, v in cloud_dict.items():
        break
    fig = mayavi_utils.draw_lidar(v)
    mayavi_wrapper.mlab.view(figure=fig)
    mayavi_wrapper.mlab.show()

# Stack 5 Occupancy grids and sort by timestamp

cloud_dict = load_all_clouds()
cloud_dict = perform_SE3(cloud_dict)
cloud_dict = create_occupancy_grids(cloud_dict)
'''
save_occupancy_grids(cloud_dict)
loaded = load_occupancy_grids()
for k, v in loaded.items():
    break
plt.imshow(v[:,:,0], cmap='gray')
plt.show()
'''