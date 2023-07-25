import scipy.io as sio
import os
import numpy as np
annotations_path = "./data/sketchyCOCO/Scene/Annotation/paper_version/"
import glob
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")


mode = "trainInTrain" #trainInTrain, val, valInTrain

bbox_mat_file_path = os.path.join(annotations_path, mode, "BBOX")
bbox_mat_files = glob.glob(f"{bbox_mat_file_path}/*.mat")[6:]
class_gt_mat_file_path = os.path.join(annotations_path, mode, "CLASS_GT")
class_gt_files = glob.glob(f"{class_gt_mat_file_path}/*.mat")[6:]
instance_gt_mat_file_path = os.path.join(annotations_path, mode, "INSTANCE_GT")
instance_gt_files = glob.glob(f"{instance_gt_mat_file_path}/*.mat")[6:]


for idx,i in enumerate(bbox_mat_files):
    print(i)
    bbox = sio.loadmat(i)
    print(bbox.keys())
    img = bbox["BBOX"]
    # img = bbox["BBOX"][:,:,[0,1]]
    # print(img)
    print(img.shape)
    print(np.unique(img, return_counts=True))


    img = Image.fromarray(img)
    im_array = np.asarray(img)
    # plt.imshow(im_array)
    # plt.show()
    # img.save(f"output_{current_datetime}.png", "PNG")
    if idx>50:
        break

# for i in class_gt_files:
#     print(i)
#     bbox = sio.loadmat(i)
#     # print(bbox.keys())
    
#     img = bbox["CLASS_GT"]
#     # print(img)
#     print(img.shape)
#     print(np.unique(img, return_counts=True))

#     img = Image.fromarray(img)
#     im_array = np.asarray(img)
#     plt.imshow(im_array)
#     plt.show()
#     # img.save(f"output_{current_datetime}.png", "PNG")
#     break

for i in instance_gt_files:
    print(i)
    bbox = sio.loadmat(i)
    # print(bbox.keys())
    img = bbox["INSTANCE_GT"]
    # img = bbox["INSTANCE_GT"][:,:,[0,1]]
    # print(img)
    print(img.shape)
    
    print(np.unique(img, return_counts=True))

    img = Image.fromarray(img)
    im_array = np.asarray(img)
    plt.imshow(im_array)
    plt.show()

    # img.save(f"output_{current_datetime}.png", "PNG")
    break




