from sys import platform
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
if platform == 'win32':
    disk_dir = "E:/"
    disk_dir_R = "R:/"
    disk_dir_P = "P:/"
else:
    disk_dir_R = "/home/max/Ram/"
    disk_dir = "/home/max/driveE/"
    disk_dir_P = "/home/sdc2/"

dst_dir = disk_dir_P + "7videos/" 
import os
import seaborn as sns
sns.set()
#disk_dir = "E:/"
#dst_dir = disk_dir + "VMAF_METRIX/CVPR_NSGA_new/logs_1403_7videos_compressed/"
dst_dir = disk_dir + "VMAF_METRIX/NeuralNetworkCompression/logs_black_box/"
#dst_dir = disk_dir + "VMAF_METRIX/NeuralNetworkCompression/logs_black_box_compressed/"
AV_log_lst = [dst_dir + i for i in sorted(os.listdir(dst_dir))]
AV_log_lst, AV_log_lst_Identity = [i for i in AV_log_lst if "Identity" not in i],  [i for i in AV_log_lst if "Identity" in i]
dst_dir = disk_dir_P + "7videos/" 

####

AV_log_methods = [np.load(i, allow_pickle = True) for i in AV_log_lst]
AV_log_methods = [AV_log if len(AV_log[0]) > 2 else AV_log[1:] for AV_log in AV_log_methods ]

####

groups = {}
for AV_log_1 in AV_log_lst:
    dir_name_raw = AV_log_1.split('/')[-1].lower()
    key = dir_name_raw.split('_')
    if key[2] == "gamma":
        key[2] = "gamma+unsharp"
        del key[3]
    if key[2] == "tonemap":
        key[2] = "tonemap drago"
        del key[3]
    
    
    if "vmaf" in dir_name_raw:
        key[-1] = key[-1][:-4]
        key = key[:-1]
    key = key[:3]
    if (*key,) in groups:
        groups[(*key,)].append(AV_log_1)
    else:
        groups[(*key,)] = [AV_log_1]
groups_Identity = {}
for AV_log_1 in AV_log_lst_Identity:
    dir_name_raw = AV_log_1.split('/')[-1].lower()
    key = dir_name_raw.split('_')
    del key[2]
    key = key[:2]
    if "pac2piq" in dir_name_raw:
        print(9)
    if "vmaf" in dir_name_raw:
        key = key[:-1]
    if (*key,) in groups_Identity:
        groups_Identity[(*key,)].append(AV_log_1)
    else:
        groups_Identity[(*key,)] = [AV_log_1]

####

#Black box with compression
markers = ["o", "<", "s", "p", "*", "P"]
def bitrate_to_marker(bitrate):
    if bitrate == "100k":
        return markers[0]
    if bitrate == "1M":
        return markers[1]
    if bitrate == "2M":
        return markers[2]
    if bitrate == "4M":
        return markers[3]
for group_key, group_val in groups.items():
    group_Identity_key = (group_key[0], group_key[1])
    if group_Identity_key in groups_Identity:
        groups_Identity_val = groups_Identity[group_Identity_key]
    else:
        groups_Identity_val = []
        print("Warning: No Identity method")
    plt.figure(dpi = 300)
    plt.xlabel(group_key[0].upper())
    plt.ylabel(group_key[1].upper())    
    dir_name = group_key[0] + " " + group_key[1] + " " + group_key[2]
    print(dir_name)
    plt.title(dir_name.upper())
    for idx, AV_log_1 in enumerate(group_val + groups_Identity_val):
        AV_log = np.load(AV_log_1, allow_pickle = True)
        colors = cm.rainbow(np.linspace(0, 1, len(AV_log)))
        for i,name,c in zip(AV_log, sorted(os.listdir(dst_dir)), colors):#:
            if len(i) < 3:
                continue
            p1 = [j[1] for j in i[2]]
            x =[j[0] for j in p1]
            y = [j[1] for j in p1]
            bitrate = AV_log_1.split("_")[-1].split("bitrate")[-1][:-4]
            if "Identity" in AV_log_1:
                plt.scatter(x,y, label = (name[:-4] if idx == 0 else ""), marker = bitrate_to_marker(bitrate),alpha = 1., s = 9, facecolors='black', edgecolors= c)
            else:
                plt.scatter(x,y, label = (name[:-4] if idx == 0 else ""), marker = bitrate_to_marker(bitrate),alpha = 0.8,color = c,s =4)
            plt.legend(fontsize = 5, bbox_to_anchor=(1.04,1))
    box_style=dict(boxstyle='round', facecolor='gray', alpha=0.5)
    
    plt.text(180,0.61, "Markers with black dot denote videos without preprocessing\nSingle-colour markers correspond to preprocessed videos\nVideos compressed with bitrate 100k are denoted as circles,\n1M - triangles, 2M - sqares, 4M - pentagons",{'color':'black','weight':'heavy','size':6},bbox=box_style)