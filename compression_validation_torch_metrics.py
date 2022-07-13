%run -i run_MDTVSFA.py
import os
from sys import platform
if platform == 'win32':
    disk_dir = "E:/"
    disk_dir_R = "R:/"
    disk_dir_P = "P:/"
else:
    disk_dir_R = "/home/max/Ram/"
    disk_dir = "/home/max/driveE/"
    disk_dir_P = "/home/sdc2/"

dst_dir = disk_dir_P + "7videos/"    
home_dir = disk_dir_R + "tonemap/"
calc_met_True = calc_met
env = calc_met_True(model = "MDTVSFA", home_dir1=home_dir,dataset_dir=dst_dir,codec=' -preset:v medium -x265-params log-level=error -b:v 10M ',)

try:
    os.mkdir(home_dir)
except Exception:
    pass
try:
    os.mkdir(home_dir +  'VMAF_METRIX')
except Exception:
    pass
try:
    os.mkdir(home_dir +  'VMAF_METRIX/csv_res')
except Exception:
    pass
try:
    os.mkdir(home_dir +  'VMAF_METRIX/vid')
except Exception:
    pass



all_method_name = ["gamma_unsharp", "CLAHE", "Tonemap_Drago", "Identity"]
all_func = [aug4const, expose_cv2, tonemapDrago, Identity]
args_d_Identity = args_d_CLAHE
all_argd = [args_d_aug4const, args_d_CLAHE, args_tonemapDrago, args_d_Identity]
method_name_to_func = {i:j for i,j in zip(all_method_name, all_func)}
method_name_to_argd = {i:j for i,j in zip(all_method_name, all_argd)}
SSIM_m = env.SSIM_metrix_get
PSNR_m = env.PSNR_metrix_get
all_proxy_metr = [SSIM_m, PSNR_m]
all_proxy_names = ["SSIM", "PSNR"]
proxy_name_to_obj = {i:j for i,j in zip(all_proxy_names, all_proxy_metr)}
log_dir = disk_dir + "VMAF_METRIX/NeuralNetworkCompression/logs_black_box/"
bitrate_log_dir = disk_dir + "VMAF_METRIX/NeuralNetworkCompression/logs_black_box_compressed/"
k_lst = [1]
%run -i Current_model_lib.py
env.metr = get_met
idx_i = -1
for log_i in tqdm(sorted(os.listdir(log_dir))):
    idx_i+=1
    parse_str = log_i.split(".")[0].split("_")
    met_name, cur_proxy_name, cur_method_name = parse_str[0], parse_str[1], parse_str[2]
    print(cur_method_name + ", no codec tuning for "+ met_name + "_" + cur_proxy_name +", NSGA2")
    calculate_flag = (met_name == "PAC2PIQ" and cur_method_name == "Identity")#
    #(met_name == "KONIQ" and cur_method_name == "CLAHE") or \
            #(met_name == "KONIQ" and cur_method_name == "Tonemap_Drago" and cur_proxy_name == "PSNR") 
    if not calculate_flag:
        continue
    target_lst = [met_name]
    #if met_name != "PAC2PIQ" or cur_method_name != "Identity":
        #continue
        
    logs_all = np.load(os.path.join(log_dir, log_i), allow_pickle= True)
    log_bitrate = np.copy(logs_all)
    loss_calc = Custom_enh_Loss(target_lst = target_lst, k_lst=k_lst, to_train = False).eval()    
    if len(parse_str) >= 4:
        for parse_substr in parse_str[3:]:
            cur_method_name += "_" + parse_substr
    cur_func = method_name_to_func[cur_method_name]
    env.PSNR_metrix_get = proxy_name_to_obj[cur_proxy_name]
    env.func = axis_swaper(cur_func)
    
    for bv in tqdm(["100k","1M","2M","4M"]):
        for vid, log_np_idx in tqdm(list(zip(sorted(os.listdir(dst_dir)), range(len(logs_all))))):
            if len(logs_all[log_np_idx]) <= 2:
                continue
            env.dataset = [vid]
            env.init_video()
            for log_np_ind in range(len(logs_all[log_np_idx][2][:,0])):
                env.codec = ' -preset:v medium -x265-params log-level=error -b:v ' + bv + " "
                met10M = env.get_metrix([logs_all[log_np_idx][2][:,0][log_np_ind]])#Check for rewrining 
                log_bitrate[log_np_idx][2][:,1][log_np_ind] = np.array(met10M[0][0])
            out_name = log_i.split(".npy")[0] + "_bitrate" + bv
            np.save(os.path.join(bitrate_log_dir, out_name), log_bitrate)