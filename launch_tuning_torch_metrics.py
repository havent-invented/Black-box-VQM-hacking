%run -i run_MDTVSFA.py
import os
home_dir = disk_dir_R + "tonemap/"
os.system('mkdir ' +home_dir)
os.system('mkdir ' +home_dir +  'VMAF_METRIX')
os.system('mkdir ' +home_dir +  'VMAF_METRIX/csv_res')
os.system('mkdir ' +home_dir +  'VMAF_METRIX/vid')

dst_dir = "P://7videos/" 
env = calc_met( model = "MDTVSFA", home_dir1=home_dir,dataset_dir=dst_dir)
all_method_name = ["gamma_unsharp", "CLAHE", "Tonemap_Drago", "Identity"]
all_func = [aug4const, expose_cv2, tonemapDrago, Identity]
args_d_Identity = args_d_CLAHE
all_argd = [args_d_aug4const, args_d_CLAHE, args_tonemapDrago, args_d_Identity]
SSIM_m = env.SSIM_metrix_get
PSNR_m = env.PSNR_metrix_get
all_proxy_metr = [SSIM_m, PSNR_m]
all_proxy_names = ["SSIM", "PSNR"]
exec(open('Current_model_lib.py').read())
met_name = "Linearity"
target_lst = [met_name]
k_lst = [1]
loss_calc = Custom_enh_Loss(target_lst = target_lst, k_lst=k_lst, to_train = False)
loss_calc = loss_calc.eval()
met_names = ["MDTVSFA", "Linearity", "SPAQ", "VSFA", "PAC2PIQ", "NIMA", "KONIQ"]
met_name = None

####


def get_met(X):
    global tmp
    tmp = X
    if met_name == "VSFA":
        return -loss_calc({"x_hat": torch.cat([X,X])}, torch.cat([X,X]))["loss"]
    else:
        return -loss_calc({"x_hat": X}, X)["loss"]
    
for met_name in met_names:
    print(met_name)
    target_lst = [met_name]
    k_lst = [1]
    loss_calc = Custom_enh_Loss(target_lst = target_lst, k_lst=k_lst, to_train = False)
    loss_calc.eval()
    loss_calc.get_met = get_met
    AV_log = []
    AV_av = []
    random.seed(a = 431)
    env.metr = get_met#
    toolbox = base.Toolbox()
    creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
    toolbox.register("individual", init_range, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evalOneMax)
    toolbox.register("mate", cxTwoPointCopy)
    toolbox.register("mutate", mut_cutom, indpb=1.)
    toolbox.register("select", tools.selNSGA2)
    creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    for idx_proxy, idx_f  in tqdm(list(itertools.product(range(len(all_proxy_metr)), range(len(all_func))))):
        cur_proxy = all_proxy_metr[idx_proxy]
        cur_proxy_name = all_proxy_names[idx_proxy]
        cur_method_name, cur_func, cur_argd = list(zip(all_method_name, all_func, all_argd))[idx_f] 
        args_d = cur_argd
        print(cur_method_name + ", no codec tuning for "+ met_name + "_" + cur_proxy_name +", NSGA2")
        env.func = axis_swaper(cur_func)
        env.PSNR_metrix_get = cur_proxy
        stats = None
        AV_log = []
        AV_av = []
        for vid in tqdm(sorted(os.listdir(dst_dir))):
            pop = toolbox.population(n=28)
            env.dataset = [vid]
            env.init_video()
            if cur_method_name == "Identity":
                ngen = 1
                mu = 1
                lambda_ = 1
            else:
                ngen = 11
                mu = 14
                lambda_ = 17
            pop, logbook,statsmy = My_eaMuPlusLambda(pop, toolbox ,mu = mu,lambda_ = lambda_, cxpb=0.5, mutpb=0.49, ngen=ngen, stats=stats)
            AV_log.append(logbook)
            #with open(disk_dir + 'VMAF_METRIX/NeuralNetworkCompression/logs_black_box/' + met_name + "_" + cur_proxy_name + "_" + cur_method_name + ".npy" , 'wb') as f:
                #np.save(f, np.array(AV_log))



####