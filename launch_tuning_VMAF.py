exec(open('run_VMAF.py').read())
env = calc_met( model = "vmaf_v063", home_dir1=home_dir,dataset_dir=dst_dir)
all_method_name = ["gamma_unsharp", "CLAHE", "Tonemap_Drago"]
all_func = [aug4const, expose_cv2, tonemapDrago]
all_argd = [args_d_aug4const, args_d_CLAHE, args_tonemapDrago]
SSIM_m = env.get_ssim_vqmt
PSNR_m = env.PSNR_metrix_get
all_proxy_metr = [SSIM_m, PSNR_m]
all_proxy_names = ["SSIM", "PSNR"]


AV_log = []
AV_av = []
random.seed(a = 431)
toolbox = base.Toolbox()
creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
toolbox.register("individual", init_range, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", mut_cutom, indpb=1.)
toolbox.register("select", tools.selNSGA2)
for idx_proxy,idx_f  in tqdm(list(itertools.product(range(len(all_proxy_metr)), range(len(all_func))))):
    if idx_proxy == 0:
        continue
    cur_proxy = all_proxy_metr[idx_proxy]
    cur_proxy_name = all_proxy_names[idx_proxy]
    cur_method_name, cur_func, cur_argd = list(zip(all_method_name, all_func, all_argd))[idx_f] 
    print(cur_method_name + ", no codec tuning for VMAF +"+ cur_proxy_name +", NSGA2")
    args_d = cur_argd
    env.func = cur_func
    env.PSNR_metrix_get = cur_proxy
    stats = None
    AV_log = []
    AV_av = []
    for vid in tqdm(sorted(os.listdir(dst_dir))):
        env.dataset = [vid]
        creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
        pop = toolbox.population(n=28)
        pop, logbook,statsmy = My_eaMuPlusLambda(pop, toolbox ,mu = 14,lambda_ = 17, cxpb=0.5, mutpb=0.49, ngen=11, stats=stats)
        AV_log.append(logbook)
        with open('/home/max/driveE/VMAF_METRIX/CVPR_NSGA_new/logs/' + "VMAF_" + cur_proxy_name + "_" + cur_method_name + ".npy" , 'wb') as f:
            np.save(f, np.array(AV_log))