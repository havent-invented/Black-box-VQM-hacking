

exec(open('run_VMAF.py').read())
env_proxy = calc_met(model = "vmaf_v063", home_dir1=home_dir,dataset_dir=dst_dir)

exec(open('run_MDTVSFA.py').read())
env = calc_met( model = "MDTVSFA", home_dir1=home_dir,dataset_dir=dst_dir)
cur_arg_global = None

env_get_metrix_primary = env.get_metrix2

def get_metr_first(args):
    global cur_arg_global
    global env_get_metrix_primary
    global env
    cur_arg_global = args
    return env_get_metrix_primary(args)


def get_metr_second():
    global cur_arg_global
    global env_proxy
    return env_proxy.get_metrix2(cur_arg_global)[0]

env.get_metrix2 = get_metr_first

all_method_name = ["gamma_unsharp", "CLAHE", "Tonemap_Drago"]
all_func = [aug4const, expose_cv2, tonemapDrago]
all_argd = [args_d_aug4const, args_d_CLAHE, args_tonemapDrago]
SSIM_m = env.SSIM_metrix_get
PSNR_m = env.PSNR_metrix_get
all_proxy_metr = [get_metr_second]
all_proxy_names = ["VMAF"]



def My_eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                   stats=None, halloffame=None, verbose=__debug__):
    
    global offspring
    statsmy = []
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.evaluate(invalid_ind)
    
    for ind, fit in zip(invalid_ind, *fitnesses):
        ind.fitness.values = fit
    # Begin the generational process
    bst_val = 0
    for gen in range(1, ngen + 1):
        
        # Vary the population
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        ##########fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        fitnesses = toolbox.evaluate(invalid_ind)
        #print(fitnesses)
        
        for ind, fit in zip(invalid_ind, *fitnesses):
            ind.fitness.values = fit
    
        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)
        
        #LOGGING
        log_min = 100
        log_meean = 0
        log_max = 0
        bst_ind = [0, 0]
        for ind in population :
            if ind.fitness.valid:
                #print(ind.fitness.values)
                ind_value = ind.fitness.values[0]
                #print(ind_value,ind)
                if bst_ind[-1] < ind_value:
                    bst_ind = np.array(ind),ind.fitness.values[0], np.array([[[*i], [*i.fitness.values]] for i in pop ]), ind_value
                log_min = min(log_min, ind_value)
                log_max = max(log_max, ind_value)
                log_meean += ind_value
        log_meean = log_meean / len(population + offspring)
        print("GEN:" + str(gen) + " ", log_min, log_meean, log_max)
        statsmy.append([population,log_min,log_meean,log_max])
    print("BEST IND:", bst_ind)
    return population, bst_ind,statsmy



AV_log = []
AV_av = []
random.seed(a = 431)
env.metr = env.MDTVSFA
toolbox = base.Toolbox()
creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
toolbox.register("individual", init_range, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", mut_cutom, indpb=1.)
toolbox.register("select", tools.selNSGA2)
idx_ = 0
for idx_proxy,idx_f  in tqdm(list(itertools.product(range(len(all_proxy_metr)), range(len(all_func))))):
    cur_proxy = all_proxy_metr[idx_proxy]
    cur_proxy_name = all_proxy_names[idx_proxy]
    cur_method_name, cur_func, cur_argd = list(zip(all_method_name, all_func, all_argd))[idx_f] 
    print(cur_method_name + ", no codec tuning for MDTVSFA +"+ cur_proxy_name +", NSGA2")
    args_d = cur_argd
    env.func = axis_swaper(cur_func)
    env_proxy.func = cur_func
    if idx_ == 0:
        idx_ += 1
        continue
    
    env.PSNR_metrix_get = cur_proxy
    stats = None
    AV_log = []
    if idx_ == 1:
        AV_log = list(np.load("/home/max/driveE/VMAF_METRIX/CVPR_NSGA_new/logs/MDTVSFA_VMAF_CLAHE.npy", allow_pickle = True))
    AV_av = []
    
    for idx_vid, vid in enumerate(tqdm(sorted(os.listdir(dst_dir)))):
        if idx_ == 1 and idx_vid <= 26:
            continue
            
        env.dataset = [vid]
        env_proxy.dataset = [vid]
        env.init_video()
        creator.create("FitnessMax", base.Fitness, weights=(100.0, 1.0))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
        pop = toolbox.population(n=28)
        pop, logbook,statsmy = My_eaMuPlusLambda(pop, toolbox ,mu = 14,lambda_ = 17, cxpb=0.5, mutpb=0.49, ngen=11, stats=stats)
        AV_log.append(logbook)
        #with open('/home/max/driveE/VMAF_METRIX/CVPR_NSGA_new/logs/' + "MDTVSFA_" + cur_proxy_name + "_" + cur_method_name + ".npy" , 'wb') as f:
        #    np.save(f, np.array(AV_log))
    idx_ += 1