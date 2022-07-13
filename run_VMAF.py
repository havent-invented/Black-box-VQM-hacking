#CLAHE Preprocessing + PSNR + VMAF
import sys
import torch
#from torchvision import transforms
import skvideo.io
from PIL import Image
import numpy as np
from MDTVSFA.CNNfeatures import get_features
from MDTVSFA.VQAmodel import VQAModel
from argparse import ArgumentParser
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from PIL import Image
import torch
import numpy as np
def axis_swaper(func):
    def wrapper(*args):
        args = np.swapaxes(args[0],0 , -1), *args[1:]
        return np.swapaxes(func(*args),0, -1)
    return wrapper
#disk_dir = "E:/"
#disk_dir_R = "R:/"
disk_dir = "/home/max/driveE/"
disk_dir_R = "/home/max/Ram/"
home_dir = disk_dir_R + "tonemap/"
prev = ''
cdir = disk_dir + "/VMAF_METRIX/CVPR_results/dataset_CC_10/"
dst_dir = disk_dir +"/VMAF_METRIX/CVPR_results/dataset_CC_10/"

def tonemapDrago(img, arg1,arg2,arg3):#[0,1]bst 0,7,0.9,
    # bias 0,1
    #sturation 0,1
    return np.nan_to_num( (cv2.createTonemapDrago(arg1,arg2,arg3).process(img.astype("float32"))),copy=True, nan = 0.5)#float version
from deap import *####ZXC
from skimage.metrics import mean_squared_error
from tqdm.notebook import tqdm
import skimage
import skimage.filters
import skimage.metrics
import skimage.io
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.color import rgb2hsv, hsv2rgb
from skimage.filters import unsharp_mask
from skimage import img_as_float, img_as_ubyte
import time
import numpy as np
import os
import itertools
import skvideo.io
import pandas as pd
import skimage.metrics
import shlex
import subprocess
import pandas as pd
from multiprocessing import Pool
from line_profiler import LineProfiler
import cv2
import shutil
import albumentations as A
import random

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
def Identity(img,arg1,arg2):
    return img


#args_d = {'n':2,'a1_min': 0.00001 ,'a1_max':20., 'a2_min' : 1, 'a2_max' : 100, 'type1': 'float','type2': 'int'}
args_tonemapDrago = {'n':3,'a0_min': 0. ,'a0_max': 2.5, 
          'a1_min' : 0., 'a1_max' : 3. , 
          'a2_min' : 0.0, 'a2_max' : 1. , 
          }
args_tonemapMantiuk = {'n':3,'a0_min': 0. ,'a0_max':2.5, 
          'a1_min' : 0.0, 'a1_max' : 1. , 
          'a2_min' : 0.00001, 'a2_max' : 1. , 
          }
args_tonemapReinhard = {'n':4,'a0_min': 0. ,'a0_max':2.5, 
          'a1_min' : -8., 'a1_max' : 8. , 
          'a2_min' : 0., 'a2_max' : 1. , 
          'a3_min' : 0., 'a3_max' : 1. , 
          }
args_d = {'n':2,'a0_min': 1 ,'a0_max':100, 'a1_min' : 0.00001, 'a1_max' : 20. , 'type0': 'int','type1': 'float'}



def init_range(icls):
    global args_d
    ret_val = []
    for i in range(args_d['n']):
        val1 = float(random.uniform(args_d['a' + str(i) + "_min"] , 
                                    args_d['a' + str(i) + "_max"] )) 
        ret_val.append(val1)
        
            
    #int_1_10_part = np.random.randint(1, 11, 3)
    #int_0_1_part = np.random.randint(0, 2, 3)
    #float_part = np.random.rand(4) * 0.1
    #print(ret_val)
    ind = np.array(ret_val)
    return icls(ind)
#NEW

def evalOneMax(individual):
    individual = np.array(individual)
    #individual[:,1] = np.maximum(individual[:,0]+1, individual[:,1])
    #individual[:,1] = np.maximum(0,individual[:,1])
    res = env.get_metrix(individual)
    #print(res,*individual)
    return res

def cxTwoPointCopy(ind1, ind2):

    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
        
    return ind1, ind2


def mut_cutom(individual, indpb = 0.5):
    
    arg1 = individual[0]
    
    
    n_i = random.randint(0,args_d['n']-1)
    
    arg1 = individual[n_i]  + np.random.randn() * (args_d['a' + str(n_i) + "_max"] - args_d['a' + str(n_i) + "_min"]) / 15.
    arg1 = np.clip(arg1, args_d['a' + str(n_i) +  '_min']  , args_d['a' + str(n_i) +  '_max'])
    
    individual[n_i] = arg1
    
    return (individual, )


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

def axis_swaper(func):
    def wrapper(*args):
        args = np.swapaxes(args[0],0 , -1), *args[1:]
        return np.swapaxes(func(*args),0, -1)
    return wrapper

os.system('mkdir ' +home_dir)
os.system('mkdir ' +home_dir +  'VMAF_METRIX')
os.system('mkdir ' +home_dir +  'VMAF_METRIX/csv_res')
os.system('mkdir ' +home_dir +  'VMAF_METRIX/vid')



class calc_met:# 
    def __init__(self,dataset1 = ["Run439.Y4M"], func1 = Identity, convKer1 = None, home_dir1 = "R:/", creat_dir = False, calc_SSIM_PSNR = False, calc_model_features = False, model = "vmaf_v063" , codec = '   -preset:v medium -x265-params log-level=error ',dataset_dir = "dataset/"):
        "Init env"
        self.lot_of_frames = False
        self.dataset_err = None
        self.dataset_np = None
        self.dataset = []
        self.crf_arr = []
        self.dataset_dir = dataset_dir
        self.home_dir = home_dir1
        self.codec = codec
        self.model = model
        self.calc_model_features = calc_model_features
        self.calc_SSIM_PSNR = calc_SSIM_PSNR
        self.global_sats = []
        self.Results = []
        
        if convKer1 is None:
            self.convKer = np.array([[ 0.54744114, -0.03412791, -0.10635577],
           [ 0.0590166 ,  1.15195262,  0.05838338],
           [-0.15729069, -0.14071838,  0.21105842]])
        else:
            self.convKer = convKer1
        self.dataset = dataset1
        self.Results = []
        self.func = func1
        if creat_dir:
            try:
                if not os.path.exists(home_dir1):
                    os.mkdir(home_dir1)
            except Exception:
                ignore
        try:
            if not os.path.exists(self.home_dir + "VMAF_METRIX/ "):
                os.mkdir(self.home_dir + "VMAF_METRIX/")
                os.mkdir(self.home_dir + "VMAF_METRIX/csv_res")
        except Exception:
            pass
        
    def Read_frames(self):
        self.dataset_err = [frameGT for frameGT in skvideo.io.FFmpegReader( self.home_dir + "0YES.Y4M", outputdict={"-c:v" :" rawvideo","-f": "rawvideo"}).nextFrame()]
        
    def get_metrix2(self, args):
    
        """
        args is a list of args. [[arg_for_first], [arg_for_second],...]
        Return value:
        """
        i,j = 1,1
        saved_N = 0
        saved_O = 0
        for el in self.dataset:    
            self.full_el = self.dataset_dir + self.dataset[0] 
            sub_str = "0"#el + "_" + str(i) + "_" + str(j)
            res_str =  el + "_" + str(i) + "_" + str(j) + ".csv"
            shutil.copyfile(self.full_el,  self.home_dir + "GT.Y4M")
            first_one = False
            self.datagen = [frameGT for frameGT in skvideo.io.FFmpegReader( self.home_dir + "GT.Y4M", outputdict={"-c:v" :" rawvideo","-f": "rawvideo"}).nextFrame()]
            Result = [el, i, j]
            ret = 1
            self.per_frame_args = []
            retGT = 1
            L2 = 0
            countL2 = 0
            for idx, iarg in enumerate(args):
                out1 = skvideo.io.FFmpegWriter( self.home_dir +  sub_str + "YES.Y4M",inputdict = {"-pix_fmt": "rgb24"}, outputdict = {"-pix_fmt": "yuv420p"})
                self.dataset_np = self.datagen
                self.dataset_err = []
                for frameGT in self.datagen:
                    self.err =  self.func(frameGT, *iarg)
                    self.dataset_err.append(self.err)
                    out1.writeFrame(self.err)
                out1.close()
                      
            vqmt_opt = "vqmt -quiet -slots 8 -orig " + self.home_dir + "GT.Y4M -in " + self.home_dir +  sub_str + "YES.Y4M" + " -csv -csv-dir " + self.home_dir + "VMAF_METRIX/csv_res -cng CUSTOM " + sub_str + "name -metr vmaf over Y -set disable_clip=true -dev OpenCL0 -set model_preset=" #+ self.model + " "
            if self.calc_model_features:
                vqmt_opt += "standard_features_neg "
            else: 
                vqmt_opt +=  self.model + " "
            """
            if self.calc_SSIM_PSNR and self.calc_model_features:
                print("self.calc_SSIM_PSNR and self.calc_model_features is not supported")
                raise Exception
            """    
            if self.calc_SSIM_PSNR:
                vqmt_opt += " -metr ssim over Y  -metr psnr over Y "
            if self.codec != None:
                self.codec_comress(self.home_dir, self.codec)
                self.Read_frames()
            subprocess.run(shlex.split(vqmt_opt))
            az = pd.read_csv(self.home_dir + "VMAF_METRIX/csv_res/" + sub_str + 'name.csv')
            arr_str = ((az[3:4].to_numpy()[0][1:]).astype('float'))
            SSIM = (arr_str[0] + arr_str[0] + arr_str[0]) / 3
            
            PSNR = arr_str[-1]
            fixed = arr_str[-1]
            orig = arr_str[-1]
            if first_one:
                first_one = False
                #saved_N  = fixed
                #saved_O = orig
            #Result.append(saved_O)
            Result.append(fixed)
            #Result.append(saved_N)
    
            #esult.append(fixed - saved_N)
            
            #print(Result)
            ##Result.append(str(self.func))
            ##self.Results.append(Result)
            
        Aq = np.zeros(len(args))
        #print(Aq)#
        frames_begin = 11 + int(self.calc_SSIM_PSNR)
        if self.lot_of_frames:
            for p in range(1,10):#dropping 0 due to first frame anomaly 
                #print(az["Netflix VMAF_VMAF061neg"][11:].to_numpy(dtype = 'float')[p::10][:len(args)])
                #print(Aq)
                Aq += az["Netflix VMAF_VMAF063"][frames_begin:].to_numpy(dtype = 'float')[p::10][:len(args)]
            Aq = Aq / 9
        else:
            Aq[0] = np.mean([float(i) for i in az["Netflix VMAF_VMAF063"][12:]])
        
        for i in zip(args,Aq):
            Result = [el] + list(i[0]) + [i[1]]
            self.Results.append(Result)
            
        #print(self.per_frame_args, az["Netflix VMAF_VMAF061neg"][11:].to_numpy(dtype = 'float'))
        if self.per_frame_args != []:
            for idx,i,j in zip(np.arange(len(args)*10) ,self.per_frame_args,az["Netflix VMAF_VMAF063"][frames_begin:].to_numpy(dtype = 'float')):#including 0-frame
                cur_global_stats = [el,idx%10, i, j, args[idx//10] ]
                for met_name in az.keys()[1:]:
                    cur_global_stats.append(az[met_name][frames_begin:].to_numpy(dtype = 'float')[idx])
                self.global_sats.append(cur_global_stats)
        else:
            for idx,j in zip(np.arange(len(args)*10) ,az["Netflix VMAF_VMAF063"][frames_begin:].to_numpy(dtype = 'float')):#including 0-frame
                cur_global_stats = [el,idx%10, j, args[idx//10] ]
                for met_name in az.keys()[1:]:
                    cur_global_stats.append(az[met_name][frames_begin:].to_numpy(dtype = 'float')[idx])
                self.global_sats.append(cur_global_stats)
        return Aq
        
    def get_ssim_vqmt(self):
        vqmt_opt = "vqmt -quiet -orig " + self.home_dir + "GT.Y4M -in " + self.home_dir +  "0YES.Y4M" + " -csv -csv-dir " + self.home_dir + "VMAF_METRIX/csv_res -cng CUSTOM "+ "0name -metr ssim over Y -dev OpenCL0 " #+ self.model + " "
        subprocess.run(shlex.split(vqmt_opt))
        az = pd.read_csv(self.home_dir + "VMAF_METRIX/csv_res/" + '0name.csv')
        return float(az['SSIM OpenCL0'][3])
    
    def PSNR_metrix_get(self):
        psnr = 0
        for i,j in zip(self.dataset_err, self.dataset_np):
            err = mean_squared_error(np.clip(i/255.,0,1), np.clip(j/255.,0,1))
            if err == 0:
                err = 0.000001
            psnr += 10 * np.log10((1 ** 2) / err)
        psnr = psnr / len(self.dataset_err)
        return psnr
    
    def codec_comress(self, home_dir, codec, input_dir = None, compressed_dir = None, output_dir = None):
        if compressed_dir == None:
            compressed_dir = home_dir + "/VMAF_METRIX/vid/a.mp4"
        if output_dir == None:
            output_dir = home_dir + "/0YES.Y4M"
        if input_dir == None:
            input_dir = home_dir + "/0YES.Y4M"
        os.system("ffmpeg -hide_banner -loglevel error -y -i " + input_dir + " " + codec + "  -pix_fmt yuv420p " + compressed_dir)
        os.system("ffmpeg -hide_banner -loglevel error -y  -i " + compressed_dir + " " + "  -pix_fmt yuv420p " + " " + output_dir)
        
    def get_metrix(self, args):
        retval = []
        self.full_el = self.dataset_dir + self.dataset[0]     
        
        for i in args:
            metr1 = self.get_metrix2( [i,])[0]
            psnr_val = self.PSNR_metrix_get()
            retval.append((metr1, psnr_val))
        return (retval,)
    
    
def expose_cv2(frameGT, tilegridsize , cliplimit):#FOR VMAF
    """
    CV2 CPU expose with createCLAHE
    """
    err = np.uint8(np.copy(frameGT))
    
    ker_cv = cv2.createCLAHE(clipLimit = cliplimit, tileGridSize=( int(tilegridsize), int(tilegridsize)))
    err[:,:,0] = ker_cv.apply(err[:,:,0])
    err[:,:,1] = ker_cv.apply(err[:,:,1])
    err[:,:,2] = ker_cv.apply(err[:,:,2])
    return np.float32(err) / 1. # / 255.

args_d_CLAHE = {'n':2,'a0_min': 1 ,'a0_max':100, 'a1_min' : 0.00001, 'a1_max' : 20. , 'type0': 'int','type1': 'float'}


def tonemapDrago(img, arg1,arg2,arg3):#[0,1]bst 0,7,0.9,
    # bias 0,1
    #sturation 0,1
    return (cv2.createTonemapDrago(arg1,arg2,arg3).process(img.astype("float32")/255.)*255).astype("uint8")
args_tonemapDrago = {'n':3,'a0_min': 0. ,'a0_max': 2.5, 
          'a1_min' : 0., 'a1_max' : 3. , 
          'a2_min' : 0.0, 'a2_max' : 1. , 
          }

def aug4const(img,arg1, sigma, amount):
    """
    gamma+unsharp_mask
    arg1,arg2 -- gamma limits arg2 > arg1
    sigma, ammout -- unsharp_mask  parameters
    Return value: transformed framse
    """
    arg1 = int(arg1)
    sigma = int(sigma)

        
    light = A.Compose([
    A.RandomGamma(p=1,gamma_limit=(arg1, arg1),always_apply=True),
    ], p=1)
    image = light(image = img)['image']
    
    kernel_size = (17, 17)
    
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = cv2.addWeighted(image, (amount + 1.), blurred, - (amount), 0)
    sharpened = np.minimum(sharpened,255)
    sharpened = np.maximum(sharpened,0)
    sharpened = sharpened.round().astype(np.uint8)
    sharpened = sharpened.astype(np.uint8)
        
    return sharpened
args_d_aug4const = {'n':3,'a0_min': 30 ,'a0_max':120, 'a1_min' : 30, 'a1_max' : 120, 'a2_min' : 0., 'a2_max' : 2.0, 'type0': 'int','type1': 'int','type2': 'float',}

