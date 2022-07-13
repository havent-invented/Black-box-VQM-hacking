from tqdm.notebook import tqdm
from deap import *

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
from hackvmaf import *
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
count = 0
prev = ''
cdir = "/home/max/driveC/DeadlineStuff/VMAF_METRIX/dataset/"
dst_dir = "/home/max/driveE/VMAF_METRIX/dataset/"


home_dir = "/home/max/Ram/conv/"
#os.mkdir("/home/max/Ram/Gamma135")
#envtmp = calc_met(home_dir1=home_dir)
#envtmp.crf_arr = [32,32,40,32,32]
#print(envtmp.get_metrix([[1,3]]))


os.system('mkdir ' +home_dir)
os.system('mkdir ' +home_dir +  'VMAF_METRIX')
os.system('mkdir ' +home_dir +  'VMAF_METRIX/csv_res')
os.system('mkdir ' +home_dir +  'VMAF_METRIX/vid')

class calc_met:# 
    def __init__(self,dataset1 = ["Run439.Y4M"], func1 = Identity, convKer1 = None, home_dir1 = "R:/", creat_dir = False, calc_SSIM_PSNR = False, calc_model_features = False, model = "vmaf_v063" , codec = '   -preset:v medium -x265-params log-level=error ',dataset_dir = "dataset/"):
        "Init env"
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
            #shutil.copyfile(self.full_el,  self.home_dir + "GT.Y4M")
            first_one = False
    
                     
            self.datagen = [frameGT for frameGT in skvideo.io.FFmpegReader( self.home_dir + "GT.Y4M", outputdict={"-c:v" :" rawvideo","-f": "rawvideo"}).nextFrame()]
            
            Result = [el, i, j]
            ret = 1
            self.per_frame_args = []
            retGT = 1
            L2 = 0
            countL2 = 0
            for idx, iarg in enumerate(args):
                out1 = skvideo.io.FFmpegWriter( self.home_dir +  sub_str + "YES.Y4M",inputdict = {"-pix_fmt": "rgb24"}, outputdict = {"-pix_fmt": "yuv420p"})#,inputdict= {"-pix_fmt": "yuv420p"}
                for frameGT in self.datagen:
                    self.err =  self.func(frameGT, *iarg)#GPU_expose_cv2
                    out1.writeFrame(self.err)
                
                out1.close()#-loglevel panic
                      
                            #os.system("ffmpeg -hide_banner -loglevel error -y -i " + self.home_dir +  sub_str + "YES.Y4M " + self.codec + "  -pix_fmt yuv420p " + self.home_dir + 'VMAF_METRIX/vid/' + el + "_" +str(iarg[0])+ "_" + str(iarg[1]) + ".mp4" )
                      #os.system("ffmpeg -hide_banner -loglevel error -y -i " + self.home_dir  + 'VMAF_METRIX/vid/' + el + "_" +str(iarg[0])+ "_" + str(iarg[1])+ ".mp4" + " -pix_fmt yuv420p  " + self.home_dir +  str(idx)+"0temp_optN.Y4M")
                      #self.tmp_bitrate_dir = self.home_dir + 'VMAF_METRIX/vid/' + el + "_" +str(iarg[0])+ "_" + str(iarg[1]) + ".mp4"
                  #print("ffmpeg -hide_banner -loglevel error -y  -i " + self.home_dir +  sub_str + "YES.Y4M " + " -vcodec h264_nvenc  -b:v 3M -preset:v medium  -forced-idr 1  -force_ key_frames 10 -keyint_min 10  -pix_fmt yuv420p " + self.home_dir +  sub_str + "YES.h264")
                  
                  
                  
                  #for idx in range(len(args)):
                  #    if idx == 0:
                  #        os.system("echo file '"+ str(idx)+"0temp_optN.Y4M' > " + self.home_dir + "list.txt")
                  #    else:
                  #        os.system("echo file '"+ str(idx)+"0temp_optN.Y4M' >> " + self.home_dir + "list.txt")
                  
                  #os.system("ffmpeg -hide_banner -loglevel error -y -f concat -i " + self.home_dir + "list.txt -c:a copy " + self.home_dir + "ab.Y4M")    
                  
            
            
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
            
            subprocess.run(shlex.split(vqmt_opt))
            #subprocess.run(shlex.split("vqmt -slots 8 -metric-parallelism 8 -threads 8 -quiet -orig R:\GT.Y4M -in " + self.home_dir + "temp_optN.Y4M -csv -csv-dir " + self.home_dir + "VMAF_METRIX/csv_res -cng CUSTOM name -metr ssim over Y,U,V -dev OpenCL0 -metr vmaf over Y -dev OpenCL0 -set model_preset=vmaf_v061_neg  -metr vmaf over Y -dev OpenCL0 -set model_preset=vmaf_v063 -metr psnr over YUV"))
            
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
        
        for p in range(1,10):#dropping 0 due to first frame anomaly 
            #print(az["Netflix VMAF_VMAF061neg"][11:].to_numpy(dtype = 'float')[p::10][:len(args)])
            #print(Aq)
            Aq += az["Netflix VMAF_VMAF063"][frames_begin:].to_numpy(dtype = 'float')[p::10][:len(args)]
        Aq = Aq / 9
        
        
        #print(Aq)    
    
            
        
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
    
    def get_metrix(self, args):
        retval = []
        self.full_el = self.dataset_dir + self.dataset[0]     
        shutil.copyfile(self.full_el,  self.home_dir + "GT.Y4M")
        for i in args:
            retval.append(*self.get_metrix2( [i,]))

            
        return np.array(retval)
Curr_metrix = 0

convKer_default=np.array([[0,0,0],
         [0,1.,0],
         [0,0,0]])

#convKer = np.random.randn(3,3)
#postimg=cv2.filter2D(frame,-1,convKer)
def fker(img,arg1 = 0 ,arg2 = 0):
    global convKer
    return cv2.filter2D(img,-1,convKer)

AV_log = []



for vid in tqdm(os.listdir(dst_dir)):
    try:
        #datagen = [frameGT for frameGT in skvideo.io.FFmpegReader("R:/GT.Y4M", outputdict={"-c:v" :" rawvideo","-f": "rawvideo"}).nextFrame()]
        env = calc_met( model = "vmaf_v063", home_dir1=home_dir,dataset_dir=dst_dir)
        top_metrix = 0
        delta = 0.1
        env.dataset = [vid]
        lamda = 0.1
        saved = np.copy(convKer_default)
        convKer = np.copy(convKer_default)
        cross = 0
        for epoch in range(15):
            env.func = fker
            saved_metrix = env.get_metrix([[1]])#pipe(['Mystery.Y4M'],  [62], [77], aug)
            print("epoch:",epoch,"metrix:",saved_metrix )
            convKer = np.copy(saved)
            grad_arr = np.zeros_like(convKer)
            for i in range(convKer.shape[0]):
                for j in range(convKer.shape[1]):
                    convKer = np.copy(saved)
                    convKer[i,j] += delta
                    env.func = fker
                    new_metrix = env.get_metrix([[1]])
                    #print(new_metrix, new_metrix-saved_metrix)
                    grad_arr[i,j] = new_metrix - saved_metrix
            if(np.linalg.norm(grad_arr) > lamda):
                grad_arr = grad_arr / np.linalg.norm(grad_arr)
            saved += grad_arr * lamda #/ max(np.sum(np.sum(grad_arr)), 0.000001)
            if(cross > saved_metrix):
                delta = delta/10
                lamda = lamda / 10
            cross = saved_metrix
            top_metrix = max(saved_metrix,top_metrix)
        print("Result:",top_metrix,saved)
        AV_log.append([saved,top_metrix])
    except Exception:
            print("exception")
            raise