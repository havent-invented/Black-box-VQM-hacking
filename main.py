#CLAHE Preprocessing + PSNR + VMAF
import sys
sys.path.insert(1, '..')
import torch
#from torchvision import transforms
import skvideo.io
from PIL import Image
import numpy as np
from CNNfeatures import get_features
from VQAmodel import VQAModel
from argparse import ArgumentParser
import time
from PIL import Image
import torch
import numpy as np
#home_dir = "/home/max/Ram/tonemap/"
#prev = ''
#cdir = "/home/max/driveE/VMAF_METRIX/CVPR_results/dataset_CC_10/"
#dst_dir = "/home/max/driveE/VMAF_METRIX/CVPR_results/dataset_CC_10/"
#
#home_dir = "/home/max/Ram/tonemap/"
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
import random

def Identity(img,arg1,arg2):
    return img

def axis_swaper(func):
    def wrapper(*args):
        args = np.swapaxes(args[0],0 , -1), *args[1:]
        return np.swapaxes(func(*args),0, -1)
    return wrapper

from skimage.metrics import mean_squared_error

class calc_met:#    
    def __init__(self,dataset1 = ["Run439.Y4M"], func1 = Identity, convKer1 = None, home_dir1 = "R:/", creat_dir = False, calc_SSIM_PSNR = False, calc_model_features = False, model = "vmaf_v063" , codec = None,dataset_dir = "dataset/"):
        "Init env"
        self.single_frame = True
        self.one_batch = True
        self.device = "cuda:0"
        self.model = VQAModel().to(device)
        self.model.load_state_dict(torch.load('../models/MDTVSFA.pt'))
        self.model.eval()
        self.frame_batch_size = 1
        self.dataset_err = None
        self.dataset_err_torch = None
        self.dataset_np = None
        self.dataset_torch = None
        self.datagen = None
        self.dataset = []
        self.crf_arr = []
        self.dataset_dir = dataset_dir
        self.home_dir = home_dir1
        self.codec = codec
        self.calc_model_features = calc_model_features
        self.calc_SSIM_PSNR = calc_SSIM_PSNR
        self.global_sats = []
        self.Results = []
        self.metr = None
        self.relative_score, self.mapped_score, self.aligned_score = 0,0,0
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
        
    def MDTVSFA(self, transformed_video):
        self.features = get_features(transformed_video / 255., frame_batch_size=self.frame_batch_size, device=self.device)
        self.features = torch.unsqueeze(self.features, 0) 

        if len(self.features.shape) == 2:
            self.features = self.features.unsqueeze(0)
        with torch.no_grad():
            input_length = self.features.shape[1] * torch.ones(1, 1, dtype=torch.long)
            self.relative_score, self.mapped_score, self.aligned_score = self.model([(self.features, input_length, ['K'])])
            y_pred = self.mapped_score[0][0].to('cpu').numpy()
        return y_pred
    
    def SSIM_metrix_get(self):
        psnr = 0
        from IQA_pytorch import SSIM
        ssim_iqa = SSIM()
        ssim_val = ssim_iqa(self.dataset_err_torch, self.dataset_torch, as_loss=False)
        psnr += ssim_val
        psnr = psnr / len(self.dataset_err)
        return psnr.cpu().numpy()[0]
    
    def PSNR_metrix_get(self):
        psnr = 0
        for i,j in zip(self.dataset_err, self.dataset_np):
            err = mean_squared_error(np.clip(i/255.,0,1), np.clip(j/255.,0,1))
            if err == 0:
                err = 0.000001
            psnr += 10 * np.log10((1 ** 2) / err)
        psnr = psnr / len(self.dataset_err)
        return psnr

    def init_video(self):
        self.datagen = [frameGT for frameGT in skvideo.io.FFmpegReader(self.dataset_dir + self.dataset[0], outputdict={"-c:v" :" rawvideo","-f": "rawvideo"}).nextFrame()]
        self.device = "cuda:0"
        if self.single_frame:
            self.datagen  = self.datagen[:1]
            self.dataset_np = np.expand_dims(np.array([np.array(image)[...,i] for i in range(3) for image in self.datagen]), 0) / 1.
        else:
            self.dataset_np = np.array(self.datagen).swapaxes(-1,-2).swapaxes(-2,-3) / 1.
        self.dataset_torch = torch.tensor(self.dataset_np , dtype = torch.float32).to(self.device) 
        
    def Write_frames(self):
        out1 = skvideo.io.FFmpegWriter( self.home_dir + "0YES.Y4M",inputdict = {"-pix_fmt": "rgb24"}, outputdict = {"-pix_fmt": "yuv420p"})
        for frameGT in self.dataset_err:
            out1.writeFrame(frameGT.swapaxes(-2, -3).swapaxes(-1,-2))
        out1.close()
        
    def Read_frames(self):
        self.dataset_err = [frameGT.swapaxes(-1,-2).swapaxes(-2,-3) for frameGT in skvideo.io.FFmpegReader( self.home_dir + "0YES.Y4M", outputdict={"-c:v" :" rawvideo","-f": "rawvideo"}).nextFrame()]
        
    def codec_comress(self, home_dir, codec, input_dir = None, compressed_dir = None, output_dir = None):
        if compressed_dir == None:
            compressed_dir = home_dir + "/VMAF_METRIX/vid/a.mp4"
        if output_dir == None:
            output_dir = home_dir + "/0YES.Y4M"
        if input_dir == None:
            input_dir = home_dir + "/0YES.Y4M"
        os.system("ffmpeg -hide_banner -loglevel error -y -i " + input_dir + " " + codec + "  -pix_fmt yuv420p " + compressed_dir)
        os.system("ffmpeg -hide_banner -loglevel error -y  -i " + compressed_dir + " " + "  -pix_fmt yuv420p " + " " + output_dir)
        
    def get_metrix2(self, args):

        '''
        args is a list of args. [[arg_for_first], [arg_for_second],...]
        Return value:
        WARNING: if one_batch == False, Proxy metrics is callculated on the last frame only
        '''
        ret = 1
        self.per_frame_args = []
        retGT = 1
        L2 = 0
        countL2 = 0
        Aq = np.zeros(len(args))
        for idx, iarg in enumerate(args):
            self.dataset_err = [self.func(frameGT, *iarg) for frameGT in self.dataset_np]
            if self.codec != None:
                self.Write_frames()
                self.codec_comress(self.home_dir, self.codec)
                self.Read_frames()
            if self.one_batch:
                for j in range(len(self.dataset_err)):
                    self.dataset_err_torch = torch.tensor(np.array([self.dataset_err[j]]), dtype = torch.float32).to(self.device)
                    self.err = self.metr(self.dataset_err_torch)
                    Aq[idx] = self.err
            else:
                self.dataset_err_torch = torch.tensor(np.array(self.dataset_err), dtype = torch.float32).to(self.device)
                self.err = self.metr(self.dataset_err_torch)
                Aq[idx] = self.err
        return Aq

    def get_metrix(self, args):
        retval = []
        self.full_el = self.dataset_dir + self.dataset[0]     
        for i in args:
            metr1 = np.mean(self.get_metrix2( [i,]))
            psnr_val = self.PSNR_metrix_get()
            retval.append((metr1, psnr_val))
        return (retval,)
    
    
