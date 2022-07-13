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



def get_metrix(args):

    """
    args is a list of args. [[arg_for_first], [arg_for_second],...]
    Return value:
    """
    i,j = 1,1
    global func
    global dataset
    global home_dir
    global err
    global Results
    saved_N = 0
    saved_O = 0
    for el in dataset:    
        full_el = "dataset/" + el 
        sub_str = "0"#el + "_" + str(i) + "_" + str(j)
        res_str =  el + "_" + str(i) + "_" + str(j) + ".csv"
        shutil.copyfile(full_el,  home_dir + "GT.Y4M")
        first_one = False

        os.system("echo file 'GT.Y4M' >  " + home_dir + "list.txt")
        for i in range(len(args)-1):
            #!echo file 'v.Y4M' >> " + home_dir + "list.txt
            os.system("echo file 'GT.Y4M' >> " + home_dir + "list.txt")

        os.system("ffmpeg -hide_banner -loglevel error -y -f concat -i " + home_dir + "list.txt -c:a copy " + home_dir + "a.Y4M")
                    
        datagen = [frameGT for frameGT in skvideo.io.FFmpegReader( home_dir + "GT.Y4M", outputdict={"-c:v" :" rawvideo","-f": "rawvideo"}).nextFrame()]
        
        Result = [el, i, j]
        ret = 1
        
        retGT = 1
        L2 = 0
        countL2 = 0
        for idx, iarg in enumerate(args):
            out1 = skvideo.io.FFmpegWriter( home_dir +  sub_str + "YES.Y4M ",inputdict = {"-pix_fmt": "rgb24"}, outputdict = {"-pix_fmt": "yuv420p"})#,inputdict= {"-pix_fmt": "yuv420p"}
            for frameGT in datagen:
                err =  func(frameGT, *iarg)#GPU_expose_cv2
                out1.writeFrame(err)
            
            out1.close()
            os.system("ffmpeg -hide_banner -loglevel error -y -i " + home_dir +  sub_str + "YES.Y4M " + " -vcodec h264_nvenc   -preset:v medium  -forced-idr 1  -force_key_frames 14 -keyint_min 4  -pix_fmt yuv420p " + home_dir +  sub_str + "YES.h264")
            os.system("ffmpeg -hide_banner -loglevel error -y -i " + home_dir +  sub_str + "YES.h264 -pix_fmt yuv420p  " + home_dir +  str(idx)+"0temp_optN.Y4M")
        #print("ffmpeg -hide_banner -loglevel error -y  -i " + home_dir +  sub_str + "YES.Y4M " + " -vcodec h264_nvenc  -b:v 3M -preset:v medium  -forced-idr 1  -force_ key_frames 10 -keyint_min 10  -pix_fmt yuv420p " + home_dir +  sub_str + "YES.h264")
        
        
        
        for idx in range(len(args)):
            if idx == 0:
                os.system("echo file '"+ str(idx)+"0temp_optN.Y4M' > " + home_dir + "list.txt")
            else:
                os.system("echo file '"+ str(idx)+"0temp_optN.Y4M' >> " + home_dir + "list.txt")
        os.system("ffmpeg -hide_banner -loglevel error -y -f concat -i " + home_dir + "list.txt -c:a copy " + home_dir + "ab.Y4M")    
        
        
        
        
        subprocess.run(shlex.split("vqmt -quiet -slots 8 -orig " + home_dir + "a.Y4M -in " + home_dir + "ab.Y4M  -csv -csv-dir " + home_dir + "VMAF_METRIX/csv_res -cng CUSTOM " + sub_str + "name -metr vmaf over Y -dev OpenCL0 -set model_preset=vmaf_v061_neg"))
        #subprocess.run(shlex.split("vqmt -slots 8 -metric-parallelism 8 -threads 8 -quiet -orig R:\GT.Y4M -in " + home_dir + "temp_optN.Y4M -csv -csv-dir " + home_dir + "VMAF_METRIX/csv_res -cng CUSTOM name -metr ssim over Y,U,V -dev OpenCL0 -metr vmaf over Y -dev OpenCL0 -set model_preset=vmaf_v061_neg  -metr vmaf over Y -dev OpenCL0 -set model_preset=vmaf_v063 -metr psnr over YUV"))
        
        az = pd.read_csv( home_dir + "VMAF_METRIX/csv_res/" + sub_str + 'name.csv')
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
        ##Result.append(str(func))
        ##Results.append(Result)
        
    Aq = np.zeros(len(args))
    for p in range(10):
        Aq += az["Netflix VMAF_VMAF061neg"][11:].to_numpy(dtype = 'float')[p::10][:len(args)]
    Aq = Aq / 10
    
    for i in zip(args,Aq):
        Result = [el] + list(i[0]) + [i[1]]
        Results.append(Result)
        
    
    return Aq
    
def aug4(img,arg1 = 40,arg2 = 60, sigma = 1, amount = 1.0):
    """
    gamma+unsharp_mask
    arg1,arg2 -- gamma limits arg2 > arg1
    sigma, ammout -- unsharp_mask  parameters
    Return value: transformed framse
    """
    arg1 = int(arg1)
    arg2 = int(arg2)
    sigma = int(sigma)
    if(arg1 > arg2):
        arg2 = arg1 + 15
        
    light = A.Compose([
    #A.RandomBrightnessContrast(p=1,always_apply=True),    #NO
    A.RandomGamma(p=1,gamma_limit=(arg1, arg2),always_apply=True),
    #A.CLAHE(p=1),    
    ], p=1)
    
    #global hey
    #try:
    #    hey = light(image = img)
    #except Exception:
    #    print(img,arg1, arg2, sigma, amount, )
    #    raise
    image = light(image = img)['image']
    
    kernel_size = (17, 17)
    
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = cv2.addWeighted(image, (amount + 1.), blurred, - (amount), 0)
    sharpened = np.minimum(sharpened,255)
    sharpened = np.maximum(sharpened,0)
    sharpened = sharpened.round().astype(np.uint8)
    sharpened = sharpened.astype(np.uint8)
        
    return sharpened

def replaceUV(func):
    def wrapper(*args):
        gt_img = args[0]
        result = func(*args)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2YUV )
        gt_img  = cv2.cvtColor(gt_img, cv2.COLOR_RGB2YUV )
        result[:,:,1] = gt_img[:,:,1]
        result[:,:,2] = gt_img[:,:,2]
        result = cv2.cvtColor(result, cv2.COLOR_YUV2RGB )
        return result
    return wrapper

def unsharp_mask(image, kernel_size=(11, 11), sigma=1.0, amount=1.0, threshold=0):
    """
    CV2 CPU unsharp_mask
    """
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    #sharpened = (amount + 1.) * image - (amount) * blurred
    sharpened = cv2.addWeighted(image, (amount + 1.), blurred, - (amount), 0)
    #cv2.min(sharpened, 255,sharpened)
    #cv2.max(sharpened, 0,sharpened)
    sharpened = np.minimum(sharpened,255)
    sharpened = np.maximum(sharpened,0)
    sharpened = sharpened.round().astype(np.uint8)
    sharpened = sharpened.astype(np.uint8)
    
    #if threshold > 0:
        #low_contrast_mask = np.absolute(image - blurred) < threshold
        #np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def expose_cv2(frameGT, tilegridsize , cliplimit):
    """
    CV2 CPU expose with createCLAHE
    """
    err = frameGT
    ker_cv = cv2.createCLAHE(clipLimit = cliplimit, tileGridSize=( tilegridsize, tilegridsize))
    err[:,:,0] = ker_cv.apply(frameGT[:,:,0])
    err[:,:,1] = ker_cv.apply(frameGT[:,:,1])
    err[:,:,2] = ker_cv.apply(frameGT[:,:,2])
    return err

def GPU_expose_cv2(frameGT, tilegridsize , cliplimit):
    """
    Expose CV2 GPU with cuda.createCLAHE
    """
    err = frameGT
    fr = cv2.cuda_GpuMat()
    frv = cv2.cuda_GpuMat()
    cuda_ker =  cv2.cuda.createCLAHE(clipLimit=cliplimit, tileGridSize=(tilegridsize, tilegridsize))
    fr.upload(frameGT[:,:,0])
    v = cuda_ker.apply(fr,cv2.cuda_Stream.Null())
    err[:,:,0] = v.download()
    fr.upload(frameGT[:,:,1])
    v = cuda_ker.apply(fr,cv2.cuda_Stream.Null())
    err[:,:,1] = v.download()
    fr.upload(frameGT[:,:,2])
    v = cuda_ker.apply(fr,cv2.cuda_Stream.Null())
    err[:,:,2] = v.download()
    return err

def aug(img,arg1 = 0,arg2 = 0):
    """
    GAMMA
    arg1,arg2 -- clip limits arg2 > arg1
    """
    
    light = A.Compose([
    #A.RandomBrightnessContrast(p=1,always_apply=True),    #NO
    A.RandomGamma(p=1,gamma_limit=(arg1, arg2),always_apply=True),
    #A.CLAHE(p=1),    
    ], p=1)
    v = light(image = img)
    return v['image']
    

def fker(img,arg):
    """
    convKer should be provided in global scope
    arg -- i,j, delta for ker[i,j] += delta
    """
    global convKer
    i,j, delta = arg
    ker = np.copy(convKer)
    ker[i,j] += delta
    return cv2.filter2D(img,-1,ker)
    
def Identity(img,arg1 = 0,arg2 = 0,arg3 = 0):
    """
    Identity function 
    Returns imgage argument, other arguments are ignored
    Usefull
    """
    return img
    
def br_toch(img,arg1,arg2 = 0):
    """
    adjust_brightness_torchvision with factor of arg1
    arg2 is ignored    
    """
    return A.adjust_brightness_torchvision(img,arg1)
    
    
    
def Init_env(dataset1 = ["Park.Y4M"], func1 = Identity, convKer1 = None, home_dir1 = "R:/", creat_dir = False):
    "Init env"
    global dataset
    global Results
    global func
    global convKer
    global home_dir
    home_dir = home_dir1
    if convKer1 is None:
        convKer = np.array([[ 0.54744114, -0.03412791, -0.10635577],
       [ 0.0590166 ,  1.15195262,  0.05838338],
       [-0.15729069, -0.14071838,  0.21105842]])
    else:
        convKer = convKer1
    dataset = dataset1
    Results = []
    func = func1
    if creat_dir:
        try:
            if not os.path.exists(home_dir1):
                os.mkdir(home_dir1)
        except Exception:
            ignore
    try:
        if not os.path.exists(home_dir + "VMAF_METRIX/ "):
            os.mkdir(home_dir + "VMAF_METRIX/")
            os.mkdir(home_dir + "VMAF_METRIX/csv_res")
    except Exception:
        pass