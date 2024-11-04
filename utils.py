import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from time import time
import os
import matplotlib.pyplot as plt
from torchvision.io.image import read_image
from torchvision.transforms import Compose, Resize, CenterCrop, RandomCrop,ToTensor, Normalize, Grayscale, RandomRotation
from torchvision.transforms.v2 import  RandomResize

from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC # resize the input image using bicubic interpolation, producing a smoother result compared to other interpolation methods like nearest-neighbor or bilinear.

def _transfRRC3(pad,rot,nx): # applied to a PIL image
    return Compose([
        RandomResize(nx-pad,nx+pad,interpolation=InterpolationMode.BILINEAR),
        RandomRotation(rot),
        RandomCrop(nx,pad,padding_mode='edge'),
        #ColorJitter(),
        #Resize([nx2,nx2], interpolation=InterpolationMode.BILINEAR),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ])

def _transfRRC4(pad,rot,nx): # applied to a PIL image
    return Compose([
        RandomResize(nx-pad,nx+pad,interpolation=InterpolationMode.BILINEAR),
        RandomRotation(rot),
        RandomCrop(nx,pad,padding_mode='edge'),
        #ColorJitter(),
        #Resize([nx2,nx2], interpolation=InterpolationMode.BILINEAR),
        Normalize(mean=[0.485, 0.456, 0.406, 0.0], std=[0.229, 0.224, 0.225, 1.0])
     ])

def _transfRRCR4(nx,pad,nx2): # applied to a PIL image
    return Compose([
        RandomResize(nx-pad,nx+pad,interpolation=InterpolationMode.BILINEAR),
        RandomRotation(pad),
        RandomCrop(nx,pad,padding_mode='edge'),
        #ColorJitter(),
        Resize([nx2,nx2], interpolation=InterpolationMode.BILINEAR),
        Normalize(mean=[0.485, 0.456, 0.406, 0.0], std=[0.229, 0.224, 0.225, 1.0])
     ])

def normalize(): 
    return Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def resize(nx2): 
    return Resize([nx2,nx2], interpolation=InterpolationMode.BILINEAR)

def load_images(path, names, device, ext='jpg'):
    n=len(names)
    images = []
    sz=[]
    for i in range(n):
        name=path+'/'+names[i]+'.'+ext
        rgb_img = read_image(name).to(device)
        images.append(rgb_img)
    
    images=torch.cat(images,dim=0)
    return images

def iou1(a,b):
    """
    Function to get the IOU between two 1D-tensors
    """
    io=torch.sum(a*b)/(torch.sum(torch.max(a,b))+0.00001)
    return io.item()

def iou(a,b):
    """
    Function to get the IOU between two 2D-tensors
    """
    return torch.sum(a*b,1)/(torch.sum(torch.max(a,b),1)+0.00001)

def iouLoss(a,b,sigma=1):
    """
    Finds the IOU between img_tensors a and b
    """
    ioui=torch.sum(a*b)/torch.sum(torch.max(a,b))
    return ioui

def load_PCA(name,device,nc=64):
    m=torch.load(name)
    P=m['P']  # list of PC-tensors, each representing one mask: contains PC npa patches ------>each element of size (256,128*128)
    mx=m['mx'] # running average list

    for i in range(len(mx)):
        mx[i]=mx[i].to(device)
    for i in range(len(P)):
        P[i]=P[i][0:nc,:].to(device)
    return mx,P

def load_1PCA(name,device,nc=64):
    m=torch.load(name)
    P=m['P']  # list of PC-tensors, each representing one mask: contains PC npa patches ------>each element of size (256,128*128)
    mx=m['mx'].to(device) # running average list
    P=P[0:nc,:].to(device)
    return mx,P

def make_divisible(img,nx=32):
    # make image size divisible by nx
    _, h, w = img.shape
    nr,nc=int(np.ceil(h/nx)),int(np.ceil(w/nx))
    newh,neww=nr*nx,nc*nx
    padding=(0, neww-w, 0, newh-h)
    padimg=Fn.pad(img, padding, mode='constant')
    return padimg, padding 

def make_square(img):
    # make the image square by padding
    # also return the cropping information
    _, h, w = img.shape
    size = max(h, w)
    difw = size-w
    difh = size-h
    x0,y0=difw//2,difh//2
    padding=(x0, difw-x0,y0, difh-y0)
    padimg=Fn.pad(img, padding, mode='constant')
    crop=(y0,y0+h,x0,x0+w)
    return padimg,crop 

def get_edge(mask):
    nx,ny=mask.shape
    out=torch.zeros(nx,ny)
    for x in range(nx):
        for y in range(ny):
            if mask[x,y] == 0:
                continue
            neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
            for neighbor_x, neighbor_y in neighbors:
                if (0 <= neighbor_x < nx
                    and 0 <= neighbor_y < ny
                    and mask[neighbor_x, neighbor_y] == 0):
                    out[x, y]=1
                    break
    return out

def plot_edge_overlay(img,mask,val):
    a=img.permute(1,2,0).cpu().clone().view(-1,3)
    e=get_edge(mask.cpu()).view(-1)
    a[e>0,0]=val
    a[e>0,1]=val
    a[e>0,2]=0.
    plt.imshow(a.reshape(mask.shape[0],mask.shape[1],3))
    plt.show()

def get_nov_patches(x,nx):   
    # get non-overlapping patches of size nx x nx
    n=x.shape[0]
    nr=x.shape[1]//nx
    nc=x.shape[2]//nx
    pa=torch.zeros(n,nr,nc,nx**2,device=x.device)
    for r in range(0,nr):
        for c in range(0,nc):
            p=x[:,r*nx:r*nx+nx,c*nx:c*nx+nx]
            pa[:,r,c,:]=p.reshape(-1,nx**2)
    return pa

def set_nov_patches(xr,nx):
    n=xr.shape[0] # number of images
    nr=xr.shape[1] # number of rows
    nc=xr.shape[2] # number of cols

    xr=xr.reshape(xr.shape[0],xr.shape[1],xr.shape[2],nx,nx)
    x=torch.zeros(n,nr*nx,nc*nx,device=xr.device)
    for r in range(nr):
        for c in range(nc): # controls the patch
            x[:,r*nx:r*nx+nx,c*nx:c*nx+nx]=xr[:,r,c,:,:] 
    return x
