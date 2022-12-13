import os
import shutil
import cv2
from PIL import Image
from datetime import datetime
import time
import numpy as np
import random

bases = ['detection-june','detection-nov']
bases2 = ['detection-nov']
basedir = 'detection-nov'
base = os.listdir(basedir)
skipsave = True
debug = True

def folderRemover(path):
    try:
        shutil.rmtree(path)
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))

folderRemover("Data")
folderRemover("PigRoboflow")

time.sleep(1)

os.makedirs(os.path.join("Data","HeatStress"))
os.makedirs(os.path.join("Data","Normal"))
os.makedirs("PigRoboflow")

def randomizer( img, targMin, targMax , min, max):
    r, c, _ = img.shape
    for y in range(r):
        for x in range(c):
            foc = img[y,x,0]
            if foc >= targMin and foc <= targMax:
                tempgen = random.randint(min, max)
                img[y][x] = (tempgen, tempgen, tempgen)

    return img

def reference(BS):
    dataname = 0
    for BASE in BS:
        bs = os.listdir(BASE)
        for f in bs:
            pth2 = os.path.join(BASE,f,'Target')
            lists = os.listdir(pth2)
            for foc in lists:
                lastpath = os.path.join(pth2, foc)
                if "unprocessed" in foc :
                    stmp = time.time()
                    if(dataname == 5 and debug): return
                    img = cv2.imread(lastpath)
                    print(f'Moving->{BASE} Reference/{stmp}.jpg -> {np.average(img)}', end='\r ')
                    if skipsave : continue
                    cv2.imwrite(f'Reference/{stmp}.jpg', img)
                    dataname += 1

def HeatStress(BS):
    dataname = 0
    for BASE in BS:
        bs = os.listdir(BASE)
        for f in bs:
            pth2 = os.path.join(BASE,f,'Target')
            lists = os.listdir(pth2)
            for foc in lists:
                lastpath = os.path.join(pth2, foc)
                if "unprocessed" in foc :
                    stmp = time.time()
                    if(dataname == 5 and debug): return
                    img = cv2.imread(lastpath)
                    img = cv2.resize(img, (10,20))
                    bfravg = np.average(img)
                    img = randomizer(img, 33, 40, 36, 43)
                    print(img)
                    # print(f'Moving->{BASE} Data/HeatStress/{stmp}.jpg -> {bfravg} : {np.average(img)}', end='\r ')
                    if skipsave : continue
                    cv2.imwrite(f'Data/HeatStress/{stmp}.jpg', img)
                    dataname += 1

def Normal(BS):
    dataname = 0
    for BASE in BS:
        bs = os.listdir(BASE)
        for f in bs:
            pth2 = os.path.join(BASE,f,'Target')
            lists = os.listdir(pth2)
            for foc in lists:
                lastpath = os.path.join(pth2, foc)
                if "unprocessed" in foc :
                    stmp = time.time()
                    if(dataname == 5 and debug): return
                    img = cv2.imread(lastpath)
                    img = cv2.resize(img, (10,20))
                    bfravg = np.average(img)
                    img = randomizer(img, 18, 33, 18, 36)
                    print(img)
                    # print(f'Moving->{BASE} Data/Normal/{stmp}.jpg -> {bfravg} : {np.average(img)}', end='\r ')
                    if skipsave : continue
                    cv2.imwrite(f'Data/Normal/{stmp}.jpg', img)
                    dataname += 1

def YoloDataGen(BS):
    dataname = 0
    for BASE in BS:
        bs = os.listdir(BASE)
        for f in bs:
            imgPth = os.path.join(BASE, f, 'img_normal.png')
            img = cv2.imread(imgPth)
            if(img is None): continue
            stmp = time.time()
            print(f"Moving->{BASE} PigRoboflow/{stmp}.png", end='\r     ')
            if skipsave : continue
            cv2.imwrite(f'PigRoboflow/{stmp}.png', img)
            dataname += 1

# reference()
print("Pass 1 Heat")
HeatStress(bases2);
print("Pass 1 Norm")
Normal(bases2);
