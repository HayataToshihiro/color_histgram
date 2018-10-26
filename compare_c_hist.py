# -*- coding: UTF-8 -*-

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def decreseValue(value):
    if value<64:
        return 32
    elif value<128:
        return 96
    elif value<196:
        return 160
    else:
        return 224


def decreaseColor(img):
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            img[y,x,0] = decreseValue(img[y,x,0])
            img[y,x,1] = decreseValue(img[y,x,1])
            img[y,x,2] = decreseValue(img[y,x,2])

def rgb2bin(blue, green, red):
    blueNo = int(blue /64)
    greenNo = int(green /64)
    redNo = int(red /64)
    return 16*blueNo + 4*greenNo + redNo

def img2bin(img,IMG_SIZE):

    binimg = np.zeros((img.shape[0],img.shape[1]),np.uint8)
    #binimg = Image.new('L',IMG_SIZE)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            binimg[y,x] = rgb2bin(img[y,x,0],img[y,x,1],img[y,x,2])
    return binimg

def h0to1(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j,0]==0:
                img[i,j,0]=1

def plot_histgram(img):
    color = ('b', 'g', 'r')
    for channel, col in enumerate(color):
        histr = cv2.calcHist([img], [channel], None, [256], [1, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])

def compare_by_hist():
    TARGET_FILE = '/home/amsl/ピクチャ/img_1.png' 
    COMPARING_FILE1 =  '/home/amsl/ピクチャ/img_3.png'  
    # COMPARING_FILE2 =  '/home/amsl/ピクチャ/img_5.png' 
    COMPARING_FILE2 =  '/home/amsl/ピクチャ/img_2.png' 
    
    TARGET_MASK_FILE = '/home/amsl/ピクチャ/img_1_.png' 
    COMPARING_MASK_FILE1 =  '/home/amsl/ピクチャ/img_3_.png'  
    # COMPARING_MASK_FILE2 =  '/home/amsl/ピクチャ/img_5_.png' 
    COMPARING_MASK_FILE2 =  '/home/amsl/ピクチャ/img_2_.png' 
    IMG_SIZE = (100, 200)
    ranges_h = 100 
    ranges = 256
    ranges_bin = 64

    ######################################################################

    #比較するイメージファイルを読み込み、ヒストグラムを計算
    target_img_path = TARGET_FILE
    target_img = cv2.imread(target_img_path)
    #target_img = cv2.resize(target_img, IMG_SIZE)
    target_mask_img_path = TARGET_MASK_FILE
    target_mask_img = cv2.imread(target_mask_img_path,cv2.IMREAD_GRAYSCALE)
    #target_mask_img = cv2.resize(target_mask_img, IMG_SIZEtwise_not)
    #target_mask_img = cv2.bitwise_not(target_mask_img)
    
    #decreaseColor(target_img)
    #target_img_bin = img2bin(target_img,IMG_SIZE)
    target_img_hsv = cv2.cvtColor(target_img, cv2.COLOR_BGR2HSV)
    #h0to1(target_img_hsv)
    
    #target_hist = cv2.calcHist([target_img_hsv], [0], target_mask_img, [ranges_h], [0, ranges_h])
    target_hist = cv2.calcHist([target_img], [0], target_mask_img, [ranges], [0, ranges]) \
                + cv2.calcHist([target_img], [1], target_mask_img, [ranges], [0, ranges]) \
                + cv2.calcHist([target_img], [2], target_mask_img, [ranges], [0, ranges])
    #target_hist = cv2.calcHist([target_img_bin], [0], None, [ranges_bin], [0, ranges_bin])
    
    ######################################################################
    
    
    #比較されるイメージファイル1を読み込み、ヒストグラムを計算
    comparing_img_path1 = COMPARING_FILE1
    comparing_img1 = cv2.imread(comparing_img_path1)
    #comparing_img1 = cv2.resize(comparing_img1, IMG_SIZE)
    comparing_mask_img_path1 = COMPARING_MASK_FILE1
    comparing_mask_img1 = cv2.imread(comparing_mask_img_path1,cv2.IMREAD_GRAYSCALE)
    #comparing_mask_img1 = cv2.resize(comparing_mask_img1, IMG_SIZE)
    #comparing_mask_img1 = cv2.bitwise_not(comparing_mask_img1)

    #decreaseColor(comparing_img1)
    #comparing_img_bin1 = img2bin(comparing_img1,IMG_SIZE)
    comparing_img1_hsv = cv2.cvtColor(comparing_img1, cv2.COLOR_BGR2HSV)
    #h0to1(comparing_img1_hsv)
    
    #comparing_hist1 = cv2.calcHist([comparing_img1_hsv], [0], comparing_mask_img1, [ranges_h], [0, ranges_h])
    comparing_hist1 = cv2.calcHist([comparing_img1], [0], comparing_mask_img1, [ranges], [0, ranges]) \
                    + cv2.calcHist([comparing_img1], [1], comparing_mask_img1, [ranges], [0, ranges]) \
                    + cv2.calcHist([comparing_img1], [2], comparing_mask_img1, [ranges], [0, ranges])
    #comparing_hist1 = cv2.calcHist([comparing_img_bin1], [0], None, [ranges_bin], [0, ranges_bin])

    ######################################################################


    #比較されるイメージファイル2を読み込み、ヒストグラムを計算
    comparing_img_path2 = COMPARING_FILE2
    comparing_img2 = cv2.imread(comparing_img_path2)
    #comparing_img2 = cv2.resize(comparing_img2, IMG_SIZE)
    comparing_mask_img_path2 = COMPARING_MASK_FILE2
    comparing_mask_img2 = cv2.imread(comparing_mask_img_path2,cv2.IMREAD_GRAYSCALE)
    #comparing_mask_img2 = cv2.resize(comparing_mask_img2, IMG_SIZE)
    #comparing_mask_img2 = cv2.bitwise_not(comparing_mask_img2)
    
    #decreaseColor(comparing_img2)
    #comparing_img_bin2 = img2bin(comparing_img2,IMG_SIZE)
    comparing_img2_hsv = cv2.cvtColor(comparing_img2, cv2.COLOR_BGR2HSV)
    # h0to1(comparing_img2_hsv)
    
    #comparing_hist2 = cv2.calcHist([comparing_img2_hsv], [0], comparing_mask_img2, [ranges_h], [0, ranges_h])
    comparing_hist2 = cv2.calcHist([comparing_img2], [0], comparing_mask_img2, [ranges], [0, ranges]) \
                    + cv2.calcHist([comparing_img2], [1], comparing_mask_img2, [ranges], [0, ranges]) \
                    + cv2.calcHist([comparing_img2], [2], comparing_mask_img2, [ranges], [0, ranges])
    #comparing_hist2 = cv2.calcHist([comparing_img_bin2], [0], None, [ranges_bin], [0, ranges_bin])
    
    ######################################################################
    
    
    #ヒストグラムを比較する
    result1 = cv2.compareHist(target_hist, comparing_hist1, 0)
    result2 = cv2.compareHist(target_hist, comparing_hist2, 0)
    print "comparing1 : %.2f" % result1
    print "comparing2 : %.2f" % result2
    
    #画像,ヒストグラムを表示
    fig_num=1
    with plt.style.context('classic'):
        plt.figure(fig_num)
        img_masked1 = cv2.bitwise_and(target_img,target_img,mask=target_mask_img)
        img_r1 = cv2.cvtColor(img_masked1, cv2.COLOR_BGR2RGB)
        plt.subplot(1,3,1)
        plt.imshow(img_r1)

        img_masked2 = cv2.bitwise_and(comparing_img1,comparing_img1,mask=comparing_mask_img1)
        img_r2 = cv2.cvtColor(img_masked2, cv2.COLOR_BGR2RGB)
        plt.subplot(1,3,2)
        plt.imshow(img_r2)
        
        img_masked3 = cv2.bitwise_and(comparing_img2,comparing_img2,mask=comparing_mask_img2)
        img_r3 = cv2.cvtColor(img_masked3, cv2.COLOR_BGR2RGB)
        plt.subplot(1,3,3)
        plt.imshow(img_r3)
    
    fig_num+=1
    with plt.style.context('classic'):
        plt.figure(fig_num)
        plt.subplot(3,1,1)
        plot_histgram(img_r1)
        plt.subplot(3,1,2)
        plot_histgram(img_r2)
        plt.subplot(3,1,3)
        plot_histgram(img_r3)

    fig_num+=1
    with plt.style.context('classic'):
        plt.figure(fig_num)
        plt.subplot(2,2,1)
        plt.imshow(img_r1)
        plt.subplot(2,2,2)
        plt.imshow(img_r2)
        plt.subplot(2,2,3)
        plt.imshow(img_r1)
        plt.subplot(2,2,4)
        plt.imshow(img_r3)
        
        
    fig_num+=1
    with plt.style.context('classic'):
        plt.figure(fig_num)
        plt.subplot(4,2,1)
        plot_histgram(img_r1)
        plt.subplot(4,2,2)
        plot_histgram(img_r2)
        plt.subplot(4,2,3)
        plot_histgram(img_r1)
        plt.subplot(4,2,4)
        plot_histgram(img_r3)



        

    
    plt.show()

    

if __name__ == '__main__':
    compare_by_hist()
