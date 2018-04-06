import skimage.io
import skimage.transform
from PIL import Image
from PIL import ImageChops
import os
import ipdb
import cv2 as cv
import scipy.misc
import numpy as np
import os
import random

#Dingo
#result_path='/Users/niloleart/Desktop/'

#Opentrends
result_path = '/home/noleart/Escritorio/results'
mask_path = '/home/noleart/Escritorio/TFG/DATASET/Masks/'


#def load_image( path, height=128, width=128 ):

def load_image( path, pre_height=146, pre_width=146, height=128, width=128 ):

    try:
        #img = skimage.io.imread( path ).astype( float )
        #img = skimage.io.imread(path)
        img = scipy.misc.imread(path).astype(float)
    except:
        return None

    img /= 255.

    if img is None: return None
    if len(img.shape) < 2: return None
    if len(img.shape) == 4: return None
    if len(img.shape) == 2: img=np.tile(img[:,:,None], 3)
    if img.shape[2] == 4: img=img[:,:,:3]
    if img.shape[2] > 4: return None

    short_edge = min( img.shape[:2] )
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy+short_edge, xx:xx+short_edge]
    resized_img = skimage.transform.resize( crop_img, [pre_height,pre_width] )

    rand_y = np.random.randint(0, pre_height - height)
    rand_x = np.random.randint(0, pre_width - width)

    resized_img = resized_img[ rand_y:rand_y+height, rand_x:rand_x+width, : ]
    #scipy.misc.imsave(os.path.join(result_path,'loadedimage.png'), resized_img)
    return resized_img#(resized_img * 2)-1 #(resized_img - 127.5)/127.5


def load_image_2( image_path, mask_path, pre_height=146, pre_width=146, height=128, width=128 ):

    try:
        img = skimage.io.imread( image_path ).astype( float )
    except:
        return None

    img /= 255.

    if img is None: return None
    if len(img.shape) < 2: return None
    if len(img.shape) == 4: return None
    if len(img.shape) == 2: img=np.tile(img[:,:,None], 3)
    if img.shape[2] == 4: img=img[:,:,:3]
    if img.shape[2] > 4: return None

    short_edge = min( img.shape[:2] )
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy+short_edge, xx:xx+short_edge]
    resized_img = skimage.transform.resize( crop_img, [pre_height,pre_width] )

    rand_y = np.random.randint(0, pre_height - height)
    rand_x = np.random.randint(0, pre_width - width)

    resized_img = resized_img[ rand_y:rand_y+height, rand_x:rand_x+width, : ]
    #scipy.misc.imsave(os.path.join(result_path,'loadedimage.png'), resized_img)
    return (resized_img * 2)-1 #(resized_img - 127.5)/127.5


def crop_random(image_ori, width=64,height=64, x=None, y=None, overlap=7):
    if image_ori is None: return None
    random_y = np.random.randint(overlap,height-overlap) if x is None else x
    random_x = np.random.randint(overlap,width-overlap) if y is None else y

    image = image_ori.copy()
    crop = image_ori.copy()
    crop = crop[random_y:random_y+height, random_x:random_x+width]
    image[random_y + overlap:random_y+height - overlap, random_x + overlap:random_x+width - overlap, 0] = 2*117. / 255. - 1.
    image[random_y + overlap:random_y+height - overlap, random_x + overlap:random_x+width - overlap, 1] = 2*104. / 255. - 1.
    image[random_y + overlap:random_y+height - overlap, random_x + overlap:random_x+width - overlap, 2] = 2*123. / 255. - 1.

    return image, crop, random_x, random_y


def crop_random_2(image_ori, width=64,height=64, x=None, y=None, overlap=7):
    if image_ori is None: return None
    random_y = np.random.randint(overlap,height-overlap) if x is None else x
    random_x = np.random.randint(overlap,width-overlap) if y is None else y

    image = image_ori.copy()
    crop = image_ori.copy()
    crop = crop[0:0+height, 32:32+width]

    #this puts cropped region to grey
    image[0+overlap:0+height - overlap, 32 + overlap:32+width - overlap, 0] = 2*117. / 255. - 1.
    image[0+overlap:0+height - overlap, 32 + overlap:32+width - overlap, 1] = 2*104. / 255. - 1.
    image[0+overlap:0+height - overlap, 32 + overlap:32+width - overlap, 2] = 2*123. / 255. - 1.

    return image, crop, random_x, random_y


def crop_random_3(image_ori, width=64,height=64, x=None, y=None, overlap=7):
    if image_ori is None: return None
    image = image_ori.copy()
    mask_name = random.choice([x for x in os.listdir(mask_path) if os.path.isfile(os.path.join(mask_path,x))])
    mask = skimage.io.imread(os.path.join(mask_path,mask_name)).astype(float)
    mask /= 255.
    #scipy.misc.imsave(os.path.join(result_path, 'mask.png'), mask)
    resized_mask = skimage.transform.resize(mask, [128, 128])
    crop = np.multiply(image_ori, resized_mask)
    # try:
    #     scipy.misc.imsave(os.path.join(result_path, 'crop.png'), crop)
    # except:
    #     print ('unable to save image')
    crop = skimage.transform.resize(crop,[64,64])
    resized_mask = skimage.transform.resize(mask, [64, 64])
    return image, crop, resized_mask


def crop_random_4(image_ori, mask_ori, width=64,height=64, x=None, y=None, overlap=7):
    if image_ori is None: return None
    image = image_ori.copy()
    scipy.misc.imsave(os.path.join(result_path, 'loaded_image.png'), image)

    mask = mask_ori.copy()
    scipy.misc.imsave(os.path.join(result_path, 'mask.png'), mask)

    resized_mask = skimage.transform.resize(mask, [128, 128])
    crop = np.multiply(image_ori, resized_mask)
    #crop = cv.bitwise_and(image, mask)
    scipy.misc.imsave(os.path.join(result_path, 'img_mult_mask.png'), crop.astype(float))

    crop = skimage.transform.resize(crop,[64,64])
    resized_mask = skimage.transform.resize(mask, [64, 64])
    return image, crop, resized_mask


def merge_mask(image_ori,  mask, x=None, y=None):
    if image_ori is None: return None
    rsz_y = 64 if x is None else x
    rsz_x = 64 if y is None else y
    masked_image = image_ori * (1 - mask)
    # masked_image[np.where((masked_image > 1))] = 1
    masked_image = skimage.transform.resize(masked_image, [rsz_y, rsz_x])
    image_ori = skimage.transform.resize(image_ori, [128,128])
    return image_ori, masked_image, x, y

