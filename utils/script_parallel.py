#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
import numpy as np
import os
import h5py
from os import walk
import imageio
import skimage.io as io
import random
import scipy
import scipy.ndimage.filters as filters
import cv2
import random
import multiprocessing
import SharedArray
import time

BASIC_H5_PATH = "/home/elefevre/Documents/Data_dypfish/basic.h5"


def rotate_img(path, img_name):
    img = cv2.imread(path + img_name, cv2.IMREAD_UNCHANGED)
    # get image height, width
    (h, w) = img.shape[:2]
    # calculate the center of the image
    center = (w / 2, h / 2)
    angles = [90, 180, 270]

    scale = 1.0

    # Perform the counter clockwise rotation holding at the center
    for angle in angles:
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(img, M, (h, w))
        cv2.imwrite(path + str(angle) + "_" + img_name, rotated)


def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = filters.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = filters.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    return scipy.ndimage.interpolation.map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


def elastic_transform_wrapped(img_path, mask_path, dir_path, dir_mask_path, tiff=True):
    if tiff:
        im = io.imread(dir_path + img_path, plugin="tifffile")
        im_mask = io.imread(dir_mask_path + mask_path, plugin="tifffile")
    else:
        im = io.imread(dir_path + img_path)
        im_mask = io.imread(dir_mask_path + mask_path)
    im_merge = np.concatenate((im[..., None], im_mask[..., None]), axis=2)

    # im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 3, im_merge.shape[1] * 0.09, im_merge.shape[1] * 0.09)
    im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08,
                                   im_merge.shape[1] * 0.08)  # soft transform

    im_t = im_merge_t[..., 0]
    im_mask_t = im_merge_t[..., 1]
    io.imsave(dir_path + "t_" + img_path, im_t)
    io.imsave(dir_mask_path + "t_" + mask_path, im_mask_t)

def get_max_projection(image_3d):
    num_slices = len(image_3d[:,0,0])
    max_proj = image_3d[0,:,:]
    for slice_idx in range(1, num_slices):
        slice = image_3d[slice_idx,:,:]
        max_proj = np.maximum(max_proj, slice)
    min = np.min(max_proj)
    max_proj[max_proj < min + np.sqrt(min)]=0
    return max_proj


def get_mask(filename, type):
    tab = filename.split('/') # 0 1 useless, 2 molecule, 3 mtype + timepoint, 4 number, 5 'image'
    molecule = tab[2]
    mtype = tab[3].split('_')[0]
    timepoint = tab[3].split('_')[1]
    image = tab[4].split('_')[1]
    cell_mask = np.ones((512, 512))
    with h5py.File(BASIC_H5_PATH, 'r') as file_handler:
        try:
            cell_mask = file_handler[mtype][molecule][timepoint][str(image)][type][()]
        except KeyError:
            print('no cell mask in h5 for', filename)
    return cell_mask


def create_mtoc_data(max_proj, cell_mask, filename):
    tab = filename.split('/') # 0 1 useless, 2 molecule, 3 mtype + timepoint, 4 number, 5 'image'
    molecule = tab[2]
    mtype = tab[3].split('_')[0]
    timepoint = tab[3].split('_')[1]
    nb = tab[4].split('_')[1]
    image = max_proj.astype(np.float32) * np.array(cell_mask).astype(np.float32) # input for mtoc and spots
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    with h5py.File(BASIC_H5_PATH, 'r') as file_handler:
        try:
            mtoc_position = file_handler[mtype][molecule][timepoint][str(nb)]["mtoc_position"][()]
            # image = io.imread(filename)
            pos = (int(mtoc_position[0]), int(mtoc_position[1]))
            image = (image/np.amax(image))*255
            cv2.circle(image, pos, 10, color=(255, 255, 255), thickness=5)
            return max_proj * np.array(cell_mask), image
        except KeyError:
            print('no mtoc pos in h5 for', mtype+" "+molecule+" "+timepoint+" "+str(nb))



def stack_all_3d_images(dirpath, filenames):  # stocker les splits pour aller voir dans le h5
    save_path = "data/"
    analyse = ['tubulin', 'dapi', 'fish']
    for filename in filenames:
        if filename.split('.')[1] == 'tif' and filename.split('.')[0].strip().lower() in analyse:
            full_filename = dirpath + '/' + filename
            file = dirpath.split("/")[-3]+'_'+dirpath.split("/")[-2]+'_'+dirpath.split("/")[-1] + '_' +filename.split('.')[0].strip().lower()+ '.png'
            new_filename = filename.split('.')[0].strip().lower() + '/' + file
            cell_mask_filename = filename.split('.')[0].strip().lower() + "/cell_mask/" + file
            nucleus_mask_filename = filename.split('.')[0].strip().lower() + "/nucleus_mask/" + file
            mtoc_input = filename.split('.')[0].strip().lower() + "/mtoc/" + file
            mtoc_target = filename.split('.')[0].strip().lower() + "/mtoc/target/" + file
            print(full_filename)
            try:
                im = io.imread(full_filename, plugin="tifffile")
                max_proj = get_max_projection(im).astype(np.float32)
                max_proj = (max_proj / np.amax(max_proj))*255
                cell_mask = (get_mask(full_filename, 'cell_mask')*255)
                nucleus_mask = (get_mask(full_filename, 'nucleus_mask')*255)
                mtoc_data, mtoc_label = create_mtoc_data(max_proj, cell_mask, full_filename)
                imageio.imwrite(save_path + '/' + new_filename, max_proj)
                imageio.imwrite(save_path + '/' + cell_mask_filename, cell_mask)
                imageio.imwrite(save_path + '/' + nucleus_mask_filename, nucleus_mask)
                imageio.imwrite(save_path + '/' + mtoc_input, mtoc_data)
                imageio.imwrite(save_path + '/' + mtoc_target, mtoc_label)
            except Exception as e:
                print(e)


def get_dirpath(raw_image_data):
    dirpaths = []
    for dirpath, dirnames, filenames in os.walk(raw_image_data):
        if len(filenames) == 0:
            continue
        dirpaths.append((dirpath, filenames))
    return dirpaths


def create_sub_folders(folder):
    os.system("mkdir -p "+folder+"/train/" )
    os.system("mkdir -p "+folder+"/test/" )
    os.system("mkdir -p "+folder+"/cell_mask/train/" )
    os.system("mkdir -p "+folder+"/cell_mask/test/" )
    os.system("mkdir -p "+folder+"/nucleus_mask/train/" )
    os.system("mkdir -p "+folder+"/nucleus_mask/test/" )
    os.system("mkdir -p "+folder+"/mtoc/target/train/" )
    os.system("mkdir -p "+folder+"/mtoc/target/test/" )
    os.system("mkdir -p "+folder+"/mtoc/train/" )
    os.system("mkdir -p "+folder+"/mtoc/test/" )

def clean():
    os.system("rm -rf data/dapi/cell_mask")
    os.system("rm -rf data/dapi/mtoc")

    os.system("rm -rf data/tubulin/nucleus_mask")

    os.system("rm -rf data/fish/mtoc")
    os.system("rm -rf data/fish/cell_mask")
    os.system("rm -rf data/fish/nucleus_mask")



def split_train_test(folder, rate):
    for dirpath, dirnames, filenames in os.walk(folder):
        tmp = []
        for file in filenames:
            tmp.append(dirpath + "/" + file)
        if len(tmp)!=0 and not "test" in dirpath and not "train" in dirpath:
            random.seed(42)
            random.shuffle(tmp)
            size = rate * len(tmp)
            for i, elt in enumerate(tmp):
                if i<size:
                    os.system("mv {} {}/test/".format(elt, dirpath))
                else:
                    os.system("mv {} {}/train/".format(elt, dirpath))


def list_files(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

def multi_process_fun(file_list, dirpath, function):
    list_size = len(file_list)
    num_workers = 6
    worker_amount = int(list_size/num_workers)

    processes = []
    for worker_num in range(num_workers):
        process = multiprocessing.Process(target=function, args=(dirpath, file_list[worker_amount*worker_num : worker_amount*worker_num + worker_amount]))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()


def multi_process_transform(img_list, mask_list, img_folder, mask_folder):
    list_size = len(img_list)
    num_workers = 6
    worker_amount = int(list_size/num_workers)

    def rewrapped_transform(img_list, mask_list, img_folder, mask_folder):
        for i, img in enumerate(list_img):
            mask = list_mask[i]
            print(img)
            elastic_transform_wrapped(img, mask, img_folder, mask_folder, tiff=False)

    processes = []
    for worker_num in range(num_workers):
        process = multiprocessing.Process(target=rewrapped_transform, args=(img_list[worker_amount*worker_num: worker_amount*worker_num+worker_amount],
                                                                    mask_list[worker_amount*worker_num: worker_amount*worker_num+worker_amount],
                                                                    img_folder, mask_folder))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

def wrap_project(test, dir_list):
    for directory, filenames in dir_list:
        stack_all_3d_images(directory, filenames)

def wrap_rotate(test, dir_list):
    for directory, filenames in dir_list:
        for file in filenames:
            rotate_img(directory+"/", file)

if __name__ == "__main__":
    RAW_IMG_DATA = "RawData/raw_image_data/"
    FOLDERS = ["./data/fish", "./data/tubulin", "./data/dapi"]
    for folder in FOLDERS:
        create_sub_folders(folder)

    dir_list = get_dirpath(RAW_IMG_DATA)

    # for directory, filenames in dir_list:
        # stack_all_3d_images(directory, filenames)  # get mask from h5 at the same time
    multi_process_fun(dir_list, '', wrap_project) # need empty string otherwise don't work (?)

    for fold in FOLDERS:
        split_train_test(fold, 0.3)
    clean()
    time.sleep(1)
    new_list = get_dirpath('./data/')

    multi_process_fun(new_list, '', wrap_rotate)

    T_FOLDERS = [("./data/tubulin/train/", "./data/tubulin/cell_mask/train/"),
                ("./data/tubulin/test/", "./data/tubulin/cell_mask/test/"),
                ("./data/dapi/train/", "./data/dapi/nucleus_mask/train/"),
                ("./data/dapi/test/", "./data/dapi/nucleus_mask/test/") ]

    for img_folder, mask_folder in T_FOLDERS:
        list_img = list_files(img_folder)
        list_mask = list_files(mask_folder)
        multi_process_transform(list_img, list_mask, img_folder, mask_folder)
