'''
Module that converts raw data to ML inputs (preprocessing)

Input : 1. Raw data directory (ex. cifar_10_dataset/train)
Required format: parent_directory/label(1..n)/data_in_image_format
2. Path to output preprocessed files (_x, _y will be added to distinguish picture detail/label)

Output : processed x values of shape (N, Y, X, 3) and y (label) used for ML
N : number of data in directory (of all labels)
Y : height of the image (in pixels)
X : width of the image (in pixels)
3 : R, G, B values of a pixel (normalized)
'''

import numpy as np
import glob
from PIL import Image
import os
import time
import argparse
from multiprocessing import Pool
from itertools import product

def preprocess_multi(image_labels_combined):
    image_directory, label_array = image_labels_combined
    image = Image.open(image_directory)
    pixelvalues = np.array(image.convert('RGB')) / 255 # convert image into RGB format then normalize
    label = image_directory.split('/')[-2] # get label from the directory name
    y = np.array([int(label == lvalue) for lvalue in label_array])
    
    return pixelvalues, y


# wrapper for preprocessing unit (to use on other classes)
def preprocess(data_dir, num_pool=1, outputx_dir=None, outputy_dir=None):
    
    print('Begin running : preprocess.py')
    begin_time = time.time()
    
    imagefiledirs = sorted(glob.glob('{}/*/*'.format(data_dir))) # full directory of all data (image)
    label_array = np.unique([image_dir.split('/')[-2] for image_dir in imagefiledirs]) # list of possible labels (from directory names)
    params = list(product(imagefiledirs,[label_array]))
    
    # Run preprocessing methods defined above
    if num_pool > 1: # using more than 1 process (user defined), multiprocessing
        print('Multiprocessing, number of processes used: {}'.format(num_pool))
        p = Pool(num_pool)
        processed_data = np.array(p.map(preprocess_multi, params))
        p.close()
        p.join()
    
    else:
        print('Multiprocessing is not used.')
        processed_data = np.array(list(map(preprocess_multi, params)))
    
    # Save the preprocessed data
    processed_image = np.stack(processed_data[:,0])
    processed_label = np.stack(processed_data[:,1])
    
    if outputx_dir is not None:
        np.save('{}'.format(outputx_dir), processed_image)
    if outputy_dir is not None:
        np.save('{}'.format(outputy_dir), processed_label)
    
    print('Preprocess complete.\nDimension of x: {}\nDimension of y: {}'.format(processed_image.shape, processed_label.shape))
    print('Time elapsed: {:.2f} seconds'.format(time.time()-begin_time))
    
    return processed_image, processed_label


if __name__=="__main__":
    
    # parse arguments required for preprocess to function
    parser = argparse.ArgumentParser(description = 'Arguments required for preprocess (works well with cifar-10)')
    parser.add_argument('-r', '--rawdata', type=str, help='Define raw data directory', default='cifar_10_dataset/train')
    parser.add_argument('-ox', '--outputx', type=str, help='Define preprocessed output directory/filename (RGB)', default='x.npy')
    parser.add_argument('-oy', '--outputy', type=str, help='Define preprocessed output directory/filename (label)', default='y.npy')
    parser.add_argument('-p', '--pool', type=int, help='Define the number of processes (in CPU) to run preprocess on', default=1)
    args = parser.parse_args()
    
    preprocess(args.rawdata, args.pool, args.outputx, args.outputy)
