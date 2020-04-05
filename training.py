'''
Trains the model given the model and data inputs

Inputs:
1. json file directory
2. npy files of training data x and y
3. directory of where to save the weight file (.h5 format)
4. (optional) npy files of validation data x and y

Output:
Weight file (.h5) to the specified directory defined in input (3)
'''

from keras.models import model_from_json
import keras
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import numpy as np
import argparse
from data_generator import Generator
from utils import compile_model
import time

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
KTF.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

def train(json_file, trainx_npy, trainy_npy, weight_directory, valx_npy=None, valy_npy=None, 
          epochs=100, nbatch=32, verb=1):
    
    # load data
    trainx = np.load(trainx_npy, mmap_mode='r+')
    trainy = np.load(trainy_npy, mmap_mode='r+')    
    train_data_gen = Generator(trainx, trainy, batch_size=nbatch)
    
    if valx_npy is None or valy_npy is None:
        val_data_gen = None
    else:
        valx = np.load(valx_npy, mmap_mode='r+')
        valy = np.load(valy_npy, mmap_mode='r+')    
        val_data_gen = Generator(valx, valy, batch_size=nbatch)
    
    # model load and compile
    json_file = open(json_file, 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model = compile_model.RMSprop(model)
    
    
    # train model (get time elapsed on fitting only)
    print('Begin model fit')
    begin_time = time.time()
    
    hist = model.fit_generator(generator=train_data_gen, validation_data=val_data_gen, epochs=epochs, verbose=verb, workers=5)
    
    print('End fit. Time elapsed: {:.2f} seconds'.format(time.time()-begin_time))
    
    # save final weights
    model.save_weights(weight_directory)
    
    return model, hist


if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Arguments required for training model (works well with cifar-10)')
    parser.add_argument('-m', '--modeljson', type=str, help='Define location of json file', required=True) # mandatory input
    parser.add_argument('-x', '--trainx', type=str, help='Define directory of ML input (x)', required=True)
    parser.add_argument('-y', '--trainy', type=str, help='Define directory of ML output (y)', required=True)
    parser.add_argument('-vx', '--valx', type=str, help='Define directory of validation x (optional)', default=None)
    parser.add_argument('-vy', '--valy', type=str, help='Define directory of validation y (optional)', default=None)
    parser.add_argument('-w', '--weightdir', type=str, help='Define location of the weights after training', default='model_weights.h5')
    parser.add_argument('-e', '--epoch', type=int, help='Decide how many epochs to train', default=100)
    parser.add_argument('-b', '--batch', type=int, help='Decide the batch size for training', default=32)
    parser.add_argument('-v', '--verbose', type=int, help='Decide what value verbose should be for fitting', default=1)
    args = parser.parse_args()
    
    train(args.modeljson, args.trainx, args.trainy, args.weightdir, args.valx, args.valy, args.epoch, args.batch, args.verbose)
    