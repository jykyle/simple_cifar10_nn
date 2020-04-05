'''
This module aims to predict and output the values of test data given the model details.

Inputs:
Model architecture (.json) and weight (.h5) files
Test data x
Test data y (optional, needed to calculate accuracy)
Output directory (place to save the resulting values in npy and log file)
Number of batches to run predict generator (optional)

Outputs:
The predicted y-values based on model and test data input (npy)
Log file with timestamp and parameters used, along with accuracy if needed.
'''

from keras.models import model_from_json
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

import argparse
from data_generator import Generator, Generator_no_y
import numpy as np
from utils import compile_model
import datetime
import time

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
KTF.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

def calc_accuracy(predy, testy):
    pred_labels = []
    for row in predy:
        pred_labels.append(np.argmax(row))

    num_correct = 0
    for i in range(len(pred_labels)):
        if pred_labels[i] == np.argmax(testy[i,:]):
            num_correct += 1
    return num_correct / testy.shape[0]

def predict(model_json, model_weights, testx_npy, testy_npy=None, output_dir=None, nbatch=128):
    print('Begin running predictor.py.')
    begin_time = time.time()
    
    json_file = open(model_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_weights)
    
    testx = np.load(testx_npy, mmap_mode='r+')
    test_data_gen = Generator_no_y(testx, batch_size=nbatch, shuffle=False)
    
    predy = model.predict_generator(test_data_gen)
    npy_filename = '{}.npy'.format(output_dir)
    np.save(npy_filename, predy)
    
    accuracy = None
    if testy_npy is not None:
        testy = np.load(testy_npy, mmap_mode='r+')
        accuracy = calc_accuracy(predy, testy)
    
    if output_dir is not None:
        npy_filename = '{}.npy'.format(output_dir)
        np.save(npy_filename, predy)
        currentDT = datetime.datetime.now()
        with open('{}.log'.format(output_dir), 'a') as f:
            f.write('{}\nModel: {}\nWeights: {}\nTest data directory: {}\nPrediction values saved at: {}\n'.format(currentDT, model_json, model_weights, testx_npy, npy_filename))
            if accuracy is not None:
                f.write('Test y directory: {}\nAccuracy = {}\n'.format(testy_npy, accuracy))
        print('Wrote npy and log file to: {}'.format(output_dir))
    
    print('predictor.py done. Time elapsed: {:.2f} seconds'.format(time.time()-begin_time))
    
    return predy
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Arguments required for training model (works well with cifar-10)')
    parser.add_argument('-m', '--modeljson', type=str, help='Define location of json file', required=True) # mandatory input
    parser.add_argument('-w', '--modelweights', type=str, help='Define location of model weight file', required=True)
    parser.add_argument('-x', '--testx', type=str, help='Define directory of ML input (x)', required=True)
    parser.add_argument('-y', '--testy', type=str, help='Define directory of ML output (y)', default=None)
    parser.add_argument('-o', '--outdir', type=str, help='Define location/name of where the result files should go', default=None)
    parser.add_argument('-nb', '--nbatch', type=int, help='Define batch size when evaluating', default=128)
    args = parser.parse_args()
    
    predict(args.modeljson, args.modelweights, args.testx, args.testy, args.outdir, args.nbatch)