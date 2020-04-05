'''
CNN example (2 convolution blocks - 4 conv layers)
Input: directory to save json file on
(optional: ML input and output to define the dimensions of input/output layer)
Output: constructed model json file
'''

import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import plot_model

import numpy as np
import os
import argparse

def save_json(model, json_filename):
    model_json = model.to_json()
    open('{}'.format(json_filename), 'w').write(model_json)
    imgfile = '{}.png'.format('.'.join(json_filename.split('.')[:-1]))
    plot_model(model, to_file='{}'.format(imgfile), show_shapes=True, show_layer_names=True)


def build_model(inputshape=(32,32,3), num_classes=10, output_json_dir=None):
    
    print('Build cnn_simple model')
    
    x_input = Input(shape=inputshape)
    x = Conv2D(32, kernel_size=(3,3), padding='same', activation='relu')(x_input)
    x = Conv2D(32, kernel_size=(3,3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(64, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = Conv2D(64, kernel_size=(3,3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.25)(x)
    
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    y = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=x_input, outputs=y)
    
    if output_json_dir is not None:
        save_json(model, output_json_dir)
        print('Model saved at: {}'.format(output_json_dir))
    
    return model


if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description = 'Arguments required for building cnn_simple')
    parser.add_argument('-o', '--outputjson', type=str, help='Define directory in which to drop the model json file on (directory needs to exist)',
                        default='model.json')
    parser.add_argument('-x', '--refinput', type=str, help='Directory of ML input to use model on - for dimension information (optional)', default=None)
    parser.add_argument('-y', '--refoutput', type=str, help='Directory of ML output - for dimension information (optional)', default=None)
    args = parser.parse_args()
    
    # if reference files are parsed, change dimension for the construction of model, otherwise build model based on CIFAR-10 example
    if args.refinput is not None:
        x = np.load(args.refinput, mmap_mode='r+')
        input_shape = x.shape[1:]
    else:
        input_shape = (32,32,3)
    
    if args.refoutput is not None:
        y = np.load(args.refoutput, mmap_mode='r+')
        output_shape = y.shape[-1]
    else:
        output_shape = 10
    
    build_model(inputshape=input_shape, num_classes=output_shape, output_json_dir=args.outputjson)
    
    