'''
Resnet-v1 example (3 resblocks)
Input: directory to save json file on
(optional: ML input and output to define the dimensions of input/output layer)
Output: constructed model json file
(work in progress)
'''

import keras
from keras.models import Model
from keras.layers import Dense, Flatten, Input, BatchNormalization, Activation
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.utils import plot_model
from keras.regularizers import l2

import numpy as np
import os
import argparse

def save_json(model, json_filename):
    model_json = model.to_json()
    open('{}'.format(json_filename), 'w').write(model_json)
    imgfile = '{}.png'.format('.'.join(json_filename.split('.')[:-1]))
    plot_model(model, to_file='{}'.format(imgfile), show_shapes=True, show_layer_names=True)


def build_model(inputshape=(32,32,3), feature_maps=[16,16,32,64], filter_size=[3,3,3,3], num_classes=10, output_json_dir=None):
    
    # input
    x_in = Input(shape=inputshape)
    
    # begin layer (conv2D -> BN -> Pooling)
    conv_1 = Conv2D(feature_maps[0], kernel_size=(filter_size[0],filter_size[0]), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x_in)
    bn_1 = BatchNormalization()(conv_1)
    first_part = Activation('relu')(bn_1)
    #pool_1 = MaxPooling2D(pool_size=(2, 2))(bn_1)
    
    # resblock : run for loop to get it deeper
    rn_res = first_part
    for i in range(1,len(feature_maps)):
        for j in range(3):
            if i > 1 and j == 0:
                rn_res = resblock(rn_res, feature_maps[i], (filter_size[i],filter_size[i]), 2) 
            else:
                rn_res = resblock(rn_res, feature_maps[i], (filter_size[i],filter_size[i]), 1) 
    
    # finally: pool, flatten then output layer
    avgpool = AveragePooling2D(pool_size=8)(rn_res)
    flat = Flatten()(avgpool)
    out = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(flat)
    
    model = Model(inputs=x_in, outputs=out)
    
    if output_json_dir is not None:
        save_json(model, output_json_dir)
        print('Model saved at: {}'.format(output_json_dir))
    
    return model


def resblock(begin_layer, featuremap, filtersize, stridemode):
    # 'the block' method
    rb_conv1 = Conv2D(featuremap, kernel_size=filtersize, strides=stridemode, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(begin_layer)
    rb_bn1 = BatchNormalization()(rb_conv1)
    rb_act1 = Activation('relu')(rb_bn1)
    
    rb_conv2 = Conv2D(featuremap, kernel_size=filtersize, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(rb_act1)
    rb_bn2 = BatchNormalization()(rb_conv2)
    
    if stridemode > 1:
        begin_layer = Conv2D(featuremap, kernel_size=(1,1), strides=stridemode, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(begin_layer)
    # add and apply activation to all
    addition = keras.layers.add([rb_bn2, begin_layer])
    final = Activation('relu')(addition)
    
    return final

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description = 'Arguments required for building vgg_v1')
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