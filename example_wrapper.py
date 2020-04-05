'''
This is a wrapper file to help users get started.
Script will run preprocessing, construct simple model and train then produce output of prediction of cifar-10 data in the repository
Input files : train / test data directory
Output : directory to save preprocessed files, model architecture and weights, and predictor results
'''

from preprocess_data import preprocess
from models import cnn_simple
from training import train
from predictor import predict
import argparse
import os

if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Arguments required for training model (works well with cifar-10)')
    parser.add_argument('-tr', '--traindata', type=str, help='Define train data directory', default='cifar_10_dataset/train')
    parser.add_argument('-te', '--testdata', type=str, help='Define test data directory', default='cifar_10_dataset/test')
    parser.add_argument('-o', '--outputdir', type=str, help='Define output directory (files will all be saved here)', default='output')
    args = parser.parse_args()
    
    # creating directory and defining files which will be saved under the directory
    os.makedirs('./{}'.format(args.outputdir), exist_ok=True)
    trainx_dir = '{}/trainx.npy'.format(args.outputdir)
    trainy_dir = '{}/trainy.npy'.format(args.outputdir)
    testx_dir = '{}/testx.npy'.format(args.outputdir)
    testy_dir = '{}/testy.npy'.format(args.outputdir)
    modeljson_dir = '{}/model.json'.format(args.outputdir)
    weight_dir = '{}/weights.h5'.format(args.outputdir)
    output_dir = '{}/result'.format(args.outputdir)
    
    # preprocess training data and testing data, output preprocessed npy files
    trainx, trainy = preprocess(args.traindata, num_pool=10, outputx_dir=trainx_dir, outputy_dir=trainy_dir)
    preprocess(args.testdata, num_pool=10, outputx_dir=testx_dir, outputy_dir=testy_dir)
    
    # build model based on training x,y dimensions, output model json file
    cnn_simple.build_model(inputshape=trainx.shape[1:], num_classes=trainy.shape[-1], output_json_dir=modeljson_dir)
    
    # train model with the training data, output weight h5 file
    train(json_file=modeljson_dir, trainx_npy=trainx_dir, trainy_npy=trainy_dir, weight_directory=weight_dir)
    
    # predict testing data, output predicted values in npy and write result log
    predict(model_json=modeljson_dir, model_weights=weight_dir, testx_npy=testx_dir, testy_npy=testy_dir, output_dir=output_dir)