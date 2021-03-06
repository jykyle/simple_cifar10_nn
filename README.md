<h2>A simple neural network for multi-class classification on images represented in RGB
(based on CIFAR-10, also works on other datasets)</h2>

<p>&nbsp;</p>
Q. What makes this project different?
<p>&nbsp;</p>
A. This repository includes a preprocessing unit that handles real images; 

it is possible to import directly by importing from keras: 'keras.datasets.cifar10',

this repository is made to also suit other datasets which requires preprocessing before training.

<p>&nbsp;</p>
<b>Note:</b>

For preprocessing purposes, the data needs to be sorted in:

(defined dataset directory)/labels/raw_data

e.g. CIFAR dataset in this repository, for each train/test directory, 

there are directory with labels which have raw image data (which belong in that specified label),

(python preprocess.py -r cifar_10_dataset/train is a working command which illustrates this)


<p>&nbsp;</p>
<b>Major libraries and versions used to build this project:</b>

Python==3.6.8

Keras==2.2.4

tensorflow-gpu==1.14.0

Pillow==6.1.0

numpy==1.18.1



<p>&nbsp;</p>
<b>Example of a run (in order):</b>

python preprocess_data.py -r cifar_10_dataset/train -ox outputs/trainx.npy -oy outputs/trainy.npy -p 10

python preprocess_data.py -r cifar_10_dataset/test -ox outputs/testx.npy -oy outputs/testy.npy -p 10

python models/cnn_simple.py -o outputs/cnn_example.json -x outputs/trainx.npy -y outputs/trainy.npy

python training.py -m outputs/cnn_example.json -x outputs/trainx.npy -y outputs/trainy.npy -vx outputs/testx.npy -vy 
outputs/testy.npy -w outputs/res_weights.h5

python predictor.py -m outputs/cnn_example.json -w outputs/res_weights.h5 -x outputs/testx.npy -y outputs/testy.npy -o outputs/result
<p>&nbsp;</p>
<b>OR try this sample run (on cifar-10 data included in the repository):</b>

python example_wrapper.py
