'''
This module is called if model needs to be compiled for fit/evaluate (mainly for training.py)
This script is to be expanded if other optimizer functions are going to be used.
'''

import keras
    
def RMSprop(model, learn_rate=0.0001, decay_rate=1e-6):
    
    opt = keras.optimizers.RMSprop(lr=learn_rate, decay=decay_rate)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model