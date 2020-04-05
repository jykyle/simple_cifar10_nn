# data generator used in training / predicting
# this is to prevent RAM being overloaded when training/predicting

from keras.utils.data_utils import Sequence
import numpy as np

# generator that includes both x/y as inputs
class Generator(Sequence):
    def __init__(self, x, y, shuffle=True, batch_size=256):
        self.x = x
        self.y = y
        self.select_idx = np.arange(self.x.shape[0])
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.copy(self.select_idx)

    def __len__(self):
        length = int(np.ceil(self.indices.shape[0] / self.batch_size))
        return length
                            
    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x[inds]
        batch_y = self.y[inds]
        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indices)
    
    def initialize_after_train(self):
        self.indices = np.copy(self.select_idx)

# generator without y value (for predicting)
class Generator_no_y(Sequence):
    def __init__(self, x, shuffle=True, batch_size=256):
        self.x = x
        self.select_idx = np.arange(self.x.shape[0])
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.copy(self.select_idx)

    def __len__(self):
        length = int(np.ceil(self.indices.shape[0] / self.batch_size))
        return length
                            
    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x[inds]
        return batch_x

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indices)
    
    def initialize_after_train(self):
        self.indices = np.copy(self.select_idx)