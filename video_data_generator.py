import os
import numpy as np
from tensorflow import keras
from load_video import load_video

class VideoDataGenerator(keras.utils.Sequence):
    """
    A custom video data generator to read videos and create batches of samples
    for the Keras' model.fit_generator interface.

    Attributes
    ----------
    df : pandas.DataFrame
        a dataframe with video pathname and corresponding label
    path : str
        root folder of all the video
    batch_size : int, optional
        number of samples in each batch
    shuffle : bool, optional
        whether to shuffle the data at the end of each epoch
    file_col : str, optional
        column name of the video pathname column
    y_col : str, optional
        column name of the label column
    mapping : dict, optional
        custom mapping for the label
    max_frames : int, optional
        max number of captured frames in each video
    resize : int, optional
        resolution of converted video
    step : int, optional
        skip how many frames before capturing 1 frame
    """

    def __init__(self, df, path, batch_size=4, shuffle=True, file_col="filename", y_col="class", mapping=None, max_frames=120, img_size=224, step=4):
        self.df = df.copy()
        self.path = path
        self.indices = self.df.index.to_list()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.file_col = file_col
        self.y_col = y_col
        self.mapping = mapping
        self.max_frames = max_frames
        self.resize = (img_size, img_size)
        self.step = step

    def __len__(self):
        """Return the number of batches in the data"""
        return len(self.df) // self.batch_size

    def __getitem__(self, index):
        """Return a batch of data"""
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        return self.__generate_data(indices)

    def on_epoch_end(self):
        """this is called by Keras at the end of each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __generate_data(self, indices):
        """helper method for generating a batch of data"""
        X, mask = [], []
        for i in indices:
            v, m = load_video(os.path.join(self.path, self.df.loc[i, self.file_col]), self.max_frames, self.resize, self.step)
            X.append(v)
            mask.append(m)
        y = self.df.loc[indices, self.y_col]
        if self.mapping:
            y = y.map(self.mapping)

        return (np.array(X), np.array(mask)), np.array(y)
