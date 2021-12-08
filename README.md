# Video classification using Keras.

Architecture used is CNN+RNN. For CNN, ResNet50 is used. For RNN, GRU is used.
Expects a folder with all the video samples and 3 CSV files train.csv, val.csv and test.csv specifying the labels of the training, validation, and test set respectively.
The CSVs should have two columns: `video_name` specifies the video pathname and `tag` specifies the label of the video.
Project includes a custom video data generator to read videos and create batches of samples for the keras' `model.fit_generator` interface.
Latest Cudnn and CudaToolkits are needed.
Include data in the correct format and run python train.py.
