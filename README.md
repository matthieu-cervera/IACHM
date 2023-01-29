# IACHM

Trainmodel.ipynb : file where the model is created and trained (on a GPU given by google colab)

data_generation_augmentation_prepro.py : file for data generation and augmentation 

loadmodel.py : the file that uses the model to evaluate the live gestures

mainlive.py : the file that allows connection between our model and Max/MSP live. We record a new gesture each 2s

testRealLive.py : a test file (not used in the final version) to record every gesture and evaluate them live. However not used because there are too much computing processes and with a new 
gesture to evaluate each 0.03s, the computers are too slow/
