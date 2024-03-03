# IACHM
Real-time movement-based music player using phone accelerometric and gyroscopic data. The core of the project is a Max/MSP file that is linked to python files using a trained deep learning model to add special movements possibilities to the pipeline.

This project aims to play a chord of three notes (initialy Cminor chord : C, E, G). Each note can be changed individually to every notes of the scale with a set of movements using gyroscopic and accelerometric data collected from a phone and processed by the Max/MSP file. This idea was inspired by Jacob Collier's video : https://www.youtube.com/watch?v=3KsF309XpJo&ab_channel=JacobCollier.

An interesting part of the project was adding 2 other movements that : 
1. start or stop the sound
2. change the sound of the chord

These two movements are to be detected using a Deep Learning model. To this purpose we created our own training data, trained a LSTM based model to classify a movement. Then, we used our trained model for a real-time detection and evaluation of the player movements. 

For more informations, see the project presentation slides.

Files Description :

Trainmodel.ipynb : file where the model is created and trained (on a GPU given by google colab)

data_generation_augmentation_prepro.py : file for data generation and augmentation 

loadmodel.py : the file that uses the model to evaluate the live gestures

mainlive.py : the file that allows connection between our model and Max/MSP live. We record a new gesture each 2s

testRealLive.py : a test file (not used in the final version) to record every gesture and evaluate them live. However not used because there are too much computing processes and with a new 
gesture to evaluate each 0.03s, the computers are too slow/

main.ZigSim.maxpat : main Max/MSP file that is used to process data and play the sounds.
