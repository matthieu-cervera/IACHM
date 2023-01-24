##================================================ IACHMLoadmodel.py ===============================================##

# This loads the model to evaluate gestures. You can load the magic wand model (took from ) or load our LSTM
# based model trained with google GPU and our database of gestures.

# Imports
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn

# Our LSTM based model - we load a trained checkpoint to it.
class LSTMModel(nn.Module):
    def __init__(self, device):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(
            input_size=6, hidden_size=12, batch_first=True)
        self.lstm = nn.LSTM(
            input_size=12, hidden_size=256,batch_first=True,dropout = 0.2)
        self.linear = nn.Linear(256, 3)
        self.sigmoid = nn.Sigmoid()
        self.device = device

    def forward(self, x):
        # h0 = torch.zeros((1, 32, 1024)).to(self.device)
        # c0 = torch.zeros((1, 32, 1024)).to(self.device)
        x, _ = self.lstm1(x)

        # h1 = torch.zeros((1, x.size(0), 256)).to(self.device)
        # c1 = torch.zeros((1, x.size(0), 256)).to(self.device)
        x, (ht, ct) = self.lstm(x)
        # print('x_shape:',x.shape)
        x = self.linear(ht[-1])
        x = self.sigmoid(x)
        return x

model = LSTMModel('cpu')
state_dict = torch.load('checkpoint_97.pth',map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

##======================== MAGIC WAND MODEL =============================##
# Load TFLite model and allocate tensors.
#interpreter = tf.lite.Interpreter(model_path="Magic_wand_model.tflite")
## Get input and output tensors.
#input_details = interpreter.get_input_details()
#output_details = interpreter.get_output_details()
#interpreter.allocate_tensors()
##=======================================================================##

def evaluate(df):
    ##======================== MAGIC WAND MODEL ==============================##
    ## Test model on random input data.
    #input_shape = input_details[0]['shape']
    #input_data = df.to_numpy(dtype=np.float32)
    #input_data = input_data.reshape(input_shape)
    #interpreter.set_tensor(input_details[0]['index'], input_data)
    #interpreter.invoke()
    ## The function `get_tensor()` returns a copy of the tensor data.
    ## Use `tensor()` in order to get a pointer to the tensor.
    #output_data = interpreter.get_tensor(output_details[0]['index'])
    #return(np.argmax(np.copy(output_data)))
    ##========================================================================##

    ##============================ OUR MODEL =================================##
    df.dropna(inplace=True)
    numpy_data = df.to_numpy(dtype=np.float32)
    input_data = torch.from_numpy(numpy_data).float()
    output = model(input_data)
    return(torch.argmax(output))
    ##=======================================================================##