##=================================================== mainlive.py ==================================================##
'''
# Collect live data from Max/MSP, and send the result of the gesture to Max/MSP. Also can generate data to train
the model
'''

# Imports
import argparse
import random
import pandas as pd
from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc import udp_client
import socket
import struct
import loadmodel

# Global variables
global data_for_eval
global data_line
#global nb_gest
global line_nb

data_for_eval = []
line_nb = 0
data_line = []
#nb_gest = 145


# Parameters
threshold = 127

'''
# Main function : it will run every time the Max/MSP server sends data
'''
def main(path: str, *osc_arguments):
    global data_line
    global data_for_eval
    global line_nb
    #global nb_gest

    msg = osc_arguments[-1]
    #print("input message: {}".format(msg))
    #print("path: {}".format(path))

    if path == '/accx' and len(data_line) != 0:
        '''
        # We have gathered all data for a single line (Max/MSP sends data in the same order so every time the first
        one appears, it's a new line of data).
        '''
        data_for_eval.append(data_line)
        line_nb += 1
        data_line = []

    '''
    # Save the received data
    '''
    data_line.append(msg)

    if line_nb > threshold:
        '''
        # We have gathered enough lines to pass them through the gesture recognition process
        '''
        df = pd.DataFrame(data_for_eval, columns=['/accx', '/accy', '/accz', '/gyrox', '/gyroy', '/gyroz'])
        df = df.iloc[:128]

        msgOUT = loadmodel.evaluate(df)
        data_for_eval = []
        line_nb = 0

        ''' 
        Data generation 
        '''
        # print("Gesture generation "+str(nb_gest))
        # df.to_csv('gesture'+str(nb_gest)+'.csv', index=False)
        #nb_gest += 1

        ''' 
        Send the gesture result to Max/MSP
        '''
        print("output message: {}".format(msgOUT))
        ipOUT = osc_arguments[0][0] # '192.168.0.10'
        portOUT = osc_arguments[0][1]
        pathOUT = osc_arguments[0][2]
        talk2Max(ipOUT,portOUT,pathOUT,str(msgOUT))

def listen2Max(addrIN,addrOUT):
    '''
    Set up server
    '''
    # input address
    ipIN   = addrIN[0]
    portIN = addrIN[1]
    pathIN = addrIN[2]
    # output address
    portOUT = addrOUT[0]
    pathOUT = addrOUT[1]
    # dispatcher to receive message
    disp = dispatcher.Dispatcher()
    disp.map(pathIN, main, ipIN, portOUT, pathOUT)
    # server to listen
    server = osc_server.OSCUDPServer((ipIN,portIN), disp)
    print("Serving on {}".format(server.server_address))
    server.serve_forever()

    '''
    Other method
    '''
    # s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # server_address = (ipIN, portIN)
    # s.bind(server_address)
    # data, address = s.recvfrom(5096)
    # print("\n\n Server received: ", data.decode('utf-8'), "\n\n")
    # decoded = str(data,encoding='utf-8')
    # print(decoded)
    # print(" Server address : ", address, "\n\n")



def talk2Max(ip,port,path,message):
    '''
    Set up client and send message
    '''
    client = udp_client.SimpleUDPClient(ip,port)
    client.send_message(path, message)
    print('message sent')

if __name__ == "__main__":
    # generate parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-II","--ipIN", type=str, default="127.0.0.1", help="The ip to listen on")
    parser.add_argument("-PI", "--portIN", type=int, default=5005, help="The port to listen on")
    parser.add_argument("-UI", "--uripathIN", type=str, default="*", help="MAX's URI path")
    parser.add_argument("-PO", "--portOUT", type=int, default=5006, help="The port to send messages to")
    parser.add_argument("-UO", "--uripathOUT", type=str, default="/filter", help="output URI path")
    args = parser.parse_args()
    # wrap up inputs
    outputAddress = [args.portOUT, args.uripathOUT]
    inputAddress = [args.ipIN, args.portIN, args.uripathIN]
    # listen to max
    while True:
        listen2Max(inputAddress, outputAddress)
