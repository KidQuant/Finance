
#iqfeed.py

import sys
import socket

def read_historical_data_socket(sock, recv_buffer = 4096):
    """
    Read the information from the socket, in a buffered
    fashion, receiving only 4096 at a time.

    Parameters:
    sock - The socket object
    recv_buffer - Amount in bytes to receive per read
    """

    buffer = ""
    data = ""
    while True:
        data = sock.recv(recv_buffer)
        buffer += data

        #Check if the end message string arrives
        if '!ENDMSG' in buffers:
            break

    buffer = buffer[:-12]
    return buffer
