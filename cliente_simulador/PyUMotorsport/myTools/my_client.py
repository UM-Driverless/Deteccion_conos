'''
Plugin to establish connection and exchange text messages between two platforms. Producing
this communication on the same machine.
'''
import socket

def bind2server(PORT=12345):
    '''
    Establishes a connection with the server, which is listening on port 12345 of this machine.
    '''
    HOST = socket.gethostname()
    HOST = socket.gethostbyname(HOST)
    print(f'IP: {HOST}, Port: {PORT}')

    # Create a socket (SOCK_STREAM means a TCP socket)
    mySocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # mySocket = socket.socket(socket.AF_IPX, socket.SOCK_STREAM)
    # HOST = '127.0.1.1:12345'
    # mySocket.connect(HOST)
    mySocket.connect((HOST, PORT))
    return mySocket

def send_msg(mySocket, data):
    mySocket.sendall((data + "\n").encode())

def receive_msg(mySocket):
    return mySocket.recv(1024)

def print_msg(data, received):
    print("Sent:     {}".format(data))
    print("Received: {}".format(received))

# from my_client import bind2server, send_msg, receive_msg, print_msg
'''
Script, communication test.
'''
if __name__ == '__main__':
    print("Running client test")
    mySocket = bind2server()
    try:
        data = "hola"
        send_msg(mySocket, data)
        serverMsg = receive_msg(mySocket)
        print_msg(data, serverMsg)

        data = "adios"
        send_msg(mySocket, data)
        serverMsg = receive_msg(mySocket)
        print_msg(data, serverMsg)
    finally:
        mySocket.close()

