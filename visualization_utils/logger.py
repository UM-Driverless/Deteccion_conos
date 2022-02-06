import datetime
import os
import sys

class Logger:
    def __init__(self, logs_path, init_message="logger ready"):
        # Create a file for debug logs
        date = datetime.datetime.now()
        self.debug_path = os.path.join(logs_path, date.strftime("%Y_%m_%d__%H_%M_") + 'debug_logger' + '.txt')

        if not os.path.exists(logs_path):
            # Create a new directory because it does not exist
            os.makedirs(logs_path)

        # Create a file for run logs. No consigo que funcione para recoger lo que se pinta por pantalla.
        # self.run_path = os.path.join(logs_path, date.strftime("%Y_%m_%d__%H_%M_") + 'run_logger' + '.txt')
        # sys.stdout = Unbuffered(sys.stdout, self.run_path)

        self.write_main_log("Initialization. " + init_message)

    def write_main_log(self, msg):
        with open(self.debug_path, 'a', encoding="utf-8") as f:
            f.write("[MAIN]: " + str(msg) + " [" + str(datetime.datetime.now().time()) + "]" + '\n')

    def write_can_log(self, msg):
        with open(self.debug_path, 'a', encoding="utf-8") as f:
            f.write("[CAN]: " + str(msg) + " [" + str(datetime.datetime.now().time()) + "]" + '\n')

class Unbuffered:
    def __init__(self, stream, path):
        self.stream = stream
        self.te = open(path, 'w')  # File where you need to keep the logs

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        self.te.write(data)    # Write the data of stdout here to a text file as well

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass