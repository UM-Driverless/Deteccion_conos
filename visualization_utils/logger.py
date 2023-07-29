import datetime
import os
import sys

class Logger:
    '''
    Appends to (or creates) log file ./logs/{year}_{month}_{day}__{hour}_{minute}.txt
    Writes a line with {time of day}: {message}
    
    Use example:
    logger = Logger()
    logger.write(f'Temperature: {temp}')
    '''
    def __init__(self, logs_path = os.path.join(os.getcwd(), 'logs')):
        self.path = os.path.join(logs_path, datetime.datetime.now().strftime('%Y_%m_%d__%H_%M') + '.txt')

        # Create the 'logs' dir if it doesn't exist
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
            self.write(f'File created by Logger class at {datetime.datetime.now().strftime("%Y_%m_%d__%H_%M")}')


    def write(self, message):
        # Appends message to the file in self.path, which remains constant during the execution
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(f'{str(datetime.datetime.now().time())}: {str(message)}\n')
