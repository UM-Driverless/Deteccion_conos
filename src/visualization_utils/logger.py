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
            print('LOGGER STARTED')


    def write(self, message: str, print_msg=False):
        """
        Appends a message to the file in self.path and optionally prints it.

        Parameters:
            message (str): The message to write to the file.
            print_msg (bool): If True, the message will also be printed.

        Returns:
            None
        """
        with open(self.path, 'a', encoding='utf-8') as f:
            # Use the full date
            timestamp = f'{datetime.datetime.now().strftime("%Y_%m_%d__%H:%M:%S.%f")}'
            # Now like before, but with microsecond precision
            formatted_message = f'{timestamp}: {message}\n'
            f.write(formatted_message)
            if print_msg:
                print(formatted_message)
