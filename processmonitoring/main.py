import os
import sys
import json
import logging

from processmonitoring.experiments import runner

def _start_logger():
    logging.basicConfig(filename = 'file.log',
                        level = logging.INFO,
                        format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')

def _create_new_folder(directory_path: str):
    """
    Opens a new folder inside the config file directory with the _plots suffix to store all 
    plots made during experiments.

    Args:
    directory_path (str): The path to the directory in which the new folder should be created.
    
    Returns:
    str: The full path of the newly created folder.
    """
    new_folder_name = os.path.splitext(os.path.basename(directory_path))[0] + '_plots'
    
    # Create the full path for the new folder
    new_folder_path = os.path.join(os.path.dirname(directory_path), new_folder_name)
    
    # Check if the new folder already exists
    if os.path.isdir(new_folder_path):
        return new_folder_path
    
    # Attempt to create the new folder
    try:
        os.mkdir(new_folder_path)
        return new_folder_path
    except OSError:
        raise ValueError("Failed to create the new folder.")

def main(argv):
    """
    Program entry point. 
    Opens a logger.
    Attempts to load a config file from directory given as commandline argument.
    Generates folder to store plots in same directory as config file.
    Hands config file to experiment runner class.

    Args:
        argv (str): Absolute to config file. Only single argument supported.
    """

    _start_logger()

    try:
        with open(argv, 'r') as f:
            config = json.load(f)
    except Exception:
        logging.exception('Unable to find or open json config file.')
        raise

    try:
        plot_dir = _create_new_folder(argv)
    except:
        # just continue without plotting
        logging.exception("Exception during folder creation.")

    try:
        exp = runner.ExperimentRunner(config, plot_dir if plot_dir else None)
        exp.run()
    except:
        logging.exception("Exception occurred somewhere else in experiment. Traceback below")
        raise

if __name__ == '__main__':
    main(sys.argv[1])