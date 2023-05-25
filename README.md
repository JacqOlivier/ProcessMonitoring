# Process Monitoring

## About
A Python package to test various feature extraction methods on time series data. 

The package enables using supervised learning algorithms for the normally unsupervised process monitoring (Statistical Process Control - SPC) task. This is done using permuted or surrogate copies of the original data. 

The main objective of the experiments is to identify whether generating these surrogate copies from different data generating distributions improves the quality of featured extracted by the supervised learning models. The quality of the features can either be inspected visually, or quantified using an SPC framework.

## Setup

Setup steps for this stage of the project:

1. Make sure you have git installed and clone the package into desired directory using the commandline:

    ```git clone https://github.com/JacqOlivier/ProcessMonitoring.git``` 

2.  Create a Python virtual environment, from the commandline:

    `python -m venv env`

3. Activate the virtual environment (for windows) from commandline:

    `./env/Scripts/activate`

4. Install required packages in the requirements.txt file:

    `pip install -r /path/to/package/requirements.txt`

5. From inside the project folder (with setup.py), install the processmonitoring package using pip:

    `pip install -e .`

This will install the package in editable mode, important for development.

## Running the program
Experiments are configured with a configuration file. The path to this configuration file is then passed as the only commandline argument to the run.py method (program entry point). A VS Code launch file is also added in the repository as an example.
## Project structure
An experiment consists of the following components:

* dataset with a known fault
* a permutation to apply to the dataset
* a feature extraction method

