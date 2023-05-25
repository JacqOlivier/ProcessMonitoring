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
Experiments are configured with a configuration file. The path to this configuration file is then passed as the only commandline argument to the main.py method (program entry point). A VS Code launch file is also added in the repository as an example. Example config files can also be found in the TestConfigdirectory.

From the commandline:

`python /path/to/main.py /path/to/config/file`

Attempt to run the test config files from within the project folder. These will each run an experiment using different permutations and a random forest feature extractor. The dataset features will be projected to a feature space and 2D plots will be saved.

The program will attempt to make a folder with the same directory as the config file, with the '_plots' suffix. If successful, the program will save all generated plots inside this folder.

Currently, this will run a single experiment and exit. Future development will include writing a wrapper module for hyperparameter gridsearches or optimisation. Provision will also be made for parallell processing to run multiple instances of each experiment with different Pseudo-RNG seeds to test the robustness of the methods.

## Project structure
An experiment consists of the following components:

* dataset with a known fault
* a permutation to apply to the dataset
* a feature extraction method

These components have all been implemented as separate modules in the package. The dataset and feature extraction modules contain base classes from which all dataset and feature extraction implementations must inherit. 

This allows for easy addition of additional methods and datasets, using the base class as a template.

Datasets must inherit from:     `processmonitoring.datasets.dataset.DatasetWithPermutations`

Feature Extractors must inherit from:

`processmonitoring.feature_extraction.GenericFeatureExtractor.FeatureExtractor`

The added class must also be given a `@register_function(config_file_name)` decorator and name to identify with in the config file. This enables the factory methods inside each class.

## Future:

Currently developing a statistical process control module that will project the feature set to T2 and SPE space for monitoring. 

Decide on experiment datasets and add these as classes for testing.

Hyperparameter searches will be required since the algorithms and experiments have many possible configurations. One option is bayesian minimisation using the [Chocolate](https://chocolate.readthedocs.io/en/latest/) library. But this will require formal metrics for optimisation, as well as significant computing power and multiprocessing support.

CNN and AutoEncoder feature extractors still need to be reworked, and are not currently functional. When using NN's, a decision on small architecture decisions, such as additional layers between a bottle neck in autoencoders and input layers etc., needs to be made.I ​use github (and git) for version control. If you would like to contribute or make any changes, please make these in a new branch and send a pull request, so that it can be reviewed and merged into the main branch.

## Contributing

If you would like to contribute or make any changes, please make these in a new branch and send a pull request so that it can be reviewed and merged into the main branch.
