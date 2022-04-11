# Roman Numeral Classification via Multi-Headed Self-Attention

## Description

This is a forked branch of Micchi et al.'s _[functional-harmony](https://gitlab.com/algomus.fr/functional-harmony)_ repository. 
This branch contains code used in the development of Brenner Swenson's Master's Thesis for City University of London, 
titled: _Automatic Harmonic Analysis: Roman Numerals via Deep Learning_. This project sought to implement self-attention based
models in the context of Roman numeral classification.

Note:
It was forked in May of 2021 and does not reflect updates made to the parent repository after said date.

## Repository Structure
Micchi et al.'s parent repository contains a large amount of bespoke code used to convert, manipulate, and augment piano roll
data prior to ingestion by a machine learning model. In avoidance of redundancy, the _[functional-harmony](https://gitlab.com/algomus.fr/functional-harmony)_ 
pre-processing code was highly leveraged in the research and development of this project's models. The directory structure of this project's source code is found below.

The `src/micchi_et_al` directory contains all of the pre-existing code leveraged during this project, save for the `src/shared_config.py` file that contains constants and various 
configurations used by both this project's models and Micchi et al.'s models. The `src/encoder_model` directory contains the majority of this project's source code.
There are many functions and model components copied/pasted from various reputable sources like [TensorFlow's Transformer tutorial](https://www.tensorflow.org/text/tutorials/transformer),
and their [tensor2tensor](https://github.com/tensorflow/tensor2tensor) Python library. Whenever outside code is used, the source URL is cited in the relevant function/method's docstring.

```markdown
src/
├── micchi_et_al  # Micchi et al.'s source code with comments and refactored import statements for compatability
│   ├── README.md
│   ├── __init__.py
│   ├── data_manipulation  # code related to data preprocessing, analysis, and manipulation
│   │   ├── __init__.py
│   │   ├── bps_conversion.py
│   │   ├── dataset_analysis.py
│   │   ├── krn2xml.py
│   │   ├── load_data.py  # functions to load TFRecords datasets
│   │   ├── playground.py
│   │   ├── preprocessing.py  # primary preprocessing module
│   │   └── train_validation_split.py  # split TFRecords in to train/validation
│   ├── models  # module related to model construction and training
│   │   ├── __init__.py
│   │   ├── key_detection_temperley.py
│   │   ├── model.py  # where Micchi et al.'s model is defined
│   │   ├── run_full.py  # script to use trained model to analyse provided scores
│   │   ├── run_model.h5
│   │   └── train.py  # model training script
│   ├── tests/  # unit tests
│   └── utils
│       ├── __init__.py
│       ├── analyse_results.py  # used to analyse results of model outputs and compare models
│       ├── converters.py
│       ├── utils.py  # various utils functions
│       └── utils_music.py  # utils related to musical concepts
├── __init__.py  
├── encoder_model  # source code used to construct and analyse the encoder-based RN model of this project
│   ├── __init__.py
│   ├── attention.py  # defines the self-attention portion components of the model
│   ├── attention_utils.py  # utils functions related to self-attention calculation.
│   ├── encoder_layer.py  # module containing the EncoderLayer class
│   ├── encoder_model.py  # the RN Encoder model is defined in this module, as well as functions for constructing it
│   ├── encoder_utils.py  # utils functions for constructing RN Encoder model.
│   ├── performance.py  # code related to the analysis of the RN model's outputs. 
│   └── utils.py  # utils functions related to the RN model
├── encoder_grid_search.py  # executable script to run grid search to find optimal hyperparameter configurations
├── encoder_test.py  # script to evaluate this project's best model
├── encoder_train.py  # code to instantiate and train an individual RN model. Called by grid search module.
├── grid_search_config.py  # file where hyperparameter options are defined for grid search
├── shared_config.py  # micchi et al constants that are shared between their model's and ours
└── trained_models  # where models and their analyses are saved
```

## Model Architecture
Below is the final model architecture achieved in this project. It consists of a stack of EncoderLayers 
for each task, where certain tasks are informed by the outputs of others. Arrows in the diagram indicate which tasks inform subsequent ones.
![Alt text](assets/RN_encoder_diagram.png?raw=true "Model Architecture")


## Using This Repo

### Environment and Dependencies
This project was developed using Python 3.6.9 and TensorFlow 2.4.1 on a Windows operating system. To ensure full compatability with other 

To set up your environment, ensure you are in the working directory, and run:
- `python -m venv venv` - This sets up a virtual environment
- `source venv/Scripts/activate` - Activate the virtual environment
- `pip install -r requirements.txt` - Install all of this project's exact dependencies

### Training
Prior to training a model, ensure that you have ran `src/micchi_et_al/data_manipulation/preprocessing.py`, as well as `train_validation_split.py` in the same directory.
Those modules will iterate through the dataset and create TFRecord files in lengths of `CHUNK_SIZE`. 

The score XML and chord annotation files are locaed in the `/data` directory. Due to GitHub's limitations
on dataset sizes, the already processed TFRecords for train/validation could not be uploaded. These data
will be provided separately in the submission via a OneDrive link.

To train a model with the best configurations identified during this project, run this command:
- `python -m src.encoder_train`

The above command will use the default values of the argparse arguments, and train the model as it was trained during this project. 


### Validation
To evaluate a model that has been saved to the `trained_models` directory, simply run:
- `python -m src.encoder_test`

The above command will execute for the best model achieved during this project. It will load the `.h5` model file, then generate predictions all validation TFRecords. 
The outputs are percentage accuracies, F1, recall, and precision scores for various RN metrics discussed in the accompanying report. It will also generate confusion matrices
for the predictions of each individual RN task. The matrices are saved as .png files in the directory the test script is ran from. 

