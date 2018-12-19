This repository contains the code and data for the final project in DD2418.

This is the directory structure

data/ : contains the dataset we use
models/ : contains the model weights of the trained models, and the weights for the pretrained BERT in models/pretrained/
results : output directory for graphs and training history
src/ : contains the source code. The python files are:
    Attention.py : The file defining the Keras attention layer
    DataLoader.py : module that does the data preprocessing, conversion of sentences to dense representations with BERT, and generationg the                      input for the networks
    fully_connected_classification.py : contains the baseline fully connected network
    hierarchical_attention_classification.py : contains the bidirectional GRU with attention classification network
    hierarchical_classification.py : contains the bidirectional GRU classification network

The required packages to run the code are in requirements.txt
The code is written in Python 3.5
Training took place in a machine with 32GB of RAM, 4 CPU cores and a Tesla P100

The experiments are contained within the three scripts that define the networks.
To run any experiment, at first you have to start the BERT provider server by running ./start_BERT_server.sh
When the server indicates that it's ready, you may run any of the experiments with python [script_name.py]
