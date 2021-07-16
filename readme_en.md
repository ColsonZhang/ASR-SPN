# The ASR based on the SPN

## Introduction

This project is a SPN solution for the re-micro-cup digital one-game problem - that is, the use of SPN network to build an isolated word speech recognition model. The project file gives the code for waveform data pre-processing and the code for model training.

## The project catalog

```
----ASR\
    |----PreProcess\           
    |    |----preprocess.py    
    |----Train\                
    |    |----datasplit.py     
    |    |----dtwsplit.py      
    |    |----train_spn.ipynb  
    |----requirements.txt       
    |----readme.md
```

## Instructions for use

1. Environmental requirements: Python3.8, you need to install the pypi package in requirement.txt, you can install it directly through the command `pip install -r requirements.txt`

2. Data preparation: Please prepare the data collection of Google Speech Command and place it in folders according to the label;

3. Data preprocessing: open the file `preprocess.py` to modify the path of the input waveform data file and configure the output path;

4. Model training: When performing model training, configure the path in each file in the `./Train/` directory, and then `execute `dtwsplit.py`, `datasplit.py` and `train_spn.ipynb` files in sequence. Among them, please use jupyter lab/notebook to open `train_spn.ipynb`, and execute them in order from top to bottom.

5. Save model: There is a function to export SPN model in `train_spn.ipynb`, you only need to configure the path.