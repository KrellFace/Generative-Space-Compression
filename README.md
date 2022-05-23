# Compressing Generative Spaces With Dimmensionality Reduction

A platform for compressing the generative spaces of PCG systems for a selection of 2D tile-based games and validating the compressed spaces against Behavioral Characteristics of the input levels.

![MarioMCAFullVisualCropped](https://user-images.githubusercontent.com/16071406/161237863-cb730510-34e8-47de-a30b-f0e2f47863ef.png)

# Workflow Steps

1. Import sets of tile-based levels and calculate behavioral characteristics of those levels based on their representations
2. Assemble the level sets into dataframes in which every row represents a level and every column represents a tile or tile+tiletype in the levels
3. Apply one of a selection of dimmensionality reduction algorithms to the compressed dataframe and select the two most explanatory new variables for visualisation
4. Calculate the linear correlation between the pair-wise vector distances for levels in the compressed space, with their difference in behavioral characteristic values

# Installation

To install and repeat the experiments you simply need to clone this repository, and then clone the respositories for the game domains of interest (implemented options listed below) into the inputLevels folder. Then after confirming that the folder structure for each game matches what is specified for the root/path variables hard coded in src/config/enumAndConfig then you should be ready to configure the variables in GenSpaceCompression_Main.py and run the experiment.

# Requirements

- Python 3.8
- VSCode

## Python Packages
-scikit-learn (https://scikit-learn.org/stable/) 
-scipy (https://scipy.org/)
-prince (https://pypi.org/project/prince/)
-pandas (https://pandas.pydata.org/)
-numpy (https://numpy.org/)


# Usage 

## Basic Operation

GenSpaceCompression_Main.py is the master file for running experiments.

To initialise a new experiment the following parameters need to be set:

games - Specify the list of game domains that you will be analysing. Set as an array of Game enums (i.e [Game.Boxoban, Game.Mario])  
algolist = Specify the list of compression algorithms that you wawnt to use. Set as an array of CompressionType enums (i.e [CompressionType.PCA, CompressionType.SVD, CompressionType.MCA])  
tot_lvls_evaled = Specify the number of levels you want to be evaluated for each game. Will be chosen randomly from those available  
runs_per_game = Number of runs for each combination of game and algorithm (Each run will randomly select different levels to analyse)  
visualise = Boolean for whether you want to generate .png images of the compressed spaces  
fileprefix =  String name for the folder for creating output files  

## Currently Implemented Game Domains

The system is currently capable of importing and processing levels from:
- Super Mario Bros, using the encoding from the Mario AI Framework(https://github.com/amidos2006/Mario-AI-Framework)
- Loderunner, using the encoding from the VGLC (https://github.com/TheVGLC/TheVGLC)
- Sokoban/Boxoban, using the encoding used in Deepmind's research (https://github.com/deepmind/boxoban-levels)


# Limitations

- Every level representation has to be the same size, as the majority of compression algorithms expect each observation to have the same number of variables 

# Issues to Address

- Storage of game attributes and similar in variables and functions in EnumsAndConfig.py and HelperMthds.py modules. 
- Easier workflow for adding new games and compression algorithms


# Citation

If you use this platform in your research or writing, please cite it as:

@misc{genspacecompression,  
author = {Oliver Withington},  
title = {Compressing and Understanding Generative Spaces: Experiment Platform},  
howpublished= {https://github.com/KrellFace/Generative-Space-Compression/},  
year = "2022",  
}
