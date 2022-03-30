# Compressing Generative Spaces With Dimmensionality Reduction

A platform for compressing the generative spaces of PCG systems for a selection of 2D tile-based games and validating the compressed spaces against Behavioral Characteristics of the input levels.

![MarioMCAFullVisual](https://user-images.githubusercontent.com/16071406/160801083-54abec0b-572e-4f55-89f7-515e24607794.png)


# Workflow Steps

1. Import sets of tile-based levels and calculate behavioral characteristics of those levels based on their representations
2. Assemble the level sets into dataframes in which every row represents a level and every column represents a tile or tile+tiletype in the levels
3. Apply one of a selection of dimmensionality reduction algorithms to the compressed dataframe and select the two most explanatory new variables for visualisation
4. Calculate the linear correlation between the pair-wise vector distances for levels in the compressed space, with their difference in behavioral characteristic values

#Installation

- To write

##Requirements

- To write 

#Usage 

##Basic Operation

GenSpaceCompression_Main.py is the master file for running experiments.

To initialise a new experiment the following parameters need to be set:

games - Specify the list of game domains that you will be analysing. Set as an array of Game enums (i.e [Game.Boxoban, Game.Mario])
algolist = Specify the list of compression algorithms that you wawnt to use. Set as an array of CompressionType enums (i.e [CompressionType.PCA, CompressionType.SVD, CompressionType.MCA])
tot_lvls_evaled = Specify the number of levels you want to be evaluated for each game. Will be chosen randomly from those available
runs_per_game = Number of runs for each combination of game and algorithm (Each run will randomly select different levels to analyse)
visualise = Boolean for whether you want to generate .png images of the compressed spaces
fileprefix =  String name for the folder for creating output files

##Currently Implemented Game Domains

The system is currently capable of importing and processing levels from:
- Super Mario Bros, using the encoding from the Mario AI Framework(https://github.com/amidos2006/Mario-AI-Framework)
- Loderunner, using the encoding from the VGLC (https://github.com/TheVGLC/TheVGLC)
- Sokoban/Boxoban, using the encoding used in Deepmind's research (https://github.com/deepmind/boxoban-levels)

##Importing a new game level type
- Add it to the Game class in EnumsAndConfig.py
- Add the heigh and width of its levels to get_level_heightandwidth_for_game() in EnumsAndConfig.py
- Add any game specific Behavioral Characteristics to the BCType enum in EnumsAndConfig.py
- Add a new dictionary to EnumsAndConfig.py, mapping each possible character in the level representation to a unique integer 
- Add a dictionary of folder locations for the input files to EnumsAndConfig.py
- Create a new level class file which inherits from LevelWrapper. Needs a calc_behavioral_features(self, charrep) method to calculate its behavioral characteristics (See MarioLevel.py for an example)
- Add the new game, the tile type dict and folder dict to the get_folder_and_tiletypedict_for_game() method in HelperMethods.py
- Add game and levelwrapper constructor to generate_levelwrapper_for_game() in helpermethods.py

##Adding a new compression algorithm

#Limitations

- Every level representation has to be the same size, as the majority of compression algorithms expect each observation to have the same number of variables 

#Bugs and Issues to Address

- Lots of game attributes and similar are stored in variables and functions when they should be stored in a more sophisticated data structure (See the number of steps required to add a new game type). 


#Citation

If you use this platform in your research or writing, please cite it as:

@misc{genspacecompression,
author = {Oliver Withington},
title = {Compressing and Understanding Generative Spaces: Experiment Platform},
howpublished= {https://github.com/KrellFace/Generative-Space-Compression/},
year = "2022",
}
