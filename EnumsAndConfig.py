from enum import Enum

class Game(Enum):
    Mario = 1,
    Loderunner = 2,
    Boxoban = 3 

#Hard coding the size of window we grab from each level for each game type
mario_width = 200
mario_height = 16
boxoban_width = 10
boxoban_height = 10
loderunner_width = 32
loderunner_height = 22

def get_level_heightandwidth_for_game(game):
    if (game == Game.Mario):
        return [mario_height, mario_width]
    elif (game == Game.Boxoban):
        return [boxoban_height, boxoban_width]
    elif (game == Game.Loderunner):
        return [loderunner_height, loderunner_width]

class CompressionType(Enum):
    PCA = 1,
    MCA = 2,
    SVD = 3,
    TSNE = 4,
    KPCA_POLY = 5,
    KPCA_RBF = 6,
    KPCA_SIGMOID = 7,
    SparsePCA = 8,
    KPCA_COSINE = 9

class BCType(Enum):
    EmptySpace = 1,
    EnemyCount = 2,
    Linearity = 3,
    Contiguity = 4

lr_tiletypes_dict = {
    "CountOfNumericTileTypes" : int(8),
    "B":int(0), "b":int(1), ".":int(2), "-":int(3), "#":int(4), "G":int(5), "E":int(6), "M":int(7)
}

#Modifications:
#Start and end pos are empty 0; All enemies are the same, 1 ; Apart from bullet bill = empty space; All solid blocks = 2; All pipe blocks = 3; All blocks containing rewards = 4
mario_tiletypes_dict_condensed = {
    "CountOfNumericTileTypes":int(5),
    #Empty space, including start and end tiles, and bullet bills
    "-":int(0),"M":int(0),"F":int(0),"|":int(0),"*":int(0),"B":int(0),"b":int(0),
    #All enemies apart from bulletbills
    "y":int(1),"Y":int(1),"E":int(1),"g":int(1),"G":int(1),"k":int(1),"K":int(1),"r":int(1),
    #Solid blocks
    "X":int(2),"#":int(2),"%":int(2),"D":int(2),"S":int(2),
    #Pipe blocks
    "t":int(3),"T":int(3),"<":int(3),">":int(3),"[":int(3),"]":int(3),
    #Reward blocks
    "?":int(4),"@":int(4),"Q":int(4),"!":int(4),"1":int(4),"2":int(4),"C":int(4),"U":int(4),"L":int(4),"o":int(4)
}

#Dictionary of tiletypes in boxoban, the google deepmind clone of sokoban

boxoban_tiletypes_dict = {
    "CountOfNumericTileTypes" : int(5),
    "#":int(0), "@":int(1), "$":int(2), ".":int(3), 
    " ":int(4)
    #"E":int(4)
}

#Level file locations
mario_root = 'C:/Users/owith/Documents/External Repositories/Mario-AI-Framework/levels/'
loderunnder_path = "C:/Users/owith/Documents/External Repositories/VGLC/TheVGLC/Lode Runner/Processed/"
boxoban_root = "C:/Users/owith/Documents/External Repositories/boxoban-levels/"

#Dictionary of Mario generator names and respective folders 
mario_folders_dict = {
    'Notch' : (mario_root + 'notch/'),
    'Notch_Param': (mario_root + 'notchParam/'),
    'Notch_ParamRand': (mario_root + 'notchParamRand/'),
    'GE': (mario_root + 'ge/'),
    #'Original': (mario_root + 'original/'),
    'Hopper': (mario_root + 'hopper/'),
    'Ore': (mario_root + 'ore/'),
    'Pattern_Count': (mario_root + 'patternCount/'),
    'Pattern_Occur': (mario_root + 'patternOccur/'),
    'Pattern_WeightCount': (mario_root + 'patternWeightCount/')
}

loderunnder_folders_dict = {
    'Processed': loderunnder_path
}

boxoban_folders_dict = {
    'Medium' : (boxoban_root + 'medium/train/'),
    'Hard' : (boxoban_root + 'hard/'),
    'unfiltered': (boxoban_root +'unfiltered/train/')
}

color_dict = dict({0:'brown',
                1:'green',
                2: 'orange',
                3: 'red',
                4: 'dodgerblue',
                5: 'darkmagenta',
                6: 'fuchsia',
                7: 'lime',
                8: 'cyan',
                9: 'cadetblue'})
