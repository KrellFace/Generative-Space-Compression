import glob
from math import nan
from enum import Enum
import os
import math
import random
from turtle import distance
#from statistics import LinearRegression
import matplotlib
from datetime import datetime
from more_itertools import difference
import numpy as np
import pandas as pd
from pyparsing import col
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import KernelPCA
from sklearn import linear_model
from sklearn.manifold import TSNE
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import prince
import itertools as it 
import timeit
import copy
from io import BytesIO
from csv import writer 
from LevelWrapper import LevelWrapper
from BoxobanLevel import BoxobanLevel
from MarioLevel import MarioLevel
from LoderunnerLevel import LoderunnerLevel

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

class CompressionType(Enum):
    PCA = 1,
    MCA = 2,
    SVD = 3,
    TSNE = 4,
    PCATSNE = 5,
    MCATSNE = 6,
    SVDTSNE = 7,
    SparsePCA = 8,
    KernelPCA = 9

class BCType(Enum):
    EmptySpace = 1,
    EnemyCount = 2,
    Linearity = 3

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
    "#":int(0), "@":int(1), "$":int(2), ".":int(3), " ":int(4)
}

#Level file locations
mario_root = 'C:/Users/owith/Documents/External Repositories/Mario-AI-Framework/levels/'
loderunnder_path = "C:/Users/owith/Documents/External Repositories/VGLC/TheVGLC/Lode Runner/Processed/"
boxoban_root = "C:/Users/owith/Documents/External Repositories/boxoban-levels/"

#Dictionary of Mario generator names and respective folders 
mario_folders_dict = {
    'Notch_Param': (mario_root + 'notchParam/'),
    'GE': (mario_root + 'ge/'),
    #'Original': (mario_root + 'original/'),
    'Hopper': (mario_root + 'hopper/'),
    'Ore': (mario_root + 'ore/'),
    'Pattern_Count': (mario_root + 'patternCount/')
}

loderunnder_folders_dict = {
    'Processed': loderunnder_path
}

boxoban_folders_dict = {
    'Medium' : (boxoban_root + 'medium/train/'),
    'Hard' : (boxoban_root + 'hard/'),
    'unfiltered': (boxoban_root +'unfiltered/train/')
}

boxoban_files_dict = {
    '000 - Medium' : (boxoban_root + 'medium/train/000.txt'),
    '001 - Medium' : (boxoban_root + 'medium/train/001.txt'),
    '000 - Hard' : (boxoban_root + 'hard/000.txt'),
    '001 - Hard' : (boxoban_root + 'hard/001.txt'),
    '000 - Unfiltered': (boxoban_root + 'unfiltered/train/000.txt'),
    '001 - Unfiltered': (boxoban_root + 'unfiltered/train/001.txt')
}

color_dict = dict({0:'brown',
                1:'green',
                2: 'orange',
                3: 'red',
                4: 'dodgerblue',
                5: 'darkmagenta',
                6: 'lightcoral'})

#######################################
#FILE IMPORTING METHODS
#######################################

#Get a 2D character matrix from a level file 
def char_matrix_from_file(path):
    with open(path) as f:
        charlist = f.read()
        width = 0
        width_calculated = False
        height = 0
        charlist_newlinesremoved = list()

        for char in charlist:
            if char == '\n':
                width_calculated = True
                height+=1
            else:
                charlist_newlinesremoved.append(char)
            if not width_calculated:
                width+=1
        output_matrix = np.reshape(charlist_newlinesremoved,(height, width), order = 'C')
        #print("Level processed: "  + path)
        return output_matrix

#Custom function for boxoban files as they are stored with multiple in each file
#Returns a dictionary of level names and associated character matrixes 
def get_boxoban_leveldict_from_file(file_name):
    level_reps = dict()
    line_counter = 0
    buffer = list()
    temp_levelname = ""
    with open(file_name) as file:
        charlist = file.read()
        for char in charlist:
            if (char == '\n'):
                #Check if we are on a level name line
                if (line_counter%12 == 0):
                    temp_levelname = temp_levelname.join(buffer)
                    buffer.clear()
                #Check if we are at the end of a level rep. If we are, add it to our dictionary
                elif ((line_counter+1)%12 == 0):
                    char_matrix = np.reshape(buffer,(boxoban_height, boxoban_width), order = 'C')
                    #level_reps[int(temp_levelname)] = char_matrix
                    #level_reps[file_dict_key +':'+ temp_levelname] = LevelWrapper(temp_levelname, file_dict_key, char_matrix)
                    new_level = BoxobanLevel(temp_levelname, file_name, char_matrix)
                    #new_level.calc_behavioral_features()
                    level_reps[file_name +':'+ temp_levelname] = new_level

                    temp_levelname = ""
                    buffer.clear()
                
                line_counter+=1
            #Only append numeric characters if we are on the level name line as level names are numbers
            elif (line_counter%12 == 0):
                if (char.isnumeric()):
                    buffer.append(char)
            #If its not a level name line or a newline character, add to buffer
            else:
                buffer.append(char)
            
    return level_reps

#Def get an dict of LevelWrappers from a folder in form (Key: 'Folder + File Name, Value: LevelWrapper)
def get_leveldict_from_folder(path, folder_key, game):
    file_names = get_filenames_from_folder(path)
    window_height, window_width = get_level_heightandwidth_for_game(game)
    #folder_name = os.path.basename(os.path.normpath(path))
    level_reps = dict()

    for level in file_names:
        level_name = os.path.basename(level)
        char_rep = char_matrix_from_file(level)
        char_rep_window = take_window_from_bottomright(char_rep, window_width, window_height)
        level_reps[folder_key +':'+ level_name] = generate_levelwrapper_for_game(game, level_name, folder_key, char_rep_window)

    return level_reps

#Get a randomly selected set of levels of size filecount from a folder
#Needs refactoring as it mirrors a lot of get_levledict_from_folder
def get_randnum_levelwrappers_folder(path, folder_key, game, count):
    file_names = get_filenames_from_folder(path)
    window_height, window_width = get_level_heightandwidth_for_game(game)
    #folder_name = os.path.basename(os.path.normpath(path))
    level_reps = dict()
    counter = 0
    while counter < count:
        #print("Counter at: " + str(counter) + " picking from filename list length:  " + str(len(file_names)))
        #Pick random file and then remove it from the list
        level = random.choice(file_names)
        file_names.remove(level)
        level_name = os.path.basename(level)
        char_rep = char_matrix_from_file(level)
        char_rep_window = take_window_from_bottomright(char_rep, window_width, window_height)
        level_reps[folder_key +':'+ level_name] = generate_levelwrapper_for_game(game, level_name, folder_key, char_rep_window)
        counter+=1

    return level_reps


#Get a combined levelwrapper dictionary from a folder dictionary
def get_leveldicts_from_folder_set(game):
    level_dict = dict()
    game_info = get_folder_and_tiletypedict_for_game(game)
    folder_dict = game_info['Folder_Dict']

    for folder in folder_dict:
        #Get all one for for specific folder
        temp_dict = get_leveldict_from_folder(folder_dict[folder], folder, game)
        level_dict = level_dict|temp_dict
    return level_dict

def get_randnum_levelwrappers(game, count):
    level_dict = dict()
    game_info = get_folder_and_tiletypedict_for_game(game)
    folder_dict = game_info['Folder_Dict']

    for folder in folder_dict:
        #Get all one for for specific folder
        temp_dict = get_randnum_levelwrappers_folder(folder_dict[folder], folder, game, count)
        level_dict = level_dict|temp_dict
    return level_dict   

"""
def get_randomlyselected_leveldicts_from_folder_set(game, filecount):
    level_dict = dict()
    game_info = get_folder_and_tiletypedict_for_game(game)
    folder_dict = game_info['Folder_Dict']

    for folder in folder_dict:
        #Get all one for for specific folder
        temp_dict = get_randomlyselected_leveldict_from_folder(folder_dict[folder], folder, game, filecount)
        level_dict = level_dict|temp_dict
    return level_dict

"""

#Get a dictionary of dictionarys (BoxobanFilename: Level Dict) from a Boxoban file
def get_leveldicts_from_boxoban_files(files_dict):
    files_level_dict = dict()
    for file in files_dict:
        temp_dict = get_boxoban_leveldict_from_file(files_dict[file], file)
        #files_level_dict[file] = temp_dict
        files_level_dict = files_level_dict|temp_dict
    return files_level_dict

def get_randnum_levelwrappers_boxoban(folders_dict, count):
    levelwrapper_dict = dict()
    counter = 0
    for folder in folders_dict:
        #List all files in folder
        file_list = get_filenames_from_folder(folders_dict[folder])
        while counter < count:
            #print("Counter at: " + str(counter) + " picking from filename list length:  " + str(len(file_list)) + " for folder: " + folder)
            randfile = random.choice(file_list)
            temp_dict = get_boxoban_leveldict_from_file(randfile)
            file_list.remove(randfile)
            levelwrapper_dict = levelwrapper_dict|temp_dict
            counter+=1
        counter = 0
    return levelwrapper_dict


#############################################
#WNDOW GRABBING  METHODS

#Generic method for taking windows from matrices 
def take_window_from_matrix(input_matrix, top_corner_x, top_corner_y, width, height):

    output_window = np.chararray((height, width), unicode = True)
    for y in range(height):
        output_window[y,] = input_matrix[y+top_corner_y,top_corner_x:(top_corner_x+width)]
    return output_window

#Capture a window of specified size from the bottom right of a matrix
def take_window_from_bottomright(input_matrix, width, height):
    x_corner = input_matrix.shape[1] - width
    y_corner = input_matrix.shape[0] - height
    return (take_window_from_matrix(input_matrix, x_corner, y_corner, width, height))


############################################
#LEVEL PROCESSING METHODS

#Generates a onehot 3D array from a character matrix, using mappings between characters and integers specified in a tile dictionary
def onehot_from_cm_tiletypecountspecified(input_matrix, tile_dict, num_tile_type):
    #Create our empty 3D matrix to populate
    input_shape = np.shape(input_matrix)
    one_hot = np.zeros((input_shape[0], input_shape[1], num_tile_type))

    #Loop through full matrix to populate it
    for x in range(input_shape[0]):
        for y in range(input_shape[1]):
            #print("Setting index " + str(x) +"," +str(y) +"," + str(lr_tiletypes_dict[input_matrix[x,y]]) + " to 1")
            one_hot[x,y,tile_dict[input_matrix[x,y]]] = int(1)

    return one_hot

#Generates a 3D one hot array, using a dictionary field specifying the number of options 
def onehot_from_charmatrix(input_matrix, tile_dict):
    return onehot_from_cm_tiletypecountspecified(input_matrix, tile_dict, tile_dict['CountOfNumericTileTypes'])

def get_compiled_char_representations_from_level_dict(level_dict, window_height, window_width):
    colname_list = generate_2dmatrix_col_names(window_height, window_width)
    alllevels_df_list = []
    for level in level_dict:
        char_rep = level_dict[level].char_rep
        flat_rep = np.ndarray.flatten(char_rep)
        level_df = pd.DataFrame(flat_rep.reshape(-1, len(flat_rep)), columns=colname_list, index=[level])
        #level_df.insert(0,"level_name",[level])
        level_df.insert(0,"generator_name",level_dict[level].generator_name)
        alllevels_df_list.append(level_df)
    return pd.concat(alllevels_df_list, ignore_index=False)  

def get_compiled_onehot_from_leveldict(level_dict, tile_dict, height, width):
    colname_list = generate_onehot_col_names(height, width, tile_dict["CountOfNumericTileTypes"])
    alllevels_df_list = []
    for key in level_dict:
        onehot_rep = onehot_from_charmatrix(level_dict[key].char_rep, tile_dict)
        #Update levelwrapper with the onehot rep
        level_dict[key].onehot_rep = onehot_rep

        flat_rep = np.ndarray.flatten(onehot_rep)
        level_df = pd.DataFrame(flat_rep.reshape(-1, len(flat_rep)), columns=colname_list, index=[key])
        #level_df.insert(0,"level_name",[key])
        #level_df.set_index(key)
        level_df.insert(0,"generator_name",level_dict[key].generator_name)
        alllevels_df_list.append(level_df)
    return pd.concat(alllevels_df_list, ignore_index=False)

##################################################
#HELPER METHODS

#Generates basic list of column names for 1D one hot grids
#Each Col name of format: X coord, Y coord and TileType Number
def generate_onehot_col_names(height, width, num_tiletypes):
    output = list()
    for y in range(height):
        for x in range(width):
            for t in range(num_tiletypes):
                output.append(str(y)+","+str(x)+","+str(t))

    return output

#Generates a basic list of column names for flattened 2D grids
#Each Col name of format X coord, Y coord
def generate_2dmatrix_col_names(height, width):
    output = list()
    for y in range(height):
        for x in range(width):
            output.append(str(y)+","+str(x))
    return output   

#Returns list of filenames from a folder
def get_filenames_from_folder(path):
    return glob.glob(path + "*.txt")

#Return a dictionary of coordinates from lists of coordinates 
def return_coord_dict_fromcoord_lists(names, xcoords, ycoords):
    output_dict = dict()
    for v in range(0, len(names)):
        output_dict[v] = [names[v],xcoords[v], ycoords[v]]
    return output_dict


#Get most extreme x and y values from a dictionary
def get_extreme_coords(level_coord_dict, extremecount = 5):
    output_dict = dict()
    #Get largest x
    for x in range (0, extremecount):
        currmaxkey = max(level_coord_dict, key = lambda k: level_coord_dict[k][1])
        output_dict[currmaxkey] = level_coord_dict[currmaxkey]
        level_coord_dict.pop(currmaxkey)
    #Get largest y
    for x in range (0, extremecount):
        currmaxkey = max(level_coord_dict, key = lambda k: level_coord_dict[k][2])
        output_dict[currmaxkey] = level_coord_dict[currmaxkey]
        level_coord_dict.pop(currmaxkey)
    #Get smallest x
    for x in range (0, extremecount):
        currminkey = min(level_coord_dict, key = lambda k: level_coord_dict[k][1])
        output_dict[currminkey] = level_coord_dict[currminkey]
        level_coord_dict.pop(currminkey)
    #Get smallest y
    for x in range (0, extremecount):
        currminkey = min(level_coord_dict, key = lambda k: level_coord_dict[k][2])
        output_dict[currminkey] = level_coord_dict[currminkey]
        level_coord_dict.pop(currminkey)
    
    return output_dict

def calculateDistance(x1,y1,x2,y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

#Generate labels of type "String" + n 
def gen_component_labels_for_n(label, n):
    output_labels = list()
    for x in range (1, n+1):
        output_labels.append(label + str(x))
    return output_labels

#Get folder dict for game type
def get_file_dict_for_gametype(game):
    if game == Game.Boxoban:
        return get_leveldicts_from_boxoban_files(boxoban_files_dict)
    else: 
        return get_leveldicts_from_folder_set(game)

def get_randnum_levelwrappers_for_game(game, count):
    if game == Game.Boxoban:
        return get_randnum_levelwrappers_boxoban(boxoban_folders_dict, count)
    else: 
        return get_randnum_levelwrappers(game, count)    

def get_folder_and_tiletypedict_for_game(game):
    output = dict()
    if (game == Game.Mario):
        output['Folder_Dict'] = mario_folders_dict
        output['Tile_Type_Dict'] = mario_tiletypes_dict_condensed
    elif (game == Game.Boxoban):
        output['Folder_Dict']= boxoban_files_dict
        output['Tile_Type_Dict'] = boxoban_tiletypes_dict
    elif (game == Game.Loderunner):
        output['Folder_Dict'] = loderunnder_folders_dict
        output['Tile_Type_Dict'] = lr_tiletypes_dict
    else:
        print("Game type not recognised")

    return output

def generate_levelwrapper_for_game(game, level_name, folder_key, char_rep):
    if (game == Game.Mario):
        return MarioLevel(level_name, folder_key, char_rep)
    elif (game == Game.Boxoban):
        return BoxobanLevel(level_name, folder_key, char_rep)
    elif (game == Game.Loderunner):
        return LoderunnerLevel(level_name, folder_key, char_rep)

#This is a hack, should be in a clever datastructure
def get_feature_names_for_compression_type(algotype):
    if (algotype == CompressionType.PCA):
        return ['PC1Val', 'PC2Val']
    elif (algotype == CompressionType.MCA):
        return ['MCA1Val', 'MCA2Val']
    elif (algotype == CompressionType.TSNE):
        return ['TSNEVal1', 'TSNEVal2']    
    elif (algotype == CompressionType.KernelPCA):
        return ['KernelPCA1', 'KernelPCA2']
    elif (algotype == CompressionType.SVD):
        return ['SVD1Val', 'SVD2Val']

def get_compvals_for_algolist_for_levelpair(level1, level2, algolist):
    vals = []
    for algo in algolist:
        if (algo == CompressionType.PCA):
            vals+= [level1.PC1Val, level1.PC2Val, level2.PC1Val, level2.PC2Val]
        elif (algo == CompressionType.SVD):
            vals+= [level1.SVD1Val, level1.SVD2Val, level2.SVD1Val, level2.SVD2Val]
        elif (algo == CompressionType.MCA):
            vals+= [level1.MCA1Val, level1.MCA2Val, level2.MCA1Val, level2.MCA2Val]
        elif (algo == CompressionType.TSNE):
            vals+= [level1.TSNEVal1, level1.TSNEVal2, level2.TSNEVal1, level2.TSNEVal2]
        elif (algo == CompressionType.KernelPCA):
            vals+=[level1.KernelPCA1, level1.KernelPCA2, level2.KernelPCA1, level2.KernelPCA2]
        else:
            print("Algo not recognised in get vals method")
    return vals
def get_distances_for_algolist_for_levelpair(level1, level2, algolist):
    distances = []
    for algo in algolist:
        if (algo == CompressionType.PCA):
            distances.append(calculateDistance(level1.PC1Val, level1.PC2Val, level2.PC1Val, level2.PC2Val))
        elif (algo == CompressionType.SVD):
            distances.append(calculateDistance(level1.SVD1Val, level1.SVD2Val, level2.SVD1Val, level2.SVD2Val))
        elif (algo == CompressionType.MCA):
            distances.append(calculateDistance(level1.MCA1Val, level1.MCA2Val, level2.MCA1Val, level2.MCA2Val))
        elif (algo == CompressionType.TSNE):
            distances.append(calculateDistance(level1.TSNEVal1, level1.TSNEVal2, level2.TSNEVal1, level2.TSNEVal2))
        elif (algo == CompressionType.KernelPCA):
            distances.append(calculateDistance(level1.KernelPCA1, level1.KernelPCA2, level2.KernelPCA1, level2.KernelPCA2))
        else:
            print("Algo not recognised in get distances method")
    return distances

def get_bcvals_for_bclist_for_levelpair(level1, level2, bclist):
    vals = []
    for bc in bclist:
        if (bc == BCType.EmptySpace):
            vals+=[level1.empty_space, level2.empty_space]
        elif (bc == BCType.EnemyCount):
            vals+=[level1.enemy_count, level2.enemy_count]
        elif (bc == BCType.Linearity):
            vals+=[level1.linearity, level2.linearity]
    return vals

def get_differences_for_bclist_for_levelpair(level1, level2, bclist):
    differences = []
    for bc in bclist:
        if (bc == BCType.EmptySpace):
            differences.append(abs(level1.empty_space - level2.empty_space))
        elif (bc == BCType.EnemyCount):
            differences.append(abs(level1.enemy_count - level2.enemy_count))
        elif (bc == BCType.Linearity):
            differences.append(abs(level1.linearity - level2.linearity))
    return differences

def gen_distnames_for_algos(algolist):
    returnlist = []
    for algo in algolist:
        returnlist.append(algo.name + "Dist") 
    return returnlist

def gen_valanddist_colnames_for_algos(algolist):
    returnlist = []
    #Value column names
    for algo in algolist:
        returnlist.append("1stLvlVal "+ algo.name + " 1") 
        returnlist.append("1stLvlVal  "+ algo.name + " 2") 
        returnlist.append("2ndLvlVal  "+ algo.name + " 1") 
        returnlist.append("2ndLvlVal  "+ algo.name + " 2") 
    returnlist+=gen_distnames_for_algos(algolist)
    return returnlist
    
def gen_diffnames_for_bcs(bclist):
    returnlist = []
    for bc in bclist:
        returnlist.append(bc.name + "Dist")
    return returnlist

def gen_valanddiff_colnames_for_bcs(bclist):
    returnlist = []
    for bc in bclist:
        returnlist.append("1stLvlVal "+ bc.name) 
        returnlist.append("2ndLvlVal "+ bc.name) 
    returnlist+=gen_diffnames_for_bcs(bclist)
    return returnlist

def get_level_heightandwidth_for_game(game):
    if (game == Game.Mario):
        return [mario_height, mario_width]
    elif (game == Game.Boxoban):
        return [boxoban_height, boxoban_width]
    elif (game == Game.Loderunner):
        return [loderunner_height, loderunner_width]

###################################
#Level Wrapper Update Methods
def update_levelwrapper_datacomp_features(level_dict, compdf, compression_type):
    """
    #print("Updating level wrappers with data from " + compression_type.name)
    #print(compdf.head())
    feature1, feature2 = get_feature_names_for_compression_type(compression_type)
    #print("Feature names : " + feature1 +"," + feature2)
    counter = 0
    for level in level_dict:
        level_dict[level].feature1 = compdf.loc[level][compression_type.name+' 1']
        level_dict[level].feature2 = compdf.loc[level][compression_type.name+' 2']
        #if (counter <5):
        #    print("Updating Level " +level_dict[level].name + " algo " + compression_type.name + " with values: " + str(level_dict[level].feature1) + "," + str(level_dict[level].feature2) )
        #    counter+=1
    return level_dict
    """
    if (compression_type == CompressionType.PCA):
        for level in level_dict:
            #print('PCA1 Val for level = ' + level + ". " + str(compdf.loc[level]['PC 1']))
            level_dict[level].PC1Val = compdf.loc[level][compression_type.name+' 1']
            level_dict[level].PC2Val = compdf.loc[level][compression_type.name+' 2']
    elif (compression_type == CompressionType.MCA):
        for level in level_dict:
            #print('MCAVal for level = ' + level + ":")
            #print(str(compdf.loc[level]['MCA-PC1']))
            level_dict[level].MCA1Val = compdf.loc[level][compression_type.name+' 1']
            level_dict[level].MCA2Val = compdf.loc[level][compression_type.name+' 2']
    elif (compression_type == CompressionType.SVD):
        for level in level_dict:
            #print('PCA1 Val for level = ' + level + ". " + str(compdf.loc[level]['PC 1']))
            level_dict[level].SVD1Val = compdf.loc[level][compression_type.name+' 1']
            level_dict[level].SVD2Val = compdf.loc[level][compression_type.name+' 2']
    elif (compression_type == CompressionType.PCATSNE):
        for level in level_dict:
            #print('PCA1 Val for level = ' + level + ". " + str(compdf.loc[level]['PC 1']))
            level_dict[level].TSNE_PCA1 = compdf.loc[level][compression_type.name+' 1']
            level_dict[level].TSNE_PCA2 = compdf.loc[level][compression_type.name+' 2']
    elif (compression_type == CompressionType.TSNE):
        for level in level_dict:
            #print('PCA1 Val for level = ' + level + ". " + str(compdf.loc[level]['PC 1']))
            level_dict[level].TSNEVal1 = compdf.loc[level][compression_type.name+' 1']
            level_dict[level].TSNEVal2 = compdf.loc[level][compression_type.name+' 2']
    elif (compression_type == CompressionType.KernelPCA):
        for level in level_dict:
            #print('PCA1 Val for level = ' + level + ". " + str(compdf.loc[level]['PC 1']))
            level_dict[level].KernelPCA1 = compdf.loc[level][compression_type.name+' 1']
            level_dict[level].KernelPCA2 = compdf.loc[level][compression_type.name+' 2']        
    else:
        print('Algo type not recognised')
    return level_dict

#Returns a dictionary of level wrappers with their dimensionality reduction algorithm locations specified
"""
def get_and_update_levels_for_algo_list(game, component_count, algolist, visualise = False):
    level_wrapper_dict = get_file_dict_for_gametype(game)
    #pca_output = multigenerator_pca_analysis(game, height, width, component_count, visualise=True)

    for algo in algolist:
        algo_output = multigenerator_compression(level_wrapper_dict, game, algo, component_count, visualise)
        level_wrapper_dict = update_levelwrapper_datacomp_features(level_wrapper_dict, algo_output, algo)

    return copy.deepcopy(level_wrapper_dict)
"""
def get_and_update_X_levels_for_algo_list(game, component_count, algolist, count, visualise = False):
    print("Starting level wrapper generation")
    level_wrapper_dict = get_randnum_levelwrappers_for_game(game, count)
    #pca_output = multigenerator_pca_analysis(game, height, width, component_count, visualise=True)

    print("Starting compression algorithm process")
    for algo in algolist:
        print("Running algo : " + algo.name)
        algo_output = multigenerator_compression(level_wrapper_dict, game, algo, component_count, visualise)
        level_wrapper_dict = update_levelwrapper_datacomp_features(level_wrapper_dict, algo_output, algo)

    return copy.deepcopy(level_wrapper_dict)
#########################
#Graphing and Visualisation Methods

def simple_scatter(frame, col1, col2, title):
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 

    ax.set_xlabel(col1, fontsize = 15)
    ax.set_ylabel(col2, fontsize = 15)        
    ax.set_title(title , fontsize = 20)

    ax.scatter(frame.loc[:, col1]
                , frame.loc[:, col2]
                #, c = color
                , s = 5)       
    ax.grid()
    plt.show()

def plot_compressed_data(toplot, var_exp, compTyp, gen_names=[],):
    print("Variance explained of plotted" + compTyp.name)
    print(var_exp)

    col1name = compTyp.name + ' 1'
    col2name = compTyp.name + ' 2'

    #print(toplot.tail())


    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    if len(var_exp)>0:
        ax.set_xlabel(compTyp.name + ' 1: ' + str("{0:.3%}".format(var_exp[0])), fontsize = 15)
        ax.set_ylabel(compTyp.name +' 2: ' + str("{0:.3%}".format(var_exp[1])), fontsize = 15)
    else:
        ax.set_xlabel(compTyp.name + ' 1', fontsize = 15)
        ax.set_ylabel(compTyp.name +' 2', fontsize = 15)        
    ax.set_title('2 component ' + compTyp.name, fontsize = 20)

    #Color each generators points differently if we are running for multiple alternatives
    if len(gen_names)>0:
        plot_col = 0
        for generator in gen_names:
            #Generate a random color for the generator
            rgb = color_dict[plot_col]
            plot_col+=1 
            #Limit our targets to just current generator
            to_keep = toplot['generator_name'] == generator
            ax.scatter(toplot.loc[to_keep, col1name]
                        , toplot.loc[to_keep, col2name]
                        , c = [rgb]
                        , s = 50)
    #For single generator
    else:
        ax.scatter(toplot[0].loc[:, col1name]
                    , toplot[0].loc[:, col2name]
                    #, c = color
                    , s = 50)       

    #Get the most outlying values to label
    #coord_dict = return_coord_dict_fromcoord_lists(toplot['level_name'].tolist(), toplot[col1name].tolist(), toplot[col2name].tolist())
    coord_dict = return_coord_dict_fromcoord_lists(toplot.index, toplot[col1name].tolist(), toplot[col2name].tolist())
    extreme_coords_for_labeling = get_extreme_coords(coord_dict, 10)

    for key in extreme_coords_for_labeling:
        ax.annotate(extreme_coords_for_labeling[key][0], (extreme_coords_for_labeling[key][1],extreme_coords_for_labeling[key][2] ))

    ax.legend(gen_names)
    ax.grid()
    plt.show()

##################################
#Compression Algorithm and Feature Correlation Methods

def get_compression_algo_projection(input, compTyp, columnPrefix = '', component_count = 2):

    projectedValues = None
    varExplained = None

    if compTyp == CompressionType.PCA:
        scaledinput = StandardScaler().fit_transform(input)
        pca = PCA(n_components=component_count)
        projectedValues = pca.fit_transform(scaledinput)
        varExplained = pca.explained_variance_ratio_
    elif compTyp == CompressionType.MCA:
        mca = prince.MCA(n_components=component_count)
        mca.fit(input)
        projectedValues = mca.fit_transform(input).to_numpy()
        varExplained = mca.explained_inertia_
    elif compTyp == CompressionType.SVD:
        #scaledinput = StandardScaler().fit_transform(input)
        svd = TruncatedSVD(n_components=component_count, n_iter=7, random_state=42)
        svd.fit(input)
        projectedValues = svd.fit_transform(input)
        varExplained = svd.explained_variance_ratio_
    elif compTyp == CompressionType.TSNE:
        scaledinput = StandardScaler().fit_transform(input)
        tsne = TSNE(n_components=component_count, n_iter=250, random_state=42)
        tsne.fit(scaledinput)
        projectedValues = tsne.fit_transform(scaledinput) 
        varExplained = []
    elif compTyp == CompressionType.KernelPCA:
        scaledinput = StandardScaler().fit_transform(input)
        tsne = KernelPCA(n_components=component_count, kernel='poly')
        tsne.fit(scaledinput)
        projectedValues = tsne.fit_transform(scaledinput) 
        #Calculate Explained Variance
        explained_variance = np.var(projectedValues, axis=0)
        varExplained = explained_variance / np.sum(explained_variance)
    else:
        print("Compression type not recognised")      
                
    labels = gen_component_labels_for_n(columnPrefix+compTyp.name + " ", component_count)

    outputDF = pd.DataFrame(data = projectedValues
                , columns = labels, index = input.index)

    #principalDf['level_name'] = levelnames       
    return (outputDF,varExplained)

def get_linear_correlations_from_df(df, algolist, bclist, filename):

    dists_list = gen_distnames_for_algos(algolist)
    bc_diff_list = gen_diffnames_for_bcs(bclist)

    curr_time = datetime.now().strftime("%m_%d_%H_%M_%S")

    outputfile = open(filename + "correlations" + curr_time + ".txt", "x")

    for compression_dist in dists_list:
        vals = df[[compression_dist]].values.reshape(-1)
        for bc in bc_diff_list:
            bcvals = df[[bc]].values.reshape(-1)

            #pcacorr, pcapval = pearsonr(vals, bcvals)
            #print('Pearsons correlation on ' + compression_dist + ' for BC: ' + bc + ' : %.3f' % pcacorr + " with P Value: " + str("{:.2f}".format(pcapval)))
            pcacspear, sppcapval = spearmanr(vals, bcvals)
            text = ('Spearmans correlation on ' + compression_dist + ' for BC: ' + bc + ' : %.3f' % pcacspear + " with P Value: " + str("{:.2f}".format(sppcapval)))
            outputfile.write(text + "\n")
            print('Spearmans correlation on ' + compression_dist + ' for BC: ' + bc + ' : %.3f' % pcacspear + " with P Value: " + str("{:.2f}".format(sppcapval)))
    outputfile.close()

################################
#MULTIGENERATOR METHODS

def multigenerator_compression(levelwrapper_dict, game, comp_algo, component_count = 2, visualise = False):
    game_info = get_folder_and_tiletypedict_for_game(game)
    folder_dict = game_info['Folder_Dict']
    tile_dict = game_info['Tile_Type_Dict']
    #level_dict = get_file_dict_for_gametype(game)


    height, width = get_level_heightandwidth_for_game(game)

    processed_levels = None
    if (comp_algo == CompressionType.MCA):
        processed_levels = get_compiled_char_representations_from_level_dict(levelwrapper_dict, height, width)
    else:
        processed_levels = get_compiled_onehot_from_leveldict(levelwrapper_dict, tile_dict, height, width)

    gen_name_list = processed_levels['generator_name'].tolist()
    #tsne_info = get_tsne_projection_from_onehot(all_levels_onehot.drop('generator_name', axis=1))
    compressed_info = get_compression_algo_projection(processed_levels.drop('generator_name', axis=1), comp_algo, component_count=component_count)
    #Readding the name of the generator for each level to the list of all levels and their PCs
    compressed_info[0]['generator_name'] = gen_name_list
    if visualise == True:
        plot_compressed_data(compressed_info[0], compressed_info[1], comp_algo, list(folder_dict.keys()))
    #print("Head of comp info for algo " + comp_algo.name)
    #print(compressed_info[0].head())
    return compressed_info[0]


#def apply_tsne_to_compressed_output(folders_dict, tiletypes_dict, algotype, height, width, initial_components, isboxoban = False):
def apply_tsne_to_compressed_output(game, initial_comp_algo, initial_components, visualise = False):
    initial_compression = pd.DataFrame()

    initial_compression = multigenerator_compression(game,initial_comp_algo, initial_components)
    gen_name_list = initial_compression['generator_name'].tolist()
    #tsneinfo = get_tsne_projection_from_onehot(initial_compression.drop('generator_name', axis=1), prev_algo = algotype)
    tsneinfo = get_compression_algo_projection(initial_compression.drop('generator_name', axis=1), CompressionType.TSNE, columnPrefix = initial_comp_algo.name)
    tsneinfo[0]['generator_name'] = gen_name_list
    folders_dict = get_folder_and_tiletypedict_for_game(game)['Folder_Dict']
    #plot_tsne(tsneinfo, list(file_levels_dict.keys()))
    if visualise == True:
        plot_compressed_data(tsneinfo[0],[], CompressionType.PCATSNE, list(folders_dict.keys()))
    return tsneinfo[0]

###################################
#WRAPPER METHODS

""""
#Generates a compiled onehot dataframe from a boxoban file
def get_all_one_hot_boxoban_from_file(path, tile_dict):
     
    level_dict = get_boxoban_leveldict_from_file(path)
    return get_compiled_onehot_from_leveldict(level_dict, tile_dict, 10, 10)
"""

#Generates a dataframe of level pairs and the feature distances between them
def gen_compression_dist_df_from_leveldict(level_wrapper_dict, algolist, bclist, maxpairs, output_file_name):
    counter = 0
    start_time = datetime.now()
    output_dict = dict()

    for (x,y) in ((x,y) for x in level_wrapper_dict for y in level_wrapper_dict if x!=y):
        level1 = level_wrapper_dict[x]
        level2 = level_wrapper_dict[y]
        algo_vals_list = get_compvals_for_algolist_for_levelpair(level1, level2,algolist)
        algo_dist_list = get_distances_for_algolist_for_levelpair(level1, level2,algolist)
        bc_vals_list = get_bcvals_for_bclist_for_levelpair(level1, level2, bclist)
        bc_dist_list = get_differences_for_bclist_for_levelpair(level1, level2, bclist)
        
        #tsnepca_distance = calculateDistance(level1.TSNE_PCA1, level1.TSNE_PCA2, level2.TSNE_PCA1, level2.TSNE_PCA2)
        #bc_dist_list.append(abs(level1.empty_space - level2.empty_space))
        levelpair_row = [level1.name, level1.generator_name, level2.name , level2.generator_name] + algo_vals_list+ algo_dist_list + bc_vals_list + bc_dist_list
        output_dict[counter] = levelpair_row
        
        #empty_space_dist = abs(level1.empty_space - level2.empty_space)
        #output_dict[counter] = [level1.name, level1.generator_name, level2.name , level2.generator_name, pca_distance, svd_distance, mca_distance, empty_space_dist]
        counter+=1

        if (counter%500000 == 0):
            print("500000 level pairs processed. Counter: " + str(counter))
            print("Runtime: " + str(datetime.now () -start_time) + " seconds")

        if (counter>maxpairs):
            break

    algo_colnames = gen_valanddist_colnames_for_algos(algolist)
    bc_colnames = gen_valanddiff_colnames_for_bcs(bclist)

    outputdf = pd.DataFrame.from_dict(output_dict, orient = 'index', columns = (['Level1', 'Level1 Generator',  'Level2', 'Level2 Generator'] + algo_colnames + bc_colnames))

    print("Total runtime: " + str(datetime.now () -start_time) + " seconds")
    print(str(counter) + " Level Pairs assessed for game: " + game.name)

    #print(outputdf.head())
    curr_time = datetime.now().strftime("%m_%d_%H_%M_%S")
    outputdf.to_csv(output_file_name + curr_time +'.csv', index = False)
    return outputdf

#Generates a feature distance dataframe for all level pairs in a folder
def generate_analytics_for_all_level_pairs(game, levelspersetcount, component_count, output_file_name, algolist, bclist, maxpairs = 100000, visualise = False):
    
    #complete_level_dict = get_and_update_levels_for_algo_list(game, component_count, algolist, visualise)
    print("Starting level dict generation")
    complete_level_dict = get_and_update_X_levels_for_algo_list(game, component_count, algolist, levelspersetcount, visualise)
    print("Starting compression df generation")
    return gen_compression_dist_df_from_leveldict(complete_level_dict, algolist,bclist, maxpairs, output_file_name)

"""
def generate_feature_dataframe(game,  component_count, output_file_name, algolist, visualise = False):
    complete_level_dict = get_and_update_levels_for_algo_list(game, component_count, algolist, visualise)

    counter = 0
    start_time = datetime.now()
    output_dict = dict()

    for x in complete_level_dict:
        level = complete_level_dict[x]

        output_dict[counter] = [level.name, level.generator_name, level.PC1Val, level.PC2Val, level.SVD1Val, level.SVD2Val, level.MCA1Val,level.MCA2Val,level.TSNE_PCA1,level.TSNE_PCA2, level.empty_space]
        counter+=1

    outputdf = pd.DataFrame.from_dict(output_dict, orient = 'index', columns = ['Level_Name', 'Level_Generator',  'PCA1','PCA2','SVD1', 'SVD2', 'MCA1', 'MCA2', 'TSNE1', 'TSNE2', 'EmptSpaceDiff'])

    print("Total runtime: " + str(datetime.now () -start_time) + " seconds")

    #print(outputdf.head())
    curr_time = datetime.now().strftime("%m_%d_%H_%M_%S")
    outputdf.to_csv(output_file_name + curr_time +'.csv', index = False)
    return outputdf
"""


#Testing wrapper function
#Note: Will only work on Boxoban for now, we need custom methods for getting column names for game specific features etc
test_comp = 2
game = Game.Boxoban
algolist = [CompressionType.PCA, CompressionType.MCA, CompressionType.SVD, CompressionType.KernelPCA, CompressionType.TSNE]
#bclist = [BCType.EmptySpace, BCType.EnemyCount]
bclist = [BCType.EmptySpace]
max_pairs = 100000
visualise = False
levelcountperset = 3
filename = game.name + "-boxoconsistancy"

test_output = generate_analytics_for_all_level_pairs(game, levelcountperset, test_comp, filename,algolist, bclist, max_pairs, visualise)

#print("head of test output:")
#print(test_output.head())

get_linear_correlations_from_df(test_output, algolist, bclist, filename)

"""
PCAVals = test_output[['PCADist']].values.reshape(-1)
MCAVals = test_output[['MCADist']].values.reshape(-1)
SVDVals = test_output[['SVDDist']].values.reshape(-1)
EmptVals = test_output[['EmptSpaceDiff']].values.reshape(-1)
#print("PCAVals dtype: " + str(PCAVals.dtype))
#print("Shapes of value arrays: " + str(PCAVals.shape))

#simple_scatter(test_output, 'PCADist', 'EmptSpaceDiff', 'PCA vs Empty Space in game: ' + game.name)
#simple_scatter(test_output, 'MCADist', 'EmptSpaceDiff', 'MCA vs Empty Space in game: ' + game.name)
#simple_scatter(test_output, 'SVDDist', 'EmptSpaceDiff', 'SVD vs Empty Space in game: ' + game.name)
#simple_scatter(test_output, 'TSNEPCADist', 'EmptSpaceDiff', 'TSNE vs Empty Space in game: ' + game.name)


pcacorr, pcapval = pearsonr(PCAVals, EmptVals)
mcacorr, mcapval = pearsonr(MCAVals, EmptVals)
svdcorr, svdpval = pearsonr(SVDVals, EmptVals)

print('Pearsons correlation on PCA: %.3f' % pcacorr + " with P Value: " + str("{:.2f}".format(pcapval)))
print('Pearsons correlation on MCA: %.3f' % mcacorr + " with P Value: " + str("{:.2f}".format(mcapval)))
print('Pearsons correlation on SVD: %.3f' % svdcorr + " with P Value: " + str("{:.2f}".format(svdpval)))


pcacspear, sppcapval = spearmanr(PCAVals, EmptVals)
mcaspear, spmcapval = spearmanr(MCAVals, EmptVals)
svdspear, pcsvdpval = spearmanr(SVDVals, EmptVals)

print('Spearmans correlation on PCA: %.3f' % pcacspear + " with P Value: " + str("{:.2f}".format(sppcapval)))
print('Spearmans correlation on MCA: %.3f' % mcaspear + " with P Value: " + str("{:.2f}".format(spmcapval)))
print('Spearmans correlation on SVD: %.3f' % svdspear + " with P Value: " + str("{:.2f}".format(pcsvdpval)))
"""


#Testing Multiple Regression Analysis
"""
test_width = 80
test_height = 10
test_comp = 50
game = Game.Mario
max_pairs = 10
visualise = False
test_output = generate_feature_dataframe(game, test_height, test_width, test_comp, 'wrapped_mario_output', visualise)

independent = test_output[['TSNE1', 'TSNE2']]
dependent = test_output[['EmptSpaceDiff']]

regr = linear_model.LinearRegression()
regr.fit (independent, dependent)
print("Coefficient of determination TSNE::")
print(str(regr.score(independent, dependent)))


independent = test_output[['PCA1', 'PCA2']]
dependent = test_output[['EmptSpaceDiff']]

regr = linear_model.LinearRegression()
regr.fit (independent, dependent)
print("Coefficient of determination PCA:")
print(str(regr.score(independent, dependent)))


independent = test_output[['MCA1', 'MCA2']]
dependent = test_output[['EmptSpaceDiff']]

regr = linear_model.LinearRegression()
regr.fit (independent, dependent)
print("Coefficient of determination MCA:")
print(str(regr.score(independent, dependent)))


independent = test_output[['SVD1', 'SVD2']]
dependent = test_output[['EmptSpaceDiff']]

regr = linear_model.LinearRegression()
regr.fit (independent, dependent)
print("Coefficient of determination SVD:")
print(str(regr.score(independent, dependent)))
"""
