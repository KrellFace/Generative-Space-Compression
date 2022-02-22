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
from pathlib import Path
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

from sympy import Q, comp 
from LevelWrapper import LevelWrapper
from BoxobanLevel import BoxobanLevel
from MarioLevel import MarioLevel
from LoderunnerLevel import LoderunnerLevel
from LevelImporting import * 

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

"""
boxoban_files_dict = {
    '000 - Medium' : (boxoban_root + 'medium/train/000.txt'),
    '001 - Medium' : (boxoban_root + 'medium/train/001.txt'),
    '000 - Hard' : (boxoban_root + 'hard/000.txt'),
    '001 - Hard' : (boxoban_root + 'hard/001.txt'),
    '000 - Unfiltered': (boxoban_root + 'unfiltered/train/000.txt'),
    '001 - Unfiltered': (boxoban_root + 'unfiltered/train/001.txt')
}
"""

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
#Returns a dictionary of form Key: 'Holding filename : Level Name'  Value: LevelWrapper
def get_boxoban_leveldict_from_file(file_path, folder_name, counttoretrieve):
    level_reps = dict()
    file_name = os.path.basename(file_path)
    line_counter = 0
    buffer = list()
    temp_levelname = ""
    counter = 0
    with open(file_path) as file:
        charlist = file.read()
        for char in charlist:
            #while counter < counttoretrieve:
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
                    new_level = BoxobanLevel(temp_levelname, folder_name, char_matrix)
                    #new_level.calc_behavioral_features()
                    level_reps[folder_name +':'+ file_name+":"+temp_levelname] = new_level
                    temp_levelname = ""
                    buffer.clear()
                    counter+=1
                    #print("Level added. Level counter: " + str(counter))
                    if counter >= counttoretrieve:
                        #print("Target " + str(counter) + " reached. Returning levels")
                        return level_reps
                
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

def get_randnum_levelwrappers(game, maxlvlsevaled):
    level_dict = dict()
    game_info = get_folder_and_tiletypedict_for_game(game)
    folder_dict = game_info['Folder_Dict']
    #Calculate count of levels to get per folder
    count = math.floor((maxlvlsevaled/len(folder_dict)))

    for folder in folder_dict:
        #Get all one for for specific folder
        temp_dict = get_randnum_levelwrappers_folder(folder_dict[folder], folder, game, count)
        level_dict = level_dict|temp_dict
    return level_dict   

def get_randnum_levelwrappers_boxoban(folders_dict, maxlvlsevaled):
    levelwrapper_dict = dict()
    counter = 0
    #Calculate count of levels to get per folder
    folderlevelcount = math.floor((maxlvlsevaled/len(folders_dict)))
    for folder in folders_dict:
        #List all files in folder
        file_list = get_filenames_from_folder(folders_dict[folder])
        #Calculate level count to get per file. A random amount between the min required per file and 5 times this amount
        minrequired = math.ceil((folderlevelcount/len(file_list)))
        while counter < folderlevelcount:
            randfile = random.choice(file_list)
            filelevelcount = None
            if minrequired < (folderlevelcount-counter):
                filelevelcount = random.randint(minrequired, (folderlevelcount-counter))
            #Else, just take the remainder needed from the file
            else:
                filelevelcount = (folderlevelcount-counter)
            #print("Retrieving " + str(filelevelcount) + " levels from file. Target count for folder: " + str(folderlevelcount) + " and curr count: " + str(counter))
            temp_dict = get_boxoban_leveldict_from_file(randfile, folder, filelevelcount)
            file_list.remove(randfile)
            levelwrapper_dict = levelwrapper_dict|temp_dict
            counter+=filelevelcount
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

def get_randnum_levelwrappers_for_game(game, maxlvlsevaled):
    if game == Game.Boxoban:
        return get_randnum_levelwrappers_boxoban(boxoban_folders_dict, maxlvlsevaled)
    else: 
        return get_randnum_levelwrappers(game, maxlvlsevaled)    


def get_folder_and_tiletypedict_for_game(game):
    output = dict()
    if (game == Game.Mario):
        output['Folder_Dict'] = mario_folders_dict
        output['Tile_Type_Dict'] = mario_tiletypes_dict_condensed
    elif (game == Game.Boxoban):
        output['Folder_Dict']= boxoban_folders_dict
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
    elif (algotype == CompressionType.SVD):
        return ['SVD1Val', 'SVD2Val']    
    elif (algotype == CompressionType.KPCA_POLY):
        return ['KPCAPolyVal1', 'KPCAPolyVal2']
    elif (algotype == CompressionType.KPCA_COSINE):
        return ['KPCACosineVal1', 'KPCACosineVal2']
    elif (algotype == CompressionType.KPCA_RBF):
        return ['KPCARbfVal1', 'KPCARbfVal2']
    elif (algotype == CompressionType.KPCA_SIGMOID):
        return ['KPCASigmoidVal1', 'KPCASigmoidVal2']


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
        elif (algo == CompressionType.KPCA_SIGMOID):
            vals+=[level1.KPCASigmoidVal1, level1.KPCASigmoidVal2, level2.KPCASigmoidVal1, level2.KPCASigmoidVal2]
        elif (algo == CompressionType.KPCA_COSINE):
            vals+=[level1.KPCACosineVal1, level1.KPCACosineVal2, level2.KPCACosineVal1, level2.KPCACosineVal2]    
        elif (algo == CompressionType.KPCA_POLY):
            vals+=[level1.KPCAPolyVal1, level1.KPCAPolyVal2, level2.KPCAPolyVal1, level2.KPCAPolyVal2]           
        elif (algo == CompressionType.KPCA_RBF):
            vals+=[level1.KPCARbfVal1, level1.KPCARbfVal2, level2.KPCARbfVal1, level2.KPCARbfVal2]                        
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
        elif (algo == CompressionType.KPCA_SIGMOID):
            distances.append(calculateDistance(level1.KPCASigmoidVal1, level1.KPCASigmoidVal2, level2.KPCASigmoidVal1, level2.KPCASigmoidVal2))
        elif (algo == CompressionType.KPCA_COSINE):
            distances.append(calculateDistance(level1.KPCACosineVal1, level1.KPCACosineVal2, level2.KPCACosineVal1, level2.KPCACosineVal2))  
        elif (algo == CompressionType.KPCA_POLY):
            distances.append(calculateDistance(level1.KPCAPolyVal1, level1.KPCAPolyVal2, level2.KPCAPolyVal1, level2.KPCAPolyVal2))           
        elif (algo == CompressionType.KPCA_RBF):
            distances.append(calculateDistance(level1.KPCARbfVal1, level1.KPCARbfVal2, level2.KPCARbfVal1, level2.KPCARbfVal2))
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
        elif (bc == BCType.Contiguity):
            vals+=[level1.contiguity, level2.contiguity]
        else:
            print("BC Type not recognised")
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
        elif (bc == BCType.Contiguity):
            differences.append(abs(level1.contiguity - level2.contiguity))
        else:
            print("BC type not recognised")
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
    elif (compression_type == CompressionType.TSNE):
        for level in level_dict:
            #print('PCA1 Val for level = ' + level + ". " + str(compdf.loc[level]['PC 1']))
            level_dict[level].TSNEVal1 = compdf.loc[level][compression_type.name+' 1']
            level_dict[level].TSNEVal2 = compdf.loc[level][compression_type.name+' 2']
    elif (compression_type == CompressionType.KPCA_SIGMOID):
        for level in level_dict:
            level_dict[level].KPCASigmoidVal1 = compdf.loc[level][compression_type.name+' 1']
            level_dict[level].KPCASigmoidVal2 = compdf.loc[level][compression_type.name+' 2']
    elif (compression_type == CompressionType.KPCA_COSINE):
        for level in level_dict:
            level_dict[level].KPCACosineVal1 = compdf.loc[level][compression_type.name+' 1']
            level_dict[level].KPCACosineVal2 = compdf.loc[level][compression_type.name+' 2']  
    elif (compression_type == CompressionType.KPCA_POLY):
        for level in level_dict:
            level_dict[level].KPCAPolyVal1 = compdf.loc[level][compression_type.name+' 1']
            level_dict[level].KPCAPolyVal2 = compdf.loc[level][compression_type.name+' 2']            
    elif (compression_type == CompressionType.KPCA_RBF):
        for level in level_dict:
            level_dict[level].KPCARbfVal1 = compdf.loc[level][compression_type.name+' 1']
            level_dict[level].KPCARbfVal2 = compdf.loc[level][compression_type.name+' 2']  
    else:
        print('Algo type not recognised')
    return level_dict

#Returns a dictionary of level wrappers with their dimensionality reduction algorithm locations specified
def get_and_update_X_levels_for_algo_list(game, component_count, algolist, maxlvlsevaled, visualise = False, file_root = ""):
    print("Starting level wrapper generation")
    level_wrapper_dict = get_randnum_levelwrappers_for_game(game, maxlvlsevaled)

    print("Starting compression algorithm process")
    for algo in algolist:
        print("Running algo : " + algo.name)
        start_time = datetime.now()
        algo_output = multigenerator_compression(level_wrapper_dict, game, algo, component_count, visualise, file_root)
        level_wrapper_dict = update_levelwrapper_datacomp_features(level_wrapper_dict, algo_output, algo)
        print("Algo "+ algo.name + "runtime: " + str(datetime.now () -start_time) + " seconds")

    return level_wrapper_dict
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

def plot_compressed_data(toplot, var_exp, compTyp, file_name, gen_names=[]):

    col1name = compTyp.name + ' 1'
    col2name = compTyp.name + ' 2'

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    if len(var_exp)>0:
        #ax.set_xlabel(compTyp.name + ' 1: ' + str("{0:.3%}".format(var_exp[0])), fontsize = 15)
        #ax.set_ylabel(compTyp.name +' 2: ' + str("{0:.3%}".format(var_exp[1])), fontsize = 15)
        ax.set_xlabel(compTyp.name + ' 1: ', fontsize = 15)
        ax.set_ylabel(compTyp.name +' 2', fontsize = 15)
    else:
        ax.set_xlabel(compTyp.name + ' 1', fontsize = 15)
        ax.set_ylabel(compTyp.name +' 2', fontsize = 15) 
    title = os.path.basename(file_name)
    #Set title without .png     
    ax.set_title(title[0:len(title)-4], fontsize = 20)

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
                        , alpha = 0.5
                        , s = 50)
    #For single generator
    else:
        ax.scatter(toplot[0].loc[:, col1name]
                    , toplot[0].loc[:, col2name]
                    , s = 20)       
    
    """
    coord_dict = return_coord_dict_fromcoord_lists(toplot.index, toplot[col1name].tolist(), toplot[col2name].tolist())
    extreme_coords_for_labeling = get_extreme_coords(coord_dict, 10)

    for key in extreme_coords_for_labeling:
        ax.annotate(extreme_coords_for_labeling[key][0], (extreme_coords_for_labeling[key][1],extreme_coords_for_labeling[key][2] ))
    """

    ax.legend(gen_names)
    ax.grid()
    #plt.show()
    plt.savefig(file_name)

##################################
#Compression Algorithm and Feature Correlation Methods

def get_compression_algo_projection(input, compTyp, columnPrefix = '', component_count = 2):

    projectedValues = None
    varExplained = None

    if compTyp == CompressionType.PCA:
        #scaledinput = StandardScaler().fit_transform(input)
        pca = PCA(n_components=component_count)
        projectedValues = pca.fit_transform(input)
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
        #scaledinput = StandardScaler().fit_transform(input)
        tsne = TSNE(n_components=component_count, n_iter=250, random_state=42)
        tsne.fit(input)
        projectedValues = tsne.fit_transform(input) 
        varExplained = []
    elif compTyp == CompressionType.KPCA_POLY:
        #scaledinput = StandardScaler().fit_transform(input)
        kpca = KernelPCA(n_components=component_count, kernel='poly')
        kpca.fit(input)
        projectedValues = kpca.fit_transform(input) 
        #Calculate Explained Variance
        explained_variance = np.var(projectedValues, axis=0)
        varExplained = explained_variance / np.sum(explained_variance)
    elif compTyp == CompressionType.KPCA_COSINE:
        #scaledinput = StandardScaler().fit_transform(input)
        kpca = KernelPCA(n_components=component_count, kernel='cosine')
        kpca.fit(input)
        projectedValues = kpca.fit_transform(input) 
        #Calculate Explained Variance
        explained_variance = np.var(projectedValues, axis=0)
        varExplained = explained_variance / np.sum(explained_variance)
    elif compTyp == CompressionType.KPCA_RBF:
        #scaledinput = StandardScaler().fit_transform(input)
        kpca = KernelPCA(n_components=component_count, kernel='rbf')
        kpca.fit(input)
        projectedValues = kpca.fit_transform(input) 
        #Calculate Explained Variance
        explained_variance = np.var(projectedValues, axis=0)
        varExplained = explained_variance / np.sum(explained_variance)
    elif compTyp == CompressionType.KPCA_SIGMOID:
        #scaledinput = StandardScaler().fit_transform(input)
        kpca = KernelPCA(n_components=component_count, kernel='sigmoid')
        kpca.fit(input)
        projectedValues = kpca.fit_transform(input) 
        #Calculate Explained Variance
        explained_variance = np.var(projectedValues, axis=0)
        varExplained = explained_variance / np.sum(explained_variance)
    else:
        print("Compression type not recognised")      
                
    labels = gen_component_labels_for_n(columnPrefix+compTyp.name + " ", component_count)

    outputDF = pd.DataFrame(data = projectedValues
                , columns = labels, index = input.index)

    return (outputDF,varExplained)

def get_linear_correlations_from_df(df, algolist, bclist, filepath):

    dists_list = gen_distnames_for_algos(algolist)
    bc_diff_list = gen_diffnames_for_bcs(bclist)

    curr_time = datetime.now().strftime("%m_%d_%H_%M_%S")
    outputfile = open(filepath, "x")

    output = dict()

    for compression_dist in dists_list:
        vals = df[[compression_dist]].values.reshape(-1)
        for bc in bc_diff_list:
            bcvals = df[[bc]].values.reshape(-1)

            spcorr, pspval = spearmanr(vals, bcvals)
            text = ('Spearmans correlation on ' + compression_dist + ' for BC: ' + bc + ' : %.3f' % spcorr + " with P Value: " + str("{:.2f}".format(pspval)))
            outputfile.write(text + "\n")
            print(text)
            output[compression_dist + bc] = [compression_dist, bc, spcorr, pspval]
        
    outputfile.close()
    return output

################################
#MULTIGENERATOR METHODS

def multigenerator_compression(levelwrapper_dict, game, comp_algo, component_count = 2, visualise = False, plot_filename_root = ""):
    game_info = get_folder_and_tiletypedict_for_game(game)
    folder_dict = game_info['Folder_Dict']
    tile_dict = game_info['Tile_Type_Dict']
    height, width = get_level_heightandwidth_for_game(game)

    processed_levels = None
    if (comp_algo == CompressionType.MCA):
        processed_levels = get_compiled_char_representations_from_level_dict(levelwrapper_dict, height, width)
    else:
        processed_levels = get_compiled_onehot_from_leveldict(levelwrapper_dict, tile_dict, height, width)

    gen_name_list = processed_levels['generator_name'].tolist()
    compressed_info = get_compression_algo_projection(processed_levels.drop('generator_name', axis=1), comp_algo, component_count=component_count)
    #Readding the name of the generator for each level to the list of all levels and their PCs
    compressed_info[0]['generator_name'] = gen_name_list
    if visualise == True:
        plot_filename = plot_filename_root + " " + comp_algo.name + ".png"
        plot_compressed_data(compressed_info[0], compressed_info[1], comp_algo,plot_filename, list(folder_dict.keys()))
    return compressed_info[0]

###################################
#WRAPPER METHODS
#Generates a dataframe of level pairs and the feature distances between them
def gen_compression_dist_df_from_leveldict(level_wrapper_dict, algolist, bclist, analytics_filepath, exemplars_filepath):
    pair_counter = 0
    start_time = datetime.now()
    output_dict = dict()
    #processed_pairs = list()

    uniquepairs = list(it.combinations(level_wrapper_dict, 2))

    #Initialise storage for the closest and furthest level pairs for each compression
    #Stored as Key: [AlgoName, "Closest" or "Furthest"] Value: [CurrentExamplar Row, Current Examplar Value]
    nearfar_exemplar_dict = dict()
    for algo in algolist:
        nearfar_exemplar_dict[algo.name + " Closest"] = [-1, 10000]
        nearfar_exemplar_dict[algo.name + " Furthest"] = [-1, 0]

    for pair in uniquepairs:
        level1 = level_wrapper_dict[pair[0]]
        level2 = level_wrapper_dict[pair[1]]
        algo_vals_list = get_compvals_for_algolist_for_levelpair(level1, level2,algolist)
        algo_dist_list = get_distances_for_algolist_for_levelpair(level1, level2,algolist)
        bc_vals_list = get_bcvals_for_bclist_for_levelpair(level1, level2, bclist)
        bc_dist_list = get_differences_for_bclist_for_levelpair(level1, level2, bclist)
        
        #tsnepca_distance = calculateDistance(level1.TSNE_PCA1, level1.TSNE_PCA2, level2.TSNE_PCA1, level2.TSNE_PCA2)
        #bc_dist_list.append(abs(level1.empty_space - level2.empty_space))
        levelpair_row = [level1.name, level1.generator_name, level2.name , level2.generator_name] + algo_vals_list+ algo_dist_list + bc_vals_list + bc_dist_list
        output_dict[pair_counter] = levelpair_row

        #Update nearfar dict
        for i in range(0,len(algolist)):
            if (algo_dist_list[i]<nearfar_exemplar_dict[algolist[i].name+" Closest"][1]):
                nearfar_exemplar_dict[algolist[i].name+" Closest"] = [pair_counter, algo_dist_list[i]]
            if (algo_dist_list[i]>nearfar_exemplar_dict[algolist[i].name+" Furthest"][1]):
                nearfar_exemplar_dict[algolist[i].name+" Furthest"] = [pair_counter, algo_dist_list[i]]
        
        pair_counter+=1

        if (pair_counter%200000 == 0):
            print("200000 level pairs processed. Counter: " + str(pair_counter))
            print("Runtime: " + str(datetime.now () -start_time) + " seconds")

    algo_colnames = gen_valanddist_colnames_for_algos(algolist)
    bc_colnames = gen_valanddiff_colnames_for_bcs(bclist)

    outputdf = pd.DataFrame.from_dict(output_dict, orient = 'index', columns = (['Level1', 'Level1 Generator',  'Level2', 'Level2 Generator'] + algo_colnames + bc_colnames))

    print("Nearfar exemplar:")
    print(nearfar_exemplar_dict)
    #Extract closest furthest exemplars
    exemplardict = dict()
    for exemp in nearfar_exemplar_dict:
        counter_val = nearfar_exemplar_dict[exemp][0]
        exemplardict[exemp] = [exemp] + output_dict[counter_val]
    exemplardf = pd.DataFrame.from_dict(exemplardict, orient = 'index', columns = (['ExemplarType', 'Level1', 'Level1 Generator',  'Level2', 'Level2 Generator'] + algo_colnames + bc_colnames))
    
    exemplardf.to_csv(exemplars_filepath, index= False )

    print("Total runtime: " + str(datetime.now () -start_time) + " seconds")

    curr_time = datetime.now().strftime("%m_%d_%H_%M_%S")
    #outputdf.to_csv(analytics_filepath, index= False )
    return outputdf

#Generates a feature distance dataframe for all level pairs in a folder
def generate_analytics_for_all_level_pairs(game, maxlvlsevaled, component_count, analytics_filepath, exemplars_filepath, algolist, bclist, visualise = False, file_root = ""):
    
    #complete_level_dict = get_and_update_levels_for_algo_list(game, component_count, algolist, visualise)
    complete_level_dict = get_and_update_X_levels_for_algo_list(game, component_count, algolist, maxlvlsevaled, visualise, file_root)
    return gen_compression_dist_df_from_leveldict(complete_level_dict, algolist,bclist, analytics_filepath, exemplars_filepath)


def multidomain_multiruns(games, component_count, algolist, tot_lvls_evaled_per_run, runs_per_game, file_prefix, visualise = False):
    bclist = None
    lincorrdict = dict()
    for game in games:
        if (game == Game.Boxoban):
            bclist = [BCType.EmptySpace, BCType.Contiguity]
        else:
            bclist = [BCType.EmptySpace, BCType.EnemyCount, BCType.Linearity]
        runcount = 0
        while runcount < runs_per_game:
            runpath = file_prefix + "/" + "Run " + str(runcount+1) + "/"
            analyticsfilepath = Path(runpath + game.name + "TotalAnalytics.csv")
            analyticsfilepath.parent.mkdir(parents =True, exist_ok=True)
            exemplarsfilepath = Path(runpath + game.name + "Exemplars.csv")
            output = None
            images_root = runpath + game.name
            #Need to hardcode loderunner to be only the 150 we have available
            if (game == Game.Loderunner):
                output = generate_analytics_for_all_level_pairs(game, 150, component_count, analyticsfilepath,exemplarsfilepath, algolist, bclist, visualise,images_root)
            else:
                output = generate_analytics_for_all_level_pairs(game, tot_lvls_evaled_per_run, component_count, analyticsfilepath,exemplarsfilepath, algolist, bclist, visualise,images_root)
            linncorrs = list()
            linncorrs.append(game.name)
            lincorrsfilepath = Path(runpath + game.name + " Run " + str(runcount+1) + ".txt")
            temp_corrsdict = get_linear_correlations_from_df(output, algolist, bclist, lincorrsfilepath)
            #Look through dictionary of linear correlations, add them to outout
            for key in temp_corrsdict:
                linncorrs = list()
                linncorrs+=[game.name, (runcount+1)]
                linncorrs+=temp_corrsdict[key]
                lincorrdict[game.name + " " + str(runcount) + key] = linncorrs
            runcount += 1

    finallinncorrsdf = pd.DataFrame.from_dict(lincorrdict, orient = 'index', columns = ['Game', 'Run', 'Compression_Dist',  'BCDist', 'Spearman Coeff', 'Spearman P Val'] )
    curr_time = datetime.now().strftime("%m_%d_%H_%M_%S")
    finaloutputpath = Path(file_prefix+ "/" + "Total Lin Corrs " + file_prefix +'.csv')
    finallinncorrsdf.to_csv(finaloutputpath, index = False)



#Testing multirun wrapper functions

component_count = 2
games = [Game.Boxoban, Game.Mario, Game.Loderunner]
#algolist = [CompressionType.PCA, CompressionType.MCA, CompressionType.SVD, CompressionType.TSNE]
algolist = [CompressionType.KPCA_SIGMOID, CompressionType.KPCA_POLY, CompressionType.KPCA_RBF, CompressionType.KPCA_COSINE]
tot_lvls_evaled = 500
runs_per_game = 1
visualise = True
fileprefix =  "KPCA Method Testing"
      
multidomain_multiruns(games, component_count, algolist, tot_lvls_evaled, runs_per_game, fileprefix, visualise)

