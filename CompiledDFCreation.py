import numpy as np
import pandas as pd
from sqlalchemy import false
from HelperMethods import *

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

#Generates a 3D one hot array, using a dictionary field specifying the number of tile options 
def onehot_from_charmatrix(input_matrix, tile_dict):
    return onehot_from_cm_tiletypecountspecified(input_matrix, tile_dict, tile_dict['CountOfNumericTileTypes'])

#Generate a stacked df of characterrepresentations of levels where each row represents a full level
#Implemented to facilitate MCA which works on the character representations directly
#Optionally takes a dict that compresses the number of chars used
def get_compiled_char_representations_from_level_dict(level_dict, window_height, window_width, initial_compression = false, comp_dict = None):
    colname_list = generate_2dmatrix_col_names(window_height, window_width)
    alllevels_df_list = []
    for level in level_dict:
        char_rep = level_dict[level].char_rep
        flat_rep = np.ndarray.flatten(char_rep)
        if (initial_compression):
            for i in range(0, len(flat_rep)):
                flat_rep[i] = comp_dict[flat_rep[i]]

        level_df = pd.DataFrame(flat_rep.reshape(-1, len(flat_rep)), columns=colname_list, index=[level])
        #level_df.insert(0,"level_name",[level])
        level_df.insert(0,"generator_name",level_dict[level].generator_name)
        alllevels_df_list.append(level_df)
    return pd.concat(alllevels_df_list, ignore_index=False)  

#Generate a stacked df of one hot representations of levels from a level dict in which each row represetns a level
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