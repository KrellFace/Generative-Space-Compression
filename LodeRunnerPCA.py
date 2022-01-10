import glob
from math import nan
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import prince

lr_tiletypes_dict = {
    "CountOfNumericTileTypes" : int(8),
    "B" : int(0),
    "b" : int(1),
    "." : int(2),
    "-" : int(3),
    "#" : int(4),
    "G" : int(5),
    "E" : int(6),
    "M" : int(7)
}

#Modifications:
#Start and end pos are empty 0
#All enemies are the same, 1 
#Apart from bullet bill = empty space
#All solid blocks = 2
#All pipe blocks = 3
#All blocks containing rewards = 4

mario_tiletypes_dict_condensed = {
    "CountOfNumericTileTypes": int(5),
    "-": int(0),
    "M": int(0),
    "F": int(0),
    "y": int(1),
    "Y": int(1),
    "E": int(1),
    "g": int(1),
    "G": int(1),
    "k": int(1),
    "K": int(1),
    "r": int(1),
    "X": int(2),
    "#": int(2),
    "%": int(2),
    "|": int(0),
    "*": int(0),
    "B": int(0),
    "b": int(0),
    "?": int(4),
    "@": int(4),
    "Q": int(4),
    "!": int(4),
    "1": int(4),
    "2": int(4),
    "D": int(2),
    "S": int(2),
    "C": int(4),
    "U": int(4),
    "L": int(4),
    "o": int(4),
    "t": int(3),
    "T": int(3),
    "<": int(3),
    ">": int(3),
    "[": int(3),
    "]": int(3)
}

#Dictionary of tiletypes in boxoban, the google deepmind clone of sokoban

boxoban_tiletypes_dict = {
    "CountOfNumericTileTypes" : int(4),
    "#" : int(0),
    "@" : int(1),
    "$" : int(2),
    "." : int(3),
    " " : int(4)
}

mario_root = 'C:/Users/owith/Documents/External Repositories/Mario-AI-Framework/levels/'

loderunnder_path = "C:/Users/owith/Documents/External Repositories/VGLC/TheVGLC/Lode Runner/Processed/"

boxoban_root = "C:/Users/owith/Documents/External Repositories/boxoban-levels/"

#Dictionary of Mario generator names and respective folders 
mario_folders_dict = {
    'Notch_Param': (mario_root + 'notchParam/'),
    'GE': (mario_root + 'ge/'),
    #'Original': (mario_root + 'original/'),
    'Hopper': (mario_root + 'hopper/'),
    'ORE': (mario_root + 'ore/'),
    'Pattern_Count': (mario_root + 'patternCount/')
}


#Get a 2D character matrix from a level file 
def char_matrix_from_filename(path):
    with open(path) as f:
        charlist = f.read()
        #print("Processing: "  + path)
        #First, we calculate the levels dimensions, and create a new list with no new line characters
        width = 0
        #Flag for counting the characters in the first line as our width calculation
        width_calculated = False
        height = 0
        charlist_newlinesremoved = list()

        for char in charlist:
            #Skip newline characters and increase height
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

def get_boxoban_leveldict_from_file(file_name):

    completed = False
    curr_index = 0
    #Total characters for level name and rep in boxoban files
    tot_size = 115
    level_reps = list()

    with open(file_name) as file:
        charlist = file.read()

        while not completed:
            level_reps.append(charlist[curr_index:(curr_index+tot_size)])
            curr_index+=tot_size
            if len(charlist)<(curr_index + tot_size):
                completed = True

    return level_reps






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

#Generates a onehot 3D array from a character matrix, using mappings between characters and integers specified in a tile dictionary
def onehot_from_charmatrix_tilecountspecified(input_matrix, tile_dict, num_tile_type):
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
    return onehot_from_charmatrix_tilecountspecified(input_matrix, tile_dict, tile_dict['CountOfNumericTileTypes'])

#Generates basic list of column names for 1D one hot grids
#Each Col name of format: X coord, Y coord and TileType Number
def generate_onehot_col_names(height, width, num_tiletypes):
    output = list()
    #Add a default level name column
    #output.append("Level_Name")
    for y in range(height):
        for x in range(width):
            for t in range(num_tiletypes):
                output.append(str(y)+","+str(x)+","+str(t))

    return output

def generate_2dmatrix_col_names(height, width):
    output = list()
    #Add a default level name column
    #output.append("Level_Name")
    for y in range(height):
        for x in range(width):
            output.append(str(y)+","+str(x))
    return output   

#Returns list of filenames from a folder
def get_filenames_from_folder(path):
    return glob.glob(path + "*.txt")


def get_all_onehot_from_folder_size_specified(path, tile_dict, window_height, window_width):
    file_names = get_filenames_from_folder(path)
    
    #Generate column names
    colname_list = generate_onehot_col_names(window_height, window_width, tile_dict["CountOfNumericTileTypes"])
    #colname_list.insert(0, "level_name")
    #df_alllevels = pd.DataFrame(columns = colname_list)
    #print(df_alllevels)
    alllevels_df_list = []

    #Loop through all levels and add their onehot reps to list
    for level in file_names:
        level_name = os.path.basename(level)
        char_rep = char_matrix_from_filename(level)
        char_rep_window = take_window_from_bottomright(char_rep, window_width, window_height)
        onehot_rep = onehot_from_charmatrix(char_rep_window, tile_dict)
        flat_rep = np.ndarray.flatten(onehot_rep)
        level_df = pd.DataFrame(flat_rep.reshape(-1, len(flat_rep)), columns=colname_list)
        level_df.insert(0,"level_name",[level_name])
        alllevels_df_list.append(level_df)
        #print(level_df.columns)

    return pd.concat(alllevels_df_list, ignore_index=True)

def get_all_charrep_from_folder_size_specified(path, window_height, window_width):
    file_names = get_filenames_from_folder(path)
    
    #Generate column names
    colname_list = generate_2dmatrix_col_names(window_height, window_width)
    #colname_list.insert(0, "level_name")
    #df_alllevels = pd.DataFrame(columns = colname_list)
    #print(df_alllevels)
    alllevels_df_list = []

    #Loop through all levels and add their 2D char representations
    for level in file_names:
        level_name = os.path.basename(level)
        char_rep = char_matrix_from_filename(level)
        char_rep_window = take_window_from_bottomright(char_rep, window_width, window_height)
        flat_rep = np.ndarray.flatten(char_rep_window)
        level_df = pd.DataFrame(flat_rep.reshape(-1, len(flat_rep)), columns=colname_list)
        level_df.insert(0,"level_name",[level_name])
        alllevels_df_list.append(level_df)
        #print(level_df.columns)

    return pd.concat(alllevels_df_list, ignore_index=True)

def get_PC_values_from_compiled_onehot(onehot_input):
    #print(df_levellist)

    #Testing PCA
    #From: https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
    #Skip first column as thats the levelname column
    features = onehot_input.columns[1:]
    levelnames = onehot_input['level_name'].tolist()

    #print(rownames)
    x = onehot_input.loc[:,features].values
    x = StandardScaler().fit_transform(x)
    #print(x)

    pca = PCA(n_components=2)
    components = pca.fit_transform(x)
    #print(pca.explained_variance_ratio_)

    principalDf = pd.DataFrame(data = components
                , columns = ['PC 1', 'PC 2'])

    principalDf['level_name'] = levelnames       
    
    pc_var_explained = {
        "PC1": pca.explained_variance_ratio_[0],
        "PC2": pca.explained_variance_ratio_[1]
    }
    return (principalDf,pc_var_explained)


def plot_pca(pca_info, gen_names=[]):
    variance_explained = pca_info[1]
    pca_info_for_each_level = pca_info[0]
    print("Variance explained of plotted PCA")
    print(variance_explained)
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1: ' + str("{0:.3%}".format(variance_explained['PC1'])), fontsize = 15)
    ax.set_ylabel('Principal Component 2: ' + str("{0:.3%}".format(variance_explained['PC2'])), fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)

    #Color each generators points differently if we are running for multiple alternatives
    if len(gen_names)>0:
        for generator in gen_names:
            #Generate a random color for the generator
            rgb = np.random.rand(3,)
            #Limit our targets to just current generator
            to_keep = pca_info_for_each_level['Generator_Name'] == generator
            ax.scatter(pca_info_for_each_level.loc[to_keep, 'PC 1']
                        , pca_info_for_each_level.loc[to_keep, 'PC 2']
                        , c = [rgb]
                        , s = 50)
    #For single generator PCA
    else:
        ax.scatter(pca_info[0].loc[:, 'PC 1']
                    , pca_info[0].loc[:, 'PC 2']
                    #, c = color
                    , s = 50)       

    for index, row in pca_info_for_each_level.iterrows():
        #print(row['c1'], row['c2'])
        ax.annotate(row['level_name'],(row['PC 1'],row['PC 2']))

    ax.legend(gen_names)
    ax.grid()
    plt.show()

#Retrieve multiple folders worth of levels from different generators to simultaneously analyse
#Runs Principal Component Analysis on onehot matrix versions of game levels
def multigenerator_pca_analysis(folders_dict,tile_dict,height, width):
    #Storage for our compiled dataframe from all folders
    output_dfs = []
    generator_names =[]
    for folder in folders_dict:
        #Get all one for for specific folder
        current_set = get_all_onehot_from_folder_size_specified(folders_dict[folder], tile_dict,  height, width)
        #print("Curr set type: " + current_set.dtype)
        current_set['Generator_Name'] =  folder
        #print("Curr set type: " + current_set.dtype)
        generator_names.append(folder)
        output_dfs.append(current_set)
    all_levels_onehot =  pd.concat(output_dfs, ignore_index=False)

    gen_name_list = all_levels_onehot['Generator_Name'].tolist()
    pca_analysis = get_PC_values_from_compiled_onehot(all_levels_onehot.drop('Generator_Name', axis=1))
    
    #Readding the name of the generator for each level to the list of all levels and their PCs
    pca_analysis[0]['Generator_Name'] = gen_name_list

    #plot_pca_with_generator_name([pca_df_with_generator_names,pc_variance_explained], generator_names)
    plot_pca(pca_analysis, generator_names)

#Runs singular value decomposition on onehot representations of game levels
def multigenerator_svd(folders_dict,tile_dict,height, width):
    #Storage for our compiled dataframe from all folders
    output_dfs = []
    generator_names =[]
    for folder in folders_dict:
        #Get all one for for specific folder
        current_set = get_all_onehot_from_folder_size_specified(folders_dict[folder], tile_dict,  height, width)
        #print("Curr set type: " + current_set.dtype)
        current_set['Generator_Name'] =  folder
        #print("Curr set type: " + current_set.dtype)
        generator_names.append(folder)
        output_dfs.append(current_set)
    all_levels_onehot =  pd.concat(output_dfs, ignore_index=False)

    only_cell_columns = all_levels_onehot.drop(['level_name','Generator_Name'], axis = 1)

    svd = TruncatedSVD(n_components=2, n_iter=7, random_state=42)

    svd.fit(only_cell_columns)
    print("Mario SVD Explained Variance Ratio Sum:")
    print(svd.explained_variance_ratio_)


def multigenerator_mca(folders_dict,height, width):
    #Storage for our compiled dataframe from all folders
    output_dfs = []
    generator_names =[]
    for folder in folders_dict:
        #Get all one for for specific folder
        current_set = get_all_charrep_from_folder_size_specified(folders_dict[folder], height, width)
        #print("Curr set type: " + current_set.dtype)
        current_set['Generator_Name'] =  folder
        #print("Curr set type: " + current_set.dtype)
        generator_names.append(folder)
        output_dfs.append(current_set)
    all_levels_char_df =  pd.concat(output_dfs, ignore_index=False)

    gen_name_list = all_levels_char_df['Generator_Name'].tolist()
    only_cell_columns = all_levels_char_df.drop(['level_name','Generator_Name'], axis = 1)

    mca = prince.MCA()

    mca = mca.fit(only_cell_columns)
    print("Mario MCA Explained interia:")
    print(mca.explained_inertia_)



#Testing Boxoban
test_reps = get_boxoban_leveldict_from_file(boxoban_root+ "medium/train/000.txt")
print(test_reps[707])


#Testing MCA on mario
#multigenerator_mca(mario_folders_dict,10,80)

#Testing SVD on mario
#multigenerator_svd(mario_folders_dict,mario_tiletypes_dict_condensed ,10,80)

#TESTING MCA
#test_csv = pd.read_csv(loderunnder_path +  "level 1.txt")
#print(test_csv.head())

#test_all_levels = get_all_charrep_from_folder_size_specified(loderunnder_path, 22, 32)
#print(test_all_levels.head())

#remove_level_col = test_all_levels.drop('level_name', axis = 1)

#mca = prince.MCA()

#mca = mca.fit(remove_level_col)
#transformed = mca.transform(remove_level_col)
#print("MCA Explained interia:")
#print(mca.explained_inertia_)

#Comparing MCA loderunner results to PCA
#df_alt_levellist = get_all_onehot_from_folder_size_specified(loderunnder_path, lr_tiletypes_dict, 22, 32)

#lr_alt_pcainfo = get_PC_values_from_compiled_onehot(df_alt_levellist)
#print("PCA variance explained :")
#print(lr_alt_pcainfo[1])
#plot_pca(lr_alt_pcainfo)

#Comparing MCA loderunner results to PCA

"""
ax = mca.plot_coordinates(
     X=remove_level_col,
     ax=None,
     figsize=(6, 6),
     show_row_points=True,
     row_points_size=10,
     show_row_labels=False,
     show_column_points=True,
     column_points_size=30,
     show_column_labels=False,
     legend_n_cols=1
     )
plt.show()
"""


#TESTING MARIO PCA

#multigenerator_pca_analysis( mario_folders_dict, mario_tiletypes_dict, 10,80)

#TESTING ALT METHODS

#Deprecated method
#df_levellist = get_all_onehot_from_folder(loderunnder_path, lr_tiletypes_dict)
#print(np.size(df_levellist))

#lr_pcainfo = get_PC_values_from_compiled_onehot(df_levellist)
#plot_basic_pca(lr_pcainfo)

#Alt method
#df_alt_levellist = get_all_onehot_from_folder_size_specified(loderunnder_path, lr_tiletypes_dict, 22, 32)
#print(np.size(df_alt_levellist))

#lr_alt_pcainfo = get_PC_values_from_compiled_onehot(df_alt_levellist)
#print(lr_alt_pcainfo[0].columns)
#plot_pca(lr_alt_pcainfo)



#TESTING DYNAMIC WINDOW SIZE SETTING
#df_windowed_levelist = get_all_onehot_from_folder_size_specified(loderunnder_path, lr_tiletypes_dict,22,32)
#print(np.size(df_windowed_levelist))

#lr_window_pcainfo = get_PC_values_from_compiled_onehot(df_windowed_levelist)

#plot_basic_pca(lr_window_pcainfo)

#TESTING WINDOW GRABBING
#test_matrix = char_matrix_from_filename(mario_folders_dict['GE'] + "lvl-1.txt")
#print(test_matrix)
#test_window = take_window_from_matrix(test_matrix, 188, 13, 6, 3)
#print(test_window)

#test_corner_window = take_br_window(test_matrix, 3, 3)
#print(test_corner_window)
#test_onehot_window = onehot_from_charmatrix(test_corner_window, mario_tiletypes_dict)
#test_onehot_window =onehot_from_charmatrix_tilecountspecified(test_corner_window, mario_tiletypes_dict, 5)
#print(test_onehot_window)

#TESTING ABOVE WITH LODERUNNER
#test_matrix = char_matrix_from_filename(loderunnder_path + "Level 1.txt")
#print(test_matrix)
#print(test_matrix.dtype)
#test_window = take_br_window(test_matrix, 6, 3)
#print(test_window)
#print(np.shape(test_window))
#print(test_window.dtype)

#test_corner_window = take_br_window(test_matrix, 3, 3)
#print(test_corner_window)
#test_onehot_window = onehot_from_charmatrix(test_corner_window, lr_tiletypes_dict)
#test_onehot_window =onehot_from_charmatrix_tilecountspecified(test_corner_window, lr_tiletypes_dict)
#print(test_onehot_window)

#Generates a Dataframe of one hot representations of all levels from a folder with a specified tile dictionary
"""
def get_all_onehot_from_folder(path, tile_dict):
    file_names = get_filenames_from_folder(path)
    
    #Create empty dataframe
    #NB Height Width and Tile Count are hardcoded here -not ideal
    colname_list = generate_col_names(22, 32, 8)
    df_alllevels = pd.DataFrame(columns = colname_list)
    #print(df_alllevels)

    #Loop through all levels and add their onehot reps to our dataframe
    for level in file_names:
        level_name = os.path.basename(level)
        char_rep = char_matrix_from_filename(level)
        onehot_rep = onehot_from_charmatrix(char_rep, tile_dict)
        flat_rep = np.ndarray.flatten(onehot_rep)
        df_newlevel = pd.DataFrame(flat_rep.reshape(-1, len(flat_rep)), columns=colname_list)
        df_alllevels = pd.concat([df_alllevels, df_newlevel])
        #Hack - Rename all rows with index = '0' (should only be the most recently added) to level_name
        df_alllevels = df_alllevels.rename(index={0:level_name})
        #print("Level " + level_name +  "processed")

    
    return df_alllevels
"""

#Generates a Dataframe of one hot representations of all levels from a folder with a specified tile dictionary, and specified window size
"""
def get_all_onehot_from_folder_size_specified(path, tile_dict, window_height, window_width):
    file_names = get_filenames_from_folder(path)
    
    #Create empty dataframe
    colname_list = generate_col_names(window_height, window_width, tile_dict["CountOfNumericTileTypes"])
    df_alllevels = pd.DataFrame(columns = colname_list)
    #print(df_alllevels)

    #Loop through all levels and add their onehot reps to our dataframe
    for level in file_names:
        level_name = os.path.basename(level)
        char_rep = char_matrix_from_filename(level)
        char_rep_window = take_window_from_bottomright(char_rep, window_height, tile_dict)
        onehot_rep = onehot_from_charmatrix(char_rep_window, tile_dict)
        flat_rep = np.ndarray.flatten(onehot_rep)
        #df_newlevel = pd.DataFrame(flat_rep.reshape(-1, len(flat_rep)), columns=colname_list)
        df_newlevel = pd.DataFrame(flat_rep.reshape(-1, len(flat_rep)), columns=colname_list)
        df_alllevels = pd.concat([df_alllevels, df_newlevel])
        #df_alllevels = df_alllevels.append(df_newlevel, ignore_index = False)
        #Hack - Rename all rows with index = '0' (should only be the most recently added) to level_name
        df_alllevels = df_alllevels.rename(index={0:level_name})
        #print("Level " + level_name +  "processed")

    
    return df_alllevels
"""

#Returns a Dataframe of levels and their values for the top PCs
#Also returns a dictionionary of  the variance explained by the Top 2 PCs
"""
def get_PC_values_from_compiled_onehot(onehot_input):
    #print(df_levellist)

    #Testing PCA
    #From: https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
    features = onehot_input.columns
    rownames = onehot_input.index.tolist()

    #print(rownames)

    x = onehot_input.loc[:,features].values
    x = StandardScaler().fit_transform(x)
    #print(x)

    pca = PCA(n_components=2)

    components = pca.fit_transform(x)

    print(pca.explained_variance_ratio_)

    principalDf = pd.DataFrame(data = components
                , index = rownames
                , columns = ['PC 1', 'PC 2'])

    pc_var_explained = {
        "PC1": pca.explained_variance_ratio_[0],
        "PC2": pca.explained_variance_ratio_[1]
    }

    return (principalDf,pc_var_explained)

"""

"""
def plot_basic_pca(pca_info):
    variance_explained = pca_info[1]
    print(variance_explained)
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1: ' + str("{0:.3%}".format(variance_explained['PC1'])), fontsize = 15)
    ax.set_ylabel('Principal Component 2: ' + str("{0:.3%}".format(variance_explained['PC2'])), fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    ax.scatter(pca_info[0].loc[:, 'PC 1']
                , pca_info[0].loc[:, 'PC 2']
                #, c = color
                , s = 50)

    #Loop through all rows and annotate the plot with the level names
    for index in pca_info[0].index.tolist():
        ax.annotate(index,(pca_info[0].at[index, 'PC 1'],pca_info[0].at[index, 'PC 2']))

    ax.grid()
    plt.show()
"""



"""
def plot_basic_pca(pca_info):
    variance_explained = pca_info[1]
    pca_info_for_each_level = pca_info[0]
    
    print(variance_explained)
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1: ' + str("{0:.3%}".format(variance_explained['PC1'])), fontsize = 15)
    ax.set_ylabel('Principal Component 2: ' + str("{0:.3%}".format(variance_explained['PC2'])), fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    ax.scatter(pca_info[0].loc[:, 'PC 1']
                , pca_info[0].loc[:, 'PC 2']
                #, c = color
                , s = 50)

    #Loop through all rows and annotate the plot with the level names
    #for datapoint in pca_info[0]:
    #    ax.annotate(datapoint['level_name'],(datapoint['PC 1'],datapoint['PC 2']))

    for index, row in pca_info_for_each_level.iterrows():
        #print(row['c1'], row['c2'])
        ax.annotate(row['level_name'],(row['PC 1'],row['PC 2']))

    ax.grid()
    plt.show()
"""