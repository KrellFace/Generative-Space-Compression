import glob
from math import nan
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import prince

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
    'ORE': (mario_root + 'ore/'),
    'Pattern_Count': (mario_root + 'patternCount/')
}

boxoban_folders_dict = {
    'Medium' : (boxoban_root + 'medium/train/'),
    'Hard' : (boxoban_root + 'hard/')
}

boxoban_files_dict = {
    '000 - Medium' : (boxoban_root + 'medium/train/000.txt'),
    '000 - Hard' : (boxoban_root + 'hard/000.txt')
}

#######################################
#FILE IMPORTING METHODS
#######################################

#Get a 2D character matrix from a level file 
def char_matrix_from_filename(path):
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
                    char_matrix = np.reshape(buffer,(10, 10), order = 'C')
                    level_reps[int(temp_levelname)] = char_matrix
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

def get_leveldict_from_individual_files_in_folder_size_specified(path, window_height, window_width):
    file_names = get_filenames_from_folder(path)
    
    level_reps = dict()

    #Loop through all levels and add their onehot reps to list
    for level in file_names:
        level_name = os.path.basename(level)
        char_rep = char_matrix_from_filename(level)
        char_rep_window = take_window_from_bottomright(char_rep, window_width, window_height)
        level_reps[level_name] = char_rep_window

    return level_reps

def get_nested_generator_level_dictionaries_from_folders(folders_dict,height, width):
    folder_level_dict = dict()
    for folder in folders_dict:
        #Get all one for for specific folder
        temp_dict = get_leveldict_from_individual_files_in_folder_size_specified(folders_dict[folder], height, width)
        folder_level_dict[folder] = temp_dict
    return folder_level_dict

def get_nested_generator_level_dictionaries_from_boxoban_files(files_dict,height, width):
    files_level_dict = dict()
    for file in files_dict:
        #Get all one for for specific folder
        #temp_dict = get_leveldict_from_individual_files_in_folder_size_specified(folders_dict[folder], height, width)
        temp_dict = get_boxoban_leveldict_from_file(files_dict[file])
        files_level_dict[file] = temp_dict
    return files_level_dict

#############################################
#MATRIX PROCESSING METHODS

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
#Level processing methods

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

#For methods like MCA we want compiled dataframes of character representations
def get_compiled_char_representations_from_level_dict(level_dict, window_height, window_width):
    colname_list = generate_2dmatrix_col_names(window_height, window_width)
    alllevels_df_list = []
    for level in level_dict:
        char_rep = level_dict[level]
        flat_rep = np.ndarray.flatten(char_rep)
        level_df = pd.DataFrame(flat_rep.reshape(-1, len(flat_rep)), columns=colname_list)
        level_df.insert(0,"level_name",[level])
        alllevels_df_list.append(level_df)
    return pd.concat(alllevels_df_list, ignore_index=True)    

def get_compiled_onehot_from_leveldict(level_dict, tile_dict, height, width):
    colname_list = generate_onehot_col_names(height, width, tile_dict["CountOfNumericTileTypes"])
    alllevels_df_list = []
    for key in level_dict:
        onehot_rep = onehot_from_charmatrix(level_dict[key], tile_dict)
        flat_rep = np.ndarray.flatten(onehot_rep)
        level_df = pd.DataFrame(flat_rep.reshape(-1, len(flat_rep)), columns=colname_list)
        level_df.insert(0,"level_name",[key])
        alllevels_df_list.append(level_df)
    return pd.concat(alllevels_df_list, ignore_index=True)

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

def return_coord_dict_fromcoord_lists(names, xcoords, ycoords):
    output_dict = dict()
    for v in range(0, len(names)):
        output_dict[v] = [names[v],xcoords[v], ycoords[v]]
    return output_dict


#Get most extreme x and y values 
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


def gen_component_labels_for_n(label, n):
    output_labels = list()
    for x in range (1, n+1):
        output_labels.append(label + str(x))
    return output_labels

###################################
#Wrapper Methods

#Generates a compiled onehot dataframe from a folder containing files of individual level representations.
#Each row in final dataframe is a flattened one hot representation of a level
def get_all_onehot_from_folder_size_specified(path, tile_dict, window_height, window_width):

    level_dict = dict()
    level_dict = get_leveldict_from_individual_files_in_folder_size_specified(path, tile_dict, window_height, window_width)
    return get_compiled_onehot_from_leveldict(level_dict)

def get_all_one_hot_boxoban_from_file(path, tile_dict):
     
    level_dict = get_boxoban_leveldict_from_file(path)
    return get_compiled_onehot_from_leveldict(level_dict, tile_dict, 10, 10)


#########################
#Graphing and Visualisation Methods

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

    #Get the most outlying values to label
    coord_dict = return_coord_dict_fromcoord_lists(pca_info_for_each_level['level_name'].tolist(), pca_info_for_each_level['PC 1'].tolist(), pca_info_for_each_level['PC 2'].tolist())
    extreme_coords_for_labeling = get_extreme_coords(coord_dict, 40)

    for key in extreme_coords_for_labeling:
        ax.annotate(extreme_coords_for_labeling[key][0], (extreme_coords_for_labeling[key][1],extreme_coords_for_labeling[key][2] ))
    #for index, row in pca_info_for_each_level.iterrows():
    #    ax.annotate(row['level_name'],(row['PC 1'],row['PC 2']))

    ax.legend(gen_names)
    ax.grid()
    plt.show()

def plot_mca(mca_info, gen_names=[]):
    intertia_explained = mca_info[1].explained_inertia_
    mca_info_for_each_level = mca_info[0]
    print("Variance explained of plotted MCA")
    print(intertia_explained)
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('MCA PC 1: ' + str("{0:.3%}".format(intertia_explained[0])), fontsize = 15)
    ax.set_ylabel('MCA PC 2: ' + str("{0:.3%}".format(intertia_explained[1])), fontsize = 15)
    ax.set_title('2 component MCA', fontsize = 20)

    #Color each generators points differently if we are running for multiple alternatives
    if len(gen_names)>0:
        for generator in gen_names:
            #Generate a random color for the generator
            rgb = np.random.rand(3,)
            #Limit our targets to just current generator
            to_keep = mca_info_for_each_level['Generator_Name'] == generator
            ax.scatter(mca_info_for_each_level.loc[to_keep, 'MCA-PC1']
                        , mca_info_for_each_level.loc[to_keep, 'MCA-PC2']
                        , c = [rgb]
                        , s = 50)
    #For single generator PCA
    else:
        ax.scatter(mca_info_for_each_level.loc[:, 'MCA-PC1']
                    , mca_info_for_each_level.loc[:, 'MCA-PC2']
                    #, c = color
                    , s = 50)       

    coord_dict = return_coord_dict_fromcoord_lists(mca_info_for_each_level['level_name'].tolist(), mca_info_for_each_level['MCA-PC1'].tolist(), mca_info_for_each_level['MCA-PC2'].tolist())
    extreme_coords_for_labeling = get_extreme_coords(coord_dict)

    for key in extreme_coords_for_labeling:
        ax.annotate(extreme_coords_for_labeling[key][0], (extreme_coords_for_labeling[key][1],extreme_coords_for_labeling[key][2]))

    #for index, row in mca_info_for_each_level.iterrows():
    #    ax.annotate(row['level_name'],(row['MCA-PC1'],row['MCA-PC2']))

    ax.legend(gen_names)
    ax.grid()
    plt.show()

def plot_svd(toplot, gen_names = []):
    svdinfo = toplot[1]
    #svd_var_exp_ratio = svdinfo.explained_variance_ratio_
    svd_var_exp_ratio = svdinfo.explained_variance_ratio_
    svd_projected_on_levels = toplot[0]
    print("Variance explained of plotted SVD")
    print(svd_var_exp_ratio)
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('SVD 1: ' + str("{0:.3%}".format(svd_var_exp_ratio[0])), fontsize = 15)
    ax.set_ylabel('SVD 2: ' + str("{0:.3%}".format(svd_var_exp_ratio[1])), fontsize = 15)
    ax.set_title('2 Component Truncated SVD', fontsize = 20)

    #Color each generators points differently if we are running for multiple alternatives
    if len(gen_names)>0:
        for generator in gen_names:
            #Generate a random color for the generator
            rgb = np.random.rand(3,)
            #Limit our targets to just current generator
            to_keep = svd_projected_on_levels['Generator_Name'] == generator
            ax.scatter(svd_projected_on_levels.loc[to_keep, 'SVD 1']
                        , svd_projected_on_levels.loc[to_keep, 'SVD 2']
                        , c = [rgb]
                        , s = 50)
    #For single generator SVD
    else:
        ax.scatter(svd_projected_on_levels['SVD 1']
                    , svd_projected_on_levels['SVD 2']
                    #, c = color
                    , s = 50)       

    for index, row in svd_projected_on_levels.iterrows():
        #print(row['c1'], row['c2'])
        ax.annotate(row['level_name'],(row['SVD 1'],row['SVD 2']))

    ax.legend(gen_names)
    ax.grid()
    plt.show()

def plot_tsne(toplot, gen_names = []):
    tsneinfo = toplot[1]
    #svd_var_exp_ratio = svdinfo.explained_variance_ratio_
    #svd_var_exp_ratio = svdinfo.explained_variance_ratio_
    tsne_projected_on_levels = toplot[0]
    #print("Variance explained of plotted SVD")
    #print(svd_var_exp_ratio)
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('TSNE 1') #+ str("{0:.3%}".format(svd_var_exp_ratio[0])), fontsize = 15)
    ax.set_ylabel('TSNE 2') #+ str("{0:.3%}".format(svd_var_exp_ratio[1])), fontsize = 15)
    ax.set_title('2 Component TSNE ', fontsize = 20)

    #Color each generators points differently if we are running for multiple alternatives
    if len(gen_names)>0:
        for generator in gen_names:
            #Generate a random color for the generator
            rgb = np.random.rand(3,)
            #Limit our targets to just current generator
            to_keep = tsne_projected_on_levels['Generator_Name'] == generator
            ax.scatter(tsne_projected_on_levels.loc[to_keep, 'TSNE 1']
                        , tsne_projected_on_levels.loc[to_keep, 'TSNE 2']
                        , c = [rgb]
                        , s = 50)
    #For single generator SVD
    else:
        ax.scatter(tsne_projected_on_levels['TSNE 1']
                    , tsne_projected_on_levels['TSNE 2']
                    #, c = color
                    , s = 50)       

    for index, row in tsne_projected_on_levels.iterrows():
        #print(row['c1'], row['c2'])
        ax.annotate(row['level_name'],(row['TSNE 1'],row['TSNE 2']))

    ax.legend(gen_names)
    ax.grid()
    plt.show()

##################################
#Compression Algorithm Methods

def get_PC_values_from_compiled_onehot(onehot_input, conponent_count = 2):
    #Skip first column as thats the levelname column
    features = onehot_input.columns[1:]
    levelnames = onehot_input['level_name'].tolist()

    x = onehot_input.loc[:,features].values
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=conponent_count)
    components = pca.fit_transform(x)

    print("Reprojected PCA shape: " + str(components.shape))
    pc_labels = gen_component_labels_for_n("PC ", conponent_count)

    principalDf = pd.DataFrame(data = components
                , columns = pc_labels)

    principalDf['level_name'] = levelnames       
    
    pc_var_explained = {
        "PC1": pca.explained_variance_ratio_[0],
        "PC2": pca.explained_variance_ratio_[1]
    }
    return (principalDf,pc_var_explained)

def get_mca_values_from_compiled_char_reps(charrep_input, conponent_count = 2):
    levelnames = charrep_input['level_name'].tolist()
    levelrep_columns = charrep_input.drop(['level_name'], axis = 1)

    mca = prince.MCA(n_components=conponent_count)

    mca.fit(levelrep_columns)
    levels_mcaprojected = mca.fit_transform(levelrep_columns)

    print("Reprojected MCA shape: " + str(levels_mcaprojected.shape))

    truncmcaDF = pd.DataFrame()

    for x in range(0, conponent_count):
        truncmcaDF[('MCA-PC' + str(x))] = levels_mcaprojected[x]
    #truncmcaDF['MCA-PC1'] = levels_mcaprojected[0]
    #truncmcaDF['MCA-PC2'] = levels_mcaprojected[1]
    truncmcaDF['level_name'] = levelnames
    return (truncmcaDF, mca)


def get_projected_truc_svd_values_from_compiled_onehot(onehot_input, conponent_count = 2):
    levelnames = onehot_input['level_name'].tolist()
    levelrep_columns = onehot_input.drop(['level_name'], axis = 1)
    #print("Level rep head: ")
    #print(levelrep_columns.head())

    svd = TruncatedSVD(n_components=conponent_count, n_iter=7, random_state=42)

    #We can centre the data first, making it equivalent
    #sc = StandardScaler()
    #levelrep_columns = sc.fit_transform(levelrep_columns)

    svd.fit(levelrep_columns)
    print("SVD un transformed explained variance " + str(svd.explained_variance_))
    levels_svdprojected = svd.fit_transform(levelrep_columns)

    print("Reprojected shape: " + str(levels_svdprojected.shape))

    truncsvdDF = pd.DataFrame()

    for x in range(0, conponent_count):
        truncsvdDF[('SVD ' + str(x+1))] = levels_svdprojected[:,x]

    #truncsvdDF['SVD 1'] = levels_svdprojected[:,0]
    #truncsvdDF['SVD 2'] = levels_svdprojected[:,1]
    truncsvdDF['level_name'] = levelnames
    return (truncsvdDF, svd)

def get_tsne_projection_from_onehot(onehot_input, conponent_count = 2):
    levelnames = onehot_input['level_name'].tolist()
    levelrep_columns = onehot_input.drop(['level_name'], axis = 1)

    tsne = TSNE(n_components=conponent_count, n_iter=250, random_state=42)

    tsne.fit(levelrep_columns)
    levels_tsneprojected = tsne.fit_transform(levelrep_columns)

    print("Reprojected shape: " + str(levels_tsneprojected.shape))


    trunctsneDF = pd.DataFrame()
    for x in range(0, conponent_count):
        trunctsneDF[('TSNE ' + str(x+1))] = levels_tsneprojected[:,x]
    #trunctsneDF['TSNE 1'] = levels_tsneprojected[:,0]
    #trunctsneDF['TSNE 2'] = levels_tsneprojected[:,1]
    trunctsneDF['level_name'] = levelnames
    return (trunctsneDF, tsne)

################################
#MULTIGENERATOR METHODS

#Retrieve multiple folders worth of levels from different generators to simultaneously analyse
#Runs Principal Component Analysis on onehot matrix versions of game levels
def multigenerator_pca_analysis(generator_levels_dict,tile_dict,height, width, component_count = 2):
    
    #Storage for our compiled dataframe from all folders
    output_dfs = []
    for generator in generator_levels_dict:
        #Get all one for for specific folder
        level_dict = generator_levels_dict[generator]
        compiled_onehot_set = get_compiled_onehot_from_leveldict(level_dict, tile_dict, height, width)
        compiled_onehot_set['Generator_Name'] =  generator
        output_dfs.append(compiled_onehot_set)
    all_levels_onehot =  pd.concat(output_dfs, ignore_index=False)

    gen_name_list = all_levels_onehot['Generator_Name'].tolist()
    pca_analysis = get_PC_values_from_compiled_onehot(all_levels_onehot.drop('Generator_Name', axis=1), component_count)
    
    #Readding the name of the generator for each level to the list of all levels and their PCs
    pca_analysis[0]['Generator_Name'] = gen_name_list
    plot_pca(pca_analysis, list(generator_levels_dict.keys()))
    return pca_analysis[0]


#Runs singular value decomposition on onehot representations of game levels
def multigenerator_svd(generator_levels_dict,tile_dict,height, width, component_count = 2):
    
    #Storage for our compiled dataframe from all folders
    output_dfs = []
    for generator in generator_levels_dict:
        #Get all one for for specific folder
        level_dict = generator_levels_dict[generator]
        compiled_onehot_set = get_compiled_onehot_from_leveldict(level_dict, tile_dict, height, width)
        compiled_onehot_set['Generator_Name'] =  generator
        output_dfs.append(compiled_onehot_set)
    all_levels_onehot =  pd.concat(output_dfs, ignore_index=False)

    gen_name_list = all_levels_onehot['Generator_Name'].tolist()

    svd_info = get_projected_truc_svd_values_from_compiled_onehot(all_levels_onehot.drop('Generator_Name', axis=1), component_count)
    #Readding the name of the generator for each level to the list of all levels and their PCs
    svd_info[0]['Generator_Name'] = gen_name_list
    plot_svd(svd_info, list(generator_levels_dict.keys()))

def multigenerator_tsne(generator_levels_dict,tile_dict,height, width):
    
    #Storage for our compiled dataframe from all folders
    output_dfs = []
    for generator in generator_levels_dict:
        #Get all one for for specific folder
        level_dict = generator_levels_dict[generator]
        compiled_onehot_set = get_compiled_onehot_from_leveldict(level_dict, tile_dict, height, width)
        compiled_onehot_set['Generator_Name'] =  generator
        output_dfs.append(compiled_onehot_set)
    all_levels_onehot =  pd.concat(output_dfs, ignore_index=False)

    gen_name_list = all_levels_onehot['Generator_Name'].tolist()

    tsne_info = get_tsne_projection_from_onehot(all_levels_onehot.drop('Generator_Name', axis=1))
    #Readding the name of the generator for each level to the list of all levels and their PCs
    tsne_info[0]['Generator_Name'] = gen_name_list
    #print('TSNE KL divergence : ' + str(tsne_info[1].kl_divergence_))
    plot_tsne(tsne_info, list(generator_levels_dict.keys()))

def multigenerator_mca(generator_levels_dict, height, width, conponent_count = 2):
    #Storage for our compiled dataframe from all folders
    output_dfs = []
    for generator in generator_levels_dict:
        level_dict = generator_levels_dict[generator]
        current_set = get_compiled_char_representations_from_level_dict(level_dict, height, width)
        current_set['Generator_Name'] =  generator
        output_dfs.append(current_set)
    all_levels_char_df =  pd.concat(output_dfs, ignore_index=False)

    gen_name_list = all_levels_char_df['Generator_Name'].tolist()

    mca_info = get_mca_values_from_compiled_char_reps(all_levels_char_df.drop('Generator_Name', axis=1), conponent_count)
    #Readding the name of the generator for each level to the list of all levels and their PCs
    mca_info[0]['Generator_Name'] = gen_name_list

    plot_mca(mca_info, list(generator_levels_dict.keys()))

#Testing TSNE on output from others
boxoban_file_levels_dict = get_nested_generator_level_dictionaries_from_boxoban_files(boxoban_files_dict, 10, 10)
pca50 = multigenerator_pca_analysis(boxoban_file_levels_dict, boxoban_tiletypes_dict, 10, 10, 10)
gen_name_list = pca50['Generator_Name'].tolist()
tsneinfo = get_tsne_projection_from_onehot(pca50.drop('Generator_Name', axis=1))
tsneinfo[0]['Generator_Name'] = gen_name_list
plot_tsne(tsneinfo, list(boxoban_files_dict.keys()))

#Testing TSNE
#boxoban_file_levels_dict = get_nested_generator_level_dictionaries_from_boxoban_files(boxoban_files_dict, 10, 10)
#multigenerator_tsne(boxoban_file_levels_dict, boxoban_tiletypes_dict, 10, 10)

#Testing All methods on Mario
#generator_levels_dict = get_nested_generator_level_dictionaries_from_folders(mario_folders_dict, 10, 80)
#multigenerator_pca_analysis(generator_levels_dict, mario_tiletypes_dict_condensed, 10, 80)
#multigenerator_svd(generator_levels_dict, mario_tiletypes_dict_condensed, 10, 80)
#multigenerator_mca(generator_levels_dict,10,80)

#Testing ALL methods on Boxoban
boxoban_file_levels_dict = get_nested_generator_level_dictionaries_from_boxoban_files(boxoban_files_dict, 10, 10)
#multigenerator_svd(boxoban_file_levels_dict, boxoban_tiletypes_dict, 10, 10, 10)
#multigenerator_mca(boxoban_file_levels_dict,10,10, 5)
#multigenerator_tsne(boxoban_file_levels_dict, boxoban_tiletypes_dict, 10,10)



#Testing all methods on Loderunner
#lr_levels_dict = get_leveldict_from_individual_files_in_folder_size_specified(loderunnder_path, 22, 32)
#test_dict = dict()
#test_dict['Default'] = lr_levels_dict
#multigenerator_pca_analysis(test_dict, lr_tiletypes_dict,  22, 32)
#multigenerator_svd(test_dict, lr_tiletypes_dict,  22, 32)
#multigenerator_mca(test_dict, 22, 32)
