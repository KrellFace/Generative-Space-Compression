import glob
from math import nan
import os
import matplotlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import prince
from LevelWrapper import LevelWrapper

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
    'Hard' : (boxoban_root + 'hard/')
}

boxoban_files_dict = {
    '000 - Medium' : (boxoban_root + 'medium/train/000.txt'),
    '001 - Medium' : (boxoban_root + 'medium/train/001.txt'),
    '000 - Hard' : (boxoban_root + 'hard/000.txt'),
    '001 - Hard' : (boxoban_root + 'hard/001.txt')
}

color_dict = dict({0:'brown',
                1:'green',
                2: 'orange',
                3: 'red',
                4: 'dodgerblue',
                5: 'darkmagenta'})

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
def get_boxoban_leveldict_from_file(file_name, file_dict_key):
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
                    #level_reps[int(temp_levelname)] = char_matrix
                    level_reps[file_dict_key +':'+ temp_levelname] = LevelWrapper(temp_levelname, file_dict_key, char_matrix)
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
def get_leveldict_from_folder(path, folder_key, window_height, window_width):
    file_names = get_filenames_from_folder(path)
    #folder_name = os.path.basename(os.path.normpath(path))
    level_reps = dict()

    #Loop through all levels and add their onehot reps to list
    for level in file_names:
        level_name = os.path.basename(level)
        char_rep = char_matrix_from_file(level)
        char_rep_window = take_window_from_bottomright(char_rep, window_width, window_height)
        level_reps[folder_key +':'+ level_name] = LevelWrapper(level_name, folder_key, char_rep_window)

    return level_reps

#Get a combined levelwrapper dictionary from a folder dictionary
def get_leveldicts_from_folder_set(folders_dict,height, width):
    folder_level_dict = dict()
    for folder in folders_dict:
        #Get all one for for specific folder
        temp_dict = get_leveldict_from_folder(folders_dict[folder], folder,  height, width)
        folder_level_dict = folder_level_dict|temp_dict
    return folder_level_dict

#Get a dictionary of dictionarys (BoxobanFilename: Level Dict) from a Boxoban file
def get_leveldicts_from_boxoban_files(files_dict,height, width):
    files_level_dict = dict()
    for file in files_dict:
        temp_dict = get_boxoban_leveldict_from_file(files_dict[file], file)
        #files_level_dict[file] = temp_dict
        files_level_dict = files_level_dict|temp_dict
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

#Generate labels of type "String" + n 
def gen_component_labels_for_n(label, n):
    output_labels = list()
    for x in range (1, n+1):
        output_labels.append(label + str(x))
    return output_labels

#Get folder dict for game type
def get_file_dict_for_gametype(folder_dict, height, width, isBoxoban):

    if isBoxoban:
        return get_leveldicts_from_boxoban_files(folder_dict, height, width)
    else: 
        return get_leveldicts_from_folder_set(folder_dict, height, width)
    
###################################
#Level Wrapper Update Methods
def update_levelwrapper_datacomp_features(level_dict, compdf, compression_type):
    #print(compdf.head())
    if (compression_type == 'PCA'):
        for level in level_dict:
            #print('PCA1 Val for level = ' + level + ". " + str(compdf.loc[level]['PC 1']))
            level_dict[level].PC1Val = compdf.loc[level]['PC 1']
            level_dict[level].PC2Val = compdf.loc[level]['PC 2']
    elif (compression_type == 'MCA'):
        for level in level_dict:
            #print('MCAVal for level = ' + level + ":")
            #print(str(compdf.loc[level]['MCA-PC1']))
            level_dict[level].MCA1Val = compdf.loc[level]['MCA-PC1']
            level_dict[level].MCA2Val = compdf.loc[level]['MCA-PC2']
    elif (compression_type == 'SVD'):
        for level in level_dict:
            #print('PCA1 Val for level = ' + level + ". " + str(compdf.loc[level]['PC 1']))
            level_dict[level].SVD1Val = compdf.loc[level]['SVD 1']
            level_dict[level].SVD2Val = compdf.loc[level]['SVD 2']
    else:
        print('Algo type not recognised')
    return level_dict


###################################
#Wrapper Methods

#Generates a compiled onehot dataframe from a boxoban file
def get_all_one_hot_boxoban_from_file(path, tile_dict):
     
    level_dict = get_boxoban_leveldict_from_file(path)
    return get_compiled_onehot_from_leveldict(level_dict, tile_dict, 10, 10)


#########################
#Graphing and Visualisation Methods

def plot_compressed_data(toplot, var_exp, algoname, col1name, col2name, gen_names=[],):
    print("Variance explained of plotted" + algoname)
    print(var_exp)
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    if len(var_exp)>0:
        ax.set_xlabel(algoname + ' 1: ' + str("{0:.3%}".format(var_exp[0])), fontsize = 15)
        ax.set_ylabel(algoname +' 2: ' + str("{0:.3%}".format(var_exp[1])), fontsize = 15)
    else:
        ax.set_xlabel(algoname + ' 1', fontsize = 15)
        ax.set_ylabel(algoname +' 2', fontsize = 15)        
    ax.set_title('2 component ' + algoname, fontsize = 20)

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
#Compression Algorithm Methods

def get_PC_values_from_compiled_onehot(onehot_input, conponent_count = 2):
    #Skip first column as thats the levelname column
    #print(onehot_input.head())
    #features = onehot_input.columns[1:]
    #levelnames = onehot_input['level_name'].tolist()

    #x = onehot_input.loc[:,features].values
    x = StandardScaler().fit_transform(onehot_input)
    pca = PCA(n_components=conponent_count)
    components = pca.fit_transform(x)

    print("Reprojected PCA shape: " + str(components.shape))
    pc_labels = gen_component_labels_for_n("PC ", conponent_count)

    principalDf = pd.DataFrame(data = components
                , columns = pc_labels, index = onehot_input.index)

    #principalDf['level_name'] = levelnames       

    return (principalDf,pca.explained_variance_ratio_)

def get_sparsePC_values_from_compiled_onehot(onehot_input, conponent_count = 2):
    #Skip first column as thats the levelname column
    #features = onehot_input.columns[1:]
    #levelnames = onehot_input['level_name'].tolist()

    #x = onehot_input.loc[:,features].values
    x = StandardScaler().fit_transform(onehot_input)
    sparsepca = SparsePCA(n_components=conponent_count, random_state=0)
    components = sparsepca.fit_transform(x)

    print("Reprojected PCA shape: " + str(components.shape))
    pc_labels = gen_component_labels_for_n("SparsePC ", conponent_count)

    principalDf = pd.DataFrame(data = components
                , columns = pc_labels, index = onehot_input.index)

    #principalDf['level_name'] = levelnames       

    return (principalDf,[])

def get_mca_values_from_compiled_char_reps(charrep_input, conponent_count = 2):
    #levelnames = charrep_input['level_name'].tolist()
    #levelrep_columns = charrep_input.drop(['level_name'], axis = 1)

    mca = prince.MCA(n_components=conponent_count)

    mca.fit(charrep_input)
    levels_mcaprojected = mca.fit_transform(charrep_input)

    print("MCA Explained Interia: " + str(levels_mcaprojected.shape))

    truncmcaDF = pd.DataFrame()

    for x in range(0, conponent_count):
        truncmcaDF[('MCA-PC' + str(x+1))] = levels_mcaprojected[x]

    #truncmcaDF['level_name'] = levelnames
    truncmcaDF.set_index(charrep_input.index)
    return (truncmcaDF, mca.explained_inertia_)


def get_projected_truc_svd_values_from_compiled_onehot(onehot_input, conponent_count = 2):
    #levelnames = onehot_input['level_name'].tolist()
    #levelrep_columns = onehot_input.drop(['level_name'], axis = 1)

    svd = TruncatedSVD(n_components=conponent_count, n_iter=7, random_state=42)

    #We can centre the data first, making it equivalent
    #sc = StandardScaler()
    #levelrep_columns = sc.fit_transform(levelrep_columns)
    #print("One hot input head:")
    #print(onehot_input.head())

    svd.fit(onehot_input)
    #print("SVD un transformed explained variance " + str(svd.explained_variance_))
    levels_svdprojected = svd.fit_transform(onehot_input)

    #print("Reprojected shape: " + str(levels_svdprojected.shape))

    #print("svd projeced head:")
    #print(levels_svdprojected.head())

    truncsvdDF = pd.DataFrame(index = onehot_input.index)

    for x in range(0, conponent_count):
        truncsvdDF[('SVD ' + str(x+1))] = levels_svdprojected[:,x]

    #truncsvdDF['level_name'] = levelnames
    #truncsvdDF.set_index(onehot_input.index)
    print("truncdf  head:")
    print(truncsvdDF.head())
    return (truncsvdDF, svd.explained_variance_ratio_)

def get_tsne_projection_from_onehot(onehot_input, conponent_count = 2):
    #levelnames = onehot_input['level_name'].tolist()
    #levelrep_columns = onehot_input.drop(['level_name'], axis = 1)

    tsne = TSNE(n_components=conponent_count, n_iter=250, random_state=42)

    tsne.fit(onehot_input)
    levels_tsneprojected = tsne.fit_transform(onehot_input)

    print("Reprojected shape: " + str(levels_tsneprojected.shape))

    trunctsneDF = pd.DataFrame()
    for x in range(0, conponent_count):
        trunctsneDF[('TSNE ' + str(x+1))] = levels_tsneprojected[:,x]
    #trunctsneDF['level_name'] = levelnames
    trunctsneDF.set_index(onehot_input.index)
    return (trunctsneDF, tsne)

################################
#MULTIGENERATOR METHODS

#Retrieve multiple folders worth of levels from different generators to simultaneously analyse
#Runs Principal Component Analysis on onehot matrix versions of game levels
def multigenerator_pca_analysis(folder_dict,tile_dict,height, width, component_count = 2, isboxoban = False):
    
    level_dict = get_file_dict_for_gametype(folder_dict, height, width, isboxoban)
    all_levels_onehot = get_compiled_onehot_from_leveldict(level_dict, tile_dict, height, width)

    gen_name_list = all_levels_onehot['generator_name'].tolist()
    pca_analysis = get_PC_values_from_compiled_onehot(all_levels_onehot.drop('generator_name', axis=1), component_count)
    
    #Readding the name of the generator for each level to the list of all levels and their PCs
    pca_analysis[0]['generator_name'] = gen_name_list

    plot_compressed_data(pca_analysis[0],pca_analysis[1], 'PCA', 'PC 1', 'PC 2', list(folder_dict.keys()))
    return pca_analysis[0]

def multigenerator_sparsepca_analysis(folder_dict,tile_dict,height, width, component_count = 2, isboxoban = False):
    
    level_dict = get_file_dict_for_gametype(folder_dict, height, width, isboxoban)
    all_levels_onehot = get_compiled_onehot_from_leveldict(level_dict, tile_dict, height, width)

    gen_name_list = all_levels_onehot['generator_name'].tolist()
    pca_analysis = get_sparsePC_values_from_compiled_onehot(all_levels_onehot.drop('generator_name', axis=1), component_count)
    
    #Readding the name of the generator for each level to the list of all levels and their PCs
    pca_analysis[0]['generator_name'] = gen_name_list
    plot_compressed_data(pca_analysis[0],pca_analysis[1], 'PCA', 'SparsePC 1', 'SparsePC 2', list(folder_dict.keys()))
    return pca_analysis[0]

#Runs singular value decomposition on onehot representations of game levels
def multigenerator_svd(folder_dict,tile_dict,height, width, component_count = 2, isboxoban = False):
    level_dict = get_file_dict_for_gametype(folder_dict, height, width, isboxoban)
    all_levels_onehot = get_compiled_onehot_from_leveldict(level_dict, tile_dict, height, width)
    print("all levels onehot head")
    print(all_levels_onehot.head())

    gen_name_list = all_levels_onehot['generator_name'].tolist()

    svd_info = get_projected_truc_svd_values_from_compiled_onehot(all_levels_onehot.drop('generator_name', axis=1), component_count)
    #Readding the name of the generator for each level to the list of all levels and their PCs
    svd_info[0]['generator_name'] = gen_name_list
    #print('svd0 head:')
    #print(svd_info[0].head())
    plot_compressed_data(svd_info[0], svd_info[1], 'SVD', 'SVD 1', 'SVD 2', list(folder_dict.keys()))
    return svd_info[0]

def multigenerator_tsne(folder_dict,tile_dict,height, width, isboxoban = False):
    level_dict = get_file_dict_for_gametype(folder_dict, height, width, isboxoban)
    #all_levels_onehot = get_onehot_df_from_gen_and_level_list(generator_levels_dict, tile_dict, height, width)
    all_levels_onehot = get_compiled_onehot_from_leveldict(level_dict, tile_dict, height, width)

    gen_name_list = all_levels_onehot['generator_name'].tolist()
    tsne_info = get_tsne_projection_from_onehot(all_levels_onehot.drop('generator_name', axis=1))
    #Readding the name of the generator for each level to the list of all levels and their PCs
    tsne_info[0]['generator_name'] = gen_name_list
    plot_compressed_data(tsne_info[0], tsne_info[1], 'T-SNE', 'TSNE 1', 'TSNE 2', list(folder_dict.keys()))

def multigenerator_mca(folder_dict, height, width, conponent_count = 2, isboxoban = False):

    level_dict = get_file_dict_for_gametype(folder_dict, height, width, isboxoban)
    all_levels_char_df = get_compiled_char_representations_from_level_dict(level_dict, height, width)
    gen_name_list = all_levels_char_df['generator_name'].tolist()

    mca_info = get_mca_values_from_compiled_char_reps(all_levels_char_df.drop('generator_name', axis=1), conponent_count)
    #Readding the name of the generator for each level to the list of all levels and their PCs
    mca_info[0]['generator_name'] = gen_name_list

    plot_compressed_data(mca_info[0], mca_info[1], 'MCA', 'MCA-PC1', 'MCA-PC2', list(folder_dict.keys()))
    return mca_info[0]

def apply_tsne_to_compressed_output(folders_dict, tiletypes_dict, algotype, height, width, initial_conponents, isboxoban = False):
    initial_compression = pd.DataFrame()
    if (algotype == 'PCA'):
        initial_compression = multigenerator_pca_analysis(folders_dict, tiletypes_dict, height, width, initial_conponents, isboxoban)
    elif (algotype == 'sparsePCA'):
        initial_compression = multigenerator_sparsepca_analysis(folders_dict, tiletypes_dict, height, width, initial_conponents, isboxoban)
    elif (algotype == 'SVD'):
        initial_compression = multigenerator_svd(folders_dict, tiletypes_dict, height, width, initial_conponents, isboxoban)
    elif (algotype == 'MCA'):
        initial_compression = multigenerator_mca(folders_dict, height, width, initial_conponents, isboxoban)
    else:
        print("Algorithm not found, please check your call")
        return
    gen_name_list = initial_compression['generator_name'].tolist()
    tsneinfo = get_tsne_projection_from_onehot(initial_compression.drop('generator_name', axis=1))
    tsneinfo[0]['generator_name'] = gen_name_list
    #plot_tsne(tsneinfo, list(file_levels_dict.keys()))
    plot_compressed_data(tsneinfo[0],[], 'T-SNE', 'TSNE 1', 'TSNE 2', list(folders_dict.keys()))

#Testing update level wrapper method
test_width = 80
test_height = 10
test_comp = 5
test_mario_dict = get_file_dict_for_gametype(mario_folders_dict, test_height, test_width, False)
test_pca_output = multigenerator_pca_analysis(mario_folders_dict, mario_tiletypes_dict_condensed, test_height, test_width, test_comp)
updated_levelwrappers = update_levelwrapper_datacomp_features(test_mario_dict, test_pca_output, 'PCA')
test_mca_output = multigenerator_mca(mario_folders_dict, test_height, test_width, test_comp)
updated_levelwrappers = update_levelwrapper_datacomp_features(updated_levelwrappers, test_mca_output, 'MCA')
test_svd_output = multigenerator_svd(mario_folders_dict, mario_tiletypes_dict_condensed, test_height, test_width, test_comp)
updated_levelwrappers = update_levelwrapper_datacomp_features(updated_levelwrappers, test_svd_output, 'SVD')

print('MCA values for level Pattern_Count:lvl-888.txt:')
print(str(updated_levelwrappers['Pattern_Count:lvl-888.txt'].MCA1Val) + " , " + str(updated_levelwrappers['Pattern_Count:lvl-888.txt'].MCA2Val))
print('SVD values for level Pattern_Count:lvl-546.txt:')
print(str(updated_levelwrappers['Pattern_Count:lvl-546.txt'].SVD1Val) + " , " + str(updated_levelwrappers['Pattern_Count:lvl-546.txt'].SVD2Val))

#Testing refactor on Boxoban
#test_width = 10
#test_height = 10
#test_comp = 10
#bbn_test_file_dict = get_leveldicts_from_boxoban_files(boxoban_files_dict, test_height, test_width)
#apply_tsne_to_compressed_output(boxoban_files_dict, boxoban_tiletypes_dict, 'PCA', test_height, test_width, test_comp, isboxoban=True)
#apply_tsne_to_compressed_output(boxoban_files_dict, boxoban_tiletypes_dict, 'SVD', test_height, test_width, test_comp, isboxoban=True)
#apply_tsne_to_compressed_output(boxoban_files_dict, boxoban_tiletypes_dict, 'MCA', test_height, test_width, test_comp, isboxoban=True)


#Testing sparsePCA on LR
#test_width = 22
#test_height = 32
#test_comp = 10
#lr_levels_dict = get_leveldict_from_folder(loderunnder_path, test_width, test_height)
#test_dict = dict()
#test_dict['Default'] = lr_levels_dict
#apply_tsne_to_compressed_output(loderunnder_folders_dict, lr_tiletypes_dict, 'sparsePCA', test_height, test_width, test_comp)

#Testing whether stuff still works after LevelWrapper Refactor
#test_width = 80
#test_height = 10
#test_comp = 5
#mario_test_file_dict = get_leveldicts_from_folder_set(mario_folders_dict, test_height, test_width)
#apply_tsne_to_compressed_output(mario_folders_dict, mario_tiletypes_dict_condensed, 'PCA', test_height, test_width, test_comp)
#apply_tsne_to_compressed_output(mario_folders_dict, mario_tiletypes_dict_condensed, 'SVD', test_height, test_width, test_comp)
#apply_tsne_to_compressed_output(mario_folders_dict, mario_tiletypes_dict_condensed, 'MCA', test_height, test_width, test_comp)
