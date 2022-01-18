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

#Get a dictionary (LevelName: LevelRep) from a folder
def get_leveldict_from_folder(path, window_height, window_width):
    file_names = get_filenames_from_folder(path)
    level_reps = dict()

    #Loop through all levels and add their onehot reps to list
    for level in file_names:
        level_name = os.path.basename(level)
        char_rep = char_matrix_from_file(level)
        char_rep_window = take_window_from_bottomright(char_rep, window_width, window_height)
        level_reps[level_name] = char_rep_window

    return level_reps

#Get a dictionary of dictionarys (FolderName: Level Dict) from a folder dictionary
def get_leveldicts_from_folder_set(folders_dict,height, width):
    folder_level_dict = dict()
    for folder in folders_dict:
        #Get all one for for specific folder
        temp_dict = get_leveldict_from_folder(folders_dict[folder], height, width)
        folder_level_dict[folder] = temp_dict
    return folder_level_dict

#Get a dictionary of dictionarys (BoxobanFilename: Level Dict) from a Boxoban file
def get_leveldicts_from_boxoban_files(files_dict,height, width):
    files_level_dict = dict()
    for file in files_dict:
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

#Generates a dataframe of char representations of levels where each row is a level (used for applying MCA)
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

###################################
#Wrapper Methods

#Generates a compiled onehot dataframe from a folder containing files of individual level representations.
#Each row in final dataframe is a flattened one hot representation of a level
def get_all_onehot_from_folder_size_specified(path, tile_dict, window_height, window_width):

    level_dict = dict()
    level_dict = get_leveldict_from_folder(path, tile_dict, window_height, window_width)
    return get_compiled_onehot_from_leveldict(level_dict)

#Generates a compiled onehot dataframe from a boxoban file
def get_all_one_hot_boxoban_from_file(path, tile_dict):
     
    level_dict = get_boxoban_leveldict_from_file(path)
    return get_compiled_onehot_from_leveldict(level_dict, tile_dict, 10, 10)

def get_onehot_df_from_gen_and_level_list(generator_levels_dict, tile_dict, height, width):  
    levelsets=[]
    for generator in generator_levels_dict:
        #Get all one for for specific folder
        level_dict = generator_levels_dict[generator]
        compiled_onehot_set = get_compiled_onehot_from_leveldict(level_dict, tile_dict, height, width)
        compiled_onehot_set['Generator_Name'] =  generator
        levelsets.append(compiled_onehot_set)
    return pd.concat(levelsets, ignore_index=False)

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
        for generator in gen_names:
            #Generate a random color for the generator
            rgb = np.random.rand(3,)
            #Limit our targets to just current generator
            to_keep = toplot['Generator_Name'] == generator
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
    coord_dict = return_coord_dict_fromcoord_lists(toplot['level_name'].tolist(), toplot[col1name].tolist(), toplot[col2name].tolist())
    extreme_coords_for_labeling = get_extreme_coords(coord_dict, 10)

    for key in extreme_coords_for_labeling:
        ax.annotate(extreme_coords_for_labeling[key][0], (extreme_coords_for_labeling[key][1],extreme_coords_for_labeling[key][2] ))
    #for index, row in pca_info_for_each_level.iterrows():
    #    ax.annotate(row['level_name'],(row['PC 1'],row['PC 2']))

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

    return (principalDf,pca.explained_variance_ratio_)

def get_mca_values_from_compiled_char_reps(charrep_input, conponent_count = 2):
    levelnames = charrep_input['level_name'].tolist()
    levelrep_columns = charrep_input.drop(['level_name'], axis = 1)

    mca = prince.MCA(n_components=conponent_count)

    mca.fit(levelrep_columns)
    levels_mcaprojected = mca.fit_transform(levelrep_columns)

    print("Reprojected MCA shape: " + str(levels_mcaprojected.shape))

    truncmcaDF = pd.DataFrame()

    for x in range(0, conponent_count):
        truncmcaDF[('MCA-PC' + str(x+1))] = levels_mcaprojected[x]

    truncmcaDF['level_name'] = levelnames
    return (truncmcaDF, mca.explained_inertia_)


def get_projected_truc_svd_values_from_compiled_onehot(onehot_input, conponent_count = 2):
    levelnames = onehot_input['level_name'].tolist()
    levelrep_columns = onehot_input.drop(['level_name'], axis = 1)

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

    truncsvdDF['level_name'] = levelnames
    return (truncsvdDF, svd.explained_variance_ratio_)

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
    trunctsneDF['level_name'] = levelnames
    return (trunctsneDF, tsne)

################################
#MULTIGENERATOR METHODS

#Retrieve multiple folders worth of levels from different generators to simultaneously analyse
#Runs Principal Component Analysis on onehot matrix versions of game levels
def multigenerator_pca_analysis(generator_levels_dict,tile_dict,height, width, component_count = 2):
    
    all_levels_onehot = get_onehot_df_from_gen_and_level_list(generator_levels_dict, tile_dict, height, width)

    gen_name_list = all_levels_onehot['Generator_Name'].tolist()
    pca_analysis = get_PC_values_from_compiled_onehot(all_levels_onehot.drop('Generator_Name', axis=1), component_count)
    
    #Readding the name of the generator for each level to the list of all levels and their PCs
    pca_analysis[0]['Generator_Name'] = gen_name_list
    plot_compressed_data(pca_analysis[0],pca_analysis[1], 'PCA', 'PC 1', 'PC 2', list(generator_levels_dict.keys()))
    return pca_analysis[0]


#Runs singular value decomposition on onehot representations of game levels
def multigenerator_svd(generator_levels_dict,tile_dict,height, width, component_count = 2):
    all_levels_onehot = get_onehot_df_from_gen_and_level_list(generator_levels_dict, tile_dict, height, width)
    gen_name_list = all_levels_onehot['Generator_Name'].tolist()

    svd_info = get_projected_truc_svd_values_from_compiled_onehot(all_levels_onehot.drop('Generator_Name', axis=1), component_count)
    #Readding the name of the generator for each level to the list of all levels and their PCs
    svd_info[0]['Generator_Name'] = gen_name_list
    plot_compressed_data(svd_info[0], svd_info[1], 'SVD', 'SVD 1', 'SVD 2', list(generator_levels_dict.keys()))
    return svd_info[0]

def multigenerator_tsne(generator_levels_dict,tile_dict,height, width):
    all_levels_onehot = get_onehot_df_from_gen_and_level_list(generator_levels_dict, tile_dict, height, width)
    gen_name_list = all_levels_onehot['Generator_Name'].tolist()
    tsne_info = get_tsne_projection_from_onehot(all_levels_onehot.drop('Generator_Name', axis=1))
    #Readding the name of the generator for each level to the list of all levels and their PCs
    tsne_info[0]['Generator_Name'] = gen_name_list
    plot_compressed_data(tsne_info[0], tsne_info[1], 'T-SNE', 'TSNE 1', 'TSNE 2', list(generator_levels_dict.keys()))

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

    plot_compressed_data(mca_info[0], mca_info[1], 'MCA', 'MCA-PC1', 'MCA-PC2', list(generator_levels_dict.keys()))
    return mca_info[0]

def apply_tsne_to_compressed_output(file_levels_dict, tiletypes_dict, algotype, height, width, initial_conponents):
    initial_compression = pd.DataFrame()
    if (algotype == 'PCA'):
        initial_compression = multigenerator_pca_analysis(file_levels_dict, tiletypes_dict, height, width, initial_conponents)
    elif (algotype == 'SVD'):
        initial_compression = multigenerator_svd(file_levels_dict, tiletypes_dict, height, width, initial_conponents)
    elif (algotype == 'MCA'):
        initial_compression = multigenerator_mca(file_levels_dict, height, width, initial_conponents)
    else:
        print("Algorithm not found, please check your call")
        return
    gen_name_list = initial_compression['Generator_Name'].tolist()
    tsneinfo = get_tsne_projection_from_onehot(initial_compression.drop('Generator_Name', axis=1))
    tsneinfo[0]['Generator_Name'] = gen_name_list
    #plot_tsne(tsneinfo, list(file_levels_dict.keys()))
    plot_compressed_data(tsneinfo[0],[], 'T-SNE', 'TSNE 1', 'TSNE 2', list(file_levels_dict.keys()))

#Testing TSNE wrapper function
#test_width = 80
#test_height = 10
#test_comp = 2
#mario_test_file_dict = get_leveldicts_from_folder_set(mario_folders_dict, test_height, test_width)
#apply_tsne_to_compressed_output(mario_test_file_dict, mario_tiletypes_dict_condensed, 'PCA', test_height, test_width, test_comp)
#apply_tsne_to_compressed_output(mario_test_file_dict, mario_tiletypes_dict_condensed, 'SVD', test_height, test_width, test_comp)
#apply_tsne_to_compressed_output(mario_test_file_dict, mario_tiletypes_dict_condensed, 'MCA', test_height, test_width, test_comp)

#Testing all TSNE wrapper methods on boxoban
#test_width = 22
#test_height = 32
#test_comp = 10
#bbn_test_file_dict = get_leveldict_from_folder(, test_height, test_width)
#apply_tsne_to_compressed_output(bbn_test_file_dict, boxoban_tiletypes_dict, 'PCA', test_height, test_width, test_comp)
#apply_tsne_to_compressed_output(bbn_test_file_dict, boxoban_tiletypes_dict, 'SVD', test_height, test_width, test_comp)
#apply_tsne_to_compressed_output(bbn_test_file_dict, boxoban_tiletypes_dict, 'MCA', test_height, test_width, test_comp)

#Testing TSNE wrapper functionS on LodeRunner
#test_width = 22
#test_height = 32
#test_comp = 10
#lr_levels_dict = get_leveldict_from_folder(loderunnder_path, test_width, test_height)
#test_dict = dict()
#test_dict['Default'] = lr_levels_dict
#apply_tsne_to_compressed_output(test_dict, lr_tiletypes_dict, 'PCA', test_height, test_width, test_comp)
#apply_tsne_to_compressed_output(test_dict, lr_tiletypes_dict, 'SVD', test_height, test_width, test_comp)
#apply_tsne_to_compressed_output(test_dict, lr_tiletypes_dict, 'MCA', test_height, test_width, test_comp)

#Testing TSNE on output from others
#boxoban_file_levels_dict = get_leveldicts_from_boxoban_files(boxoban_files_dict, 10, 10)
#pca50 = multigenerator_pca_analysis(boxoban_file_levels_dict, boxoban_tiletypes_dict, 10, 10, 10)
#gen_name_list = pca50['Generator_Name'].tolist()
#tsneinfo = get_tsne_projection_from_onehot(pca50.drop('Generator_Name', axis=1))
#tsneinfo[0]['Generator_Name'] = gen_name_list
#plot_tsne(tsneinfo, list(boxoban_files_dict.keys()))

#Testing TSNE
#boxoban_file_levels_dict = get_leveldicts_from_boxoban_files(boxoban_files_dict, 10, 10)
#multigenerator_tsne(boxoban_file_levels_dict, boxoban_tiletypes_dict, 10, 10)

#Testing All methods on Mario
#generator_levels_dict = get_leveldicts_from_folder_set(mario_folders_dict, 10, 80)
#multigenerator_pca_analysis(generator_levels_dict, mario_tiletypes_dict_condensed, 10, 80)
#multigenerator_svd(generator_levels_dict, mario_tiletypes_dict_condensed, 10, 80)
#multigenerator_mca(generator_levels_dict,10,80)

#Testing ALL methods on Boxoban
#boxoban_file_levels_dict = get_leveldicts_from_boxoban_files(boxoban_files_dict, 10, 10)
#multigenerator_svd(boxoban_file_levels_dict, boxoban_tiletypes_dict, 10, 10, 10)
#multigenerator_mca(boxoban_file_levels_dict,10,10, 5)
#multigenerator_tsne(boxoban_file_levels_dict, boxoban_tiletypes_dict, 10,10)



#Testing all methods on Loderunner
#lr_levels_dict = get_leveldict_from_folder(loderunnder_path, 22, 32)
#test_dict = dict()
#test_dict['Default'] = lr_levels_dict
#multigenerator_pca_analysis(test_dict, lr_tiletypes_dict,  22, 32)
#multigenerator_svd(test_dict, lr_tiletypes_dict,  22, 32)
#multigenerator_mca(test_dict, 22, 32)
