import glob
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

lr_tiletypes_dict = {
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

mario_tiletypes_dict = {
    '-': int(0),
    'M': int(0),
    'F': int(0),
    'y': int(1),
    'Y': int(1),
    'E': int(1),
    'g': int(1),
    'G': int(1),
    'k': int(1),
    'K': int(1),
    'r': int(1),
    'X': int(2),
    '#': int(2),
    '%': int(2),
    '|': int(0),
    '\*': int(0),
    'B': int(0),
    'b': int(0),
    '?': int(4),
    '@': int(4),
    'Q': int(4),
    '!': int(4),
    '1': int(4),
    '2': int(4),
    'D': int(2),
    'S': int(2),
    'C': int(4),
    'U': int(4),
    'L': int(4),
    'o': int(4),
    't': int(3),
    'T': int(3),
    '<': int(3),
    '>': int(3),
    '[': int(3),
    ']': int(3)
}

mario_root = 'C:/Users/owith/Documents/External Repositories/Mario-AI-Framework/levels/'

#Dictionary of Mario generator names and respective folders 
mario_folders_dict = {
    'Notch_Param': (mario_root + 'notchParam/'),
    'GE': (mario_root + 'ge/'),
    'Original': (mario_root + 'original/'),
    'Hopper': (mario_root + 'hopper/'),
    'ORE': (mario_root + 'ore/'),
    'Pattern_Count': (mario_root + 'patternCount/')
}

loderunnder_path = "C:/Users/owith/Documents/External Repositories/VGLC/TheVGLC/Lode Runner/Processed/"

def char_matrix_from_filename(path):
    with open(path) as f:
        charlist = f.read()

        #First, we calculate the levels dimensions, and create a new list with no new line characters
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
        #print(width)
        #print(charlist_newlinesremoved)
        #print("Width " + str(width))
        #print("Height: " + str(height))
        output_matrix = np.reshape(charlist_newlinesremoved,(height, width), order = 'C')
        #print(output_matrix)
        return output_matrix

def take_window_from_matrix(input_matrix, top_corner_x, top_corner_y, width, height):

    output_window = np.chararray((height, width))
  
  #for (y in 1:window_size_y){
    for y in range(height):

        output_window[y,] = input_matrix[y+top_corner_y,top_corner_x:(top_corner_x+width)]
    
    return output_window

def take_br_window(input_matrix, width, height):
    x_corner = input_matrix.shape[1] - width
    y_corner = input_matrix.shape[0] - height
    return (take_window_from_matrix(input_matrix, x_corner, y_corner, width, height))

def onehot_from_charmatrix_tilecountspecified(input_matrix, tile_dict, num_tile_type):
    #Create our empty 3D matrix to populate
    input_shape = np.shape(input_matrix)
    one_hot = np.zeros((input_shape[0], input_shape[1], num_tile_type))
    #print(np.shape(one_hot))

    #Loop through full matrix to populate it
    for x in range(input_shape[0]):
        for y in range(input_shape[1]):
            #print("Setting index " + str(x) +"," +str(y) +"," + str(lr_tiletypes_dict[input_matrix[x,y]]) + " to 1")
            one_hot[x,y,tile_dict[input_matrix[x,y]]] = int(1)

    return one_hot

def onehot_from_charmatrix(input_matrix, tile_dict):
    return onehot_from_charmatrix_tilecountspecified(input_matrix, tile_dict, len(tile_dict))

def generate_col_names(height, width, num_tiletypes):
    output = list()
    for y in range(height):
        for x in range(width):
            for t in range(num_tiletypes):
                output.append(str(y)+","+str(x)+","+str(t))

    return output

def get_filenames_from_folder(path):
    return glob.glob(path + "*.txt")

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
        #df_alllevels = df_alllevels.append(df_newlevel, ignore_index = False)
        #Hack - Rename all rows with index = '0' (should only be the most recently added) to level_name
        df_alllevels = df_alllevels.rename(index={0:level_name})
        #print("Level " + level_name +  "processed")

    
    return df_alllevels

def get_top_pcs_from_compiled_onehot(onehot_input):
    #print(df_levellist)

    #Testing accuracy of compiled one hot
    #print(df_levellist.iloc[:20, 2000:2020])

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
    return principalDf

def add_gen_name_column(df_input, gen_name):
    output = df_input['Generator_Name'] = gen_name
    return output

def plot_basic_pca(pca_info):
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    ax.scatter(pca_info.loc[:, 'PC 1']
                , pca_info.loc[:, 'PC 2']
                #, c = color
                , s = 50)

    #Loop through all rows and annotate the plot with the level names
    for index in pca_info.index.tolist():
        ax.annotate(index,(pca_info.at[index, 'PC 1'],pca_info.at[index, 'PC 2']))

    ax.grid()
    plt.show()

def mario_pca_analysis(width, height):
    

#TESTING PCA

df_levellist = get_all_onehot_from_folder(loderunnder_path, lr_tiletypes_dict)

lr_pcainfo = get_top_pcs_from_compiled_onehot(df_levellist)

#plot_basic_pca(lr_pcainfo)

"""
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
ax.scatter(lr_pcainfo.loc[:, 'PC 1']
            , lr_pcainfo.loc[:, 'PC 2']
            #, c = color
            , s = 50)

#Loop through all rows and annotate the plot with the level names
for index in lr_pcainfo.index.tolist():
   ax.annotate(index,(lr_pcainfo.at[index, 'PC 1'],lr_pcainfo.at[index, 'PC 2']))

ax.grid()

plt.show()

#for index in lr_principalDf.index.tolist():
#    print (index)
"""

#TESTING WINDOW GRABBING
test_matrix = char_matrix_from_filename(mario_folders_dict['GE'] + "lvl-1.txt")
#print(test_matrix)
#test_window = take_window_from_matrix(test_matrix, 188, 13, 6, 3)
#print(test_window)

test_corner_window = take_br_window(test_matrix, 20, 3)
print(test_corner_window)