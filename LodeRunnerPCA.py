import glob
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#Print all file names in a folder
#print(glob.glob("C:/Users/owith/Documents/External Repositories/VGLC/TheVGLC/Lode Runner/Processed/*.txt"))


#Open file and read it as a character series 
#with open("C:/Users/owith/Documents/External Repositories/VGLC/TheVGLC/Lode Runner/Processed/Level 2.txt") as f:
#    content = f.read()
#    print(list(content))

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

def onehot_from_charmatrix(input_matrix):
    #Create our empty 3D matrix to populate
    input_shape = np.shape(input_matrix)
    one_hot = np.zeros((input_shape[0], input_shape[1], len(lr_tiletypes_dict)))
    #print(np.shape(one_hot))

    #Loop through full matrix to populate it
    for x in range(input_shape[0]):
        for y in range(input_shape[1]):
            #print("Setting index " + str(x) +"," +str(y) +"," + str(lr_tiletypes_dict[input_matrix[x,y]]) + " to 1")
            one_hot[x,y,lr_tiletypes_dict[input_matrix[x,y]]] = int(1)

    return one_hot

def generate_col_names(height, width, num_tiletypes):
    output = list()
    for y in range(height):
        for x in range(width):
            for t in range(num_tiletypes):
                output.append(str(y)+","+str(x)+","+str(t))

    return output

def get_filenames_from_folder(path):
    return glob.glob(path + "*.txt")

def get_all_onehot_from_folder(path):
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
        onehot_rep = onehot_from_charmatrix(char_rep)
        flat_rep = np.ndarray.flatten(onehot_rep)
        df_newlevel = pd.DataFrame(flat_rep.reshape(-1, len(flat_rep)), columns=colname_list)
        df_alllevels = pd.concat([df_alllevels, df_newlevel])
        #df_alllevels = df_alllevels.append(df_newlevel, ignore_index = False)
        #Hack - Rename all rows with index = '0' (should only be the most recently added) to level_name
        df_alllevels = df_alllevels.rename(index={0:level_name})
        #print("Level " + level_name +  "processed")

    
    return df_alllevels

def get_top_pcs_from_compiled_onehot(onehot_input):
    df_levellist = get_all_onehot_from_folder(loderunnder_path)
    #print(df_levellist)

    #Testing accuracy of compiled one hot
    #print(df_levellist.iloc[:20, 2000:2020])

    #Testing PCA
    #From: https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
    features = df_levellist.columns
    rownames = df_levellist.index.tolist()

    #print(rownames)

    x = df_levellist.loc[:,features].values
    x = StandardScaler().fit_transform(x)
    #print(x)

    pca = PCA(n_components=2)

    components = pca.fit_transform(x)

    print(pca.explained_variance_ratio_)

    principalDf = pd.DataFrame(data = components
                , index = rownames
                , columns = ['PC 1', 'PC 2'])
    return principalDf



#TESTING Char Matrix
#level2_matrix = char_matrix_from_filename("C:/Users/owith/Documents/External Repositories/VGLC/TheVGLC/Lode Runner/Processed/Level 2.txt")
#print(level2_matrix)

#Test whether we can retrieve ints from our dictionary using char grid
#print(lr_tiletypes_dict[level2_matrix[21,1]])

#TESTING one got functionality
#one_hot = onehot_from_charmatrix(level2_matrix)
#print(np.shape(one_hot))
#print(one_hot[21,0,])
#print(one_hot.ravel())

#TESTING Column name generation and file retrieval
#print(generate_col_names(np.shape(one_hot)[0], np.shape(one_hot)[1], np.shape(one_hot)[2]))
#print(get_filenames_from_folder(loderunnder_path))

df_levellist = get_all_onehot_from_folder(loderunnder_path)
#print(df_levellist)

#Testing accuracy of compiled one hot
#print(df_levellist.iloc[:20, 2000:2020])

#Testing PCA
#From: https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
#features = df_levellist.columns
#rownames = df_levellist.index.tolist()

#print(rownames)

#x = df_levellist.loc[:,features].values
#x = StandardScaler().fit_transform(x)
#print(x)

#pca = PCA(n_components=2)

#lr_components = pca.fit_transform(x)

#print(pca.explained_variance_ratio_)

#lr_principalDf = pd.DataFrame(data = lr_components
#             , index = rownames
#             , columns = ['PC 1', 'PC 2'])
#print(lr_principalDf)

lr_pcainfo = get_top_pcs_from_compiled_onehot(df_levellist)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
#targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
#colors = ['r', 'g', 'b']
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