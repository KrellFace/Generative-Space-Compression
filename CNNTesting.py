from src.func.mainComp import *
from src.config.enumsAndConfig import *
from src.config.helperMthds import *
import numpy as np
from sklearn.model_selection import RepeatedKFold
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import datasets, layers, models, Model
import csv


OUTPUT_PATH  = '../output'

def get_model(in_shape, bc_count):
	#From CNN tutorial
    #Create convolutional base using a common pattern
    #A stack of Conv2d and MaxPooling2D layers
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=in_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding = "same"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding = "same"))

    #Add Dense layers to the top of the model
    #This does classification on the last tensor output
    #Takes 1D vectors as an input, so first we need to flatten
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu', name = "penultimate_layer"))

    #Two Outputs to the final dense layer as we have 2 BCs
    model.add(layers.Dense(bc_count, name = "final_layer"))
    #model.summary()
    model.compile(loss='mae', optimizer='adam')
    return model

def evaluate_model(X, y, input_shape, bc_count):
	results = list()
	# define evaluation procedure
	cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
	# enumerate folds
	for train_ix, test_ix in cv.split(X):
		# prepare data
		X_train, X_test = X[train_ix], X[test_ix]
		y_train, y_test = y[train_ix], y[test_ix]
		# define model
		model = get_model(input_shape, bc_count)
		# fit model
		model.fit(X_train, y_train, verbose=0, epochs=100)
		# evaluate model on test set
		mae = model.evaluate(X_test, y_test, verbose=0)
		# store result
		print('>%.3f' % mae)
		results.append(mae)
	return results

#Get arrays of level wrappers, level one hot matrixes, and level BCs
def get_level_data(game, levelCount):
    leveldict = get_randnum_levelwrappers_for_game(game, levelCount)
    tile_dict = get_folder_and_tiletypedict_for_game(game)['Tile_Type_Dict']

    matrixes_array = []
    bcs_array = []

    count = 1
    for key in leveldict:
        onehot_rep = onehot_from_charmatrix(leveldict[key].char_rep, tile_dict)
        #Update levelwrapper with the onehot rep
        leveldict[key].onehot_rep = onehot_rep
        #flat_rep = np.ndarray.flatten(onehot_rep)
        matrixes_array.append(onehot_rep)
        if(game == Game.Loderunner or game == Game.Mario):
            bcs_array.append([leveldict[key].bc_vals[BCType.EmptySpace], leveldict[key].bc_vals[BCType.Linearity], leveldict[key].bc_vals[BCType.EmptySpace]])
        elif(game == Game.Boxoban):
            bcs_array.append([leveldict[key].bc_vals[BCType.EmptySpace], leveldict[key].bc_vals[BCType.Contiguity]])
        count+=1
    return leveldict, matrixes_array, bcs_array

def save_intermediate_output(intermediate_layer_model, x, lvl_name, filename=OUTPUT_PATH + '/img_dataset.csv'):
    #intermediate_layer_model = Model(inputs=model.input,
    #                                 outputs=model.layers[layer].output)
    intermediate_output = intermediate_layer_model.predict(x[np.newaxis, ...])
    #print(intermediate_output.shape)
    #info_img = np.array([lvl_name])
    row = np.append(lvl_name, intermediate_output)
    with open(filename, 'a+', newline ='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    csvFile.close()
    return intermediate_output

def run_cnn_layer_extraction_and_tsne_bc_correlation(game, level_count, input_shape, bc_list, output_files_name, comp_type):
    level_dict, level_onehot_matrixes, bcs = get_level_data(game, level_count)
    model = get_model(input_shape, len(bc_list))
    split = int((level_count/5)*4)
    train_levels = np.array(level_onehot_matrixes[0:split])
    test_levels = np.array(level_onehot_matrixes[split:])
    train_bcs = np.array(bcs[0:split])
    test_bcs = np.array(bcs[split:])

    if not os.path.exists(output_files_name):
        os.makedirs(output_files_name)

    model.fit(train_levels, train_bcs, verbose=0, epochs=100)

    prediction = model.predict(test_levels)
    total_intermediate_outputs = list()

    intermediate_layer_model = Model(inputs=model.input,
                                    outputs=model.layers[-2].output)

    for level_key in level_dict:
        #Save weights from penultimate layer
        matrix = save_intermediate_output(intermediate_layer_model, level_dict[level_key].onehot_rep, level_key, filename=output_files_name + 'intermediate_output.csv')
        flat_rep = np.ndarray.flatten(matrix)
        matrix_df = pd.DataFrame(flat_rep.reshape(-1, len(flat_rep)), columns=range(0, 64), index=[level_key])
        total_intermediate_outputs.append(matrix_df)


    #print("I M Outputs size: " + str(len(total_intermediate_outputs)))

    compiled = pd.concat(total_intermediate_outputs, ignore_index=False)

    print(compiled.head)

    corr_data = get_compression_algo_projection(compiled, comp_type)
    #print(corr_data[0].head)

    #plot_compressed_data(corr_data, [], comp_type, "Test_Plot", show_plot = True)

    update_levelwrapper_datacomp_features(level_dict, corr_data[0], comp_type)

    final_data = gen_compression_dist_df_from_leveldict(level_dict, game, [comp_type], bc_list, output_files_name )

    linncorrs = list()
    linncorrs.append(game.name)
    lincorrsfilepath = Path(output_files_name + "linn_cors_output.txt")
    temp_corrsdict = get_linear_correlations_from_df(final_data, [comp_type], bc_list, lincorrsfilepath)



#Config
game = Game.Loderunner
level_shape = get_level_heightandwidth_for_game(game)
tile_count = get_folder_and_tiletypedict_for_game(game)['Tile_Type_Dict']["CountOfNumericTileTypes"]
input_shape = (level_shape[0], level_shape[1], tile_count)
level_count = 0
bc_list = []
comp_type = CompressionType.PCA
output_files_name = OUTPUT_PATH +'/CNNTestTest2/'
if(game == Game.Loderunner):
    #input_shape = (22,32,8)
    level_count = 150
    bc_list = [BCType.EmptySpace, BCType.EnemyCount,BCType.Linearity]
elif (game == Game.Boxoban):
    #input_shape = (10,10,5)
    level_count = 600
    bc_list = [BCType.EmptySpace, BCType.Contiguity]
elif (game == Game.Mario):
    #input_shape = (10,10,5)
    level_count = 800
    bc_list = [BCType.EmptySpace, BCType.EnemyCount, BCType.Linearity]

run_cnn_layer_extraction_and_tsne_bc_correlation(game, level_count, input_shape, bc_list, output_files_name, comp_type)



#Testing Prediction
#level_dict, level_onehot_matrixes, bcs = get_level_data(game, level_count)
#model = get_model(input_shape, bc_count)
#split = int((level_count/5)*4)
#train_levels = np.array(level_onehot_matrixes[0:split])
#test_levels = np.array(level_onehot_matrixes[split:])
#train_bcs = np.array(bcs[0:split])
#test_bcs = np.array(bcs[split:])

#if not os.path.exists(output_files_name):
#    os.makedirs(output_files_name)

#model.fit(train_levels, train_bcs, verbose=0, epochs=100)

#prediction = model.predict(test_levels)
#total_intermediate_outputs = list()

#intermediate_layer_model = Model(inputs=model.input,
#                                 outputs=model.layers[-2].output)

#for level_key in level_dict:
    #Save weights from penultimate layer
#    matrix = save_intermediate_output(intermediate_layer_model, level_dict[level_key].onehot_rep, level_key, layer=-2, filename=output_files_name + 'intermediate_output.csv')
#    flat_rep = np.ndarray.flatten(matrix)
#    matrix_df = pd.DataFrame(flat_rep.reshape(-1, len(flat_rep)), columns=range(0, 64), index=[level_key])
#    total_intermediate_outputs.append(matrix_df)


#print("I M Outputs size: " + str(len(total_intermediate_outputs)))

#compiled = pd.concat(total_intermediate_outputs, ignore_index=False)

#print(compiled.head)

#corr_data = get_compression_algo_projection(compiled, CompressionType.TSNE)
#print(corr_data[0].head)

#plot_compressed_data(corr_data, [], CompressionType.TSNE, "Test_Plot", show_plot = True)

#update_levelwrapper_datacomp_features(level_dict, corr_data[0], CompressionType.TSNE)

#final_data = gen_compression_dist_df_from_leveldict(level_dict, game, [CompressionType.TSNE], bc_list, output_files_name )

#linncorrs = list()
#linncorrs.append(game.name)
#lincorrsfilepath = Path(output_files_name + "linn_cors_output.txt")
#temp_corrsdict = get_linear_correlations_from_df(final_data, [CompressionType.TSNE], bc_list, lincorrsfilepath)

#for x in range(len(test_levels)):

    #print("Level " + str(x))
    #print("Actual BCs: " + str(test_bcs[x][0]) + ", " + str(test_bcs[x][1]))
    #print("Predicted BCs: " + str(prediction[x][0]) + ", " + str(prediction[x][1]))

    #Save weights from penultimate layer
    #matrix = save_intermediate_output(model, test_levels[x], "Level " + str(x), layer=-2, filename='img_dataset3.csv')
    #colname_list = generate_onehot_col_names(32, 22, lr_tiletypes_dict["CountOfNumericTileTypes"])
    #flat_rep = np.ndarray.flatten(matrix)
    #matrix_df = pd.DataFrame(flat_rep.reshape(-1, len(flat_rep)), columns=range(0, 64), index=[test_levels[x].name])
    #matrix_df = pd.DataFrame(matrix.flatten())
    #total_intermediate_outputs.append(matrix_df)

    #print("Length of frame index: " + str(len(matrix_df.index)) + " df shape[0]: " + str(matrix_df.shape[0]))
    #print("Column count: " + str(len(matrix_df.columns)))

#Print the weights of the penultimate layer
#weights = model.get_layer("penultimate_layer").weights
#print(weights)
#lastweights = model.get_layer("final_layer").weights
#print(lastweights)

#Run Main
#levels, bcs = get_level_data(Game.Boxoban, level_count)

#nplevels = np.array(levels)
#npbcs = np.array(bcs)

#results = evaluate_model(nplevels, npbcs, input_shape)
#print('MAE: %.3f (%.3f)' % (np.mean(results), np.std(results)))






#train_levels = np.array(levels[0:160])
#test_levels = np.array(levels[160:])
#train_bcs = np.array(bcs[0:160])
#test_bcs = np.array(bcs[160:])

#train_data = tf.data.Dataset.from_tensor_slices((train_levels, train_bcs))
#test_data = tf.data.Dataset.from_tensor_slices((test_levels, test_bcs))
#model.fit(train_data, epochs=10, validation_data=test_data)

#model.fit(train_levels, train_bcs, verbose=0, epochs=100)
#mae = model.evaluate(test_levels, test_bcs, verbose=0)
#print('>%.3f' % mae)