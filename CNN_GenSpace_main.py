from src.func.mainComp import *
from src.config.enumsAndConfig import *
from src.config.helperMthds import *
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import datasets, layers, models, Model
import csv


OUTPUT_PATH  = '../output'


#########################################################################################
#FUNCTIONALITY FOR PRODUCING, TRAINING AND SAVING CONVOLUTIONAL NEURAL NETWORKS
########################################################################################

#This module is for training a CNN to predict the BCs of input game levels
#It takes in a set of level_wrappers with BCs calculated, and is instructed to not train on one of the BCs
#It outputs the trained CNN, which should be saved in some form
def train_and_save_CNN(level_wrappers, game, level_shape, tile_type_count, bcs_to_train, output_path, print_eval = False, save_model = False):


    input_shape = (level_shape[0], level_shape[1], tile_type_count)

    model = get_model(input_shape, len(bcs_to_train))

    onehot_matrixes, bcs = get_model_ready_data_from_levelwrapperdict_and_bclist(level_wrappers,bcs_to_train, game)

    model.fit(onehot_matrixes, bcs, verbose=0, epochs=100)

    #prediction = model.predict(onehot_matrixes)
    #print("Prediction:")

    if(print_eval):		
        mae = model.evaluate(onehot_matrixes, bcs, verbose=0)
        # store result
        print("MeanAverageError:" + '>%.3f' % mae)
    
    if(save_model):
        model.save(output_path)


    return model


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
    #model.add(layers.Dense(2, activation='relu', name = "penultimate_layer"))

    #Two Outputs to the final dense layer as we have 2 BCs
    model.add(layers.Dense(bc_count, name = "final_layer"))
    #model.summary()
    model.compile(loss='mae', optimizer='adam')
    return model


#########################################################################################
#FUNCTIONALITY FOR GENERATING GENERATIVE SPACE VISUALISATIONS FROM LEVELS+CNN
########################################################################################

def generate_visualisation_from_cnn(level_wrappers, model, game, comp_algo, save_intermediate_output = False, intermed_output_path = '', save_comp_weights = False, comp_weights_path = ''):
    #First we need to create our intermediate layer model
    compiled_level_weights = get_penultimate_layer_levelweights(level_wrappers,model, save_intermediate_output,  intermed_output_path)

    
    data = get_compression_algo_projection(compiled_level_weights, comp_algo)

    compressed_weights = data[0]

    if(save_comp_weights):
        
        compressed_weights.to_csv(comp_weights_path)

    return compressed_weights

#Return a Dataframe of all level specific extracted weights
def get_penultimate_layer_levelweights(level_wrappers, model, save_output = False, output_path = ''):

    intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[-2].output)

    total_intermediate_outputs = list()

    for level_key in level_wrappers:
        onehot_rep = level_wrappers[level_key].onehot_rep
        level_weights = intermediate_layer_model.predict(onehot_rep[np.newaxis, ...])
        #print(intermediate_output.shape)
        flat = np.ndarray.flatten(level_weights)
        level_weights_df = pd.DataFrame(flat.reshape(-1, len(flat)), columns=range(0, 64), index=[level_key])
        total_intermediate_outputs.append(level_weights_df)
    
    compiled = pd.concat(total_intermediate_outputs, ignore_index=False)

    #print("Compiled head:")
    #print(compiled.head)

    if(save_output):
        compiled.to_csv(output_path)
    
    return compiled


###########################################################################################
#FUNCTIONALITY FOR EVALUATING A 2D PROJECTION OF LEVELS, BASED ON CORRELATION BETWEEN VECTOR DISTANCES AND A SPECIFIED BC
############################################################################################

def calculate_linncorr_of_projection_for_bc(level_dict, correlation_data, bc, comp_type, base_file_path, linn_corrs_path = 'LinearCorrelations.txt', ):
    update_levelwrapper_datacomp_features(level_dict, correlation_data, comp_type)

    final_data = gen_compression_dist_df_from_leveldict(level_dict, game, [comp_type], [bc], base_file_path )

    linncorrs = list()
    linncorrs.append(game.name)
    temp_corrsdict = get_linear_correlations_from_df(final_data, [comp_type], [bc], linn_corrs_path)
    return temp_corrsdict

###########################################################################################
#FUNCTIONALITY FOR VANILLA DIMENSIONALITY REDUCTION RUNS ON SETS OF LEVELS
###########################################################################################

def vanilla_dr_for_levelset(level_wrappers, game, comp_type, bc_list, base_file_path):
    algo_output = multigenerator_compression(level_wrappers, game, comp_type, 2, False, base_file_path)
    updated_dict = update_levelwrapper_datacomp_features(level_wrappers, algo_output, comp_type)
    comp_dist_df = gen_compression_dist_df_from_leveldict(updated_dict,game, [comp_type],bc_list, base_file_path)

    lincorrdict = dict()
    linncorrs.append(game.name)
    lincorrsfilepath = Path(base_file_path + "BenchmarkDR_linncorrs.txt")
    temp_corrsdict = get_linear_correlations_from_df(comp_dist_df, [comp_type], bc_list, lincorrsfilepath)
    #Look through dictionary of linear correlations, add them to outout
    for key in temp_corrsdict:
        linncorrs = list()
        linncorrs+=[game.name, (runcount+1)]
        linncorrs+=temp_corrsdict[key]
        lincorrdict[game.name + " " + str(runcount) + key] = linncorrs
    runcount += 1

    finallinncorrsdf = pd.DataFrame.from_dict(lincorrdict, orient = 'index', columns = ['Game', 'Run', 'Compression_Dist',  'BCDist', 'Spearman Coeff', 'Spearman P Val'] )
    finaloutputpath = Path(base_file_path+ "BenchmarkDR_linncorrs.csv")
    finallinncorrsdf.to_csv(finaloutputpath, index = False)

    return


########################################
#HELPER AND MISC METHODS
########################################

def get_model_ready_data_from_levelwrapperdict_and_bclist(levelwrappers, bc_list, game):
    matrixes_array = []
    bcs_array = []
    tile_dict = get_folder_and_tiletypedict_for_game(game)['Tile_Type_Dict']

    count = 1
    for key in levelwrappers:
        #onehot_rep = onehot_from_charmatrix(levelwrappers[key].char_rep, tile_dict)
        #Update levelwrapper with the onehot rep
        #levelwrappers[key].onehot_rep = onehot_rep
        onehot_rep = levelwrappers[key].onehot_rep
        #flat_rep = np.ndarray.flatten(onehot_rep)
        matrixes_array.append(onehot_rep)
        levelbcs = []
        for bc in bc_list:
            levelbcs.append(leveldict[key].bc_vals[bc])
        bcs_array.append([levelbcs])



        #if(game == Game.Loderunner or game == Game.Mario):
        #    bcs_array.append([leveldict[key].empty_space, leveldict[key].enemy_count, leveldict[key].linearity])
        #elif(game == Game.Boxoban):
        #    bcs_array.append([leveldict[key].empty_space, leveldict[key].contiguity])
        count+=1
    return np.array(matrixes_array),np.array(bcs_array)

def update_levelwrappers_with_onehot(levelwrappers, game):

    tile_dict = get_folder_and_tiletypedict_for_game(game)['Tile_Type_Dict']

    for key in levelwrappers:
        onehot_rep = onehot_from_charmatrix(levelwrappers[key].char_rep, tile_dict)
        #Update levelwrapper with the onehot rep
        levelwrappers[key].onehot_rep = onehot_rep
        #flat_rep = np.ndarray.flatten(onehot_rep)
    return

#####################################################################
#WRAPPER METHODS FOR FULL RUNS
###################################################################

def fullrun_cnn_and_dr_benchmark(game, bc_list, comp_algo, output_path_root):
    return

def generate_and_validate_compressions_for_bc_set(game, level_wrappers, bc_list, comp_algo, output_path_root):
    
    level_shape = get_level_heightandwidth_for_game(game)
    tile_type_count = get_folder_and_tiletypedict_for_game(game)['Tile_Type_Dict']["CountOfNumericTileTypes"]
    update_levelwrappers_with_onehot(level_wrappers, game)

    #for bc in bc_list:
    for i in range(len(bc_list)):
        train_bcs = bc_list.copy()
        train_bcs.pop(i)
        curr_bc = bc_list[i]
        #print("Bc list: " + str(bc_list))
        #print("Curr BC: " + str(curr_bc))
        #print("Train BC list: " + str(train_bcs))

        #print("Train bcs length: " + str(len(train_bcs)))

        bcrun_fileroot = output_path_root + "/"+curr_bc.name + "_Run/"

        lvlwrapseries = pd.Series(level_wrappers)
        #print(lvlwrapseries.head)
        training_data , test_data  = [i.to_dict() for i in train_test_split(lvlwrapseries, train_size=0.8)] 
        

        full_model = train_and_save_CNN(training_data, game, level_shape, tile_type_count, train_bcs, bcrun_fileroot + 'model', True, True)

        #get_penultimate_layer_levelweights(leveldict, test_model, True, output_files_name + 'IntermediateOutput.csv')

        comp_weights = generate_visualisation_from_cnn(test_data, full_model, game, comp_algo, True, bcrun_fileroot + 'IntermediateWeights.csv',  True, bcrun_fileroot + 'CompressedWeights.csv' )


        linncorrs = calculate_linncorr_of_projection_for_bc(test_data, comp_weights, curr_bc, compalgo, bcrun_fileroot, output_path_root +"LinearCorrelations.txt")

        print("Linn corrs for BC: "  + curr_bc.name)
        print(linncorrs)



##############################################################
#FUNCTIONALITY TESTING
#####################################################

#Testing model generation

game = Game.Boxoban
compalgo = CompressionType.PCA

#LR Full list
#bc_full_list = [BCType.EmptySpace, BCType.EnemyCount,BCType.Linearity, BCType.Density]
#Boxoban BCs
bcs = [BCType.EmptySpace, BCType.Contiguity]
output_files_name = OUTPUT_PATH +'/DRBenchmarkTest/'

leveldict = get_randnum_levelwrappers_for_game(game, 1000)

#Testing a full run
#generate_and_validate_compressions_for_bc_set(game, leveldict, bcs, compalgo, output_files_name)

#Testing DR benchmarking
vanilla_dr_for_levelset(leveldict, game, compalgo, bcs, output_files_name)


#Testing modules individually
#level_shape = get_level_heightandwidth_for_game(game)
#tile_type_count = get_folder_and_tiletypedict_for_game(game)['Tile_Type_Dict']["CountOfNumericTileTypes"]
#Loderunner and Mario Test BCs
#bcs_to_train = [BCType.EmptySpace, BCType.EnemyCount,BCType.Linearity]
#bc_to_test = BCType.Density

#test_model = train_and_save_CNN(leveldict, game, level_shape, tile_type_count, bcs_to_train, output_files_name + 'testModel', True, True)

#get_penultimate_layer_levelweights(leveldict, test_model, True, output_files_name + 'IntermediateOutput.csv')

#comp_weights = generate_visualisation_from_cnn(leveldict, test_model, game, compalgo, True, output_files_name + 'IntermediateWeights.csv',  True, output_files_name + 'CompressedWeights.csv' )

#linncorrs = calculate_linncorr_of_projection_for_bc(leveldict, comp_weights, bc_to_test, compalgo, output_files_name, output_files_name +"LinearCorrelations.txt")

#print("Linn corrs:")
#print(linncorrs)