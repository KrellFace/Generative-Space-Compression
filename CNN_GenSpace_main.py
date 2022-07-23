from inspect import CO_ASYNC_GENERATOR
from typing import List
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
from tensorflow.keras.optimizers import Adam
import csv
import time
from datetime import timedelta
from datetime import datetime


OUTPUT_PATH  = '../output'


#########################################################################################
#FUNCTIONALITY FOR PRODUCING, TRAINING AND SAVING CONVOLUTIONAL NEURAL NETWORKS
########################################################################################

class CNNType(Enum):
    Basic = 1,
    VGG16 = 2

#This module is for training a CNN to predict the BCs of input game levels
#It takes in a set of level_wrappers with BCs calculated, and is instructed to not train on one of the BCs
#It outputs the trained CNN, which should be saved in some form
def train_and_save_CNN(level_wrappers, game, level_shape, tile_type_count, bcs_to_train, cnn_type, output_path, logfile, print_eval = False, save_model = False):

    input_shape = (level_shape[0], level_shape[1], tile_type_count)

    model = get_model(input_shape, len(bcs_to_train), cnn_type)

    
    #print("Pre training model weights:")
    #print(str(np.ndarray.flatten(model.layers[-2].get_weights()[0])[50:80]))

    onehot_matrixes, bcs = get_model_ready_data_from_levelwrapperdict_and_bclist(level_wrappers, bcs_to_train, game)

    model.fit(onehot_matrixes, bcs, verbose=0, epochs=100)
    
    """
    print("Main Model penultimate weights:")
    print("Config:")
    print(model.layers[-2].get_config())
    print("Weights:")
    print(model.layers[-2].get_weights())
    """

    #prediction = model.predict(onehot_matrixes)
    #print("Prediction:")

    #if(print_eval):		
    mae = model.evaluate(onehot_matrixes, bcs, verbose=0)
    # store result
    
    eval_cnn_line = ("MeanAverage Error for CNN of type: " + cnn_type.name + " " + '>%.3f' % mae)
    add_line_to_logfile(logfile, eval_cnn_line)
    #print("MeanAverageError:" + '>%.3f' % mae)
    
    if(save_model):
        model.save(output_path)

    return model


def get_model(in_shape, bc_count, cnn_type):
    #Create convolutional base using a common pattern
    #A stack of Conv2d and MaxPooling2D layers
    if (cnn_type == CNNType.Basic):
        model= models.Sequential()
        model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=in_shape))
        model.add(layers.MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same'))
        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = "same"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same'))
        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = "same"))

        #Add Dense layers to the top of the model
        #This does classification on the last tensor output
        #Takes 1D vectors as an input, so first we need to flatten
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu', name = "penultimate_layer"))
        #model.add(layers.Dense(2, activation='relu', name = "penultimate_layer"))

        model.add(layers.Dense(bc_count))
        #model.summary()
        opt = Adam(bl_adam_opt_lr)
        model.compile(loss='mae', optimizer=opt)
        #print("Model summary:")
        #print(model.summary())
        
        return model
    elif (cnn_type == CNNType.VGG16):
        model = models.Sequential()
        model.add(layers.Conv2D(input_shape=in_shape,filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2), padding='same'))
        model.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2), padding='same'))
        model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2), padding='same'))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2), padding='same'))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2), padding='same'))
        model.add(layers.Flatten())
        model.add(Dense(units=4096,activation="relu"))
        model.add(Dense(units=4096,activation="relu"))
        model.add(Dense(units=1000,activation="relu"))
        model.add(Dense(units=bc_count))
        opt = Adam(vgg_adam_opt_lr)
        model.compile(loss='mae', optimizer=opt)
        #print("Model summary:")
        #print(model.summary())
        return model
    else:
        print("CNN Type not recognised, please check input")


#########################################################################################
#FUNCTIONALITY FOR GENERATING GENERATIVE SPACE VISUALISATIONS FROM LEVELS+CNN
########################################################################################

def generate_visualisation_from_cnn(level_wrappers, model, game, comp_algo, save_intermediate_output = False, intermed_output_path = '', save_comp_weights = False, comp_weights_path = '', save_visual = False, visual_path = '', cnn_pure = False):
    #First we need to create our intermediate layer model
    compiled_level_weights = get_penultimate_layer_leveloutput(level_wrappers,model, save_intermediate_output,  intermed_output_path)

    data = None
    compressed_weights = None
    folder_dict =  get_folder_and_tiletypedict_for_game(game)['Folder_Dict']

    #print("Cnn impure, following normal process")
    gen_name_list = compiled_level_weights['generator_name'].tolist()
    data = get_compression_algo_projection(compiled_level_weights.drop('generator_name', axis=1), comp_algo)
    data[0]['generator_name'] = gen_name_list
    plot_compressed_data(data[0], data[1], [(comp_algo.name + ' 1'), (comp_algo.name + ' 2')],visual_path, list(folder_dict.keys()))
    
    compressed_weights = data[0]
    #Otherwise, just rename the columns
    """
    else:
        cnn_puredata = compiled_level_weights.rename(columns={0: 'CNN_Output 1', 1: 'CNN_Output 2'})
        #print("Pure data head:")
        #print(cnn_puredata.head)
        plot_compressed_data(cnn_puredata, 0, [('CNN_Output 1'), ('CNN_Output 2')],visual_path, list(folder_dict.keys()))
        
        compressed_weights = cnn_puredata
    """

    #print("Comp data head:")
    #print(data[0].head)

    #folder_dict =  get_folder_and_tiletypedict_for_game(game)['Folder_Dict']
    #plot_compressed_data(data[0], data[1], [(comp_algo.name + ' 1'), (comp_algo.name + ' 2')],visual_path, list(folder_dict.keys()))

    #compressed_weights = data[0]

    if(save_comp_weights):
        
        compressed_weights.to_csv(comp_weights_path)

    return compressed_weights

#Return a Dataframe of all level specific extracted weights
def get_penultimate_layer_leveloutput(level_wrappers, model, save_output = False, output_path = ''):

    intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[-2].output)

    total_intermediate_outputs = list()

    """
    print("Intermediate Model final layer weights:")
    print("Config:")
    print(intermediate_layer_model.layers[-1].get_config())
    print("Weights:")
    print(intermediate_layer_model.layers[-1].get_weights())
    """
    #print("Intermed model weights post training:")
    #print(str(np.ndarray.flatten(intermediate_layer_model.layers[-1].get_weights()[0])[50:80]))

    for level_key in level_wrappers:
        #print("Level key being predicted with intermed model: " +  level_key)
        #print("And onehot rep: : " +  str(level_wrappers[level_key].onehot_rep))
        onehot_rep = level_wrappers[level_key].onehot_rep
        level_weights = intermediate_layer_model.predict(onehot_rep[np.newaxis, ...])
        #print(intermediate_output.shape)
        flat = np.ndarray.flatten(level_weights)

        #print("Onehot Rep (200 to 230:) for level key: " + level_key)
        #print(str(np.ndarray.flatten(onehot_rep)[200:230]))
        
        #print("Intermed level weights (200 to 230:) for level key: " + level_key)
        #print(str(flat[200:230]))


        #level_weights_df = pd.DataFrame(flat.reshape(-1, len(flat)), columns=range(0, 64), index=[level_key])
        #level_weights_df = pd.DataFrame(flat.reshape(-1, len(flat)), columns=range(0, 4096), index=[level_key])
        level_weights_df = pd.DataFrame(flat.reshape(-1, len(flat)), columns=range(0, len(flat)), index=[level_key])
        level_weights_df.insert(0,"generator_name",level_wrappers[level_key].generator_name)
        total_intermediate_outputs.append(level_weights_df)
    
    compiled = pd.concat(total_intermediate_outputs, ignore_index=False)

    if(save_output):
        compiled.to_csv(output_path)
    
    return compiled


###########################################################################################
#FUNCTIONALITY FOR EVALUATING A 2D PROJECTION OF LEVELS, BASED ON CORRELATION BETWEEN VECTOR DISTANCES AND BCS
############################################################################################

def calculate_linncorr_of_projection_for_bc(game, level_dict, correlation_data, bcs, comp_type, base_file_path, linn_corrs_path = 'CNNComp_LinearCorrelations.txt'):

    #print("Corr data head")
    #print(correlation_data.head)
    update_levelwrapper_datacomp_features(level_dict, correlation_data, comp_type)

    final_data = gen_compression_dist_df_from_leveldict(level_dict, game, [comp_type], bcs, base_file_path )

    linncorrs = list()
    linncorrs.append(game.name)
    temp_corrsdict = get_linear_correlations_from_df(final_data, [comp_type], bcs, linn_corrs_path)
    #Extract only BCDist name, linncorr and P value
    #onlykey = list(temp_corrsdict)[0]
    #output = list()
    #output+=[[temp_corrsdict[onlykey][1]], [temp_corrsdict[onlykey][2]], [temp_corrsdict[onlykey][3]]]
    return temp_corrsdict


###########################################################################################
#FUNCTIONALITY FOR VANILLA DIMENSIONALITY REDUCTION RUNS ON SETS OF LEVELS
###########################################################################################

def vanilla_dr_for_levelset(level_wrappers, game, comp_type, bc_list, base_file_path):

    #Create Folder
    os.makedirs(base_file_path)

    algo_output = multigenerator_compression(level_wrappers, game, comp_type, 2, True, base_file_path)
    updated_dict = update_levelwrapper_datacomp_features(level_wrappers, algo_output, comp_type)
    comp_dist_df = gen_compression_dist_df_from_leveldict(updated_dict,game, [comp_type],bc_list, base_file_path)

    lincorrdict = dict()
    lincorrsfilepath = Path(base_file_path + "BenchmarkDR_linncorrs.txt")
    temp_corrsdict = get_linear_correlations_from_df(comp_dist_df, [comp_type], bc_list, lincorrsfilepath)
    #Look through dictionary of linear correlations, add them to outout
    for key in temp_corrsdict:
        linncorrs = list()
        linncorrs.append(game.name)
        linncorrs+=temp_corrsdict[key]
        lincorrdict[game.name + " " + key] = linncorrs

    #finallinncorrsdf = pd.DataFrame.from_dict(lincorrdict, orient = 'index', columns = ['Game', 'Compression_Dist',  'BCDist', 'Spearman Coeff', 'Spearman P Val'] )
    #finaloutputpath = Path(base_file_path+ "BenchmarkDR_linncorrs.csv")
    #finallinncorrsdf.to_csv(finaloutputpath, index = False)

    return lincorrdict


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
            levelbcs.append(levelwrappers[key].bc_vals[bc])
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

def add_line_to_logfile(logfile, line):
    print(line)
    with open(logfile, "a") as file_object:
        file_object.write("\n")
        file_object.write(datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
        file_object.write("\n")
        file_object.write(line)

#####################################################################
#WRAPPER METHODS FOR FULL RUNS
###################################################################

def fullrun_VGG16_and_benchmarks(game, bc_list, cnn_output_comp_algo, baseline_comp_algo, output_path_root, run_count, lvls_per_run, tt_split, validate_bcs_individually):

    all_runs_vgg16_dict = dict()
    all_runs_basiccnn_dict = dict()
    all_runs_dr_dict = dict()

    os.makedirs(output_path_root)
    logfile = output_path_root +"log.txt"
    f = open(logfile, 'w')

    start = time.time()

    startline = "Commencing full experiment for game: " + game.name + " with Run Count: " + str(run_count)
    add_line_to_logfile(logfile, startline)
    bcs_line = "BCs to be processed: "
    for bc in bc_list:
        bcs_line+= (bc.name + ", ")
    add_line_to_logfile(logfile, bcs_line)
    lvls_line = "Levels per run: " + str(lvls_per_run) + ", and train/test split ratio: " + str(tt_split)
    add_line_to_logfile(logfile, lvls_line)
    adam_line = "optimizer learning rates. VGG: " + str(vgg_adam_opt_lr) + " Basic: " + str(bl_adam_opt_lr) 
    add_line_to_logfile(logfile, adam_line)


    for i in range(run_count):

        run_log_line = "Run " + str(i) + " - Starting with runtime " + str(timedelta(seconds=(time.time()-start)))
        add_line_to_logfile(logfile, run_log_line)

        run_path = output_path_root +"/RUN "+str(i)+"/"

        r_levels_line = ("Retrieving " + str(lvls_per_run) + " levels for Run: "+ str(i) + " for game: " + game.name + " at time " + str(timedelta(seconds=(time.time()-start))))
        add_line_to_logfile(logfile, r_levels_line)

        level_wrappers = dict()
        if(game == Game.Loderunner):
            level_wrappers=  get_randnum_levelwrappers_for_game(game, 150)
        else:
            level_wrappers = get_randnum_levelwrappers_for_game(game, lvls_per_run)

        #VGG16 Compression
        start_vgg16_line = ("Levels retrieved. Starting VGG16 compression and visualisation" + " at time " + str(timedelta(seconds=(time.time()-start))))
        add_line_to_logfile(logfile, start_vgg16_line)
        
        vgg16_corr_dict = generate_and_validate_CNN_compressions_for_bc_set(game, level_wrappers, bc_list, CNNType.VGG16, cnn_output_comp_algo, run_path+"/VGG16_RUNS/", logfile, start, validate_bcs_individually, tt_split)

        for key in vgg16_corr_dict:
            all_runs_vgg16_dict["Run "+str(i)+" "+key] = vgg16_corr_dict[key]

        #Basic CNN Compression
        start_basicnn_line = ("VGG16 Compression completed. Starting Basic CNN compression and visualisation" + " at time " + str(timedelta(seconds=(time.time()-start))))
        add_line_to_logfile(logfile, start_basicnn_line)
        
        basiccnn_corr_dict = generate_and_validate_CNN_compressions_for_bc_set(game, level_wrappers, bc_list, CNNType.Basic, cnn_output_comp_algo, run_path+"/CNNBasic_RUNS/", logfile, start, validate_bcs_individually, tt_split)

        for key in basiccnn_corr_dict:
            all_runs_basiccnn_dict["Run "+str(i)+" "+key] = basiccnn_corr_dict[key]

        #Vanillda DR Compression
        vanilla_dr_line = ("Run: "+ str(i) + " CNN processes complete. Starting vanilla DR" + " at time " + str(timedelta(seconds=(time.time()-start))))
        add_line_to_logfile(logfile, vanilla_dr_line)
        dr_corr_dict = vanilla_dr_for_levelset(level_wrappers, game, baseline_comp_algo, bc_list, run_path + "/DR_RUNS/")

        for key in dr_corr_dict:
            all_runs_dr_dict["Run "+str(i)+" "+key] = dr_corr_dict[key]
        
        run_complete_line = ("Run: "+ str(i) + " completed" + " at time " + str(timedelta(seconds=(time.time()-start))))
        add_line_to_logfile(logfile, run_complete_line)
        
    allruns_line = ("All runs completed. Generating final data " + " at time " + str(timedelta(seconds=(time.time()-start))))
    add_line_to_logfile(logfile, allruns_line)
    finalVGG16linncorrsdf = pd.DataFrame.from_dict(all_runs_vgg16_dict, orient = 'index', columns = ['Compression_Dist',  'BCDist', 'Spearman Coeff', 'Spearman P Val'] )
    finalvgg16outputpath = Path(output_path_root+ "/Total VGG16 Lin Corrs.csv")
    finalVGG16linncorrsdf.to_csv(finalvgg16outputpath, index = True)

    
    finalbasicCNNlindoorsdf = pd.DataFrame.from_dict(all_runs_basiccnn_dict, orient = 'index', columns = ['Compression_Dist',  'BCDist', 'Spearman Coeff', 'Spearman P Val'] )
    finalbaseoutputpath = Path(output_path_root+ "/Total Basic CNN Lin Corrs.csv")
    finalbasicCNNlindoorsdf.to_csv(finalbaseoutputpath, index = True)

    finalDRlinncorrsdf = pd.DataFrame.from_dict(all_runs_dr_dict, orient = 'index', columns = ['Game', 'Compression_Dist',  'BCDist', 'Spearman Coeff', 'Spearman P Val'] )
    finalDRoutputpath = Path(output_path_root+ "/Total DR Lin Corrs.csv")
    finalDRlinncorrsdf.to_csv(finalDRoutputpath, index = True)

    return



def generate_and_validate_CNN_compressions_for_bc_set(game, level_wrappers, bc_list,cnn_type, comp_algo, output_path_root, logfile, starttime, bcs_individually, tt_split):
    
    level_shape = get_level_heightandwidth_for_game(game)
    tile_type_count = get_folder_and_tiletypedict_for_game(game)['Tile_Type_Dict']["CountOfNumericTileTypes"]
    update_levelwrappers_with_onehot(level_wrappers, game)

    all_bcs_dicts = list()

    if bcs_individually:
        for i in range(len(bc_list)):
            
            train_bcs = bc_list.copy()
            train_bcs.pop(i)
            curr_bc = bc_list[i]

            start_cnn_line = ("Starting CNN training for BC" + curr_bc.name + " with train/test split: " + str(tt_split) + " with runtime " + str(timedelta(seconds=(time.time()-starttime))))
            add_line_to_logfile(logfile, start_cnn_line)

            bcrun_fileroot = output_path_root + "/"+curr_bc.name + "_Run/"

            lvlwrapseries = pd.Series(level_wrappers)
            #print(lvlwrapseries.head)
            training_data , test_data  = [i.to_dict() for i in train_test_split(lvlwrapseries, train_size=tt_split)] 

            full_model = train_and_save_CNN(training_data, game, level_shape, tile_type_count, train_bcs, cnn_type, bcrun_fileroot + 'model', logfile, True, True)

            #get_penultimate_layer_leveloutput(leveldict, test_model, True, output_files_name + 'IntermediateOutput.csv')
            
            start_visual_line = ("Starting CNN trained. Starting visualisation generation with runtime " + str(timedelta(seconds=(time.time()-starttime))))
            add_line_to_logfile(logfile, start_visual_line)

            comp_weights = generate_visualisation_from_cnn(test_data, full_model, game, comp_algo, True, bcrun_fileroot + 'IntermediateWeights.csv',  True, bcrun_fileroot + 'CompressedWeights.csv', True, bcrun_fileroot + "CNN_Compression_Visual.png")

            
            start_visual_line = ("Visualisation generation finished. Starting lincoor calculation with runtime " + str(timedelta(seconds=(time.time()-starttime))))
            add_line_to_logfile(logfile, start_visual_line)

            linncorrs = calculate_linncorr_of_projection_for_bc(game, test_data, comp_weights, [curr_bc], comp_algo, bcrun_fileroot, output_path_root +"CNNComp_LinearCorrelations.txt")

            #print("Linn corrs for BC: "  + curr_bc.name)
            #print(linncorrs)

            all_bcs_dicts.append(linncorrs)
    else:
        start_cnn_line = ("Starting CNN training for full BC list with train/test split: " + str(tt_split) + " with runtime " + str(timedelta(seconds=(time.time()-starttime))))
        add_line_to_logfile(logfile, start_cnn_line)

        bcrun_fileroot = output_path_root + "/Combined BCs_Run/"

        lvlwrapseries = pd.Series(level_wrappers)
        #print(lvlwrapseries.head)
        training_data , test_data  = [i.to_dict() for i in train_test_split(lvlwrapseries, train_size=tt_split)] 

        full_model = train_and_save_CNN(training_data, game, level_shape, tile_type_count, bc_list, cnn_type, bcrun_fileroot + 'model', logfile, True, True)

        #get_penultimate_layer_leveloutput(leveldict, test_model, True, output_files_name + 'IntermediateOutput.csv')
        
        start_visual_line = ("Starting CNN trained. Starting visualisation generation with runtime " + str(timedelta(seconds=(time.time()-starttime))))
        add_line_to_logfile(logfile, start_visual_line)

        comp_weights = generate_visualisation_from_cnn(test_data, full_model, game, comp_algo, True, bcrun_fileroot + 'IntermediateWeights.csv',  True, bcrun_fileroot + 'CompressedWeights.csv', True, bcrun_fileroot + "CNN_Compression_Visual.png")

        
        start_visual_line = ("Visualisation generation finished. Starting lincoor calculation with runtime " + str(timedelta(seconds=(time.time()-starttime))))
        add_line_to_logfile(logfile, start_visual_line)

        linncorrs = calculate_linncorr_of_projection_for_bc(game, test_data, comp_weights, bc_list, comp_algo, bcrun_fileroot, output_path_root +"CNNComp_LinearCorrelations.txt")

        #print("Linn corrs for BC: "  + curr_bc.name)
        #print(linncorrs)

        all_bcs_dicts.append(linncorrs)

    assembled_bc_dict = dict()
    for d in all_bcs_dicts:
        assembled_bc_dict.update(d)

    return assembled_bc_dict

def batches_of_full_runs(games, vgg_lrs, basic_lrs, batches_path, rn_cnt, levels_per_run, tt_split,  process_bcs_individually):

    #Check that input arrays match
    if not (len(games) == len(vgg_lrs) and len(games) == len(basic_lrs) ):
        print("Input arrays did not have matching sizes")
        return

    for i in range(len(games)):
        gamebcs = get_BCs_for_game(games[i])
        global vgg_adam_opt_lr
        vgg_adam_opt_lr = vgg_lrs[i]
        global bl_adam_opt_lr
        bl_adam_opt_lr = basic_lrs[i]

        batch_name = batches_path + "/Batch-" + str(i)+"-Game-" + games[i].name +"VGLR-"+str(vgg_lrs[i])+"BaseLR-"+str(basic_lrs[i])+"/"
        fullrun_VGG16_and_benchmarks(games[i], gamebcs, CompressionType.PCA, CompressionType.PCA, batch_name, rn_cnt,levels_per_run, tt_split,  process_bcs_individually)




##############################################################
#RUN EXPERIMENTS
#####################################################

#Individual Run Set generation

rungame = Game.Boxoban
bl_compalgo = CompressionType.PCA
cnn_compmode = CompressionType.PCA
#cnn_type = CNNType.VGG16
process_bcs_individually = False
rn_cnt = 5
levels_per_run = 1000
tt_split = .8
vgg_adam_opt_lr = 0.00005
bl_adam_opt_lr = 0.1

output_files_name = OUTPUT_PATH +'/5 Runs Boxoban VGG-00005 BL-0.1/'

#bcs = get_BCs_for_game(game)

#fullrun_VGG16_and_benchmarks(game, bcs, cnn_compmode, bl_compalgo, output_files_name, rn_cnt,levels_per_run, tt_split,  process_bcs_individually)


#Testing batches
game_batch = [Game.Mario, Game.Mario, Game.Boxoban, Game.Boxoban]
vglr_batch = [0.001, 0.0001, 0.00001, 0.000001]
baselr_batch = [0.01, 0.01, 0.0001, 0.00001]
batch_path = OUTPUT_PATH + '/VarriedLR_Batches/'

batches_of_full_runs(game_batch, vglr_batch, baselr_batch, batch_path, 1, 40, 0.8,  False)

#leveldict = get_randnum_levelwrappers_for_game(game, 1000)

#Testing a full run
#generate_and_validate_CNN_compressions_for_bc_set(game, leveldict, bcs, compalgo, output_files_name)

#Testing DR benchmarking
#vanilla_dr_for_levelset(leveldict, game, compalgo, bcs, output_files_name)


#Testing modules individually
#level_shape = get_level_heightandwidth_for_game(game)
#tile_type_count = get_folder_and_tiletypedict_for_game(game)['Tile_Type_Dict']["CountOfNumericTileTypes"]
#Loderunner and Mario Test BCs
#bcs_to_train = [BCType.EmptySpace, BCType.EnemyCount,BCType.Linearity]
#bc_to_test = BCType.Density

#test_model = train_and_save_CNN(leveldict, game, level_shape, tile_type_count, bcs_to_train, output_files_name + 'testModel', True, True)

#get_penultimate_layer_leveloutput(leveldict, test_model, True, output_files_name + 'IntermediateOutput.csv')

#comp_weights = generate_visualisation_from_cnn(leveldict, test_model, game, compalgo, True, output_files_name + 'IntermediateWeights.csv',  True, output_files_name + 'CompressedWeights.csv' )

#linncorrs = calculate_linncorr_of_projection_for_bc(leveldict, comp_weights, bc_to_test, compalgo, output_files_name, output_files_name +"LinearCorrelations.txt")

#print("Linn corrs:")
#print(linncorrs)