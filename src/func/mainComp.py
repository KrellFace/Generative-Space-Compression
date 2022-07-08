
import itertools as it 

from src.config.enumsAndConfig import *
from src.config.helperMthds import *
from src.lvlClasses.lvlWrapUpdate import *
from src.visualisation.plotGene import *
from src.func.dfCreate import *
from src.func.compAndCorr import *
from src.visualisation.levelImageGen import *


################################
#MULTIGENERATOR METHODS

def multigenerator_compression(levelwrapper_dict, game, comp_algo, component_count = 2, visualise = False, plot_filename_root = ""):
    game_info = get_folder_and_tiletypedict_for_game(game)
    folder_dict = game_info['Folder_Dict']
    tile_dict = game_info['Tile_Type_Dict']
    height, width = get_level_heightandwidth_for_game(game)

    processed_levels = None
    if (comp_algo == CompressionType.MCA):
        if (game == Game.Mario):
            processed_levels = get_compiled_char_representations_from_level_dict(levelwrapper_dict, height, width, True, mario_tiletypes_charcompression_dict)
        else:
            processed_levels = get_compiled_char_representations_from_level_dict(levelwrapper_dict, height, width)
    else:
        processed_levels = get_compiled_onehot_from_leveldict(levelwrapper_dict, tile_dict, height, width)

    gen_name_list = processed_levels['generator_name'].tolist()

    compressed_info = get_compression_algo_projection(processed_levels.drop('generator_name', axis=1), comp_algo, component_count=component_count)
    #Readding the name of the generator for each level to the list of all levels and their PCs
    compressed_info[0]['generator_name'] = gen_name_list

    if visualise == True:
        plot_filename = plot_filename_root + " " + comp_algo.name + ".png"
        plot_compressed_data(compressed_info[0], compressed_info[1], comp_algo,plot_filename, list(folder_dict.keys()))
    return compressed_info[0]


#Returns a dictionary of level wrappers with their dimensionality reduction algorithm locations specified
def get_and_update_X_levels_for_algo_list(game, component_count, algolist, maxlvlsevaled, visualise = False, file_root = ""):
    print("Starting level wrapper generation")
    level_wrapper_dict = get_randnum_levelwrappers_for_game(game, maxlvlsevaled)

    print("Starting compression algorithm process")
    for algo in algolist:
        print("Running algo : " + algo.name)
        start_time = datetime.now()
        algo_output = multigenerator_compression(level_wrapper_dict, game, algo, component_count, visualise, file_root)
        level_wrapper_dict = update_levelwrapper_datacomp_features(level_wrapper_dict, algo_output, algo)
        print("Algo "+ algo.name + " compression runtime: " + str(datetime.now () -start_time) + " seconds")

    return level_wrapper_dict

#Generates a dataframe of level pairs and the feature distances between them
def gen_compression_dist_df_from_leveldict(level_wrapper_dict, game, algolist, bclist, output_filepath):
    pair_counter = 0
    start_time = datetime.now()
    output_dict = dict()
    #processed_pairs = list()

    uniquepairs = list(it.combinations(level_wrapper_dict, 2))

    #Initialise storage for the closest and furthest level pairs for each compression
    #Stored as Key: [AlgoName, "Closest" or "Furthest"] Value: [CurrentExamplar Row, Current Examplar Value]
    nearfar_exemplar_dict = dict()
    for algo in algolist:
        nearfar_exemplar_dict[algo.name + " Closest"] = [-1, 10000 ,None, None]
        nearfar_exemplar_dict[algo.name + " Furthest"] = [-1, 0,None, None]

    for pair in uniquepairs:
        level1 = level_wrapper_dict[pair[0]]
        level2 = level_wrapper_dict[pair[1]]
        algo_vals_list = get_compvals_for_algolist_for_levelpair(level1, level2,algolist)
        algo_dist_list = get_distances_for_algolist_for_levelpair(level1, level2,algolist)
        bc_vals_list = get_bcvals_for_bclist_for_levelpair(level1, level2, bclist)
        bc_dist_list = get_differences_for_bclist_for_levelpair(level1, level2, bclist)

       # print("BC Vals: " + str(bc_vals_list))
        
        levelpair_row = [level1.name, level1.generator_name, level1.source_file, level2.name , level2.generator_name, level2.source_file] + algo_vals_list+ algo_dist_list + bc_vals_list + bc_dist_list
        output_dict[pair_counter] = levelpair_row

        #Update nearfar dict
        for i in range(0,len(algolist)):
            if (algo_dist_list[i]<nearfar_exemplar_dict[algolist[i].name+" Closest"][1]):
                nearfar_exemplar_dict[algolist[i].name+" Closest"] = [pair_counter, algo_dist_list[i], level1, level2]
            if (algo_dist_list[i]>nearfar_exemplar_dict[algolist[i].name+" Furthest"][1]):
                nearfar_exemplar_dict[algolist[i].name+" Furthest"] = [pair_counter, algo_dist_list[i], level1, level2]
        
        pair_counter+=1

        if (pair_counter%200000 == 0):
            print("200000 level pairs processed. Counter: " + str(pair_counter))
            print("Runtime: " + str(datetime.now () -start_time) + " seconds")

    algo_colnames = gen_valanddist_colnames_for_algos(algolist)
    bc_colnames = gen_valanddiff_colnames_for_bcs(bclist)

    outputdf = pd.DataFrame.from_dict(output_dict, orient = 'index', columns = (['Level1', 'Level1 Generator', 'Level1 File',  'Level2', 'Level2 Generator','Level2 File'] + algo_colnames + bc_colnames))

    #Extract closest furthest exemplars
    exemplardict = dict()
    for exemp in nearfar_exemplar_dict:
        counter_val = nearfar_exemplar_dict[exemp][0]
        exemplardict[exemp] = [exemp] + output_dict[counter_val]
        #Generate images of the exemplar levels
        lvl1outputpath = output_filepath + exemp +  " Level1.png"
        lvl2outputpath = output_filepath + exemp +  " Level2.png"
        generate_image(game, nearfar_exemplar_dict[exemp][2],lvl1outputpath)
        generate_image(game, nearfar_exemplar_dict[exemp][3],lvl2outputpath)
    exemplardf = pd.DataFrame.from_dict(exemplardict, orient = 'index', columns = (['ExemplarType', 'Level1', 'Level1 Generator', 'Level1 File',  'Level2', 'Level2 Generator','Level2 File'] + algo_colnames + bc_colnames))
    
    exemplarsfilepath = output_filepath + "Exemplars.csv"
    exemplardf.to_csv(exemplarsfilepath, index= False )

    curr_time = datetime.now().strftime("%m_%d_%H_%M_%S")
    analytics_filepath= os.path.join(output_filepath + "Analytics.csv")
    return outputdf

#Generates a feature distance dataframe for all level pairs in a folder
def generate_analytics_for_all_level_pairs(game, maxlvlsevaled, component_count, output_filepath, algolist, bclist, visualise = False, file_root = ""):
    
    complete_level_dict = get_and_update_X_levels_for_algo_list(game, component_count, algolist, maxlvlsevaled, visualise, file_root)
    return gen_compression_dist_df_from_leveldict(complete_level_dict,game, algolist,bclist, output_filepath)


def multidomain_multiruns(games, component_count, algolist, tot_lvls_evaled_per_run, runs_per_game, file_prefix, visualise = False):
    bclist = None
    lincorrdict = dict()
    start_time = datetime.now()
    for game in games:

        #if (game == Game.Boxoban):
        #    bclist = [BCType.EmptySpace, BCType.Contiguity]
        #else:
        #    bclist = [BCType.EmptySpace, BCType.EnemyCount, BCType.Linearity]
        bclist = get_BCs_for_game(game)
        runcount = 0
        while runcount < runs_per_game:
            runpath = file_prefix + "/" +game.name +"/Run " + str(runcount+1) + "/"
            output_filepath = runpath + game.name
            #Create parent folder
            Path(output_filepath).parent.mkdir(parents =True, exist_ok=True)
            output = None
            images_root = runpath + game.name
            #Need to hardcode loderunner to be only the 150 we have available
            if (game == Game.Loderunner):
                output = generate_analytics_for_all_level_pairs(game, 150, component_count, output_filepath, algolist, bclist, visualise,images_root)
            else:
                output = generate_analytics_for_all_level_pairs(game, tot_lvls_evaled_per_run, component_count, output_filepath, algolist, bclist, visualise,images_root)
                
            linncorrs = list()
            linncorrs.append(game.name)
            lincorrsfilepath = Path(runpath + game.name + " Run " + str(runcount+1) + ".txt")
            temp_corrsdict = get_linear_correlations_from_df(output, algolist, bclist, lincorrsfilepath)
            #Look through dictionary of linear correlations, add them to outout
            for key in temp_corrsdict:
                linncorrs = list()
                linncorrs+=[game.name, (runcount+1)]
                linncorrs+=temp_corrsdict[key]
                lincorrdict[game.name + " " + str(runcount) + key] = linncorrs
            runcount += 1

            runtime_seconds=  datetime.now () -start_time
            runtime_minutes = runtime_seconds/60
            print("Runtime: " + str(runtime_minutes) + " minutes at run " + str(runcount) + " for game: " + game.name)

    finallinncorrsdf = pd.DataFrame.from_dict(lincorrdict, orient = 'index', columns = ['Game', 'Run', 'Compression_Dist',  'BCDist', 'Spearman Coeff', 'Spearman P Val'] )
    curr_time = datetime.now().strftime("%m_%d_%H_%M_%S")
    finaloutputpath = Path(file_prefix+ "/Total Lin Corrs.csv")
    finallinncorrsdf.to_csv(finaloutputpath, index = False)


