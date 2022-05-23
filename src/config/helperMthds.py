import glob
import math
from typing import Union
from pathlib import Path

from src.config.enumsAndConfig import *
from src.lvlClasses.boxobanLevel import BoxobanLevel
from src.lvlClasses.marioLevel import MarioLevel
from src.lvlClasses.loderunnerLevel import LoderunnerLevel
import src.func.lvlImport as lvlImport

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

def calculateDistance(x1,y1,x2,y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

#Generate labels of type "String" + n 
def gen_component_labels_for_n(label, n):
    output_labels = list()
    for x in range (1, n+1):
        output_labels.append(label + str(x))
    return output_labels

def get_randnum_levelwrappers_for_game(game, maxlvlsevaled):
    if game == Game.Boxoban:
        return lvlImport.get_randnum_levelwrappers_boxoban(boxoban_folders_dict, maxlvlsevaled)
    else: 
        return lvlImport.get_randnum_levelwrappers(game, maxlvlsevaled)    


def get_folder_and_tiletypedict_for_game(game):
    output = dict()
    if (game == Game.Mario):
        output['Folder_Dict'] = mario_folders_dict
        output['Tile_Type_Dict'] = mario_tiletypes_dict_condensed
    elif (game == Game.Boxoban):
        output['Folder_Dict']= boxoban_folders_dict
        output['Tile_Type_Dict'] = boxoban_tiletypes_dict
    elif (game == Game.Loderunner):
        output['Folder_Dict'] = loderunnder_folders_dict
        output['Tile_Type_Dict'] = lr_tiletypes_dict
    else:
        print("Game type not recognised")
    return output


def generate_levelwrapper_for_game(game, level_name, generator_name, source_file, char_rep):
    if (game == Game.Mario):
        return MarioLevel(level_name, generator_name,source_file, char_rep)
    elif (game == Game.Boxoban):
        return BoxobanLevel(level_name, generator_name, source_file,char_rep)
    elif (game == Game.Loderunner):
        return LoderunnerLevel(level_name, generator_name, source_file,char_rep)

#Get the location values in compressed space for a pair of levels for a specified algorithm
def get_compvals_for_algolist_for_levelpair(level1, level2, algolist):
    vals = []
    for algo in algolist:
        if (algo == CompressionType.PCA):
            vals+= [level1.PC1Val, level1.PC2Val, level2.PC1Val, level2.PC2Val]
        elif (algo == CompressionType.SVD):
            vals+= [level1.SVD1Val, level1.SVD2Val, level2.SVD1Val, level2.SVD2Val]
        elif (algo == CompressionType.MCA):
            vals+= [level1.MCA1Val, level1.MCA2Val, level2.MCA1Val, level2.MCA2Val]
        elif (algo == CompressionType.TSNE):
            vals+= [level1.TSNEVal1, level1.TSNEVal2, level2.TSNEVal1, level2.TSNEVal2]
        elif (algo == CompressionType.KPCA_SIGMOID):
            vals+=[level1.KPCASigmoidVal1, level1.KPCASigmoidVal2, level2.KPCASigmoidVal1, level2.KPCASigmoidVal2]
        elif (algo == CompressionType.KPCA_COSINE):
            vals+=[level1.KPCACosineVal1, level1.KPCACosineVal2, level2.KPCACosineVal1, level2.KPCACosineVal2]    
        elif (algo == CompressionType.KPCA_POLY):
            vals+=[level1.KPCAPolyVal1, level1.KPCAPolyVal2, level2.KPCAPolyVal1, level2.KPCAPolyVal2]           
        elif (algo == CompressionType.KPCA_RBF):
            vals+=[level1.KPCARbfVal1, level1.KPCARbfVal2, level2.KPCARbfVal1, level2.KPCARbfVal2]                        
        else:
            print("Algo not recognised in get vals method")
    return vals

#Get the distance between the location of two levels in compressed space for a given algorithm
def get_distances_for_algolist_for_levelpair(level1, level2, algolist):
    distances = []
    for algo in algolist:
        if (algo == CompressionType.PCA):
            distances.append(calculateDistance(level1.PC1Val, level1.PC2Val, level2.PC1Val, level2.PC2Val))
        elif (algo == CompressionType.SVD):
            distances.append(calculateDistance(level1.SVD1Val, level1.SVD2Val, level2.SVD1Val, level2.SVD2Val))
        elif (algo == CompressionType.MCA):
            distances.append(calculateDistance(level1.MCA1Val, level1.MCA2Val, level2.MCA1Val, level2.MCA2Val))
        elif (algo == CompressionType.TSNE):
            distances.append(calculateDistance(level1.TSNEVal1, level1.TSNEVal2, level2.TSNEVal1, level2.TSNEVal2))
        elif (algo == CompressionType.KPCA_SIGMOID):
            distances.append(calculateDistance(level1.KPCASigmoidVal1, level1.KPCASigmoidVal2, level2.KPCASigmoidVal1, level2.KPCASigmoidVal2))
        elif (algo == CompressionType.KPCA_COSINE):
            distances.append(calculateDistance(level1.KPCACosineVal1, level1.KPCACosineVal2, level2.KPCACosineVal1, level2.KPCACosineVal2))  
        elif (algo == CompressionType.KPCA_POLY):
            distances.append(calculateDistance(level1.KPCAPolyVal1, level1.KPCAPolyVal2, level2.KPCAPolyVal1, level2.KPCAPolyVal2))           
        elif (algo == CompressionType.KPCA_RBF):
            distances.append(calculateDistance(level1.KPCARbfVal1, level1.KPCARbfVal2, level2.KPCARbfVal1, level2.KPCARbfVal2))
        else:
            print("Algo not recognised in get distances method")
    return distances

def get_bcvals_for_bclist_for_levelpair(level1, level2, bclist):
    vals = []
    for bc in bclist:
        if (bc == BCType.EmptySpace):
            vals+=[level1.empty_space, level2.empty_space]
        elif (bc == BCType.EnemyCount):
            vals+=[level1.enemy_count, level2.enemy_count]
        elif (bc == BCType.Linearity):
            vals+=[level1.linearity, level2.linearity]
        elif (bc == BCType.Contiguity):
            vals+=[level1.contiguity, level2.contiguity]
        else:
            print("BC Type not recognised")
    return vals

def get_differences_for_bclist_for_levelpair(level1, level2, bclist):
    differences = []
    for bc in bclist:
        if (bc == BCType.EmptySpace):
            differences.append(abs(level1.empty_space - level2.empty_space))
        elif (bc == BCType.EnemyCount):
            differences.append(abs(level1.enemy_count - level2.enemy_count))
        elif (bc == BCType.Linearity):
            differences.append(abs(level1.linearity - level2.linearity))
        elif (bc == BCType.Contiguity):
            differences.append(abs(level1.contiguity - level2.contiguity))
        else:
            print("BC type not recognised")
    return differences

def gen_distnames_for_algos(algolist):
    returnlist = []
    for algo in algolist:
        returnlist.append(algo.name + "Dist") 
    return returnlist

def gen_valanddist_colnames_for_algos(algolist):
    returnlist = []
    #Value column names
    for algo in algolist:
        returnlist.append("1stLvlVal "+ algo.name + " 1") 
        returnlist.append("1stLvlVal  "+ algo.name + " 2") 
        returnlist.append("2ndLvlVal  "+ algo.name + " 1") 
        returnlist.append("2ndLvlVal  "+ algo.name + " 2") 
    returnlist+=gen_distnames_for_algos(algolist)
    return returnlist
    
def gen_diffnames_for_bcs(bclist):
    returnlist = []
    for bc in bclist:
        returnlist.append(bc.name + "Dist")
    return returnlist

def gen_valanddiff_colnames_for_bcs(bclist):
    returnlist = []
    for bc in bclist:
        returnlist.append("1stLvlVal "+ bc.name) 
        returnlist.append("2ndLvlVal "+ bc.name) 
    returnlist+=gen_diffnames_for_bcs(bclist)
    return returnlist

def get_single_subpath_part(base_dir: Union[Path, str], n:int) -> str:
    if n ==0:
        return Path(base_dir).name
    for _ in range(n):
        base_dir = Path(base_dir).parent
    return getattr(base_dir, "name")

def get_n_last_subparts_path(base_dir: Union[Path, str], n:int) -> Path:
    return Path(*Path(base_dir).parts[-n-1:])