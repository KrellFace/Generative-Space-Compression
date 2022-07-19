from src.config.enumsAndConfig import *
from src.func.lvlImport import *


#Method for updating the attributes of a set of levelwrappers based on compressed data
def update_levelwrapper_datacomp_features(level_dict, compdf, compression_type):
    if (compression_type == CompressionType.PCA):
        for level in level_dict:
            level_dict[level].PC1Val = compdf.loc[level][compression_type.name+' 1']
            level_dict[level].PC2Val = compdf.loc[level][compression_type.name+' 2']
            #print("While updating levelwrappers. Density is " + str(level_dict[level].bc_vals[BCType.Density]))
    elif (compression_type == CompressionType.MCA):
        for level in level_dict:
            level_dict[level].MCA1Val = compdf.loc[level][compression_type.name+' 1']
            level_dict[level].MCA2Val = compdf.loc[level][compression_type.name+' 2']
    elif (compression_type == CompressionType.SVD):
        for level in level_dict:
            level_dict[level].SVD1Val = compdf.loc[level][compression_type.name+' 1']
            level_dict[level].SVD2Val = compdf.loc[level][compression_type.name+' 2']
    elif (compression_type == CompressionType.TSNE):
        for level in level_dict:
            level_dict[level].TSNEVal1 = compdf.loc[level][compression_type.name+' 1']
            level_dict[level].TSNEVal2 = compdf.loc[level][compression_type.name+' 2']
    elif (compression_type == CompressionType.KPCA_SIGMOID):
        for level in level_dict:
            level_dict[level].KPCASigmoidVal1 = compdf.loc[level][compression_type.name+' 1']
            level_dict[level].KPCASigmoidVal2 = compdf.loc[level][compression_type.name+' 2']
    elif (compression_type == CompressionType.KPCA_COSINE):
        for level in level_dict:
            level_dict[level].KPCACosineVal1 = compdf.loc[level][compression_type.name+' 1']
            level_dict[level].KPCACosineVal2 = compdf.loc[level][compression_type.name+' 2']  
    elif (compression_type == CompressionType.KPCA_POLY):
        for level in level_dict:
            level_dict[level].KPCAPolyVal1 = compdf.loc[level][compression_type.name+' 1']
            level_dict[level].KPCAPolyVal2 = compdf.loc[level][compression_type.name+' 2']            
    elif (compression_type == CompressionType.KPCA_RBF):
        for level in level_dict:
            level_dict[level].KPCARbfVal1 = compdf.loc[level][compression_type.name+' 1']
            level_dict[level].KPCARbfVal2 = compdf.loc[level][compression_type.name+' 2']  
    elif(compression_type == CompressionType.CNN_Output):
        for level in level_dict:
            level_dict[level].CNN_OutputVal1 = compdf.loc[level][compression_type.name+' 1']
            level_dict[level].CNN_OutputVal2 = compdf.loc[level][compression_type.name+' 2']  
            #print("New level_dict vals for cnnoutput: " + str(level_dict[level].CNN_OutputVal1) + ", " + str(level_dict[level].CNN_OutputVal2))

    else:
        print('Algo type not recognised')
    return level_dict

