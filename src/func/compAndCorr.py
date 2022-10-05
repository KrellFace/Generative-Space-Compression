from src.config.enumsAndConfig import *
from src.config.helperMthds import *

from sklearn.decomposition import  PCA, TruncatedSVD, KernelPCA, SparsePCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from scipy.stats import spearmanr
import prince
import numpy as np
import pandas as pd
from datetime import datetime
from os.path import exists


#Run compression algorithm on representations of game levels and return the reprojected data 
def get_compression_algo_projection(input, compTyp, columnPrefix = '', component_count = 2):

    projectedValues = None
    varExplained = None

    if compTyp == CompressionType.PCA:
        #scaledinput = StandardScaler().fit_transform(input)
        pca = PCA(n_components=component_count)
        projectedValues = pca.fit_transform(input)
        varExplained = pca.explained_variance_ratio_
    elif compTyp == CompressionType.MCA:
        mca = prince.MCA(n_components=component_count)
        mca.fit(input)
        projectedValues = mca.fit_transform(input).to_numpy()
        varExplained = mca.explained_inertia_
    elif compTyp == CompressionType.SVD:
        #scaledinput = StandardScaler().fit_transform(input)
        svd = TruncatedSVD(n_components=component_count, n_iter=7, random_state=42)
        svd.fit(input)
        projectedValues = svd.fit_transform(input)
        varExplained = svd.explained_variance_ratio_
    elif compTyp == CompressionType.TSNE:
        #scaledinput = StandardScaler().fit_transform(input)
        tsne = TSNE(n_components=component_count, n_iter=250, random_state=42)
        tsne.fit(input)
        projectedValues = tsne.fit_transform(input) 
        varExplained = []
    elif compTyp == CompressionType.KPCA_POLY:
        #scaledinput = StandardScaler().fit_transform(input)
        kpca = KernelPCA(n_components=component_count, kernel='poly')
        kpca.fit(input)
        projectedValues = kpca.fit_transform(input) 
        #Calculate Explained Variance
        explained_variance = np.var(projectedValues, axis=0)
        varExplained = explained_variance / np.sum(explained_variance)
    elif compTyp == CompressionType.KPCA_COSINE:
        #scaledinput = StandardScaler().fit_transform(input)
        kpca = KernelPCA(n_components=component_count, kernel='cosine')
        kpca.fit(input)
        projectedValues = kpca.fit_transform(input) 
        #Calculate Explained Variance
        explained_variance = np.var(projectedValues, axis=0)
        varExplained = explained_variance / np.sum(explained_variance)
    elif compTyp == CompressionType.KPCA_RBF:
        #scaledinput = StandardScaler().fit_transform(input)
        kpca = KernelPCA(n_components=component_count, kernel='rbf')
        kpca.fit(input)
        projectedValues = kpca.fit_transform(input) 
        #Calculate Explained Variance
        explained_variance = np.var(projectedValues, axis=0)
        varExplained = explained_variance / np.sum(explained_variance)
    elif compTyp == CompressionType.KPCA_SIGMOID:
        #scaledinput = StandardScaler().fit_transform(input)
        kpca = KernelPCA(n_components=component_count, kernel='sigmoid')
        kpca.fit(input)
        projectedValues = kpca.fit_transform(input) 
        #Calculate Explained Variance
        explained_variance = np.var(projectedValues, axis=0)
        varExplained = explained_variance / np.sum(explained_variance)
    else:
        print("Compression type not recognised")      
                
    labels = gen_component_labels_for_n(columnPrefix+compTyp.name + " ", component_count)

    outputDF = pd.DataFrame(data = projectedValues
                , columns = labels, index = input.index)

    return (outputDF,varExplained)

#Get the spearmans correlation coefficients for each algorithm and behavioral characteristic and print to file
def get_linear_correlations_from_df(df, algolist, bclist, filepath):

    dists_list = gen_distnames_for_algos(algolist)
    bc_diff_list = gen_diffnames_for_bcs(bclist)

    curr_time = datetime.now().strftime("%m_%d_%H_%M_%S")
    outputfile = open(filepath, "a")
    

    output = dict()

    for compression_dist in dists_list:
        vals = df[[compression_dist]].values.reshape(-1)
        for bc in bc_diff_list:
            bcvals = df[[bc]].values.reshape(-1)

            spcorr, pspval = spearmanr(vals, bcvals)
            text = ('Spearmans correlation on ' + compression_dist + ' for BC: ' + bc + ' : %.3f' % spcorr + " with P Value: " + str("{:.2f}".format(pspval)))
            outputfile.write(text + "\n")
            #print(text)
            output[compression_dist + bc] = [compression_dist, bc, spcorr, pspval]
        
    outputfile.close()
    return output