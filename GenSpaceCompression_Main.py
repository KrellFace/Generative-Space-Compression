from turtle import distance
from datetime import datetime
from more_itertools import difference
import numpy as np
import pandas as pd
from pathlib import Path
from pyparsing import col
import itertools as it 
from io import BytesIO
from csv import writer

from sympy import Q, comp 
from LevelWrapper import LevelWrapper
from BoxobanLevel import BoxobanLevel
from MarioLevel import MarioLevel
from LoderunnerLevel import LoderunnerLevel
import LevelImporting 
from EnumsAndConfig import *
import WindowGrabbing
from HelperMethods import *
from LevelWrapperUpdateMethods import *
from PlotGeneration import *
from CompiledDFCreation import *
from CompressionAndCorrelation import *
from LevelImageGeneration import *
from MainCompressionMethods import *


overall_start_time = datetime.now()
component_count = 2
games = [Game.Boxoban, Game.Mario]
#games = [Game.Boxoban]
algolist = [CompressionType.PCA, CompressionType.SVD, CompressionType.MCA]
#algolist = [CompressionType.KPCA_SIGMOID, CompressionType.KPCA_POLY, CompressionType.KPCA_RBF, CompressionType.KPCA_COSINE]
#algolist = [CompressionType.PCA, CompressionType.MCA, CompressionType.SVD, CompressionType.TSNE, CompressionType.KPCA_SIGMOID, CompressionType.KPCA_POLY, CompressionType.KPCA_RBF, CompressionType.KPCA_COSINE]
tot_lvls_evaled = 100
runs_per_game = 1
visualise = True
fileprefix =  "Output Folder Name"
multidomain_multiruns(games, component_count, algolist, tot_lvls_evaled, runs_per_game, fileprefix, visualise)
runtime_seconds=  datetime.now () -overall_start_time
runtime_minutes = runtime_seconds/60
print("Total Runtime: " + str(runtime_minutes) + " minutes")

