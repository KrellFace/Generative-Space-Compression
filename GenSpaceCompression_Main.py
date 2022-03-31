from datetime import datetime
from src.func.mainCompressionMethods import *


#Parameters to set
games = [Game.Boxoban, Game.Mario]
algolist = [CompressionType.PCA, CompressionType.SVD, CompressionType.MCA]
tot_lvls_evaled = 100
runs_per_game = 1
visualise = True
fileprefix =  "TestNewFoldsGen19"

overall_start_time = datetime.now()
component_count = 2

multidomain_multiruns(games, component_count, algolist, tot_lvls_evaled, runs_per_game, fileprefix, visualise)
runtime_seconds=  datetime.now () -overall_start_time
runtime_minutes = runtime_seconds/60
print("Total Runtime: " + str(runtime_minutes) + " minutes")

