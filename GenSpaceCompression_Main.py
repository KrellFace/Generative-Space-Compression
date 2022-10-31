import time
from src.func.mainComp import *
from src.config.enumsAndConfig import *


#Parameters to set
games = [Game.Loderunner, Game.Mario, Game.Boxoban]
algolist = [CompressionType.PCA, CompressionType.SVD, CompressionType.MCA]
tot_lvls_evaled = 100
runs_per_game = 1
visualise = True
fileprefix =  "ScatterPlotTest1"
component_count = 2

#overall_start_time = datetime.now()
overall_start_time = time.perf_counter()
multidomain_multiruns(games, component_count, algolist, tot_lvls_evaled, runs_per_game, f'output/{fileprefix}', visualise)
runtime_seconds=  time.perf_counter() -overall_start_time
runtime_minutes = runtime_seconds/60
print(f"Total Runtime: {runtime_minutes} minutes")

def main():

    overall_start_time = time.perf_counter()
    multidomain_multiruns(games, component_count, algolist, tot_lvls_evaled, runs_per_game, f'output/{fileprefix}', visualise)
    runtime_seconds=  time.perf_counter() -overall_start_time
    runtime_minutes = runtime_seconds/60
    print(f"Total Runtime: {runtime_minutes} minutes")

if __name__ == "__main__":
    main()
