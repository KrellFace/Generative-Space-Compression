from src.func.mainComp import *
from src.config.enumsAndConfig import *
from src.config.helperMthds import *
from scipy.stats import spearmanr



test_game = Game.Mario
lvl_count = 100
bcs = [BCType.EmptySpace, BCType.EnemyCount, BCType.Linearity, BCType.Density] 


print("Start")
overall_start_time = datetime.now()


#Retrieve level wrappers

level_wrappers = get_randnum_levelwrappers_for_game(test_game, lvl_count)

print("Wrap count: " + str(len(level_wrappers)))

#Convert Levelwrapper dictionary into dataframe of BC values ready for analysis

lvl_dict = dict()
for lvlkey in level_wrappers.keys():

    lvlbcs = []
    for bc in bcs:
        lvlbcs.append(level_wrappers[lvlkey].bc_vals[bc])
    lvl_dict[lvlkey] = lvlbcs

df_cols = ['EmptySpace','EnemyCount','Linearity','Density']

lvl_df = pd.DataFrame.from_dict(lvl_dict, orient='index', columns=df_cols)

#print("Lvl df head:")
#print(lvl_df.head())

#print("-1:")
#print(lvl_df[['EmptySpace']].values.reshape(-1))

#Calculate Linear Correlation for each BC pair
#for bctype1 in df_cols:
for bctype1 in bcs:
    #print(bctype1)
    for bctype2 in bcs:
        if(bctype1!=bctype2):
            print("Data for BC Pair: " + bctype1.name + ", " + bctype2.name )
            bc1list = lvl_df[[bctype1.name]].values.reshape(-1)
            bc2list = lvl_df[[bctype2.name]].values.reshape(-1)

            spcorr, pspval = spearmanr(bc1list, bc2list)
            print('Sp corr for BC pair' + ' : %.3f' % spcorr + " with P Value: " + str("{:.2f}".format(pspval)))

            tot_othercorr = 0

            #Calculate correlation with alt BCs
            for otherbc in bcs:
                if (otherbc!=bctype1 and otherbc!= bctype2):
                    maxcorr = 0
                    otherbclist = lvl_df[[otherbc.name]].values.reshape(-1)
                    spcorr1, pspval1 = spearmanr(bc1list, otherbclist)
                    print('Sp corr for BC sub pair: ' + bctype1.name + ", " + otherbc.name + ' : %.3f' % spcorr1 + " with P Value: " + str("{:.2f}".format(pspval)))
                    if(abs(spcorr1)>maxcorr):
                        maxcorr = abs(spcorr1)
                    spcorr2, pspval2 = spearmanr(bc2list, otherbclist)
                    print('Sp corr for BC sub pair: ' + bctype2.name + ", " + otherbc.name + ' : %.3f' % spcorr2 + " with P Value: " + str("{:.2f}".format(pspval)))
                    if(abs(spcorr2)>maxcorr):
                        maxcorr = abs(spcorr2)
                    print('Best abs corr found with sub BC: ' + otherbc.name+ " = " + str(maxcorr) )
                    tot_othercorr+=maxcorr
            
            print("Average other BC corr: " + str(tot_othercorr/(len(bcs)-2)))
                    


#Calculate linear correlations 
        
#self.bc_vals[BCType.EmptySpace] =  temp_emptyspace
#self.bc_vals[BCType.EnemyCount] =  temp_enemycount
#self.bc_vals[BCType.Linearity] =  temp_linearity
#self.bc_vals[BCType.Density] =  total_density/len(char_rep[0])


runtime_seconds=  datetime.now () -overall_start_time
runtime_minutes = runtime_seconds/60
print("Total Runtime: " + str(runtime_minutes) + " minutes")