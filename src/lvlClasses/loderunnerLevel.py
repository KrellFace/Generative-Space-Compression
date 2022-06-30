from src.lvlClasses.levelWrapper import LevelWrapper
from src.config.enumsAndConfig import BCType

class LoderunnerLevel(LevelWrapper):

    #BC Storage
    #empty_space = None
    #enemy_count = None
    #linearity = None  
    #density = None

    
    solidtiles = ["B","b"]
    standabletiles = ["G", ".", "E","#"]

    def __init__(self, name, generator_name, source_file,char_rep):
        super(LoderunnerLevel, self).__init__(name, generator_name, source_file,char_rep)

    def calc_behavioral_features(self, char_rep):
        temp_emptyspace = 0
        temp_enemycount = 0   
        temp_linearity = 0     
        total_density = 0
        for y in range(0, len(char_rep)):
            for x in range(0,len(char_rep[0])):
                if char_rep[y][x]=='.':
                    temp_emptyspace+=1
                if char_rep[y][x]=='E':
                    temp_enemycount+=1
                #Increase linearity score for each block adjacent in the x axis
                if char_rep[y][x] in ['b', 'B'] and x>0:
                    if char_rep[y][x-1]  in ['b', 'B']:
                        temp_linearity+=1
                if char_rep[y][x]  in ['b', 'B'] and x<len(char_rep[0])-1:
                    if char_rep[y][x+1]  in ['b', 'B']:
                        temp_linearity+=1
                
                #Density calculation - counting blocks that could be stood on
                if char_rep[y][x] in self.solidtiles and y>0:
                    if char_rep[y-1][x] in self.standabletiles:
                        total_density+=1
                
        #self.empty_space = temp_emptyspace
        #self.enemy_count = temp_enemycount
        #self.linearity = temp_linearity
        #self.density = total_density/len(char_rep[0])
        self.bc_vals[BCType.EmptySpace] =  temp_emptyspace
        self.bc_vals[BCType.EnemyCount] =  temp_enemycount
        self.bc_vals[BCType.Linearity] =  temp_linearity
        self.bc_vals[BCType.Density] =  total_density/len(char_rep[0])

        #print("Density for level: " + self.name + ', generator:  ' + self.generator_name +  ', '  + str(self.bc_vals[BCType.Density]))


