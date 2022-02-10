from LevelWrapper import LevelWrapper

class LoderunnerLevel(LevelWrapper):

    #BC Storage
    empty_space = None
    enemy_count = None
    linearity = None  

    def __init__(self, name, generator_name, char_rep):
        super(LoderunnerLevel, self).__init__(name, generator_name, char_rep)
        #self.calc_behavioral_features(char_rep)

    def calc_behavioral_features(self, char_rep):
        temp_emptyspace = 0
        temp_enemycount = 0   
        temp_linearity = 0     
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
                
        self.empty_space = temp_emptyspace
        self.enemy_count = temp_enemycount
        self.linearity = temp_linearity


