from LevelWrapper import LevelWrapper

class MarioLevel(LevelWrapper):

    #BC Storage
    empty_space = None
    enemy_count = None 

    def __init__(self, name, generator_name, char_rep):
        super(MarioLevel, self).__init__(name, generator_name, char_rep)
        #self.calc_behavioral_features(char_rep)

    def calc_behavioral_features(self, char_rep):
        temp_emptyspace = 0        
        temp_enemycount = 0
        for x in range(0, len(char_rep)):
            for y in range(0,len(char_rep[0])):
                if char_rep[x][y]=='-':
                    temp_emptyspace+=1
                elif char_rep[x][y] in ["y","Y","E","g","G","k","K","r"]:
                    temp_enemycount+=1
        self.empty_space = temp_emptyspace
        self.enemy_count = temp_enemycount
        #print("Empty space for level: " + self.name + ', generator:  ' + self.generator_name +  ', '  + str(self.empty_space))


