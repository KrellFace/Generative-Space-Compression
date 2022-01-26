from LevelWrapper import LevelWrapper

class LoderunnerLevel(LevelWrapper):

    #BC Storage
    box_count = None 
    empty_space = None

    def __init__(self, name, generator_name, char_rep):
        super(LoderunnerLevel, self).__init__(name, generator_name, char_rep)
        #self.calc_behavioral_features(char_rep)

    def calc_behavioral_features(self, char_rep):
        temp_emptyspace = 0        
        for x in range(0, len(char_rep)):
            for y in range(0,len(char_rep[0])):
                if char_rep[x][y]=='.':
                    temp_emptyspace+=1
        self.empty_space = temp_emptyspace


