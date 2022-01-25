from LevelWrapper import LevelWrapper

class LoderunnerLevel(LevelWrapper):

    #BC Storage
    box_count = None 
    empty_space = None

    def __init__(self, name, generator_name, char_rep):
        super(LoderunnerLevel, self).__init__(name, generator_name, char_rep)
        #self.calc_behavioral_features(char_rep)

    def calc_behavioral_features(self, char_rep):
        return 


