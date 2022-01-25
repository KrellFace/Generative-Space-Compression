from LevelWrapper import LevelWrapper

class BoxobanLevel(LevelWrapper):

    #BC Storage
    box_count = None 
    empty_space = None

    def __init__(self, name, generator_name, char_rep):
        super(BoxobanLevel, self).__init__(name, generator_name, char_rep)
        #self.calc_behavioral_features(char_rep)

    def calc_behavioral_features(self, char_rep):
        temp_boxcount = 0
        temp_emptyspace = 0
        #print("Char rep dimensions: " )
        #print(len(char_rep))
        #print(len(char_rep[0]))
        for x in range(0, len(char_rep)):
            for y in range(0,len(char_rep[0])):
                if char_rep[x][y]=='$':
                    temp_boxcount+=1
                elif char_rep[x][y]==' ':
                    temp_emptyspace+=1
        self.box_count = temp_boxcount
        self.empty_space = temp_emptyspace

        #print('Level: ' + self.name + " box count: " + str(self.box_count) + ' & empty space: ' + str(self.empty_space))


