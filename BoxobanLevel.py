from LevelWrapper import LevelWrapper

class BoxobanLevel(LevelWrapper):

    #BC Storage
    box_count = None 
    empty_space = None
    contiguity = None

    def __init__(self, name, generator_name, char_rep):
        super(BoxobanLevel, self).__init__(name, generator_name, char_rep)
        #self.calc_behavioral_features(char_rep)

    def calc_behavioral_features(self, char_rep):
        temp_boxcount = 0
        temp_emptyspace = 0
        temp_contiguity = 0
        #print("Char rep dimensions: " )
        #print(len(char_rep))
        #print(len(char_rep[0]))
        for x in range(0, len(char_rep)):
            for y in range(0,len(char_rep[0])):
                if char_rep[x][y]=='$':
                    temp_boxcount+=1
                elif char_rep[x][y]==' ':
                    temp_emptyspace+=1
                #Calculate contiguity score:
                #x axis
                if char_rep[y][x] == '#'and x>0:
                    if char_rep[y][x-1]  == '#':
                        temp_contiguity+=1
                if char_rep[y][x] == '#' and x<len(char_rep[0])-1:
                    if char_rep[y][x+1]  == '#':
                        temp_contiguity+=1
                #y axis
                if char_rep[y][x] == '#'and y>0:
                    if char_rep[y-1][x]  == '#':
                        temp_contiguity+=1
                if char_rep[y][x] == '#' and y<len(char_rep[0])-1:
                    if char_rep[y+1][x]  == '#':
                        temp_contiguity+=1                
            
        self.box_count = temp_boxcount
        self.empty_space = temp_emptyspace
        self.contiguity = temp_contiguity

        #print('Level: ' + self.name + " box count: " + str(self.box_count) + ' & empty space: ' + str(self.empty_space))


