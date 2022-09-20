from src.lvlClasses.levelWrapper import LevelWrapper
from src.config.enumsAndConfig import BCType

class BoxobanLevel(LevelWrapper):

    #BC Storage
    #box_count = None 
    #empty_space = None
    #contiguity = None


    def __init__(self, name, generator_name, source_file, char_rep):
        super(BoxobanLevel, self).__init__(name, generator_name,source_file,char_rep)
        #self.calc_behavioral_features(char_rep)

    def calc_behavioral_features(self, char_rep):
        temp_boxcount = 0
        temp_emptyspace = 0
        temp_contiguity = 0
        temp_corriscore = 0
        temp_solidcount = 0
        for y in range(0, len(char_rep)):
            for x in range(0,len(char_rep[0])):
                if char_rep[y][x]=='$':
                    temp_boxcount+=1
                elif char_rep[y][x]==' ':
                    temp_emptyspace+=1
                elif char_rep[y][x] =='#':
                    temp_solidcount+=1
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
                if char_rep[y][x] == '#' and y<len(char_rep)-1:
                    if char_rep[y+1][x]  == '#':
                        temp_contiguity+=1

                #Calculate CorridorScore
                if  char_rep[y][x]==' ' and y>0 and y<len(char_rep)-1 and x>0 and x<len(char_rep[0])-1:
                    if char_rep[y-1][x]  == '#' and char_rep[y+1][x]  == '#' and char_rep[y][x-1]  != '#' and char_rep[y][x+1]  != '#':
                        temp_corriscore +=1
                    elif char_rep[y-1][x]  != '#' and char_rep[y+1][x]  != '#' and char_rep[y][x-1]  == '#' and char_rep[y][x+1]  == '#':
                        temp_corriscore +=1
            
        #self.box_count = temp_boxcount
        #self.empty_space = temp_emptyspace
        #self.contiguity = temp_contiguity
        self.bc_vals[BCType.EmptySpace] =  temp_emptyspace
        self.bc_vals[BCType.Contiguity] =  temp_contiguity
        self.bc_vals[BCType.AdjustedContiguity] =  (temp_contiguity/temp_solidcount)
        self.bc_vals[BCType.CorriCount] =  temp_corriscore


