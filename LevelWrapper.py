

#Wrapper class for game levels. Containing all relevant information about them including their representations and attributes
class LevelWrapper:
    
    #Stores Level Name
    name = ''
    #Parent Generator
    generator_name= ''
    #Stores the character matrix representation of the level
    char_rep = None
    
    onehot_rep =  None

    def __init__(self, name, generator_name, char_rep):
        self.data = []
        self.name = name
        self.generator_name = generator_name
        self.char_rep = char_rep
    
    def print_self(self):
        print("Level Name = " + self.name)
        print("Level Char Rep: ")
        print(self.char_rep)

