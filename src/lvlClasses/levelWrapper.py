from abc import abstractmethod
from pickle import NONE

#Wrapper class for game levels. Containing all relevant information about them including their representations and attributes
class LevelWrapper:
    
    #Stores Level Name
    name = ''
    #Parent Generator
    generator_name= ''

    #Source File
    source_file = ''
    
    #level_type = ''

    #Stores the character matrix representation of the level
    char_rep = None
    
    onehot_rep =  None

    PC1Val = None
    PC2Val = None 

    MCA1Val = None
    MCA2Val = None

    SVD1Val = None
    SVD2Val = None

    TSNEVal1 = None
    TSNEVal2 = None 

    KPCAPolyVal1 = None
    KPCAPolyVal2 = None 

    KPCARbfVal1 = None
    KPCARbfVal2 = None 

    KPCASigmoidVal1 = None
    KPCASigmoidVal2 = None 

    KPCACosineVal1 = None
    KPCACosineVal2 = None 

    def __init__(self, name, generator_name, source_file, char_rep):
        self.data = []
        self.name = name
        self.generator_name = generator_name
        self.source_file = source_file
        self.char_rep = char_rep

        self.calc_behavioral_features(char_rep)
    
    def print_self(self):
        print("Level Name = " + self.name)
        print("Level Char Rep: ")
        print(self.char_rep)

    def print_features(self):
        print("PCA Vals: " + str(self.PC1Val) + "," + str(self.PC2Val))
        print("MCA Vals: " + str(self.MCA1Val) + "," + str(self.MCA2Val))
        print("TSNE Vals: " + str(self.TSNEVal1) +" ," + str(self.TSNEVal2))
    
    @abstractmethod
    def calc_behavioral_features(self, char_rep):
        pass

