from abc import abstractmethod
from pickle import NONE

#Wrapper class for game levels. Containing all relevant information about them including their representations and attributes
class LevelWrapper:
    
    #Stores Level Name
    name = ''
    #Parent Generator
    generator_name= ''
    
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

    KernelPCA1 = None
    KernalPCA2 = None 

    TSNE_PCA1 = None
    TSNE_PCA2 = None

    TSNE_MCA1 = None
    TSNE_MCA2 = None

    TSNE_SVD1 = None
    TSNE_SVD2 = None


    def __init__(self, name, generator_name, char_rep):
        self.data = []
        self.name = name
        self.generator_name = generator_name
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

