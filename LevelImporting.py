import os
import random
from EnumsAndConfig import *
from WindowGrabbing import *
from HelperMethods import *

#Get a 2D character matrix from a level file 
def char_matrix_from_file(path):
    with open(path) as f:
        charlist = f.read()
        width = 0
        width_calculated = False
        height = 0
        charlist_newlinesremoved = list()

        for char in charlist:
            if char == '\n':
                width_calculated = True
                height+=1
            else:
                charlist_newlinesremoved.append(char)
            if not width_calculated:
                width+=1
        output_matrix = np.reshape(charlist_newlinesremoved,(height, width), order = 'C')
        #print("Level processed: "  + path)
        return output_matrix

#Custom function for boxoban files as they are stored with multiple in each file
#Returns a dictionary of form Key: 'Holding filename : Level Name'  Value: LevelWrapper
def get_boxoban_leveldict_from_file(file_path, folder_name, counttoretrieve):
    level_reps = dict()
    file_name = os.path.basename(file_path)
    line_counter = 0
    buffer = list()
    temp_levelname = ""
    counter = 0
    with open(file_path) as file:
        charlist = file.read()
        for char in charlist:
            #while counter < counttoretrieve:
            if (char == '\n'):
                #Check if we are on a level name line
                if (line_counter%12 == 0):
                    temp_levelname = temp_levelname.join(buffer)
                    buffer.clear()
                #Check if we are at the end of a level rep. If we are, add it to our dictionary
                elif ((line_counter+1)%12 == 0):
                    char_matrix = np.reshape(buffer,(boxoban_height, boxoban_width), order = 'C')
                    #level_reps[int(temp_levelname)] = char_matrix
                    #level_reps[file_dict_key +':'+ temp_levelname] = LevelWrapper(temp_levelname, file_dict_key, char_matrix)
                    new_level = BoxobanLevel(temp_levelname, get_n_last_subparts_path(file_path, 2), char_matrix)
                    #new_level.calc_behavioral_features()
                    level_reps[folder_name +':'+ file_name+":"+temp_levelname] = new_level
                    temp_levelname = ""
                    buffer.clear()
                    counter+=1
                    #print("Level added. Level counter: " + str(counter))
                    if counter >= counttoretrieve:
                        #print("Target " + str(counter) + " reached. Returning levels")
                        return level_reps
                
                line_counter+=1
            #Only append numeric characters if we are on the level name line as level names are numbers
            elif (line_counter%12 == 0):
                if (char.isnumeric()):
                    buffer.append(char)
            #If its not a level name line or a newline character, add to buffer
            else:
                buffer.append(char)
            
    return level_reps

#Def get an dict of LevelWrappers from a folder in form (Key: 'Folder + File Name, Value: LevelWrapper)
def get_leveldict_from_folder(path, folder_key, game):
    file_names = get_filenames_from_folder(path)
    window_height, window_width = get_level_heightandwidth_for_game(game)
    #folder_name = os.path.basename(os.path.normpath(path))
    level_reps = dict()

    for level in file_names:
        level_name = os.path.basename(level)
        char_rep = char_matrix_from_file(level)
        char_rep_window = take_window_from_bottomright(char_rep, window_width, window_height)
        level_reps[folder_key +':'+ level_name] = generate_levelwrapper_for_game(game, level_name, folder_key, char_rep_window)

    return level_reps

#Get a randomly selected set of levels of size filecount from a folder
#Needs refactoring as it mirrors a lot of get_levledict_from_folder
def get_randnum_levelwrappers_folder(path, folder_key, game, count):
    file_names = get_filenames_from_folder(path)
    window_height, window_width = get_level_heightandwidth_for_game(game)
    #folder_name = os.path.basename(os.path.normpath(path))
    level_reps = dict()
    counter = 0
    while counter < count:
        #print("Counter at: " + str(counter) + " picking from filename list length:  " + str(len(file_names)))
        #Pick random file and then remove it from the list
        level = random.choice(file_names)
        file_names.remove(level)
        level_name = os.path.basename(level)
        char_rep = char_matrix_from_file(level)
        char_rep_window = take_window_from_bottomright(char_rep, window_width, window_height)
        level_reps[folder_key +':'+ level_name] = generate_levelwrapper_for_game(game, level_name, folder_key, char_rep_window)
        counter+=1

    return level_reps


#Get a combined levelwrapper dictionary from a folder dictionary
def get_leveldicts_from_folder_set(game):
    level_dict = dict()
    game_info = get_folder_and_tiletypedict_for_game(game)
    folder_dict = game_info['Folder_Dict']

    for folder in folder_dict:
        #Get all one for for specific folder
        temp_dict = get_leveldict_from_folder(folder_dict[folder], folder, game)
        level_dict = level_dict|temp_dict
    return level_dict

def get_randnum_levelwrappers(game, maxlvlsevaled):
    level_dict = dict()
    game_info = get_folder_and_tiletypedict_for_game(game)
    folder_dict = game_info['Folder_Dict']
    #Calculate count of levels to get per folder
    count = math.floor((maxlvlsevaled/len(folder_dict)))

    for folder in folder_dict:
        #Get all one for for specific folder
        temp_dict = get_randnum_levelwrappers_folder(folder_dict[folder], folder, game, count)
        level_dict = level_dict|temp_dict
    return level_dict   

def get_randnum_levelwrappers_boxoban(folders_dict, maxlvlsevaled):
    levelwrapper_dict = dict()
    counter = 0
    #Calculate count of levels to get per folder
    folderlevelcount = math.floor((maxlvlsevaled/len(folders_dict)))
    for folder in folders_dict:
        #List all files in folder
        file_list = get_filenames_from_folder(folders_dict[folder])
        #Calculate level count to get per file. A random amount between the min required per file and 5 times this amount
        minrequired = math.ceil((folderlevelcount/len(file_list)))
        while counter < folderlevelcount:
            randfile = random.choice(file_list)
            filelevelcount = None
            if minrequired < (folderlevelcount-counter):
                filelevelcount = random.randint(minrequired, (folderlevelcount-counter))
            #Else, just take the remainder needed from the file
            else:
                filelevelcount = (folderlevelcount-counter)
            #print("Retrieving " + str(filelevelcount) + " levels from file. Target count for folder: " + str(folderlevelcount) + " and curr count: " + str(counter))
            temp_dict = get_boxoban_leveldict_from_file(randfile, folder, filelevelcount)
            file_list.remove(randfile)
            levelwrapper_dict = levelwrapper_dict|temp_dict
            counter+=filelevelcount
        counter = 0
    return levelwrapper_dict
