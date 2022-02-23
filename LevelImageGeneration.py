from tkinter import X
from PIL import Image
import numpy as np
import LevelWrapper
from EnumsAndConfig import *

boxoban_tilecolors_dict = {
    #Grey walls
    "#":[125, 125, 124], 
    #Blue Player Character
    "@":[0, 162, 255],
    #Brown boxes 
    "$":[204, 129, 0], 
    #Golden goals
    ".":[255, 218, 10], 
    #Pale grey empty space
    " ":[242, 242, 240]
}
empty_space_col = [242, 242, 240]
#Enemy red
enemy_col = [255, 10, 10]
#Dark brown solid
solid_col = [74, 39, 1]
#Pipe green
pipe_col = [26, 171, 0]
#Gold reward
reward_col = [255, 218, 10]

mario_tilecolors_dict_condensed = {
    #Empty space, including start and end tiles, and bullet bills
    "-":empty_space_col,"M":empty_space_col,"F":empty_space_col,"|":empty_space_col,"*":empty_space_col,"B":empty_space_col,"b":empty_space_col,
    #All enemies apart from bulletbills
    "y":enemy_col,"Y":enemy_col,"E":enemy_col,"g":enemy_col,"G":enemy_col,"k":enemy_col,"K":enemy_col,"r":enemy_col,
    #Solid blocks
    "X":solid_col,"#":solid_col,"%":solid_col,"D":solid_col,"S":solid_col,
    #Pipe blocks
    "t":pipe_col,"T":pipe_col,"<":pipe_col,">":pipe_col,"[":pipe_col,"]":pipe_col,
    #Reward blocks
    "?":reward_col,"@":reward_col,"Q":reward_col,"!":reward_col,"1":reward_col,"2":reward_col,"C":reward_col,"U":reward_col,"L":reward_col,"o":reward_col
}


pixelSize = 16

def generate_image(game, LevelWrapper, filepath):
    level_charrep = LevelWrapper.char_rep
    lvl_width = len(level_charrep[0])
    lvl_height = len(level_charrep)

    image_pixel_matrix = np.zeros((lvl_height*pixelSize, lvl_width*pixelSize, 3), dtype = np.uint8)

    for y in range(0, len(level_charrep)):
        for x in range(0,len(level_charrep[0])):
            targety = [y*pixelSize, ((y+1) * pixelSize)]
            targetx = [x*pixelSize, ((x+1) * pixelSize)]
            if (game == Game.Boxoban):
                image_pixel_matrix[targety[0] : targety[1], targetx[0] : targetx[1]] = boxoban_tilecolors_dict[level_charrep[y][x]]
            elif (game == Game.Mario):
                image_pixel_matrix[targety[0] : targety[1], targetx[0] : targetx[1]] = mario_tilecolors_dict_condensed[level_charrep[y][x]]
    img = Image.fromarray(image_pixel_matrix, 'RGB')
    img.save(filepath)
    
        


        
