from tkinter import X
from PIL import Image
import numpy as np

import LevelWrapper
from EnumsAndConfig import *

mario_tilemap_path = 'mario_mapsheet.png'

mario_enemytile_path = 'mario_enemysheet.png'

#Taken from https://opengameart.org/content/sokoban-100-tiles
boxoban_tilemap_path = 'sokoban_spritesheet.png'


pixelSize = 16


"""
boxoban_tilecolors_dict = {
    #Grey walls
    "#":[125, 125, 124, 1], 
    #Blue Player Character
    "@":[0, 162, 255, 1],
    #Brown boxes 
    "$":[204, 129, 0, 1], 
    #Golden goals
    ".":[255, 218, 10, 1], 
    #Pale grey empty space
    " ":[242, 242, 240, 1]
}
"""

#Locations of tiles in form x1, x2, y1, y2
boxoban_tilesize = 60
boxoban_tiletypes_map = {
    #Red Walls
    "#":[450,510,2, 62], 
    #Player Character
    "@":[450, 510, 230,290],
    #Brown boxes 
    "$":[130, 190,64, 124], 
    #Goals
    ".":[194, 254,338, 398], 
    #Pale grey empty space
    " ":[192, 252, 530, 590]
}

#Store mario tile locations in tilemap
empty_space_loc = [2, 5]
base_enemy_loc = [4, 0]
base_solid_loc = [6,0]
base_pipe_loc = [4, 2]
reward_loc = [0,1]
brick_loc = [1,0]
ground_loc = [1,0]
pyramid_loc = [2,0]
top_pipe_left_loc = [2,2]
top_pipe_right_loc = [3,2]
pipe_left_loc = [4,2]
pipe_right_loc = [5,2]
small_pipe_top_loc = [4,6]
small_pipe_loc = [4,6]
coin_loc = [7,1]
jumpthrough_loc = [6,5]
jumpthrough_background_loc = [7,5]

#Locations of mario tiles in the tilemap in x, y coords per tile:
mario_tiletypes_map = {
    "-":empty_space_loc,
    "M":empty_space_loc,
    "F":empty_space_loc,
    "|":jumpthrough_background_loc,
    "*":empty_space_loc,
    "B":empty_space_loc,
    "b":empty_space_loc,
    #Solid blocks
    "X":ground_loc,
    "#":pyramid_loc,
    "%":jumpthrough_loc,
    "D":base_solid_loc,
    "S":brick_loc,
    #Pipe blocks
    "t":small_pipe_loc,
    "T":small_pipe_top_loc,
    "<":top_pipe_left_loc,
    ">":top_pipe_right_loc,
    "[":pipe_left_loc,
    "]":pipe_right_loc,
    #Reward blocks
    "?":reward_loc,"@":reward_loc,"Q":reward_loc,"!":reward_loc,"1":reward_loc,"2":reward_loc,
    "C":brick_loc,
    "U":brick_loc,
    "L":reward_loc,
    "o":coin_loc
}

mario_enemy_list = ["y","Y","E","g","G","k","K","r"]

#Store Enemy types in the enemy sprite sheet
spiky_loc = [0, 3]
goomba_loc = [0,2]
greenkoopa_loc = [0,1]
redkoopa_loc = [0,0]

mario_enemytypes_map = {
        "y":spiky_loc,
        "Y":spiky_loc,
        "E":goomba_loc,
        "g":goomba_loc,
        "G":goomba_loc,
        "k":greenkoopa_loc,
        "K":greenkoopa_loc,
        "r":redkoopa_loc,
}

def generate_image(game, LevelWrapper, filepath):
    level_charrep = LevelWrapper.char_rep
    lvl_width = len(level_charrep[0])
    lvl_height = len(level_charrep)

    image_pixel_matrix = None
    if (game == Game.Mario):
        image_pixel_matrix = np.zeros((lvl_height*pixelSize, lvl_width*pixelSize, 4), dtype = np.uint8)
    elif (game == Game.Boxoban):
        image_pixel_matrix = np.zeros((lvl_height*boxoban_tilesize, lvl_width*boxoban_tilesize, 4), dtype = np.uint8)

    tile_map = None
    enemy_map = None
    if (game == Game.Mario):
        mapimg = Image.open(mario_tilemap_path)
        tile_map = np.asarray(mapimg)
        enemyimg = Image.open(mario_enemytile_path)
        enemy_map = np.asarray(enemyimg)
    elif(game == Game.Boxoban):
        mapimg = Image.open(boxoban_tilemap_path)
        tile_map = np.asarray(mapimg)
    for y in range(0, len(level_charrep)):
        for x in range(0,len(level_charrep[0])):
            if (game == Game.Boxoban):
                targety = [y*boxoban_tilesize, ((y+1) * boxoban_tilesize)]
                targetx = [x*boxoban_tilesize, ((x+1) * boxoban_tilesize)]
                maploc = boxoban_tiletypes_map[level_charrep[y][x]]
                image_pixel_matrix[targety[0] : targety[1], targetx[0] : targetx[1]] = tile_map[maploc[2]:maploc[3], maploc[0]: maploc[1]]
            elif (game == Game.Mario):
                targety = [y*pixelSize, ((y+1) * pixelSize)]
                targetx = [x*pixelSize, ((x+1) * pixelSize)]
                #Logic for adding enemy sprites
                if level_charrep[y][x] in mario_enemy_list:
                    enemymaploc = mario_enemytypes_map[level_charrep[y][x]]
                    #print("Adding enemy of type: " + level_charrep[y][x] + " with map loc: " + str(enemymaploc)) 
                    #Enemy tiles are 16 x 32, the height of two tiles, so the logic reflects this
                    maptargetx = [enemymaploc[0]*pixelSize, ((enemymaploc[0]+1) * pixelSize)]
                    maptargety = [enemymaploc[1]*(pixelSize*2), ((enemymaploc[1]+1) *(pixelSize*2))]
                    """
                    #Do not set pixels which are white (I need to work out how to do this with alpha)
                    for y2 in range(maptargety[0], maptargety[1]):
                        for x2 in range(maptargetx[0], maptargetx[1]):
                            #Only set pixel if pixel in enemy sprite sheet is not white
                            if not (enemy_map[y2, x2] == [255, 255, 255]):
                                print("ImgPixelMatrix dimensions : " + str(image_pixel_matrix.shape) + " enemy map shape : " + str(enemy_map.shape))
                                print("Img Matrix target shape : " + str(image_pixel_matrix[(targety[0]-(pixelSize*2))+y2, targetx[0]+x2].shape) + " enemy target shape: " + str(enemy_map[y2,x2].shape))
                                image_pixel_matrix[(targety[0]-(pixelSize*2))+y2, targetx[0]+x2] = enemy_map[y2,x2]
                    """
                    image_pixel_matrix[targety[0]-pixelSize : targety[1], targetx[0] : targetx[1]] = enemy_map[maptargety[0]:maptargety[1], maptargetx[0]: maptargetx[1]]
                else:
                    maploc = mario_tiletypes_map[level_charrep[y][x]]
                    maptargetx = [maploc[0]*pixelSize, ((maploc[0]+1) * pixelSize)]
                    maptargety = [maploc[1]*pixelSize, ((maploc[1]+1) * pixelSize)]
                    image_pixel_matrix[targety[0] : targety[1], targetx[0] : targetx[1]] = tile_map[maptargety[0]:maptargety[1], maptargetx[0]: maptargetx[1]]
    if game==Game.Boxoban or game == Game.Mario: 
        img = Image.fromarray(image_pixel_matrix, 'RGBA')
        img.save(filepath)
    
        


        
