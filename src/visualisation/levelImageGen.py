from tkinter import X
from PIL import Image
import numpy as np

from src.config.enumsAndConfig import *

sprites_root = 'assets/'

mario_tilemap_path = sprites_root +'mario_mapsheet.png'

mario_enemytile_path = sprites_root + 'mario_enemysheet.png'

#Taken from https://opengameart.org/content/sokoban-100-tiles
boxoban_tilemap_path = sprites_root +'sokoban_spritesheet.png'


mario_tileSize = 16

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
        image_pixel_matrix = np.zeros((lvl_height*mario_tileSize, lvl_width*mario_tileSize, 4), dtype = np.uint8)
    elif (game == Game.Boxoban):
        image_pixel_matrix = np.zeros((lvl_height*boxoban_tilesize, lvl_width*boxoban_tilesize, 4), dtype = np.uint8)

    tile_map = None
    enemy_map = None
    if (game == Game.Mario):
        tile_map = np.asarray(Image.open(mario_tilemap_path))
        enemy_map = np.asarray(Image.open(mario_enemytile_path))
    elif(game == Game.Boxoban):
        tile_map = np.asarray(Image.open(boxoban_tilemap_path))
    for y in range(0, len(level_charrep)):
        for x in range(0,len(level_charrep[0])):
            if (game == Game.Boxoban):
                targety = [y*boxoban_tilesize, ((y+1) * boxoban_tilesize)]
                targetx = [x*boxoban_tilesize, ((x+1) * boxoban_tilesize)]
                maploc = boxoban_tiletypes_map[level_charrep[y][x]]
                image_pixel_matrix[targety[0] : targety[1], targetx[0] : targetx[1]] = tile_map[maploc[2]:maploc[3], maploc[0]: maploc[1]]
            elif (game == Game.Mario):
                targety = [y*mario_tileSize, ((y+1) * mario_tileSize)]
                targetx = [x*mario_tileSize, ((x+1) * mario_tileSize)]
                #Logic for adding enemy sprites
                if level_charrep[y][x] in mario_enemy_list:
                    enemymaploc = mario_enemytypes_map[level_charrep[y][x]]
                    #print("Adding enemy of type: " + level_charrep[y][x] + " with map loc: " + str(enemymaploc)) 
                    #Enemy tiles are 16 x 32, the height of two tiles, so the logic reflects this
                    maptargetx = [enemymaploc[0]*mario_tileSize, ((enemymaploc[0]+1) * mario_tileSize)]
                    maptargety = [enemymaploc[1]*(mario_tileSize*2), ((enemymaploc[1]+1) *(mario_tileSize*2))]
                    image_pixel_matrix[targety[0]-mario_tileSize : targety[1], targetx[0] : targetx[1]] = enemy_map[maptargety[0]:maptargety[1], maptargetx[0]: maptargetx[1]]
                else:
                    maploc = mario_tiletypes_map[level_charrep[y][x]]
                    maptargetx = [maploc[0]*mario_tileSize, ((maploc[0]+1) * mario_tileSize)]
                    maptargety = [maploc[1]*mario_tileSize, ((maploc[1]+1) * mario_tileSize)]
                    image_pixel_matrix[targety[0] : targety[1], targetx[0] : targetx[1]] = tile_map[maptargety[0]:maptargety[1], maptargetx[0]: maptargetx[1]]
    if game==Game.Boxoban or game == Game.Mario: 
        img = Image.fromarray(image_pixel_matrix, 'RGBA')
        img.save(filepath)
    
        


        
