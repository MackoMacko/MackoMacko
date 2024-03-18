#%%### --------- DEPENDENCIES AND IMPORTS --------- ###
import numpy as np
from scipy.spatial import *
import matplotlib.pyplot as plt
from scipy.ndimage import label
import time
# import easygui as eg
import glob
import pandas as pd
import re
from scipy.interpolate import make_interp_spline
from matplotlib.colors import Normalize
### ------------------------------------------- ###


#%%### --------- GRID LOAD PREPARATION --------- ###
info2 = """
loading grid from .png file into numpy array with only 0 (vacuum) and 1 (metal)
"""
def grid_preparation(filepath, treshold):
    #loading grid
    grid = plt.imread(filepath)
    
    x,y = np.where(grid < treshold)
    grid = grid[min(x):max(x), min(y):max(y)]
    
    #normalizacja
    grid = (grid < treshold)*1.0 
    
    return grid


#%%### --------- PHYSICAL MODEL FOR CALCULATIONS --------- ###

def model_3D(d: np.array = np.array([]),
             px_scale: float = 1,
             diffusion: float = 100) -> np.array:
    
    #applied model for the probability calculation
    P = np.exp(-px_scale*d/diffusion)
        
    return P



def heat_map(grid_in: np.array, 
             func, 
             diff: float, 
             unit_size: float,
             line_width_px: int = 1,
             plotting: int = 0) -> float:

    """
    grid - np.array filled with 0 and 1 to express the presence or absence of metal
    
    func - model of the probability 
    
    diff - diffusion coefficient (or mean free path in the material), expressed in nm
    
    unit_size - calculation area in nm, later on the grid is scaled to match the given number
     
    line_width_px - line width of the metal stated in px rather than nm
    
    plotting - plots or not the results
    
    RETURN: efficiency defined as a probabilty of collection a generated charge
    """
    
    grid = grid_in[:]    
    
    px_scale = unit_size / max(grid.shape)
   
    #usuwanie niepolaczonych elementow przy pomocy modulu scipy.ndimage
    lw, _ = label(grid) #przypisanie numeru do wszystkich klastrow
    grid = (lw == 1)*1.0 #zakladamy, ze ramka ma numer 1 klastra, co jest prawda, chyba, ze wyjdziemy poza ten warunek, czyli modelowanie bez ramki kwadratowej, jednak na razie powinien on zostac
    
    metal = np.sum(grid[grid==1.0]) #suma metalu w badanej powierzchni
    
    #teraz grid ma 1 tylko na polaczonych z ramka punktach
    nonzero = grid.nonzero() #array z indeksami metalu
    nonzero = np.asarray([nonzero[0].T, nonzero[1].T]).T # transpozycja do postaci par (x,y)
    
    zero = np.nonzero(grid == 0) #array z indeksami pustki 
    zero = np.asarray([zero[0].T, zero[1].T]).T # transpozycja do postaci par (x,y) tak j.w.
    tree = cKDTree(nonzero) # drzewo przypisujace najblizyszhc sasiadow z biblioteki scipy.spatial
    
    d, _ = tree.query(zero,1) #wlasciwy moment wywolania algorytmu poszukujacego jednego najblizszego sasiada

 
       
    grid[zero.T[0], zero.T[1]] = func(d, px_scale, diff) #przypisanie mapy z wartosciami otrzymanymi z alg. 
    
    scale_labs_x = unit_size/grid.shape[0]
    scale_labs_y = unit_size/grid.shape[1]
        
    if plotting:
        
        cmap = plt.colormaps['hot']
        ticks_step = 1000 # w nm
       
        scale_labs = max(grid.shape)/unit_size
        
        ticks_labels = np.arange(0,unit_size+1,1000)
        
        ticks_x = np.arange(0, grid.shape[1]+1, ticks_step*scale_labs)
        ticks_y = np.arange(0, grid.shape[0]+1, ticks_step*scale_labs)
          
        fig, ax = plt.subplots(figsize = (9,6), dpi = 500)
        #ax.imshow(grid)
        
        norm = Normalize(vmin=0, vmax=1)
        
        im = ax.imshow(grid, cmap=cmap, norm = norm, interpolation = 'bicubic')
        
        cbar = fig.colorbar(im, ax=ax)
        
        ax.set_xlim([0,grid.shape[1]])
        ax.set_ylim([0,grid.shape[0]])
        
          
        ax.set_xticks(ticks = ticks_x,
                 labels = ticks_labels[0:len(ticks_x)] )
        
        ax.set_yticks(ticks = ticks_y,
                 labels = ticks_labels[0:len(ticks_y)] )
        
        ax.annotate(f"diffusion: {diff} nm, unit: {unit_size} nm", xy=(0,0), xytext=(-10, -50),
            textcoords='offset points', ha='left', va='top')
    

        ax.set_xlabel("X [nm]")
        ax.set_ylabel("Y [nm]") 

        eff = (np.sum(grid) - metal)/grid.size
        # print('grid_sum: ',np.sum(grid))
        # print('metal: ', metal)
        
    return eff





