import numpy as np
from scipy.spatial import *
import itertools
import matplotlib.pyplot as plt
import cmath
import random


def simul_init(L,N):
    """parameters setting:
        L - box size,
        N - birds #
        """

    step = L/(np.sqrt(N))
    birds_starting_params = [[i,j,2*np.pi*random.random()] for i in np.arange(0,L,step) 
                              for j in np.arange(0,L,step)]
    return np.asarray(birds_starting_params)

def bird_of_prey(position_angle):
    if position_angle:
        bird_of_prey_start = np.asarray(position_angle,dtype = 'float64')
        return bird_of_prey_start
    else: return [None]


def update_position(birds, prey, L = 10, N = 800,v_0 = 2, a = 0.15,r_b = 3,dt = 0.02):
    """taking Nx3 list where:
        birds[i][0] - X pos.
        birds[i][1] - Y pos.
        birds[i][2] - angle"""
    
    #creating a KD-tree with nearest image periodic condition: LxL 
    birds_tree = cKDTree(birds[:,0:2], boxsize = [L,L])
    
    #updating nearest neighboors of the every bird:
    nearest_birds_indexes = birds_tree.query(birds[:,0:2],5)[1]
    
    #if we decide to attach a bird of prey:
    if len(prey) == 3:
        #looking for the all birds inside the r_b radius:
        bird_of_prey_close = birds_tree.query_ball_point(prey[0:2], r_b)
        #looking for the nearest bird:
        bird_of_prey_chase = birds_tree.query(prey[0:2],1)[1]
        #assiging chase direction to the nearest bird:
        prey[2] = birds[bird_of_prey_chase,2] + 2*np.pi*a*np.random.random(1)
    
    
    angle_noise_term = (2*np.pi*a)*np.random.random(len(nearest_birds_indexes))
    mean_angles = np.mean(birds[:,2][nearest_birds_indexes],1) + angle_noise_term
    birds[:,2] = mean_angles
    
    
    if len(prey) == 3:
        birds[:,2][bird_of_prey_close] =  prey[2] + 2*np.pi*a*np.random.random(len(bird_of_prey_close))
        prey[0:2] += v_0*dt*np.array([np.cos(prey[2]), np.sin(prey[2])]).T
        
    birds[:,0:2] += v_0*dt*np.array([np.cos(birds[:,2]), np.sin(birds[:,2])]).T
  
    
    # periodic boundary conditions:
    birds[:,0:2][birds[:,0:2] > L] -= L
    birds[:,0:2][birds[:,0:2] < 0] += L

def simulation_main(prey = None, L = 10, N = 800,v_0 = 2, a = 0.025,r_b = 3,dt = 0.02, time_steps = 1000):
    """parameters setting:
    L - box size,
    N - birds #
    r_b - radius of the prey sight
    v_0 - bird velocity
    a - noise term
    dt - time_step
    time_steps - simulation length
    """
 
        
    birds = simul_init(L,N)
    prey = bird_of_prey(prey)
    for i in range(time_steps):
        update_position(birds,prey,L,N,v_0,a,r_b,dt)
        if i % 10 == 0:
            print(round(100*i/time_steps,1))
            plt.xlim([-0.05*L,1.05*L])
            plt.ylim([-0.05*L,1.05*L])
            plt.quiver(birds[:,0],birds[:,1],
                       np.cos(birds[:,2]),np.sin(birds[:,2]), 
                       birds[:,2])
            plt.show()
    return birds




#%%
simulation_main(prey = [10,10,0],L=32,N=8000,v_0=2,a=0.1,r_b=4,dt= 0.04,time_steps=1000)














