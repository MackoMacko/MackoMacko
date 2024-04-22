#%% imports needed

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import *

#%% a chemical particle class - describing the properties of one molecule


class Particle:
    
    #constructor of the class
    def __init__(self, 
                 compound: str, 
                 position: np.array, 
                 velocity: np.array, 
                 mass: float,
                 hindrance: float,
                 chemical_properties: dict):
        
         
        self.compound = compound
        self.position = position
        self.velocity = velocity
        self.mass = mass
        self.hindrance = hindrance
        self.adsorption_state = 0
        self.particle_parameters = {
            'structure': [],
            'prop1': [],
            'prop2': [],
            'prop3': []
            }
           
        
    #setters
    def set_compound(self, new_compound: str):
        self.compound = new_compound
    
    def set_position(self, new_position: np.array):
        self.position = new_position
        
    def set_velocity(self, new_velocity: np.array):
        self.velocity = new_velocity
        
    def set_mass(self, new_mass: float):
        self.mass = new_mass
        
    def set_hindrance(self, new_hindrance: float):
        self.hindrance = new_hindrance
        
    def set_adsorbtion_state(self, new_ads_state: int):
        self.adsorption_state = new_ads_state
    
    def set_particle_parameters(self, new_chemical_properties: dict):
        new_prop = new_chemical_properties.keys()
        for single_key in new_prop:
            self.particle_parameters[single_key] = new_chemical_properties[single_key]
            
        
    #getters
    def get_compound(self) -> str: 
        return self.compound

    def get_position(self) -> np.array:
        return self.position

    def get_velocity(self) -> np.array:
        return self.velocity

    def get_mass(self) -> float:
        return self.mass

    def get_hindrance(self) -> float:
        return self.hindrance

    def get_adsorption_state(self) -> int:
        return self.adsorption_state

    def get_particle_parameters(self) -> dict:
        return self.chemical_properties
    
    def get_particle_parameters_single(self, key):
        return self.chemical_properties[key]
    
    
    #info
    def info(self):
        string = f"""Compound: {self.compound},
Position and speed: {np.round(self.position,2)}, {np.round(self.velocity,2)},
Mass: {self.mass},
Hindrance: {self.hindrance},
Adsorbed:   {self.adsorbtion_state},
Chemical properties: {str(val[key]) for key in self.particle_parameters}
                """
        print(string)
                
                
#%%   
  
class Enviroment:
    
    def __init__(self,
        size: np.array, 
        temperature: float,
        pressure: float,
        surface_type: str, 
        trench_potential: np.array):
   
         
        self.size = size
        self.temperature = temperature
        self.pressure = pressure
        self.surface_type = None
        self.trench_potential = None
        self.boxes = []
        
        if surface_type:
            self.surface_type = Enviroment.set_surface_type(self,surface_type)
            
        if len(trench_potential):
            self.trench_potential = Enviroment.set_trench_potential(self,trench_potential)
        
        
        
    #setters                   
    def set_size(self, new_size: np.array):
        self.size = new_size
    
    def set_temperature(self, new_temperature: float):
        self.temperature = new_temperature
        
    def set_presussure(self, new_velocity: np.array):
        self.velocity = new_velocity
        
    def set_surface_type(self, surface_type):
        
        surf_silicon = {'name': 'Silicon',
                        'functional groups': 'OH',
                        'sticking potential': 0,
                        'group density': 0
                        }

        surf_glass = {'name': 'glass',
                        'functional groups': 'OH',
                        'sticking potential': 0,
                        'group density': 0
                        }
        
        surf_ZnO = {'name': 'ZnO',
                    'functional groups': 'OH',
                    'sticking potential': 0,
                    'group density': 0
                    }
        
        surf_types = [surf_silicon, surf_glass, surf_ZnO]
        
        self.surface_type = [i['name'] for i in surf_types if i['name'] == surface_type][0]
        print([i['name'] for i in surf_types if i['name'] == surface_type][0])
        
        ### to be implemented
        
    def set_trench_potential(self, trench_potential):
        trench_x = trench_potential[0]
        trench_end_x = trench_potential[1]
        trench_y = trench_potential[2]
        trench_depth = trench_potential[3]
        #to be implemented
        
    def set_boxes(self, boxes):
        self.boxes = boxes
     
    def append_box(self, box):
        self.boxes.append(box)
        
    #getters of basic properties             
    def get_size(self):
        return self.size
    
    def get_temperature(self):
        return self.temperature
        
    def get_presussure(self):
        return self.velocity
        
    def get_surface_type(self):
        return self.string
        
    def get_trench_potential(self):
        return self.trench_potential
    
    def get_boxes(self):
        return self.boxes
    
    def collisions_to_boundary(self):
        xxx = 1
        return xxx
    
    
    
    
    
    
    

#%%
        
class Simulation:
        
    def __init__(self, 
            particles: list = [], 
            time: float = 0,
            enviroment: Enviroment = None, 
           
           ):
        
            self.particles = particles
            self.part_number = len(self.particles)
            self.enviroment = enviroment
            self.model_parameters = {
                'r_minimum_collision': 0.6,
                'LJ_potential': 0.0005,
                'time_step': 0.001,
                'cut_off_radius': 5
                }
            
            #to do:
                # - parameters of the model getters and setters, variable stored in dictionary
                # - model description
                # - adsorption handling
                # - temperature and pressure setting
                
            
    #setters       
    def set_particles(self, particles):
        self.particles = particles
        
    def add_particles(self, particle):
        self.particles.append(particle)
        
    def set_enviroment(self, enviroment):
        self.enviroment = enviroment
    
    def set_particles_position(self,positions):
        len_part = self.part_number
        len_pos = len(positions)
        size = self.enviroment.size
        
        if len_part > len_pos:
            pass
            # for i in range(len_part - len_pos):
            #     print('len of particles: ', len_part)
            #     print('len of positions: ', len(positions))
            #     positions.append(np.random.random(3)*size)
        
        print('len of particles: ', len_part)
        print('len of positions: ', len(positions))
        
        for ind, particle in enumerate(particles):
            
            self.particles[ind].position = positions[ind] 
            
        
    def set_particles_velocity(self,velocities):
        for ind, particle in enumerate(self.particles):
            self.particles[ind].velocity = velocities[ind]
            
    def set_particles_adsorption(self, states):
        for ind, particle in enumerate(self.particles):
            self.particles[ind].adsorbtion_state = states[ind]
    
    def set_model_parameters(self, params):
        for par in params.items():
            self.model_parameters[par[0]] = par[1]
            
        
    #getters
    
    def get_model_parameters(self):
        return self.model_parameters
    
    def get_particles(self):
        return self.particles
    
    def get_particles_position(self):
        loc_xyz = [particle.position for particle in self.particles]
        return loc_xyz
    
    def get_particles_velocity(self):
        velocity_xyz = [particle.velocity for particle in self.particles]
        return velocity_xyz
    
    def get_particles_adsorbed(self):
        adsorbed_state = [particle.adsorption_state for particle in self.particles]
        return adsorbed_state 
    
    def get_particles_chem_properties(self):
        pass

    # caching the simulation data for speed
    
    def position_velocity_adsorb_to_dict(self):
        loc = Simulation.get_particles_position(self)
        velocity = Simulation.get_particles_velocity(self)
        chem = Simulation.get_particles_chem_properties(self)
        return {'loc': loc, 'velocity': velocity, 'chem': chem}
        
    def set_position_velocity_adsorb_dict(self, loc_velo_chem_dict):
        loc_velo_chem_dict['loc'] = loc_velo_chem_dict['loc']
        loc_velo_chem_dict['velocity'] = loc_velo_chem_dict['velocity']
        loc_velo_chem_dict['chem'] = loc_velo_chem_dict['chem']
        return loc_velo_chem_dict
    
        
    # init the simulation
    def assign_init_location(self, init_mode):
        #fix the number of particles divergence problem
        #generate the particles only in the available spots, based on the enviroment

        size = self.enviroment.size
        part_number = self.part_number
        particle_positions = []
        # particle_positions = np.random.random((part_number,3))
        # particle_positions *= size
        
        #correct this function        
        if init_mode == 'uniform':
            total_volume = np.prod(size)
            # Calculate the volume each particle should occupy
            particle_volume = total_volume / part_number
            # Calculate the number of particles that can fit along each dimension
            num_particles_per_dim = [int(round(dim / np.cbrt(particle_volume))) for dim in size]

            
            for x in np.linspace(1, size[0], num_particles_per_dim[0], endpoint=False):
                for y in np.linspace(1, size[1], num_particles_per_dim[1], endpoint=False):
                    for z in np.linspace(1, size[2], num_particles_per_dim[2], endpoint=False):
                        particle_positions.append(np.array([x, y, z]))
            
                        
            Simulation.set_particles_position(self,particle_positions)
            
        #correct this function            
        if init_mode == 'streamline':
            step = 0.1 #to do: play with hindrance
            y_0 = size[1]
            z_0 = size[2]/2
            
            num_particles_per_y = min (int(y_0/step), part_number)
            num_particles_per_x = int(part_number / num_particles_per_y) + 1 

            
            for x in np.linspace(0, num_particles_per_x*step, num_particles_per_x, endpoint = False):
                for y in np.linspace(0, y_0, num_particles_per_y, endpoint = False):
                        particle_positions.append(np.array([x, y, z_0]))
           
            print(len(particle_positions))
            Simulation.set_particles_position(self,particle_positions)
            
         
    def assign_init_velocities(self, init_mode, velo_avg, distortion):
        """
        3 modes to choose from:
        - gauss - ideal gas distribution
        - streamline - all particles go in one direction with a similar velocity (+some noise)
        - modulated streamline - the front of the particles go with modulated velocity in one direction
        """
        
        particle_velocities = []
        size = self.enviroment.size
        part_number = self.part_number
        dist_degree = velo_avg*distortion
        
        
        if init_mode == 'gauss':
            
           gauss_velo =  velo_avg*np.random.normal(0, dist_degree, size=3*size).reshape(3,-1).T
           Simulation.set_particles_velocity(self, gauss_velo)
                            
        if init_mode == 'streamline':
           drift_velo = np.zeros((part_number,3))
           drift_velo[:,0] = velo_avg
           streamline_velo = velo_avg*dist_degree*(np.random.random((part_number,3))-0.5) + drift_velo
           Simulation.set_particles_velocity(self, streamline_velo)
            
        if init_mode == 'modulated_streamline':
           pass
       
           
    def periodic_boundary_conditions_check(size, positions):
        
        if len(positions):
            positions[:,0] = positions[:,0] % size[0]
            positions[:,1] = positions[:,1] % size[1]
            positions[:,2] = positions[:,2] % size[2]

        return positions
    
    
        
    def box_periodic_boundary_conditions(size, positions, velocities, boxes):
        #find the set of particles close to walls
        #select them and assign the proper position and velocity change
        #pass the new velocities and positions back
        
        def boxes_to_conditions(positions, boxes):
           
            conditions_x = np.zeros(len(positions), dtype = bool)
            conditions_y = np.zeros(len(positions), dtype = bool)
            conditions_z = np.zeros(len(positions), dtype = bool)
            box_cond = np.zeros(len(positions), dtype = bool)
            
            for box in boxes:
                
                center = box[0]
                x1 = box[1] / 2
                y1 = box[2] / 2
                z1 = box[3] / 2
                
                box_x = np.array([center[0] - x1, center[0] + x1]) 
                box_y = np.array([center[1] - y1, center[1] + y1]) 
                box_z = np.array([center[2] - z1, center[2] + z1]) 
                
                trench_x  =  (positions[:,0] > box_x[0]) & (positions[:,0] < box_x[1])
                trench_y  =  (positions[:,1] > box_y[0]) & (positions[:,1] < box_y[1])
                trench_z  =  (positions[:,2] > box_z[0]) & (positions[:,2] < box_z[1])
                
                box_cond = (box_cond) | (trench_x & trench_y & trench_z)
                conditions_x = conditions_x | trench_x
                conditions_y = conditions_y | trench_y
                conditions_z = conditions_z | trench_z
        
            return box_cond, conditions_x, conditions_y, conditions_z

        wall_x = (positions[:,0] < 0) | (positions[:,0] > size[0])
        wall_y = (positions[:,1] < 0) | (positions[:,1] > size[1])
        wall_z = (positions[:,2] < 0) | (positions[:,2] > size[2])
          
        if len(boxes):
            
            box_conditions = boxes_to_conditions(positions, boxes)
            
            inside_condition = box_conditions[0]
            
            conditions_x = box_conditions[1]
            conditions_y = box_conditions[2]
            conditions_z = box_conditions[3]

        velocities[(wall_x), 0] *= -1 
        velocities[(wall_y), 1] *= -1
        velocities[(wall_z), 2] *= -1
        
        
        velocities[(inside_condition & conditions_x), 0] *= -1 
        velocities[(inside_condition & conditions_y), 1] *= -1
        velocities[(inside_condition & conditions_z), 2] *= -1
   

        return velocities
    
    
    def ideal_gas_forces(self, epoche, update_step):
        
        #parameters to tune:
        r_min = self.model_parameters['r_minimum_collision']
        LJ_pot = self.model_parameters['LJ_potential']
        time_step = self.model_parameters['time_step']
        cut_off_radius = self.model_parameters['cut_off_radius']
        
        
          
        positions = Simulation.position_velocity_adsorb_to_dict(self)['loc']
        velocities = Simulation.position_velocity_adsorb_to_dict(self)['velocity']
        adsorbed = Simulation.position_velocity_adsorb_to_dict(self)['ad']
        size = self.enviroment.get_size()
        
        boxes = self.enviroment.get_boxes()
        
        
        for step in range(update_step*epoche): 

            if step % update_step == 0:

                positions = Simulation.position_velocity_adsorb_to_dict(self)['loc']
                velocities = Simulation.position_velocity_adsorb_to_dict(self)['velocity']
                        
            tree = cKDTree(positions) # drzewo przypisujace najblizyszhc sasiadow z biblioteki scipy.spatial
            
            pairs_to_solve = tree.query_pairs(r = cut_off_radius) #wlasciwy moment wywolania algorytmu poszukujacego jednego najblizszego sasiada
    
            force_on_particle = np.zeros((len(positions),3))
            
            for pair in pairs_to_solve:
                
                force_directions = positions[pair[1]] - positions[pair[0]]

                r = np.linalg.norm(force_directions)
                
                if r < r_min:
                    force_magnitude = LJ_pot * (2 / r_min**14 - 1 / r_min**8)
                
                else:
                    force_magnitude = LJ_pot * (2 / r**14 - 1 / r**8)
                
                 
                force_on_particle[pair[0]] += force_magnitude * force_directions / r
                force_on_particle[pair[1]] -= force_magnitude * force_directions / r
               
                
            #Verlet + boundaries
            velocities += force_on_particle * time_step / 2
            positions = (positions + velocities * time_step)
            #positions = Simulation.boundary_conditions_check(size, positions, 'trench')            
            velocities += force_on_particle * time_step / 2
            velocities = Simulation.box_periodic_boundary_conditions(size, positions, velocities, boxes)
            
        Simulation.set_particles_position(self, positions) 
        Simulation.set_particles_velocity(self, velocities)
        
        Simulation.adsorbtion_check(self)
        
     
   
    def adsorption_conditions(self):
        
       pass
   
    
   
    
   
    def adsorption_check(self):        
        
        adsorbed = self.get_particles_adsorbed()
        positions = self.get_particles_position()
        
        
        
        
        if  
        
    
    
    def cycle():
        #dose
        
        #purge
        
        #dose 2
        pass
    
    def DLA():
        pass
    
               
    def one_step(self, time_step = 0.001):
        # to do - boundary condition
        for particle in self.particles:
            particle.position += particle.velocity*time_step
            
    
    def simulation_steps(self, time_step, number_of_steps):
        for i in range(number_of_steps):
            Simulation.one_step(self, time_step)
        
    
    def visualize_particles(self):
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        dims = self.enviroment.size
        
        boxes = self.enviroment.get_boxes()
        
        
        positions = np.array([particle.position for particle in self.particles])
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c = 'blue', s = 1)

        
        
        if len(boxes):

            Simulation.visualise_boxes(self, ax)
            
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Particle Positions')
        
                
        ax.set_xlim([0,dims[0]])
        ax.set_ylim([0,dims[1]])
        ax.set_zlim([0,dims[2]])
        
        plt.show()   
        
           
    def visualise_boxes(self, ax):
        
        size = self.enviroment.get_size()
        boxes = self.enviroment.get_boxes()
        
        for box in boxes:
            
            
            center = np.array(box[0])
            x = box[1] / 2
            y = box[2] / 2
            z = box[3] / 2

            #up
            ax.plot([center[0] - x , center[0]  + x, center[0]  + x,center[0]  - x,center[0] - x], 
                    [center[1] - y, center[1] - y,center[1]  + y,center[1]  + y, center[1]  - y],
                    [center[2] + z ,center[2] + z ,center[2] + z ,center[2] +z ,center[2] + z ], 
                    color = 'red', alpha = 0.4)
            #bottom
            ax.plot([center[0]  - x , center[0]  + x,center[0]  + x,center[0]  - x,center[0] - x], 
                    [center[1] - y, center[1] - y,center[1]  + y,center[1]  + y, center[1]  - y],
                    [center[2] -z ,center[2] -z ,center[2] -z ,center[2] -z ,center[2] -z ],
                    color = 'red', alpha = 0.4)
            #frames
            ax.plot([center[0]  - x, center[0]  - x], [center[1]  - y, center[1]  - y],[center[2] -z ,center[2] +z ],color = 'red', alpha = 0.4)
            ax.plot([center[0]  + x, center[0]  + x], [center[1]  - y, center[1]  - y],[center[2] -z ,center[2] +z ],color = 'red', alpha = 0.4)
            ax.plot([center[0]  - x, center[0]  - x], [center[1]  + y, center[1]  + y],[center[2] -z ,center[2] +z ],color = 'red', alpha = 0.4)
            ax.plot([center[0]  + x, center[0]  + x], [center[1]  + y, center[1]  + y],[center[2] -z ,center[2] +z ],color = 'red', alpha = 0.4)

                

    def gif_simulation(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        dims = self.enviroment.size
        collisions = self.enviroment.collisions
        
        positions = np.array([particle.position for particle in self.particles])
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2])
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Particle Positions')
        ax.text(np.mean(Simulation.get_particles_velocity(self,)))
        
        
        ax.set_xlim([0,dims[0]])
        ax.set_ylim([0,dims[1]])
        ax.set_zlim([0,dims[2]])
        
  
        
        plt.show()   
        
        
    


#%% testing the simulation interactions and funcionalities

part_list = [Particle('zno', np.array([0,0,0]), np.array([0,0,0]),0,0, {}) for i in range(200)]


part1 = Particle('zno', np.array([0,0,0]),
                 np.array([0,0,0]),
                 2.5,0, {})

part2 = Particle('tma', np.array([1,1,1]),
                 np.array([1,2,3]),
                 2.5,0, {})   

part3 = Particle('dez', np.array([2,2,2]),
                 np.array([5,5,5]),
                 4,1, {})                 
                 


particles = part_list


env = Enviroment(np.array([200,100,90]),100,1,'Silicon', np.array([1,2,3,4]))


box1_coord = [[90,30,20], 200,150,31]

box2_coord = [[300,30,20], 200,150,31]

env.set_boxes([box1_coord, box2_coord] )


sim  = Simulation(particles, 0, env)

sim.set_model_parameters({'r_minimum_collision': 0.5,
                         'LJ_potential': 0.001,
                         'time_step': 0.005,
                         'cut_off_radius': 4})



sim.assign_init_location('streamline')

sim.assign_init_velocities('streamline', velo_avg = 5, distortion = 2)


#sim.simulation_steps(0.1,100)

sim.visualize_particles()
epoch = 1000

for i in range(epoch):
    sim.ideal_gas_forces(2, 10)
    
    sim.visualize_particles()

    
