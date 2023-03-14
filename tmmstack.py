import numpy as np
import tmm
from scipy import interpolate
import pandas as pd


class Layer:
    
    
    def __init__(self, thickness, refractive_index, name ):
        
        if name == None:
            self.name = 'layer'
        else:
            self.name = name
        
        self.thickness = thickness
        
        self.ref_data = 'ref. index is not taken from file'
        
        
        
        if type(refractive_index) in [int, float, complex]:
            self.refractive_index = refractive_index
            
        if type(refractive_index) == list:
            #refractive_index = np.array(refractive_index)
            self.refractive_index = np.array(refractive_index)
            # if self.refractive_index.shape[1] == 3:
                
            #     self.refractive_index[:,1] = self.refractive_index[:,1]+ 1j*self.refractive_index[:,2]
            #     self.refractive_index = self.refractive_index[:,0:1]
                
            self.refractive_index = interpolate.interp1d(self.refractive_index[:,0].real,
                          self.refractive_index[:,1], kind='quadratic')
        
        if type(refractive_index) == str:
            """plik musi byc postaci:
            wl n k
            np.
            300 1.5 0.9
            """
            self.ref_data = refractive_index
            ref = np.genfromtxt(refractive_index, dtype = float, delimiter = ' ')
            #converting to nm instead of um:
            if ref[0,0] < 100:     
                ref[:,0] = ref[:,0]*1000
                print('Ref. index adjusted to nm!')
                #converting to complex number
            ref = np.vstack([ref[:,0].astype(complex), ref[:,1]+ 1j*ref[:,2]]).T

            self.refractive_index = interpolate.interp1d(ref[:,0].real,
                                    ref[:,1], kind='quadratic')
            
            
    
    def info(self):
        ref = 'function'
        if type(self.refractive_index) in [int, float, complex]:
            ref = str(self.refractive_index)
            
        if type(self.refractive_index) == list:
            ref = str(self.refractive_index[0]) + ' - ' + str(self.refractive_index[-1])
            
        return f'{self.name}: {self.thickness} nm, n: {ref}            taken from: {self.ref_data}'
    
    def print_info(self):
        print(self.info())     
        

class Stack(Layer):
    
    def __init__(self, name = None, layers = None):
        
        if name is None:
            self.name = 'Unnamed_stack'
        else:
            self.name = name 

        if layers is None:
            self.layers = []
        else:
            self.layers = layers
    
    def print_stack(self):
        for i,layer in enumerate(self.layers):
            
            print( f'Layer {i}: {layer.info()}')
            
            
    def stack_info_string(self):
        s = ''
        for i, layer in enumerate(self.layers):
            s += '\n' + layer.info() + '\n'
        
        return s
            
            
            
            
    def defining_thicknesses(self):
        thicknesses = []
        for layer in self.layers:
            thicknesses.append(layer.thickness)
        
        return thicknesses
    
    def defining_ref(self):
        ref = []
        
        
        for layer in self.layers:
            ref.append(layer.refractive_index)
        
        return ref
        
         
    def tmm_calculation_norm(self, wavelengths = [300,1200], coherence_list = ['c']):
        """
        Funkcja do obliczen TMM dla kata padania 0. Zwraca T i R w formie arraya. 
        material_nk_data - lista wg wzoru: [ wl, real + j imaginary ]
        wavelengths - lista dlugosci fal [ start, end ] lub [n_1, n_2, n_3], gdzie n_i to kolejne wartosci warstwy
        """
        
        def ref_definition(self, wavelength):
            refs = []
                
            for layer in self.layers:
                ref_ind = layer.refractive_index
                if type(ref_ind) in [int, float, complex]:
                    refs.append(ref_ind)
                
                else:

                    refs.append(ref_ind(wavelength))
                    
            return refs
            
                
        Rnorm = []
        Tnorm = []
        
        wavelengths = np.linspace(wavelengths[0],wavelengths[1],
                                  wavelengths[1] - wavelengths[0] + 1 )
        
        
        if coherence_list[0] == 'c':
            for wl in wavelengths:
            
                Rnorm.append(tmm.coh_tmm('p', ref_definition(self, wl), self.defining_thicknesses(), 0, wl)['R'])
                Tnorm.append(tmm.coh_tmm('p', ref_definition(self, wl), self.defining_thicknesses(), 0, wl)['T'])
        
        if coherence_list[0] == 'i':
            for wl in wavelengths:
            
                Rnorm.append(tmm.inc_tmm('p', ref_definition(self, wl), self.defining_thicknesses(),coherence_list, 0, wl)['R'])
                Tnorm.append(tmm.inc_tmm('p', ref_definition(self, wl), self.defining_thicknesses(),coherence_list, 0, wl)['T'])
        
        

        return np.array(Rnorm), np.array(Tnorm) 
    
    @staticmethod
    def TRweight_calculation(opt_data, ref_spectrum_data):
        """opt_data = np.array([[wl, R(wl)],
                              [wl, R(wl)]...])"""
        
        
        opt_w = sum(ref_spectrum_data(opt_data[:,0])*opt_data[:,1]) / sum(ref_spectrum_data(opt_data[:,0]))

        
        return opt_w
    
    @staticmethod
    def nrel_data_format():
        """provide a file with AM spectrum and return interp object"""
        nrel = pd.read_csv('C:\\Users\\admin\\Desktop\\Freiburg_rough\\Desktop\\Data Collected\\References\\ASTMG173.csv', skiprows = 1, delimiter = ',')
        nrel = np.array(nrel)
        nrel = np.delete(nrel,(1,3),1)
        ph_flux = nrel[:,1]*nrel[:,0]*5.03*10**15
        
        nrel = np.array([nrel[:,0], ph_flux]).T
        nrel_interpolated = interpolate.interp1d(nrel[:,0], nrel[:,1], kind='quadratic')
        
        return nrel_interpolated    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    