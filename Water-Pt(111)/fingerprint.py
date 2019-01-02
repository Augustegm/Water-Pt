from ase.io import read, write
from ase.neighborlist import neighbor_list
from ase.neighborlist import primitive_neighbor_list
import numpy as np

def fingerprint(images, cutoff, etas, Rc,  elements, cutoff_function = False):
    for count, atoms in enumerate(images):
        species = sorted(list(set(atoms.get_chemical_symbols())))
        N_atoms = len(atoms)
        N_species = len(species)
        N_etas = len(etas)
        neighbors = neighbor_list('ijdD', atoms, cutoff/2 * np.ones(N_atoms))

        for i, atom in enumerate(atoms):
            if atom.symbol in elements:
                fingerprint = np.zeros([3,N_species,N_etas])
                for j, element in enumerate(species):
                    for k, (eta,R) in enumerate(zip(etas,Rc)):
                        indices = np.argwhere(neighbors[0] == atom.index)
                        neighbor_indices = neighbors[1][indices] #get indices of neighbor atoms
                        for index, neighbor_index in zip(indices, neighbor_indices):
                            if atoms[neighbor_index[0]].symbol == element:
                                Rij = neighbors[2][index[0]] #distance from central atom to neighbor
                                rij = neighbors[3][index[0]] #distances vector from central atom to neighbor
                            
                                if cutoff_function:
                                    fingerprint[:,j,k] += rij/Rij * np.exp(-((Rij-R)/eta)**2) * 0.5 * (np.cos(np.pi*Rij/cutoff) + 1)
                                else:
                                    fingerprint[:,j,k] += rij/Rij * np.exp(-((Rij-R)/eta)**2)
                                
                fingerprint = fingerprint.reshape(3, N_species * N_etas)

                try:
                    fingerprints = np.concatenate((fingerprints, fingerprint), axis = 0)
                except UnboundLocalError: #fingerprints not defined in very first iteration
                    fingerprints = fingerprint 
            else:
                continue
                    
    return fingerprints




