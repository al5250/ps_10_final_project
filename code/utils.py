import numpy as np
from scipy.special import hyp1f1

def load_basis_set(path_to_file='basis_sets/STO-3G.txt'):
    '''
    Loads dictionary of Gaussian basis set exponents and coefficients from external file.  Currently, only S and SP orbitals are supported.
    
    Args:
      path_to_file: A string denoting path to file of basis set.
    
    Returns:
      A dictionary indexed by element abbreviation.  Contents for each element include an (N,) array of orbital types, an (N, K) array of Gaussian exponents, and an (N, K) array of Gaussian coefficients, where N is the number of orbitals and K is the number of primitive Gaussians per orbital.       
    '''
    basis_dict = {}
    with open(path_to_file, 'r') as fp:
        line = fp.readline()
        while line:
            if line[0] == '-':
                atom = line.split()[0][1:]
                orbital_types = []
                expos = []
                coefs = []
                line = fp.readline()
                while line[0] != '*':
                    orbital = line.split()[0]
                    if orbital == 'S':
                        orb_expos = []
                        orb_coefs = []
                        for _ in range(3):
                            line = fp.readline()
                            vals = [float(x) for x in list(line.split())]
                            orb_expos.append(vals[0])
                            orb_coefs.append(vals[1])
                        orbital_types.append('S')
                        expos.append(orb_expos)
                        coefs.append(orb_coefs)
                    elif orbital == 'SP':
                        orb_expos = []
                        orb_coefs_S = []
                        orb_coefs_P = []
                        for _ in range(3):
                            line = fp.readline()
                            vals = [float(x) for x in list(line.split())]
                            orb_expos.append(vals[0])
                            orb_coefs_S.append(vals[1])
                            orb_coefs_P.append(vals[2])
                        orbital_types.append('S')
                        expos.append(orb_expos)
                        coefs.append(orb_coefs_S)
                        for t in ['Px', 'Py', 'Pz']:
                            orbital_types.append(t)
                            expos.append(orb_expos)
                            coefs.append(orb_coefs_P)
                    else:
                        raise ValueError('Orbital {} not supported!'.format(orbital))
                    line = fp.readline()
                basis_dict[atom] = {'orbital_types' : np.array(orbital_types), 
                                    'expos' : np.array(expos),
                                    'coefs': np.array(coefs)}
            line = fp.readline()
        return basis_dict

def angular_vector(ao_type):
    '''
    Matches orbital type string to array of quantum numbers.  Currently, only S and P orbitals are supported.
    
    Args:
      ao_type: Input string denoting atomic orbital type.  Valid options include ['S', 'P_x', 'P_y', 'P_z'].
    
    Returns:
      A (3,) array of Cartesian coordinates representing the azimuthal and magnetic quantum numbers.
    '''
    if ao_type == 'S':
        return np.array([0, 0, 0])
    elif ao_type == 'Px':
        return np.array([1, 0, 0])
    elif ao_type == 'Py':
        return np.array([0, 1, 0])
    elif ao_type == 'Pz':
        return np.array([0, 0, 1])
    else:
        raise NotImplementedError()
        
def gaussian_norm_coef(a, l_A):
    '''
    Calculates the normalization constant for a 3D Gaussian.  Currently, only S and P orbitals are supported.
    
    Args:
      a: Scalar exponent of Gaussian.
      l_A: Azimuthal quantum number.  Valid options include [0, 1].
      
    Returns:
      Normalization constant for corresponding orbital type.
    '''
    if l_A == 0:
        return (2 * a / np.pi) ** (3/4)
    elif l_A == 1:
        return (128 * a ** 5 / np.pi ** 3) ** (1/4)
    else:
        raise NotImplementedError('Input `l_A` must be an integer <= 1.')

def boys(n, T):
    '''
    Evaluates the nth Boys function for input argument T.
    
    Args:
      n: Non-negative integer.
      T: Input argument.
      
    Returns:
      Output of function.
    '''
    return hyp1f1(n + 0.5, n + 1.5, -T) / (2.0 * n + 1.0) 

def overlap_expansion_coefs(a, b, R_A, R_B, l_A_max=0, l_B_max=0):
    '''
    Recursively computes expansion coefficients for the overlap distribution of two Gaussians.  Helper function for computing Hartree-Fock integrals.  
    
    Args:
      a: Exponent of Gaussian A.
      b: Exponent of Gaussian B.
      R_A: (3,) array denoting center of Gaussian A.
      R_B: (3,) array denoting center of Gaussian B.
      l_A_max: Maximum azimuthal quantum number for Gaussian A.
      l_B_max: Maximum azimuthal quantum number for Gaussian B.
    
    Returns:
      A (3, t_max+1, l_A_max+1, l_B_max+1) tensor where t_max = l_A_max + l_B_max.  The [i, t, l_A, l_B]-th value corresponds to dimension i (where i=0 denotes 'x', i=1 denotes 'y', and i=2 denotes 'z'), indicator value t, azimuthal quantum number l_A for Gaussian A, and azimuthal quantum number l_B for Gaussian B. 
    '''
    t_max = l_A_max + l_B_max + 1
    E = np.full((3, t_max + 1, l_A_max + 1, l_B_max + 1), np.nan)
    
    for i in range(3):
        
        p = a + b
        R_P_i = (a * R_A[i] + b * R_B[i]) / p
        R_AB_i = R_B[i] - R_A[i]
        R_AP_i = R_P_i - R_A[i]
        R_BP_i = R_P_i - R_B[i]
        
        # Base Case (l_A = 0)
        E[i, 0, 0, 0] = np.exp(-a * b / p * R_AB_i ** 2)
        E[i, 1:, 0, 0] = 0
        
        E[i, t_max, :, :] = 0
        
        for l_A in range(1, l_A_max + 1):
            for t in range(0, t_max):
                if t > l_A:
                    E[i, t, l_A, 0] = 0
                else:
                    E[i, t, l_A, 0] = 1 / (2*p) * E[i, t-1, l_A-1, 0] + \
                                        R_AP_i * E[i, t, l_A-1, 0] + \
                                        (t+1) * E[i, t+1, l_A-1, 0]
        
        for l_B in range(1, l_B_max + 1):
            for l_A in range(0, l_A_max + 1):
                for t in range(0, t_max):
                    E[i, t, l_A, l_B] = 1 / (2*p) * E[i, t-1, l_A, l_B-1] + \
                                        R_BP_i * E[i, t, l_A, l_B-1] + \
                                        (t+1) * E[i, t+1, l_A, l_B-1]                
    
    return E

def coulomb_auxiliary_integrals(p, R_P, R_C, t_max=1, u_max=1, v_max=1):
    '''
    Recursively computes Coulomb auxiliary integrals for a given exponent and two Gaussian centers.  Helper function for computing Hartree-Fock integrals involving Coulombic forces.    
    
    Args:
      p: Combined Gaussian exponent.
      R_P: (3,) array denoting center of Gaussian P.
      R_C: (3,) array denoting center of Gaussian C.
      t_max: Maximum value for dimension 'x'.  Corresponds to azimuthal quantum number in 'x' direction.
      u_max: Maximum value for dimension 'y'.  Corresponds to azimuthal quantum number in 'y' direction.
      v_max: Maximum value for dimension 'z'.  Corresponds to azimuthal quantum number in 'z' direction.
    
    Returns:
      A (n_max+1, t_max+1, u_max+1, v_max+1) tensor where n_max = t_max + u_max + v_max.  The [n, t, u, v]-th value corresponds to indicator value n, azimuthal quantum number t in 'x' direction, azimuthal quantum number u in 'y' direction, and azimuthal quantum number v in 'z' direction.  Typically, only the n=0 sub-tensor is used in external calculations.  
    '''
    R_PC = R_P - R_C
    R2_PC = np.sum(R_PC ** 2)
    
    n_max = t_max + u_max + v_max
    R = np.full((n_max + 1, t_max + 1, u_max + 1, v_max + 1), np.nan)
    
    # Base Case (n, t=0, u=0, v=0)
    for n in range(0, n_max + 1):
        R[n, 0, 0, 0] = (-2 * p) ** n * boys(n, p * R2_PC)
    
    if t_max >= 1:
        # (n, t=1, u=0, v=0)
        for n in range(0, n_max):
                R[n, 1, 0, 0] = R_PC[0] * R[n+1, 0, 0, 0]

        # (n, t>1, u=0, v=0)
        for t in range(2, t_max + 1):
            for n in range(n_max + 1 - t):
                R[n, t, 0, 0] = R_PC[0] * R[n+1, t-1, 0, 0] + (t-1) * R[n+1, t-2, 0, 0]
    
    if u_max >= 1:
        # (n, t, u=1, v=0)
        for t in range(0, t_max + 1):
            for n in range(0, n_max - t):
                R[n, t, 1, 0] = R_PC[1] * R[n+1, t, 0, 0]

        # (n, t, u>1, v=0)
        for u in range(2, u_max + 1):
            for t in range(0, t_max + 1):
                for n in range(0, n_max + 1 - u - t):
                    R[n, t, u, 0] = R_PC[1] * R[n+1, t, u-1, 0] + (u-1) * R[n+1, t, u-2, 0]
    
    if v_max >=1 :
        # (n, t, u, v=1)
        for u in range(0, u_max + 1):
            for t in range(0, t_max + 1):
                for n in range(0, n_max - u - t):
                    R[n, t, u, 1] = R_PC[2] * R[n+1, t, u, 0]

        # (n, t, u, v>1)
        for v in range(2, v_max + 1):
            for u in range(0, u_max + 1):
                for t in range(0, t_max + 1):
                    for n in range(0, n_max + 1 - v - u - t):
                        R[n, t, u, v] = R_PC[2] * R[n+1, t, u, v-1] + (v-1) * R[n+1, t, u, v-2]
                    
    return R