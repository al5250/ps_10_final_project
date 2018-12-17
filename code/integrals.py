import numpy as np
from utils import gaussian_norm_coef, coulomb_auxiliary_integrals 

def overlap(E, a, b, L_A, L_B):
    '''
    Computes the normalized overlap integral between two 3D Gaussians.
    
    Args:
      E: (3, t_max+1, l_A_max+1, l_B_max+1) array for expansion coefficients for overlap distribution, where t_max >= 0, l_A_max >= L_A.max(), and l_B_max >= L_B.max().
      a: Exponent of Gaussian A.
      b: Exponent of Gaussian B.
      L_A: (3,) array representing angular vector for Gaussian A.
      L_B: (3,) array representing angular vector for Gaussian B.
    
    Returns:
      Normalized overlap integral.
    '''
    p = a + b
    n_A = gaussian_norm_coef(a, L_A.sum())
    n_B = gaussian_norm_coef(b, L_B.sum())
    S_x = (np.pi / p) ** (1/2) * E[0, 0, L_A[0], L_B[0]]
    S_y = (np.pi / p) ** (1/2) * E[1, 0, L_A[1], L_B[1]]
    S_z = (np.pi / p) ** (1/2) * E[2, 0, L_A[2], L_B[2]]
    S = n_A * n_B * S_x * S_y * S_z
    return S

def _kinetic_1d(E, a, b, l_A, l_B):
    '''
    Computes the unnormalized kinetic energy integral in one dimension between two Gaussians.
    
    Args:
      E: Expansion coefficients for overlap distribution.
      a: Exponent for Gaussian A.
      b: Exponent for Gaussian B.
      l_A: Component of angular vector for Gaussian A in given dimension.
      l_B: Component of angular vector for Gaussian B in given dimension.
    
    Returns:
      Unnormalized kinetic energy integral in given dimension.
    '''
    p = a + b
    S_i_jp2 = (np.pi / p) ** (1/2) * E[0, l_A, l_B + 2]
    S_i_j = (np.pi / p) ** (1/2) * E[0, l_A, l_B]
    if l_B >= 2:
        S_i_jm2 = (np.pi / p) ** (1/2) * E[0, l_A, l_B - 2]
    else:
        S_i_jm2 = 0
    T = -2 * b ** 2 * S_i_jp2 + b * (2 * l_B + 1) * S_i_j - 1/2 * l_B * (l_B - 1) * S_i_jm2
    return T

def kinetic(E, a, b, L_A, L_B):
    '''
    Computes the kinetic energy integral between two 3D Gaussians.  
    
    Args:
      E: (3, t_max+1, l_A_max+1, l_B_max+1) array for expansion coefficients for overlap distribution, where t_max >= 0, l_A_max >= L_A.max(), and l_B_max >= L_B.max() + 2.
      a: Exponent of Gaussian A.
      b: Exponent of Gaussian B.
      L_A: (3,) array representing angular vector for Gaussian A.
      L_B: (3,) array representing angular vector for Gaussian B.
    
    Returns:
      Normalized kinetic energy integral.
    '''
    p = a + b
    n_A = gaussian_norm_coef(a, L_A.sum())
    n_B = gaussian_norm_coef(b, L_B.sum())
    S_x = (np.pi / p) ** (1/2) * E[0, 0, L_A[0], L_B[0]] 
    S_y = (np.pi / p) ** (1/2) * E[1, 0, L_A[1], L_B[1]] 
    S_z = (np.pi / p) ** (1/2) * E[2, 0, L_A[2], L_B[2]]
    T_x = _kinetic_1d(E[0], a, b, L_A[0], L_B[0])
    T_y = _kinetic_1d(E[1], a, b, L_A[1], L_B[1])
    T_z = _kinetic_1d(E[2], a, b, L_A[2], L_B[2])
    return n_A * n_B * (T_x * S_y * S_z + S_x * T_y * S_z + S_x * S_y * T_z)

def _single_nucleus_attraction(E, R, Z_C, a, b, L_A, L_B):
    '''
    Computes the unnormalized Coulombic nuclear attraction integral between two Gaussians and a given nucleus.
    
    Args:
      E: Expansion coefficients for overlap distribution.
      R: (n_max+1, t_max+1, u_max+1, v_max+1) array for Coulomb auxiliary integrals, where n_max >= 0, t_max >= L_A[0] + L_B[0], u_max >= L_A[1] + L_B[1], v_max >= L_A[2] + L_B[2].
      Z_C: Charge of nucleus C.
      a: Exponent for Gaussian A.
      b: Exponent for Gaussian B.
      L_A: Angular vector for Gaussian A.
      L_B: Angular vector for Gaussian B.
    
    Returns:
      Unnormalized Coulombic nuclear attraction integral.
    '''
    p = a + b
    Theta = 0
    for t in range(L_A[0] + L_B[0] + 1):
        for u in range(L_A[1] + L_B[1] + 1):
            for v in range(L_A[2] + L_B[2] + 1):
                Theta += E[0, t, L_A[0], L_B[0]] * E[1, u, L_A[1], L_B[1]] * E[2, v, L_A[2], L_B[2]] * R[0, t, u, v]
    return -2 * np.pi / p * Z_C * Theta 

def nuclear_attraction(E, a, b, R_A, R_B, L_A, L_B, R_Cs, Z_Cs):
    '''
    Computes the normalized Coulombic nuclear attraction integral between two Gaussians for all nuclei.
    
    Args:
      E: (3, t_max+1, l_A_max+1, l_B_max+1) array for expansion coefficients for overlap distribution, where t_max >= L_A.max() + L_B.max(), l_A_max >= L_A.max(), and l_B_max >= L_B.max().
      a: Exponent of Gaussian A.
      b: Exponent of Gaussian B.
      R_A: (3,) array denoting center of Gaussian A.
      R_B: (3,) array denoting center of Gaussian B.
      L_A: (3,) array representing angular vector for Gaussian A.
      L_B: (3,) array representing angular vector for Gaussian B.
      R_Cs: (n_nuclei, 3) array denoting centers of nuclei. 
      Z_Cs: (n_nuclei,) array denoting charges of nuclei.
    
    Returns:
      Normalized Coulombic nuclear attraction integral.
    '''
    n_A = gaussian_norm_coef(a, L_A.sum())
    n_B = gaussian_norm_coef(b, L_B.sum())
    V = 0
    p = a + b
    R_P = (a * R_A + b * R_B) / p
    L_AB = L_A + L_B
    for R_C, Z_C in zip(R_Cs, Z_Cs):
        R = coulomb_auxiliary_integrals(p, R_P, R_C, t_max=L_AB[0], u_max=L_AB[1], v_max=L_AB[2])
        V += _single_nucleus_attraction(E, R, Z_C, a, b, L_A, L_B)
    return n_A * n_B * V

def ee_repulsion(E1, E2, a, b, c, d, R_A, R_B, R_C, R_D, L_A, L_B, L_C, L_D):
    '''
    Computes the normalized Coulombic two electron repulsion integral between four Gaussians.
    
    Args:
      E1: (3, t_max+1, l_A_max+1, l_B_max+1) array for overlap expansion coefficients between Gaussian A and Gaussian B, where t_max >= L_A.max() + L_B.max(), l_A_max >= L_A.max(), and l_B_max >= L_B.max().
      E2: (3, t_max+1, l_C_max+1, l_D_max+1) array for overlap expansion coefficients between Gaussian C and Gaussian D, where t_max >= L_C.max() + L_D.max(), l_C_max >= L_A.max(), and l_D_max >= L_D.max().
      a: Exponent of Gaussian A.
      b: Exponent of Gaussian B.
      c: Exponent of Gaussian C.
      d: Exponent of Gaussian D.
      R_A: (3,) array denoting center of Gaussian A.
      R_B: (3,) array denoting center of Gaussian B.
      R_C: (3,) array denoting center of Gaussian C.
      R_D: (3,) array denoting center of Gaussian D.
      L_A: (3,) array representing angular vector for Gaussian A.
      L_B: (3,) array representing angular vector for Gaussian B.
      L_C: (3,) array representing angular vector for Gaussian C.
      L_D: (3,) array representing angular vector for Gaussian D.
    
    Returns:
      Normalized Coulombic two electron repulsion integral.
    '''
    n_A = gaussian_norm_coef(a, L_A.sum())
    n_B = gaussian_norm_coef(b, L_B.sum())
    n_C = gaussian_norm_coef(c, L_C.sum())
    n_D = gaussian_norm_coef(d, L_D.sum())
    p = a + b
    q = c + d
    alpha = p * q / (p + q)
    R_P = (a * R_A + b * R_B) / p
    R_Q = (c * R_C + d * R_D) / q
    L_ABCD = L_A + L_B + L_C + L_D
    R = coulomb_auxiliary_integrals(alpha, R_P, R_Q, t_max=L_ABCD[0], u_max=L_ABCD[1], v_max=L_ABCD[2])
    ee_repuls = 0
    for t1 in range(L_A[0] + L_B[0] + 1):
        for u1 in range(L_A[1] + L_B[1] + 1):
            for v1 in range(L_A[2] + L_B[2] + 1):
                for t2 in range(L_C[0] + L_D[0] + 1):
                    for u2 in range(L_C[1] + L_D[1] + 1):
                        for v2 in range(L_C[2] + L_D[2] + 1):
                            ee_repuls += E1[0, t1, L_A[0], L_B[0]] * E1[1, u1, L_A[1], L_B[1]] * E1[2, v1, L_A[2], L_B[2]] * \
                                         E2[0, t2, L_C[0], L_D[0]] * E2[1, u2, L_C[1], L_D[1]] * E2[2, v2, L_C[2], L_D[2]] * \
                                         (-1)**(t2+u2+v2) * R[0, t1+t2, u1+u2, v1+v2]
    return n_A * n_B * n_C * n_D * 2 * (np.pi) ** (5/2) / p / q / np.sqrt(p + q) * ee_repuls