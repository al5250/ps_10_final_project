import numpy as np
from utils import overlap_expansion_coefs, angular_vector
from integrals import overlap, kinetic, nuclear_attraction, ee_repulsion
import time

def compute_all_integrals(ao_exps, ao_coefs, ao_types, Rs, Zs, ao2nuc, ee_thres=1e-7):
    '''
    Computes all four revelant integrals for the Hartree-Fock algorithm - overlap, kinetic energy, nuclear attraction, and two electron repulsion.  Currently, only S and P type orbitals are supported.
    
    Args:
      ao_exps: (n_orbitals, K) array of Gaussian exponents, where n_orbitals is the number of atomic orbitals and K is the number of Gaussians used to characterize each orbital.
      ao_coefs: (n_orbitals, K) array of Gaussian coefficients.
      ao_types: (n_orbitals,) array of strings denoting atomic orbital types.
      Rs: (n_nuclei, 3) array of Gaussian centers, where n_nuclei is the number of atomic nuclei.
      Zs: (n_nuclei,) array of nucleus charges.
      ao2nuc: (n_orbitals,) array where value [i] ~ 1,...,n_nuclei denotes which nucleus the ith orbital is centered around.
      ee_thres: Scalar threshold for minimum test value needed to compute two electron repulsion integral.
    
    Returns:
      S: (n_orbitals, n_orbitals) array of overlap integrals.
      T: (n_orbitals, n_orbitals) array of kinetic energy integrals.
      V: (n_orbitals, n_orbitals) array of nuclear attraction integrals.
      ee_repulsion: (n_orbitals, n_orbitals, n_orbitals, n_orbitals) array of two electron repulsion integrals.
    '''
    
    n_orbitals = ao_exps.shape[0]
    
    # Compute overlap expansion coefficients    
    Es = {}
    for ao_A in range(n_orbitals):
        for ao_B in range(n_orbitals):
            a, b = ao_exps[ao_A], ao_exps[ao_B]
            R_A, R_B = Rs[ao2nuc[ao_A]], Rs[ao2nuc[ao_B]]
            for i in range(a.size):
                for j in range(b.size):
                    Es[(ao_A, ao_B, i, j)] = overlap_expansion_coefs(a[i], b[j], R_A, R_B, 3, 3)
    
    S = np.full((n_orbitals, n_orbitals), np.nan)
    T = np.full((n_orbitals, n_orbitals), np.nan)
    V = np.full((n_orbitals, n_orbitals), np.nan)
    ee_repuls = np.full((n_orbitals, n_orbitals, n_orbitals, n_orbitals), np.nan)
    
    # Compute overlap, kinetic energy, and nuclear attraction integrals
    for ao_A in range(n_orbitals):
        for ao_B in range(n_orbitals):

            a, b = ao_exps[ao_A], ao_exps[ao_B]
            d_A, d_B = ao_coefs[ao_A], ao_coefs[ao_B]
            R_A, R_B = Rs[ao2nuc[ao_A]], Rs[ao2nuc[ao_B]]
            L_A, L_B = angular_vector(ao_types[ao_A]), angular_vector(ao_types[ao_B])

            S_AB = np.zeros((a.size, b.size))
            T_AB = np.zeros((a.size, b.size))
            V_AB = np.zeros((a.size, b.size))
            for i in range(a.size):
                for j in range(b.size):
                    E = Es[(ao_A, ao_B, i, j)]
                    S_AB[i, j] = d_A[i] * d_B[j] * overlap(E, a[i], b[j], L_A, L_B)
                    T_AB[i, j] = d_A[i] * d_B[j] * kinetic(E, a[i], b[j], L_A, L_B)
                    V_AB[i, j] = d_A[i] * d_B[j] * nuclear_attraction(E, a[i], b[j], R_A, R_B, L_A, L_B, Rs, Zs)

            S[ao_A, ao_B] = S_AB.sum()
            T[ao_A, ao_B] = T_AB.sum()
            V[ao_A, ao_B] = V_AB.sum()

    # Pre-screen electron-electron repulsion integrals
    for ao_A in range(n_orbitals):
        for ao_B in range(n_orbitals):
            if np.isnan(ee_repuls[ao_A, ao_B, ao_A, ao_B]):
                a, b = ao_exps[ao_A], ao_exps[ao_B]
                d_A, d_B = ao_coefs[ao_A], ao_coefs[ao_B]
                R_A, R_B = Rs[ao2nuc[ao_A]], Rs[ao2nuc[ao_B]]
                L_A, L_B = angular_vector(ao_types[ao_A]), angular_vector(ao_types[ao_B])
                ee_repuls_ABAB = np.zeros((a.size, b.size, a.size, b.size))
                for i in range(a.size):
                    for j in range(b.size):
                        for k in range(a.size):
                            for l in range(b.size):
                                E1 = Es[(ao_A, ao_B, i, j)]
                                E2 = Es[(ao_A, ao_B, k, l)]
                                ee_repuls_ABAB[i, j, k, l] = d_A[i] * d_B[j] * d_A[k] * d_B[l] * \
                                                             ee_repulsion(E1, E2, a[i], b[j], a[k], b[l], 
                                                                          R_A, R_B, R_A, R_B, L_A, L_B, L_A, L_B) 
                val = ee_repuls_ABAB.sum()
                ee_repuls[ao_A, ao_B, ao_A, ao_B] = val
                ee_repuls[ao_B, ao_A, ao_B, ao_A] = val
                ee_repuls[ao_B, ao_A, ao_A, ao_B] = val
                ee_repuls[ao_A, ao_B, ao_B, ao_A] = val            
    
    # Compute electron-electron repulsion integrals
    n_screened = 0
    for ao_A in range(n_orbitals):
        for ao_B in range(n_orbitals):
            for ao_C in range(n_orbitals):
                for ao_D in range(n_orbitals):
                    if np.isnan(ee_repuls[ao_A, ao_B, ao_C, ao_D]):

                        # Prescreen integral value
                        test = np.sqrt(ee_repuls[ao_A, ao_B, ao_A, ao_B]) * np.sqrt(ee_repuls[ao_C, ao_D, ao_C, ao_D])
                        if test <= ee_thres:
                            val = 0
                            n_screened += 1
                        else:
                            a, b, c, d = ao_exps[ao_A], ao_exps[ao_B], ao_exps[ao_C], ao_exps[ao_D]
                            d_A, d_B, d_C, d_D = ao_coefs[ao_A], ao_coefs[ao_B], ao_coefs[ao_C], ao_coefs[ao_D]
                            R_A, R_B, R_C, R_D = Rs[ao2nuc[ao_A]], Rs[ao2nuc[ao_B]], Rs[ao2nuc[ao_C]], Rs[ao2nuc[ao_D]]
                            L_A, L_B = angular_vector(ao_types[ao_A]), angular_vector(ao_types[ao_B])
                            L_C, L_D = angular_vector(ao_types[ao_C]), angular_vector(ao_types[ao_D])
                            ee_repuls_ABCD = np.zeros((a.size, b.size, c.size, d.size))
                            for i in range(a.size):
                                for j in range(b.size):
                                    for k in range(c.size):
                                        for l in range(d.size):
                                            E1 = Es[(ao_A, ao_B, i, j)]
                                            E2 = Es[(ao_C, ao_D, k, l)]
                                            ee_repuls_ABCD[i, j, k, l] = d_A[i] * d_B[j] * d_C[k] * d_D[l] * \
                                                                         ee_repulsion(E1, E2, a[i], b[j], c[k], d[l], 
                                                                                      R_A, R_B, R_C, R_D, L_A, L_B, L_C, L_D) 
                            val = ee_repuls_ABCD.sum()
                        ee_repuls[ao_A, ao_B, ao_C, ao_D] = val
                        ee_repuls[ao_C, ao_D, ao_A, ao_B] = val
                        ee_repuls[ao_B, ao_A, ao_D, ao_C] = val
                        ee_repuls[ao_D, ao_C, ao_B, ao_A] = val
                        ee_repuls[ao_B, ao_A, ao_C, ao_D] = val
                        ee_repuls[ao_D, ao_C, ao_A, ao_B] = val
                        ee_repuls[ao_A, ao_B, ao_D, ao_C] = val
                        ee_repuls[ao_C, ao_D, ao_B, ao_A] = val
    
    return S, T, V, ee_repuls

def hartree_fock(n_electrons, ao_exps, ao_coefs, ao_types, Rs, Zs, ao2nuc, alg_thres=1e-8, ee_thres=1e-7, max_iters=100, verbose=False):
    
    n_orbitals = ao_exps.shape[0]
    n_nuclei = Rs.shape[0]
        
    # Compute nuclear-nuclear repulsion energy
    V_nn = 0
    for i in range(n_nuclei):
        for j in range(i+1, n_nuclei):
            V_nn += Zs[i] * Zs[j] / np.sqrt(np.sum((Rs[i] - Rs[j]) ** 2))
    if verbose:
        print('************************ Hartree-Fock Algorithm ************************')
        print('num ele: %2d | num orb: %2d | nuc energy: %.3f' % (n_electrons, n_orbitals, V_nn))
    
    # Compute electron integrals
    t = time.time()
    S, T, V, ee_repuls = compute_all_integrals(ao_exps, ao_coefs, ao_types, Rs, Zs, ao2nuc, ee_thres)
    delta_t = time.time() - t
    if verbose:
        print('int time: %.2f sec' % delta_t)
    
    # Construct orthonormal basis
    eig_S, U = np.linalg.eigh(S)
    eig_S_sqrt = eig_S ** (-1/2) * np.eye(len(eig_S))
    X = U @ eig_S_sqrt @ U.conj().T
    
    # Run main algorithm
    H = T + V
    F = H.copy()
    # F = np.zeros((n_orbitals, n_orbitals))
    # for mu in range(n_orbitals):
        # for nu in range(n_orbitals):
            # F[mu, nu] = 1.75 * S[mu, nu] * (H[mu, mu] + H[nu, nu]) / 2
    n_iter = 0
    P = np.zeros((n_orbitals, n_orbitals))
    P_prev = np.full((n_orbitals, n_orbitals), np.inf)
        
    while np.max(np.abs(P - P_prev)) > alg_thres and n_iter < max_iters:
        F_ = X.conj().T @ F @ X 
        eps, C_ = np.linalg.eigh(F_)
        C = X @ C_
        P_prev = P
        P = np.zeros((n_orbitals, n_orbitals))
        for mu in range(n_orbitals):
            for nu in range(n_orbitals):
                for m in range(n_electrons // 2):
                    P[mu, nu] += C[mu, m] * C[nu, m]
        G = np.zeros((n_orbitals, n_orbitals))
        for ao1 in range(n_orbitals):
            for ao2 in range(n_orbitals):
                for ao3 in range(n_orbitals):
                    for ao4 in range(n_orbitals):
                        G[ao1, ao2] += (P[ao3, ao4] * (2 * ee_repuls[ao1, ao2, ao4, ao3] - ee_repuls[ao1, ao3, ao4, ao2]))
        F = H + G
        E_ele = np.sum(P * (H + F))
        E_tot = E_ele + V_nn
        if verbose:
            print('iter: #%3d | ele energy: %.8f | tot mol energy: %.8f' % (n_iter, E_ele, E_tot))

        n_iter += 1
    
    if n_iter < max_iters:
        status = 'converged'
    else:
        status = 'diverged'
    
    if verbose:
        print('************************************************************************')
    
    return E_tot, eps, F, P, H, status