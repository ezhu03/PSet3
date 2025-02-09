import numpy as np
from numpy.linalg import inv, det
from scipy.integrate import nquad

def gaussian_integrand(*v, A, w):
    """
    The integrand f(v) = exp( -1/2 v^T A v + v^T w ),
    where v is treated as an N-tuple, A is an (N x N) matrix, and w is an N-vector.
    """
    v = np.array(v)
    exponent = -0.5 * v.T @ A @ v + np.dot(v.T,w)
    return np.exp(exponent)

def numerical_gaussian_integral(A, w, L=5.0):
    """
    Numerically approximate the integral of exp(-1/2 v^T A v + v^T w) 
    over v in R^N, by truncating each dimension to [-L, L].
    """
    N = len(w)
    # Create the bounds for each dimension: [(-L, L), (-L, L), ... ]
    bounds = [(-L, L)] * N
    
    integrand = lambda *coords: gaussian_integrand(*coords, A=A, w=w)
    
    val, err = nquad(integrand, bounds)
    return val, err

def closed_form_gaussian_integral(A, w):
    """
    Returns the closed-form value of the integral:
      I = sqrt(det(A^-1)/(2 pi)^N) * exp((1/2) w^T A^-1 w).

    Requires A to be symmetric, positive-definite. Otherwise, the integral 
    is not well-defined (the expression would be infinite or undefined).
    """
    N = len(w)
    A_inv = inv(A)
    # det(A^-1) = 1 / det(A), so sqrt(det(A^-1)) = 1 / sqrt(det(A))
    # or equivalently np.sqrt(det(A_inv)) = 1/np.sqrt(det(A))
    prefactor = np.sqrt(((2 * np.pi) ** N)/det(A))
    exponent_part = 0.5 * np.dot(np.dot(w.T, A_inv),w)
    return prefactor * np.exp(exponent_part)

def verify_gaussian_integral(A, w, L=5.0):
    """
    Compare the numerical integral with the closed-form expression,
    but only if A is positive-definite (all eigenvalues > 0).
    """
    # First check if A is positive-definite
    eigvals = np.linalg.eigvals(A)
    if np.any(eigvals <= 0):
        print("Matrix A is not positive-definite => Gaussian integral diverges.")
        print("No finite result or closed-form expression possible.\n")
        return
    
    # If we get here, A should be positive-definite
    num_val, num_err = numerical_gaussian_integral(A, w, L=L)
    ana_val = closed_form_gaussian_integral(A, w)
    
    print(f"Numerical integral  = {num_val:.6g} +/- {num_err:.2g}")
    print(f"Closed-form result  = {ana_val:.6g}")
    print(f"Ratio (num/closed) = {num_val / ana_val:.6g}")
    print()
    return num_val

# --------------------------------------------------------------------------

    # Symmetric positive-definite matrix
A = np.array([
        [4, 2, 1],
        [2, 5, 3],
        [1, 3, 6]
])

    # Indefinite (or not positive-definite) matrix
A_prime = np.array([
        [4, 2, 1],
        [2, 1, 3],
        [1, 3, 6]
])

w = np.array([1, 2, 3])

print("Testing with A (positive-definite):")
Z = verify_gaussian_integral(A, w, L=5)

print("Testing with A' (indefinite):")
verify_gaussian_integral(A_prime, w, L=5)

print("Calculating moments with A:")
def moment_integrand(*v, A, w, powers):
    v = np.array(v)
    exponent = -0.5 * v @ A @ v + v @ w
    # e.g. if powers=[2,0,1], we return v[0]^2 * v[2]^1, etc.
    factor = np.prod([v[i]**p for i,p in enumerate(powers)])
    return factor * np.exp(exponent)

# <v1> = int v1 * p(v) dv ================================================
def calculate_moment(A, w, powers, L=5):
    N = len(w)
        # Create the bounds for each dimension: [(-L, L), (-L, L), ... ]
    bounds = [(-L, L)] * N
    val_raw, err_raw = nquad(
        lambda *coords: moment_integrand(*coords, A=A, w=w, powers=powers),
        bounds
    )
    return val_raw
mu = np.dot(inv(A),w)
with open("results.txt", "w") as f:
    f.write("Analytical first moment: mu_i = A^-1 w_i \n")
    f.write('\n')
    moment_v1 = calculate_moment(A,w,[1,0,0],5) / Z
    f.write(f"Numerical first moment <v1> = {moment_v1:.6g} ")
    f.write('\n')
    f.write(f"Analytical first moment <v1> = {mu[0]:.6g}")
    f.write('\n')
    moment_v2 = calculate_moment(A,w,[0,1,0],5) / Z
    f.write(f"Numerical first moment <v2> = {moment_v2:.6g} ")
    f.write('\n')
    f.write(f"Analytical first moment <v2> = {mu[1]:.6g}")
    f.write('\n')
    moment_v3 = calculate_moment(A,w,[0,0,1],5) / Z
    f.write(f"Numerical first moment <v3> = {moment_v3:.6g} ")
    f.write('\n')
    f.write(f"Analytical first moment <v3> = {mu[2]:.6g} ")
    f.write('\n')
    f.write('\n')
    f.write("Analytical second moment:  A-1_ij + mu_i mu_j \n")
    f.write('\n')
    moment_v1v2 = calculate_moment(A,w,[1,1,0],5) / Z
    f.write(f"Numerical second moment <v1v2> = {moment_v1v2:.6g}")
    f.write('\n')
    f.write(f"Analytical second moment <v1v2> = {inv(A)[0,1]+mu[0]*mu[1]:.6g}")
    f.write('\n')
    moment_v1v3 = calculate_moment(A,w,[1,0,1],5) / Z
    f.write(f"Numerical second moment <v1v3> = {moment_v1v3:.6g}")
    f.write('\n')
    f.write(f"Analytical second moment <v1v3> = {inv(A)[0,2]+mu[0]*mu[2]:.6g}")
    f.write('\n')
    moment_v2v3 = calculate_moment(A,w,[0,1,1],5) / Z
    f.write(f"Numerical second moment <v2v3> = {moment_v2v3:.6g}" )
    f.write('\n')
    f.write(f"Analytical second moment <v2v3> = {inv(A)[1,2]+mu[1]*mu[2]:.6g}")
    f.write('\n')
    f.write('\n')
    f.write("Higher order moments, (mu_i + x_i)^a (mu_j + x_j)^b (mu_k + x_k)^c \n")
    moment_v12v2 = calculate_moment(A,w,[2,1,0],5) / Z
    f.write(f"Numerical third moment <v1^2v2> = {moment_v12v2:.6g}")
    f.write('\n')
    f.write(f"Analytical third moment <v1^2v2> = {mu[0]*mu[1]**2+mu[0]*inv(A)[1,1]+2*mu[1]*inv(A)[0,1]:.6g} ")
    f.write('\n')
    moment_v2v32 = calculate_moment(A,w,[0,1,2],5) / Z
    f.write(f"Numerical third moment <v2v3^2> = {moment_v2v32:.6g}")
    f.write('\n')
    f.write(f"Analytical third moment <v2v3^2> = {mu[1]*mu[2]**2+mu[1]*inv(A)[2,2]+2*mu[2]*inv(A)[1,2]:.6g} ")
    f.write('\n')
    moment_v12v22 = calculate_moment(A,w,[2,2,0],5) / Z
    f.write(f"Numerical fourth moment <v1^2v2^2> = {moment_v12v22:.6g}")
    f.write('\n')
    f.write(f"Analytical fourth moment <v1^2v2^2> = {inv(A)[0,0]*inv(A)[1,1]+2*inv(A)[0,1]**2+inv(A)[1,1]*mu[0]**2+inv(A)[0,0]*mu[1]**2+4*mu[0]*mu[1]*inv(A)[0,1] +(mu[0]**2)*(mu[1]**2):.6g}")
    f.write('\n')
    moment_v22v33 = calculate_moment(A,w,[0,2,2],5) / Z
    f.write(f"Numerical fourth moment <v2^2v3^2> = {moment_v22v33:.6g}")
    f.write('\n')
    f.write(f"Analytical fourth moment <v2^2v3^2> = {inv(A)[1,1]*inv(A)[2,2]+2*inv(A)[1,2]**2+inv(A)[2,2]*mu[1]**2+inv(A)[1,1]*mu[2]**2+4*mu[1]*mu[2]*inv(A)[1,2] +(mu[1]**2)*(mu[2]**2):.6g}")
    f.write('\n')
print("Results written to results.txt")