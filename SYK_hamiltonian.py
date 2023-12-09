# This code generates one instance of SYK Hamiltonian
# given the number of Majorana fermions and
# Pauli decomposes the Hamiltonian for digital quantum computation. 

# Please cite our article if you find this useful
# Email: raghav.govind.jha @ gmail.com  

import sys
import time
import math
import numpy as np
from functools import reduce
from datetime import datetime
import scipy.io as sio
import warnings
warnings.filterwarnings("ignore")

if len(sys.argv) < 2:
  print("Usage: python",str(sys.argv[0])," 'N' ")
  sys.exit(1)

N = int(sys.argv[1])

if N%2 !=0:
    print ("N has to be even, it is", N)
    sys.exit(1)

start = time.time()
print ("STARTED: ", datetime.now())

def dagger(a):
        return np.transpose(a).conj()

def pretty_print_matrix(matrix):
    print(('\n'.join(['\t'.join([str(cell) for cell in row]) for row in matrix])))


def make_Majorana(N):

    # Make 'N' Majorana fermions. Set of N Hermitian matrices psi_i, i=1,..N
    # obeying anti-commutation relations {psi_i,psi_j} = δ_{ij}
    # Note: Not 2δ_{ij}
    
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    I = np.array([[1, 0], [0, 1]])

    psi = dict()

    for i in range(1, N+1):

        if (i % 2) == 1:
            matlist = [Z] * int((i-1)/2)
            matlist.append(X)
            matlist = matlist + [I] * int((N/2 - (i+1)/2))
            psi[i] = 1/np.sqrt(2)*reduce(np.kron, matlist)
        else:
            matlist = [Z] * int((i - 2) / 2)
            matlist.append(Y)
            matlist = matlist + [I] * int((N/2 - i/2))
            psi[i] = 1/np.sqrt(2)*reduce(np.kron, matlist)


    for i in range(1, N+1):
        for j in range(1, N+1):

            if i != j:
                if np.allclose(psi[i] @ psi[j], -psi[j] @ psi[i]) == False:
                    print ("Does not satisfy algebra for i != j")

            if i == j:
                if np.allclose(psi[i] @ psi[j] + psi[j] @ psi[i], np.eye(int(2**(N/2)))) == False:
                    print ("Does not satisfy algebra for i = j")

    return psi


def make_Hamiltonian(psi, N, instances, J_squared):
    
    # Creates multiple realisations of the SYK Hamiltonian
    # Variance of couplings is given by 'J_squared * 3!/N^3'.

    H = 0
    J = dict()
    J_prob = dict()
    sigma_sq = 6.*J_squared/(N**3) # or q!*J_squared/(N**(q-1)) for general q
    sigma = math.sqrt(sigma_sq)

    for i in range(1, N+1):
        for j in range(i+1, N+1):
            for k in range(j+1, N+1):
                for l in range(k+1, N+1):

                    J[i, j, k, l] = np.random.normal(loc=0, scale=sigma,size=1)
                    M = psi[i] @ psi[j] @ psi[k] @ psi[l]
                    H = H + np.array([element * M for element in J[i, j, k, l]])


    if np.allclose(dagger(H[0]), H[0]) == False: 
        print ("Hamiltonian is not Hermitian")
         
    return H[0] 


def numberToBase(n, b, n_qubits):
    if n == 0:
        return np.zeros(n_qubits,dtype=int)
    digits = np.zeros(n_qubits,dtype=int)
    counter=0
    while n:
        digits[counter]=int(n % b)
        n //= b
        counter+=1
    return digits[::-1]

def decomposePauli(H):
    
    sx = np.array([[0, 1],  [ 1, 0]], dtype=np.complex128)
    sy = np.array([[0, -1j],[1j, 0]], dtype=np.complex128)
    sz = np.array([[1, 0],  [0, -1]], dtype=np.complex128)
    id = np.array([[1, 0],  [ 0, 1]], dtype=np.complex128)

    labels = ['I', "X", "Y", "Z"]
    S = [id, sx, sy, sz]
    dim_matrix=np.shape(H)[0]
    n_qubits=int(np.log2(dim_matrix))
    if(dim_matrix!=2**n_qubits):
        raise NameError("Matrix is not a power of 2!")
    hilbertspace=2**n_qubits
    n_paulis=4**n_qubits
    pauli_list=np.zeros([n_paulis,n_qubits],dtype=int)
    for k in range(n_paulis):
        pauli_list[k,:]=numberToBase(k,4,n_qubits)
    weights=np.zeros(n_paulis,dtype=np.complex128)
    for k in range(n_paulis):
        pauli=S[pauli_list[k][0]]

        for n in range(1,n_qubits):
            pauli=np.kron(pauli,S[pauli_list[k][n]])
        weights[k] = 1/hilbertspace* np.dot(pauli,H).trace()

    nnz = [i for i, e in enumerate(weights) if e != 0]

    for i in nnz:

        if abs(weights[i]) > 1e-14:
            print (pauli_list[i],",", weights[i].real) 
            

ferm = make_Majorana(N)
H = make_Hamiltonian(ferm, N, 1, 1.)

file_write = 'H_N' + str(N) + '.mtx' 
sio.mmwrite(file_write,H)

# Decompose the constructed H in terms of Paulis. 
decomposePauli(H)

print ("FINISHED: ", datetime.now())
end = time.time()
print("Run time (in seconds):", round(end-start,2))