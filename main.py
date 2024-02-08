#%%
import numpy as np
from scipy.optimize import minimize

# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile, Aer, BasicAer
from qiskit import IBMQ, execute
from qiskit.compiler import transpile
# from qiskit.tools.jupyter import *
from qiskit.visualization import *
from qiskit.providers.aer import QasmSimulator
from qiskit.circuit.library import MCXGate
#%%
import os
from pathlib import Path
from functools import partial

# set julia project path to current dir
os.environ["PYTHON_JULIAPKG_PROJECT"] = str(Path(__file__).absolute().parents[0])

from juliacall import Main as jl

# for development I have Revice in my global env
jl.seval("using Revise")

# load Compromise into global namespace
jl.seval("import Compromise as C")
#%%
jl.seval("struct PyFunction <: Function func::Py end")
jl.seval("as_vec(x::Number)=[x,]")
jl.seval("as_vec(x) = x")
jl.seval("(pf::PyFunction)(args...)=as_vec(pyconvert(Float64, pf.func(args...)))")
    
#%%

### Ansatz: Create U(lambda)
def Get_U_lambda(Q_Amount, vec_lambda):
    qc_U = QuantumCircuit(Q_Amount)
    index = 0
    for i in range(Q_Amount):
        qc_U.ry(vec_lambda[index],i)
        index += 1
    ### erste Reihe #####
    qc_U.cx(Q_Amount-1, Q_Amount-2)
    qc_U.ry(vec_lambda[index],Q_Amount-2)
    index +=1
    qc_U.cx(Q_Amount-1, Q_Amount-2)
    ### zweite Reihe ### only for 3+ qubits
    if(Q_Amount < 3):
        return qc_U
    qc_U.cx(Q_Amount-2, Q_Amount-3)
    qc_U.ry(vec_lambda[index],Q_Amount-3)
    index +=1
    qc_U.cx(Q_Amount-1, Q_Amount-3)
    qc_U.ry(vec_lambda[index],Q_Amount-3)
    index +=1
    qc_U.cx(Q_Amount-2, Q_Amount-3)
    qc_U.ry(vec_lambda[index],Q_Amount-3)
    index +=1
    qc_U.cx(Q_Amount-1, Q_Amount-3)
    ### dritte Reihe ### nur für 4+ qubits
    if (Q_Amount < 4):
        return qc_U
    Cind4_arr=np.array([Q_Amount-3,Q_Amount-2,Q_Amount-3,Q_Amount-1,Q_Amount-3,Q_Amount-2,Q_Amount-3])
    for i in range(0,7):
        qc_U.cx(Cind4_arr[i], Q_Amount-4)
        qc_U.ry(vec_lambda[index],Q_Amount-4)
        index +=1
    qc_U.cx(Q_Amount-1, Q_Amount-4)
    if (Q_Amount < 5):
        return qc_U
    Cind5_arr=np.array([Q_Amount-4,Q_Amount-3,Q_Amount-4,Q_Amount-2,Q_Amount-4,Q_Amount-3,Q_Amount-4, Q_Amount-1,\
                        Q_Amount-4,Q_Amount-3,Q_Amount-4,Q_Amount-2,Q_Amount-4,Q_Amount-3,Q_Amount-4])
    for i in range(0,15):
        qc_U.cx(Cind5_arr[i], Q_Amount-5)
        qc_U.ry(vec_lambda[index],Q_Amount-5)
        index +=1
    qc_U.cx(Q_Amount-1, Q_Amount-5)
    if (Q_Amount < 6):
        return qc_U
    Cind6_arr=np.array([Q_Amount-5,Q_Amount-4,Q_Amount-5,Q_Amount-3,Q_Amount-5,Q_Amount-4,Q_Amount-5, Q_Amount-2,\
                        Q_Amount-5,Q_Amount-4,Q_Amount-5,Q_Amount-3,Q_Amount-5,Q_Amount-4,Q_Amount-5, Q_Amount-1,\
                        Q_Amount-5,Q_Amount-4,Q_Amount-5,Q_Amount-3,Q_Amount-5,Q_Amount-4,Q_Amount-5, Q_Amount-2,\
                        Q_Amount-5,Q_Amount-4,Q_Amount-5,Q_Amount-3,Q_Amount-5,Q_Amount-4,Q_Amount-5])
    for i in range(0,31):
        qc_U.cx(Cind6_arr[i], Q_Amount-6)
        qc_U.ry(vec_lambda[index],Q_Amount-6)
        index +=1
    qc_U.cx(Q_Amount-1, Q_Amount-6)
    return qc_U

### Calculate Costs which are independend of lambda
def Get_Cost_Contribution(vec_fk_input, Qubits):
    N = 2**Qubits #States for calculation
    vec_fk = vec_fk_input/ np.sqrt(np.sum(abs(vec_fk_input)**2)) #normiert
    # vec_fk = np.sqrt(vec_fk_input/ np.sum(abs(vec_fk_input))) #normiert
    ### Implement U(Psi_t)
    qc_fk = QuantumCircuit(Qubits)
    qc_fk.initialize(vec_fk, np.arange(0,Qubits).tolist())
    result = transpile(qc_fk, basis_gates=['u3', 'cx'], optimization_level= 1)
    Psi = result.to_gate().control()
    Psi_dagger = result.to_gate().inverse().control()
    ###Implement Shiftplus############################
    qc1 = QuantumCircuit(Qubits+1)
    qc1.h(0)
    qc1.append(Psi, np.arange(0,Qubits+1).tolist())
    for m in range(0,Qubits):
        turn = Qubits-m
        gate = MCXGate(turn)
        qc1.append(gate, np.arange(0,turn+1).tolist())
    qc1.append(Psi_dagger, np.arange(0,Qubits+1).tolist())
    qc1.h(0)
    backend_2 = BasicAer.get_backend('statevector_simulator')
    job = execute(qc1, backend_2)#für plot histogram, shots = 2**17)
    counts1 = job.result().get_statevector()    
    p1_array = np.zeros(2**(Qubits+1))
    for k in range(1,2**(Qubits+1),2):
        p1_array[k] = abs(counts1[k])**2
    p1 = np.sum(p1_array)
    p0_array = np.zeros(2**(Qubits+1))
    for k in range(0,2**(Qubits+1),2):
        p0_array[k] = abs(counts1[k])**2
    p0 = np.sum(p0_array)
    C3_Shiftplus = (p0 - p1)
    ###Implement Shiftminus############################
    qc2 = QuantumCircuit(Qubits+1)
    qc2.h(0)
    qc2.append(Psi, np.arange(0,Qubits+1).tolist())
    for m in range(0,Qubits):
        gate = MCXGate(m+1)
        qc2.append(gate, np.arange(0,m+2).tolist())
    qc2.append(Psi_dagger, np.arange(0,Qubits+1).tolist())
    qc2.h(0)
    backend_2 = BasicAer.get_backend('statevector_simulator')
    job2 = execute(qc2, backend_2)#für plot histogram, shots = 2**17)
    counts2 = job2.result().get_statevector()    
    p1_array = np.zeros(2**(Qubits+1))
    for k in range(1,2**(Qubits+1),2):
        p1_array[k] = abs(counts2[k])**2
    p1 = np.sum(p1_array)
    p0_array = np.zeros(2**(Qubits+1))
    for k in range(0,2**(Qubits+1),2):
        p0_array[k] = abs(counts2[k])**2
    p0 = np.sum(p0_array)
    C3_Shiftminus = (p0 - p1)
    ###Implement Shiftplusplus############################
    qc3 = QuantumCircuit(Qubits+1)
    qc3.h(0)
    qc3.append(Psi, np.arange(0,Qubits+1).tolist())
    for m in range(0,Qubits):
        turn = Qubits-m
        gate = MCXGate(turn)
        qc3.append(gate, np.arange(0,turn+1).tolist())
    for m in range(0,Qubits):
        turn = Qubits-m
        gate = MCXGate(turn)
        qc3.append(gate, np.arange(0,turn+1).tolist())
    qc3.append(Psi_dagger, np.arange(0,Qubits+1).tolist())
    qc3.h(0)
    backend_2 = BasicAer.get_backend('statevector_simulator')
    job3 = execute(qc3, backend_2)#für plot histogram, shots = 2**17)
    counts3 = job3.result().get_statevector()    
    p1_array = np.zeros(2**(Qubits+1))
    for k in range(1,2**(Qubits+1),2):
        p1_array[k] = abs(counts3[k])**2
    p1 = np.sum(p1_array)
    p0_array = np.zeros(2**(Qubits+1))
    for k in range(0,2**(Qubits+1),2):
        p0_array[k] = abs(counts3[k])**2
    p0 = np.sum(p0_array)
    C4_Shiftplusplus = (p0 - p1)
    ###Implement Shiftminusminus############################
    qc4 = QuantumCircuit(Qubits+1)
    qc4.h(0)
    qc4.append(Psi, np.arange(0,Qubits+1).tolist())
    for m in range(0,Qubits):
        gate = MCXGate(m+1)
        qc4.append(gate, np.arange(0,m+2).tolist())
    for m in range(0,Qubits):
        gate = MCXGate(m+1)
        qc4.append(gate, np.arange(0,m+2).tolist())
    qc4.append(Psi_dagger, np.arange(0,Qubits+1).tolist())
    qc4.h(0)
    backend_2 = BasicAer.get_backend('statevector_simulator')
    job4 = execute(qc4, backend_2)#für plot histogram, shots = 2**17)
    counts4 = job4.result().get_statevector()    
    p1_array = np.zeros(2**(Qubits+1))
    for k in range(1,2**(Qubits+1),2):
        p1_array[k] = abs(counts4[k])**2
    p1 = np.sum(p1_array)
    p0_array = np.zeros(2**(Qubits+1))
    for k in range(0,2**(Qubits+1),2):
        p0_array[k] = abs(counts4[k])**2
    p0 = np.sum(p0_array)
    C4_Shiftminusminus = (p0 - p1)
    
    return C3_Shiftplus, C4_Shiftplusplus, C3_Shiftminus, C4_Shiftminusminus


### Calculate Costs
def Get_Costs(vec_lambda, vec_fk_input, Qubits, D, tau, u, C3_Shiftplus, C4_Shiftplusplus, C3_Shiftminus, C4_Shiftminusminus):
    N = 2**Qubits #States for calculation
    ### Implement help circuit for Udagger ------------------------------------
    vec_fk = vec_fk_input/ np.sqrt(np.sum(abs(vec_fk_input)**2)) #normiert
    qc_fk = QuantumCircuit(Qubits)
    qc_fk.initialize(vec_fk, np.arange(0,Qubits).tolist())
    result = transpile(qc_fk, basis_gates=['u3', 'cx'], optimization_level= 1)
    Udagger = result.to_gate().inverse().control()
    ### Implement U(lambda) ---------------------------------------------------
    qc_U = Get_U_lambda(Qubits, vec_lambda[1:])
    U_lambda = qc_U.to_gate()    # qc_U.to_gate() 
    U_lambda_c = U_lambda.control()
    ### Implement S+ Circuit -------------------------------------------------- 
    qc1 = QuantumCircuit(Qubits+1)
    qc1.h(0)
    qc1.append(U_lambda_c, np.arange(0,Qubits+1).tolist())
    for m in range(0,Qubits):
        turn = Qubits-m
        gate = MCXGate(turn)
        qc1.append(gate, np.arange(0,turn+1).tolist())
    qc1.append(Udagger, np.arange(0,Qubits+1).tolist())
    qc1.h(0)
    ### Simulate circuit
    backend_2 = BasicAer.get_backend('statevector_simulator')
    job = execute(qc1, backend_2)#für plot histogram, shots = 2**17)
    counts1 = job.result().get_statevector()    
    p1_array = np.zeros(2**(Qubits+1))
    for k in range(1,2**(Qubits+1),2):
        p1_array[k] = abs(counts1[k])**2
    p1 = np.sum(p1_array)
    p0_array = np.zeros(2**(Qubits+1))
    for k in range(0,2**(Qubits+1),2):
        p0_array[k] = abs(counts1[k])**2
    p0 = np.sum(p0_array)
    C2_shiftplus = (p0 - p1) #measure z-basis
    ### Implement S- Circuit --------------------------------------------------
    qc3 = QuantumCircuit(Qubits+1)
    qc3.h(0)
    qc3.append(U_lambda_c, np.arange(0,Qubits+1).tolist())
    for m in range(0,Qubits):
        gate = MCXGate(m+1)
        qc3.append(gate, np.arange(0,m+2).tolist())
    qc3.append(Udagger, np.arange(0,Qubits+1).tolist())
    qc3.h(0)
    ### Simulate circuit
    job3 = execute(qc3, backend_2)#für plot histogram, shots = 2**17)
    counts3 = job3.result().get_statevector()    
    p1_array = np.zeros(2**(Qubits+1))
    for k in range(1,2**(Qubits+1),2):
        p1_array[k] = abs(counts3[k])**2
    p1 = np.sum(p1_array)
    p0_array = np.zeros(2**(Qubits+1))
    for k in range(0,2**(Qubits+1),2):
        p0_array[k] = abs(counts3[k])**2
    p0 = np.sum(p0_array)
    C2_shiftminus = (p0 - p1) #measure z-basis
    ### Implement identity circuit --------------------------------------------
    qc2 = QuantumCircuit(Qubits+1,1)
    qc2.h(0)
    qc2.append(U_lambda_c, np.arange(0,Qubits+1).tolist())
    qc2.append(Udagger, np.arange(0,Qubits+1).tolist())
    qc2.h(0)
    ### Simulate circuit
    job2 = execute(qc2, backend_2)#für plot histogram, shots = 2**17)
    counts2 = job2.result().get_statevector()    
    p1_array = np.zeros(2**(Qubits+1))
    for k in range(1,2**(Qubits+1),2):
        p1_array[k] = abs(counts2[k])**2
    p1 = np.sum(p1_array)
    p0_array = np.zeros(2**(Qubits+1))
    for k in range(0,2**(Qubits+1),2):
        p0_array[k] = abs(counts2[k])**2
    p0 = np.sum(p0_array)
    C1 = (p0 - p1) #measure z-basis
    ### Calculate costs -------------------------------------------------------
    ### Calculate a,b,c
    lam0T = np.sqrt(np.sum(abs(vec_fk_input)**2))
    a = tau * D * N**2 - tau * u * N/2
    b = 2 * tau * D * N**2
    c = tau * D * N**2 + tau * u * N/2
    ### Cost function
    C = vec_lambda[0]**2 - 2 * vec_lambda[0] * lam0T * ((1-b)*C1 + a*C2_shiftplus + c*C2_shiftminus) \
        + lam0T**2 *(1 + 2*(a*C3_Shiftplus - b + c*C3_Shiftminus) + a**2 + b**2 + c**2 \
                     - a*b*C3_Shiftminus + a*c*C4_Shiftminusminus - b*a*C3_Shiftplus - b*c*C3_Shiftminus \
                     + c*a*C4_Shiftplusplus - c*b*C3_Shiftplus)
    return C

### Optimization method: Nelder-Mead
def Optimize(vec_fk, Qubits, D, tau, u, vec0, C3_Shiftplus, C4_Shiftplusplus, C3_Shiftminus, C4_Shiftminusminus):
    sol = minimize(Get_Costs, vec0, args= (vec_fk, Qubits, D, tau, u, C3_Shiftplus, C4_Shiftplusplus, C3_Shiftminus, C4_Shiftminusminus), method = 'Nelder-Mead')
    values = sol.x
    return values

def jl_opt(vec_fk, Qubits, D, tau, u, vec0, C3_Shiftplus, C4_Shiftplusplus, C3_Shiftminus, C4_Shiftminusminus):
    num_vars = vec_fk.size
    mop = jl.C.MutableMOP(num_vars=num_vars)
    objective_func = jl.PyFunction(lambda x : Get_Costs(x, vec_fk, Qubits, D, tau, u, C3_Shiftplus, C4_Shiftplusplus, C3_Shiftminus, C4_Shiftminusminus))
    jl.C.add_objectives_b(mop, objective_func, jl.Symbol("rbf"), dim_out=1, func_iip=False)
    algo_opts = jl.C.AlgorithmOptions(stop_max_crit_loops=20)
    final_vals, ret_code = jl.C.optimize(mop, vec0, algo_opts=algo_opts)
    return final_vals.x

### Return result state for a given lambda
def Get_result(Qubits, vec_lambda):
    qc_U = Get_U_lambda(Qubits, vec_lambda[1:])
    backend_2 = BasicAer.get_backend('statevector_simulator')
    job = execute(qc_U, backend_2)
    counts2 = job.result().get_statevector()    
    res_array = np.zeros(2**(Qubits))
    for k in range(0,2**(Qubits)):
        res_array[k] = abs(counts2[k])
    return res_array

#%%
### User Input ----------------------------------------------------------------
T = 2 #Number of time steps
D = 1 #Diffusion constant
u = 10 #velocity
### Activate for N=8
#Q = 2 #Qubit amount
#tau = 0.008 #time constant
### Activate for N=16
Q = 4 #Qubit amount
tau = 0.001 #time constant
### Activate for N=32
# Q = 5 #Qubit amount
# tau = 0.00024 #time constant
### End Input -----------------------------------------------------------------

N = 2**Q
N2= int(N/2)
### Initialize arrays
L_qc = np.zeros([N,T]) #result of for concentration
L_qc[N2,0] = 1 #initialize peak as initial condition
C1 = np.zeros(T) #for costs
Lam=np.zeros([2**(Q),T]) #for all optimal lambdas
vec_lam_alt = np.ones(N) #initialization for optimization

#%%
### Run loop
# import time
# for i in range(1,T):
i = 1

# start = time.time()
C3_Shiftplus, C4_Shiftplusplus, C3_Shiftminus, C4_Shiftminusminus = Get_Cost_Contribution(L_qc[:,i-1], Q)
    #vec_lam = Optimize(L_qc[:,i-1], Q, D, tau, u, vec_lam_alt, C3_Shiftplus, C4_Shiftplusplus, C3_Shiftminus, C4_Shiftminusminus)
vec_lam = jl_opt(L_qc[:,i-1], Q, D, tau, u, vec_lam_alt, C3_Shiftplus, C4_Shiftplusplus, C3_Shiftminus, C4_Shiftminusminus)
    # ende = time.time()
vec_lam_alt = vec_lam
Lam[:,i] = vec_lam
L_qc[:,i] = Get_result(Q, vec_lam)*vec_lam[0]
C1[i-1] = Get_Costs(vec_lam, L_qc[:,i-1], Q, D, tau, u, C3_Shiftplus, C4_Shiftplusplus, C3_Shiftminus, C4_Shiftminusminus)   
#    np.savetxt(str(Q)+'qu_Adv-Diff_conc.txt', L_qc) #activate to return result file
#    np.savetxt(str(Q)+'qu_Adv-Diff_Cost.txt', C1) #activate to return result file
# %%
