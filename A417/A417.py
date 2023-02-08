import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

N=30 
n = 3 
A = np.array([[-1,0.4,0.8],[1,0,0],[0,1,0]]) 
b = np.array([1,0,0.3])
x_des = np.array([7,2,-6]) 
x0 = np.zeros((n,)) 


#define x variable 
#define u variable 
x = cp.Variable((N, n))
u = cp.Variable(N)
 
obj = 0
for t in range(N):
    obj += cp.maximum(cp.abs(u[t]), 2 * cp.abs(u[t]) - 1)
obj = cp.Minimize(obj)

constraints = [ x[i+1] == (A @ (x[i])) + b*u[i] for i in range(N-1) ]
constraints.append(x[0] == x0)
constraints.append(x[N-1] == x_des)
prob = cp.Problem(obj, constraints) 
prob.solve() 

plt.figure() 
plt.step(range(N),u.value) 
plt.title('$u(t)$ vs time') 
plt.xlabel('$t$') 
plt.ylabel('$u(t)$')
plt.show()