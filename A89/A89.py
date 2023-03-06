import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


P1=np.array([[1,0,0,0],[0,1,0,0], [0, 0,1, 0]])
P2=np.array([[1,0,0,0], [0,0,1,0], [0,-1,0,10]])
P3=np.array([[1,1,1,-10], [-1,1,1,0], [-1,-1,1,10]])
P4=np.array([[0, 1, 1, 0], [0, -1, 1, 0], [-1, 0, 0, 10]])

y1 = np.array([0.98, 0.93])
y2 = np.array([1.01, 1.01])
y3 = np.array([0.95, 1.05])
y4 = np.array([2.04, 0.00])

def temptest(P, x, y):
    A = P[:-1, :-1]
    c = P[-1, :-1]
    b = P[:-1, -1]
    d = P[-1, -1]
    return A @ x + d + (c @ x + b)*y

def temp2(P, x):
    A = P[:-1, :-1]
    c = P[-1, :-1]
    b = P[:-1, -1]
    d = P[-1, -1]
    return c @ x + d

def fk(P, x):
    # c is 1x3
    # A is 2x3
    A = P[:-1, :-1]
    c = P[-1, :-1]
    b = P[:-1, -1]
    d = P[-1, -1]
    fkx = (A @ x + b)/(c @ x + d)
    return fkx

def fk_norm(P, x, y):
    return np.linalg.norm(fk(P,x) - y)

def max_fx(x):
    constraints = np.array([fk_norm(P1, x, y1), fk_norm(P2, x, y2), fk_norm(P3, x, y3), fk_norm(P4, x, y4)])
    print(constraints)
    return np.amax(constraints)

l = 0
u = 10
tol = 1e-4 
while u-l > tol: 
  t = (l+u)/2 
  x = cp.Variable(3)
  obj = cp.Minimize(1)

  constraints = [  
            cp.atoms.norm2(temptest(P1, x, y1)) - t*temp2(P1,x) <= 0,
            cp.atoms.norm2(temptest(P2, x, y2)) - t*temp2(P2,x) <= 0,
            cp.atoms.norm2(temptest(P3, x, y3)) - t*temp2(P3,x) <= 0,
            cp.atoms.norm2(temptest(P4, x, y4)) - t*temp2(P4,x) <= 0
                ] 
  prob = cp.Problem(obj, constraints) 
  #assert prob.is_dqcp() 
  prob.solve()
 
  if prob.value == float('Inf'): 
    l = t
    print('infeasible', l, u)
  else: 
    lastx = x.value
    u = t
    print('feasible', l, u)
    print(lastx)
print(lastx)
max_fx(lastx)