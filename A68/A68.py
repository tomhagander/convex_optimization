import numpy as np
import cvxpy as cp
from spline_data import t, y
from bsplines import bsplines
import matplotlib.pyplot as plt

A = np.zeros((len(y), 13))
G = np.zeros((len(y), 13))
b = y.reshape(-1,1)

for i in range(len(t)):
    g, gp, gpp = bsplines(t[i])
    A[i] = g
    G[i] = gpp

x = cp.Variable(13)
# Define the objective function
obj = cp.Minimize(cp.norm(A @ x - y, 2)**2)
# Define the constraints
constraints = [-G @ x <= 0]
# Solve the optimization problem
prob = cp.Problem(obj, constraints)

result = prob.solve()
x0 = np.array(x.value).reshape(-1,1)
print(x0)
#LS estimation without the convex constraint
x_ls,_,_,_ = np.linalg.lstsq(A.astype('float'),b.astype('float'))
print(x_ls)
#Plot the results
h = A @ x0
plt.figure()
plt.scatter(t,y,color='g')
plt.plot(t,h)
plt.plot(t,A @ x_ls)
legends = ['convex estimation','Least squares estimation','actual data'] 
plt.legend(legends)
plt.xlabel("$t$")
plt.ylabel("$f(t)$")
plt.title('estimation evaluation')
plt.show()

