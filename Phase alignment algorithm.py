import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
plt.close("all")
N = 200

"""Fonctions de construction des matrices circulantes antisymétriques"""
def J(j, n = N):
    if j%n == 0:
        return sps.eye(n, dtype = float, format = "csc")
    return sps.diags([np.ones(n-j%n), np.ones(j%n)], [j%n, -n+j%n], dtype = float, format = "csc")

def A(th_arr, index, n = N):
    A_th = 0 * sps.eye(n, dtype = float, format = "csc")
    if index is None:
        for i, theta in enumerate(th_arr):
            A_th += (J(i+1, n) - J(-i-1, n)) * theta
    else:
        for i, j in enumerate(index):
            A_th += (J(j+1, n) - J(-j-1, n)) * th_arr[i]
    return A_th
#Exemple
#print(A([2,3],n = 7).toarray())

"""Choix du support de minimisation"""
x = np.linspace(0,1,N)

param = 0.3
g = (1 + np.tanh((param - x)*30))/2
x_zone = np.linspace(0,param, 30)
def F_target(t):
    return np.exp(-100 * (t-0.3)**2)
F = F_target(x)
G = sps.diags([g], [0], format = "csc")

plt.figure(0)
plt.title("objectif : approcher la F sur une zone en imposant TFD en module")
plt.plot(x, g, "k--", label = "zone")
plt.plot(x, F, label = "F")
plt.legend()
plt.show()
plt.pause(1)

"""Choix de la fonction d'étude"""
V = np.copy(np.exp(-2000 * (x-0.2)**2))*4
V0 = np.copy(V)
"""Définitions des fonctionnelles"""
Id = sps.eye(N, dtype = float, format = "csc")

def full_f(th, V, index = None):
        return sps.linalg.spsolve((Id - A(th, index)), (Id + A(th, index)).dot(V))

def f(th, V, index = None):
    return G.dot(sps.linalg.spsolve((Id - A(th, index)), (Id + A(th, index)).dot(V) )- F)

def df(th, h, V, index = None):
    A_th = A(th, index)
    return G.dot(A(h, index).dot(sps.linalg.spsolve(Id - A_th, V) + sps.linalg.spsolve((Id - A_th).dot((Id - A_th)), (Id + A_th).dot(V))))

def Hf(th, V, index = None):
    A_th = A(th, index)
    M = sps.linalg.spsolve(Id - A_th, V) + sps.linalg.spsolve((Id - A_th).dot((Id - A_th)), (Id + A_th).dot(V))
    return np.array([G.dot(A((np.arange(len(th))==i).astype(float), index).dot(M)) for i in range(len(th))])
    
"""Verification de la bonne définition de la différentielle df"""
th_arr = np.random.random(3)
index = [1,2,5]
h = np.random.random(3)
err = []
scale = np.logspace(-1, -3, 10)
for s in scale:
    err.append(np.linalg.norm(f(th_arr + h*s, V, index) - f(th_arr, V, index) - df(th_arr, h*s, V, index)))
    
plt.figure(1)
plt.title("l'erreur est bien d'ordre 2")
plt.loglog(scale, err,"ro-", label = "||f(x+h) - f(x) - df(x).h||")
plt.plot(scale, scale**2, "k--", label = "||h||²")
plt.legend()
plt.show()

"""Initialisation de l'algorithme de descente de gradient"""

n_th = 2
max_n_th = 5
index = None
last_res = 1
th = np.zeros(n_th, dtype = float)
r0 = 1e-1
slowness = 0
running = True
j = 0
while running:
    j+=1
    jac = Hf(th, V, index)
    value = f(th, V, index)
    res = np.linalg.norm(value)
    if res < last_res:
        if res > last_res*(1-1e-4):
            if slowness > 3:
                if n_th >= max_n_th:
                    running = False
                else:
                    print("expansion")
                    n_th +=1
                    th = np.hstack((th, 0))*0
                    jac = Hf(th, V, index)
                    slowness = 0
            else:
                slowness +=1
        else:
            slowness = 0
        r0 *= 0.9
        
    else:
        r0 *= 1.1
    last_res = res
        
    #th -= r0 * jac.dot(value) # gradient descent
    th -= np.linalg.solve(jac.dot(jac.T) + r0 *  np.eye(n_th), jac.dot(value)) + np.exp(-j/10)*np.random.random(n_th) #mix simulated Annealing and Levenberg-Marquardt methods.
    
    if j%10==0:
        print(np.linalg.norm(value))
        plt.figure(2)
        plt.clf()
        plt.plot(x, V0,"b", label = "Initial signal")
        plt.plot(x, full_f(th, V, index), "r", label = "Phase-aligned signal")
        plt.plot(x_zone, F_target(x_zone), 'k--', label = "Target")
        plt.legend()
        plt.show()
        plt.pause(0.01)
    if j%10==0:
        V = full_f(th, V, index)
        th*=0
        







    