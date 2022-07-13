import numpy as np
from sklearn.metrics import mean_absolute_error
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx

def PSO_UQ(y_pred,y_true,iters=40000):
    """
    Implementation of Uncertainty quantification method using triangualr fuzzy numbers
    and the principle of justifiable granularity
    Parameters
    ----------
    y_pred : numpy array
        Predictions
    y_true : numpy array
        True or actual values
    iters: integer
        Number of iterations for particle sward optimization algorithm

    Returns
    -------
    ((alpha,beta),cost)
    alpha : float
        Left spread of triangular fuzzy number representing prediction uncertainty
    beta : float
        Right spread of triangular fuzzy number representing prediction uncertainty
    cost:float
        Cost function defined as -specificity*coverage
        
    """
    def coverage(alpha,beta):
        n=len(y_true)
        cost=[]
        for j in range(len(alpha)):
            c=0
            for i in range(len(y_true)):
                if y_true[i]<=y_pred[i]-alpha[j]:
                    c+=0
                elif y_true[i]<=y_pred[i]:
                    c+= 1-(y_pred[i]-y_true[i])/alpha[j]
                elif y_true[i]<=y_pred[i]+beta[j]:
                    c+= 1-(y_true[i]-y_pred[i])/beta[j]
                else:
                    c+=0
            cost.append(c)
        return cost

    def func(solution):
        alpha=solution[:, 0]
        beta=solution[:,1]
        return -(np.exp(.5*(-alpha-beta))*coverage(alpha,beta))
    
    e=mean_absolute_error(y_true, y_pred)
    max_bound = e*100 * np.ones(2)
    min_bound = np.zeros(2)
    bounds = (min_bound, max_bound)

    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

    # Perform optimization
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options, bounds=bounds)
    cost, pos = optimizer.optimize(func, iters)
    return pos,cost

y_true=np.arange(10)
y_pred=y_true+2
print("***********PSO_UQ**********")
print(PSO_UQ(y_pred,y_true))