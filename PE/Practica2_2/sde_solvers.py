# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 17:29:26 2020

@author: Alberto Suárez
"""
# Load packages
import numpy as np


def euler_maruyana(t0, x0, T, a, b, M, N):
    """ Numerical integration of an SDE using the stochastic Euler scheme

    x(t0) = x0
    dx(t) = a(t, x(t))*dt + b(t, x(t))*dW(t)   [Itô SDE]

    Parameters
    ----------
    t0 : float
        Initial time for the simulation
    x0 : float
        Initial level of the process
    T : float
        Length of the simulation interval [t0, t0+T]
    a :
        Function a(t,x(t)) that characterizes the drift term
    b :
        Function b(t,x(t)) that characterizes the diffusion term
    M: int
        Number of trajectories in simulation
    N: int
        Number of intervals for the simulation

    Returns
    -------
    t: numpy.ndarray of shape (N+1,)
        Regular grid of discretization times in [t0, t0+T]
    X: numpy.ndarray of shape (M,N+1)
        Simulation consisting of M trajectories.
        Each trajectory is a row vector composed of the values
        of the process at t.

    Example
    -------

    >>> import matplotlib.pyplot as plt
    >>> import sde_solvers as sde
    >>> t0, S0, T, mu, sigma = 0, 100.0, 2.0, 0.3,  0.4
    >>> M, N = 20, 1000
    >>> def a(t, St): return mu*St
    >>> def b(t, St): return sigma*St
    >>> t, S = sde.euler_maruyana(t0, S0, T, a, b, M, N)
    >>> _ = plt.plot(t,S.T)
    >>> _= plt.xlabel('t')
    >>> _=  plt.ylabel('S(t)')
    >>> _= plt.title('Geometric BM (Euler scheme)')

    """

    t = np.linspace(t0,t0+T, N+1)
    X = np.ndarray(shape=(M, N+1))

    #Fill all trajectories with initial condition
    X[:, 0] = np.full(M, x0)

    #Value for time step
    dT = T/N

    #Generate values simultaneously for all trajectories at intervals n = {1..N+1}
    for (tn,n) in zip(t[1:],range(N)):
        Zn = np.random.randn(M)
        X[:,n+1] = X[:,n] + a(tn, X[:,n])*dT + b(tn, X[:,n])*np.sqrt(dT)*Zn

    return t, X



def milstein(t0, x0, T, a, b, db_dx, M, N):
    """ Numerical integration of an SDE using the stochastic Milstein scheme

    x(t0) = x0
    dx(t) = a(t, x(t))*dt + b(t, x(t))*dW(t)   [Itô SDE]

    Parameters
    ----------
    t0 : float
        Initial time for the simulation
    x0 : float
        Initial level of the process
    T : float
        Length of the simulation interval [t0, t0+T]
    a :
        Function a(t, x(t)) that characterizes the drift term
    b :
        Function b(t, x(t)) that characterizes the diffusion term
    db_dx:
        Derivative wrt the second argument of b(t, x)
    M: int
        Number of trajectories in simulation
    N: int
        Number of intervals for the simulation

    Returns
    -------
    t: numpy.ndarray of shape (N+1,)
        Regular grid of discretization times in [t0, t0+T]
    X: numpy.ndarray of shape (M,N+1)
        Simulation consisting of M trajectories.
        Each trajectory is a row vector composed of the
        values of the process at t.

    Example
    -------

    >>> import matplotlib.pyplot as plt
    >>> import sde_solvers as sde
    >>> t0, S0, T, mu, sigma = 0, 100.0, 2.0, 0.3,  0.4
    >>> M, N = 20, 1000
    >>> def a(t, St): return mu*St
    >>> def b(t, St): return sigma*St
    >>> def db_dSt(t, St): return sigma
    >>> t, S = sde.milstein(t0, S0, T, a, b, db_dSt, M, N)
    >>> _ = plt.plot(t,S.T)
    >>> _= plt.xlabel('t')
    >>> _=  plt.ylabel('S(t)')
    >>> _= plt.title('Geometric BM (Milstein scheme)')

    """

    t = np.linspace(t0, t0+T, N+1)
    X = np.ndarray(shape=(M, N+1))

    #Fill all trajectories with initial condition
    X[:, 0] = np.full(M, x0)

    #Value for time step
    dT = T/N

    #Generate values simultaneously for all trajectories at intervals n = {1..N+1}
    for (tn,n) in zip(t[1:],range(N)):
        Zn = np.random.randn(M)
        X[:,n+1] = (X[:,n] + a(tn, X[:,n])*dT 
                   + b(tn, X[:,n])*np.sqrt(dT)*Zn 
                   + 0.5*b(tn, X[:,n])*db_dx(tn, X[:,n]) 
                   * (Zn**2 - 1)*dT)

    return t, X


def simulate_jump_process(t0, T, simulator_arrival_times, simulator_jumps, M):
    """ Simulation of jump process

    Parameters
    ----------
    t0 : float
        Initial time for the simulation
    T : float
        Length of the simulation interval [t0, t0+T]
    simulator_arrival_times: callable with arguments (t0,T)
        Function that returns a list of M arrays of arrival times in [t0, t0+T]
    simulator_jumps: callable with argument N
        Function that returns a list of M arrays with the sizes of the jumps
    M: int
        Number of trajectories in the simulation

    Returns
    -------
    times_of_jumps: list of lists
         list of M sublists.
         Each sublist contains the time instants in [t0,t1] 
         at which the jumps take place. 
    sizes_of_jumps: list of M lists
         list of M sublists.
         Each sublist contains the sizes of the jumps.  

    """

    times_of_jumps = [[] for _ in range(M)]
    sizes_of_jumps = [[] for _ in range(M)]
    for m in range(M):
        times_of_jumps[m] = simulator_arrival_times(t0, T)
        max_jumps = len(times_of_jumps[m])
        sizes_of_jumps[m] = simulator_jumps(max_jumps)
    return times_of_jumps, sizes_of_jumps


def euler_jump_diffusion(t0, x0, T, a, b, c,
                         simulator_jump_process,
                         M, N):
    """ Stochastic Euler scheme for the simulation of jump diffusion process

    x(t0) = x0
    dx(t) = a(t, x(t))*dt + b(t, x(t))*dW(t) + c(t, x(t)) dJ(t)

    [Itô SDE with a jump term]


    Parameters
    ----------
    t0 : float
        Initial time for the simulation
    x0 : float
        Initial level of the process
    T : float
        Length of the simulation interval [t0, t0+T]
    a : Function a(t,x(t)) that characterizes the drift term
    b : Function b(t,x(t)) that characterizes the diffusion term
    c : Function c(t,x(t)) that characterizes the jump term
    simulator_jump_process: Function that returns times and sizes of jumps
    M: int
        Number of trajectories in simulation
    N: int
        Number of intervals for the simulation

    Returns
    -------
    t: numpy.ndarray of shape (N+1,)
        Regular grid of discretization times in [t0,t1]
    X: numpy.ndarray of shape (M,N+1)
        Simulation consisting of M trajectories.
        Each trajectory is a row vector composed of the
        values of the process at t
    """
    
    t = np.linspace(t0,t0+T, N+1)
    X = np.ndarray(shape=(M, N+1))

    #Fill all trajectories with initial condition
    X[:, 0] = np.full(M, x0)

    #Value for time step
    dT = T/N
    
    #Gaussian noise
    Z = np.random.randn(M, N)
    
    #Generate jump times and sizes
    times_of_jumps, sizes_of_jumps = simulator_jump_process(t0, T, M)
    
    #We transform the times of jumps to the indexes where they take place
    jumps_dT = list(map(lambda ts: ((ts - t0)/dT).astype(int), times_of_jumps))
    
    for m in range(M):
        for n in range(1, N+1):
            jump_term = 0
            jump_indexes = np.where(jumps_dT[m]==n)[0]
            
            #There may be multiple jumps in a single dT, so we add all of them
            if len(jump_indexes) > 0:
                for idx in jump_indexes:
                    jump_term += c(t[n-1], X[m, n-1])*sizes_of_jumps[m][idx]
                
            X[m, n] = X[m, n-1] + a(t[n-1], X[m, n-1])*dT + b(t[n-1], X[m, n-1])*np.sqrt(dT)*Z[m, n-1] + jump_term
    
    return t, X


