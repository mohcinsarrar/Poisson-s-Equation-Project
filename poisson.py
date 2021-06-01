#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 09:34:21 2020

Created by :

SARRAR Mohcin
BOULLOUZ Hamza
MOATADID Ismail
BAJJOU Abderrazzak


/*
 *   Solving the Poisson's equation discretized on the [0,1]x[0,1] domain
 *   using the finite difference method and a Jacobi's iterative solver.
 *
 *   Delta u = f(x,y)= 2*(x*x-x+y*y -y)
 *   u equal 0 on the boudaries
 *   The exact solution is u = x*y*(x-1)*(y-1)
 *
 *   The u value is :
 *    coef(1) = (0.5*hx*hx*hy*hy)/(hx*hx+hy*hy)
 *    coef(2) = 1./(hx*hx)
 *    coef(3) = 1./(hy*hy)
 *
 *    u(i,j)(n+1)= coef(1) * (  coef(2)*(u(i+1,j)+u(i-1,j)) &
 *               + coef(3)*(u(i,j+1)+u(i,j-1)) - f(i,j))
 *
 *   ntx and nty are the total number of interior points along x and y, respectivly.
 * 
 *   hx is the grid spacing along x and hy is the grid spacing along y.
 *    hx = 1./(ntx+1)
 *    hy = 1./(nty+1)
 *
 *   On each process, we need to:
 *   1) Split up the domain
 *   2) Find our 4 neighbors
 *   3) Exchange the interface points
 *   4) Calculate u
 *
 *   @author: kissami
 */


"""
import numpy as np
from mpi4py import MPI
from psydac.ddm.partition import mpi_compute_dims

import matplotlib.pyplot as plt
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
from numba import njit

comm = MPI.COMM_WORLD
nb_procs = comm.Get_size()
rank = comm.Get_rank()

nb_neighbours = 4
N = 0
E = 1
S = 2
W = 3

neighbour = np.zeros(nb_neighbours, dtype=np.int8)
ntx = 11
nty = 11

Nx = ntx+2
Ny = nty+2

npoints  =  [ntx, nty]
p1 = [2,2]
P1 = [False, False]
reorder = True

coef = np.zeros(3)
# Grid spacing '''
hx = 1/(ntx+1.)
hy = 1/(nty+1.)

# Equation Coefficients '''
coef[0] = (0.5*hx*hx*hy*hy)/(hx*hx+hy*hy)
coef[1] = 1./(hx*hx)
coef[2] = 1./(hy*hy)

def create_2d_cart(npoints, p1, P1, reorder):
    
    # Store input arguments                                                                                                                                                                                                                                               
    npts    = tuple(npoints)
    pads    = tuple(p1)
    periods = tuple(P1)
    reorder = reorder
    
    nprocs, block_shape = mpi_compute_dims(nb_procs, npts, pads )
    
    dims = nprocs
    
    if (rank == 0):
        print("Execution poisson with",nb_procs," MPI processes\n"
               "Size of the domain : ntx=",npoints[0], " nty=",npoints[1],"\n"
               "Dimension for the topology :",dims[0]," along x", dims[1]," along y\n"
               "-----------------------------------------")  
    
    # Creation of the Cartesian topology

    cart2d = comm.Create_cart(dims = dims ,periods = periods ,reorder = reorder)
    
    return dims, cart2d

def create_2dCoords(cart2d, npoints, dims):
   # Create 2d coordinates of each process'''

    b = [npoints[0]/dims[0],npoints[1]/dims[1]]
    
    coords = cart2d.Get_coords(rank)
    
    sx = coords[0]*b[0]+1
    ex = coords[0]*b[0]+b[0]
    
    sy = coords[1]*b[1]+1
    ey = coords[1]*b[1]+b[1]

    print("Rank in the topology :",rank," Local Grid Index :", sx, " to ",ex," along x, ",sy, " to", ey," along y")
    
    return int(sx), int(ex), int(sy), int(ey)

def create_neighbours(cart2d):

    # Get my northern and southern neighbours '''

    # Get my western and eastern neighbours '''
    
    neighbour[W] , neighbour[E] = cart2d.Shift( direction = 0 , disp=1 )
    neighbour[S] , neighbour[N] = cart2d.Shift( direction = 1 , disp=1 )

    
    print("Process", rank," neighbour: N", neighbour[N]," E",neighbour[E] ," S ",neighbour[S]," W",neighbour[W])
    
    return neighbour
#Creation of the derived datatypes needed to exchange points with neighbours'''
def create_derived_type(sx, ex, sy, ey):
    
    #Creation of the type_line derived datatype to exchange points with northern to southern neighbours '''

    # in the communication function we use slice instead of derived types

    # count = ex-sx+3
    # counts = [1 for i in range(count)]
    # displacements = [i for i in range(IDX(sx-1,ey),IDX(ex+1,ey)+1,count-1)]
    # type_ligne = MPI.DOUBLE.Create_indexed(counts,displacements)
    # type_ligne.Commit()

    # Creation of the type_column derived datatype to exchange points with western to eastern neighbours 

    # count = ey-sy+3
    # counts = [1 for i in range(count)]
    # displacements = [i for i in range(IDX(ex,sy-1),IDX(ex,ey+1)+1)]
    # type_column = MPI.DOUBLE.Create_indexed(counts,displacements)
    # type_column.Commit()
    type_ligne = 0
    type_column = 0
    return type_ligne, type_column

# Exchange the points at the interface '''
def communications(u, sx, ex, sy, ey, type_column, type_ligne):
    # Send to neighbour N and receive from neighbour S '''
    if(neighbour[N]>=0):########

        sendbuff = u[IDX(sx-1,ey):IDX(ex+1,ey)+1:ey-sy+3].copy()
        comm.Send( sendbuff ,neighbour[N])
        
        recvbuff = np.zeros(ex-sx+3)
        comm.Recv( recvbuff ,neighbour[N])
        u[IDX(sx-1,sy-1):IDX(ex+1,sy-1)+1:ey-sy+3] = recvbuff

    # Send to neighbour S and receive from neighbour N '''
    if(neighbour[S]>=0):########

        sendbuff = u[IDX(sx-1,sy):IDX(ex+1,sy)+1:ey-sy+3].copy()
        comm.Send( sendbuff ,neighbour[S])

        recvbuff = np.zeros(ex-sx+3)
        comm.Recv(recvbuff ,neighbour[S])
        u[IDX(sx-1,ey+1):IDX(ex+1,ey+1)+1:ey-sy+3] = recvbuff

        
    # Send to neighbour W and receive from neighbour E '''
    if(neighbour[W]>=0):#######

        sendbuff = u[IDX(sx,sy-1):IDX(sx,ey+1)+1].copy()
        comm.Send( sendbuff,neighbour[W])

        recvbuff = np.zeros(ey-sy+3)
        comm.Recv(recvbuff ,neighbour[W])
        u[IDX(sx-1,sy-1):IDX(sx-1,ey+1)+1] = recvbuff

    # Send to neighbour E  and receive from neighbour W '''
    if(neighbour[E]>=0):#######

        sendbuff = u[IDX(ex,sy-1):IDX(ex,ey+1)+1].copy()
        comm.Send( sendbuff,neighbour[E])

        recvbuff = np.zeros(ey-sy+3)
        comm.Recv(recvbuff ,neighbour[E])
        u[IDX(ex+1,sy-1):IDX(ex+1,ey+1)+1] = recvbuff
       

#
# * IDX(i, j) : indice de l'element i, j dans le tableau u
# * sx-1 <= i <= ex+1
# * sy-1 <= j <= ey+1

def IDX(i, j):
    return int(( ((i)-(sx-1))*(ey-sy+3) + (j)-(sy-1) ))

def initialization(sx, ex, sy, ey):
    # Grid spacing in each dimension'''
    # Solution u and u_new at the n and n+1 iterations '''
    
    SIZE = (ex-sx+3) * (ey-sy+3)

    u       = np.zeros(int(SIZE))
    u_new   = np.zeros(int(SIZE))
    f       = np.zeros(int(SIZE))
    u_exact = np.zeros(int(SIZE))
    
    # Initialition of rhs f and exact soluction '''
 
    for i in range(sx-1, ex+1, 1):
        x = i*hx
        for j in range(sy-1, ey+1, 1):
            y = j*hy
            u_exact[IDX(i,j)] = x*y*(x-1)*(y-1)
            f[IDX(i,j)] = 2*(x*x-x+y*y-y)

    return u, u_new, u_exact, f

def computation(u, u_new):
    
    # Compute the new value of u using'''

    for i in range(sx,ex+1):
        for j in range(sy,ey+1):
            u_new[IDX(i,j)] = coef[0]*(coef[1]*(u[IDX(i+1,j)]+u[IDX(i-1,j)])+coef[2]*(u[IDX(i,j+1)]+u[IDX(i,j-1)])-f[IDX(i,j)])
 
def output_results(u, u_exact):
    
    print("Exact Solution u_exact - Computed Solution u - difference")
    for itery in range(sy, ey+1, 1):
        print(u_exact[IDX(1, itery)], '-', u[IDX(1, itery)], u_exact[IDX(1, itery)]-u[IDX(1, itery)] );

# Calcul for the global error (maximum of the locals errors) '''
def global_error(u, u_new):
   
    local_error = 0
     
    for iterx in range(sx, ex+1, 1):
        for itery in range(sy, ey+1, 1):
            temp = np.fabs( u[IDX(iterx, itery)] - u_new[IDX(iterx, itery)]  )
            if local_error < temp:
                local_error = temp
    
    return local_error

import meshio

def plot_2d(f):

    f = np.reshape(f, (ex-sx+3, ey-sy+3))
    
    x = np.linspace(0, 1, ey-sy+3)
    y = np.linspace(0, 1, ex-sx+3)
    
    fig = plt.figure(figsize=(7, 5), dpi=100)
    ax = fig.gca(projection='3d')                      
    X, Y = np.meshgrid(x, y)      

    ax.plot_surface(X, Y, f, cmap=cm.viridis)
    
    plt.show()

dims, cart2d   = create_2d_cart(npoints, p1, P1, reorder)
neighbour      = create_neighbours(cart2d)

sx, ex, sy, ey = create_2dCoords(cart2d, npoints, dims)

type_ligne, type_column = create_derived_type(sx, ex, sy, ey)
u, u_new, u_exact, f = initialization(sx, ex, sy, ey)

# Time stepping '''
it = 0
convergence = False
it_max = 100000
eps = 2.e-16

# Elapsed time '''
t1 = MPI.Wtime()

#import sys; sys.exit()
while (not(convergence) and (it < it_max)):
    it = it+1

    temp = u.copy() 
    u = u_new.copy() 
    u_new = temp.copy()
    
    # Exchange of the interfaces at the n iteration '''
    communications(u, sx, ex, sy, ey, type_column, type_ligne)
   
    # Computation of u at the n+1 iteration '''
    computation(u, u_new)
    
    # Computation of the global error '''
    local_error = global_error(u, u_new)
    diffnorm = comm.allreduce(np.array(local_error), op=MPI.MAX )   
   
    # Stop if we obtained the machine precision '''
    convergence = (diffnorm < eps)
    
    # Print diffnorm for process 0 '''
    if ((rank == 0) and ((it % 10) == 0)):
        print("Iteration", it, " global_error = ", diffnorm)
        
# Elapsed time '''
t2 = MPI.Wtime()


if (rank == 0):

    # Print convergence time for process 0 '''
    print("Convergence after",it, 'iterations in', t2-t1,'secs')

    # Compare to the exact solution on process 0 '''
   
    output_results(u, u_exact)

    

    plot_2d(u)
    plot_2d(u_exact)
