# Poisson-s-Equation-Project
 # Contributors
 1) BOULOUZ HAMZA
 2) SARRAR MOHCIN
 3) MOATADID ISMAIL
 4) BAJJOU ABDERRAZAK
 # Supervisor
 KISSAMI IMAD

   Solving the Poisson's equation discretized on the [0,1]x[0,1] domain
   using the finite difference method and a Jacobi's iterative solver.
 
    Delta u = f(x,y)= 2*(x*x-x+y*y -y)
    u equal 0 on the boudaries
    The exact solution is u = x*y*(x-1)*(y-1)
 
    The u value is :
    coef(1) = (0.5*hx*hx*hy*hy)/(hx*hx+hy*hy)
    coef(2) = 1./(hx*hx)
    coef(3) = 1./(hy*hy)
 
 *    u(i,j)(n+1)= coef(1) * (  coef(2)*(u(i+1,j)+u(i-1,j)) + coef(3)*(u(i,j+1)+u(i,j-1)) - f(i,j))

 *   ntx and nty are the total number of interior points along x and y, respectivly.
 
 *   hx is the grid spacing along x and hy is the grid spacing along y.
 *    hx = 1./(ntx+1)
 *    hy = 1./(nty+1)
 ###   On each process, we need to:
   1) Split up the domain
   2) Find our 4 neighbors
   3) Exchange the interface points
   4) Calculate u

