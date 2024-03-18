import numpy as np

def busqueda_dicotomica(f, xmin, xmax, niter = 1000, l = 0.001, eps = 0.005):

    a_k = xmin
    b_k = xmax
    
    lamb = (a_k + b_k) / 2 - eps
    mu = (a_k + b_k) / 2 + eps

    for k in range(niter):
        if (abs(b_k - a_k)) <= l:
            print(f"El mínimo está entre {a_k} y {b_k}, con n = {k} iteraciones")
            return(a_k, b_k)
        else:
            lamb = (a_k + b_k) / 2 - eps
            mu = (a_k + b_k) / 2 + eps
            if (f(lamb) < f(mu)):
                a_k, b_k = a_k, mu
            else:
                a_k, b_k = lamb, b_k

def seccion_aurea(f, xmin, xmax, niter = 1000, l = 0.001):
    alpha = 0.618
    
    a_k = xmin
    b_k = xmax
    
    lambd = a_k + (1 - alpha)*(b_k - a_k)
    mu = a_k + alpha*(b_k - a_k)

    for k in range(niter):
        if (abs(a_k - b_k) < l):
            print(f"El mínimo está entre {a_k} y {b_k}, con n = {k} iteraciones")
            return(a_k, b_k)
        if (f(lambd) > f(mu)):
            a_k = lambd
            lambd = mu
            mu = a_k + alpha*(b_k - a_k)
            k = k + 1
        if (f(lambd) < f(mu)):
            b_k = mu
            mu = lambd
            lambd = a_k + (1 - alpha)*(b_k - a_k)
            k = k + 1


def gradiente(f, fp, x0, niter = 1000, lambd = 0.1, prec = 0.001):
    
    x = x0

    for k in range(niter):
        if (abs(fp(x)) <= prec):
            break
        else:
            d_k = - fp(x)
            min = f(x + lambd*d_k)
            x = x + lambd*d_k
            
    print(f"El mínimo de la función se encuentra en {x}, con un valor {f(x)} y n = {k} iteraciones")
    return(x)

def newton(grad, hessian, xk, lr, eps, max_iter):

  x = xk.copy()
  converged = False

  for iter in range(max_iter):

    H = hessian(x)

    H_inv = np.linalg.solve(H, np.eye(H.shape[0]))

    xk_new = x - lr * grad(x) @ H_inv

    if np.linalg.norm(xk_new - x) < eps:
      print(f"Convergencia en iteración: {iter}")
      converged = True
      break

    x = xk_new

  if not converged:
        print("[newton] No converge.")

  return xk_new


def linesearch_secant(x, grad, d, eps=1e-3, max_iter=100):

  curr_alpha = 0
  alpha = 0.5

  dphi_zero = grad(x).T @ d
  dphi_curr = dphi_zero

  for i in range(max_iter):
    
    
    alpha_old = curr_alpha
    curr_alpha = alpha
    dphi_old = dphi_curr
    dphi_curr = grad(x + curr_alpha * d).T @ d

    if dphi_curr < eps:
      break

    alpha = (dphi_curr * alpha_old - dphi_old * curr_alpha) / (dphi_curr - dphi_old)

    if np.abs(dphi_curr) < eps * np.abs(dphi_zero):
      break

  return alpha

    
def fletcher_reeves(x_0, grad, eps=1e-3, max_iter=100, tol=1e-8):

  x_k = x_0.copy()
  conv = False
  alpha = 0

  for i in range(max_iter):

    g = grad(x_k)

    if np.linalg.norm(g) < eps:
      conv = True
      break

    if i == 0:

      d = -g
      g_old = g

    alpha_k = (g.T @ g) / (g_old.T @ g_old)

    d = -g + alpha_k * d

    alpha = linesearch_secant(x_k, grad, d)

    x_n = x_k + alpha * d

    if np.linalg.norm(x_n - x_k) < tol * np.linalg.norm(x_k):
      print(f"Tolerancia alcanzada en {i+1} iteraciones")
      break

    g_old = g
    x_k = x_n

  return x_k


