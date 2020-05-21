import numpy as np
from itertools import product

class dgatom:
    def __init__(self,coef,sigma,mu):
        self.coef = coef
        self.sigma = sigma
        self.mu = mu

    def eval(self,x):
        return self.coef/(np.pi*self.sigma)**0.25*np.exp(-(x-self.mu)**2.0/self.sigma/2.0)

class dgamm:
    def __init__(self,nterms=0):
        self.nterms = nterms
        self.atoms = []

    def eval(self,x):
        return np.sum([self.atoms[i].eval(x) for i in range(self.nterms)],axis=0)

def dinnerprod(g1,g2,use_coef=False):
    res = np.sqrt(2.0)*(g1.sigma*g2.sigma)**0.25/np.sqrt(g1.sigma+g2.sigma)*exp(-(g1.mu-g2.mu)**2.0/(g1.sigma+g2.sigma)/2.0)
    if use_coef:
        res *= g1.coef*g2.coef
    return res

def dgradinnerprod(g1,g2,use_coef=False):
    """
    Compute the inner product of the gradients
    int_{R}  d/dx(f(x)) * d/dx(g(x)) dx
    """
    res = (4.0*g1.sigma*g2.sigma)**0.25/np.sqrt(g1.sigma+g2.sigma)*\
        (-(g1.mu-g2.mu)**2.0/(g1.sigma+g2.sigma)**2.0+1.0/(g1.sigma+g2.sigma))*\
        np.exp(-(g1.mu-g2.mu)**2.0/(g1.sigma+g2.sigma)/2.0)
    if use_coef:
        res *= g1.coef*g2.coef
    return res


def dprodgatoms(g1,g2):
    coef = g1.coef*g2.coef/(np.pi*(g1.sigma+g2.sigma))^0.25*exp(-(g1.mu-g2.mu)^2.0/(g1.sigma+g2.sigma)/2.0)
    sigma = g1.sigma*g2.sigma/(g1.sigma+g2.sigma)
    mu = (g1.sigma*g2.mu+g2.sigma*g1.mu)/(g1.sigma+g2.sigma)
    return dgatom(coef,sigma,mu)

def dconvgatoms(g1,g2):
    """
    Compute the convolution of two Gaussian atoms
    int_{R}g1(x-y)g2(y)dy
    where the kernel k(x-y) = coef*gatom(x,σ,y)
    and g(x) = coef*gatom(y,σ,μ).
    """
    coef = (4.0*np.pi*g1.sigma*g2.sigma/(g1.sigma+g2.sigma))^0.25*g1.coef*g2.coef
    sigma = g1.sigma+g2.sigma
    mu = g2.mu
    return dgatom(coef,sigma,mu)

def sfastchol(g,krankmax,eps):
    """
    Reduce the number of terms in a Gaussian mixture model.
    The reduced mixture has up to 7~8 digits of accuracy
    (depending on the input eps).
    """

    eps2 = eps*eps
    if eps2 < 1.0e-15:
        eps2 = 1e-15
        print("input eps < 1e-7, eps2 is reset to 1e-15")

    L = np.empty([krankmax,g.nterms],dtype=float)
    diag = np.ones(g.nterms,dtype=float)
    ipivot = np.arange(g.nterms)
    newnterms = 0

    for i in range(g.nterms):

        # Find largest diagonal element
        dmax = diag[ipivot[i]]
        imax = i
        for j in range(i+1,g.nterms):
            if dmax < diag[ipivot[j]]:
                dmax = diag[ipivot[j]]
                imax = j
        
        # Swap to the leading position
        ipivot[i],ipivot[imax] = ipivot[imax],ipivot[i]

        # Check if the diagonal element is large enough
        if diag[ipivot[i]] < eps2:
            break

        L[i,ipivot[i]] = sqrt(diag[ipivot[i]])

        for j in range(i+1,g.nterms):
            r1 = np.dot(L[0:i,ipivot[i]],L[0:i,ipivot[j]])
            r2 = dinnerprod(g.atoms[ipivot[i]],g.atoms[ipivot[j]])
            L[i,ipivot[j]] = (r2-r1)/L[i,ipivot[i]]
            diag[ipivot[j]] -= L[i,ipivot[j]]**2.0

        newnterms += 1

    # solve for new coefficients via forward/backward substitution
    newcoefs = np.zeros(newnterms,dtype=float)

    for i in range(newnterms):
        for j in range(newnterms,g.nterms):
            newcoefs[i] += g.atoms[ipivot[j]].coef*np.dot(L[0:i+1,ipivot[i]],L[0:i+1,ipivot[j]])
    
    # forward substitution
    for i in range(newnterms):
        r1 = 0.0
        for j in range(i):
            r1 += L[j,ipivot[i]]*newcoefs[j]
        newcoefs[i] = (newcoefs[i]-r1)/L[i,ipivot[i]]
    
    # backward substitution
    for i in range(newnterms-1,-1,-1):
        r1 = 0.0
        for j in range(i+1,newnterms):
            r1 += L[i,ipivot[j]]*newcoefs[j]
        newcoefs[i] = (newcoefs[i]-r1)/L[i,ipivot[i]]

    # add "skeleton" terms and copy exponents and shifts
    newg = dgamm(newnterms)
    for i in range(newnterms):
        newg.atoms.append(dgatom(newcoefs[i]+g.atoms[ipivot[i]].coef,
            g.atoms[ipivot[i]].sigma,
            g.atoms[ipivot[i]].mu))

    return newg


def get_exp_as_gaussian(acc,dnear,dfar):
    """
    Generates an approximation
    f(r) = sum_{j=1}^{nt} wei_j exp(- pp_j r^2)
    with the number of terms "nt", so that
    |f(r) - exp(-r)| <= acc for dnear <= r <= dfar.
    """
    eps = 1e-17
    aa = 0.037037037037037037
    bb = -0.253086419753086420

    # set stepsize
    hstep = 1.0/(aa+bb*np.log10(acc))

    # generate an equally spaced logarithmic grid
    rstart = -73.0
    rstop = 5.0
    nstep = np.int64(np.ceil((rstop-rstart)/hstep))
    nterms = nstep

    # generate an excessive collection of gaussians
    ppini = np.zeros(nterms,dtype=float)
    weiini = np.zeros(nterms,dtype=float)

    for i in range(nterms):
        ppini[-i-1] = np.exp(-rstart-i*hstep)
        weiini[-i-1] = hstep/2.0/np.sqrt(np.pi)*np.exp(-np.exp(rstart+i*hstep)/4.0+(rstart+i*hstep)/2.0)

    # evaluate error of approximation at lower limit "dnear" and drop terms
    ss = 0.0
    nstop = 0
    for i in range(nterms):
        ss += weiini[i]*np.exp(-ppini[i]*dnear*dnear)
        err = np.log10(abs(ss*np.exp(dnear)-1.0)+eps)
        if err < np.log10(acc):
            break
        else:
            nstop += 1

    nt = nstop
    pp = ppini[0:nt]
    wei = weiini[0:nt]

    return nt,pp,wei


def compute_energy(phi,v):
    """
    """
    res = 0.0
    for (a1,a2) in product(phi.atoms,phi.atoms):
        res += dgradinnerprod(a1,a2,use_coef=True)

    for (a1,a2,a3) in product(v.atoms,phi.atoms,phi.atoms):
        res += dinnerprod(dprodgatoms(a1,a2),a3,use_coef=True)
    return 0.5*res
    

def tise_solve(maxiter=20):
    # get representation of exp(-r) as sum of Gaussians
    nt,pp,wei = get_exp_as_gaussian(1e-8,1e-8,1e3)

    # v(x)=-8*exp(-x^2)
    v = dgamm(1)
    v.atoms.append(dgatom(-8.0/(np.pi/2)**0.25,0.5,0.0))

    # Initial solution
    phi0 = dgamm(2)
    phi0.atoms.append(dgatom(1.0,1.0,1.0))
    phi0.atoms.append(dgatom(1.0,1.0,-1.0))

    # Compute energy
    energy = compute_energy(phi0,v)
    mu = np.sqrt(-2.0*energy)

    eps = 1.0e-10
    for _ in range(maxiter):
        vphi = dgatom()

        for (a1,a2) in product(v,phi0):
            a3 = dprodgatoms(a1,a2)
            if a3.coef > eps:
                vphi.atoms.append(a3)
        vphi.nterms = len(vphi.atoms)

        # Reduce the number
        vϕ = sfastchol(vϕ,krankmax,1e-7)

        # Get Green's function
        gf = dgamm(nt)
        for i in range(nt):
            gf.atoms[i].append(dgatom(-1.0/mu*wei[i]*(np.pi/(2.0*mu*mu*pp[i]))**25,0.5/mu/mu/pp[i],0.0))

        phi1 = dgamm()
        for (a1,a2) in product(gf.atoms,vphi.atoms):
            a3 = dconvgatoms(a1,a2)
            if abs(a3.coef) > eps:
                phi1.atoms.append(a3)
        phi1.nterms = len(phi.atoms)
        phi1 = sfastchol(phi1,krankmax,1e-7)

        # Normalize phi1
        phi_1_norm = 0.0
        for (a1,a2) in product(phi1.atoms,phi1.atoms):
            phi_1_norm += dinnerprod(a1,a2,use_coef=True)
        for a in phi1.atoms:
            a.coef /= np.sqrt(phi_1_norm)

        # Update energy
        newenergy = compute_energy(phi1,v)
        newmu = np.sqrt(-newenergy)

        # Update solution
        phi0 = phi1
        energy,mu = newenergy,newmu

    return energy,phi0


def test_get_exp_as_gaussian():
    acc = 1.0e-10
    dnear = 1.0e-5
    dfar = 1.0e3
    nt,pp,wei = get_exp_as_gaussian(acc,dnear,dfar)
    print(nt)
    print(pp)
    print(wei)

    nx = 256
    xx = np.linspace(-10.0,0.0,nx)
    ff = np.exp(-10**xx)
    ff_approx = np.zeros(nx,dtype=float)
    for i in range(nt):
        ff_approx += wei[i]*np.exp(-pp[i]*10**xx**2)
    
    # plt.plot(10**xx,ff)
    plt.plot(10**xx,ff_approx)
    plt.show()

# class zgatom:

#     def __init__(self,coef,sigma,mu):
#         self.coef = coef
#         self.sigma = sigma
#         self.mu = mu

#     def eval(self,x):
#         return self.coef*((1.0/self.sigma+1.0/np.conj(self.sigma))/2.0/np.pi)^0.25*
#             np.exp(-np.imag(self.μ)^2.0/np.real(self.sigma)/2.0).*np.exp.(-(x-self.μ).^2.0/self.sigma/2.0)

import matplotlib.pyplot as plt
if __name__ == '__main__':

    test_get_exp_as_gaussian()