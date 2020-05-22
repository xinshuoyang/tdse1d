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
    res = np.sqrt(2.0)*(g1.sigma*g2.sigma)**0.25/np.sqrt(g1.sigma+g2.sigma)*np.exp(-(g1.mu-g2.mu)**2.0/(g1.sigma+g2.sigma)/2.0)
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
    coef = g1.coef*g2.coef/(np.pi*(g1.sigma+g2.sigma))**0.25*np.exp(-(g1.mu-g2.mu)**2.0/(g1.sigma+g2.sigma)/2.0)
    sigma = g1.sigma*g2.sigma/(g1.sigma+g2.sigma)
    mu = (g1.sigma*g2.mu+g2.sigma*g1.mu)/(g1.sigma+g2.sigma)
    return dgatom(coef,sigma,mu)

def dconvgatoms(g1,g2):
    """
    Compute the convolution of two Gaussian atoms
    int_{R}g1(x-y)g2(y)dy
    where the kernel k(x-y) = coef*gatom(x,sigma,y)
    and g(x) = coef*gatom(y,sigma,mu).
    """
    coef = (4.0*np.pi*g1.sigma*g2.sigma/(g1.sigma+g2.sigma))**0.25*g1.coef*g2.coef
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

    L = np.empty([krankmax,g.nterms],dtype=np.float64)
    diag = np.ones(g.nterms,dtype=np.float64)
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

        L[i,ipivot[i]] = np.sqrt(diag[ipivot[i]])

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
    res *= 0.5
    for (a1,a2,a3) in product(v.atoms,phi.atoms,phi.atoms):
        res += dinnerprod(dprodgatoms(a1,a2),a3,use_coef=True)
    return res

def tise_solve(maxiter=20):
    # get representation of exp(-r) as sum of Gaussians
    nt,pp,wei = get_exp_as_gaussian(1e-8,1e-8,1e3)

    # v(x)=-8*exp(-x^2)
    v = dgamm(1)
    v.atoms.append(dgatom(-8.0*(np.pi/2.0)**0.25,0.5,0.0))

    # Initial solution
    phi0 = dgamm(2)
    phi0.atoms.append(dgatom(1.0,1.0,-1.0))
    phi0.atoms.append(dgatom(1.0,1.0,1.0))

    # Compute energy
    energy = compute_energy(phi0,v)
    mu = np.sqrt(-2.0*energy)

    print(mu,energy)

    eps = 1.0e-10
    krankmax = 500
    for _ in range(maxiter):
        vphi = dgamm()

        for (a1,a2) in product(v.atoms,phi0.atoms):
            a3 = dprodgatoms(a1,a2)
            if abs(a3.coef) > eps:
                vphi.atoms.append(a3)
        vphi.nterms = len(vphi.atoms)

        # Reduce the number
        vphi = sfastchol(vphi,krankmax,1e-7)

        # Get Green's function
        gf = dgamm(nt)
        for i in range(nt):
            gf.atoms.append(dgatom(-1.0/mu*wei[i]*(np.pi/(2.0*mu*mu*pp[i]))**0.25,0.5/mu/mu/pp[i],0.0))

        phi1 = dgamm()
        for (a1,a2) in product(gf.atoms,vphi.atoms):
            a3 = dconvgatoms(a1,a2)
            if abs(a3.coef) > eps:
                phi1.atoms.append(a3)
        phi1.nterms = len(phi1.atoms)
        phi1 = sfastchol(phi1,krankmax,1e-7)

        # Normalize phi1
        phi_1_norm = 0.0
        for (a1,a2) in product(phi1.atoms,phi1.atoms):
            phi_1_norm += dinnerprod(a1,a2,use_coef=True)
        for a in phi1.atoms:
            a.coef /= np.sqrt(phi_1_norm)

        # Update energy
        newenergy = compute_energy(phi1,v)
        newmu = np.sqrt(-2.0*newenergy)

        # Update solution
        phi0 = phi1
        energy,mu = newenergy,newmu

        print(energy,mu)

    return energy,phi0


def test_get_exp_as_gaussian():
    acc = 1.0e-10
    dnear = 1.0e-8
    dfar = 1.0e3
    nt,pp,wei = get_exp_as_gaussian(acc,dnear,dfar)
    print(nt)
    print(pp)
    print(wei)

    nx = 256
    xx = np.linspace(-10.0,3.0,nx)
    ff = np.exp(-10**xx)
    ff_approx = np.zeros(nx,dtype=float)
    for i in range(nt):
        ff_approx += wei[i]*np.exp(-pp[i]*(10**xx)**2)
    
    plt.plot(xx,np.log10(np.abs(ff-ff_approx)+1.0e-16))
    plt.show()

def test_tise_solve():
    energy,phi = tise_solve()
    xx = np.linspace(-5.0,5.0,256)
    plt.plot(xx,phi.eval(xx))
    plt.show()
    print(energy)

########################################################################################################################

class zgatom:
    def __init__(self,coef,sigma,mu):
        self.coef = coef
        self.sigma = sigma
        self.mu = mu

    def eval(self,x):
        return self.coef*((1.0/self.sigma+1.0/np.conj(self.sigma))/2.0/np.pi)**0.25*\
            np.exp(-np.imag(self.mu)**2.0/np.real(self.sigma)/2.0)*np.exp(-(x-self.mu)**2.0/self.sigma/2.0)

class zgmm():
    def __init__(self,nterms=0):
        self.nterms = nterms
        self.atoms = []

    def eval(self,x):
        return np.sum([self.atoms[i].eval(x) for i in range(self.nterms)],axis=0)

def zinnerprod(g1,g2,use_coef=True):
    res = np.sqrt(2.0)*(np.real(g1.sigma)*np.real(g2.sigma))**0.25/\
            np.sqrt(np.abs(g1.sigma)*np.abs(g2.sigma))/np.sqrt(1.0/g1.sigma+1.0/np.conj(g2.sigma))*\
            np.exp(-np.imag(g1.mu)**2.0/np.real(g1.sigma)/2.0)*\
            np.exp(-np.imag(g2.mu)**2.0/np.real(g2.sigma)/2.0)*\
            np.exp(-(g1.mu-np.conj(g2.mu))**2.0/(g1.sigma+np.conj(g2.sigma))/2.0)
    if use_coef:
        res *= g1.coef*g2.coef
    return res

def zconvheatgatom(t,g):
    """
    Compute the convolution of the heat kernel (with imaginary argument)
    and a Gaussian atom:
    int_{R}  1/(2πit)^(1/2) exp(i(x-y)^2/(2t)) zg(y) dy.
    """
    coef = g.coef*np.sqrt(np.abs(t*1.0j+g.sigma)/(1.0j*t/g.sigma+1.0)/np.abs(g.sigma))
    sigma = 1.0j*t+g.sigma
    mu = g.mu
    return zgatom(coef,sigma,mu)

def  zprodgatoms(g1,g2):
    """
    Compute the product of two Gaussian atoms
    """
    sigma = 1.0/(1.0/g1.sigma+1.0/g2.sigma)
    mu = sigma*(g1.mu/g1.sigma+g2.mu/g2.sigma)
    coef = g1.coef*g2.coef*((1.0/g1.sigma+1.0/np.conj(g1.sigma))/2.0/np.pi)**0.25*\
        ((1.0/g2.sigma+1.0/np.conj(g2.sigma))/2.0/np.pi)^0.25*\
        np.exp(-(g1.mu-g2.mu)^2.0/(g1.sigma+g2.sigma)/2.0)*\
        np.exp(-np.imag(g1.mu)**2.0/np.real(g1.sigma)/2.0)*\
        np.exp(-np.imag(g2.mu)**2.0/np.real(g2.sigma)/2.0)/\
        ((1.0/sigma+1.0/np.conj(sigma))/2.0/np.pi)**0.25*np.exp(np.imag(mu)**2.0/np.real(sigma)/2.0)
    return zgatom(coef,sigma,mu)

def cfastchol(g,krankmax,eps):
    """
    Reduce the number of terms in a wave function (complex64) that is
    represented as a Gaussian mixture.
    The reduced mixture has up to 7~8 digits of accuracy
    (depending on the input eps).

    Inputs:

    zw            --- wave function to be reduced
    krankmax      --- maximum expected number of terms;
                      if exceeded info = 1 on exit, otherwise info = 0

    eps           --- accuracy, the resulting accuracy is eps,
                      eps^2 can be 1d-14 - 1d-15 or so
                      if ifl = 1 then it is reset to the value reached
                      at krankmax

    Outputs:

    ipivot        --- pivot vector
    zw            --- reduced representation

    info          --- info = 1 if number of terms exceeds krankmax,
                      info = 0, otherwise
    """

    # set error threshold
    eps2 = eps*eps
    if eps2 < 1e-15:
        eps2 = 1e-15
        print("input eps < 1e-7, eps2 is reset to 1e-15")

    L = np.empty([krankmax,g.nterms],dtype=np.complex128)
    diag = np.ones(g.nterms,dtype=np.float64)
    ipivot = np.arange(g.nterms)
    newnterms = 0

    for i in range(g.nterms):
        # find largest diagonal element
        dmax = diag[ipivot[i]]
        imax = i
        for j in range(i+1,g.nterms):
            if dmax < diag[ipivot[j]]:
                dmax = diag[ipivot[j]]
                imax = j

        # swap to the leading position
        ipivot[i],ipivot[imax] = ipivot[imax],ipivot[i]

        # check if the diagonal element large enough
        if diag[ipivot[i]] < eps2:
            break
        
        L[i,ipivot[i]] = sqrt(diag[ipivot[i]])

        for j in range(i+1,g.nterms):
            r1 = np.dot(L[0:i-1,ipivot[i]],L[0:i-1,ipivot[j]])
            r2 = zinnerprod(g.atoms[ipivot[i]],g.atoms[ipivot[j]])
            L[i,ipivot[j]] = (r2-r1)/L[i,ipivot[i]]
            diag[ipivot[j]] -= L[i,ipivot[j]]*np.conj(L[i,ipivot[j]])
        newnterms += 1

    # solve for new coefficients using forward/backward substitution
    newcoefs = np.zeros(newnterms,dtype=np.complex128)

    for i in range(newnterms):
        for j in range(newnterms+1,g.nterms):
            newcoefs[i] += g.atoms[ipivot[j]].coef*\
                np.dot(L[1:i,ipivot[j]],L[1:i,ipivot[i]])
            # newcoefs[i] += zw.atoms[ipivot[j]].coef*
            #     zinnerprod(zw.atoms[ipivot[j]],zw.atoms[ipivot[i]])

    # forward substitution
    for i in range(newnterms):
        r1 = 0.0
        for j in range(i-1):
            r1 += L[j,ipivot[i]]*newcoefs[j]
        newcoefs[i] = (newcoefs[i]-r1)/L[i,ipivot[i]]

    # backward substitution
    for i in range(newnterms-1,-1,-1):
        r1 = 0.0
        for j in range(i+1,newnterms):
            r1 += np.conj(L[i,ipivot[j]])*newcoefs[j]
        newcoefs[i] = (newcoefs[i]-r1)/np.conj(L[i,ipivot[i]])

    # add "skeleton" terms and copy exponents and shifts
    newg = zgmm(newnterms)
    for i in range(newnterms):
        newg.atoms.append(zgatom(newcoefs[i]+g.atoms[ipivot[i]].coef,
            g.atoms[ipivot[i]].sigma,g.atoms[ipivot[i]].mu))

    return newg

def get_potential():
    pass

def irk1_solve_single_step(tstep,psi,V):
    """
    G.B. derivation. Probably wrong.

    ψ -- solution at current step
    """
    V,Vre,Vim = get_potential(tstep)

    # Compute ϕ0(y,τ/2)
    psi0h = zgmm(psi.nterms)

    for i in range(psi.nterms):
        psi0h.atoms.append(zconvheatgatom(tstep/2.0,psi.atoms[i]))

    psinext = zgmm()

    # psi0(x,τ)
    for i in range(psi.nterms):
        psinext.atoms.append(zconvheatgatom(tstep,psi.atoms[i]))
    psinext.nterms = len(psinext.atoms)

    # Convolution ∫ G0(x-y,τ/2)*V(y,τ/2)*ψ0(y,τ/2) dy
    for (a1,a2) in product(V.atoms,psi0h.atoms):
        psinext.atoms.append(zconvheatgatom(tstep/2.0,zprodgatoms(a1,a2)))
        psinext.atoms[-1].coef *= -tstep*1.0j

    # Convolution ∫ G0(x-y,τ/2) Vre(y,τ/2) ψ0(y,τ/2) dy
    for (a1,a2) in product(Vre.atoms,psi0h.atoms):
        psinext.atoms.append(zconvheatgatom(tstep/2.0,zprodgatoms(a1,a2)))
        psinext.atoms[-1].coef *= -0.5*tstep*tstep

    for (a1,a2) in product(Vim.atoms,psi0h.atoms):
        psinext.atoms.append(zconvheatgatom(tstep/2.0,zprodgatoms(a1,a2)))
        psinext.atoms[-1].coef *= -0.25j*tstep*tstep*tstep

    return cfastchol(psinext,1000,1e-7)

import matplotlib.pyplot as plt
if __name__ == '__main__':

    test_tise_solve()
    # g = dprodgatoms(dgatom(1.0,2.0,3.0),dgatom(4.0,5.0,6.0))
    # print(g.coef,g.sigma,g.mu)