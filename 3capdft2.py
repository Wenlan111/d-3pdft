import os
import sys
from scipy import optimize
import numpy as np
import time
import pickle
from pyscf import gto, scf, dft
import fragments
import psutil
from pyscf.scf import chkfile as scfchk
from pyscf.scf.addons import remove_linear_dep_
MAX_MEMORY =  int(psutil.virtual_memory().available / 1e6) 
print("pyscf max mem", MAX_MEMORY)

def get_E_wo_vp(casscf, Vp):
    Es = []
    for E, ci in zip(casscf.casscf.e_states, casscf.casscf.ci):
        D = casscf.casscf_.make_rdm1(casscf.casscf.mo_coeff, ci)
        Es.append(E - np.sum(Vp * D))
    return Es
    
def v2V(v, *phis):
    ngrid = len(phis)
    V = 0.
    for phi in phis:
        V += np.einsum("p,pu,pv->uv", w*v, phi, phi, optimize=True)
    V /= ngrid
    V = 0.5 * (V + V.T)
    return V

def D2n(D, *phis):
    ngrid = len(phis)
    n = 0.
    for phi in phis:
        n += np.einsum("pu,pv,uv->p", phi, phi, D, optimize=True)
    n /= ngrid
    return n

#basis
obasis = gto.basis.parse('''O    S
   8588.500                  0.00189515
   1297.230                  0.0143859
    299.2960                 0.0707320
     87.37710                0.2400010
     25.67890                0.5947970
      3.740040               0.2808020
O    SP
     42.11750                0.113889               0.0365114
      9.628370               0.920811               0.237153
      2.853320              -0.00327447             0.819702
O    SP
      0.905661               1.000000               1.000000
O    SP
      0.255611               1.000000               1.000000
O    SP
      0.0845000              1.0000000              1.0000000
O    D
      1.292                  1.000000''')

albasis = gto.basis.parse('''Al    S
  54866.489                  0.000839
   8211.7665                 0.006527
   1866.1761                 0.033666
    531.12934                0.132902
    175.11797                0.401266
     64.005500               0.531338
Al    S
     64.005500               0.202305
     25.292507               0.624790
     10.534910               0.227439
Al    S
      3.2067110              1.000000
Al    S
      1.152555               1.000000
Al    S
      0.1766780              1.000000
Al    S
      0.0652370              1.000000
Al    P
    259.28362                0.009448
     61.076870               0.070974
     19.303237               0.295636
      7.0108820              0.728219
Al    P
      2.6738650              0.644467
      1.0365960              0.417413
Al    P
      0.3168190              1.000000
Al    P
      0.1142570              1.000000
Al    SP
      0.0318000              1.0000000              1.0000000
Al    P
      0.041397               1.000000
Al    D
      0.3250000              1.0000000''')
#ref pbe calculation for O2/Al50 at 2.4A
mol, scf_dict = scfchk.load_scf('al2.4.chk')
mf = dft.UKS(mol)
mo_coeff  = scf_dict['mo_coeff']
mo_occ    = scf_dict['mo_occ']
mo_energy = scf_dict['mo_energy']
Daref, Dbref = mf.make_rdm1(mo_coeff, mo_occ) 
#nref = D2n(Daref+Dbref, phi)

#total geo
geo = gto.Mole()
geo.atom = "3.xyz"
geo.unit = 'angstrom'
geo.basis = {'O': obasis, 'Al': albasis}
geo.charge = 0
geo.spin = 2
geo.build()
#lgeo1
ncas = 12
nelec1 = (7,5)
w1o = 0.9664
w2o = 0.0336
lgeo1=gto.Mole()
lgeo1.atom= '''
O -2.64917    1.46446    9.98876
O -1.55673    0.82557    9.98876
'''
lgeo1.unit='angstrom'
lgeo1.basis = {'O': obasis}
#lgeo1.basis = 'def2-SVP'
lgeo1.spin = 2
lgeo1.build()
lhf1 = scf.UHF(lgeo1)
lhf1.kernel()
lcasscfs1 = fragments.FragmentCASSCF(lhf1, ncas, nelec1, omega=None, omega_sa=None)
lcasscfs1.solver.max_memory = MAX_MEMORY
lcasscfs1.kernel()
#ldft1 = fragments.FragmentDFT(lgeo1,'pbe0')
#lgeo2
nelec2 = (7,7)
lgeo2=gto.Mole()
lgeo2.atom= '''
O -2.64917    1.46446    9.98876
O -1.55673    0.82557    9.98876
'''
lgeo2.unit='angstrom'
lgeo2.basis = {'O': obasis}
#lgeo2.basis = 'def2-SVP'
lgeo2.charge = -2
lgeo2.spin = 0
lgeo2.build()
lhf2 = scf.RHF(lgeo2)
lhf2.kernel()
lcasscfs2 = fragments.FragmentCASSCF(lhf2, ncas, nelec2, omega=None, omega_sa=None)
lcasscfs2.solver.max_memory = MAX_MEMORY
lcasscfs2.kernel() 
#ldft2 = fragments.FragmentDFT(lgeo2,'pbe0')

#rgeo1
w1al = 0.9328
w2al = 0.0672

rgeo1 = gto.Mole()
rgeo1.atom = "rgeo.xyz"
rgeo1.unit = 'angstrom'
rgeo1.basis = {'ghost-O': '6-31g', 'Al': albasis}
rgeo1.spin = 0
rgeo1.build()
rdft1 = fragments.FragmentDFT(rgeo1,'pbe',metal = True)
#rdft1.dftsolver = remove_linear_dep_(rdft1.dftsolver, lindep=1e-4)
#rgeo2
rgeo2 = gto.Mole()
rgeo2.atom = "rgeo.xyz"
rgeo2.unit = 'angstrom'
rgeo2.basis = {'ghost-O': '6-31g', 'Al': albasis}
rgeo2.charge = 1
rgeo2.spin = 1
rgeo2.build()
rdft2 = fragments.FragmentDFT(rgeo2,'pbe',metal = True)
#rdft2.dftsolver = remove_linear_dep_(rdft2.dftsolver, lindep=1e-4)
#fragment ensembles
print("begin fragment ensemble")
l = fragments.ens([lcasscfs1,lcasscfs2],[w1o,w2o])
#l = fragments.ens([ldft1,ldft2],[w1o,w2o])
print("left ensemble electron number", l.get_nelec())
r = fragments.ens([rdft1,rdft2],[w1al,w2al])
print("right ensemble electron number", r.get_nelec())

#grid
#gridall
grid = dft.gen_grid.Grids(geo)
grid.level = 3
grid.build()
coords = grid.coords
w = grid.weights 
ao_values = dft.numint.eval_ao(geo, coords, deriv=1)
phi, phi_x, phi_y, phi_z = ao_values

ao_values_r1 = dft.numint.eval_ao(rgeo1, coords, deriv=1)
phi_r1, phi_r1_x, phi_r1_y, phi_r1_z = ao_values_r1
ao_values_r2 = dft.numint.eval_ao(rgeo2, coords, deriv=1)
ao_values_l1 = dft.numint.eval_ao(lgeo1, coords, deriv=1)
ao_values_l2 = dft.numint.eval_ao(lgeo2, coords, deriv=1)
phi_l1, phi_l1_x, phi_l1_y, phi_l1_z = ao_values_l1  
phi_l2, phi_l2_x, phi_l2_y, phi_l2_z = ao_values_l2 

#data = pickle.load(open("pdft3_checkpoint.pkl", "rb"))
# define init parameters
#vp = data["vp"]
vp = 0
maxiter = 35000
#step_size = 0.12
steps = np.hstack((np.linspace(0.5, 0.1, 5000), 0.1*np.ones(maxiter-5000)))
# convergence info
dVperrconv = 1e-1
Efconv = 1e-7
# VH
#load the results from previous run
#data = pickle.load(open("pdft_checkpointpbe.pkl", "rb"))
#Vpl  = data["Vpl"]
#Vpr  = data["Vpr"]

# initial run
l.scf(None)
r.scf(None)
El = l.get_E()
Er = r.get_E()
Ef = El+Er
#Ef = data["Ef"]
nref = D2n(Daref+Dbref, phi)
Dal, Dbl = l.get_D()
Dar, Dbr = r.get_D()
#Dal, Dbl = data["Dal"], data["Dbl"] 
#Dar, Dbr = data["Dar"], data["Dbr"]
#nl = data["nl"]
#nr = data["nr"]
nl = D2n(Dal+Dbl,phi_l1)
nr = D2n(Dar+Dbr,phi_r1)
nf = nl + nr
print("N:",np.sum(nf*w))
L1 = np.sum(np.abs(nf-nref)*w)
# initial run
print(f"init Ef={Ef:.5f} L1={L1:.4f}.")
Efold = Ef
nold = nf
checkpoint = "pdft3_checkpoint.pkl"
#PDFT-scf-LOOP
#start = data["step"] + 1
#for itera in range(start, len(steps)):
#    thisstep = steps[itera]
for itera, thisstep in enumerate(steps):
    t = -time.time()
    #dVp = get_dVp(Dfa, Dfb, geo,basis, pbs, ao_values, w)
    vp += thisstep * (nf- nref)
    Vpl = v2V(vp, phi_l1)
    Vpr = v2V(vp, phi_r1)
    #print(Vp)
    l.scf(Vpl)
    r.scf(Vpr)
    #print(Vp)
    El = l.get_E()
    Er = r.get_E()
    Ef = El + Er
    Dal, Dbl = l.get_D()
    Dar, Dbr = r.get_D()
    nl = D2n(Dal+Dbl, phi_l1)
    nr = D2n(Dbr+Dar, phi_r1)
    nf = nl + nr
    print("Npdft",np.sum(nf*w))
    L1 = np.sum(np.abs(nf-nref)*w)
    dEf = np.abs(Efold - Ef)
    #dVperr = np.abs(np.sum(0.5 * (dVpa + dVpb.T) * (Dfa + Dfb)))
    t += time.time()
    print(f"step={itera:3d} Ef={Ef:.5f} dEf={dEf:.2e} L1={L1:.4f} stepsize={thisstep:.1e} t={t:.1f}s.")    
    Efold = Ef

    # check some convergence
    Data = {
        'step' : itera,
        'vp'  : vp,
        'Vpl' : Vpl,
        'Vpr' : Vpr,
        'Dal' : Dal,
        'Dbl' : Dbl,
        'Dar' : Dar,
        'Dbr' : Dbr,
        'nr'  : nr,
        'nl'  : nl,
        'Ef'  : Ef,
        'L1'  : L1,
    }

    tmpname = checkpoint + ".tmp"
    # atomic write: write → flush → rename
    with open(tmpname, "wb") as f:
        pickle.dump(Data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmpname, checkpoint)
    print(f"[Checkpoint] updated {checkpoint}")
    # check some convergence
    if (dEf<Efconv):# and (dVperr<dVperrconv):
        print("converged.")
        break
