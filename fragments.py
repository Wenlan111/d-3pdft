"""
2024.04.230
Yuming Shi
Here I will discuss how to define PDFT fragments such that:
    1) Vp is added;
    2) Computation is continued,
    3) fractional electrons are dealt with by ensembles.
All the implementation here is based on the adding a Vp properities to the pyscf.scf.hf object and to pyscf.scf.hf.get_hcore():
class SCF(lib.StreamObject):
    ....
    Vp = 0.
    ....
    def get_hcore(self, mol=None):
        if mol is None: mol = self.mol
        return get_hcore(mol) + self.Vp
Everything here is unrestricted calculation to reflect ensemble components.
"""
from pyscf import scf, dft, cc, mcscf, fci
import numpy as np
from functools import reduce
from pyscf import dft
from pyscf.scf.addons import remove_linear_dep_, smearing_
from pyscf.dft.gen_grid import treutler_prune


# HF
class FragmentHF:
    def __init__(self, mol):
        self.mol = mol
        self.solver = self.hfsolver = scf.UHF(mol) # does not necessarily have to be UHF, right?
        self.S = self.mol.intor('int1e_ovlp')
        self.E = None # total energy WITHOUT vp contribution
        self.Eprime = None # total energy WITH vp contribution
        self.D = None # on AO, (Da, Db)
        self.nroots = 1
        # ---------------------------------------
        # The following are unnecessary attributes. You might want to trigger them when necessary.
        # self.V = self.mol.intor('int1e_nuc')
        # self.T = self.mol.intor('int1e_kin')
        # self.eri = self.mol.intor('int2e') # 4-rank, large.
        # self.Exc = None
        # self.EH = None
        # self.Eext = None
        
    def get_rdm1(self):
        Da, Db = self.hfsolver.make_rdm1()
        
        # checker, could be deleted.
        assert np.isclose(np.sum(Da * self.S), self.mol.nelec[0], atol=1e-3), (np.sum(Da * self.S), self.mol.nelec[0])
        assert np.isclose(np.sum(Db * self.S), self.mol.nelec[1], atol=1e-3), (np.sum(Db * self.S), self.mol.nelec[1])
        
        self.D = np.array((Da, Db)) # save the density for ref and for initial guess for next calculation.
        return Da, Db
        
    def kernel(self, *args, Vp=None, dm0=None, **kwargs):
        """
        SCF runner.
        dm0: initial guess. If None, use the dm of last iteration.
        """
        if dm0 is None:
            dm0 = self.D
        
        if Vp is not None:
            self.hfsolver.Vp = Vp # will trigger the get_hcore() function to give T+V+Vp
            self.hfsolver.kernel(*args, dm0=dm0, **kwargs)
            Da, Db = self.get_rdm1()
            self.Eprime = self.solver.e_tot 
            self.E = self.Eprime - np.sum((Da+Db) * Vp) # subtract the vp contribution to the total energy to get Ef.
        else:
            self.hfsolver.Vp = 0.
            self.hfsolver.kernel(*args, dm0=dm0, **kwargs)
            Da, Db = self.get_rdm1()
            self.E = self.solver.e_tot
            self.Eprime = self.E
        return self.E

# DFT
class FragmentDFT:
    def _dm_to_DaDb(self, dm):
        arr = np.asarray(dm)

        if isinstance(dm, (tuple, list)) and len(dm) == 2:
            Da, Db = dm
        elif arr.ndim == 3 and arr.shape[0] == 2:
            Da, Db = arr[0], arr[1]
        elif arr.ndim == 2:
            Da = Db = 0.5 * arr
        else:
            raise ValueError(f"Unexpected dm shape: {arr.shape}")

        return np.array(Da, copy=True), np.array(Db, copy=True)

    def _cache_last_normal_dm(self, envs):
        cyc = envs.get("cycle", None)
        if cyc is None or cyc < 0:
            return  

        dm = envs.get("dm", None)
        if dm is None:
            return

        Da, Db = self._dm_to_DaDb(dm)
        self._last_normal_dm = (Da, Db)
        self._last_normal_cycle = int(cyc)
    def __init__(self, mol, xc, metal=False,
                 smearing=None, newton=None,
                 sigma=None, smearing_method=None,
                 fix_spin_smearing=None):
        self.mol = mol

        if metal:
             if (mol.spin==0):
                 mf = dft.RKS(mol)
             else:
                 mf = dft.UKS(mol)
                 # RI-J + linear-dep handling (ghosts)
             mf.xc = xc
             mf = mf.density_fit(auxbasis='weigend')
             mf = remove_linear_dep_(mf, lindep=1e-4)

             mf.direct_scf = True
             mf.max_cycle = 200
             mf.conv_tol = 1e-6 
             mf.verbose = 4

             mf.grids.level = 3
             mf.grids.prune = treutler_prune
             mf.small_rho_cutoff = 1e-6
             mf.diis_space = 12
        else:
             mf = dft.UKS(mol)
             mf.xc = xc
        if smearing:
             if sigma is None:
                 sigma = 0.02 
             if smearing_method is None:
                 smearing_method = "fermi"
             if fix_spin_smearing is None:
                 fix_spin_smearing = (mol.spin != 0)
             mf = smearing_(mf, sigma=sigma, method=smearing_method, fix_spin=fix_spin_smearing)
             mf.level_shift = 0.2
             mf.damp = 0.15
        if newton:
             mf = mf.newton()
             mf.level_shift = 0.2
             mf.damp = 0.15

        if smearing and newton:
             raise ValueError("Choose smearing OR newton (not both).")

        self.dftsolver = mf
        self.solver = self.dftsolver
        self._last_normal_dm = None
        self._last_normal_cycle = None

        old_cb = getattr(self.dftsolver, "callback", None)

        def chained_cb(envs):
            if callable(old_cb):
                old_cb(envs)
            self._cache_last_normal_dm(envs)

        self.dftsolver.callback = chained_cb
        self.dftsolver.xc = xc
        # one could also specify the dft grid point 
        
        self.S = self.mol.intor('int1e_ovlp')
        self.E = None # total energy WITHOUT vp contribution
        self.Eprime = None # total energy WITH vp contribution
        self.D = None # on AO, (Da, Db)
        self.nroots = 1
        
        # ---------------------------------------
        # The following are unnecessary attributes. You might want to trigger them when necessary.
        # self.V = self.mol.intor('int1e_nuc')
        # self.T = self.mol.intor('int1e_kin')
        # self.eri = self.mol.intor('int2e') # 4-rank, large.
        # self.Exc = None
        # self.EH = None
        # self.Eext = None
            
    def get_rdm1(self):
        if getattr(self, "_last_normal_dm", None) is not None:
            Da, Db = self._last_normal_dm
        else:
            dm = self.dftsolver.make_rdm1()
            Da, Db = self._dm_to_DaDb(dm)
            dm = self.dftsolver.make_rdm1()
            arr = np.asarray(dm)
        
        self.D =np.array((Da, Db))
        #checker, could be deleted.
        #assert np.isclose(np.sum((Da+Db) * self.S), self.mol.nelec[], atol=1e-3), (np.sum(Da * self.S), self.mol.nelec[0])
        #assert np.isclose(np.sum(Db * self.S), self.mol.nelec[1], atol=1e-3), (np.sum(Db * self.S), self.mol.nelec[1])
        if self.D is None:
            raise RuntimeError(" self.D is None: SCF will fall back to minao.")

        return Da, Db

    #def mo_coeff(self):
        #Ca, Cb = self.dftsolver.mo_coeff
        
    def kernel(self, *args, Vp=None, dm0=None, **kwargs):
        """
        SCF runner.
        dm0: initial guess. If None, use the dm of last iteration.
        """
        if dm0 is None:
            if self.D is not None:
        # reuse previous density
               dm0 = self.D
        # collapse spin if RKS
               if isinstance(self.dftsolver, dft.rks.RKS):
                  arr = np.asarray(dm0)
                  if arr.ndim == 3 and arr.shape[0] == 2:
                      dm0 = arr[0] + arr[1]
        else:
        # first SCF: allow PySCF to construct its default guess
            dm0 = None

        print("dm0 check",dm0)
        if Vp is not None:
            self.dftsolver.Vp = Vp
            self.dftsolver.kernel(*args, dm0=dm0, **kwargs)
            Da, Db = self.get_rdm1()
            self.Eprime = self.solver.e_tot 
            self.E = self.Eprime - np.sum((Da+Db) * Vp) # subtract the vp contribution to the total energy to get Ef.
        else:
            self.dftsolver.Vp = 0.
            self.dftsolver.kernel(*args, dm0=dm0, **kwargs)
            Da, Db = self.get_rdm1()
            self.E = self.solver.e_tot
            self.Eprime = self.E
        return self.E

# CCSD
class FragmentCCSD:
    def __init__(self, myhf):
        self.myhf = myhf
        self._mo_coeff = myhf.mo_coeff
        self.mol = myhf.mol
        self.solver = self.ccsolver = cc.UCCSD(myhf)
        self.S = self.mol.intor('int1e_ovlp')
        self.E = None
        self.Eprime = None
        self.D = None
        self.nroots = 1
        
    def get_rdm1(self):
        Da, Db = self.ccsolver.make_rdm1()
        Da = self._mo_coeff[0] @ Da @ self._mo_coeff[0].T
        Db = self._mo_coeff[1] @ Db @ self._mo_coeff[1].T
        assert np.isclose(np.sum(Da * self.S), self.mol.nelec[0], atol=1e-3), (np.sum(Da * self.S), self.mol.nelec[0])
        assert np.isclose(np.sum(Db * self.S), self.mol.nelec[1], atol=1e-3), (np.sum(Db * self.S), self.mol.nelec[1])
        self.D = np.array((Da, Db))
        return Da, Db
        
    def kernel(self, *args, Vp=None, **kwargs):
        """I am not sure if there is a way to continue the CCSD calculations. I attempted to input t1 and t2. But not sure this is good. One should check this."""
        if Vp is not None:
            self.myhf.Vp = Vp
            self.myhf.kernel() # such that you have new MO consistent with the given Vp
            self._mo_coeff = self.myhf.mo_coeff
            self.solver.mo_coeff = self._mo_coeff
            
            self.ccsolver.kernel(t1=self.ccsolver.t1, t2=self.ccsolver.t2)
            Da, Db = self.get_rdm1()
            self.Eprime = self.solver.e_tot 
            self.E = self.Eprime - np.sum((Da+Db) * Vp)
        else:
            self.myhf.kernel()
            self._mo_coeff = self.myhf.mo_coeff
            self.solver.mo_coeff = self._mo_coeff
            self.ccsolver.kernel(t1=self.ccsolver.t1, t2=self.ccsolver.t2)
            Da, Db = self.get_rdm1()
            self.E = self.solver.e_tot
            self.Eprime = self.E
        return self.E
        


# CASSCF
class FragmentCASSCF:
    """
    Here I added the attributs self.omega and self.nroots. Those specifies the weights for the state-averaging. 
    It also gives you access to ensembles that have excited states.
    There is a difference between omega and omega_SA for this design.
    omega is the weight for the ensemble one needs to construct.
    omega_sa is the weight for SA-CASSCF to optimize the orbital.
    They must of the same length. Except when omega is not None but omega_sa is None, then by default omega will be omega_sa, which is ture for most cases.
    TODO: very importantly, I found we must use UHF+UCASSCF or RHF+CASSCF. Otherwise the implementation of Vp is abstent.
    """
    def __init__(self, myhf, ncas, nelec, omega=None, omega_sa=None):
        self.myhf = myhf
        is_uhf = isinstance(myhf, scf.uhf.UHF)
        assert np.isclose(self.myhf.Vp, 0.)
        self.mol = myhf.mol
        if omega is not None:
            if is_uhf:
                self.casscf_ = mcscf.UCASSCF(myhf, ncas, nelec) # state-specific CASSCF
            else:
                self.casscf_ = mcscf.CASSCF(myhf, ncas, nelec)
            if omega_sa is None:
                self.solver = self.casscf = self.casscf_.state_average(omega) # main CASSCF
            else:
                assert len(omega)==len(omega_sa)
                self.solver = self.casscf = self.casscf_.state_average(omega_sa) # main CASSCF
        else:
            # assert len(omega) == len(omega_sa)
            if is_uhf:
                self.solver = self.casscf_ = self.casscf = mcscf.UCASSCF(myhf, ncas, nelec) # state-specific CASSCF
            else:
                self.solver = self.casscf_ = self.casscf = mcscf.CASSCF(myhf, ncas, nelec) # state-specific CASSCF
        self.E = None
        self.Eprime = None
        self.Es = None
        self.Eprimes = None
        self.D = None # density matirces on AO, (Da, Db)
        self.Dhf = None # HF density matrices
        self.S = self.mol.intor('int1e_ovlp')
        self.omega = omega # this is now the weights of ensemble NOT necessarily the weight for sa-casscf.
        self.omega_sa = omega_sa
        self.nroots = len(omega) if omega is not None else 1
        if self.omega_sa is not None and self.omega is None:
            raise ValueError("It is invalid to have the omega as None but omega/weights for state averaging not None.")

    def update_omega(self, omega, omega_sa):
        if self.omega is None and self.omega_sa is None:
            raise ValueError("Currently this object is initialized without stateaveraging, thus one can not modify the weights.")
        elif omega is not None and omega_sa is None: # 
            self.omega = omega
            if self.omega_sa is None:
                self.casscf.weights = omega
        elif omega is None and omega_sa is not None:
            self.omega_sa = omega_sa
            self.casscf.weights = omega_sa
        else:
            self.omega = omega
            self.omega_sa = omega_sa
            self.casscf.weights = omega
        return
        
    def get_rdm1(self):
        """
        There are two ways to deal with density matrices. The SA-CASSCF will automatically give SA-DM. 
        But one could call self.casscf_.make_rdm1s, which is not SA-CASSCF but State-Specific-CASSCF, for each component with SA-CASSCF.ci[idx].
        I will use auto SA DM.
        """
        if self.omega_sa is None:
            Da, Db = self.casscf.make_rdm1s()
        else:
            Da = 0.
            Db = 0.
            for w, state_ci in zip(self.omega, self.casscf.ci):
                Da_, Db_ = self.casscf_.make_rdm1s(self.casscf.mo_coeff, state_ci)
                Da += w * Da_
                Db += w * Db_
                
        # checker, could be deleted.
        assert np.isclose(np.sum(Da * self.S), self.mol.nelec[0], atol=1e-3), (np.sum(Da * self.S), self.mol.nelec[0])
        assert np.isclose(np.sum(Db * self.S), self.mol.nelec[1], atol=1e-3), (np.sum(Db * self.S), self.mol.nelec[1])

        self.D = np.array((Da, Db))
        
        # I should probably also update HF dm. Currently, the only purpose is to provide initial guess for HF scf. But I do not know if HF scf is necessary. 
        self.Dhf = self.myhf.make_rdm1()
        return Da, Db

    def get_rdm1s(self):
        """
        get_rdm1 for each state in the ensemble.
        """
        if self.nroots == 1:
            Das, Dbs = self.casscf.make_rdm1s()
        else:
            Das = []
            Dbs = []
            for state_ci in self.casscf.ci:
                Da_, Db_ = self.casscf_.make_rdm1s(self.casscf.mo_coeff, state_ci)
                Das.append(Da_)
                Dbs.append(Db_)
        return np.array(Das), np.array(Dbs)
        
    def kernel(self, *args, dm0hf=None, Vp=None, mo_coeff=None, ci0=None, **kwargs):
        """
        Again the energy is state averaged by default. One should go to e_states for specific components.
        """
        if dm0hf is None:
            dm0hf = self.Dhf
        if mo_coeff is None:
            mo_coeff = self.casscf.mo_coeff  # there are two options: CASSCF orbitals with last Vp, or HF orbitals with current Vp. Your call.
        if ci0 is None:
            ci0 = self.casscf.ci
        
        if Vp is not None:
            self.myhf.Vp = Vp
            self.myhf.kernel(dm0=dm0hf) # Here, I do not know whether it is necessary to re-run the HF scf. I think it is not.
            self.casscf.kernel(*args, mo_coeff=mo_coeff, ci0=ci0, **kwargs)
        else:
            self.myhf.Vp = 0.
            self.myhf.kernel(dm0=dm0hf)
            self.casscf.kernel(*args, mo_coeff=mo_coeff, ci0=ci0, **kwargs)
        
        if self.omega is None:
            self.Eprime = self.solver.e_tot
        else:
            self.Eprime = 0.
            for idx, w in enumerate(self.omega):
                self.Eprime += self.solver.e_states[idx] * w
                
        self.Es = []
        self.Eprimes = []
        if Vp is None:
            self.E = self.Eprime
            if self.nroots == 1:
                self.Es = self.E
                self.Eprimes = self.Eprime
            else:
                for idx, w in enumerate(self.omega):
                    Eprime = self.solver.e_states[idx]
                    E = Eprime
                    self.Es.append(E)
                    self.Eprimes.append(Eprime)
        else:
            Da, Db = self.get_rdm1()
            self.E = self.Eprime - np.sum((Da+Db) * Vp)
            if self.nroots == 1:
                self.Es = self.E
                self.Eprimes = self.Eprime
            else:
                Das, Dbs = self.get_rdm1s()
                for idx, w in enumerate(self.omega):
                    Eprime = self.solver.e_states[idx]
                    E = Eprime - np.sum((Das[idx]+Dbs[idx]) * Vp)
                    self.Es.append(E)
                    self.Eprimes.append(Eprime)
        return self.E
    
# FCI
class FragmentFCI:
    def __init__(self, myhf, omega=None):
        self.myhf = myhf
        self._mo_coeff = myhf.mo_coeff
        assert np.isclose(self.myhf.Vp, 0.)
        self.mol = myhf.mol
        self.solver = self.fcisolver = fci.FCI(myhf)
        self.E = None
        self.Eprime = None
        self.Es = None
        self.Eprimes = None
        self.D = None # density matirces on AO, (Da, Db)
        self.S = self.mol.intor('int1e_ovlp')
        self.omega = omega # omega is the weight for the ens description. |><| = \sum_i \omega_i |i><i|.  Here I believe omega must be fixed.
        self.nroots = 1 if self.omega is None else len(self.omega)
        self.hcore = np.copy(self.myhf.get_hcore()) # T+V
        
    def get_rdm1(self):
        if self.omega is None:
            Da, Db = self.fcisolver.make_rdm1s(self.fcisolver.ci, self.fcisolver.norb, self.fcisolver.nelec)
        else:
            Da = 0.
            Db = 0.
            for idx, w in enumerate(self.omega):
                Da_, Db_ = self.fcisolver.make_rdm1s(self.fcisolver.ci[idx], self.fcisolver.norb, self.fcisolver.nelec)
                Da += w * Da_
                Db += w * Db_
        if self._mo_coeff.ndim == 3:
            Da = self._mo_coeff[0] @ Da @ self._mo_coeff[0].T
            Db = self._mo_coeff[1] @ Db @ self._mo_coeff[1].T
        elif self._mo_coeff.ndim == 2:
            Da = self._mo_coeff @ Da @ self._mo_coeff.T
            Db = self._mo_coeff @ Db @ self._mo_coeff.T
        else:
            raise ValueError(self._mo_coeff.ndim)
        assert np.isclose(np.sum(Da * self.S), self.mol.nelec[0], atol=1e-3), (np.sum(Da * self.S), self.mol.nelec[0])
        assert np.isclose(np.sum(Db * self.S), self.mol.nelec[1], atol=1e-3), (np.sum(Db * self.S), self.mol.nelec[1])
        self.D = np.array((Da, Db))
        
        return Da, Db
    
    def get_rdm1s(self):
        """get_rdm1 for each fragment"""
        if self.nroots == 1:
            Das, Dbs = self.fcisolver.make_rdm1s(self.fcisolver.ci, self.fcisolver.norb, self.fcisolver.nelec)
        else:
            Das = []
            Dbs = []
            for idx, w in enumerate(self.omega):
                Da_, Db_ = self.fcisolver.make_rdm1s(self.fcisolver.ci[idx], self.fcisolver.norb, self.fcisolver.nelec)
                if self._mo_coeff.ndim == 3:
                    Da = self._mo_coeff[0] @ Da_ @ self._mo_coeff[0].T
                    Db = self._mo_coeff[1] @ Db_ @ self._mo_coeff[1].T
                elif self._mo_coeff.ndim == 2:
                    Da = self._mo_coeff @ Da_ @ self._mo_coeff.T
                    Db = self._mo_coeff @ Db_ @ self._mo_coeff.T
                else:
                    raise ValueError(self._mo_coeff.ndim)
                Das.append(Da)
                Dbs.append(Db)
        return np.array(Das), np.array(Dbs)
        
    def kernel(self, *args, Vp=None, ci0=None, **kwargs):
        if ci0 is None:
            ci0 = self.fcisolver.ci
        if Vp is not None:
            if self._mo_coeff.ndim == 3:
                h1e = [reduce(np.dot, (self._mo_coeff[0].conj().T, self.hcore+Vp, self._mo_coeff[0])),
                    reduce(np.dot, (self._mo_coeff[1].conj().T, self.hcore+Vp, self._mo_coeff[1]))]
                self.fcisolver.kernel(*args, h1e=h1e, nroots=self.nroots, ci0=ci0, **kwargs)
            elif self._mo_coeff.ndim == 2:
                h1e = reduce(np.dot, (self._mo_coeff.conj().T, self.hcore+Vp, self._mo_coeff))
                self.fcisolver.kernel(*args, h1e=h1e, nroots=self.nroots, ci0=ci0, **kwargs)
            else:
                raise ValueError(self._mo_coeff.ndim)
        else:
            self.myhf.Vp = 0.
            self.fcisolver.kernel(*args, nroots=self.nroots, ci0=ci0, **kwargs)
            
        if self.omega is None:
            self.Eprime = self.fcisolver.e_tot
        else:
            self.Eprime = 0.
            for idx, w in enumerate(self.omega):
                self.Eprime += self.fcisolver.e_tot[idx] * w
                
        self.Es = []
        self.Eprimes = []
        if Vp is None:
            self.E = self.Eprime
            if self.nroots == 1:
                self.Es = self.E
                self.Eprimes = self.Eprime
            else:
                for idx, w in enumerate(self.omega):
                    Eprime = self.fcisolver.e_tot[idx]
                    E = Eprime
                    self.Es.append(E)
                    self.Eprimes.append(Eprime)
        else:
            Da, Db = self.get_rdm1()
            self.E = self.Eprime - np.sum((Da+Db) * Vp)
            if self.nroots == 1:
                self.Es = self.E
                self.Eprimes = self.Eprime
            else:
                Das, Dbs = self.get_rdm1s()
                for idx, w in enumerate(self.omega):
                    Eprime = self.fcisolver.e_tot[idx]
                    E = Eprime - np.sum((Das[idx]+Dbs[idx]) * Vp)
                    self.Es.append(E)
                    self.Eprimes.append(Eprime)
        return self.E
    
    
# ==============================================
# ensemble treatment
class ens():
    """
    this is an extra layer of ensemble for spin and number of electrons. with spin-restricted Vp.
    Ensembles of excitates states are calcualteds in FCI with nroots or SA-CASSCF.
    """
    def recursive_sum(self, t):
        total = 0
        for item in t:
            if isinstance(item, (list, tuple)):
                assert len(item) == 2, item
                total += sum(item)
            else:
                total += item
        return total
    
    def __init__(self, fragments, omega):
        """
        fragments: a list of fragment calculate defined above
        omega: a list ensemble weights.
            The current rule is, if the element is a number, it is the weight of the correspoinding fragment in self.fragments.
            If the element is itself a list of two numbers, then the a spin fliped result is automatically generated without calculation for those two numbers.
            All the numbers must sum to 1.
            E.g. (spin-up, spin-down)
            fragments = [|(4,5)>, |(5,5)>]
            omega = [(0.4, 0.4), 0.2]
            real ensemble:
            |ens> = 0.4 * |(4,5)> + 0.4 * |(5,4)> + 0.2 * |(5,5)>.
            None = state with no electron = |(0,0)>
        """
        assert np.isclose(self.recursive_sum(omega), 1), self.recursive_sum(omega)
        assert len(fragments) == len(omega)
        self.fragments = fragments
        self.omega = omega
        
    def scf(self, Vp, *args, **keywords):
        for frag in self.fragments:
            if frag is None:
                continue
            frag.kernel(Vp=Vp, *args, **keywords)
        return
    
    def get_D(self):
        Da = 0.
        Db = 0.
        for w, frag in zip(self.omega, self.fragments):
            if frag is None:
                continue
            Da_, Db_ = frag.get_rdm1()
            if isinstance(w, (list, tuple)):
                wa, wb = w
                Da += Da_ * wa
                Db += Db_ * wa
                # flip the dm
                Da += Db_ * wb
                Db += Da_ * wb
            else:
                Da += Da_ * w
                Db += Db_ * w
        return Da, Db
    
    def get_Ds(self):
        """return a list for each fragment. 
        The element of the list can also have multiple components (three dims with first-dim the excitate states) if we incluse some excited states.
        """
        Das = []
        Dbs = []
        for frag in self.fragments:
            if frag is None:
                Da_, Db_ = None, None
            else:
                if frag.nroots == 1:
                    Da_, Db_ = frag.get_rdm1()
                else:
                    Da_, Db_ = frag.get_rdm1s()
            Das.append(Da_)
            Dbs.append(Db_)
        return Das, Dbs
    
    def get_omegas(self):
        """
        omegas: all the omega of all the individual states |N,n>. N is the num of electron. n is the states/excitation.
            it will be [[wNn, wNn+1, wNn+2, ...], [wN+1n, wN+1n+1, wN+1n+2, ...], ...]
            sum_Nn (omegas) = 1
        excitation_omegas: excitation of each N of |N,n> with the difference:
            sum_n (excitation_omegas) = 1
        """
        omegas = []
        excitation_omegas = []
        for wN, frag in zip(self.omega, self.fragments):
            if isinstance(wN, tuple) or isinstance(wN, list):
                wN = sum(wN) # disgard spin
            if frag.nroots == 1:
                omegas.append([wN])
                excitation_omegas.append([1])
            else:
                omegas.append([wN * _ for _ in frag.omega])
                excitation_omegas.append([_ for _ in frag.omega])
        assert np.isclose(sum(sum(sublist) for sublist in omegas), 1), sum(sum(sublist) for sublist in omegas)
        return omegas, excitation_omegas
    
    def get_E(self):
        E = 0.
        for w, frag in zip(self.omega, self.fragments):
            if frag is None:
                continue
            if isinstance(w, (list, tuple)):
                wa, wb = w
                E += frag.E * (wa+wb)
            else:
                E += frag.E * w
        return E
    
    def get_Es(self):
        """return a list for each fragment. 
        The element of the list can also have multiple components (a list) if we incluse some excited states.
        """
        Es = []
        for frag in self.fragments:
            if frag is None:
                Es.append(None)
            else:
                if frag.nroots == 1:
                    Es.append(frag.E)
                else:
                    Es.append(frag.Es)
        return Es
    
    def get_Eprimes(self):
        """return a list for each fragment. 
        The element of the list can also have multiple components (a list) if we incluse some excited states.
        """
        Es = []
        for frag in self.fragments:
            if frag is None:
                Es.append(None)
            else:
                if frag.nroots == 1:
                    Es.append(frag.Eprime)
                else:
                    Es.append(frag.Eprimes)
        return Es
    
    def get_nelec(self):
        Nas = []
        Nbs = []
        for w, frag in zip(self.omega, self.fragments):
            if frag is None:
                Na_, Nb_ = 0, 0,
            else:
                Na_, Nb_ = frag.mol.nelec
            if isinstance(w, (list, tuple)):
                Nas.append((Na_, Nb_))
                Nbs.append((Nb_, Na_))
            else:
                Nas.append(Na_)
                Nbs.append(Nb_)
        return Nas, Nbs
    

