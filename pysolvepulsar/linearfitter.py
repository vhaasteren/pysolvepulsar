#!/usr/bin/python
# -*- coding: utf-8 -*-
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab
"""
candidate:  Representation of candidate solutions

"""

from __future__ import print_function
from __future__ import division

import numpy as np
import scipy.linalg as sl
import logging

class LinearFitter(object):
    """Linear fitter that works, regardless of the condition number"""

    def __init__(self, **kwargs):
        """The values passed to the fitter must already obey the linear
        constraints.
        
        :param residuals:
            The timing residuals, as returned by the timing package (sec)

        :param toaerrs:
            TOA uncertainties

        :param Mj:
            The inter-patch jump matrix

        :param Mo:
            The total offset design matrix

        :param Mt:
            The timing model parameter design matrix

        :param Gj:
            The jump G-matrix, orthogonal to the Mj matrix

        :param parlabelst:
            Timing model parameter labels (list)

        :param parlabelso:
            Offset labels (list)

        :param Phivect:
            Prior covariance diagonal for the timing model parameters

        :param Phiveco:
            Prior covariance diagonal for the offset parameters

        :param prparst:
            Prior parameter values for the timing model

        :param prparso:
            Prior parameter values for the offsets

        :param candpars:
            Current timing model parameters from the candidate
        """
        # Read in the parameters
        for key in ['toaerrs', 'residuals', 'Mj', 'Mo', 'Mt', 'Gj',
                'parlabelst', 'parlabelso', 'Phivect', 'Phiveco', 'prparst',
                'prparso', 'candpars', '_logger']:
            setattr(self, key, kwargs.pop(key))

        # Set the normalization
        if 'normalize' in kwargs:
            self.do_normalize = kwargs.pop('normalize')
        else:
            self.do_normalize = True
        self._logger.info("Normalizing the solution: {0}".format(self.do_normalize))

        # Combine the timing model and the offset
        self.Mtot = np.append(self.Mt, self.Mo, axis=1)
        self.parlabels = self.parlabelst + self.parlabelso

        # Numbers of parameters and observations
        self.npart = self.Mtot.shape[1]
        self.nparj = self.Mj.shape[1]
        self.nobs = len(self.residuals)

        # Create the noise and prior matrices
        self.Nvec = self.toaerrs**2
        self.Phivec = np.append(self.Phivect, self.Phiveco)

        # Prior information
        self.Phivec = np.append(self.Phivect, self.Phiveco)
        self.Phivec_inv = 1.0 / self.Phivec
        self.prpars = np.append(self.prparst, self.prparso)
        self.prpars_delta = self.prpars - np.append(self.candpars, self.prparso)
        self.dpars = np.zeros_like(self.prpars)
        self.phipar = self.prpars_delta * self.Phivec_inv

        # Return matrices, init to zero
        self.MNt = np.zeros(self.npart)
        self.Mp = np.zeros_like(self.Mtot)
        self.MNM = np.zeros((self.npart, self.npart))
        self.Sigma_inv = np.zeros_like(self.MNM)
        self.Sigma= np.zeros_like(self.MNM)

        self.normalize(normalize=self.do_normalize, forward=True)

        # And finally, also subtract the jumps from the residuals (since
        # tempo2/PINT does not do that for us, and they are not in candidate
        # solutions)
        self.dt, self.jvals = self.subtract_jumps(self.residuals,
                self.Mj, self.Nvec)

    def get_tmp_normalization(self):
        """Given the prior, and the TOA uncertainties, get the design matrix
        normalizing constant"""
        nv = np.mean(self.Nvec)
        mu = np.sum(self.Mtot**2, axis=0)
        Phivec_inv = 1.0/self.Phivec
        return np.sqrt(mu/nv + Phivec_inv)

    def normalize(self, normalize=True, forward=True):
        """Perform the normalization for numerical stability"""
        self.norm = self.get_tmp_normalization() if normalize else 1.0

        if forward:
            self.Mtot_n = self.Mtot / self.norm
            self.Mp_n = self.Mp / self.norm
            self.MNt_n = self.MNt / self.norm          # TODO: Check this!!!!
            self.MNt_n = self.MNt * self.norm          # TODO: Check this!!!!
            self.Phivec_n = self.Phivec * self.norm**2
            self.Phivec_inv_n = self.Phivec_inv / self.norm**2
            self.MNM_n = ((self.MNM*self.norm).T*self.norm).T
            self.Sigma_n = ((self.Sigma*self.norm).T*self.norm).T
            self.Sigma_inv_n = ((self.Sigma_inv/self.norm).T/self.norm).T
            self.dpars_n = self.dpars * self.norm
            self.phipar_n = self.phipar / self.norm    # TODO: Check this!!!!
            self.phipar_n = self.phipar * self.norm    # TODO: Check this!!!!
            self.prpars_delta_n = self.prpars_delta * self.norm
            self.prpars_n = self.prpars * self.norm
        else:
            self.Mtot = self.Mtot_n * self.norm
            self.Mp = self.Mp_n * self.norm
            self.MNt = self.MNt_n * self.norm          # TODO: Check this!!!!
            self.MNt = self.MNt_n / self.norm          # TODO: Check this!!!!
            self.Phivec = self.Phivec_n / self.norm**2
            self.Phivec_inv = self.Phivec_inv_n * self.norm**2
            self.MNM = ((self.MNM_n/self.norm).T/self.norm).T
            self.Sigma = ((self.Sigma_n/self.norm).T/self.norm).T
            self.Sigma_inv = ((self.Sigma_inv_n*self.norm).T*self.norm).T
            self.dpars = self.dpars_n / self.norm
            self.phipar = self.phipar_n * self.norm    # TODO: Check this!!!!
            self.phipar = self.phipar_n / self.norm    # TODO: Check this!!!!
            self.prpars_delta = self.prpars_delta_n / self.norm
            self.prpars = self.prpars_n / self.norm

    def subtract_jumps(self, dt, Mj, Nvec):
        """Subtract the best-fit value of the patch jumps

        Because libstempo does not know about the inter-patch jumps, we need to
        remove the best-fit jumps from the residuals. Fortunately, we know that
        we have converged to a proper solution, so just fitting for the jumps
        should be sufficient.

        :param dt:
            The non-jump-subtracted residuals as given by libstempo

        :param Mj:
            The jump-only design matrix

        :param Nvec:
            The diagonal elements of the noise (co)variance matrix
        """
        MNM = np.dot(Mj.T / Nvec, Mj)       # Diagonal matrix
        Sigma = np.diag(1.0/np.diag(MNM))

        jvals = np.dot(Sigma, np.dot(Mj.T, dt / Nvec))
        dtj = np.dot(Mj, jvals)

        return dt - dtj, jvals

    def prior_inverse(self):
        """Perform the linear least-squares fit inverse, using only the prior"""
        # We know nothing but the prior. No fit for parameters
        self.Sigma_inv_n = np.diag(self.Phivec_inv_n)
        self.Sigma_n = np.diag(self.Phivec_n)

        self.dpars = np.dot(self.Sigma, self.phipar)
        # TODO: Check rp. Needs to work, even with pars =/= ML prior
        self.rp = 0.0 * self.dt         # Only true if pars chosen as ML prior
        self.rr = np.dot(self.Mtot_n, np.dot(self.Sigma_n, self.Mtot_n.T))

        self.Np = np.zeros((0,0))
        self.MNM_n = np.zeros((self.npart,self.npart))
        self.MNt_n = np.zeros(0)
        self.Mp_n = np.zeros((0, self.Mtot_n.shape[1]))
        self.dtp = np.zeros(0)
        self.Np_cf = (np.zeros((0,0)), False)

        self.loglik = 0.0
        self.loglik_ml = 0.0

        # Do this elsewhere
        # Transform the units back
        #self.dpars = self.dpars / self.norm
        #self.Sigma_inv = ((self.Sigma_inv*self.norm).T*self.norm).T
        #self.Sigma = ((self.Sigma/self.norm).T/self.norm).T

    def reversed_woodbury_inverse(self):
        """Perform the linear least-squares fit inverse, #pars > #eff. obsns"""
        self._logger.info("RevWood {0}, {1}".format(self.Gj.shape, self.Mtot_n.shape))
        self.Mp_n = np.dot(self.Gj.T, self.Mtot_n)
        self.dtp = np.dot(self.Gj.T, self.dt)
        self.Np = np.dot(self.Gj.T * self.Nvec, self.Gj)

        try:
            self.Np_cf = sl.cho_factor(self.Np)
            self.MNM_n = np.dot(self.Mp_n.T, sl.cho_solve(self.Np_cf, self.Mp_n))
            self.Sigma_inv_n = self.MNM_n + np.diag(self.Phivec_inv_n)

            C = self.Np + np.dot(self.Mp_n * self.Phivec, self.Mp_n.T)
            C_cf = sl.cho_factor(C)
            CiMP = sl.cho_solve(C_cf, self.Mp_n * self.Phivec_n)
            self.Sigma_n = np.diag(self.Phivec_n) - \
                    np.dot((self.Mp_n * self.Phivec_n).T, CiMP)
        except np.linalg.LinAlgError as err:
            print("Reverse Woodbury also has problems... :(")
            print("Perhaps try rank-reduced Cholesky code to invert Sigma_i?")
            # Use rank-reduced Cholesky code to invert Sigma_inv
            # self.Sigma = get_rr_CiA(Phivec_inv, self.Mp_n.T, 1.0/np.diag(self.Np), np.eye(len(Phivec_inv)))
            raise

    def canonical_woodbury_inverse(self):
        """Perform the linear least-squares fit inverse, #eff. obsns > #pars """
        self.Mp_n = np.dot(self.Gj.T, self.Mtot_n)
        self.dtp = np.dot(self.Gj.T, self.dt)
        self.Np = np.dot(self.Gj.T * self.Nvec, self.Gj)

        try:
            self.Np_cf = sl.cho_factor(self.Np)
        except np.linalg.LinAlgError as err:
            print("Cholesky, Np = ", self.Np)
            raise

        self.MNM_n = np.dot(self.Mp_n.T, sl.cho_solve(self.Np_cf, self.Mp_n))
        self.Sigma_inv_n = self.MNM_n + np.diag(self.Phivec_inv_n)

        try:
            Sigma_inv_cf = sl.cho_factor(self.Sigma_inv_n)
            self.Sigma_n = sl.cho_solve(Sigma_inv_cf, np.eye(len(self.MNM_n)))
        except np.linalg.LinAlgError as err:
            print("Cholesky, Sigma_inv = ", self.Sigma_inv_n)
            raise

    def finalize(self):
        """Finalize the fit, utilizing the already inverted Sigma matrix"""
        # Calculate the prediction quantities
        if len(self.dtp) > 0:
            self.MNt_n = np.dot(self.Mp_n.T, sl.cho_solve(self.Np_cf, self.dtp))
            dtNdt = np.dot(self.dtp, sl.cho_solve(self.Np_cf, self.dtp))
        else:
            self.MNt_n = np.zeros(self.Mp_n.shape[1])
            dtNdt = 0.0
        self.dpars_n = np.dot(self.Sigma_n, self.MNt_n + self.phipar_n)

        # TODO: should use dpars, instead of MNt below here???
        self.rp = np.dot(self.Mtot_n, np.dot(self.Sigma_n, self.MNt_n))   # Should be approx~0.0
        self.rr = np.dot(self.Mtot_n, np.dot(self.Sigma_n, self.Mtot_n.T))

        # Calculate the log-likelihood
        logdetN2 = np.sum(np.log(np.diag(self.Np_cf[0])))
        logdetphi2 = 0.5*np.sum(np.log(self.Phivec_n))
        chi2dt = 0.5*dtNdt
        chi2phi = 0.5*np.sum(self.prpars_delta_n**2/self.Phivec_n)
        chi2phi1 = 0.5*np.dot(self.dpars_n, np.dot(self.Sigma_inv_n, self.dpars_n))
        chi2_active = 0.5*np.dot(self.dpars_n, np.dot(self.Sigma_inv_n, self.dpars_n))

        # NOTE: chi2_active is zero _if_ we move to ML solution. We are dpars
        #       away from there. That's why we subtract it from loglik.
        #       Note also that, now, chi2phi1 and chi2_active are the same in
        #       this rescaling
        self.loglik = -logdetN2-logdetphi2-chi2dt-chi2phi+chi2phi1-chi2_active
        self.loglik_ml = -logdetN2-logdetphi2-chi2dt-chi2phi+chi2phi1

        # Transform the units back
        # Do this elsewhere
        #self.MNM = ((self.MNM/self.norm).T/self.norm).T
        #self.MNt = self.MNt / self.norm
        #self.Phivec = self.Phivec / self.norm**2
        #self.Phivec_inv = self.Phivec_inv*self.norm**2
        #self.Sigma_inv = ((self.Sigma_inv*self.norm).T*self.norm).T
        #self.Sigma = ((self.Sigma/self.norm).T*self.norm).T
        #self.dpars = self.dpars / self.norm

    def makedict(self):
        """Create a dictionary of all results"""
        dd = dict()
        dd['dt'] = self.dt                      # Timing residuals
        dd['dtp'] = self.dtp                    # Projected timing residuals
        dd['Mj'] = self.Mj                      # Jump design matrix
        dd['Mt'] = self.Mt                      # Timing model design matrix
        dd['Mo'] = self.Mo                      # Offset design matrix
        dd['Mtot'] = self.Mtot                  # Full design matrix
        dd['Mp'] = self.Mp                      # Projected design matrix
        dd['Gj'] = self.Gj                      # Jump design matrix G-matrix
        dd['Nvec'] = self.Nvec                  # Noise matrix diagonal
        dd['Np'] = self.Np                      # Projected noise matrix
        dd['Np_cf'] = self.Np_cf                # Cholesky factorized Np
        dd['MNM'] = self.MNM                    # Data-only Sigma^-1
        dd['MNt'] = self.MNt                    # Data-only parameters
        dd['Phivec'] = self.Phivec              # Prior diagnoal
        dd['Phivec_inv'] = self.Phivec_inv      # Inverse prior diagonal
        dd['Sigma_inv'] = self.Sigma_inv        # Parameter covariance (inv)
        dd['Sigma'] = self.Sigma                # Inverse parameter covariance
        dd['dpars'] = self.dpars                # Delta-parameters (the fit)
        dd['rp'] = self.rp                      # Residual projection
        # TODO: The abs here is totally incorrect!!!
        dd['stdrp'] = np.sqrt(np.diag(np.abs(self.rr)))      # Residuals projection std
        dd['parlabels'] = self.parlabels        # Parameter labels
        dd['loglik'] = self.loglik              # Log-likelihood
        dd['loglik_ml'] = self.loglik_ml        # Log-likelihood (ML)
        dd['newpars'] = self.candpars                # The candidate parameters
        dd['phipar'] = self.phipar                   # TODO: unnecessary?
        dd['norm'] = self.norm
        return dd

    def fit(self):
        """Perform the linear least-squares fit, and extrapolate"""
        if self.nparj == self.nobs:
            # No information in data
            self._logger.info("Doing the prior-inverse (nobs = nparj)")
            self.prior_inverse()
        elif self.nobs - self.nparj > self.npart:
            # More parameters than observations (under constrained)
            self._logger.info("Doing the canonical-Woodbury fit")
            self._logger.info("{0} (nobs) - {1} (nparj) > {2} (npart)".format(self.nobs, self.nparj, self.npart))
            self.canonical_woodbury_inverse()
        else:
            # More observations than parameters (over constrained)
            self._logger.info("Doing the reverse-Woodbury fit")
            self._logger.info("{0} (nobs) - {1} (nparj) < {2} (npart)".format(self.nobs, self.nparj, self.npart))
            self.reversed_woodbury_inverse()

        self.finalize()
        self.normalize(normalize=self.do_normalize, forward=False)

        return self.makedict()
