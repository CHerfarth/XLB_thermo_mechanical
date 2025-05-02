import warp as wp
from xlb.precision_policy import PrecisionPolicy
from typing import Any
import sympy
import numpy as np
import xlb.experimental.thermo_mechanical.solid_utils as utils
from xlb import DefaultConfig


# these are the global variables needed throughout the simulation
class SimulationParams:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._E = None
            cls._instance._nu = None
            cls._instance._T = None
            cls._instance._L = None
            cls._instance._kappa = None
            cls._instance._theta = None
            cls._instance._K = None
            cls._instance._mu = None
            cls._instance._lamb = None
            cls._instance._precision_policy = None

        return cls._instance

    def set_parameters(self, E, nu, dx, dt, L, T, kappa, theta, precision_policy=None):
        self._E_unscaled = E
        self._E = E
        self._nu_unscaled = nu
        self._nu = nu
        self._dx = dx
        self._dt = dt
        self._T = T
        self._L = L
        self._kappa = kappa
        self._theta = theta
        #self._precision_policy = precision_policy

        # Calculate derived parameters
        self._K_unscaled = E / (2 * (1 - nu))
        self._mu_unscaled = E / (2 * (1 + nu))
        self._lamb_unscaled = self._K_unscaled - self._mu_unscaled

        # Make material params dimensionless
        self._mu = self._mu_unscaled * self._T / (self._L * self._L * self._kappa)
        self._lamb = self._lamb_unscaled * self._T / (self._L * self._L * self._kappa)
        self._K = self._K_unscaled * self._T / (self._L * self._L * self._kappa)
        self._E = self._E * self._T / (self._L * self._L * self._kappa)

        #set precision policy
        if (precision_policy == None):
            self._precision_policy = DefaultConfig.default_precision_policy 
        else:
            self._precision_policy = precision_policy
        utils.get_updated_params()

    @property
    def precision_policy(self):
        return self._precision_policy

    @property
    def K_unscaled(self):
        return self._K_unscaled

    @property
    def mu_unscaled(self):
        return self._mu_unscaled

    @property
    def lamb_unscaled(self):
        return self._lamb_unscaled

    @property
    def E_unscaled(self):
        return self._E_unscaled

    @property
    def nu_unscaled(self):
        return self._nu_unscaled

    # Getter for E
    @property
    def E(self):
        return self._E

    # Getter for dx
    @property
    def dx(self):
        return self._dx

    # Getter for dt
    @property
    def dt(self):
        return self._dt

    # Getter for nu
    @property
    def nu(self):
        return self._nu

    # Getter for T
    @property
    def T(self):
        return self._T

    # Getter for L
    @property
    def L(self):
        return self._L

    # Getter for kappa
    @property
    def kappa(self):
        return self._kappa

    # Getter for theta
    @property
    def theta(self):
        return self._theta

    # Getter for K
    @property
    def K(self):
        return self._K

    # Getter for mu
    @property
    def mu(self):
        return self._mu

    # Getter for lamb
    @property
    def lamb(self):
        return self._lamb

    '''@property
    def precision_policy(self):
        return self._precision_policy'''
