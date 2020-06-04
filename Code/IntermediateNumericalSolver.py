#!/usr/bin/env python3
# DataAnalysis.py
import scipy as sp
import numpy as np
import scipy.special as spec
from cycler import cycler
from scipy.integrate import trapz, solve_ivp
import scipy.signal
from scipy.fftpack import diff as psdiff
from numpy import gradient
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter,MultipleLocator
from copy import deepcopy
import texplot

### Debug options
plot_trig_funcs = False
plot_snapshots = False
plot_negative_snapshots = False
plot_pos_neg_snapshots = False
plot_skew_asymm = False
plot_skew_asymm_kh = False
plot_snapshots_cnoidal = False
plot_negative_snapshots_cnoidal = False
plot_pos_neg_snapshots_cnoidal = False
plot_skew_asymm_cnoidal = False
plot_skew_asymm_cnoidal_kh = False
plot_power_spec_Jeffreys = False
plot_power_spec_vs_time_Jeffreys = False
plot_double_power_spec_Jeffreys = False
plot_power_spec_GM = False
plot_power_spec_vs_time_GM = False
plot_pos_neg_snapshots_cnoidal_GM = False
plot_forcing_types = False

### Set global parameters
## Conversion factors
omega_to_T = 1/2/np.pi # conversion from omega to 1/T

## Physical parameters to convert from normalized, nondimensional
## variables to non-normalized, dimensional variables
eps = 0.1 # a/h
mu= 0.8 # (kh)**2
mu_solitary = 6*eps # (kh)**2 fixed for solitary waves

# For plots with a pair of epsilon and mu values
pairedEps = [eps, 2*eps]
pairedMu = [mu, 2*mu]

## Equation type: can be either 'KdVB' for KdV-Burgers (Jeffreys-type
## forcing) or 'KdVNL' for nonlocal KdV (Generalized Miles-type forcing)
diffeq = 'KdVB'

## Wind Phase; used for the (Generalized Miles-type) nonlocal
## KdV equation
psiP = 3/4*np.pi

## How long to run the code for
tLen = 3
tNum = tLen*10**4

## Old recommended code parameters: Radau solver, tLen=3, tNum=2**4
## Note: () represents sqrt(abs(C/B))
# For P = 0.1, xLen = 2**6*() and xNum = 2**8
# For P = 0.2, xLen = 2**7*() and xNum = 2**10
# For P = 0.3, xLen = 640*() and xNum = 2**13

## New recommended code parameters: Radau solver, tLen=3, tNum=2**4
# For P = 0.1, xLen = 2**6*() and xNum = 2**9
# For P = 0.2, xLen = 2**7*() and xNum = 2**10
# For P = 0.3, xLen = 2**8*() and xNum = 2**9

## Recommended code parameters: AB solver, tLen=20, tNum=10**5
# For P = 0.02, xLen = 20 and xNum = 200, H=0.0001
# For P = 0.2, xLen = 20 and xNum = 200, H=0.01

# Should be P=1 since we already pulled out the epsilon-factor, so P
# should be order-1. However, the system is too numerically unstable to
# actually use P=1, so decrease it for stability
P = 0.25

xLen = 10
xStep = 0.1

# Biharmonic viscosity; gives numerical stability
H = P*0.05

skew_asymm_tLen = 3
skew_asymm_tNum = 'density'
skew_asymm_Ps = np.array([-0.5,-0.25,0,0.25,0.5])
skew_asymm_array = np.ones(skew_asymm_Ps.shape)
skew_asymm_xLen = skew_asymm_array*10
skew_asymm_xLen_cnoidal = np.repeat('fit',skew_asymm_Ps.size)
skew_asymm_xStep = skew_asymm_array*0.1
skew_asymm_Hs = np.absolute(skew_asymm_Ps)*0.05

kh_mus = np.linspace(eps*6,eps*12,8)
kh_tLen = 1

# Values for GM
P_GM = 0.5
H_GM = 0

# Trig function plotting parameters`
trig_mode = 1

# FFT bandwidth
FFT_tLen = 9
FFT_num_peaks = 4

## Plotting parameters
JColor = '#8da0cb' # Jeffreys color (blue)
MColor = '#66c2a5' # Miles color (green)
GColor = '#fc8d62' # Generalized color (orange)
SColor = '#e7298a' # Stokes color (black)
JStyle = 'dashed' # Jeffreys line style (dashed)
GStyle = 'solid' # Generalized line style (solid)
SStyle = 'dotted' # Stokes line style (dotted)

class kdvSystem():

    def __init__(self, A=1, B=3/2, C=None, F=None, G=None, P=None, H=0,
            psiP=0, diffeq='KdVB', eps=0.1, mu=0.1):
        """Initialize kdvSystem class instance.

        Solves a KdV-Burgers equation of the form
        A*\partial_t y + F*\partial_x y + B*y*\partial_x y +
          C*\partial_xxx y = G \partial_xx y - H \partial_xxxx y
        with y(x,t) the dependent function.
        Here, B/A determines the nonlinearity, C/A determines the
        dispersivity, F/A corresponds to a background counter-current,
        and G/A determines the damping rate.
        Solves the system on a periodic domain.


        If F is not provided, choose value to give default, solitonic
        solution zero propagation velocity

        Parameters:
        A : float or None
            Sets the time scale of problem. Default is 1.
        B : float or None
            Sets the importance of the nonlinearity. Default is 3/2.
        C : float or None
            Sets the importance of the dispersivity. Default is 1/6.
        F : float, 'soltion_frame', or None
            Sets the background current strength. If 'soliton_frame', a
            value of -abs(B)/3*np.sign(C) is chosen to cancel the
            propagation of the default, solitonic solution with zero
            damping. Default is 0.
        G : float or None
            Sets the strength of the damping. If None, determine G from
            G=-1/2*P, where P is the pressure forcing strength. Default
            is 0.
        H : float or None
            Sets the strength of the "biviscosity" damping. Positive H
            is needed for stability when solving the KdV-Burgers
            equation with negative G. Default is 0.
        P : float or None
            Sets the strength of the pressure forcing. If None,
            determine P from P=-2*G, where G is the strength of the
            damping. Default is 0.
        psiP : float or None
            The wind phase, or the shift of the pressure relative to the
            surface height p(x,t) = eta(x+psiP,t). This is used for
            Generalized Miles-type forcings. Default is 0.
        diffeq : 'KdVB' or 'KdVNL'
            Specifies for which equation we are generating the sparcity
            matrix. 'KdVB' is KdV-Burgers, while 'KdVNL' is nonlocal
            KdV. Default is'KdVB'.
        """

        ## Physical parameters

        # sqrt(C/B) = width
        # A/B*sqrt(C/B) width/speed=travel time
        # A/G*C/B = decay time
        # G/sqrt(B*C) = travel time/decay time

        self.A = A
        self.B = B
        self.H = H

        self.eps = eps
        self.mu = mu

        self.psiP = psiP
        self.diffeq = diffeq

        if C is not None:
            self.C = C
        else:
            self.C = 1/6*mu/eps

        if F == 'soliton_frame':
            # Choose value to give default, solitonic solution zero
            # propagation velocity
            self.F = -abs(B)/3*np.sign(self.C)
        elif F is not None:
            self.F = F
        else:
            self.F = 0

        if G is not None and P is not None:
            raise ValueError('Cannot provide both G and P to kdvSystem constructor.')
        elif P is not None:
            self.P = P
            self.G = -1/2*P
        elif G is not None:
            self.G = G
            self.P = -2*G
        else:
            self.G = 0
            self.P = 0

        if self.diffeq == 'KdVNL' and P<0:
            # For nonlocal KdV (Generalized Miles), we always use P>0.
            # Since we previously used P<0 to denote wind blowing in the
            # opposite direction, simply change the sign of P (and G),
            # while flipping the sign of psiP (which *correctly* flips
            # the wind direction for KdVNL)
            self.P = -self.P
            self.G = -self.G
            self.psiP = -self.psiP

    def set_spatial_grid(self, xLen=2*np.pi, xNum=None,
            xStep=None, xOffset=None,
            WaveLength=2*np.pi,NumWaves=2):
        """Set the x coordinate grid.

        Parameters
        ----------
        xLen : float, 'fit', or None
            Length of x domain. If 'fit', choose xLen to fit NumWave
            wavelengths of length WaveLegngth. Default is 2*np.pi.
        xNum : float or None
            Number of grid points in x domain. If None, xStep
            must be specified. Default is None.
        xStep : float or None
            Spacing between grid points in x domain. If
            None, xNum must be specified. Default is None.
        xOffset : float, 'nice_value', or None
            Distance that origin is offset from the center of the
            domain. 'nice_value' gives a nice shift slightly to the
            right a distance of 8*np.sqrt(abs(self.C)/abs(self.B)).
            Default is 0.
        WaveLength : float or None
            Wave length. Default is 2*np.pi.
        NumWaves : float or None
            Number of (periodic) waves to include in domain. Default is
            2.

        Returns
        -------
        None
        """

        self.NumWaves = NumWaves
        self.WaveLength = WaveLength

        if xLen == 'fit':
            self.xLen = self.NumWaves*self.WaveLength
        else:
            self.xLen = xLen

        if xNum != None and xStep == None:
            self.xNum = xNum
        elif xNum == None and xStep != None:
            # xNum = xLen/xStep = NumWaves*WaveLength/xStep, but to
            # prevent rounding errors; round WaveLength/xStep, then
            # multiply by NumWaves
            self.xNum = int(round(self.WaveLength/xStep)*self.NumWaves)
        else:
            raise(ValueError('Exactly one of xNum or xStep must be specified'))

        self.dx = self.xLen / self.xNum # Usually it would be
          # dx=xLen/(xNum-1), but since we have periodic boundary
          # conditions, we don't need the -1

        # Offset origin from display window
        if xOffset == 'nice_value':
            self.xOffset = 8*np.sqrt(abs(self.C)/abs(self.B))
        elif xOffset is not None:
            self.xOffset = xOffset
        else:
            self.xOffset = 0

        self.x = np.linspace(
                -0.5*self.xLen-self.xOffset,
                0.5*self.xLen-self.xOffset,
                self.xNum,
                endpoint=False
                )

    def set_temporal_grid(self, tLen=10, tNum=2**4):
        """Set the t-coordinate grid.

        Parameters
        ----------
        tLen : float or None
            Length of t domain. Default is 10.
        tNum : float, 'density', or None
            Number of grid points in t domain. 'density' chooses tNum so
            that the temporal density (tLen/tNum) is equal to the
            spatial density (xLen/xNum) to the 4th power. Default is
            2**4.

        Returns
        -------
        None
        """

        self.tLen = tLen

        if tNum == 'density':
            self.tNum = int(self.tLen/(self.xLen/self.xNum)**4)
        else:
            self.tNum = tNum

        self.t, self.dt = np.linspace(0, self.tLen, self.tNum, retstep=True)


    def set_initial_conditions(self, y0='solitary', Height=None,
            redo_grids=False):
        """Set the initial conditions.
        Parameters
        ----------
        y0 : array_like, 'solitary', or 'cnoidal'
            Given initial condition. If array_like, must have same size as
            self.x array. 'solitary' gives a solitary wave profile which
            satisfies the unforced KdV equation. 'cnoidal' gives a
            cnoidal wave profile which satisfies the unforced KdV
            equation. Default is a 'solitary'.
        Height : float or None
            Height of initial condition. If None, then H is chosen to be
            2*sign(self.B*self.C). Default is None.
        redo_grids : boolean
            If True, re-adjust xLen to fit NumWaves, keeping the number
            of spatial points fixed, as well as updating tNum with
            'density' parameter

        Returns
        -------
        None
        """

        if Height is not None:
            self.Height = Height
        else:
            self.Height = 2*np.sign(self.B*self.C)

        if type(y0) != np.ndarray and (y0 == 'solitary' or y0 ==
                'cnoidal'):
            # If mu and eps are specified, then m is fixed
            m = self.Height*self.B/(3*self.C)

            # Round to sigFigs significant figures (to prevent machine
            # precision causing, eg, m = 1.0000000000000002 to throw and
            # error)
            # Source:
            # https://stackoverflow.com/questions/3410976/how-to-round-a-number-to-significant-figures-in-python
            sigFigs = 5
            m = round(m, sigFigs-1-int(np.floor(np.log10(np.absolute(m)))))
            if m > 1 or m < 0:
                raise(ValueError("m = 2*B/3/C*eps/mu must be at least 0"+
                    " and less than 1; m was calculated to be "+
                    str(m)))
            self.m = m

        if type(y0) == np.ndarray:
            self.y0 = y0
        elif y0 == 'solitary' or m == 1:
            if np.sign(self.Height) != np.sign(self.B*self.C):
                raise(ValueError("sgn(H) must equal sgn(B*C)"))
            # mu and eps must satisfy specific relationship for
            # solitary waves (corresponding to m=1; see below)
            if m != 1:
                raise(ValueError("Must have Height*B/(3*C) = 1 for "+
                    "solitary waves, but equals "+
                    str(self.Height*self.B/(3*self.C))))
            self.y0 = self.Height*1/np.cosh(np.sqrt(self.Height*self.B/self.C/12) \
                    *self.x)**2

        elif y0 == 'cnoidal':
            K = spec.ellipk(m)
            E = spec.ellipe(m)

            # Adjust the length of a wavelength to be 4*K(m) in
            # nondimensional units, since we want one wavelength to be
            # k*(x=lambda) = k*lambda = (2/Delta)*lambda = 4*K(m)
            # Note: we defined k=2/Delta so that it reduces to the usual
            # definition 2*pi/lambda for m=0
            self.WaveLength = 4*K

            # Also adjust xLen to fit the new wavelengths
            if redo_grids:
                self.set_spatial_grid(xLen='fit', xNum=self.xNum,
                        xOffset=self.xOffset,
                        WaveLength=self.WaveLength,
                        NumWaves=self.NumWaves)
                self.set_temporal_grid(tNum='density', tLen=self.tLen)

            cn = spec.ellipj(self.x/self.WaveLength*2*K,m)[1]
            trough = self.Height/m*(1-m-E/K)
            self.y0 = trough + self.Height*cn**2

        else:
            raise(ValueError("y0 must be array_like, 'solitary', or 'cnoidal'"))

    def set_x_window(self, xMin=None, xMax=None):
        """Set x_window, the portion of the x domain to plot.

        Parameters
        ----------
        xMin : float, 'nice_value', or None
            Minimum x coordinate of x_window. 'nice_value" gives a nice
            window of -17.5*np.sqrt(abs(C)/abs(B))-xOffset. Default is
            -xLen/2-xOffset.
        xMax : float, 'nice_value' or None
            Maximum x coordinate of x_window. 'nice_value' gives a nice
            window of 20*np.sqrt(abs(C)/abs(B))-xOffset. Default is
            xLen/2-xOffset.

        Returns
        -------
        None
        """

        if xMin == 'nice_value':
            self.xMin = -17.5*np.sqrt(abs(self.C)/abs(self.B))-self.xOffset
        elif xMin is not None:
            self.xMin = xMin
        else:
            self.xMin = -self.xLen/2-self.xOffset

        if xMax == 'nice_value':
            self.xMax = 20*np.sqrt(abs(self.C)/abs(self.B))-self.xOffset
        elif xMax is not None:
            self.xMax = xMax
        else:
            self.xMax = self.xLen/2-self.xOffset


        self.xWin = [self.xMin, self.xMax]

    def get_masked_x(self):
        """Get the x coordinate vector with coordinates outside x_window
        masked.

        Returns
        -------
        xMasked : numpy.ma.core.MaskedArray
            The x domain with regions outside x_window masked.
        """

        # Set x limits
        xMasked = np.ma.masked_outside(self.x,*self.xWin)
        return xMasked

    def set_snapshot_ts(self, snapshot_fracs=None):
        """Set the snapshot times.

        Set the snapshot times as a fraction of the total time domain
        length. Any snapshot times after the last time step are set to
        the last time step. Any snapshot times before the first time
        step are set to the first time step.

        Parameters
        ----------
        snapshot_fracs : array_like or None
            Times, as a fraction of the total time domain length, at
            which to display snapshots. Default is [0, 0.2, 0.4, 0.6,
            0.8, 1.0]

        Returns
        -------
        None
        """

        if snapshot_fracs is not None:
            snapshot_fracs = np.array(snapshot_fracs)
        else:
            snapshot_fracs = np.array([0,0.2,0.4,0.6,0.8,1])

        # Convert snapshot fractions to snapshot times
        snapshot_ts = snapshot_fracs*self.tLen

        # Convert snapshot times to snapshot indexes
        snapshot_indxs = np.floor(snapshot_ts*self.tNum/self.tLen).astype(int)

        # Ensure that indexes do not underflow
        ts_too_big_indxs = snapshot_indxs < 0
        snapshot_indxs[ts_too_big_indxs] = 0

        # Ensure that indexes do not overflow
        ts_too_big_indxs = snapshot_indxs >= self.t.size
        snapshot_indxs[ts_too_big_indxs] = self.t.size-1

        # Get actual snapshot times
        snapshot_ts = np.around(self.t[snapshot_indxs],1)

        self.snapshot_indxs = snapshot_indxs
        self.snapshot_ts = snapshot_ts


    def _kdvb(self, t, u, periodic_deriv=False):
        """Differential equation for the KdV-Burgers equation."""
        if not periodic_deriv:
            # Compute the x derivatives using the finite-difference method
            ux = np.gradient(u, self.dx, axis=0)
            uxx = np.gradient(ux, self.dx, axis=0)
            uxxx = np.gradient(uxx, self.dx, axis=0)
        else:
            # Compute the x derivatives using the pseudo-spectral method
            ux = psdiff(u, period=self.xLen)
            uxx = psdiff(u, period=self.xLen, order=2)
            uxxx = psdiff(u, period=self.xLen, order=3)

        # Compute du/dt
        dudt = -u*ux*self.B/self.A -uxxx*self.C/self.A \
                -ux*self.F/self.A + uxx*self.G/self.A

        return dudt

    def _kdvnl(self, t, u, periodic_deriv=False):
        """Differential equation for the nonlocal-KdV equation."""
        if not periodic_deriv:
            # Compute the x derivatives using the finite-difference method
            ux = np.gradient(u, self.dx, axis=0)
            uxx = np.gradient(ux, self.dx, axis=0)
            uxxx = np.gradient(uxx, self.dx, axis=0)
            uxxxx = np.gradient(uxxx, self.dx, axis=0)
            uxnl = np.roll(ux,
                    shift=int(round(-self.psiP*self.WaveLength/(2*np.pi)/self.dx)),
                    axis=0)
        else:
            # Compute the x derivatives using the pseudo-spectral method
            ux = psdiff(u, period=self.xLen)
            uxxx = psdiff(u, period=self.xLen, order=3)
            uxxxx = psdiff(u, period=self.xLen, order=4)
            uxnl = np.roll(ux,
                    shift=int(round(-self.psiP*self.WaveLength/(2*np.pi)/self.dx)),
                    axis=0)

        # Compute du/dt
        dudt = -u*ux*self.B/self.A -uxxx*self.C/self.A \
                -ux*self.F/self.A + uxnl*self.G/self.A \
                -uxxxx*self.H/self.A

        return dudt

    def _JacSparcity(self):
        """Return the Jacobian sparcity matrix.

        Returns
        -------
        jacSparcity : numpy.ndarray
            An array with zeros representing where the Jacobian should
            not have entries, and ones where it may.
        """

        def _fill_banded(a, val, width=1):
            """
            Fill an array with a banded value.

            Source: https://stackoverflow.com/questions/16131972/extract-and-set-thick-diagonal-of-numpy-array

            Parameters
            ----------
            a : numpy.ndarray
                Array to fill.
            val : float
                Value to fill band with.
            width : integer or None
                Number of diagonals to fill. 1 corresponds to main diagonal,
                2 corresponds to main diagonal as well as one superdiagonal
                and 1 subdiagonal, etc. Default is 1.

            Returns
            -------
            a : numpy.ndarray
                Array with filled-in band of values.
            """
            mask = np.abs(np.add.outer(np.arange(a.shape[0]),
                -np.arange(a.shape[1]))) < width
            a[mask] = val
            return a

        # Since we only have 3rd derivatives, create a banded sparcity matrix
        # with ones on the three upper/lower off-diagonals
        zeroArray = np.zeros((self.x.size,self.x.size))
        jacSparcity = _fill_banded(zeroArray,1,4)
        if self.diffeq == 'KdVNL':
            shift = np.roll(np.identity(self.x.size),
                    shift=int(round(-self.psiP*self.WaveLength/(2*np.pi)/self.dx)), axis=0)
            jacSparcity = np.logical_or(jacSparcity,shift)

        return jacSparcity

    def solve_system_builtin(self, periodic_deriv=True):
        """Use builtin methods to solve the differential
        equation on a periodic domain. If self.diffeq == 'KdVB', solve
        the KdV-Burgers equation; if self.diffeq == 'KdVNL', solve the
        nonlocal KdV equation."""

        if self.diffeq == 'KdVB':
            # KdV-Burgers
            eqn = self._kdvb
        elif self.diffeq == 'KdVNL':
            # Nonlocal KdV
            eqn = self._kdvnl

        if not periodic_deriv:
            sol = sp.integrate.solve_ivp(
                    eqn,
                    (0,self.tLen),
                    self.y0,
                    t_eval=self.t,
                    method='Radau',
                    vectorized=True,
                    jac_sparsity=self._JacSparcity()
                    )
        else:
            sol = sp.integrate.solve_ivp(
                    lambda t,u: eqn(t,u,periodic_deriv=True),
                    (0,self.tLen),
                    self.y0,
                    t_eval=self.t,
                    method='RK23',
                    )

        if sol['status'] != 0:
            print(sol['message'])

        self.sol = sol['y']


    def solve_system_ab2(self):
        """Use 2nd-order Adams-Bashforth method to solve the
        differential equation on a periodic domain. If self.diffeq ==
        'KdVB', solve the KdV-Burgers equation; if self.diffeq ==
        'KdVNL', solve the nonlocal KdV equation."""

        dx = self.dx
        dt = self.dt

        nx = self.x.size
        nt = self.t.size

        dx2 = dx**2
        dx3 = dx**3
        dx4 = dx**4

        y = np.empty((nt,nx),dtype=np.float128)
        y[0,:] = self.y0

        i = 0
        y0 = y[0,:]
        y2 = np.concatenate(([y0[-1]], y0, [y0[0]]))
        y4 = np.concatenate((y0[-2:], y0, y0[0:2]))

        # Center difference with periodic boundary conditions
        dydx = (y2[2:] - y2[0:-2])/(2*dx)
        dy3dx3 = (y4[4:] - 2*y4[3:-1] + 2*y4[1:-3] - y4[0:-4])/(2*dx3)
        dy4dx4 = (y4[4:] - 4*y4[3:-1] + 6*y4[2:-2] - 4*y4[1:-3] +
                y4[0:-4])/(dx4)

        if self.diffeq == 'KdVB':
            dy2dx2 = (y2[2:] - 2*y2[1:-1] + y2[0:-2])/(dx2)
            RHS0 = -(self.F*dydx + self.B*y0*dydx + self.C*dy3dx3 -
                    self.G*dy2dx2 + self.H*dy4dx4)
        elif self.diffeq == 'KdVNL':
            dydxnl = np.roll(dydx,
                    shift=int(round(-self.psiP*self.WaveLength/(2*np.pi)/dx)),
                    axis=0)
            RHS0 = -(self.F*dydx + self.B*y0*dydx + self.C*dy3dx3 -
                    self.G*dydxnl + self.H*dy4dx4)

        # First step is an Euler step
        y[i+1,:] = y[i,:] + dt*RHS0
        i = i + 1

        RHS1 = RHS0

        # Second step is a 2nd order Adams-Bashforth step
        y0 = y[i,:]
        y2 = np.concatenate(([y0[-1]], y0, [y0[0]]))
        y4 = np.concatenate((y0[-2:], y0, y0[0:2]))

        # Center difference with periodic boundary conditions
        dydx = (y2[2:] - y2[0:-2])/(2*dx)
        dy3dx3 = (y4[4:] - 2*y4[3:-1] + 2*y4[1:-3] - y4[0:-4])/(2*dx3)
        dy4dx4 = (y4[4:] - 4*y4[3:-1] + 6*y4[2:-2] - 4*y4[1:-3] +
                y4[0:-4])/(dx4)

        if self.diffeq == 'KdVB':
            dy2dx2 = (y2[2:] - 2*y2[1:-1] + y2[0:-2])/(dx2)
            RHS0 = -(self.F*dydx + self.B*y0*dydx + self.C*dy3dx3 -
                    self.G*dy2dx2 + self.H*dy4dx4)
        elif self.diffeq == 'KdVNL':
            dydxnl = np.roll(dydx,
                    shift=int(round(-self.psiP*self.WaveLength/(2*np.pi)/dx)),
                    axis=0)
            RHS0 = -(self.F*dydx + self.B*y0*dydx + self.C*dy3dx3 -
                    self.G*dydxnl + self.H*dy4dx4)


        y[i+1,:] = y[i,:] + (dt/2)*(3*RHS0 - RHS1)
        i = i+1

        RHS2 = RHS1
        RHS1 = RHS0

        for i in range(2,nt-1):
            y0 = y[i,:]
            y2 = np.concatenate(([y0[-1]], y0, [y0[0]]))
            y4 = np.concatenate((y0[-2:], y0, y0[0:2]))

            # Center difference with periodic boundary conditions
            dydx = (y2[2:] - y2[0:-2])/(2*dx)
            dy3dx3 = (y4[4:] - 2*y4[3:-1] + 2*y4[1:-3] - y4[0:-4])/(2*dx3)
            dy4dx4 = (y4[4:] - 4*y4[3:-1] + 6*y4[2:-2] - 4*y4[1:-3] +
                    y4[0:-4])/(dx4)

            if self.diffeq == 'KdVB':
                dy2dx2 = (y2[2:] - 2*y2[1:-1] + y2[0:-2])/(dx2)
                RHS0 = -(self.F*dydx + self.B*y0*dydx + self.C*dy3dx3 -
                        self.G*dy2dx2 + self.H*dy4dx4)
            elif self.diffeq == 'KdVNL':
                dydxnl = np.roll(dydx,
                        shift=int(round(-self.psiP*self.WaveLength/(2*np.pi)/dx)),
                        axis=0)
                RHS0 = -(self.F*dydx + self.B*y0*dydx + self.C*dy3dx3 -
                        self.G*dydxnl + self.H*dy4dx4)

            y[i+1,:] = y[i,:] + (dt/12)*(23*RHS0 - 16*RHS1 + 5*RHS2)

            RHS2 = RHS1
            RHS1 = RHS0

        self.sol = y.transpose()

    def solve_system_rk3(self):
        """Use 3rd-order Runge-Kutta method to solve the
        differential equation on a periodic domain. If self.diffeq ==
        'KdVB', solve the KdV-Burgers equation; if self.diffeq ==
        'KdVNL', solve the nonlocal KdV equation."""

        # RHS function
        # y'(t,x) = f(t,y(t,x))
        def f(t,y0,dx):
            y2 = np.concatenate(([y0[-1]], y0, [y0[0]]))
            y4 = np.concatenate((y0[-2:], y0, y0[0:2]))

            # Center difference with periodic boundary conditions
            dydx = (y2[2:] - y2[0:-2])/(2*dx)
            dy3dx3 = (y4[4:] - 2*y4[3:-1] + 2*y4[1:-3] - y4[0:-4])/(2*dx**3)
            dy4dx4 = (y4[4:] - 4*y4[3:-1] + 6*y4[2:-2] - 4*y4[1:-3] +
                    y4[0:-4])/(dx**4)

            if self.diffeq == 'KdVB':
                dy2dx2 = (y2[2:] - 2*y2[1:-1] + y2[0:-2])/(dx**2)
                RHS = -(self.F*dydx + self.B*y0*dydx + self.C*dy3dx3 -
                        self.G*dy2dx2 + self.H*dy4dx4)
            elif self.diffeq == 'KdVNL':
                dydxnl = np.roll(dydx,
                        shift=int(round(-self.psiP*self.WaveLength/(2*np.pi)/dx)),
                        axis=0)
                RHS = -(self.F*dydx + self.B*y0*dydx + self.C*dy3dx3 -
                        self.G*dydxnl + self.H*dy4dx4)
            return RHS

        # Number of stages
        s = 3

        # Butcher table weights
        b = np.empty(s)
        b[0]=1/6
        b[1]=1/6
        b[2]=2/3

        # Butcher table nodes
        c = np.empty(s)
        c[0]=0
        c[1]=1
        c[2]=1/2

        # RK matrix
        a = np.array([[0,0,0],[1,0,0],[1/4,1/4,0]])

        dx = self.dx
        dt = self.dt

        nx = self.x.size
        nt = self.t.size

        y = np.empty((nt,nx),dtype=np.float128)
        y[0,:] = self.y0

        n = 0
        for n in range(0,nt-1):
            yn = y[n,:]

            # RK Stages
            k = np.zeros((yn.size,s))
            tn = n*dt
            for j in range(0,s):
              k[:,j] = f(tn+c[j]*dt,yn+dt*np.dot(k,a[j,:]),dx)

            y[n+1,:] = y[n,:] + dt*np.dot(k,b)

        self.sol = y.transpose()

    def get_snapshots(self):
        """Get the snapshots at times set by set_snapshot_ts.

        Return snapshots at times specified by set_snapshot_ts (must be
        called first).

        Returns
        -------
        snapshots : numpy.ndarray, shape(x.size,t.size)
            An array with columns containing solution values over the
            spatial domain, and rows corresponding to different time
            snapshots.
        """
        return self.sol[:,self.snapshot_indxs]


    def skewness(self):
        """Get the skewness of the solution calculated with
        solve_system.

        Returns
        -------
        skewness : ndarray(t.size,)
            Returns the skewness calculated at each time step.
        """

        # Cast to type double (float64) since fft (used in Hilbert for
        # asymmetry) is unable to hand long double (float128)
        sol = np.array(self.sol, dtype=np.float64)

        averageSolCubed = sp.integrate.trapz(
                sol**3,
                dx=self.dx,
                axis=0)/(self.xLen)
        averageSolSquared = sp.integrate.trapz(
                sol**2,
                dx=self.dx,
                axis=0)/(self.xLen)

        skewness = averageSolCubed/averageSolSquared**(3/2)

        return skewness


    def asymmetry(self):
        """Get the asymmetry of the solution calculated with
        solve_system.

        Returns
        -------
        asymmetry : ndarray(t.size,)
            Returns the asymmetry calculated at each time step.
        """

        # Cast to type double (float64) since fft (used in Hilbert for
        # asymmetry) is unable to hand long double (float128)
        sol = np.array(self.sol, dtype=np.float64)

        # Calculate the hilbert transform
        # Confusingly, sp.signal.hilbert gives the analytic signal, x <- x + i*H(x)
        solHilbert = (sp.signal.hilbert(sol,axis=0) - sol)/1j

        # Throw out imaginary part since it should be zero (to within rounding error)
        solHilbert = np.real(solHilbert)

        averageHilbertCubed = sp.integrate.trapz(
                solHilbert**3,
                dx=self.dx,
                axis=0)/(self.xLen)
        averageSolSquared = sp.integrate.trapz(
                sol**2,
                dx=self.dx,
                axis=0)/(self.xLen)

        asymmetry = averageHilbertCubed/averageSolSquared**(3/2)

        return asymmetry

    def maximum(self):
        """Get the height of the function at each time step.

        Returns
        -------
        height : ndarray(t.size,)
            Returns the maximum calculated at each time step.
        """

        maximum = np.amax(self.sol, axis=0) - np.amin(self.sol, axis=0)

        return maximum

    def boost_to_lab_frame(self,velocity=-1/6):
        """
        Parameters
        ----------
        speed : float, 'solitary', 'cnoidal', or None
            The velocity of the initial, wave frame relative to the lab
            frame in units of [x]/[t]. 'solitary' boosts into the
            co-moving frame of an unforced solitary wave. 'cnoidal'
            boosts into the co-moving frame of an unforced cnoidal wave.
            Default is -1/6.

        Returns
        -------
        None
        """

        if velocity == 'solitary':
            velocity = -(self.B*self.Height/3+self.F)/self.A
        elif velocity == 'cnoidal':
            m = self.m
            Height = self.Height
            K = spec.ellipk(m)
            E = spec.ellipe(m)

            velocity = -(self.F+2/3*self.B*Height/m*(1-1/2*m-3/2*E/K))/self.A

        # Convert the physical velocity to a coordinate velocity
        coord_vel = velocity/self.dx*self.dt

        sol = self.sol

        for time in range(sol.shape[1]):
            sol[:,time] = np.roll(sol[:,time],shift=int(round(coord_vel*time)))

        self.sol = sol


if(plot_trig_funcs):
    print("Computing the solution.")

    # Create KdV-Burgers or nonlocal KdV system
    builtinSolver = kdvSystem(B=0, F=0, G=0, H=0, C=1/6)
    # Set spatial grid
    builtinSolver.set_spatial_grid(xLen=2*sp.pi,xNum=64)
    # Set initial conditions
    builtinSolver.set_initial_conditions(np.sin(trig_mode*builtinSolver.x))
    # Copy setup for finite difference solver
    FDSolver = deepcopy(builtinSolver)
    # Set temporal grid
    builtinSolver.set_temporal_grid(
            tLen=2*np.pi/trig_mode**3*builtinSolver.A/builtinSolver.C,
            tNum=501)
    FDSolver.set_temporal_grid(
            tLen=2*np.pi/trig_mode**3*builtinSolver.A/builtinSolver.C,
            tNum=10**5)
    # Set snapshot times
    builtinSolver.set_snapshot_ts([0,1/3,2/3,1])
    FDSolver.set_snapshot_ts([0,1/3,2/3,1])
    # Solve KdV-Burgers system
    builtinSolver.solve_system_builtin()
    FDSolver.solve_system_rk3()

#    # Boost to co-moving frame (moving with velocity -1/6)
#    builtinSolver.boost_to_lab_frame(velocity=1/6)
#    FDSolver.boost_to_lab_frame(velocity=1/6)

    # Convert back to non-normalized variables
    # Convert from eta' = eta/a = eta/h/eps to eta'*eps = eta/a*eps = eta/h
    # (Primes denote the nondim variables used throughout this solver)
    builtinSnapshots = builtinSolver.get_snapshots()*eps
    FDSnapshots = FDSolver.get_snapshots()*eps

    # Hide solution outside of window
    builtinSolver.set_x_window()
    xMasked = builtinSolver.get_masked_x()

    # Normalize x by wavelength
    # Convert from x' = x*k_E to x'/lambda' = x*k_E/lambda/k_E = x/lambda
    # (Primes denote the nondim variables used throughout this solver)
    xMasked = xMasked/builtinSolver.WaveLength

    print("Plotting.")
    ## Color cycle
    num_lines = builtinSnapshots[1,:].size # Number of lines
    new_colors = [plt.get_cmap('YlGnBu')(1. * (i+1)/(num_lines+1)) for i in
            reversed(range(num_lines))] # add 1 to numerator and denominator
                                        # so we don't go all the way to 0
                                        # (it's too light to see)
    linestyles = [*((0,(3+i,i)) for i in range(num_lines))]
    plt.rc('axes', prop_cycle=(cycler('color', new_colors) +
                           cycler('linestyle', linestyles)))

    # Initialize figure
    fig, ax = texplot.newfig(0.9,nrows=2,sharex=True,sharey=False,golden=True)

    # Plot x'*lambda' = x*k_E/lambda/k_E = x/lambda
    # (Primes denote the nondim variables used throughout this solver)
    ax[1].set_xlabel(r'Distance $x/\lambda$')
    ax[0].set_title(r'Builtin Solver after exactly 1 Period: $a/h={eps}$, $kh = {kh}$'.format(
        eps=eps,kh=round(np.sqrt(mu_solitary),1)))
    ax[1].set_title(r'FD Solver after exactly 1 Period: $a/h={eps}$, $kh = {kh}$'.format(
        eps=eps,kh=round(np.sqrt(mu_solitary),1)))

    ax[0].plot(xMasked,builtinSnapshots)
    ax[1].plot(xMasked,FDSnapshots)

    # Note: divide by epsilon to convert from t_1 to the full time t
    fig.legend(np.around(builtinSolver.snapshot_ts/eps,1),
            title=r'Time $t \sqrt{g/h}$',
            loc='right')

    # Make background transparent
    fig.patch.set_alpha(0)

    texplot.savefig(fig,'../Figures/Trig-Funcs')

if(plot_snapshots):
    print("Computing the solution.")

    # Create KdV-Burgers or nonlocal KdV system
    snapshotSystem = kdvSystem(P=P,H=H,psiP=psiP,diffeq=diffeq, eps=eps,
            mu=mu_solitary)
    # Set spatial and temporal grid
    snapshotSystem.set_spatial_grid(xLen=xLen,xStep=xStep)
    snapshotSystem.set_temporal_grid(tLen=tLen,tNum='density')
    # Set initial conditions
    snapshotSystem.set_initial_conditions(y0='solitary')
    # Solve KdV-Burgers system
    snapshotSystem.solve_system_rk3()

    # Boost to co-moving frame
    snapshotSystem.boost_to_lab_frame(velocity='solitary')

    # Convert back to non-normalized variables
    # Convert from eta' = eta/a = eta/h/eps to eta'*eps = eta/a*eps = eta/h
    # (Primes denote the nondim variables used throughout this solver)
    snapshotSystem.set_snapshot_ts([0,1/3,2/3,1])
    snapshots = snapshotSystem.get_snapshots()*eps

    # Hide solution outside of window
    snapshotSystem.set_x_window(xMin=-4,xMax=5)
    xMasked = snapshotSystem.get_masked_x()

    # Normalize x by wavelength
    # Convert from x' = x*k_E to x'/lambda' = x*k_E/lambda/k_E = x/lambda
    # (Primes denote the nondim variables used throughout this solver)
    xMasked = xMasked/snapshotSystem.WaveLength

    print("Plotting.")
    ## Color cycle
    num_lines = snapshots[1,:].size # Number of lines
    new_colors = [plt.get_cmap('viridis')(1. * (i)/(num_lines)) for i in
            range(num_lines)]
    linestyles = [*((0,(3+i,i)) for i in range(num_lines))]
    plt.rc('axes', prop_cycle=(cycler('color', new_colors) +
                           cycler('linestyle', linestyles)))

    # Initialize figure
    fig, ax = texplot.newfig(0.9,golden=True)

    # Plot x'*lambda' = x*k_E/lambda/k_E = x/lambda
    # (Primes denote the nondim variables used throughout this solver)
    ax.set_xlabel(r'Distance $x/\lambda$')
    # Plot eta'*eps = eta/a*eps = eta/h
    # (Primes denote the nondim variables used throughout this solver)
    ax.set_ylabel(r'Surface Height $\eta / h$')
    # Convert P' = P*k/(rho_w*g)/eps to P'*eps = P*k/(rho_w*g)
    # (Primes denote the nondim variables used throughout this solver)
    ax.set_title(r'Surface Height vs Time: $a/h={eps}$, $kh = {kh}$, $P_J k/(\rho_w g) = {P}$'.format(
        eps=eps,kh=round(np.sqrt(mu_solitary),1),P=round(eps*P,3)))

    ax.plot(xMasked,snapshots)

    # Add arrow depicting wind direction
    arrowLeft = np.array([0.05,0.4])
    arrowRight = np.array([0.25,0.4])
    textBottom = (arrowLeft+arrowRight)/2 + np.array([0,0.05])
    spacing = np.array([0.6,0])
    ax.annotate(r'Phase'+'\n'+r'Speed', xy=textBottom, xycoords='axes fraction',
            ha='center',va='bottom',ma='center')
    ax.annotate('', xy=arrowLeft, xytext=arrowRight,
            xycoords="axes fraction", arrowprops={'arrowstyle': '<-',
                'shrinkA':1,'shrinkB':0})
    ax.annotate(r'Wind', xy=textBottom+spacing,
            xycoords='axes fraction', ha='center',va='bottom',
            ma='center')
    ax.annotate('', xy=arrowLeft+spacing, xytext=arrowRight+spacing,
            xycoords="axes fraction", arrowprops={'arrowstyle': '<-',
                'shrinkA':1,'shrinkB':0})

    # Note: divide by epsilon to convert from t_1 to the full time t
    fig.legend(np.around(snapshotSystem.snapshot_ts/eps,1),
            title=r'Time $t \sqrt{g/h}$',loc='right')

    # Make background transparent
    fig.patch.set_alpha(0)

    texplot.savefig(fig,'../Figures/Snapshots')

if(plot_negative_snapshots):
    print("Computing the solution.")

    # Create KdV-Burgers or nonlocal KdV system
    snapshotSystem = kdvSystem(P=-P,H=H,psiP=psiP,diffeq=diffeq,
            eps=eps, mu=mu_solitary)
    # Set spatial and temporal grid
    snapshotSystem.set_spatial_grid(xLen=xLen,xStep=xStep)
    snapshotSystem.set_temporal_grid(tLen=tLen,tNum='density')
    # Set initial conditions
    snapshotSystem.set_initial_conditions(y0='solitary')
    # Solve KdV-Burgers system
    snapshotSystem.solve_system_rk3()

    # Boost to co-moving frame
    snapshotSystem.boost_to_lab_frame(velocity='solitary')

    # Convert back to non-normalized variables
    # Convert from eta' = eta/a = eta/h/eps to eta'*eps = eta/a*eps = eta/h
    # (Primes denote the nondim variables used throughout this solver)
    snapshotSystem.set_snapshot_ts([0,1/3,2/3,1])
    snapshots = snapshotSystem.get_snapshots()*eps

    # Hide solution outside of window
    snapshotSystem.set_x_window(xMin=-4,xMax=5)
    xMasked = snapshotSystem.get_masked_x()

    # Normalize x by wavelength
    # Convert from x' = x*k_E to x'/lambda' = x*k_E/lambda/k_E = x/lambda
    # (Primes denote the nondim variables used throughout this solver)
    xMasked = xMasked/snapshotSystem.WaveLength

    print("Plotting.")
    ## Color cycle
    num_lines = snapshots[1,:].size # Number of lines
    new_colors = [plt.get_cmap('viridis')(1. * (i)/(num_lines)) for i in
            range(num_lines)]
    linestyles = [*((0,(3+i,i)) for i in range(num_lines))]
    plt.rc('axes', prop_cycle=(cycler('color', new_colors) +
                           cycler('linestyle', linestyles)))

    # Initialize figure
    fig, ax = texplot.newfig(0.9,golden=True)

    # Plot x'*lambda' = x*k_E/lambda/k_E = x/lambda
    # (Primes denote the nondim variables used throughout this solver)
    ax.set_xlabel(r'Distance $x/\lambda$')
    # Plot eta'*eps = eta/a*eps = eta/h
    # (Primes denote the nondim variables used throughout this solver)
    ax.set_ylabel(r'Surface Height $\eta / h$')
    # Convert P' = P*k/(rho_w*g)/eps to P'*eps = P*k/(rho_w*g)
    # (Primes denote the nondim variables used throughout this solver)
    ax.set_title(r'Surface Height vs Time: $a/h={eps}$, $kh = {kh}$, $P_J k/(\rho_w g) = {P}$'.format(
        eps=eps,kh=round(np.sqrt(mu_solitary),1),P=round(eps*(-P),3)))

    ax.plot(xMasked,snapshots)

    # Add arrow depicting wind direction
    arrowLeft = np.array([0.05,0.4])
    arrowRight = np.array([0.25,0.4])
    textBottom = (arrowLeft+arrowRight)/2 + np.array([0,0.05])
    spacing = np.array([0.6,0])
    ax.annotate(r'Phase'+'\n'+r'Speed', xy=textBottom, xycoords='axes fraction',
            ha='center',va='bottom',ma='center')
    ax.annotate('', xy=arrowLeft, xytext=arrowRight,
            xycoords="axes fraction", arrowprops={'arrowstyle': '<-',
                'shrinkA':1,'shrinkB':0})
    ax.annotate(r'Wind', xy=textBottom+spacing,
            xycoords='axes fraction', ha='center',va='bottom',
            ma='center')
    ax.annotate('', xy=arrowLeft+spacing, xytext=arrowRight+spacing,
            xycoords="axes fraction", arrowprops={'arrowstyle': '->',
                'shrinkA':1,'shrinkB':0})

    # Note: divide by epsilon to convert from t_1 to the full time t
    fig.legend(np.around(snapshotSystem.snapshot_ts/eps,1),
            title=r'Time $t \sqrt{g/h}$',loc='right')

    # Make background transparent
    fig.patch.set_alpha(0)

    texplot.savefig(fig,'../Figures/Snapshots-Negative')

if(plot_pos_neg_snapshots):
    print("Computing the solution.")

    posSnapshots = [None,None]
    negSnapshots = [None,None]
    xMasked = [None,None]
    for indx, eps_val in enumerate(pairedEps):
        # Create KdV-Burgers or nonlocal KdV system
        posSystem = kdvSystem(P=P,H=H,psiP=psiP,diffeq=diffeq,
                eps=eps_val, mu=eps_val/eps*mu_solitary)
        negSystem = kdvSystem(P=-P,H=H,psiP=psiP,diffeq=diffeq,
                eps=eps_val, mu=eps_val/eps*mu_solitary)
        # Set spatial and temporal grid
        posSystem.set_spatial_grid(xLen=xLen,xStep=xStep)
        negSystem.set_spatial_grid(xLen=xLen,xStep=xStep)
        posSystem.set_temporal_grid(tLen=tLen,tNum='density')
        negSystem.set_temporal_grid(tLen=tLen,tNum='density')
        # Set initial conditions
        posSystem.set_initial_conditions(y0='solitary')
        negSystem.set_initial_conditions(y0='solitary')
        # Solve KdV-Burgers system
        posSystem.solve_system_rk3()
        negSystem.solve_system_rk3()

        # Boost to co-moving frame
        posSystem.boost_to_lab_frame(velocity='solitary')
        negSystem.boost_to_lab_frame(velocity='solitary')

        # Convert back to non-normalized variables
        # Convert from eta' = eta/a = eta/h/eps to eta'*eps = eta/a*eps = eta/h
        # (Primes denote the nondim variables used throughout this solver)
        posSystem.set_snapshot_ts([0,1/3,2/3,1])
        negSystem.set_snapshot_ts([0,1/3,2/3,1])
        posSnapshots[indx] = posSystem.get_snapshots()*eps_val
        negSnapshots[indx] = negSystem.get_snapshots()*eps_val

        # Hide solution outside of window
        posSystem.set_x_window(xMin=-4,xMax=5)
        xMasked[indx] = posSystem.get_masked_x()

        # Normalize x by wavelength
        # Convert from x' = x*k_E to x'/lambda' = x*k_E/lambda/k_E = x/lambda
        # (Primes denote the nondim variables used throughout this solver)
        xMasked[indx] = xMasked[indx]/posSystem.WaveLength

    print("Plotting.")

    ## Color cycle
    # We should have the same number of lines for all eps/mu pairs,
    # so it doesn't matter which loop we calculate this on
    num_lines = posSnapshots[0][1,:].size # Number of lines
    new_colors = [plt.get_cmap('viridis')(1. *
        (i)/(num_lines)) for i in
            range(num_lines)]
    linestyles = [*((0,(3+i,i)) for i in range(num_lines))]
    plt.rc('axes', prop_cycle=(cycler('color', new_colors) +
                           cycler('linestyle', linestyles)))

    # Initialize figure
    fig, ax = texplot.newfig(0.9,nrows=2,ncols=2,sharex=True,sharey=True,golden=True)
    fig.set_tight_layout(False)

    # Adjust figure height
    figsize = fig.get_size_inches()
    fig.set_size_inches([figsize[0],figsize[1]*1.3])

    fig.subplots_adjust(left=0.175,right=0.825,top=0.8,bottom=0.125,hspace=0.3)

    for indx in [0,1]:
        # Plot x'*lambda' = x*k_E/lambda/k_E = x/lambda
        # (Primes denote the nondim variables used throughout this solver)
        ax[1,indx].set_xlabel(r'Distance $x/\lambda$')
        # Plot eta'*eps = eta/a*eps = eta/h
        # (Primes denote the nondim variables used throughout this solver)
        ax[indx,0].set_ylabel(r'$\eta / h$')

    fig.suptitle(r'Surface Height vs Time')

    for indx in [0,1]:
        # Convert P' = P*k/(rho_w*g)/eps to P'*eps = P*k/(rho_w*g)
        # (Primes denote the nondim variables used throughout this solver)

        ax[0,indx].set_title(r'\begin{{tabular}}{{c}}$\epsilon = {eps}$, $\mu = {mu}$\\$P_J k/(\rho_w g) = {P}$\end{{tabular}}'.format(
            P=round(eps*(P),3),eps=pairedEps[indx],mu=round(pairedEps[indx]/eps*mu_solitary,3)))
        ax[1,indx].set_title(r'$P_J k/(\rho_w g) = {P}$'.format(P=round(eps*(-P),3)))

        ax[0,indx].plot(xMasked[indx],posSnapshots[indx])
        ax[1,indx].plot(xMasked[indx],negSnapshots[indx])

    # Add arrow depicting wind direction
    arrowLeft = np.array([0.05,0.4])
    arrowRight = np.array([0.25,0.4])
    textBottom = (arrowLeft+arrowRight)/2 + np.array([0,0.05])
    spacing = np.array([0.6,0])

    for indx in [0,1]:
        ax[0,indx].annotate(r'Phase'+'\n'+r'Speed', xy=textBottom, xycoords='axes fraction',
                ha='center',va='bottom',ma='center')
        ax[0,indx].annotate('', xy=arrowLeft, xytext=arrowRight,
                xycoords="axes fraction", arrowprops={'arrowstyle': '<-',
                    'shrinkA':1,'shrinkB':0})
        ax[0,indx].annotate(r'Wind', xy=textBottom+spacing,
                xycoords='axes fraction', ha='center',va='bottom',
                ma='center')
        ax[0,indx].annotate('', xy=arrowLeft+spacing, xytext=arrowRight+spacing,
                xycoords="axes fraction", arrowprops={'arrowstyle': '<-',
                    'shrinkA':1,'shrinkB':0})
        ax[1,indx].annotate(r'Phase'+'\n'+r'Speed', xy=textBottom, xycoords='axes fraction',
                ha='center',va='bottom',ma='center')
        ax[1,indx].annotate('', xy=arrowLeft, xytext=arrowRight,
                xycoords="axes fraction", arrowprops={'arrowstyle': '<-',
                    'shrinkA':1,'shrinkB':0})
        ax[1,indx].annotate(r'Wind', xy=textBottom+spacing,
                xycoords='axes fraction', ha='center',va='bottom',
                ma='center')
        ax[1,indx].annotate('', xy=arrowLeft+spacing, xytext=arrowRight+spacing,
                xycoords="axes fraction", arrowprops={'arrowstyle': '->',
                    'shrinkA':1,'shrinkB':0})

    # Note: divide by epsilon to convert from t_1 to the full time t
    fig.legend(np.around(posSystem.snapshot_ts/eps,1),
            title=r'Time'+'\n'+r'$t \sqrt{g/h}$',loc='right')

    # Make background transparent
    fig.patch.set_alpha(0)

    texplot.savefig(fig,'../Figures/Snapshots-Positive-Negative')

if(plot_skew_asymm):

    maximums = [None]*(skew_asymm_Ps.size)
    skewnesses = [None]*(skew_asymm_Ps.size)
    asymmetries = [None]*(skew_asymm_Ps.size)
    t = None

    for idx,Pval in enumerate(skew_asymm_Ps):
        print("Computing the solution.")
        # Create KdV-Burgers or nonlocal KdV system
        skewAsymSystem = kdvSystem(P=Pval,
                H=skew_asymm_Hs[idx],psiP=psiP,diffeq=diffeq, eps=eps,
                mu=mu_solitary)
        # Set spatial and temporal grid
        skewAsymSystem.set_spatial_grid(
                xLen=skew_asymm_xLen[idx],
                xStep=skew_asymm_xStep[idx])
        skewAsymSystem.set_temporal_grid(tLen=skew_asymm_tLen,tNum=skew_asymm_tNum)
        # Set initial conditions
        skewAsymSystem.set_initial_conditions(y0='solitary')
        # Solve KdV-Burgers system
        skewAsymSystem.solve_system_rk3()

        # Boost to co-moving frame
        skewAsymSystem.boost_to_lab_frame(velocity='solitary')

        # Save timesteps
        # Note: divide by epsilon to convert from t_1 to the full time t
        t = skewAsymSystem.t/eps

        print("Computing the Height.")
        maximums[idx] = skewAsymSystem.maximum()

        print("Computing the Skewness.")
        skewnesses[idx] = skewAsymSystem.skewness()

        print("Computing the Asymmetry.")
        asymmetries[idx] = skewAsymSystem.asymmetry()

    maximums = np.array(maximums).transpose()
    # Normalize maximums by t=0 maximum
    maximums = maximums/maximums[0,:]
    skewnesses = np.array(skewnesses).transpose()
    asymmetries = np.array(asymmetries).transpose()

    print("Plotting.")

    ## Color cycle
    num_lines = skew_asymm_Ps.size # Number of lines
    # Make the colors go from blue to black to red
    MaxColorAbs = 0.4
    new_colors = [plt.get_cmap('twilight')((i-num_lines/2+1/2)*2*MaxColorAbs/(num_lines+1)+0.5)
            for i in range(num_lines)]
    # Make the first half dotted
    linestyles = [*((0,(1,1+i)) for i in reversed(range(round((num_lines-1)/2))))]
    # Make the second half dashed (with the middle one solid)
    linestyles.extend([*((0,(3+i,i)) for i in range(round((num_lines+1)/2)))])
    plt.rc('axes', prop_cycle=(cycler('color', new_colors) +
                               cycler('linestyle', linestyles)))

    # Initialize figure
    fig, ax = texplot.newfig(0.9,nrows=3,sharex=True,sharey='row',golden=True)

    # Adjust figure height
    figsize = fig.get_size_inches()
    fig.set_size_inches([figsize[0],figsize[1]*1.3])

    fig.set_tight_layout(False)
    fig.subplots_adjust(left=0.175,right=0.8,top=0.875,bottom=0.15)

    ax[-1].set_xlabel(r'Time $t \sqrt{g/h}$')
    ax[0].set_ylabel(r'Height')
    ax[1].set_ylabel(r'Skewness')
    ax[2].set_ylabel(r'Asymmetry')
    fig.suptitle(r'\begin{{tabular}}{{c}}Height, Skewness, and Asymmetry: \\ $a/h={eps}$, $kh = {kh}$\end{{tabular}}'.format(
        eps=eps,kh=round(np.sqrt(mu_solitary),1),P=P))

    # Put horizontal line at y=1
    ax[0].axhline(1, color='0.75')

    # Put horizontal line at A=0
    ax[2].axhline(0, color='0.75')

    lines = ax[0].plot(t,maximums)
    ax[1].plot(t,skewnesses)
    ax[2].plot(t,asymmetries)


    # Convert P' = P*k/(rho_w*g)/eps to P'*eps = P*k/(rho_w*g)
    # (Primes denote the nondim variables used throughout this solver)
    leg = fig.legend(lines, np.around(skew_asymm_Ps*eps,3),
            title=r'Pressure'+'\n'+r'Magnitude'+'\n'+r'$P_J k/(\rho_w g)$',
            loc='right')
    leg.get_title().set_multialignment('center')

    # Make background transparent
    fig.patch.set_alpha(0)

    texplot.savefig(fig,'../Figures/Skew-Asymm')

if(plot_skew_asymm_kh):

    maximums = [None]*(kh_mus.size)
    skewnesses = [None]*(kh_mus.size)
    asymmetries = [None]*(kh_mus.size)
    t = None

    for idx,mu_val in enumerate(kh_mus):
        print("Computing the solution.")
        # Create KdV-Burgers or nonlocal KdV system
        skewAsymSystem = kdvSystem(P=P, H=H,psiP=psiP,diffeq=diffeq,
                eps=eps/mu_solitary*mu_val, mu=mu_val)
        # Set spatial and temporal grid
        skewAsymSystem.set_spatial_grid(xLen=xLen, xStep=xStep)
        skewAsymSystem.set_temporal_grid(tLen=skew_asymm_tLen,tNum=skew_asymm_tNum)
        # Set initial conditions
        skewAsymSystem.set_initial_conditions(y0='solitary',redo_grids=True)
        # Solve KdV-Burgers system
        skewAsymSystem.solve_system_rk3()

        # Boost to co-moving frame
        skewAsymSystem.boost_to_lab_frame(velocity='solitary')

        # Save t_1 (NOT t, since eps differs for different runs) timesteps
        t = skewAsymSystem.t

        print("Computing the Height.")
        maximums[idx] = skewAsymSystem.maximum()

        print("Computing the Skewness.")
        skewnesses[idx] = skewAsymSystem.skewness()

        print("Computing the Asymmetry.")
        asymmetries[idx] = skewAsymSystem.asymmetry()

    maximums = np.array(maximums).transpose()
    # Normalize maximums by t=0 maximum
    maximums = maximums/maximums[0,:]
    skewnesses = np.array(skewnesses).transpose()
    asymmetries = np.array(asymmetries).transpose()

    # Only use last (t=1) value
    maximums = maximums[-1,:]
    skewnesses = skewnesses[-1,:]
    asymmetries = asymmetries[-1,:]

    print("Plotting.")

    ## Color cycle
    num_lines = skew_asymm_Ps.size # Number of lines
    # Make the colors go from blue to black to red
    MaxColorAbs = 0.4
    new_colors = [plt.get_cmap('twilight')((i-num_lines/2+1/2)*2*MaxColorAbs/(num_lines+1)+0.5)
            for i in range(num_lines)]
    # Make the first half dotted
    linestyles = [*((0,(1,1+i)) for i in reversed(range(round((num_lines-1)/2))))]
    # Make the second half dashed (with the middle one solid)
    linestyles.extend([*((0,(3+i,i)) for i in range(round((num_lines+1)/2)))])
    plt.rc('axes', prop_cycle=(cycler('color', new_colors) +
                               cycler('linestyle', linestyles)))

    # Initialize figure
    fig, ax = texplot.newfig(0.9,nrows=3,sharex=True,sharey=False,golden=True)

    # Adjust figure height
    figsize = fig.get_size_inches()
    fig.set_size_inches([figsize[0],figsize[1]*1.3])

    fig.set_tight_layout(False)
    fig.subplots_adjust(left=0.175,right=0.9,top=0.875,bottom=0.15)

    ax[-1].set_xlabel(r'Nondimensional Depth $kh$')
    ax[0].set_ylabel(r'Height')
    ax[1].set_ylabel(r'Skewness')
    ax[2].set_ylabel(r'Asymmetry')
    fig.suptitle(r'Height, Skewness, and Asymmetry: $t \epsilon \sqrt{{g/h}} = {t}$'.format(
        P=P,t=round(t[-1],0)))

    def mu2eps(x):
        return x/mu_solitary*eps
    def eps2mu(x):
        return x/eps*mu_solitary

    ax[0].secondary_xaxis('top',
            functions=(mu2eps,eps2mu)).set_xlabel(r'Nondimensional Height $\epsilon$')
    ax[1].secondary_xaxis('top',
            functions=(mu2eps,eps2mu)).set_xlabel(r'Nondimensional Height $\epsilon$')
    ax[2].secondary_xaxis('top',
            functions=(mu2eps,eps2mu)).set_xlabel(r'Nondimensional Height $\epsilon$')

    # Put horizontal line at y=1
    ax[0].axhline(1, color='0.75')

    # Put horizontal line at A=0
    ax[2].axhline(0, color='0.75')

    ax[0].plot(np.sqrt(kh_mus),maximums)
    ax[1].plot(np.sqrt(kh_mus),skewnesses)
    ax[2].plot(np.sqrt(kh_mus),asymmetries)

    # Make background transparent
    fig.patch.set_alpha(0)

    texplot.savefig(fig,'../Figures/Skew-Asymm-kh')

if(plot_snapshots_cnoidal):
    print("Computing the solution.")

    # Create KdV-Burgers or nonlocal KdV system
    snapshotSystem = kdvSystem(P=P,H=H,psiP=psiP,diffeq=diffeq, eps=eps, mu=mu)
    # Set spatial and temporal grid
    snapshotSystem.set_spatial_grid(xLen='fit',xStep=xStep)
    snapshotSystem.set_temporal_grid(tLen=tLen,tNum='density')
    # Set initial conditions
    snapshotSystem.set_initial_conditions(y0='cnoidal',redo_grids=True)
    # Solve KdV-Burgers system
    snapshotSystem.solve_system_rk3()

    # Boost to co-moving frame
    snapshotSystem.boost_to_lab_frame(velocity='cnoidal')

    # Convert back to non-normalized variables
    # Convert from eta' = eta/a = eta/h/eps to eta'*eps = eta/a*eps = eta/h
    # (Primes denote the nondim variables used throughout this solver)
    snapshotSystem.set_snapshot_ts([0,1/3,2/3,1])
    snapshots = snapshotSystem.get_snapshots()*eps

    # Hide solution outside of window
    snapshotSystem.set_x_window()
    xMasked = snapshotSystem.get_masked_x()

    # Normalize x by wavelength
    # Convert from x' = x*k_E to x'/lambda' = x*k_E/lambda/k_E = x/lambda
    # (Primes denote the nondim variables used throughout this solver)
    xMasked = xMasked/snapshotSystem.WaveLength

    print("Plotting.")
    ## Color cycle
    num_lines = snapshots[1,:].size # Number of lines
    new_colors = [plt.get_cmap('viridis')(1. * (i)/(num_lines)) for i in
            range(num_lines)]
    linestyles = [*((0,(3+i,i)) for i in range(num_lines))]
    plt.rc('axes', prop_cycle=(cycler('color', new_colors) +
                           cycler('linestyle', linestyles)))

    # Initialize figure
    fig, ax = texplot.newfig(0.9,golden=True)

    # Plot x'*lambda' = x*k_E/lambda/k_E = x/lambda
    # (Primes denote the nondim variables used throughout this solver)
    ax.set_xlabel(r'Distance $x/\lambda$')
    # Plot eta'*eps = eta/a*eps = eta/h
    # (Primes denote the nondim variables used throughout this solver)
    ax.set_ylabel(r'Surface Height $\eta / h$')
    # Convert P' = P*k/(rho_w*g)/eps to P'*eps = P*k/(rho_w*g)
    # (Primes denote the nondim variables used throughout this solver)
    ax.set_title(r'Surface Height vs Time: $a/h={eps}$, $kh = {kh}$, $P_J k/(\rho_w g) = {P}$'.format(
        eps=eps,kh=round(np.sqrt(mu),1),P=round(eps*P,3)))

    ax.plot(xMasked,snapshots)

    # Add arrow depicting wind direction
    arrowLeft = np.array([0.175,0.4])
    arrowRight = np.array([0.375,0.4])
    textBottom = (arrowLeft+arrowRight)/2 + np.array([0,0.05])
    spacing = np.array([0.45,0])
    ax.annotate(r'Phase'+'\n'+r'Speed', xy=textBottom, xycoords='axes fraction',
            ha='center',va='bottom',ma='center')
    ax.annotate('', xy=arrowLeft, xytext=arrowRight,
            xycoords="axes fraction", arrowprops={'arrowstyle': '<-',
                'shrinkA':1,'shrinkB':0})
    ax.annotate(r'Wind', xy=textBottom+spacing,
            xycoords='axes fraction', ha='center',va='bottom',
            ma='center')
    ax.annotate('', xy=arrowLeft+spacing, xytext=arrowRight+spacing,
            xycoords="axes fraction", arrowprops={'arrowstyle': '<-',
                'shrinkA':1,'shrinkB':0})

    # Note: divide by epsilon to convert from t_1 to the full time t
    fig.legend(np.around(snapshotSystem.snapshot_ts/eps,1),
            title=r'Time $t \sqrt{g/h}$',loc='right')

    # Make background transparent
    fig.patch.set_alpha(0)

    texplot.savefig(fig,'../Figures/Snapshots-Cnoidal')

if(plot_negative_snapshots_cnoidal):
    print("Computing the solution.")

    # Create KdV-Burgers or nonlocal KdV system
    snapshotSystem = kdvSystem(P=-P,H=H,psiP=psiP,diffeq=diffeq, eps=eps, mu=mu)
    # Set spatial and temporal grid
    snapshotSystem.set_spatial_grid(xLen='fit',xStep=xStep)
    snapshotSystem.set_temporal_grid(tLen=tLen,tNum='density')
    # Set initial conditions
    snapshotSystem.set_initial_conditions(y0='cnoidal',redo_grids=True)
    # Solve KdV-Burgers system
    snapshotSystem.solve_system_rk3()

    # Boost to co-moving frame
    snapshotSystem.boost_to_lab_frame(velocity='cnoidal')

    # Convert back to non-normalized variables
    # Convert from eta' = eta/a = eta/h/eps to eta'*eps = eta/a*eps = eta/h
    # (Primes denote the nondim variables used throughout this solver)
    snapshotSystem.set_snapshot_ts([0,1/3,2/3,1])
    snapshots = snapshotSystem.get_snapshots()*eps

    # Hide solution outside of window
    snapshotSystem.set_x_window()
    xMasked = snapshotSystem.get_masked_x()

    # Normalize x by wavelength
    # Convert from x' = x*k_E to x'/lambda' = x*k_E/lambda/k_E = x/lambda
    # (Primes denote the nondim variables used throughout this solver)
    xMasked = xMasked/snapshotSystem.WaveLength

    print("Plotting.")
    ## Color cycle
    num_lines = snapshots[1,:].size # Number of lines
    new_colors = [plt.get_cmap('viridis')(1. * (i)/(num_lines)) for i in
            range(num_lines)]
    linestyles = [*((0,(3+i,i)) for i in range(num_lines))]
    plt.rc('axes', prop_cycle=(cycler('color', new_colors) +
                           cycler('linestyle', linestyles)))

    # Initialize figure
    fig, ax = texplot.newfig(0.9,golden=True)

    # Plot x'*lambda' = x*k_E/lambda/k_E = x/lambda
    # (Primes denote the nondim variables used throughout this solver)
    ax.set_xlabel(r'Distance $x/\lambda$')
    # Plot eta'*eps = eta/a*eps = eta/h
    # (Primes denote the nondim variables used throughout this solver)
    ax.set_ylabel(r'Surface Height $\eta / h$')
    # Convert P' = P*k/(rho_w*g)/eps to P'*eps = P*k/(rho_w*g)
    # (Primes denote the nondim variables used throughout this solver)
    ax.set_title(r'Surface Height vs Time: $a/h={eps}$, $kh = {kh}$, $P_J k/(\rho_w g) = {P}$'.format(
        eps=eps,kh=round(np.sqrt(mu),1),P=round(eps*(-P),3)))

    ax.plot(xMasked,snapshots)

    # Add arrow depicting wind direction
    arrowLeft = np.array([0.175,0.4])
    arrowRight = np.array([0.375,0.4])
    textBottom = (arrowLeft+arrowRight)/2 + np.array([0,0.05])
    spacing = np.array([0.45,0])
    ax.annotate(r'Phase'+'\n'+r'Speed', xy=textBottom, xycoords='axes fraction',
            ha='center',va='bottom',ma='center')
    ax.annotate('', xy=arrowLeft, xytext=arrowRight,
            xycoords="axes fraction", arrowprops={'arrowstyle': '<-',
                'shrinkA':1,'shrinkB':0})
    ax.annotate(r'Wind', xy=textBottom+spacing,
            xycoords='axes fraction', ha='center',va='bottom',
            ma='center')
    ax.annotate('', xy=arrowLeft+spacing, xytext=arrowRight+spacing,
            xycoords="axes fraction", arrowprops={'arrowstyle': '->',
                'shrinkA':1,'shrinkB':0})

    # Note: divide by epsilon to convert from t_1 to the full time t
    fig.legend(np.around(snapshotSystem.snapshot_ts/eps,1),
            title=r'Time $t \sqrt{g/h}$',loc='right')

    # Make background transparent
    fig.patch.set_alpha(0)

    texplot.savefig(fig,'../Figures/Snapshots-Negative-Cnoidal')

if(plot_pos_neg_snapshots_cnoidal):
    print("Computing the solution.")

    posSnapshots = [None,None]
    negSnapshots = [None,None]
    xMasked = [None,None]
    for indx, (eps_val,mu_val) in enumerate(zip(pairedEps,pairedMu)):
        # Create KdV-Burgers or nonlocal KdV system
        posSystem = kdvSystem(P=P,H=H,psiP=psiP,diffeq=diffeq,
                eps=eps_val, mu=mu_val)
        negSystem = kdvSystem(P=-P,H=H,psiP=psiP,diffeq=diffeq,
                eps=eps_val, mu=mu_val)
        # Set spatial and temporal grid
        posSystem.set_spatial_grid(xLen='fit',xStep=xStep)
        negSystem.set_spatial_grid(xLen='fit',xStep=xStep)
        posSystem.set_temporal_grid(tLen=tLen,tNum='density')
        negSystem.set_temporal_grid(tLen=tLen,tNum='density')
        # Set initial conditions
        posSystem.set_initial_conditions(y0='cnoidal',redo_grids=True)
        negSystem.set_initial_conditions(y0='cnoidal',redo_grids=True)
        # Solve KdV-Burgers system
        posSystem.solve_system_rk3()
        negSystem.solve_system_rk3()

        # Boost to co-moving frame
        posSystem.boost_to_lab_frame(velocity='cnoidal')
        negSystem.boost_to_lab_frame(velocity='cnoidal')

        # Convert back to non-normalized variables
        # Convert from eta' = eta/a = eta/h/eps to eta'*eps = eta/a*eps = eta/h
        # (Primes denote the nondim variables used throughout this solver)
        posSystem.set_snapshot_ts([0,1/3,2/3,1])
        negSystem.set_snapshot_ts([0,1/3,2/3,1])
        posSnapshots[indx] = posSystem.get_snapshots()*eps_val
        negSnapshots[indx] = negSystem.get_snapshots()*eps_val

        # Hide solution outside of window
        posSystem.set_x_window()
        xMasked[indx] = posSystem.get_masked_x()

        # Normalize x by wavelength
        xMasked[indx] = xMasked[indx]/posSystem.WaveLength

    print("Plotting.")

    ## Color cycle
    # We should have the same number of lines for all eps/mu pairs,
    # so it doesn't matter which loop we calculate this on
    num_lines = posSnapshots[0][1,:].size # Number of lines
    new_colors = [plt.get_cmap('viridis')(1. * (i)/(num_lines)) for i in
            range(num_lines)]
    linestyles = [*((0,(3+i,i)) for i in range(num_lines))]
    plt.rc('axes', prop_cycle=(cycler('color', new_colors) +
                           cycler('linestyle', linestyles)))

    # Initialize figure
    fig, ax = texplot.newfig(0.9,nrows=2,ncols=2,sharex=True,sharey=True,golden=True)
    fig.set_tight_layout(False)

    # Adjust figure height
    figsize = fig.get_size_inches()
    fig.set_size_inches([figsize[0],figsize[1]*1.3])

    fig.subplots_adjust(left=0.175,right=0.825,top=0.8,bottom=0.125,hspace=0.3)

    for indx in [0,1]:
        # Plot x'*lambda' = x*k_E/lambda/k_E = x/lambda
        # (Primes denote the nondim variables used throughout this solver)
        ax[1,indx].set_xlabel(r'Distance $x/\lambda$')
        # Plot eta'*eps = eta/a*eps = eta/h
        # (Primes denote the nondim variables used throughout this solver)
        ax[indx,0].set_ylabel(r'$\eta / h$')

    fig.suptitle(r'Surface Height vs Time')

    for indx in [0,1]:
        # Convert P' = P*k/(rho_w*g)/eps to P'*eps = P*k/(rho_w*g)
        # (Primes denote the nondim variables used throughout this solver)

        ax[0,indx].set_title(r'\begin{{tabular}}{{c}}$\epsilon = {eps}$, $\mu = {mu}$\\$P_J k/(\rho_w g) = {P}$\end{{tabular}}'.format(
            P=round(eps*(P),3),eps=pairedEps[indx],mu=pairedMu[indx]))
        ax[1,indx].set_title(r'$P_J k/(\rho_w g) = {P}$'.format(
            P=round(eps*(-P),3),eps=pairedEps[indx],mu=pairedMu[indx]))

        ax[0,indx].plot(xMasked[indx],posSnapshots[indx])
        ax[1,indx].plot(xMasked[indx],negSnapshots[indx])

    # Add arrow depicting wind direction
    arrowLeft = np.array([0.175,0.4])
    arrowRight = np.array([0.375,0.4])
    textBottom = (arrowLeft+arrowRight)/2 + np.array([0,0.05])
    spacing = np.array([0.45,0])

    for indx in [0,1]:
        ax[0,indx].annotate(r'Phase'+'\n'+r'Speed', xy=textBottom, xycoords='axes fraction',
                ha='center',va='bottom',ma='center')
        ax[0,indx].annotate('', xy=arrowLeft, xytext=arrowRight,
                xycoords="axes fraction", arrowprops={'arrowstyle': '<-',
                    'shrinkA':1,'shrinkB':0})
        ax[0,indx].annotate(r'Wind', xy=textBottom+spacing,
                xycoords='axes fraction', ha='center',va='bottom',
                ma='center')
        ax[0,indx].annotate('', xy=arrowLeft+spacing, xytext=arrowRight+spacing,
                xycoords="axes fraction", arrowprops={'arrowstyle': '<-',
                    'shrinkA':1,'shrinkB':0})
        ax[1,indx].annotate(r'Phase'+'\n'+r'Speed', xy=textBottom, xycoords='axes fraction',
                ha='center',va='bottom',ma='center')
        ax[1,indx].annotate('', xy=arrowLeft, xytext=arrowRight,
                xycoords="axes fraction", arrowprops={'arrowstyle': '<-',
                    'shrinkA':1,'shrinkB':0})
        ax[1,indx].annotate(r'Wind', xy=textBottom+spacing,
                xycoords='axes fraction', ha='center',va='bottom',
                ma='center')
        ax[1,indx].annotate('', xy=arrowLeft+spacing, xytext=arrowRight+spacing,
                xycoords="axes fraction", arrowprops={'arrowstyle': '->',
                    'shrinkA':1,'shrinkB':0})

    # Note: divide by epsilon to convert from t_1 to the full time t
    fig.legend(np.around(posSystem.snapshot_ts/eps,1),
            title=r'Time'+'\n'+r'$t \sqrt{g/h}$',loc='right')

    # Make background transparent
    fig.patch.set_alpha(0)

    texplot.savefig(fig,'../Figures/Snapshots-Positive-Negative-Cnoidal')

if(plot_skew_asymm_cnoidal):

    maximums = [None,None]
    skewnesses = [None,None]
    asymmetries = [None,None]
    t = [None,None]

    for indx, (eps_val,mu_val) in enumerate(zip(pairedEps,pairedMu)):
        tmp_maximums = [None]*(skew_asymm_Ps.size)
        tmp_skewnesses = [None]*(skew_asymm_Ps.size)
        tmp_asymmetries = [None]*(skew_asymm_Ps.size)
        for idx,Pval in enumerate(skew_asymm_Ps):
            print("Computing the solution.")
            # Create KdV-Burgers or nonlocal KdV system
            skewAsymSystem = kdvSystem(P=Pval,
                    H=skew_asymm_Hs[idx],psiP=psiP,diffeq=diffeq,
                    eps=eps_val, mu=mu_val)
            # Set spatial and temporal grid
            skewAsymSystem.set_spatial_grid(
                    xLen=skew_asymm_xLen_cnoidal[idx],
                    xStep=skew_asymm_xStep[idx])
            skewAsymSystem.set_temporal_grid(tLen=skew_asymm_tLen,tNum=skew_asymm_tNum)
            # Set initial conditions
            skewAsymSystem.set_initial_conditions(y0='cnoidal',redo_grids=True)
            # Solve KdV-Burgers system
            skewAsymSystem.solve_system_rk3()

            # Boost to co-moving frame
            skewAsymSystem.boost_to_lab_frame(velocity='cnoidal')

            # Save timesteps
            # Note: divide by epsilon to convert from t_1 to the full time t
            t[indx] = skewAsymSystem.t/eps_val

            print("Computing the Height.")
            tmp_maximums[idx] = skewAsymSystem.maximum()

            print("Computing the Skewness.")
            tmp_skewnesses[idx] = skewAsymSystem.skewness()

            print("Computing the Asymmetry.")
            tmp_asymmetries[idx] = skewAsymSystem.asymmetry()

        tmp_maximums = np.array(tmp_maximums).transpose()
        # Normalize maximums by t=0 maximum
        maximums[indx] = tmp_maximums/tmp_maximums[0,:]
        skewnesses[indx] = np.array(tmp_skewnesses).transpose()
        asymmetries[indx] = np.array(tmp_asymmetries).transpose()

    print("Plotting.")

    ## Color cycle
    # We should have the same number of lines for all eps/mu pairs,
    # so it doesn't matter which loop we calculate this on
    num_lines = skew_asymm_Ps.size # Number of lines
    # Make the colors go from blue to black to red
    MaxColorAbs = 0.4
    new_colors = [plt.get_cmap('twilight')((i-num_lines/2+1/2)*2*MaxColorAbs/(num_lines+1)+0.5)
            for i in range(num_lines)]
    # Make the first half dotted
    linestyles = [*((0,(1,1+i)) for i in reversed(range(round((num_lines-1)/2))))]
    # Make the second half dashed (with the middle one solid)
    linestyles.extend([*((0,(3+i,i)) for i in range(round((num_lines+1)/2)))])
    plt.rc('axes', prop_cycle=(cycler('color', new_colors) +
                               cycler('linestyle', linestyles)))

    # Initialize figure
    fig, ax = texplot.newfig(0.9,nrows=3,ncols=2,sharex=True,sharey='row',golden=True)

    # Adjust figure height
    figsize = fig.get_size_inches()
    fig.set_size_inches([figsize[0],figsize[1]*1.3])

    fig.set_tight_layout(False)
    fig.subplots_adjust(left=0.175,right=0.8,top=0.875,bottom=0.15)

    ax[-1,0].set_xlabel(r'Time $t \sqrt{g/h}$')
    ax[-1,1].set_xlabel(r'Time $t \sqrt{g/h}$')
    ax[0,0].set_ylabel(r'Height')
    ax[1,0].set_ylabel(r'Skewness')
    ax[2,0].set_ylabel(r'Asymmetry')
    fig.suptitle(r'\begin{{tabular}}{{c}}Height, Skewness, and Asymmetry: \\ $a/h={eps}$, $kh = {kh}$\end{{tabular}}'.format(
        eps=eps,kh=round(np.sqrt(mu),1),P=P))

    for indx in [0,1]:
        # Put horizontal line at y=1
        ax[0,indx].axhline(1, color='0.75')

        # Put horizontal line at A=0
        ax[2,indx].axhline(0, color='0.75')

        lines = ax[0,indx].plot(t[indx],maximums[indx])
        ax[1,indx].plot(t[indx],skewnesses[indx])
        ax[2,indx].plot(t[indx],asymmetries[indx])

    # Convert P' = P*k/(rho_w*g)/eps to P'*eps = P*k/(rho_w*g)
    # (Primes denote the nondim variables used throughout this solver)
    leg = fig.legend(lines, np.around(skew_asymm_Ps*eps,3),
            title=r'Pressure'+'\n'+r'Magnitude'+'\n'+r'$P_J k/(\rho_w g)$',
            loc='right')
    leg.get_title().set_multialignment('center')

    # Make background transparent
    fig.patch.set_alpha(0)

    texplot.savefig(fig,'../Figures/Skew-Asymm-Cnoidal')

if(plot_skew_asymm_cnoidal_kh):

    maximums = [None]*(kh_mus.size)
    skewnesses = [None]*(kh_mus.size)
    asymmetries = [None]*(kh_mus.size)
    t = None

    for idx,mu_val in enumerate(kh_mus):
        print("Computing the solution.")
        # Create KdV-Burgers or nonlocal KdV system
        skewAsymSystem = kdvSystem(P=P, H=H,psiP=psiP,diffeq=diffeq,
                eps=eps, mu=mu_val)
        # Set spatial and temporal grid
        skewAsymSystem.set_spatial_grid(xLen=xLen, xStep=xStep)
        skewAsymSystem.set_temporal_grid(tLen=skew_asymm_tLen,tNum=skew_asymm_tNum)
        # Set initial conditions
        skewAsymSystem.set_initial_conditions(y0='cnoidal',redo_grids=True)
        # Solve KdV-Burgers system
        skewAsymSystem.solve_system_rk3()

        # Boost to co-moving frame
        skewAsymSystem.boost_to_lab_frame(velocity='cnoidal')

        # Save t_1 (NOT t, since eps differs for different runs) timesteps
        t = skewAsymSystem.t

        # Since we re-scaled the x-grid and t-grid, each value of mu_val
        # gives a different tNum and hence a different array size for
        # maximum/skewness/asymmetry
        #
        # Since we only use the first and last element, just save those
        # so all outputs are the same size
        print("Computing the Height.")
        maximums[idx] = skewAsymSystem.maximum()[[0,-1]]

        print("Computing the Skewness.")
        skewnesses[idx] = skewAsymSystem.skewness()[[0,-1]]

        print("Computing the Asymmetry.")
        asymmetries[idx] = skewAsymSystem.asymmetry()[[0,-1]]

    maximums = np.array(maximums).transpose()
    # Normalize maximums by t=0 maximum
    maximums = maximums/maximums[0,:]
    skewnesses = np.array(skewnesses).transpose()
    asymmetries = np.array(asymmetries).transpose()

    # Only use last (t=1) value
    maximums = maximums[-1,:]
    skewnesses = skewnesses[-1,:]
    asymmetries = asymmetries[-1,:]

    print("Plotting.")

    ## Color cycle
    num_lines = skew_asymm_Ps.size # Number of lines
    # Make the colors go from blue to black to red
    MaxColorAbs = 0.4
    new_colors = [plt.get_cmap('twilight')((i-num_lines/2+1/2)*2*MaxColorAbs/(num_lines+1)+0.5)
            for i in range(num_lines)]
    # Make the first half dotted
    linestyles = [*((0,(1,1+i)) for i in reversed(range(round((num_lines-1)/2))))]
    # Make the second half dashed (with the middle one solid)
    linestyles.extend([*((0,(3+i,i)) for i in range(round((num_lines+1)/2)))])
    plt.rc('axes', prop_cycle=(cycler('color', new_colors) +
                               cycler('linestyle', linestyles)))

    # Initialize figure
    fig, ax = texplot.newfig(0.9,nrows=3,sharex=True,sharey=False,golden=True)

    # Adjust figure height
    figsize = fig.get_size_inches()
    fig.set_size_inches([figsize[0],figsize[1]*1.3])

    fig.set_tight_layout(False)
    fig.subplots_adjust(left=0.175,right=0.9,top=0.875,bottom=0.15)

    ax[-1].set_xlabel(r'Nondimensional Depth $kh$')
    ax[0].set_ylabel(r'Height')
    ax[1].set_ylabel(r'Skewness')
    ax[2].set_ylabel(r'Asymmetry')
    fig.suptitle(r'Height, Skewness, and Asymmetry: $t \epsilon \sqrt{{g/h}} = {t}$'.format(
        P=P,t=round(t[-1],0)))

    # Put horizontal line at y=1
    ax[0].axhline(1, color='0.75')

    # Put horizontal line at A=0
    ax[2].axhline(0, color='0.75')

    ax[0].plot(np.sqrt(kh_mus),maximums)
    ax[1].plot(np.sqrt(kh_mus),skewnesses)
    ax[2].plot(np.sqrt(kh_mus),asymmetries)

    # Make background transparent
    fig.patch.set_alpha(0)

    texplot.savefig(fig,'../Figures/Skew-Asymm-Cnoidal-kh')

if(plot_power_spec_Jeffreys):
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    print("Computing the solution.")

    # Create KdV-Burgers or nonlocal KdV system
    posSystem = kdvSystem(P=P,H=H,diffeq='KdVB', eps=eps, mu=mu)
    negSystem = kdvSystem(P=-P,H=H,diffeq='KdVB', eps=eps, mu=mu)
    # Set spatial and temporal grid
    posSystem.set_spatial_grid(xLen='fit',xStep=xStep)
    negSystem.set_spatial_grid(xLen='fit',xStep=xStep)
    posSystem.set_temporal_grid(tLen=FFT_tLen,tNum='density')
    negSystem.set_temporal_grid(tLen=FFT_tLen,tNum='density')
    # Set initial conditions
    posSystem.set_initial_conditions(y0='cnoidal',redo_grids=True)
    negSystem.set_initial_conditions(y0='cnoidal',redo_grids=True)
    # Solve KdV-Burgers system
    posSystem.solve_system_rk3()
    negSystem.solve_system_rk3()

    # Boost to co-moving frame
    posSystem.boost_to_lab_frame(velocity='cnoidal')
    negSystem.boost_to_lab_frame(velocity='cnoidal')

    # Convert back to non-normalized variables
    # Convert from eta' = eta/a = eta/h/eps to eta'*eps = eta/a*eps = eta/h
    # (Primes denote the nondim variables used throughout this solver)
    posSystem.set_snapshot_ts(np.linspace(0,1,num=6))
    negSystem.set_snapshot_ts(np.linspace(0,1,num=6))
    posSnapshots = posSystem.get_snapshots()*eps
    negSnapshots = negSystem.get_snapshots()*eps

    # Resample (via interpolation) to get a higher FFT resolution
    repeat_times = 5
    posSnapshotsRepeated = np.tile(posSnapshots, (repeat_times,1))
    negSnapshotsRepeated = np.tile(negSnapshots, (repeat_times,1))

    # Take spatial FFT (scale FFT by 1/N and multiply by 2 since we
    # ignore the negative frequencies)
    posSnapshotsFFT = 2*np.fft.fft(posSnapshotsRepeated,
            axis=0)/posSystem.xNum/repeat_times
    negSnapshotsFFT = 2*np.fft.fft(negSnapshotsRepeated,
            axis=0)/negSystem.xNum/repeat_times

    # Convert to power spectrum (ie abs squared)
    posSnapshotsPower = np.absolute(posSnapshotsFFT)**2
    negSnapshotsPower = np.absolute(negSnapshotsFFT)**2

    # Generate spatial FFT conjugate coordinate
    kappa = np.fft.fftfreq(posSystem.xNum*repeat_times,
            posSystem.dx)*posSystem.WaveLength

    # Find peaks in initial data (peaks shouldn't move over time either)
    posSnapshotsPowerPeaks = sp.signal.find_peaks(posSnapshotsPower[:,0])[0]
    negSnapshotsPowerPeaks = sp.signal.find_peaks(negSnapshotsPower[:,0])[0]

    # Only keep non-negative kappa peaks
    posSnapshotsPowerPeaks = posSnapshotsPowerPeaks[kappa[posSnapshotsPowerPeaks] >= 0]
    negSnapshotsPowerPeaks = negSnapshotsPowerPeaks[kappa[negSnapshotsPowerPeaks] >= 0]

    # Sort the peaks
    posSnapshotsPowerPeaks = posSnapshotsPowerPeaks[np.argsort(
        posSnapshotsPower[posSnapshotsPowerPeaks,0])[::-1]]
    negSnapshotsPowerPeaks = negSnapshotsPowerPeaks[np.argsort(
        negSnapshotsPower[negSnapshotsPowerPeaks,0])[::-1]]

    # Only include first Power_num_peaks+1 (we need one extra since the
    # right-sided base always gives the right window limit, so we'll use
    # the n+1th left base limit to define the nth right base limit)
    posSnapshotsPowerPeaks = posSnapshotsPowerPeaks[0:FFT_num_peaks+1]
    negSnapshotsPowerPeaks = negSnapshotsPowerPeaks[0:FFT_num_peaks+1]

    # Find peak bases in initial data (peaks shouldn't move over time either)
    posSnapshotsPowerLeftBaseIndices = sp.signal.peak_prominences(
            posSnapshotsPower[:,0], posSnapshotsPowerPeaks)[1]
    negSnapshotsPowerLeftBaseIndices = sp.signal.peak_prominences(
            negSnapshotsPower[:,0], negSnapshotsPowerPeaks)[1]

    # Use the n+1th left base limit to define the nth right base limit
    posSnapshotsPowerRightBaseIndices = posSnapshotsPowerLeftBaseIndices[1:]
    posSnapshotsPowerLeftBaseIndices= posSnapshotsPowerLeftBaseIndices[:-1]
    negSnapshotsPowerRightBaseIndices = negSnapshotsPowerLeftBaseIndices[1:]
    negSnapshotsPowerLeftBaseIndices= negSnapshotsPowerLeftBaseIndices[:-1]

    # Take the largest of the two last base indices
    lastBaseIndex = np.amax([posSnapshotsPowerRightBaseIndices[-1],
        negSnapshotsPowerRightBaseIndices[-1]])

    # Store height (specifically, the largest height as a function of
    # time) and bases of second peak for inset image
    posSnapshotsPowerSecondPeakHeight = np.amax(posSnapshotsPower[
            posSnapshotsPowerPeaks[1]])
    negSnapshotsPowerSecondPeakHeight = np.amax(negSnapshotsPower[
            negSnapshotsPowerPeaks[1]])

    # Find 2nd peak bases in initial data (peaks shouldn't move over time either)
    # Re-calculate using 0.9999 prominence since we want it cropped
    # closely to the peak; the above base finding prescription would go
    # from the left side of peak 2 to the left side of peak 3, including
    # all the intervening flat space.
    posSnapshotsPowerSecondBaseIndices = sp.signal.peak_widths(
            posSnapshotsPower[:,0], [posSnapshotsPowerPeaks[1]],
            rel_height=0.9999)[2:]
    negSnapshotsPowerSecondBaseIndices = sp.signal.peak_widths(
            negSnapshotsPower[:,0], [negSnapshotsPowerPeaks[1]],
            rel_height=0.9999)[2:]

    # Convert from indices to kappa values by multiplying by the
    # kappa[1] (since all steps are evenly spaced)
    posSnapshotsPowerSecondBase = np.array(
            posSnapshotsPowerSecondBaseIndices)*kappa[1]
    negSnapshotsPowerSecondBase = np.array(
            negSnapshotsPowerSecondBaseIndices)*kappa[1]

    # Cut off after the last base (add 1 since we want the last base
    # *inclusive)
    kappa = kappa[0:lastBaseIndex+1]
    posSnapshotsPower = posSnapshotsPower[0:lastBaseIndex+1,:]
    negSnapshotsPower = negSnapshotsPower[0:lastBaseIndex+1,:]

    print("Plotting.")
    ## Color cycle
    num_lines = posSnapshots[1,:].size # Number of lines
    new_colors = [plt.get_cmap('viridis')(1. * (i)/(num_lines)) for i in
            range(num_lines)]
    linestyles = [*((0,(3+i,i)) for i in range(num_lines))]
    plt.rc('axes', prop_cycle=(cycler('color', new_colors) +
                           cycler('linestyle', linestyles)))

    # Initialize figure
    fig, ax = texplot.newfig(0.9,nrows=2,sharex=True,sharey=False,golden=True)
    fig.set_tight_layout(False)

    # Adjust figure height
    figsize = fig.get_size_inches()
    fig.set_size_inches([figsize[0],figsize[1]*1.3])

    fig.subplots_adjust(left=0.175,right=0.825,top=0.875,bottom=0.125,hspace=0.3)

    ax[1].set_xlabel(r'Harmonic $\kappa/k$')
    ax[0].set_ylabel(r'$\abs{\hat{\eta}}^2 k^2/h^2$')
    ax[1].set_ylabel(r'$\abs{\hat{\eta}}^2 k^2/h^2$')
    # Convert P' = P*k/(rho_w*g)/eps to P'*eps = P*k/(rho_w*g)
    # (Primes denote the nondim variables used throughout this solver)
    fig.suptitle(r'Power Spectrum vs Time: $a/h={eps}$, $kh = {kh}$'.format(
        eps=eps,kh=round(np.sqrt(mu),1)))
    ax[0].set_title(r'Co-Wind: $P_J k/(\rho_w g) = {P}$'.format(
        P=round(eps*(P),3)))
    ax[1].set_title(r'Counter-Wind: $P_J k/(\rho_w g) = {P}$'.format(
        P=round(eps*(-P),3)))

    ax[0].plot(kappa,posSnapshotsPower)
    ax[1].plot(kappa,negSnapshotsPower)

    # Put insets to zoom-in around second harmonic
    axins = [zoomed_inset_axes(ax[0], zoom=2.5, loc=1),
             zoomed_inset_axes(ax[1], zoom=10, loc=1)]

    axins[0].plot(kappa,posSnapshotsPower)
    axins[1].plot(kappa,negSnapshotsPower)

    axins[0].set_xlim(*posSnapshotsPowerSecondBase)
    axins[1].set_xlim(*negSnapshotsPowerSecondBase)
    axins[0].set_ylim(0,posSnapshotsPowerSecondPeakHeight*1.05)
    axins[1].set_ylim(0,negSnapshotsPowerSecondPeakHeight*1.05)

    mark_inset(ax[0], axins[0], loc1=4, loc2=2, fc="none", ec="0.5")
    mark_inset(ax[1], axins[1], loc1=4, loc2=2, fc="none", ec="0.5")

    # Turn off tick labels
    axins[0].set_yticklabels([])
    axins[0].set_xticklabels([])
    axins[1].set_yticklabels([])
    axins[1].set_xticklabels([])

    # Note: divide by epsilon to convert from t_1 to the full time t
    fig.legend(np.around(posSystem.snapshot_ts/eps,1),
            title=r'Time'+'\n'+r'$t \sqrt{g/h}$',loc='right')

    # Make background transparent
    fig.patch.set_alpha(0)

    texplot.savefig(fig,'../Figures/Power-Spectrum-Jeffreys')

if(plot_power_spec_vs_time_Jeffreys):
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    print("Computing the solution.")

    # Create KdV-Burgers or nonlocal KdV system
    posSystem = kdvSystem(P=P,H=H,psiP=psiP,diffeq='KdVB', eps=eps, mu=mu)
    negSystem = kdvSystem(P=-P,H=H,psiP=psiP,diffeq='KdVB', eps=eps, mu=mu)
    # Set spatial and temporal grid
    posSystem.set_spatial_grid(xLen='fit',xStep=xStep)
    negSystem.set_spatial_grid(xLen='fit',xStep=xStep)
    posSystem.set_temporal_grid(tLen=FFT_tLen,tNum='density')
    negSystem.set_temporal_grid(tLen=FFT_tLen,tNum='density')
    # Set initial conditions
    posSystem.set_initial_conditions(y0='cnoidal',redo_grids=True)
    negSystem.set_initial_conditions(y0='cnoidal',redo_grids=True)
    # Solve KdV-Burgers system
    posSystem.solve_system_rk3()
    negSystem.solve_system_rk3()

    # Boost to co-moving frame
    posSystem.boost_to_lab_frame(velocity='cnoidal')
    negSystem.boost_to_lab_frame(velocity='cnoidal')

    # Convert back to non-normalized variables
    # Convert from eta' = eta/a = eta/h/eps to eta'*eps = eta/a*eps = eta/h
    # (Primes denote the nondim variables used throughout this solver)
    posSystem.set_snapshot_ts(np.linspace(0,1,num=posSystem.tNum))
    negSystem.set_snapshot_ts(np.linspace(0,1,num=posSystem.tNum))
    posSnapshots = posSystem.get_snapshots()*eps
    negSnapshots = negSystem.get_snapshots()*eps

    # Save timesteps
    # Note: divide by epsilon to convert from t_1 to the full time t
    t = posSystem.t/eps

    # Resample (via interpolation) to get a higher FFT resolution
    repeat_times = 5
    posSnapshotsRepeated = np.tile(posSnapshots, (repeat_times,1))
    negSnapshotsRepeated = np.tile(negSnapshots, (repeat_times,1))

    # Take spatial FFT (scale FFT by 1/N and multiply by 2 since we
    # ignore the negative frequencies)
    posSnapshotsFFT = 2*np.fft.fft(posSnapshotsRepeated,
            axis=0)/posSystem.xNum/repeat_times
    negSnapshotsFFT = 2*np.fft.fft(negSnapshotsRepeated,
            axis=0)/negSystem.xNum/repeat_times

    # Convert to power spectrum (ie abs squared)
    posSnapshotsPower = np.absolute(posSnapshotsFFT)**2
    negSnapshotsPower = np.absolute(negSnapshotsFFT)**2

    # Generate spatial FFT conjugate coordinate
    kappa = np.fft.fftfreq(posSystem.xNum*repeat_times,
            posSystem.dx)*posSystem.WaveLength

    # Find peaks in initial data (peaks shouldn't move over time either)
    posSnapshotsPowerPeaks = sp.signal.find_peaks(posSnapshotsPower[:,0])[0]
    negSnapshotsPowerPeaks = sp.signal.find_peaks(negSnapshotsPower[:,0])[0]

    # Only keep non-negative kappa peaks
    posSnapshotsPowerPeaks = posSnapshotsPowerPeaks[kappa[posSnapshotsPowerPeaks] >= 0]
    negSnapshotsPowerPeaks = negSnapshotsPowerPeaks[kappa[negSnapshotsPowerPeaks] >= 0]

    # Sort the peaks
    posSnapshotsPowerPeaks = posSnapshotsPowerPeaks[np.argsort(
        posSnapshotsPower[posSnapshotsPowerPeaks,0])[::-1]]
    negSnapshotsPowerPeaks = negSnapshotsPowerPeaks[np.argsort(
        negSnapshotsPower[negSnapshotsPowerPeaks,0])[::-1]]

    # Only include first 3 peaks
    posSnapshotsPowerPeaks = posSnapshotsPowerPeaks[0:3]
    negSnapshotsPowerPeaks = negSnapshotsPowerPeaks[0:3]

    # Get indices for primary (m=1), first harmonic (m=2), and second
    # harmonic (m=3)
    posPrimaryIndex = posSnapshotsPowerPeaks[0]
    posFirstHarmonicIndex = posSnapshotsPowerPeaks[1]
    posSecondHarmonicIndex = posSnapshotsPowerPeaks[2]
    negPrimaryIndex = negSnapshotsPowerPeaks[0]
    negFirstHarmonicIndex = negSnapshotsPowerPeaks[1]
    negSecondHarmonicIndex = negSnapshotsPowerPeaks[2]

    print("Plotting.")
    ## Color cycle
    num_lines = posSnapshots[1,:].size # Number of lines
    new_colors = [plt.get_cmap('viridis')(1. * (i)/(num_lines)) for i in
            range(num_lines)]
    linestyles = [*((0,(3+i,i)) for i in range(num_lines))]
    plt.rc('axes', prop_cycle=(cycler('color', new_colors) +
                           cycler('linestyle', linestyles)))

    # Initialize figure
    fig, ax = texplot.newfig(0.9,nrows=2,sharex=True,sharey=False,golden=True)
    fig.set_tight_layout(False)

    # Adjust figure height
    figsize = fig.get_size_inches()
    fig.set_size_inches([figsize[0],figsize[1]*1.3])

    fig.subplots_adjust(left=0.125,right=0.775,top=0.875,bottom=0.125,hspace=0.3)

    ax[1].set_xlabel(r'Time $t\sqrt{g/h}$')

    # Convert P' = P*k/(rho_w*g)/eps to P'*eps = P*k/(rho_w*g)
    # (Primes denote the nondim variables used throughout this solver)
    fig.suptitle(r'Power Spectrum vs Time: $a/h={eps}$, $kh = {kh}$'.format(
        eps=eps,kh=round(np.sqrt(mu),1)))
    ax[0].set_title(r'Co-Wind: $P_J k/(\rho_w g) = {P}$'.format(
        P=round(eps*(P),3)))
    ax[1].set_title(r'Counter-Wind: $P_J k/(\rho_w g) = {P}$'.format(
        P=round(eps*(-P),3)))

    # Source: https://matplotlib.org/3.1.1/gallery/ticks_and_spines/multiple_yaxis_with_spines.html
    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    firstHarmonicAx = [None,None]
    secondHarmonicAx = [None,None]
    for indx in [0,1]:
        firstHarmonicAx[indx] = ax[indx].twinx()
        secondHarmonicAx[indx] = ax[indx].twinx()

        # Offset the right spine of secondHarmoincAx.
        # The ticks and label have already been placed on the right by twinx above.
        secondHarmonicAx[indx].spines["right"].set_position(("axes", 1.2))

        # Having been created by twinx, secondHarmonicAx has its frame
        # off, so the line of its detached spine is invisible.
        # First, activate the frame but make the patch and spines invisible.
        make_patch_spines_invisible(secondHarmonicAx[indx])

        # Second, show the right spine.
        secondHarmonicAx[indx].spines["right"].set_visible(True)

    line1 = ax[0].plot(t,posSnapshotsPower[posPrimaryIndex,:],'r')[0]
    line2 = firstHarmonicAx[0].plot(t,posSnapshotsPower[posFirstHarmonicIndex,:],'g')[0]
    line3 = secondHarmonicAx[0].plot(t,posSnapshotsPower[posSecondHarmonicIndex,:],'b')[0]
    ax[1].plot(t,negSnapshotsPower[negPrimaryIndex,:],'r')
    firstHarmonicAx[1].plot(t,negSnapshotsPower[negFirstHarmonicIndex,:],'g')
    secondHarmonicAx[1].plot(t,negSnapshotsPower[negSecondHarmonicIndex,:],'b')

    for indx in [0,1]:
        ax[indx].set_ylabel(r'Primary $\abs{\hat{\eta}}^2 k^2/h^2$')
        firstHarmonicAx[indx].set_ylabel(r'First Harmonic $\abs{\hat{\eta}}^2 k^2/h^2$')
        secondHarmonicAx[indx].set_ylabel(r'Second Harmonic $\abs{\hat{\eta}}^2 k^2/h^2$')

        ax[indx].yaxis.label.set_color(line1.get_color())
        firstHarmonicAx[indx].yaxis.label.set_color(line2.get_color())
        secondHarmonicAx[indx].yaxis.label.set_color(line3.get_color())

        ax[indx].tick_params(axis='y', colors=line1.get_color())
        firstHarmonicAx[indx].tick_params(axis='y', colors=line2.get_color())
        secondHarmonicAx[indx].tick_params(axis='y', colors=line3.get_color())
        ax[indx].tick_params(axis='x')

        ax[indx].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
        firstHarmonicAx[indx].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
        secondHarmonicAx[indx].ticklabel_format(style='sci',axis='y',scilimits=(0,0))

        ax[indx].set_ylim(bottom=0)
        firstHarmonicAx[indx].set_ylim(bottom=0)
        secondHarmonicAx[indx].set_ylim(bottom=0)

    # Make background transparent
    fig.patch.set_alpha(0)

    texplot.savefig(fig,'../Figures/Power-Spectrum-vs-Time-Jeffreys')

if(plot_double_power_spec_Jeffreys):
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    print("Computing the solution.")

    # Create KdV-Burgers or nonlocal KdV system
    posSystem = kdvSystem(P=P,H=H,diffeq='KdVB', eps=eps, mu=mu)
    negSystem = kdvSystem(P=-P,H=H,diffeq='KdVB', eps=eps, mu=mu)
    # Set spatial and temporal grid
    posSystem.set_spatial_grid(xLen='fit',xStep=xStep)
    negSystem.set_spatial_grid(xLen='fit',xStep=xStep)
    posSystem.set_temporal_grid(tLen=FFT_tLen,tNum='density')
    negSystem.set_temporal_grid(tLen=FFT_tLen,tNum='density')
    # Set initial conditions
    posSystem.set_initial_conditions(y0='cnoidal',redo_grids=True)
    negSystem.set_initial_conditions(y0='cnoidal',redo_grids=True)
    # Solve KdV-Burgers system
    posSystem.solve_system_rk3()
    negSystem.solve_system_rk3()

    # Boost to co-moving frame
    posSystem.boost_to_lab_frame(velocity='cnoidal')
    negSystem.boost_to_lab_frame(velocity='cnoidal')

    # Convert back to non-normalized variables
    # Convert from eta' = eta/a = eta/h/eps to eta'*eps = eta/a*eps = eta/h
    # (Primes denote the nondim variables used throughout this solver)
    posSystem.set_snapshot_ts(np.linspace(0,1,num=posSystem.tNum))
    negSystem.set_snapshot_ts(np.linspace(0,1,num=posSystem.tNum))
    posSnapshots = posSystem.get_snapshots()*eps
    negSnapshots = negSystem.get_snapshots()*eps

    # Resample (via interpolation) to get a higher FFT resolution
    repeat_times = 20
    posSnapshotsRepeated = np.tile(posSnapshots, (repeat_times,1))
    negSnapshotsRepeated = np.tile(negSnapshots, (repeat_times,1))

    # Generate spatial FFT conjugate coordinate
    kappa = np.fft.fftfreq(posSystem.xNum*repeat_times,
            posSystem.dx)*posSystem.WaveLength
    omega = np.fft.fftfreq(posSystem.tNum,
            posSystem.dt)*posSystem.WaveLength

    # Make a mesh of unshifted frequencies first for multiplying each
    # mode by exp(sqrt(1+P)*k*t)
    omega_mesh,kappa_mesh = np.meshgrid(omega,kappa)

    # Take spatial FFT (scale FFT by 1/N and multiply by 2 since we
    # ignore the negative frequencies)
    posSnapshotsFFT = 2*np.fft.fft(posSnapshotsRepeated,
            axis=0)/(posSystem.xNum*repeat_times)
    negSnapshotsFFT = 2*np.fft.fft(negSnapshotsRepeated,
            axis=0)/(negSystem.xNum*repeat_times)

    # Set kappa-columns with small values equal to zero so we don't blow them up
    posTol = np.amax([posSnapshotsFFT.real,posSnapshotsFFT.imag])*1e-3
    negTol = np.amax([negSnapshotsFFT.real,negSnapshotsFFT.imag])*1e-3

    posLessThanTol = np.logical_and(np.abs(posSnapshotsFFT.real)<posTol,
            np.abs(posSnapshotsFFT.imag)<posTol)
    negLessThanTol = np.logical_and(np.abs(negSnapshotsFFT.real)<negTol,
            np.abs(negSnapshotsFFT.imag)<negTol)

    posColumnWiseTol = np.tile(np.all(posLessThanTol, axis=1),
            (posSnapshotsFFT.shape[1],1)).transpose()
    negColumnWiseTol = np.tile(np.all(negLessThanTol, axis=1),
            (negSnapshotsFFT.shape[1],1)).transpose()

    posSnapshotsFFT[posColumnWiseTol] = 0
    negSnapshotsFFT[negColumnWiseTol] = 0

    # Multiply each mode by exp(sqrt(1+P)*k*t)
    posSnapshotsModifiedFFT = np.exp(-np.imag(1/2*eps*1j*P)
            *2*np.pi/posSystem.WaveLength
            *np.abs(kappa_mesh)
            *posSystem.t
            /4.14027 # fudge factor
            )*posSnapshotsFFT
    negSnapshotsModifiedFFT = np.exp(-np.imag(1/2*eps*(-1j*P))
            *2*np.pi/posSystem.WaveLength
            *np.abs(kappa_mesh)
            *negSystem.t
            /4.14027 # fudge factor
            )*negSnapshotsFFT

    # Take temporal FFT (scale FFT by 1/N and multiply by 2 since we
    # ignore the negative frequencies)
    posSnapshotsDoubleFFT = 2*np.fft.fft(posSnapshotsModifiedFFT,
            axis=1)/(posSystem.tNum)
    negSnapshotsDoubleFFT = 2*np.fft.fft(negSnapshotsModifiedFFT,
            axis=1)/(negSystem.tNum)

    # Convert to power spectrum (ie abs squared)
    posSnapshotsPower = np.absolute(posSnapshotsDoubleFFT)**2
    negSnapshotsPower = np.absolute(negSnapshotsDoubleFFT)**2

    # Shift frequencies
    kappa = np.fft.fftshift(kappa)
    omega = np.fft.fftshift(omega)

    omega_mesh,kappa_mesh = np.meshgrid(omega,kappa)

    posSnapshotsPower = np.fft.fftshift(posSnapshotsPower)
    negSnapshotsPower = np.fft.fftshift(negSnapshotsPower)

    print("Plotting.")
    ## Color cycle
    num_lines = posSnapshots[1,:].size # Number of lines
    new_colors = [plt.get_cmap('viridis')(1. * (i)/(num_lines)) for i in
            range(num_lines)]
    linestyles = [*((0,(3+i,i)) for i in range(num_lines))]
    plt.rc('axes', prop_cycle=(cycler('color', new_colors) +
                           cycler('linestyle', linestyles)))

    # Initialize figure
    fig, ax = texplot.newfig(0.9,nrows=2,sharex=True,sharey=False,golden=True)
    fig.set_tight_layout(False)

    # Adjust figure height
    figsize = fig.get_size_inches()
    fig.set_size_inches([figsize[0],figsize[1]*1.3])

    fig.subplots_adjust(left=0.175,right=0.825,top=0.875,bottom=0.125,hspace=0.3)

    ax[1].set_xlabel(r'Harmonic $\kappa/k$')
    ax[0].set_ylabel(r'Frequency $\omega/\sqrt{g/h}$')
    ax[1].set_ylabel(r'Frequency $\omega/\sqrt{g/h}$')
    # Convert P' = P*k/(rho_w*g)/eps to P'*eps = P*k/(rho_w*g)
    # (Primes denote the nondim variables used throughout this solver)
    fig.suptitle(r'Wavenumber Frequency Plot of $\abs{{\hat{{\eta}}}}^2$: $a/h={eps}$, $kh = {kh}$'.format(
        eps=eps,kh=round(np.sqrt(mu),1)))
    ax[0].set_title(r'Co-Wind: $P_J k/(\rho_w g) = {P}$'.format(
        P=round(eps*(P),3)))
    ax[1].set_title(r'Counter-Wind: $P_J k/(\rho_w g) = {P}$'.format(
        P=round(eps*(-P),3)))

    cs = [None,None]

    cs[0] = ax[0].contourf(kappa_mesh,omega_mesh,posSnapshotsPower,
            locator=mpl.ticker.SymmetricalLogLocator(linthresh=np.amax(posSnapshotsPower)/1e6,base=10),
            norm=mpl.colors.SymLogNorm(linthresh=np.amax(posSnapshotsPower)/1e6,base=10))
    cs[1] = ax[1].contourf(kappa_mesh,omega_mesh,negSnapshotsPower,
            locator=mpl.ticker.SymmetricalLogLocator(linthresh=np.amax(negSnapshotsPower)/1e6,base=10),
            norm=mpl.colors.SymLogNorm(linthresh=np.amax(negSnapshotsPower)/1e6,base=10))

    # Add colorbars
    fig.colorbar(cs[0],ax=ax[0],format=mpl.ticker.LogFormatterMathtext())
    fig.colorbar(cs[1],ax=ax[1],format=mpl.ticker.LogFormatterMathtext())

    # Adjust limits
    ax[0].set_xlim(-3.5,3.5)
    ax[0].set_ylim(-100,100)
    ax[1].set_xlim(-3.5,3.5)
    ax[1].set_ylim(-100,100)

    # Make background transparent
    fig.patch.set_alpha(0)

    texplot.savefig(fig,'../Figures/Double-Power-Spectrum-Jeffreys')

if(plot_power_spec_GM):
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    print("Computing the solution.")

    # Create KdV-Burgers or nonlocal KdV system
    posSystem = kdvSystem(P=P,H=H,psiP=psiP,diffeq='KdVNL', eps=eps, mu=mu)
    negSystem = kdvSystem(P=-P,H=H,psiP=psiP,diffeq='KdVNL', eps=eps, mu=mu)
    # Set spatial and temporal grid
    posSystem.set_spatial_grid(xLen='fit',xStep=xStep)
    negSystem.set_spatial_grid(xLen='fit',xStep=xStep)
    posSystem.set_temporal_grid(tLen=FFT_tLen,tNum='density')
    negSystem.set_temporal_grid(tLen=FFT_tLen,tNum='density')
    # Set initial conditions
    posSystem.set_initial_conditions(y0='cnoidal',redo_grids=True)
    negSystem.set_initial_conditions(y0='cnoidal',redo_grids=True)
    # Solve KdV-Burgers system
    posSystem.solve_system_rk3()
    negSystem.solve_system_rk3()

    # Boost to co-moving frame
    posSystem.boost_to_lab_frame(velocity='cnoidal')
    negSystem.boost_to_lab_frame(velocity='cnoidal')

    # Convert back to non-normalized variables
    # Convert from eta' = eta/a = eta/h/eps to eta'*eps = eta/a*eps = eta/h
    # (Primes denote the nondim variables used throughout this solver)
    posSystem.set_snapshot_ts(np.linspace(0,1,num=6))
    negSystem.set_snapshot_ts(np.linspace(0,1,num=6))
    posSnapshots = posSystem.get_snapshots()*eps
    negSnapshots = negSystem.get_snapshots()*eps

    # Resample (via interpolation) to get a higher FFT resolution
    repeat_times = 5
    posSnapshotsRepeated = np.tile(posSnapshots, (repeat_times,1))
    negSnapshotsRepeated = np.tile(negSnapshots, (repeat_times,1))

    # Take spatial FFT (scale FFT by 1/N and multiply by 2 since we
    # ignore the negative frequencies)
    posSnapshotsFFT = 2*np.fft.fft(posSnapshotsRepeated,
            axis=0)/posSystem.xNum/repeat_times
    negSnapshotsFFT = 2*np.fft.fft(negSnapshotsRepeated,
            axis=0)/negSystem.xNum/repeat_times

    # Convert to power spectrum (ie abs squared)
    posSnapshotsPower = np.absolute(posSnapshotsFFT)**2
    negSnapshotsPower = np.absolute(negSnapshotsFFT)**2

    # Generate spatial FFT conjugate coordinate
    kappa = np.fft.fftfreq(posSystem.xNum*repeat_times,
            posSystem.dx)*posSystem.WaveLength

    # Find peaks in initial data (peaks shouldn't move over time either)
    posSnapshotsPowerPeaks = sp.signal.find_peaks(posSnapshotsPower[:,0])[0]
    negSnapshotsPowerPeaks = sp.signal.find_peaks(negSnapshotsPower[:,0])[0]

    # Only keep non-negative kappa peaks
    posSnapshotsPowerPeaks = posSnapshotsPowerPeaks[kappa[posSnapshotsPowerPeaks] >= 0]
    negSnapshotsPowerPeaks = negSnapshotsPowerPeaks[kappa[negSnapshotsPowerPeaks] >= 0]

    # Sort the peaks
    posSnapshotsPowerPeaks = posSnapshotsPowerPeaks[np.argsort(
        posSnapshotsPower[posSnapshotsPowerPeaks,0])[::-1]]
    negSnapshotsPowerPeaks = negSnapshotsPowerPeaks[np.argsort(
        negSnapshotsPower[negSnapshotsPowerPeaks,0])[::-1]]

    # Only include first Power_num_peaks+1 (we need one extra since the
    # right-sided base always gives the right window limit, so we'll use
    # the n+1th left base limit to define the nth right base limit)
    posSnapshotsPowerPeaks = posSnapshotsPowerPeaks[0:FFT_num_peaks+1]
    negSnapshotsPowerPeaks = negSnapshotsPowerPeaks[0:FFT_num_peaks+1]

    # Find peak bases in initial data (peaks shouldn't move over time either)
    posSnapshotsPowerLeftBaseIndices = sp.signal.peak_prominences(
            posSnapshotsPower[:,0], posSnapshotsPowerPeaks)[1]
    negSnapshotsPowerLeftBaseIndices = sp.signal.peak_prominences(
            negSnapshotsPower[:,0], negSnapshotsPowerPeaks)[1]

    # Use the n+1th left base limit to define the nth right base limit
    posSnapshotsPowerRightBaseIndices = posSnapshotsPowerLeftBaseIndices[1:]
    posSnapshotsPowerLeftBaseIndices= posSnapshotsPowerLeftBaseIndices[:-1]
    negSnapshotsPowerRightBaseIndices = negSnapshotsPowerLeftBaseIndices[1:]
    negSnapshotsPowerLeftBaseIndices= negSnapshotsPowerLeftBaseIndices[:-1]

    # Take the largest of the two last base indices
    lastBaseIndex = np.amax([posSnapshotsPowerRightBaseIndices[-1],
        negSnapshotsPowerRightBaseIndices[-1]])

    # Store height (specifically, the largest height as a function of
    # time) and bases of second peak for inset image
    posSnapshotsPowerSecondPeakHeight = np.amax(posSnapshotsPower[
            posSnapshotsPowerPeaks[1]])
    negSnapshotsPowerSecondPeakHeight = np.amax(negSnapshotsPower[
            negSnapshotsPowerPeaks[1]])

    # Find 2nd peak bases in initial data (peaks shouldn't move over time either)
    # Re-calculate using 0.9999 prominence since we want it cropped
    # closely to the peak; the above base finding prescription would go
    # from the left side of peak 2 to the left side of peak 3, including
    # all the intervening flat space.
    posSnapshotsPowerSecondBaseIndices = sp.signal.peak_widths(
            posSnapshotsPower[:,0], [posSnapshotsPowerPeaks[1]],
            rel_height=0.9999)[2:]
    negSnapshotsPowerSecondBaseIndices = sp.signal.peak_widths(
            negSnapshotsPower[:,0], [negSnapshotsPowerPeaks[1]],
            rel_height=0.9999)[2:]

    # Convert from indices to kappa values by multiplying by the
    # kappa[1] (since all steps are evenly spaced)
    posSnapshotsPowerSecondBase = np.array(
            posSnapshotsPowerSecondBaseIndices)*kappa[1]
    negSnapshotsPowerSecondBase = np.array(
            negSnapshotsPowerSecondBaseIndices)*kappa[1]

    # Cut off after the last base (add 1 since we want the last base
    # *inclusive)
    kappa = kappa[0:lastBaseIndex+1]
    posSnapshotsPower = posSnapshotsPower[0:lastBaseIndex+1,:]
    negSnapshotsPower = negSnapshotsPower[0:lastBaseIndex+1,:]

    print("Plotting.")
    ## Color cycle
    num_lines = posSnapshots[1,:].size # Number of lines
    new_colors = [plt.get_cmap('viridis')(1. * (i)/(num_lines)) for i in
            range(num_lines)]
    linestyles = [*((0,(3+i,i)) for i in range(num_lines))]
    plt.rc('axes', prop_cycle=(cycler('color', new_colors) +
                           cycler('linestyle', linestyles)))

    # Initialize figure
    fig, ax = texplot.newfig(0.9,nrows=2,sharex=True,sharey=False,golden=True)
    fig.set_tight_layout(False)

    # Adjust figure height
    figsize = fig.get_size_inches()
    fig.set_size_inches([figsize[0],figsize[1]*1.3])

    fig.subplots_adjust(left=0.175,right=0.825,top=0.875,bottom=0.125,hspace=0.3)

    ax[1].set_xlabel(r'Harmonic $\kappa/k$')
    ax[0].set_ylabel(r'$\abs{\hat{\eta}}^2 k^2/h^2$')
    ax[1].set_ylabel(r'$\abs{\hat{\eta}}^2 k^2/h^2$')
    # Convert P' = P*k/(rho_w*g)/eps to P'*eps = P*k/(rho_w*g)
    # (Primes denote the nondim variables used throughout this solver)
    fig.suptitle(r'Power Spectrum vs Time: $a/h={eps}$, $kh = {kh}$'.format(
        eps=eps,kh=round(np.sqrt(mu),1)))
    ax[0].set_title(r'Co-Wind: $P_G k/(\rho_w g) = {P}$'.format(
        P=round(eps*(P),3)))
    ax[1].set_title(r'Counter-Wind: $P_G k/(\rho_w g) = {P}$'.format(
        P=round(eps*(-P),3)))

    ax[0].plot(kappa,posSnapshotsPower)
    ax[1].plot(kappa,negSnapshotsPower)

    # Put insets to zoom-in around second harmonic
    axins = [zoomed_inset_axes(ax[0], zoom=10, loc=1),
             zoomed_inset_axes(ax[1], zoom=10, loc=1)]

    axins[0].plot(kappa,posSnapshotsPower)
    axins[1].plot(kappa,negSnapshotsPower)

    axins[0].set_xlim(*posSnapshotsPowerSecondBase)
    axins[1].set_xlim(*negSnapshotsPowerSecondBase)
    axins[0].set_ylim(0,posSnapshotsPowerSecondPeakHeight*1.05)
    axins[1].set_ylim(0,negSnapshotsPowerSecondPeakHeight*1.05)

    mark_inset(ax[0], axins[0], loc1=4, loc2=2, fc="none", ec="0.5")
    mark_inset(ax[1], axins[1], loc1=4, loc2=2, fc="none", ec="0.5")

    # Turn off tick labels
    axins[0].set_yticklabels([])
    axins[0].set_xticklabels([])
    axins[1].set_yticklabels([])
    axins[1].set_xticklabels([])

    # Note: divide by epsilon to convert from t_1 to the full time t
    fig.legend(np.around(posSystem.snapshot_ts/eps,1),
            title=r'Time'+'\n'+r'$t \sqrt{g/h}$',loc='right')

    # Make background transparent
    fig.patch.set_alpha(0)

    texplot.savefig(fig,'../Figures/Power-Spectrum-GM')

if(plot_power_spec_vs_time_GM):
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    print("Computing the solution.")

    # Create KdV-Burgers or nonlocal KdV system
    posSystem = kdvSystem(P=P,H=H,psiP=psiP,diffeq='KdVNL', eps=eps, mu=mu)
    negSystem = kdvSystem(P=-P,H=H,psiP=psiP,diffeq='KdVNL', eps=eps, mu=mu)
    # Set spatial and temporal grid
    posSystem.set_spatial_grid(xLen='fit',xStep=xStep)
    negSystem.set_spatial_grid(xLen='fit',xStep=xStep)
    posSystem.set_temporal_grid(tLen=FFT_tLen,tNum='density')
    negSystem.set_temporal_grid(tLen=FFT_tLen,tNum='density')
    # Set initial conditions
    posSystem.set_initial_conditions(y0='cnoidal',redo_grids=True)
    negSystem.set_initial_conditions(y0='cnoidal',redo_grids=True)
    # Solve KdV-Burgers system
    posSystem.solve_system_rk3()
    negSystem.solve_system_rk3()

    # Boost to co-moving frame
    posSystem.boost_to_lab_frame(velocity='cnoidal')
    negSystem.boost_to_lab_frame(velocity='cnoidal')

    # Convert back to non-normalized variables
    # Convert from eta' = eta/a = eta/h/eps to eta'*eps = eta/a*eps = eta/h
    # (Primes denote the nondim variables used throughout this solver)
    posSystem.set_snapshot_ts(np.linspace(0,1,num=posSystem.tNum))
    negSystem.set_snapshot_ts(np.linspace(0,1,num=posSystem.tNum))
    posSnapshots = posSystem.get_snapshots()*eps
    negSnapshots = negSystem.get_snapshots()*eps

    # Save timesteps
    # Note: divide by epsilon to convert from t_1 to the full time t
    t = posSystem.t/eps

    # Resample (via interpolation) to get a higher FFT resolution
    repeat_times = 5
    posSnapshotsRepeated = np.tile(posSnapshots, (repeat_times,1))
    negSnapshotsRepeated = np.tile(negSnapshots, (repeat_times,1))

    # Take spatial FFT (scale FFT by 1/N and multiply by 2 since we
    # ignore the negative frequencies)
    posSnapshotsFFT = 2*np.fft.fft(posSnapshotsRepeated,
            axis=0)/posSystem.xNum/repeat_times
    negSnapshotsFFT = 2*np.fft.fft(negSnapshotsRepeated,
            axis=0)/negSystem.xNum/repeat_times

    # Convert to power spectrum (ie abs squared)
    posSnapshotsPower = np.absolute(posSnapshotsFFT)**2
    negSnapshotsPower = np.absolute(negSnapshotsFFT)**2

    # Generate spatial FFT conjugate coordinate
    kappa = np.fft.fftfreq(posSystem.xNum*repeat_times,
            posSystem.dx)*posSystem.WaveLength

    # Find peaks in initial data (peaks shouldn't move over time either)
    posSnapshotsPowerPeaks = sp.signal.find_peaks(posSnapshotsPower[:,0])[0]
    negSnapshotsPowerPeaks = sp.signal.find_peaks(negSnapshotsPower[:,0])[0]

    # Only keep non-negative kappa peaks
    posSnapshotsPowerPeaks = posSnapshotsPowerPeaks[kappa[posSnapshotsPowerPeaks] >= 0]
    negSnapshotsPowerPeaks = negSnapshotsPowerPeaks[kappa[negSnapshotsPowerPeaks] >= 0]

    # Sort the peaks
    posSnapshotsPowerPeaks = posSnapshotsPowerPeaks[np.argsort(
        posSnapshotsPower[posSnapshotsPowerPeaks,0])[::-1]]
    negSnapshotsPowerPeaks = negSnapshotsPowerPeaks[np.argsort(
        negSnapshotsPower[negSnapshotsPowerPeaks,0])[::-1]]

    # Only include first 3 peaks
    posSnapshotsPowerPeaks = posSnapshotsPowerPeaks[0:3]
    negSnapshotsPowerPeaks = negSnapshotsPowerPeaks[0:3]

    # Get indices for primary (m=1), first harmonic (m=2), and second
    # harmonic (m=3)
    posPrimaryIndex = posSnapshotsPowerPeaks[0]
    posFirstHarmonicIndex = posSnapshotsPowerPeaks[1]
    posSecondHarmonicIndex = posSnapshotsPowerPeaks[2]
    negPrimaryIndex = negSnapshotsPowerPeaks[0]
    negFirstHarmonicIndex = negSnapshotsPowerPeaks[1]
    negSecondHarmonicIndex = negSnapshotsPowerPeaks[2]

    print("Plotting.")
    ## Color cycle
    num_lines = posSnapshots[1,:].size # Number of lines
    new_colors = [plt.get_cmap('viridis')(1. * (i)/(num_lines)) for i in
            range(num_lines)]
    linestyles = [*((0,(3+i,i)) for i in range(num_lines))]
    plt.rc('axes', prop_cycle=(cycler('color', new_colors) +
                           cycler('linestyle', linestyles)))

    # Initialize figure
    fig, ax = texplot.newfig(0.9,nrows=2,sharex=True,sharey=False,golden=True)
    fig.set_tight_layout(False)

    # Adjust figure height
    figsize = fig.get_size_inches()
    fig.set_size_inches([figsize[0],figsize[1]*1.3])

    fig.subplots_adjust(left=0.125,right=0.775,top=0.875,bottom=0.125,hspace=0.3)

    ax[1].set_xlabel(r'Time $t\sqrt{g/h}$')

    # Convert P' = P*k/(rho_w*g)/eps to P'*eps = P*k/(rho_w*g)
    # (Primes denote the nondim variables used throughout this solver)
    fig.suptitle(r'Power Spectrum vs Time: $a/h={eps}$, $kh = {kh}$'.format(
        eps=eps,kh=round(np.sqrt(mu),1)))
    ax[0].set_title(r'Co-Wind: $P_G k/(\rho_w g) = {P}$'.format(
        P=round(eps*(P),3)))
    ax[1].set_title(r'Counter-Wind: $P_G k/(\rho_w g) = {P}$'.format(
        P=round(eps*(-P),3)))

    # Source: https://matplotlib.org/3.1.1/gallery/ticks_and_spines/multiple_yaxis_with_spines.html
    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    firstHarmonicAx = [None,None]
    secondHarmonicAx = [None,None]
    for indx in [0,1]:
        firstHarmonicAx[indx] = ax[indx].twinx()
        secondHarmonicAx[indx] = ax[indx].twinx()

        # Offset the right spine of secondHarmoincAx.
        # The ticks and label have already been placed on the right by twinx above.
        secondHarmonicAx[indx].spines["right"].set_position(("axes", 1.2))

        # Having been created by twinx, secondHarmonicAx has its frame
        # off, so the line of its detached spine is invisible.
        # First, activate the frame but make the patch and spines invisible.
        make_patch_spines_invisible(secondHarmonicAx[indx])

        # Second, show the right spine.
        secondHarmonicAx[indx].spines["right"].set_visible(True)

    line1 = ax[0].plot(t,posSnapshotsPower[posPrimaryIndex,:],'r')[0]
    line2 = firstHarmonicAx[0].plot(t,posSnapshotsPower[posFirstHarmonicIndex,:],'g')[0]
    line3 = secondHarmonicAx[0].plot(t,posSnapshotsPower[posSecondHarmonicIndex,:],'b')[0]
    ax[1].plot(t,negSnapshotsPower[negPrimaryIndex,:],'r')
    firstHarmonicAx[1].plot(t,negSnapshotsPower[negFirstHarmonicIndex,:],'g')
    secondHarmonicAx[1].plot(t,negSnapshotsPower[negSecondHarmonicIndex,:],'b')

    for indx in [0,1]:
        ax[indx].set_ylabel(r'Primary $\abs{\hat{\eta}}^2 k^2/h^2$')
        firstHarmonicAx[indx].set_ylabel(r'First Harmonic $\abs{\hat{\eta}}^2 k^2/h^2$')
        secondHarmonicAx[indx].set_ylabel(r'Second Harmonic $\abs{\hat{\eta}}^2 k^2/h^2$')

        ax[indx].yaxis.label.set_color(line1.get_color())
        firstHarmonicAx[indx].yaxis.label.set_color(line2.get_color())
        secondHarmonicAx[indx].yaxis.label.set_color(line3.get_color())

        ax[indx].tick_params(axis='y', colors=line1.get_color())
        firstHarmonicAx[indx].tick_params(axis='y', colors=line2.get_color())
        secondHarmonicAx[indx].tick_params(axis='y', colors=line3.get_color())
        ax[indx].tick_params(axis='x')

        ax[indx].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
        firstHarmonicAx[indx].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
        secondHarmonicAx[indx].ticklabel_format(style='sci',axis='y',scilimits=(0,0))

        ax[indx].set_ylim(bottom=0)
        firstHarmonicAx[indx].set_ylim(bottom=0)
        secondHarmonicAx[indx].set_ylim(bottom=0)

    # Make background transparent
    fig.patch.set_alpha(0)

    texplot.savefig(fig,'../Figures/Power-Spectrum-vs-Time-GM')

if(plot_pos_neg_snapshots_cnoidal_GM):
    print("Computing the solution.")

    # Create KdV-Burgers or nonlocal KdV system
    posSystem = kdvSystem(P=P_GM,H=H_GM,psiP=psiP,diffeq='KdVNL', eps=eps, mu=mu)
    negSystem = kdvSystem(P=-P_GM,H=H_GM,psiP=psiP,diffeq='KdVNL', eps=eps, mu=mu)
    # Set spatial and temporal grid
    posSystem.set_spatial_grid(xLen='fit',xStep=xStep)
    negSystem.set_spatial_grid(xLen='fit',xStep=xStep)
    posSystem.set_temporal_grid(tLen=tLen,tNum='density')
    negSystem.set_temporal_grid(tLen=tLen,tNum='density')
    # Set initial conditions
    posSystem.set_initial_conditions(y0='cnoidal',redo_grids=True)
    negSystem.set_initial_conditions(y0='cnoidal',redo_grids=True)
    # Solve KdV-Burgers system
    posSystem.solve_system_rk3()
    negSystem.solve_system_rk3()

    # Boost to co-moving frame
    posSystem.boost_to_lab_frame(velocity='cnoidal')
    negSystem.boost_to_lab_frame(velocity='cnoidal')

    # Convert back to non-normalized variables
    # Convert from eta' = eta/a = eta/h/eps to eta'*eps = eta/a*eps = eta/h
    # (Primes denote the nondim variables used throughout this solver)
    posSystem.set_snapshot_ts([0,1/3,2/3,1])
    negSystem.set_snapshot_ts([0,1/3,2/3,1])
    posSnapshots = posSystem.get_snapshots()*eps
    negSnapshots = negSystem.get_snapshots()*eps

    # Hide solution outside of window
    posSystem.set_x_window()
    xMasked = posSystem.get_masked_x()

    # Normalize x by wavelength
    # Convert from x' = x*k_E to x'/lambda' = x*k_E/lambda/k_E = x/lambda
    # (Primes denote the nondim variables used throughout this solver)
    xMasked = xMasked/posSystem.WaveLength

    print("Plotting.")
    ## Color cycle
    num_lines = posSnapshots[1,:].size # Number of lines
    new_colors = [plt.get_cmap('viridis')(1. * (i)/(num_lines)) for i in
            range(num_lines)]
    linestyles = [*((0,(3+i,i)) for i in range(num_lines))]
    plt.rc('axes', prop_cycle=(cycler('color', new_colors) +
                           cycler('linestyle', linestyles)))

    # Initialize figure
    fig, ax = texplot.newfig(0.9,nrows=2,sharex=True,sharey=True,golden=True)
    fig.set_tight_layout(False)

    # Adjust figure height
    figsize = fig.get_size_inches()
    fig.set_size_inches([figsize[0],figsize[1]*1.3])

    fig.subplots_adjust(left=0.175,right=0.9,top=0.875,bottom=0.125,hspace=0.3)

    # Plot x'*lambda' = x*k_E/lambda/k_E = x/lambda
    # (Primes denote the nondim variables used throughout this solver)
    ax[1].set_xlabel(r'Distance $x/\lambda$')
    # Plot eta'*eps = eta/a*eps = eta/h
    # (Primes denote the nondim variables used throughout this solver)
    ax[0].set_ylabel(r'$\eta / h$')
    ax[1].set_ylabel(r'$\eta / h$')
    # Convert P' = P*k/(rho_w*g)/eps to P'*eps = P*k/(rho_w*g)
    # (Primes denote the nondim variables used throughout this solver)
    fig.suptitle(r'Surface Height vs Time: $a/h={eps}$, $kh = {kh}$'.format(
        eps=eps,kh=round(np.sqrt(mu),1)))
    ax[0].set_title(r'$P_G k/(\rho_w g) = {P}$'.format(
        P=round(eps*(P_GM),3)))
    ax[1].set_title(r'$P_G k/(\rho_w g) = {P}$'.format(
        P=round(eps*(-P_GM),3)))

    ax[0].plot(xMasked,posSnapshots)
    ax[1].plot(xMasked,negSnapshots)

    # Add arrow depicting wind direction
    arrowLeft = np.array([0.175,0.4])
    arrowRight = np.array([0.375,0.4])
    textBottom = (arrowLeft+arrowRight)/2 + np.array([0,0.05])
    spacing = np.array([0.45,0])
    ax[0].annotate(r'Phase'+'\n'+r'Speed', xy=textBottom, xycoords='axes fraction',
            ha='center',va='bottom',ma='center')
    ax[0].annotate('', xy=arrowLeft, xytext=arrowRight,
            xycoords="axes fraction", arrowprops={'arrowstyle': '<-',
                'shrinkA':1,'shrinkB':0})
    ax[0].annotate(r'Wind', xy=textBottom+spacing,
            xycoords='axes fraction', ha='center',va='bottom',
            ma='center')
    ax[0].annotate('', xy=arrowLeft+spacing, xytext=arrowRight+spacing,
            xycoords="axes fraction", arrowprops={'arrowstyle': '<-',
                'shrinkA':1,'shrinkB':0})
    ax[1].annotate(r'Phase'+'\n'+r'Speed', xy=textBottom, xycoords='axes fraction',
            ha='center',va='bottom',ma='center')
    ax[1].annotate('', xy=arrowLeft, xytext=arrowRight,
            xycoords="axes fraction", arrowprops={'arrowstyle': '<-',
                'shrinkA':1,'shrinkB':0})
    ax[1].annotate(r'Wind', xy=textBottom+spacing,
            xycoords='axes fraction', ha='center',va='bottom',
            ma='center')
    ax[1].annotate('', xy=arrowLeft+spacing, xytext=arrowRight+spacing,
            xycoords="axes fraction", arrowprops={'arrowstyle': '->',
                'shrinkA':1,'shrinkB':0})

    # Note: divide by epsilon to convert from t_1 to the full time t
    fig.legend(np.around(posSystem.snapshot_ts/eps,1),
            title=r'Time'+'\n'+r'$t \sqrt{g/h}$',loc='right')

    # Make background transparent
    fig.patch.set_alpha(0)

    texplot.savefig(fig,'../Figures/Snapshots-Positive-Negative-Cnoidal-GM')

# Plot different forcing types
if(plot_forcing_types):

    P_forcing = 1

    Height = 1

    WaveLength = 2*np.pi
    numWaves = 1
    xLen_forcing = WaveLength*numWaves
    dx_forcing = 0.01
    x = np.arange(0,xLen_forcing,dx_forcing)

    m = 0.1

    K = spec.ellipk(m)
    E = spec.ellipe(m)

    cn = spec.ellipj(x/WaveLength*2*K,m)[1]
    trough = Height/m*(1-m-E/K)
    y0 = trough + Height*cn**2

    eta = y0
    jeffreys = P_forcing*psdiff(eta, period=xLen_forcing)
    generalized = P_forcing*np.roll(eta, shift=int(round(-psiP*WaveLength/(2*np.pi)/dx_forcing)),
                    axis=0)

    # Initialize figure
    fig, ax = texplot.newfig(0.9,nrows=2,ncols=1,sharex=True,
            sharey=False)

    ax[0].set_ylabel(r'Elevation $\eta/H$')
    ax[1].set_ylabel(r'\begin{tabular}{c}Pressure $p$\end{tabular}')

    ax[1].set_xlabel(r'Distance $x/\lambda$')

    ax[0].annotate('a)', xy=(0.02,0.8), xycoords='axes fraction')
    ax[1].annotate('b)', xy=(0.02,0.8), xycoords='axes fraction')

    ax[0].plot(x,eta,color=SColor)

    ln1 = ax[1].plot(x,jeffreys,label=r'Jeffreys $p_J$',
            color=JColor,linestyle=JStyle)
    ln3 = ax[1].plot(x,generalized,label=r'Generalized $p_G$',
            color=GColor,linestyle=GStyle)

    lns = ln1+ln3
    lbls = [l.get_label() for l in lns]

    fig.legend(lns, lbls, loc='right', bbox_to_anchor=(0.5,0.5))

    # Add line depicting $\psi_P$
    ax[1].annotate('', xy=(WaveLength, 0), xytext=(WaveLength*(1-psiP/(2*np.pi)), 0),
            arrowprops=dict(arrowstyle='|-|, widthA=0.5, widthB=0.5'))
    ax[1].annotate(r'$\psi_P$', xy=(WaveLength*(1-psiP/(2*np.pi)/2.), 0), ha='center',
            va='bottom')

    texplot.savefig(fig,'../Figures/Forcing-Types')
