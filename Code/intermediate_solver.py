#!/usr/bin/env python3
# intermediate_solver.py
import sys
import scipy as sp
import numpy as np
import scipy.special as spec
from scipy.integrate import trapz, solve_ivp
import scipy.signal
from scipy.fftpack import diff as psdiff
from numpy import gradient
import xarray as xr
import data_csv

class kdvSystem():

    def __init__(self, A=1, B=3/2, C=None, F=None, G=None, P=None, H=0,
            psiP=0, diffeq='KdVB', eps=0.1, mu=0.6, *args, **kwargs):
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
            Sets the importance of the dispersivity. Default is
            1/6*mu/eps.
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

    def set_spatial_grid(self, xLen=20, xNum=None,
            xStep=None, xOffset=None,
            WaveLength=2*np.pi,NumWaves=2, *args, **kwargs):
        """Set the x coordinate grid.

        Parameters
        ----------
        xLen : float, 'fit', or None
            Length of x domain. If 'fit', choose xLen to fit NumWave
            wavelengths of length WaveLegngth. Default is 10.
        xNum : float or None
            Number of grid points in x domain. If None, xStep
            must be specified. Default is None.
        xStep : float or None
            Spacing between grid points in x domain. If
            None, xNum must be specified. Default is 0.1.
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
        elif xNum == None and xStep == None:
            # Use default value of xStep = 0.1
            xStep = 0.1
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

    def set_temporal_grid(self, tLen=3, tNum='density', *args, **kwargs):
        """Set the t-coordinate grid.

        Parameters
        ----------
        tLen : float or None
            Length of t domain. Default is 3.
        tNum : float, 'density', or None
            Number of grid points in t domain. 'density' chooses tNum so
            that the temporal density (tLen/tNum) is equal to the
            spatial density (xLen/xNum) to the 4th power. Default is
            'density'.

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
            redo_grids=True, *args, **kwargs):
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
                raise(ValueError("m = 2*B/3/C must be at least 0"+
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
            self.WaveLength = np.inf

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

    def set_x_window(self, xMin=None, xMax=None, xScale=1, *args, **kwargs):
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
        xScale : float
            Amount to scale xMin and xMax by. Default is 1.

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


        self.xWin = [self.xMin*xScale, self.xMax*xScale]

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

    def set_snapshot_ts(self, snapshot_fracs=None, *args, **kwargs):
        """Set the snapshot times.

        Set the snapshot times as a fraction of the total time domain
        length. Any snapshot times after the last time step are set to
        the last time step. Any snapshot times before the first time
        step are set to the first time step.

        Parameters
        ----------
        snapshot_fracs : array_like or None
            Times, as a fraction of the total time domain length, at
            which to display snapshots. If None, return every xNum
            points (so that the returned t-grid is the same size as the
            returned x-grid). Default is None.

        Returns
        -------
        None
        """

        if snapshot_fracs is None:
            # Choose indices so that we return xNum timesteps
            snapshot_indxs = np.arange(start=0, stop=self.tNum,
                    step=int(self.tNum/self.xNum))
            # Ensure we also capture the last timestep
            snapshot_indxs = np.append(snapshot_indxs, self.tNum-1)
            snapshot_ts = np.array(self.t)[snapshot_indxs]

        else:
            # Convert snapshot fractions to snapshot times
            snapshot_ts = np.array(snapshot_fracs)*self.tLen

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

    def _kdvb(self, t, u, periodic_deriv=False, *args, **kwargs):
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
            uxxxx = psdiff(u, period=self.xLen, order=4)

        # Compute du/dt
        dudt = -u*ux*self.B/self.A -uxxx*self.C/self.A \
                -ux*self.F/self.A + uxx*self.G/self.A \
                -uxxxx*self.H/self.A

        return dudt

    def _kdvnl(self, t, u, periodic_deriv=False, *args, **kwargs):
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

    def _JacSparcity(self, *args, **kwargs):
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

    def solve_system_builtin(self, periodic_deriv=True, *args, **kwargs):
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

    def boost_to_lab_frame(self,boostVelocity=-1/6, *args, **kwargs):
        """
        Parameters
        ----------
        boostVelocity : float, 'solitary', 'cnoidal', or None
            The velocity of the initial, wave frame relative to the lab
            frame in units of [x]/[t]. 'solitary' boosts into the
            co-moving frame of an unforced solitary wave. 'cnoidal'
            boosts into the co-moving frame of an unforced cnoidal wave.
            Default is -1/6.

        Returns
        -------
        None
        """

        if boostVelocity == 'solitary':
            boostVelocity = -(self.B*self.Height/3+self.F)/self.A
        elif boostVelocity == 'cnoidal':
            m = self.m
            Height = self.Height
            K = spec.ellipk(m)
            E = spec.ellipe(m)

            boostVelocity = -(self.F+2/3*self.B*Height/m*(1-1/2*m-3/2*E/K))/self.A

        # Convert the physical velocity to a coordinate velocity
        coord_vel = boostVelocity/self.dx*self.dt

        sol = self.sol

        for time in range(sol.shape[1]):
            sol[:,time] = np.roll(sol[:,time],shift=int(round(coord_vel*time)))

        self.sol = sol

def default_solver(y0_func=None, solver='RK3', *args, **kwargs):

    if 'boostVelocity' not in kwargs and (kwargs.get('wave_type') == 'cnoidal'
            or kwargs.get('wave_type') == 'solitary'):
        kwargs['boostVelocity'] = kwargs['wave_type']
    if 'y0' not in kwargs and (kwargs.get('wave_type') == 'cnoidal'
            or kwargs.get('wave_type') == 'solitary'):
        kwargs['y0'] = kwargs['wave_type']

    forcing_type_dict = {'Jeffreys' : 'KdVB', 'GM' : 'KdVNL'}
    if 'forcing_type' in kwargs:
        kwargs['diffeq'] = forcing_type_dict[kwargs['forcing_type']]

    # Create KdV-Burgers or nonlocal KdV system
    solverSystem = kdvSystem(**kwargs)
    # Set spatial grid
    solverSystem.set_spatial_grid(**kwargs)
    # Set temporal grid
    solverSystem.set_temporal_grid(**kwargs)
    # Set initial conditions
    if callable(y0_func):
        solverSystem.set_initial_conditions(y0=y0_func(solverSystem.x), **kwargs)
    else:
        solverSystem.set_initial_conditions(**kwargs)
    # Set snapshot times
    solverSystem.set_snapshot_ts(**kwargs)
    # Solve KdV-Burgers system
    if solver == 'Builtin':
        solverSystem.solve_system_builtin()
    elif solver == 'RK3':
        solverSystem.solve_system_rk3()
    elif solver == 'AB2':
        solverSystem.solve_system_ab2()
    else:
        raise(ValueError("'solver' must be {'Builtin','RK3','AB2'}, but "+
            solver+" was provided"))

    # Boost to co-moving frame
    solverSystem.boost_to_lab_frame(**kwargs)

    # Convert back to non-normalized variables
    # Convert from eta' = eta/a = eta/h/eps to eta'*eps = eta/a*eps = eta/h
    # (Primes denote the nondim variables used throughout this solver)
    snapshots = solverSystem.get_snapshots()*solverSystem.eps

    # Hide solution outside of window
    solverSystem.set_x_window()
    xMasked = solverSystem.get_masked_x()

    # Normalize x by wavelength
    # Convert from x' = x*k_E to x'/sqrt(mu) = x*k_E/k_E/h = x/h
    # (Primes denote the nondim variables used throughout this solver)
    xMasked = xMasked/np.sqrt(solverSystem.mu)

    # Extract snapshot times
    # t_1' = epsilon*t' = epsilon*t*sqrt(g*h)*k_E
    snapshot_ts = solverSystem.snapshot_ts

    # Package snapshots, xs, and ts together in a DataArray
    data = xr.DataArray(snapshots,
            dims=('x/h','t*eps*sqrt(g*h)*k_E'),
            coords=[xMasked, snapshot_ts])

    # Remove data for x with masked values
    data = data.where(~np.isnan(data['x/h']),drop=True)

    return data, solverSystem

def gen_trig_verf(save_prefix, H=1.25e-2):
    # Generate verification data by using sinusoidal initial conditions
    # with the Burgers equation
    NumWaves = 1
    xLen = 2*np.pi
    xNum = 64
    xs = np.linspace(0,xLen,num=xNum)
    wave_length = xLen/NumWaves
    WaveNum = 2*np.pi/wave_length
    y0_func = lambda x : np.sin(WaveNum*x)
    tNum_trig = {'Builtin': 501, 'RK3': 10**5}
    snapshot_fracs = [0,1/3,2/3,1]

    for H_val in [H, 0]:
        parameters = {
                'A' : 1,
                'B' : 0,
                'C' : 1/6, # C = 1/6*mu/eps, so this requires mu=eps
                'P' : 0,
                'H' : H_val,
                'eps' : 0.1,
                'mu' : 0.1,
                'wave_length' : wave_length,
                }

        for solver in ['Builtin','RK3']:
            data, _ = default_solver(**parameters,
                xLen=xLen, xNum=xNum, y0_func = y0_func,
                tLen=(xLen/NumWaves)**3/(2*np.pi)**2*parameters['A']\
                        /parameters['C'],
                tNum=tNum_trig[solver],
                snapshot_fracs = snapshot_fracs,
                velocity=0,
                solver=solver,
                )

            data_csv.save_data(data,
                    save_prefix+'TrigVerf-'+solver+'_H'+str(H_val),
                    **parameters, solver=solver, stack_coords = True)

def gen_long_verf(save_prefix, mu=0.8, H=1.25e-2):
    # Generate verification data by running a solitary wave profile
    # without forcing for a long time

    tLen = 30
    snapshot_fracs = [0,1/3,2/3,1]

    for H_val in [H, 0]:
        for wave_type in ['solitary', 'cnoidal']:
            parameters = {
                    'mu' : mu,
                    'P' : 0,
                    'H' : H_val,
                    'wave_type' : wave_type,
                    }

            # Use default mu for solitary waves
            if wave_type == 'solitary':
                parameters.pop('mu')

            data, dataClass = default_solver(**parameters,
                    tLen=tLen,
                    snapshot_fracs = snapshot_fracs,
                    )

            # Get default mu for solitary waves
            if wave_type == 'solitary':
                parameters['mu'] = dataClass.mu

            data_csv.save_data(data,
                    save_prefix+'LongVerf_H'+str(H_val),
                    stack_coords = True, eps=dataClass.eps,
                    **parameters)

def gen_snapshots(save_prefix, eps=0.1, mu=0.8, P=0.25, psiP=3/4*np.pi, H=1.25e-2):
    """ Generate snapshots for range of parameters. Save the results to
    the directory given by 'save_prefix'.

    Parameters
    ----------
    eps : float
        Nondimensional height a/h. Default is 0.1
    mu : float
        Nondimensional wavelength (k_E h)**2. Must be greater than or
        equal to eps*6. Default is 0.8.
    psiP : float
        Wind phase, in radians, for (GM) KdVNL equation. Default is
        3/4*pi.
    P : float
        Nondimensional pressure magnitude P*k_E/(rhoW*g). Default is
        0.25. Note: P should be O(1) since we already pulled out the
        epsilon-factor. However, the system is too numerically unstable
        to actually use P~1, so decrease it for stability
    H : float
        Biharmonic viscosity needed for numerical stability. Default is
        1.25e-2.
    """

    for forcing_type in ['Jeffreys', 'GM']:
        for wave_type in ['solitary','cnoidal']:
            if forcing_type == 'GM' and wave_type == 'solitary':
                # We don't need GM applied to solitary waves, as it
                # doesn't make as much sense
                continue
            for P_val in P*np.array([-2,-1,0,1,2]):

                for mu_val in mu*np.array([1,2]):

                    parameters = {
                            'eps' : eps,
                            'mu' : mu_val,
                            'wave_type' : wave_type,
                            'forcing_type' : forcing_type,
                            'P' : P_val,
                            'psiP' : psiP,
                            'H' : H,
                            }

                    # Jeffreys does not just psiP
                    if forcing_type == 'Jeffreys':
                        parameters.pop('psiP')

                    # Use default mu for solitary waves
                    if wave_type == 'solitary':
                        parameters.pop('mu')

                    # Run model
                    data, dataClass = default_solver(**parameters)

                    # Get default mu for solitary waves
                    if wave_type == 'solitary':
                        parameters['mu'] = dataClass.mu

                    # Save data
                    data_csv.save_data(data, save_prefix+'Snapshots',
                            wave_length=dataClass.WaveLength,
                            **parameters, stack_coords=True)

                    if wave_type == 'solitary':
                        # We don't use mu for solitary waves, so break
                        # after a single run (since changing mu won't
                        # affect the output)
                        break

def gen_depth_varying(save_prefix, eps=0.1, mu=0.6, P=0.25, psiP=3/4*np.pi,
        H=1.25e-2, forcing_type='Jeffreys'):

    # Linearly space khs
    kh_vals = np.sqrt(mu)*np.linspace(1,np.sqrt(2),num=30)
    # Don't include kh = eps*0.6 (solitary wave) since cnoidal waves
    # have num_waves=2 but cnoidal wave has num_waves=1
    kh_vals = kh_vals[1:]
    mu_vals = kh_vals**2

    for mu_val in mu_vals:
        parameters = {
                'eps' : eps,
                'mu' : mu_val,
                'wave_type' : 'cnoidal',
                'forcing_type' : forcing_type,
                'P' : P,
                'psiP' : psiP,
                'H' : H,
                }

        # Jeffreys does not just psiP
        if forcing_type == 'Jeffreys':
            parameters.pop('psiP')

        # Run model
        data, dataClass = default_solver(**parameters, tLen=3)

        # Save data
        data_csv.save_data(data, save_prefix+'DepthVarying',
                wave_length=dataClass.WaveLength,
                **parameters, stack_coords=True)

def main():
    save_prefix = '../Data/Raw/'

    callable_functions = {
            'trig_verf' : gen_trig_verf,
            'long_verf' : gen_long_verf,
            'snapshots' : gen_snapshots,
            'depth_varying' : gen_depth_varying,
            }

    if len(sys.argv) == 1:
        # No option provided; run all gens
        for function in callable_functions.values():
            function(save_prefix)
    else:
        if not sys.argv[1] in callable_functions:
            raise ValueError('Command line option must be the name of '+
                'a callable function') from None
        callable_functions[sys.argv[1]](save_prefix, *sys.argv[2:])

if __name__ == '__main__':
    main()
