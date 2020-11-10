#!/usr/bin/env python3
# intermediate_solver.py
import sys
import warnings
import scipy as sp
import numpy as np
import scipy.special as spec
from scipy.integrate import trapz, solve_ivp
import scipy.signal
from scipy.fftpack import diff as psdiff
from numpy import gradient
import xarray as xr
import data_csv
from useful_functions import round_sig_figs, derivative

class kdvSystem():

    def __init__(self, A=1, B=3/2, C=None, F=None, G=None, P=None,
            H=None, nu_bi=None, psiP=0, diffeq='KdVB', eps=0.1, mu=0.6,
            Height=None, *args, **kwargs):
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
        F : float or None
            Sets the background current strength. Default is 0.
        G : float or None
            Sets the strength of the damping. If None, determine G from
            G=-1/2*P, where P is the pressure forcing strength. Default
            is 0.
        H : float or None
            Sets the strength of the higher-order damping. Positive H is
            needed for stability when solving the KdV-Burgers equation
            with negative G. If None, determine H from H = nu_bi.
            Default is 0.
        P : float or None
            Sets the strength of the pressure forcing. If None,
            determine P from P=-2*G, where G is the strength of the
            damping. Default is 0.
        nu_bi : float or None
            Sets the strength of the biviscosity. If None, determine
            nu_bi from nu_bi=H, where H is the higher-order damping.
            Default is 0.
        psiP : float or None
            The wind phase, or the shift of the pressure relative to the
            surface height p(x,t) = eta(x+psiP,t). This is used for
            Generalized Miles-type forcings. Default is 0.
        Height : float or None
            Height of initial condition. If None, then H is chosen to be
            2*sign(self.B*self.C). Default is None.
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

        self.eps = eps
        self.mu = mu

        self.psiP = psiP
        self.diffeq = diffeq

        if C is not None:
            self.C = C
        else:
            self.C = 1/6*mu/eps

        if F is not None:
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

        if H is not None and nu_bi is not None:
            raise ValueError('Cannot provide both H and nu_bi to kdvSystem constructor.')
        elif nu_bi is not None:
            self.nu_bi = nu_bi
            self.H = nu_bi
        elif H is not None:
            self.H = H
            self.nu_bi = H
        else:
            self.H = 0
            self.nu_bi = 0

        if self.diffeq == 'KdVNL' and P<0:
            # For nonlocal KdV (Generalized Miles), we always use P>0.
            # Since we previously used P<0 to denote wind blowing in the
            # opposite direction, simply change the sign of P (and G),
            # while flipping the sign of psiP (which *correctly* flips
            # the wind direction for KdVNL)
            self.P = -self.P
            self.G = -self.G
            self.psiP = -self.psiP

        if Height is not None:
            self.Height = Height
        else:
            self.Height = 2*np.sign(self.B*self.C)

        self.m = self._calculate_m()

    def _calculate_m(self):
        # If B and C are specified, then m is fixed
        m = self.Height*self.B/(3*self.C)

        # Round to num_sig_figs significant figures (to prevent
        # machine precision causing, eg, m = 1.0000000000000002 to
        # throw an error)
        # Source:
        # https://stackoverflow.com/questions/3410976/
        num_sig_figs = 5
        m_rounded = round_sig_figs(m, num_sig_figs)
        if m_rounded > 1 or m_rounded < 0:
            raise(ValueError("m = 2*B/3/C must be at least 0"+
                " and less than 1; m was calculated to be "+
                str(m_rounded)))

        if m_rounded == 1:
            # Due to machine precision, it is possible that m is
            # slightly larger than 1 (though not too much larger, since
            # we have already checked m=1 to num_sig_figs significant
            # figures). It is also possible that, due to machine
            # precision, m is slightly less than 1 when it should be
            # identically 1. Though this is less of an issue (any value
            # of m in [0,1] is allowed), we have specific routines in
            # place for m == 1 identically to deal with K(m) = infinity.
            # Therefore, it is beneficial to ensure that all m
            # approximately equal to 1 are rounded to 1 identically. In
            # either case, set m to 1 identically.
            m = 1

        return m

    def _cnoidal_wavelength(self):
        """Return the wavelength of a cnoidal wave."""
        K = spec.ellipk(self.m)

        WaveLength = 4*K

        return WaveLength

    def _regularize_wavelength(self, WaveLength):
        """Regularize cnoidal wavelength so we don't need an
        infinitely-long domain."""

        # Cut-off WaveLength at 80
        regularized_wave_length = min(WaveLength,80)

        return regularized_wave_length

    def set_spatial_grid(self, xLen=80, xNum=None,
            xStep=None, xOffset=None, WaveType=None,
            WaveLength=2*np.pi,NumWaves=1, spectral=False, *args,
            **kwargs):
        """Set the x coordinate grid.

        Parameters
        ----------
        xLen : float or None
            Length of x domain, or maximum domain length if WaveType is
            not None. Default is 80.
        WaveType : 'int_wave_lengths', 'cnoidal', or None
            If 'int_wave_lengths', choose domain length to fit NumWaves
            wavelengths of length WaveLength. If 'cnoidal', choose
            domain length to fit NumWaves wavelengths of a cnoidal wave
            wave-length. Default is None.
        xNum : float or None
            Number of grid points in x domain. If None, xStep
            must be specified. Default is None.
        xStep : float or None
            Spacing between grid points in x domain. If
            None, xNum must be specified. Default is 0.05.
        xOffset : float, 'nice_value', or None
            Distance that origin is offset from the center of the
            domain. 'nice_value' gives a nice shift slightly to the
            right a distance of 8*np.sqrt(abs(self.C)/abs(self.B)).
            Default is 0.
        WaveLength : float or None
            Wave length. Default is 2*np.pi.
        NumWaves : float or None
            Number of (periodic) waves to include in domain. Default is
            1.

        Returns
        -------
        None
        """

        self.NumWaves = NumWaves

        if WaveType == 'cnoidal':
            self.WaveLength = self._cnoidal_wavelength()
        else:
            self.WaveLength = WaveLength

        if xLen == 'cnoidal':
            # Ensure WaveLength isn't larger than xLen
            regularized_wave_length = \
                    self._regularize_wavelength(self.WaveLength)
        else:
            regularized_wave_length = self.WaveLength

        if WaveType == 'int_wave_lengths' or WaveType == 'cnoidal':
            self.xLen = self.NumWaves*regularized_wave_length
        else:
            self.xLen = xLen

        if xNum is not None:
            if xStep is not None:
                raise(ValueError('Exactly one of xNum or xStep must be specified'))
            self.xNum = xNum
        else:
            if xStep is None:
                # Use default value of xStep = 0.05
                xStep = 0.05
            if WaveType == 'int_wave_lengths' or WaveType == 'cnoidal':
                # xNum = xLen/xStep = NumWaves*WaveLength/xStep, but to
                # prevent rounding errors; round WaveLength/xStep, then
                # multiply by NumWaves
                self.xNum = int(round(regularized_wave_length/xStep)*
                        self.NumWaves)
            else:
                self.xNum = int(round(self.xLen/xStep))

        # Usually it would be dx=xLen/(xNum-1), but since we have
        # periodic boundary conditions, we don't need the -1
        self.dx = self.xLen / self.xNum

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

        self.spectral = False
        if spectral:
            self.spectral = True

            from dedalus import public as de
            from fractions import Fraction

            # De-alias by 3/2 for quadratic nonlinearity
            dealias_factor=Fraction(3,2)

            # Add dx to right endpoint since we specify [xmin,xmax) for
            # periodic BCs, but Dedalus wants closed intervals
            # [xmin,xmax+dx]
            endpoints = (self.x[0], self.x[-1]+self.dx)

            # Ensure that the number of de-aliased points
            # xNum*dealias_factor is a whole number
            denom = dealias_factor.denominator
            self.xNum = int(np.ceil(self.xNum/denom)*denom)

            # Create basis and domain
            x_basis = de.Chebyshev('x', self.xNum, interval=endpoints,
                    dealias=dealias_factor)
            self.domain = de.Domain([x_basis], np.float64)

            self.x = self.domain.grid(0)
            # Depending on the basis, it is likely that dx is no longer
            # constant
            self.dx = np.diff(self.x)

            # Print details of domain
            print('Number of Coefficients: '+str(self.domain.all_elements()[0].size))
            print('Grid spacing (avg): '+str(np.mean(self.domain.all_grid_spacings()[0])))
            print('Grid spacing (min): '+str(np.amin(self.domain.all_grid_spacings()[0])))
            print('Grid spacing (max): '+str(np.amax(self.domain.all_grid_spacings()[0])))

    def set_temporal_grid(self, tLen=3, tNum='density', spectral=False,
            *args, **kwargs):
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

        if tNum == 'density' and not spectral:
            self.tNum = int(self.tLen/(self.xLen/self.xNum)**4)
        elif tNum == 'density' and spectral:
            self.tNum = int(10*self.tLen/(self.xLen/self.xNum))
        else:
            self.tNum = tNum

        self.t, self.dt = np.linspace(0, self.tLen, self.tNum, retstep=True)

    def set_initial_conditions(self, y0='cnoidal', *args, **kwargs):
        """Set the initial conditions.
        Parameters
        ----------
        y0 : array_like or 'kdv'
            Given initial condition. If array_like, must have same size as
            self.x array. 'kdv' gives a cnoidal wave profile which
            satisfies the unforced KdV equation. Default is a 'cnoidal'.

        Returns
        -------
        None
        """

        # If we are using 'kdv' initial conditions, ensure that the
        # previously set height has the correct sign
        if np.sign(self.Height) != np.sign(self.B*self.C) \
                and y0 == 'cnoidal':
            raise(ValueError("sgn(H) must equal sgn(B*C)"))

        if type(y0) == np.ndarray:
            self.y0 = y0
            if self.spectral:
                raise(ValueError('Cannot provide y0 initial condition'+\
                        ' as matrix after passing spectral=True to'+\
                        ' set_spatial_grid'))
        elif y0 == 'kdv':
            m = self.m

            if m == 1:
                self.y0 = self.Height*1/np.cosh(np.sqrt(
                    self.Height*self.B/self.C/12)*self.x)**2
            else:
                K = spec.ellipk(m)
                E = spec.ellipe(m)

                cn = spec.ellipj(self.x/self.WaveLength*2*K,m)[1]
                trough = self.Height/m*(1-m-E/K)
                self.y0 = trough + self.Height*cn**2

        else:
            raise(ValueError("y0 must be array_like or 'cnoidal'"))

    def boost_frame(self, boostVelocity=0, **kwargs):
        """ Boost frame to boostVelocity.

        Boost the frame to boostVelocity by adjusting the F coefficient
        (corresponding to a Galilean boost or the introduction of a
        background current).

        Parameters
        ----------
        boostVelocity : float or 'kdv'
            The type of wave. If 'cnoidal', a value is chosen to cancel
            the propagation of the solution to the unforced KdV
            equation with zero damping.
        """

        if boostVelocity == 'kdv':
            # Choose value to give default, KdV solution zero
            # propagation velocity
            m = self.m
            K = spec.ellipk(m)
            E = spec.ellipe(m)

            self.F = -2/3*self.B*self.Height/m*(1-1/2*m-3/2*E/K)
        else:
            self.F = -boostVelocity*self.A

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
            # Choose indices so that we return xNum timesteps (or tNum,
            # whichever is smaller)
            snapshot_indxs = np.arange(start=0, stop=self.tNum,
                    step=max(1,int(self.tNum/self.xNum)))
            # Ensure we also capture the last timestep
            if self.tNum-1 not in snapshot_indxs:
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
            snapshot_ts = self.t[snapshot_indxs]

        self.snapshot_indxs = snapshot_indxs
        self.snapshot_ts = snapshot_ts

    def _kdvb(self, t, u, terms=False, *args, **kwargs):
        """Differential equation for the KdV-Burgers equation."""
        ux = derivative(u, dx=self.dx, period=self.xLen, order=1, **kwargs)
        uxx = derivative(u, dx=self.dx, period=self.xLen, order=2, **kwargs)
        uxxx = derivative(u, dx=self.dx, period=self.xLen, order=3, **kwargs)
        uxxxx = derivative(u, dx=self.dx, period=self.xLen, order=4, **kwargs)

        # Compute du/dt
        dudt = -u*ux*self.B/self.A -uxxx*self.C/self.A \
                -ux*self.F/self.A + uxx*self.G/self.A \
                -uxxxx*self.H/self.A

        # Return the contributions of the individual terms
        if terms:
            return dudt, {'Change':dudt, 'Advection':self.B/self.A*u*ux,
                    'Dispersion':self.C/self.A*uxxx,
                    'Current':self.F/self.A*ux, 'Wind':-self.G/self.A*uxx,
                    'Hyperviscosity':self.H/self.A*uxxxx}
        else:
            return dudt

    def _kdvnl(self, t, u, periodic_deriv=False, terms=True, *args, **kwargs):
        """Differential equation for the nonlocal-KdV equation."""

        ux = derivative(u, dx=self.dx, period=self.xLen, order=1, **kwargs)
        uxxx = derivative(u, dx=self.dx, period=self.xLen, order=3, **kwargs)
        uxxxx = derivative(u, dx=self.dx, period=self.xLen, order=4, **kwargs)
        uxnl = np.roll(ux,
                shift=int(round(-self.psiP*self.WaveLength/(2*np.pi)/self.dx)),
                axis=0)

        # Compute du/dt
        dudt = -u*ux*self.B/self.A -uxxx*self.C/self.A \
                -ux*self.F/self.A + uxnl*self.G/self.A \
                -uxxxx*self.H/self.A

        # Return the contributions of the individual terms
        if terms:
            return dudt, {'Change':dudt, 'Advection':self.B/self.A*u*ux,
                    'Dispersion':self.C/self.A*uxxx,
                    'Current':self.F/self.A*ux, 'Wind':self.G/self.A*uxx,
                    'Hyperviscosity':self.H/self.A*uxxxx}
        else:
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
                    t_eval=self.snapshot_ts,
                    method='Radau',
                    vectorized=True,
                    jac_sparsity=self._JacSparcity()
                    )
        else:
            sol = sp.integrate.solve_ivp(
                    lambda t,u: eqn(t,u,deriv_type='periodic_fd'),
                    (0,self.tLen),
                    self.y0,
                    t_eval=self.snapshot_ts,
                    method='RK23',
                    )

        if sol['status'] != 0:
            print(sol['message'])

        self.sol = sol['y']


    def solve_system_ab3(self, max_slope=np.inf, *args, **kwargs):
        """Use 3rd-order Adams-Bashforth method to solve the
        differential equation on a periodic domain. If self.diffeq ==
        'KdVB', solve the KdV-Burgers equation; if self.diffeq ==
        'KdVNL', solve the nonlocal KdV equation."""

        dt = self.dt

        nx = self.x.size
        nt = self.t.size
        nt_snapshots = self.snapshot_indxs

        y = np.empty((nt_snapshots.size,nx), dtype=np.float128)
        y[0,:] = self.y0

        if self.diffeq == 'KdVB':
            rhs_eqn = self._kdvb
        elif self.diffeq == 'KdVNL':
            rhs_eqn = self._kdvnl

        # Save the contributions of the individual terms
        PDEterms = {k: np.empty((nt_snapshots.size,nx),
            dtype=np.float128) for k in ['Change', 'Advection',
            'Dispersion', 'Current', 'Wind', 'Hyperviscosity']}
        terms = rhs_eqn(0, y[0,:], deriv_type='periodic_fd',
                terms=True)[1]
        for key in terms:
            PDEterms[key][0,:] = terms[key]

        yn = y[0,:]
        for n in range(0,nt-1):
            if n == 0:
                # First step is an Euler step
                RHS0, terms = rhs_eqn(n*dt, y[n,:],
                        deriv_type='periodic_fd', terms=True)
                # Calculate next step
                yn = yn + dt*RHS0

                RHS1 = RHS0

            elif n == 1:
                # Second step is a 2nd order Adams-Bashforth step
                RHS0, terms = rhs_eqn(n*dt, y[n,:],
                        deriv_type='periodic_fd', terms=True)
                # Calculate next step
                yn = yn + dt*(3/2*RHS0 - 1/2*RHS1)

                RHS2 = RHS1
                RHS1 = RHS0
            else:
                # Later steps are all 3rd order Adams-Bashforth steps
                RHS0, terms = rhs_eqn(n*dt, yn,
                        deriv_type='periodic_fd', terms=True)
                # Calculate next step
                yn = yn + dt*(23/12*RHS0 - 16/12*RHS1 + 5/12*RHS2)

                RHS2 = RHS1
                RHS1 = RHS0

            # Check if maximum slope is less than max_slope
            if max_slope != np.inf:
                maxSlope = np.max(np.abs(
                    derivative(yn, dx=self.dx, period=self.xLen,
                        order=1)))
                if maxSlope > max_slope:
                    warnings.warn('Local slope '+str(maxSlope)+\
                            ' exceeds max_slope threshold')

                    # Truncate y to previous value of snapshot_indx
                    y = y[:snapshot_indx,:]

                    # Truncate times
                    self.snapshot_ts = self.snapshot_ts[:snapshot_indx]
                    self.tLen = self.snapshot_ts[-1] - self.t[0]
                    self.t = self.t[self.t <= self.tLen]
                    self.tNum = self.t.size

                    break

            if n+1 in nt_snapshots:
                # Find index of n in nt_snapshots
                snapshot_indx = np.nonzero(nt_snapshots ==
                        n+1)[0].item()
                y[snapshot_indx,:] = yn
                for key in terms:
                    PDEterms[key][snapshot_indx,:] = \
                            terms[key].transpose()

        self.sol = y.transpose()
        self.PDEterms = PDEterms

    def solve_system_rk3(self, max_slope=np.inf, *args, **kwargs):
        """Use 3rd-order Runge-Kutta method to solve the
        differential equation on a periodic domain. If self.diffeq ==
        'KdVB', solve the KdV-Burgers equation; if self.diffeq ==
        'KdVNL', solve the nonlocal KdV equation."""

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

        dt = self.dt

        nx = self.x.size
        nt = self.t.size
        nt_snapshots = self.snapshot_indxs

        y = np.empty((nt_snapshots.size,nx), dtype=np.float128)
        y[0,:] = self.y0

        if self.diffeq == 'KdVB':
            rhs_eqn = self._kdvb
        elif self.diffeq == 'KdVNL':
            rhs_eqn = self._kdvnl

        # Save the contributions of the individual terms
        PDEterms = {k: np.empty((nt_snapshots.size,nx),
            dtype=np.float128) for k in ['Change', 'Advection',
            'Dispersion', 'Current', 'Wind', 'Hyperviscosity']}
        terms = rhs_eqn(0, y[0,:], deriv_type='periodic_fd',
                terms=True)[1]
        for key in terms:
            PDEterms[key][0,:] = terms[key]

        yn = y[0,:]
        for n in range(0,nt-1):

            # RK Stages
            k = np.zeros((yn.size,s))
            tn = n*dt
            for j in range(0,s):
              # Overwrite 'terms' so we only have the final version
              k[:,j], terms = rhs_eqn(tn+c[j]*dt,
                      yn+dt*np.dot(k,a[j,:]), deriv_type='periodic_fd',
                      terms=True)

            # Calculate next step
            yn = yn + dt*np.dot(k,b)

            # Check if maximum slope is less than max_slope
            if max_slope != np.inf:
                maxSlope = np.max(np.abs(
                    derivative(yn, dx=self.dx, period=self.xLen,
                        order=1)))
                if maxSlope > max_slope:
                    warnings.warn('Local slope '+str(maxSlope)+\
                            ' exceeds max_slope threshold')

                    # Truncate y to previous value of snapshot_indx
                    y = y[:snapshot_indx,:]

                    # Truncate times
                    self.snapshot_ts = self.snapshot_ts[:snapshot_indx]
                    self.tLen = self.snapshot_ts[-1] - self.t[0]
                    self.t = self.t[self.t <= self.tLen]
                    self.tNum = self.t.size

                    break

            if n+1 in nt_snapshots:
                # Find index of n in nt_snapshots
                snapshot_indx = np.nonzero(nt_snapshots ==
                        n+1)[0].item()
                y[snapshot_indx,:] = yn
                for key in terms:
                    PDEterms[key][snapshot_indx,:] = \
                            terms[key].transpose()


        self.sol = y.transpose()
        self.PDEterms = PDEterms

    def solve_system_spectral(self, max_slope=np.inf, *args, **kwargs):
        """Use a spectral method to solve the differential equation on a
        periodic domain. This is currently only implemented for the
        KdV-Burgers equation so we must have self.diffeq == 'KdVB'."""

        if self.diffeq != 'KdVB':
            raise(ValueError("Spectral solver can only solve the"+\
                    " KdV-Burgers equation (self.diffeq='KdVB'),"+\
                    " but '"+self.diffeq+"' was passed"))

        from dedalus import public as de
        from dedalus.extras import flow_tools

        # Source: https://dedalus-project.readthedocs.io/en/latest/notebooks/dedalus_tutorial_problems_solvers.html

        # We don't use a hyperviscosity for spectral methods
        self.H = 0

        # We need to transform the third order KdV-B equation to a
        # system of first order equations
        # A*d(u)/dt + F*ux + B*u*ux + C*uxxx = G*uxx
        # to
        # (A)           (-F ,G ,-C)           (u*ux)
        # (0)*d(U)/dt = (0  ,0  ,0)*d(U)/dx + (0)
        # (0)           (0  ,0  ,0)           (0)
        # with U = (u,ux,uxx).T and (3) boundary conditions
        # (1,0,0)*U(xmin,t) = 0,
        # (0,1,0)*U(xmin,t) = 0,
        # (0,1,0)*U(xmax,t) = 0,
        # and initial conditions
        # U(x,0) = (u0, d(u0)/dx, d^2(u0)/dx^2)

        # Create problem
        problem = de.IVP(self.domain, variables=['u', 'ux', 'uxx'])

        # Apply Dirichlet preconditioning
        problem.meta[:]['x']['dirichlet'] = True

        # Define parameters
        problem.parameters['A'] = self.A
        problem.parameters['B'] = self.B
        problem.parameters['C'] = self.C
        problem.parameters['F'] = self.F
        problem.parameters['G'] = self.G

        # Main equation, with linear terms on the LHS and nonlinear terms on the RHS
        problem.add_equation("A*dt(u) + F*dx(u) + C*dx(uxx) - G*dx(ux) = -B*u*ux")
        # Auxiliary equations defining the first-order reduction
        problem.add_equation("ux - dx(u) = 0")
        problem.add_equation("uxx - dx(ux) = 0")
        # Boundary conditions (set u(xMax) = 0 since this is the
        # upstream side)
        problem.add_bc('right(u) = 0')
        problem.add_bc('left(ux) = 0')
        problem.add_bc('right(ux) = 0')

        # Select timestepping method
        solver = problem.build_solver(de.timesteppers.RK443)

        # Reference local grid and state fields
        u = solver.state['u']
        ux = solver.state['ux']
        uxx = solver.state['uxx']

        # Differentiate initial conditions and store results in
        # self.state
        u['g'] = self.y0

        u.differentiate('x', out=ux)
        ux.differentiate('x', out=uxx)

        # Stop stopping criteria
        solver.stop_sim_time = self.tLen
        solver.stop_wall_time = np.inf
        solver.stop_iteration = np.inf

        dt = self.dt

        # Save the snapshots
        # Since the evaluator requires a constant sim_dt, we need to
        # take the average dt form snapshot_ts; this is slightly less
        # flexible than allowing arbitrary snapshot_ts, but is required
        # to use dynamic timestepping
        snapshot_dt = (self.snapshot_ts[-1]-self.snapshot_ts[0])/\
                self.snapshot_ts.size

        # Define DictionaryHandler that keeps state
        from dedalus.core.evaluator import Handler
        class StatefulDictionaryHandler(Handler):
            """Handler that stores outputs in a stateful dictionary."""

            def __init__(self, *args, **kw):
                Handler.__init__(self, *args, **kw)
                self.fields = dict()

            def __getitem__(self, item):
                return self.fields[item]

            def process(self, **kw):
                """Reference fields from dictionary."""
                for task in self.tasks:
                    task['out'].set_scales(task['scales'], keep_data=True)
                    task['out'].require_layout(task['layout'])
                    if not task['name'] in self.fields:
                        self.fields[task['name']] = task['out'].data
                    else:
                        self.fields[task['name']] = np.vstack(
                                (self.fields[task['name']],
                                    task['out'].data))

        # Save the contributions of the individual terms
        analyzer = solver.evaluator.add_handler(StatefulDictionaryHandler(
            solver.domain, solver.evaluator.vars, sim_dt=snapshot_dt))
        analyzer.add_task("t", layout='g', name='t', scales=1)
        analyzer.add_task("u", layout='g', name='u', scales=1)
        analyzer.add_task(
                "-F/A*dx(u) - C/A*dx(uxx) + G/A*dx(ux) -B/A*u*ux",
                layout='g', name='Change', scales=1)
        analyzer.add_task("B/A*u*ux", layout='g', name='Advection', scales=1)
        analyzer.add_task("C/A*dx(uxx)", layout='g', name='Dispersion', scales=1)
        analyzer.add_task("F/A*dx(u)", layout='g', name='Current', scales=1)
        analyzer.add_task("-G/A*dx(ux)", layout='g', name='Wind', scales=1)
        analyzer.add_task("0", layout='g', name='Hyperviscosity', scales=1)

        # Add CFL condition to dynamically calculate dt
        CFL = flow_tools.CFL(solver, initial_dt=dt, safety=0.3,
                             max_change=1.5, min_change=0.5)
        # The local velocity u=dx(phi)+O(eps) is equal also
        # u=eta+O(eps), so the velocity is equal to our primary field
        # eta (which is called 'u' here)
        CFL.add_velocity('u', axis=0)

        # Main loop
        while solver.ok:
            solver.step(dt)

            # Check if maximum slope is less than max_slope
            if max_slope != np.inf:
                maxSlope = np.max(np.abs(u.differentiate('x')['g']))
                if maxSlope > max_slope:
                    warnings.warn('Local slope '+str(maxSlope)+\
                            ' exceeds max_slope threshold')

                    break

            # Update dt
            dt = CFL.compute_dt()

        self.sol = analyzer['u'].transpose()
        self.PDEterms = {k : analyzer[k] for k in
                ['Change', 'Advection', 'Dispersion', 'Current', 'Wind',
                    'Hyperviscosity']}
        self.snapshot_ts = analyzer['t'][:,0]
        self.tNum = self.snapshot_ts.size
        self.tLen = self.snapshot_ts[-1]

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
        return self.sol


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
        # Confusingly, sp.signal.hilbert gives the analytic signal, x -> x + i*H(x)
        # so take the imaginary part
        solHilbert = np.imag(sp.signal.hilbert(sol,axis=0))

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
        boostVelocity : float, 'kdv', or None
            The velocity of the initial, wave frame relative to the lab
            frame in units of [x]/[t]. 'kdv' boosts into the
            co-moving frame of the solution to the unforced KdV
            equation. Default is -1/6.

        Returns
        -------
        None
        """

        if boostVelocity == 'kdv':
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

def convert_array_to_xarray(system, array, DataSet=False):

    # Hide solution outside of window
    system.set_x_window()
    xMasked = system.get_masked_x()

    # Normalize x by wavelength
    # Convert from x' = x*k_E to x'/sqrt(mu) = x*k_E/k_E/h = x/h
    # (Primes denote the nondim variables used throughout this solver)
    xMasked = xMasked/np.sqrt(system.mu)

    # Extract snapshot times
    # t_1' = epsilon*t' = epsilon*t*sqrt(g*h)*k_E
    snapshot_ts = system.snapshot_ts

    if not DataSet:
        # Package snapshots, xs, and ts together in a DataArray
        data = xr.DataArray(array,
                dims=('x/h','t*eps*sqrt(g*h)*k_E'),
                coords=[xMasked, snapshot_ts])
    else:
        # Package snapshots, xs, and ts together in a DataSet
        data = xr.Dataset(data_vars={
            k : (('t*eps*sqrt(g*h)*k_E','x/h'), v) for k,v in
            array.items()},
            coords={'x/h':xMasked, 't*eps*sqrt(g*h)*k_E':snapshot_ts})

    # Remove data for x with masked values
    data = data.where(~np.isnan(data['x/h']),drop=True)

    return data

def default_solver(y0_func=None, solver='Spectral', *args, **kwargs):

    forcing_type_dict = {'Jeffreys' : 'KdVB', 'GM' : 'KdVNL'}
    if 'forcing_type' in kwargs:
        kwargs['diffeq'] = forcing_type_dict[kwargs['forcing_type']]
    if 'xLen' not in kwargs and kwargs.get('wave_type') == 'cnoidal':
        kwargs['WaveType'] = kwargs['wave_type']
    if 'boostVelocity' not in kwargs and (kwargs.get('wave_type') ==
            'cnoidal' or kwargs.get('wave_type') == 'solitary'):
        kwargs['boostVelocity'] = 'kdv'
    if 'y0' not in kwargs and (kwargs.get('wave_type') == 'cnoidal' or
            kwargs.get('wave_type') == 'solitary'):
        kwargs['y0'] = 'kdv'
    if solver == 'Spectral':
        kwargs['spectral'] = True

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
    # Boost to co-moving frame
    solverSystem.boost_frame(**kwargs)
    # Set snapshot times
    solverSystem.set_snapshot_ts(**kwargs)
    # Solve KdV-Burgers system
    if solver == 'Builtin':
        solverSystem.solve_system_builtin()
    elif solver == 'RK3':
        solverSystem.solve_system_rk3()
    elif solver == 'AB3':
        solverSystem.solve_system_ab3()
    elif solver == 'Spectral':
        solverSystem.solve_system_spectral(**kwargs)
    else:
        raise(ValueError("'solver' must be"+\
                " {'Builtin','RK3','AB3','Spectral'},"+\
                " but "+solver+" was provided"))

    # Convert back to non-normalized variables
    # Convert from eta' = eta/a = eta/h/eps to eta'*eps = eta/a*eps = eta/h
    # (Primes denote the nondim variables used throughout this solver)
    snapshots = solverSystem.get_snapshots()*solverSystem.eps

    # Extract xarray of data
    data = convert_array_to_xarray(solverSystem, snapshots)

    return data, solverSystem

def gen_trig_verf(save_prefix, nu_bi=3e-3):
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
    snapshot_fracs = np.arange(100)/99

    for nu_bi_val in [nu_bi, 0]:
        parameters = {
                'A' : 1,
                'B' : 0,
                'C' : 1/6, # C = 1/6*mu/eps, so this requires mu=eps
                'P' : 0,
                'nu_bi' : nu_bi_val,
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
                snapshot_fracs=snapshot_fracs,
                boostVelocity=0,
                solver=solver,
                )

            data_csv.save_data(data,
                    save_prefix+'TrigVerf-'+solver+'_nu_bi'+str(nu_bi_val),
                    **parameters, solver=solver, stack_coords = True)

def gen_long_verf(save_prefix, mu=0.8, nu_bi=3e-3):
    # Generate verification data by running a solitary wave profile
    # without forcing for a long time

    tLen = 100
    snapshot_fracs = np.array([0,1/3,2/3,1])
    full_snapshot_fracs = np.arange(100)/99


    for nu_bi_val in [nu_bi, 0]:
        for wave_type in ['solitary', 'cnoidal']:
            parameters = {
                    'mu' : mu,
                    'P' : 0,
                    'nu_bi' : nu_bi_val,
                    'wave_type' : wave_type,
                    }

            # Use default mu for solitary waves
            if wave_type == 'solitary':
                parameters.pop('mu')

            data, dataClass = default_solver(**parameters,
                    tLen=tLen,
                    snapshot_fracs=full_snapshot_fracs,
                    )

            # Get default mu for solitary waves
            if wave_type == 'solitary':
                parameters['mu'] = dataClass.mu

            data_csv.save_data(data,
                    save_prefix+'LongVerf_nu_bi'+str(nu_bi_val),
                    stack_coords = True, eps=dataClass.eps,
                    **parameters)

def gen_snapshots(save_prefix, eps=0.1, mu=0.8, P=0.25, psiP=3/4*np.pi,
        nu_bi=3e-3):
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
    nu_bi : float
        Nondimensional biviscosity needed for numerical stability.
        Default is 3e-3.
    """

    for forcing_type in ['Jeffreys']:
        for wave_type in ['solitary']:
            if forcing_type == 'GM' and wave_type == 'solitary':
                # We don't need GM applied to solitary waves, as it
                # doesn't make as much sense
                continue
            for P_val in P*np.array([-1,-0.5,0,0.5,1]):

                for counter, mu_val in enumerate(mu*np.array([1,7/8])):

                    parameters = {
                            'eps' : eps,
                            'mu' : mu_val,
                            'wave_type' : wave_type,
                            'forcing_type' : forcing_type,
                            'P' : P_val,
                            'psiP' : psiP,
                            'nu_bi' : nu_bi,
                            }

                    # Jeffreys does not just psiP
                    if forcing_type == 'Jeffreys':
                        parameters.pop('psiP')

                    # Use default mu for solitary waves
                    if wave_type == 'solitary':
                        if counter == 0:
                            parameters.pop('mu')
                        else:
                            # Since solitary waves all use the same mu,
                            # don't duplicate the calculation
                            continue

                    # Run model
                    data, dataClass = default_solver(**parameters)

                    # Get default mu for solitary waves
                    if wave_type == 'solitary':
                        parameters['mu'] = dataClass.mu

                    # Save data
                    data_csv.save_data(data, save_prefix+'Snapshots',
                            wave_length=dataClass.WaveLength,
                            **parameters, stack_coords=True)

                    # Extract xarray of data
                    terms = convert_array_to_xarray(dataClass,
                            dataClass.PDEterms, DataSet=True)

                    # Save individual terms of PDE
                    data_csv.save_data(terms[{'t*eps*sqrt(g*h)*k_E':-1}],
                            save_prefix+'Terms',
                            wave_length=dataClass.WaveLength,
                            **parameters, stack_coords=False)

def gen_depth_varying(save_prefix, eps=0.1, mu=0.6, P=0.25, psiP=3/4*np.pi,
        nu_bi=3e-3, forcing_type='Jeffreys'):

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
                'nu_bi' : nu_bi,
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

def gen_press_varying(save_prefix, eps=0.1, mu=0.8, P=0.25, psiP=3/4*np.pi,
        nu_bi=3e-3, forcing_type='Jeffreys'):

    # Linearly space Ps
    P_vals = P*np.linspace(-1,1,num=30)

    for P_val in P_vals:
        for wave_type in ['solitary','cnoidal']:
            parameters = {
                    'eps' : eps,
                    'mu' : mu,
                    'wave_type' : wave_type,
                    'forcing_type' : forcing_type,
                    'P' : P_val,
                    'psiP' : psiP,
                    'nu_bi' : nu_bi,
                    }

            # Jeffreys does not just psiP
            if forcing_type == 'Jeffreys':
                parameters.pop('psiP')

            # Use default mu for solitary waves
            if wave_type == 'solitary':
                parameters.pop('mu')

            # Run model
            data, dataClass = default_solver(**parameters, tLen=3)

            # Save data
            data_csv.save_data(data, save_prefix+'PressVarying',
                    wave_length=dataClass.WaveLength,
                    **parameters, stack_coords=True)

def gen_biviscosity_variation(save_prefix, eps=0.1, mu=0.8, P=0, psiP=3/4*np.pi,
        nu_bi=3e-3, forcing_type='Jeffreys', wave_type='cnoidal'):

    nu_bi_vals = np.vectorize(round_sig_figs)(nu_bi*np.logspace(-2,1,4),3)
    # Also include nu_bi=0
    nu_bi_vals = np.insert(nu_bi_vals, 0, 0)

    for nu_bi_val in nu_bi_vals:

       parameters = {
               'eps' : eps,
               'mu' : mu,
               'wave_type' : wave_type,
               'forcing_type' : forcing_type,
               'P' : P,
               'psiP' : psiP,
               'nu_bi' : nu_bi_val,
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
       data_csv.save_data(data,
               save_prefix+'Biviscosity_nubi'+str(nu_bi_val),
               wave_length=dataClass.WaveLength,
               **parameters, stack_coords=True)

def gen_decaying_no_nu_bi(save_prefix, eps=0.1, mu=0.8, P=0.25,
        psiP=3/4*np.pi, nu_bi=0, forcing_type='Jeffreys',
        wave_type='cnoidal'):

     for P_val in P*np.array([-1,-0.5,0]):

         for mu_val in mu*np.array([1,7/8]):

             parameters = {
                     'eps' : eps,
                     'mu' : mu_val,
                     'wave_type' : wave_type,
                     'forcing_type' : forcing_type,
                     'P' : P_val,
                     'psiP' : psiP,
                     'nu_bi' : nu_bi,
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
             data_csv.save_data(data, save_prefix+'Decaying-no-NuBi',
                     wave_length=dataClass.WaveLength,
                     **parameters, stack_coords=True)

             if wave_type == 'solitary':
                 # We don't use mu for solitary waves, so break
                 # after a single run (since changing mu won't
                 # affect the output)
                 break

def gen_validity_domain(save_prefix, eps=0.1, mu=0.8, P=0.25, psiP=3/4*np.pi,
        nu_bi=3e-3, xStep=1e-1, tStep=1e-2, max_tLen=100,
        max_slope=10000):

    nu_bi_vals = np.array([0])
    xStep_vals = np.vectorize(round_sig_figs)(xStep*np.logspace(0,2,4),3)
    tNum_vals = np.ceil((max_tLen/tStep*np.logspace(-1,1,5))).astype(int)

    parameter_array = np.einsum('i,j,k',xStep_vals,tNum_vals,nu_bi_vals)

    for forcing_type in ['Jeffreys']:
        for wave_type in ['solitary']:
            if forcing_type == 'GM' and wave_type == 'solitary':
                # We don't need GM applied to solitary waves, as it
                # doesn't make as much sense
                continue
            for mu_counter, mu_val in enumerate(mu*np.array([1])):

                metrics = xr.Dataset(
                        data_vars={
                    'Stopping time' : (('xStep', 'tNum', 'nu_bi'),
                        np.zeros(parameter_array.shape)),
                    'Normalized RMS change' : (('xStep', 'tNum', 'nu_bi'),
                        np.zeros(parameter_array.shape)),
                    },
                    coords={'xStep':xStep_vals, 'tNum':tNum_vals,
                        'nu_bi':nu_bi_vals})

                for xStep_counter, xStep_val in enumerate(xStep_vals):
                    for tNum_counter, tNum_val in enumerate(tNum_vals):
                        for nu_bi_counter, nu_bi_val in enumerate(nu_bi_vals):

                            parameters = {
                                    'tLen' : max_tLen,
                                    'xStep' : xStep_val,
                                    'tNum' : tNum_val,
                                    'eps' : eps,
                                    'mu' : mu_val,
                                    'wave_type' : wave_type,
                                    'forcing_type' : forcing_type,
                                    'P' : P,
                                    'psiP' : psiP,
                                    'nu_bi' : nu_bi_val,
                                    'max_slope' : max_slope,
                                    }

                            # Jeffreys does not just psiP
                            if forcing_type == 'Jeffreys':
                                parameters.pop('psiP')

                            # Use default mu for solitary waves
                            if wave_type == 'solitary':
                                if mu_counter == 0:
                                    parameters.pop('mu')
                                else:
                                    # Since solitary waves all use the same mu,
                                    # don't duplicate the calculation
                                    continue

                            # Suppress warnings
                            warnings.filterwarnings("ignore")

                            # Run model
                            data, dataClass = default_solver(**parameters)

                            # Un-suppress warnings
                            warnings.resetwarnings()

                            # Get default mu for solitary waves
                            if wave_type == 'solitary':
                                parameters['mu'] = dataClass.mu

                            # Check what the final time was
                            metrics['Stopping time'][{'xStep':xStep_counter,
                                        'tNum':tNum_counter,
                                        'nu_bi':nu_bi_counter}] = \
                                                dataClass.tLen

                            # Calculate the normalized RMS change from
                            # the initial profile

                            # Subtract last time from first time
                            difference = data[{'t*eps*sqrt(g*h)*k_E':-1}] - \
                                    data[{'t*eps*sqrt(g*h)*k_E':0}]

                            # Take the L2 norm
                            L2diff = scipy.integrate.trapz(difference**2, dx=dataClass.dx,
                                    axis=0)/(dataClass.xLen)

                            # Normalize by the original L2 norm
                            L2orig = scipy.integrate.trapz(
                                    data[{'t*eps*sqrt(g*h)*k_E':0}]**2, dx=dataClass.dx,
                                    axis=0)/(dataClass.xLen)
                            L2ratio = np.sqrt(L2diff/L2orig)

                            metrics['Normalized RMS change']\
                                    [{'xStep':xStep_counter,
                                        'tNum':tNum_counter,
                                        'nu_bi':nu_bi_counter}] = L2ratio

                # Replace tNum with tStep
                metrics = metrics.assign_coords({'tStep':
                    metrics['tNum']/max_tLen})
                metrics = metrics.swap_dims({'tNum' : 'tStep'})
                metrics = metrics.drop_vars('tNum')

                # Save data
                data_csv.save_data(metrics, save_prefix+'Metrics',
                        wave_length=dataClass.WaveLength,
                        **parameters, stack_coords=False)

def main():
    save_prefix = '../Data/Raw/'

    callable_functions = {
#            'trig_verf' : gen_trig_verf,
#            'long_verf' : gen_long_verf,
            'snapshots' : gen_snapshots,
#            'depth_varying' : gen_depth_varying,
#            'press_varying' : gen_press_varying,
#            'biviscosity' : gen_biviscosity_variation,
#            'decaying_no_nu_bi' : gen_decaying_no_nu_bi,
#            'validity_domain' : gen_validity_domain,
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
