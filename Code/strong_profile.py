#!/usr/bin/env python3
# strong_profile.py
''' Generate plots depicting the evolution of the analytic solution to
strongly wind-forced waves in shallow water
'''

import numpy as np
import scipy as sp
import scipy.signal
import scipy.special as spec
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
import matplotlib.gridspec as gridspec
import texplot

def cnoidal_profile(press_type, xs, P=1, psi_P=np.pi/2, depth=np.inf,
        eps=0.1, A_mag=1, t=0, m=0.8):
    """ Generate wave profile under a Jeffreys and Generalized Miles
    (GM) pressureforcing.

    Parameters
    ----------
    press_type : 'Jeffreys' or 'GM'
        The pressure forcing type applied. With $\eta(x,t)$ the wave
        profile, 'Jeffreys' corresponds to a pressure of the form
        $P*\partial \eta/\partial x$ while 'GM' corresponds to
        $P*\eta(x+psiP,t)$.
    xs : ndarray
        A one dimensional array of giving the x-position to at which the
        profile should be evaluated.
    P : float
        The magnitude of the nondimensional pressure forcing $P
        k/(rho_w*g)$. Default is 1.
    psi_P : float
        The wind phase of the pressure forcing. Default is np.pi/2.
    depth : float or np.inf
        The nondimensional depth $k*h$. Default is np.inf.
    eps : float
        Initial nondimensional wave height $a_0/h$. Default is 0.1.
    A_mag : float
        Initial magnitude of the first harmonic. Default is 1.
    t : float
        Time at which to evaluate the profile. Default is 0.
    explicit_result : boolean
        Use the explicitly calculated result. If false, t must be
        non-zero
    m : float
        The elliptic parameter in the range [0,1] inclusive. Default is
        0.8.

    Returns
    -------
    profile : ndarray
        A one-dimensional array with the same size as xs. Represents the
        wave profile, evaluated at each value of xs, at time t.
    """

    mP = np.sqrt(1-m**2)
    E = spec.ellipe(m)
    K = spec.ellipk(m)
    c1 = 8/3/np.pi**2*m**2*K**2 # Amplitude of wave; set by requiring the period to be 2*pi

    sn = spec.ellipj(np.sqrt(3*c1/8)*(xs+0*t*(c1*eps/4))/m,m)[0]
    cn = spec.ellipj(np.sqrt(3*c1/8)*(xs+0*t*(c1*eps/4))/m,m)[1]
    dn = spec.ellipj(np.sqrt(3*c1/8)*(xs+0*t*(c1*eps/4))/m,m)[2]

    if press_type == 'Jeffreys':
        A = c1*(cn**2 - (E/K - mP**2)/m**2)
        diffA = -2*cn*sn*dn*c1*np.sqrt(3*c1/8)
        diffDiffA = -2*(dn**2*cn**2-m*cn**2*sn**2-sn**2*dn**2)*c1*(3*c1/8)
        diffDiffDiffA = 8*sn*cn*dn*(dn**2+m*cn**2-m*sn**2)*c1*np.sqrt(3*c1/8)*(3*c1/8)

        firstOrderTerm = A
        secondOrderTerm = -P/64/np.pi**2*t \
                *(
                        5*diffA**2
                        +4*A*diffDiffA
                        +3*t*(1+c1*eps/2)*(
                            2*diffA*diffDiffA
                            -A*diffDiffDiffA
                            )
                 )

    elif press_type == 'GM':
        snpsiP = spec.ellipj(np.sqrt(3/8)*(xs+0*t*(c1*eps/4)+psi_P)/m,m)[0]
        cnpsiP = spec.ellipj(np.sqrt(3/8)*(xs+0*t*(c1*eps/4)+psi_P)/m,m)[1]
        dnpsiP = spec.ellipj(np.sqrt(3/8)*(xs+0*t*(c1*eps/4)+psi_P)/m,m)[2]

        A = c1*(cn**2 - (E/K - mP**2)/m**2)
        ApsiP = cnpsiP**2 - (E/K - mP**2)/m**2
        diffA = -2*cn*sn*dn*c1*np.sqrt(3*c1/8)
        diffApsiP = -2*cnpsiP*snpsiP*dnpsiP*c1*np.sqrt(3*c1/8)
        diffDiffA = -2*(dn**2*cn**2-m*cn**2*sn**2-sn**2*dn**2)*c1*(3*c1/8)
        diffDiffApsiP = -2*(dnpsiP**2*cnpsiP**2-m*cnpsiP**2*snpsiP**2
                -snpsiP**2*dnpsiP**2)*c1*(3*c1/8)
        diffDiffDiffA = 8*sn*cn*dn*(dn**2+m*cn**2-m*sn**2
                )*c1*np.sqrt(3*c1/8)*(3*c1/8)
        diffDiffDiffApsiP = 8*snpsiP*cnpsiP*dnpsiP*(dnpsiP**2+m*cnpsiP**2
                -m*snpsiP**2)*c1*np.sqrt(3*c1/8)*(3*c1/8)

        firstOrderTerm = A
        secondOrderTerm = -P/64/np.pi**2*t \
                *(
                        9*ApsiP*diffApsiP
                        -4*ApsiP*diffA
                        -5*A*diffApsiP
                        +3*t*(1+c1*eps/2)*(
                            diffApsiP**2
                            +ApsiP*diffDiffApsiP
                            -diffApsiP*diffA
                            -diffDiffApsiP*A
                            )
                 )
    else:
        raise(ValueError("forcing_type must be either 'Jeffreys' or 'GM', but "
            +press_type+" was given"))


    first = eps*firstOrderTerm
    second = eps**2*secondOrderTerm
    profile = first + second
    return profile

def solitary_profile(press_type, xs, P=1, psi_P=np.pi/2, depth=np.inf,
        eps=0.1, A_mag=1, t=0,explicit_result=True):
    """ Generate wave profile under a Jeffreys and Generalized Miles
    (GM) pressureforcing.

    Parameters
    ----------
    press_type : 'Jeffreys' or 'GM'
        The pressure forcing type applied. With $\eta(x,t)$ the wave
        profile, 'Jeffreys' corresponds to a pressure of the form
        $P*\partial \eta/\partial x$ while 'GM' corresponds to
        $P*\eta(x+psiP,t)$.
    xs : ndarray
        A one dimensional array of giving the x-position to at which the
        profile should be evaluated.
    P : float
        The magnitude of the nondimensional pressure forcing $P
        k/(rho_w*g)$. Default is 1.
    psi_P : float
        The wind phase of the pressure forcing. Default is np.pi/2.
    depth : float or np.inf
        The nondimensional depth $k*h$. Default is np.inf.
    eps : float
        Initial nondimensional wave height $a_0/h$. Default is 0.1.
    A_mag : float
        Initial magnitude of the first harmonic. Default is 1.
    t : float
        Time at which to evaluate the profile. Default is 0.
    explicit_result : boolean
        Use the explicitly calculated result. If false, t must be
        non-zero

    Returns
    -------
    fig : Figure
        Figure containing plot of profile under the action of Jeffreys
        and Generalized Miles (GM) forcings.
    """

    # Initial wave amplitude
    c1 = 1

    if press_type == 'Jeffreys':
        sech = 1/np.cosh(np.sqrt(3*c1/8)*(xs+0*t*(c1*eps/4)))
        tanh = np.tanh(np.sqrt(3*c1/8)*(xs+0*t*(c1*eps/4)))

        if explicit_result:
            # Explicitly calculated result: cleaner and works for t == 0
            firstOrderTerm = c1*sech**2
            secondOrderTerm = -P*c1**3*3/128/np.pi**2*t*sech**4 \
                    *(9-11*sech**2-6*np.sqrt(3*c1/8)*t*(1+c1*eps/2)*tanh)
        else:
            # This gives the same result, but it isn't as simplified;
            # also it requires t != 0

            A = c1*sech**2
            diffA = -2*tanh*sech**2*c1*np.sqrt(3*c1/8)
            diffDiffA = 2*sech**2*(2-3*sech**2)*c1*(3*c1/8)
            diffDiffDiffA = 8*sech**2*tanh*(3*sech**2-1)*c1*np.sqrt(3*c1/8)*(3*c1/8)

            firstOrderTerm = A
            secondOrderTerm = -P/64/np.pi**2*t \
                    *(
                            5*diffA**2
                            +4*A*diffDiffA
                            +3*t*(1+c1*eps/2)*(
                                2*diffA*diffDiffA
                                -A*diffDiffDiffA
                                )
                     )

    elif press_type == 'GM':
        sech = 1/np.cosh(np.sqrt(3/8)*(xs+0*t*(eps/4)))
        tanh = np.tanh(np.sqrt(3/8)*(xs+0*t*(eps/4)))
        sechPsiP = 1/np.cosh(np.sqrt(3/8)*(xs+0*t*(eps/4)+psi_P))
        tanhPsiP = np.tanh(np.sqrt(3/8)*(xs+0*t*(eps/4)+psi_P))

        if explicit_result:
            # Explicitly calculated result: cleaner and works for t == 0
            firstOrderTerm = sech**2
            secondOrderTerm = -P*c1**2/64/np.pi**2*np.sqrt(3*c1/8)*t \
                    *(-18*sechPsiP**4*tanhPsiP
                      +8*sechPsiP**2*sech**2*tanh
                      +10*sechPsiP**2*sech**2*tanhPsiP
                      +3*np.sqrt(3*c1/8)*t*(1+c1*eps/2)*(
                          4*sechPsiP**4*tanhPsiP**2
                          +2*sechPsiP**4*(2-3*sechPsiP**2)
                          -4*sechPsiP**2*tanhPsiP*sech**2*tanh
                          -2*sech**2*sechPsiP**2*(
                              2-3*sechPsiP**2)
                          )
                      )
        else:
            # This gives the same result, but it isn't as simplified (to check,
            # ensure t is non-zero)
                    A = c1*sech**2
                    ApsiP = c1*sechpsiP**2
                    diffA = -2*tanh*sech**2*c1*np.sqrt(3*c1/8)
                    diffApsiP = -2*tanhpsiP*sechpsiP**2*c1*np.sqrt(3*c1/8)
                    diffDiffA = 2*sech**2*(2-3*sech**2)*c1*(3*c1/8)
                    diffDiffApsiP = 2*sechpsiP**2*(2-3*sechpsiP**2)*c1*(3*c1/8)

                    firstOrderTerm = A
                    secondOrderTerm = -P/64/np.pi**2*t \
                            *(
                                    9*ApsiP*diffApsiP
                                    -4*ApsiP*diffA
                                    -5*A*diffApsiP
                                    +3*t*(1+c1*eps/2)*(
                                        diffApsiP**2
                                        +ApsiP*diffDiffApsiP
                                        -diffApsiP*diffA
                                        -diffDiffApsiP*A
                                        )
                             )
    else:
        raise(ValueError("forcing_type must be either 'Jeffreys' or 'GM', but "
            +press_type+" was given"))

    first = eps*firstOrderTerm
    second = eps**2*secondOrderTerm
    profile = first + second
    return profile

def plot_J_and_GM_profiles(xs, ts, Jeffreys, GM, P=1, psi_P=np.pi/2,
        depth=np.inf, eps=0.1):
    """ Generate plot showing Jeffreys and Generalized Miles profiles
    from supplied data.

    Parameters
    ----------
    xs : ndarray
        A one dimensional array of giving the x-position to at which the
        profile should be evaluated.
    ts : ndarray
        Times at which to the profiles are evaluated.
    Jeffreys : ndarray
        The data to be plotted. The first axis should be the values as a
        function of position, the second axis should be the values as a
        function of time. Must be the size (xs,ts).
    GM : ndarray
        The data to be plotted. The first axis should be the values as a
        function of position, the second axis should be the values as a
        function of time. Must be the size (xs,ts).
    P : float
        The magnitude of the nondimensional pressure forcing $P
        k/(rho_w*g)$. Default is 1.
    psi_P : float
        The wind phase of the pressure forcing. Default is np.pi/2.
    depth : float or np.inf
        The nondimensional depth $k*h$. Default is np.inf.
    eps : float
        Initial nondimensional wave height $a_0/h$. Default is 0.1.
    """

    data = {'Jeffreys' : Jeffreys,
            'GM' : GM}

    ## Color cycle
    num_lines = data['Jeffreys'].shape[1] # Number of lines
    new_colors = [plt.get_cmap('YlGnBu')(1. * (i+2)/(num_lines+2)) for i in
            reversed(range(num_lines))] # add 2 to numerator and denominator
                                        # so we don't go all the way to 0
                                        # (it's too light to see)
    linestyles = [*((0,(3+i,i)) for i in range(num_lines))]

    property_cycle = (cycler('color', new_colors) + cycler('linestyle',
        linestyles))

    # Initialize figure
    fig, ax = texplot.newfig(0.9,nrows=0,ncols=0)
    size = fig.get_size_inches()
    fig.set_size_inches([size[0],size[1]*1.2])
    fig.set_tight_layout(False)

    # Create subplots
    gs = gridspec.GridSpec(2, 1, left=0.14, right=0.88, top=0.82,
            bottom=0.14,hspace=0.3)
    ax = [None]*2
    ax[0] = fig.add_subplot(gs[0,0])
    ax[1] = fig.add_subplot(gs[1,0], sharex=ax[0])

    # Hide tick labels on upper plot (must manually do this when using
    # GridSpec)
    [label.set_visible(False) for label in ax[0].get_xticklabels()]

    # Create fake axis for shared axis label
    gs = gridspec.GridSpec(1, 1, left=0.12, right=0.85, top=0.82,
            bottom=0.14,hspace=0.3)
    fakeAx = fig.add_subplot(gs[0,0],frameon=False)

    # Hide tick labels
    fakeAx.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    # Label shared axes
    fakeAx.set_ylabel(r'Wave Amplitude $k \eta$')

    # Label horizontal axis
    ax[1].set_xlabel(r'Distance $x$')

    for indx,forcing in enumerate(data):
        # Set title
        ax[indx].set_title(
                forcing+' Profile'+
                (r' ($\psi_P = \SI{{{psiP}}}{{\\degree}}$)'.\
                        format(psiP=np.around(np.degrees(psi_P)))
                        if forcing == 'GM' else '')+
                ' vs Time (in co-moving frame):'+
                r' $\epsilon = {eps}$, $kh = {kh}$, $P = {P}$'.format(
                    eps = eps,
                    kh= [x if x!=sp.inf else r'\infty' for x in
                        np.around([depth],decimals=1)][0]
                    , P=P)
                )

        # Set property cycle
        ax[indx].set_prop_cycle(property_cycle)

        # Plot snapshots
        ax[indx].plot(xs,data[forcing])

        # Add line depicting wind direction
        ax[indx].annotate('Wind', xy=(-3./4*np.pi, 0.1),va='top',ha='center')
        ax[indx].annotate('', xy=(-0.55*np.pi,0.05), xytext=(-0.9*np.pi, 0.05),
                arrowprops=dict(arrowstyle='->'))

        # Convert numerical index to lower case letter (0->a, 1->b, etc)
        indxToAlpha = chr(indx+ord('a'))
        subplotLabel = indxToAlpha + ')'

        # Add plot labels
        ax[indx].annotate(subplotLabel, xy=(0.05,0.9),xycoords='axes fraction',
            va='top',ha='center')

    fig.legend(np.around(ts,decimals=2),
            title=r'\begin{tabular}{c} Time $t / k \sqrt{gh} $\end{tabular}',
            loc='right',
            bbox_to_anchor=(1.0,0.5))

    return fig

def calc_and_plot(wave_type, save_prefix, m=0.8):
    """Generate and plot a wave profile under the action of Jeffreys and
    Generalized Miles (GM) pressure forcings.

    Parameters
    ----------
    wave_type : 'cnoidal' or 'solitary'
        Describes the type of wave to calculate and plot.
    m : float
        The elliptic parameter in the range [0,1] inclusive. Used for
        wave_type 'cnoidal' waves. Default is 0.8.
    """

    ## Physical parameters
    eps = 0.1 # dimensionless nonlinearity parameter Ak
    P = 0.5 # dimensionless magnitude of wind forcing
    psi_P = np.radians(135) # Default wind phase in radians
    depth = np.inf # Default water depth

    ## Code parameters
    profileDepth = 1.0 # kh for fin_depth profiles
    period = 4*np.pi # period of cnoidal wave

    ## Define independent variables
    PRange = np.array([0,0.1,1], dtype=sp.float128)
    ts = np.array([0,10,20], dtype=sp.float128)

    data = {}
    for forcing in ['Jeffreys', 'GM']:
        # Generate profile
        if wave_type == 'solitary':
            # Define x-coordinates
            xs = np.arange(-5,5,0.01)

            profile = [solitary_profile(forcing, xs, P=P, psi_P=psi_P,
                depth=profileDepth, eps=eps, t=time) for time in ts]
        elif wave_type == 'cnoidal':
            # Define x-coordinates
            xs = np.linspace(-period/2, period/2, 100)

            profile = [cnoidal_profile(forcing, xs, P=P, psi_P=psi_P,
                depth=profileDepth, eps=eps, t=time) for time in ts]
        else:
            raise(ValueError(
            "wave_type must be either 'cnoidal' or 'solitary', but "
            +wave_type+" was given"))

        profile = np.transpose(np.array(profile))
        data[forcing] = profile

        # Calculate skewness and asymmetry numerically
        # Note: values differs slightly since analytic expression
        # truncates to order(epsilon), but numerical expression doesn't
        # (ie, it retains *some* higher order terms
        Skew = (np.trapz(profile**3,axis=0,dx=360/xs.size)/360)/\
                (np.trapz(profile**2,axis=0,dx=360/xs.size)/360)**(1.5)
        print('Numerical Skewness '+forcing+': ' + str(Skew))
        # Calculate Hilbert transform
        profile_Hilbert = np.imag(sp.signal.hilbert(profile,axis=0))
        Asym = (np.trapz(profile_Hilbert**3,axis=0,dx=360/xs.size)/360)/\
                (np.trapz(profile_Hilbert**2,axis=0,dx=360/xs.size)/360)**(1.5)
        print('Numerical Asymmetry '+forcing+': ' + str(Asym))

    fig = plot_J_and_GM_profiles(xs, ts, data['Jeffreys'], data['GM'], P=P,
            psi_P=psi_P, eps=eps, depth=depth)

    # Save plot
    texplot.savefig(fig,save_prefix+wave_type.capitalize()+'-Profile')

def main():
    save_prefix = '../Reports/Figures/'

    for wave_type in ['solitary','cnoidal']:
        print('Shape Parameters for '+wave_type+' wave:')
        calc_and_plot(wave_type, save_prefix)

if __name__ == '__main__':
    main()
