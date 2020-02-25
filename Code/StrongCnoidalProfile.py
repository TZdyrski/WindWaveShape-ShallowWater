#!/usr/bin/env python3
# DataAnalysis.py
import numpy as np
import scipy as sp
import scipy.io
import scipy.signal
import scipy.special as spec
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.ticker import StrMethodFormatter, MultipleLocator
from matplotlib import cm as cm
from matplotlib.legend_handler import HandlerBase
import matplotlib.gridspec as gridspec
import texplot
import fractions
import re

### Debug options
plot_profile = False

### Set global parameters
## Physical parameters
eps = 0.1 # dimensionless nonlinearity parameter Ak
P = 0.3 # dimensionless magnitude of wind forcing
psi_P = np.radians(135) # Default wind phase in radians
depth = np.inf # Default water depth

all_profiles_depth = 2 # Depth for plot_all_profiles

## Code parameters
profileDepth = 1.0 # kh for fin_depth profiles
m = 0.8 # Jacobi elliptic parameter
period = 2*np.pi

## Define independent variables
xs = np.linspace(-period/2,period/2,100)
PRange = np.array([0,0.1,1], dtype=sp.float128)
profile_times = np.array([0,10,20], dtype=sp.float128)

## Color cycle
num_lines = profile_times.size # Number of lines
new_colors = [plt.get_cmap('YlGnBu')(1. * (i+2)/(num_lines+2)) for i in
        reversed(range(num_lines))] # add 2 to numerator and denominator
                                    # so we don't go all the way to 0
                                    # (it's too light to see)
linestyles = [*((0,(3+i,i)) for i in range(num_lines))]
plt.rc('axes', prop_cycle=(cycler('color', new_colors) +
                           cycler('linestyle', linestyles)))

## Generate wave profile
# Note: we are following the writeup derivation; we convert from complex
# to real by adding the complex conjugate (or doubling the real part)

def FullProfile(press_type, phase, P=1, psi_P=np.pi/2, depth=np.inf,
        eps=0.1, A_mag=1, t=0, m=1):

    mP = np.sqrt(1-m**2)
    E = spec.ellipe(m)
    K = spec.ellipk(m)
    c1 = 8/3/np.pi**2*m**2*K**2 # Amplitude of wave; set by requiring the period to be 2*pi

    sn = spec.ellipj(np.sqrt(3*c1/8)*(phase+0*t*(c1*eps/4))/m,m)[0]
    cn = spec.ellipj(np.sqrt(3*c1/8)*(phase+0*t*(c1*eps/4))/m,m)[1]
    dn = spec.ellipj(np.sqrt(3*c1/8)*(phase+0*t*(c1*eps/4))/m,m)[2]

    if press_type == "jeffreys":
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

    elif press_type == "generalized":
        snpsiP = spec.ellipj(np.sqrt(3/8)*(phase+0*t*(c1*eps/4)+psi_P)/m,m)[0]
        cnpsiP = spec.ellipj(np.sqrt(3/8)*(phase+0*t*(c1*eps/4)+psi_P)/m,m)[1]
        dnpsiP = spec.ellipj(np.sqrt(3/8)*(phase+0*t*(c1*eps/4)+psi_P)/m,m)[2]

        A = c1*(cn**2 - (E/K - mP**2)/m**2)
        ApsiP = cnpsiP**2 - (E/K - mP**2)/m**2
        diffA = -2*cn*sn*dn*c1*np.sqrt(3*c1/8)
        diffApsiP = -2*cnpsiP*snpsiP*dnpsiP*c1*np.sqrt(3*c1/8)
        diffDiffA = -2*(dn**2*cn**2-m*cn**2*sn**2-sn**2*dn**2)*c1*(3*c1/8)
        diffDiffApsiP = -2*(dnpsiP**2*cnpsiP**2-m*cnpsiP**2*snpsiP**2-snpsiP**2*dnpsiP**2)*c1*(3*c1/8)
        diffDiffDiffA = 8*sn*cn*dn*(dn**2+m*cn**2-m*sn**2)*c1*np.sqrt(3*c1/8)*(3*c1/8)
        diffDiffDiffApsiP = 8*snpsiP*cnpsiP*dnpsiP*(dnpsiP**2+m*cnpsiP**2-m*snpsiP**2)*c1*np.sqrt(3*c1/8)*(3*c1/8)

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

    first = eps*firstOrderTerm
    second = eps**2*secondOrderTerm
    profile = first + second
    profile = np.real(profile)
    return profile

## Return tick labels in units of pi*multiple
def PiLabelFormat(x,pos=0):
    # Convert ticklabel to fractions
    ticklabel =fractions.Fraction(x/np.pi).limit_denominator()
    # Convert integer multiples of pi numbers back to ints
    if ticklabel.denominator == 1: ticklabel = ticklabel.numerator
    # Format fractions using latex
    if type(ticklabel) == fractions.Fraction:
        ticklabel = r'$' + str(r'-' if ticklabel <0 else '') + \
        r'\tfrac{'+ str(abs(ticklabel.numerator) if
                abs(ticklabel.numerator) != 1 else '') + r' \pi}{' + \
        str(ticklabel.denominator) + r'}$'
    elif ticklabel == 1:
        ticklabel = r'$\pi$'
    elif ticklabel == -1:
        ticklabel = r'$-\pi$'
    elif type(ticklabel) == int and ticklabel != 0:
        ticklabel = r'$' + str(ticklabel) + r' \pi $'
    return ticklabel

# Default wind phase in radians (inline and without '$')
psi_P_text = re.sub(r'\\tfrac\{(.*?)\}\{(.*?)\}',r'\1/\2',
        PiLabelFormat(psi_P).replace('$',''))

## Return formatter for tick labels in units of pi*multiple
## Note: PiFormatter rounds to the nearest rational multiple of pi (with
## maximum denominator given by limit_denominator's max_denominator
## default); therefore, for most accurate labels, ensure ticks are
## placed at rational multiples of pi using the PiLocator locator
def PiFormatter():

    formatter = mpl.ticker.FuncFormatter(PiLabelFormat)

    return formatter

## Return locator giving ticks in units of pi*multiple
def PiLocator(multiple):
    # Define unit as pi times multiple
    unit = np.pi*multiple

    # Round to nearest fraction
    unit = fractions.Fraction(unit/np.pi).limit_denominator()*np.pi

    # Create locators
    locator = mpl.ticker.MultipleLocator(unit)

    return locator

## Set tick mark locations as multiples of $\pi$
def PiMultipleTicks(ax,whichAxis,multiple,minorMultiple=0):

    majorLocator = PiLocator(multiple)

    if minorMultiple != 0: minorLocator = PiLocator(minorMultiple)

    if whichAxis == 'x':
        # Draw ticks
        ax.xaxis.set_major_locator(majorLocator)
        if minorMultiple != 0: ax.xaxis.set_minor_locator(minorLocator)

        # Label major tick
        ax.xaxis.set_major_formatter(PiFormatter())

        # The PGF backend has a problem aligning tick labels; manually
        # align by baseline and pad
        for lab in ax.xaxis.get_ticklabels():
            lab.set_verticalalignment('baseline')

        # Pad tick labels
        if len(ax.xaxis.get_ticklabels()) != 0:
            fontsize = ax.xaxis.get_ticklabels()[0].get_size()
            ax.tick_params(axis='x', pad=fontsize+.8)

        return
    elif whichAxis == 'y':
        # Draw ticks
        ax.yaxis.set_major_locator(majorLocator)
        if minorMultiple != 0: ax.yaxis.set_minor_locator(minorLocator)

        # Label major tick
        ax.yaxis.set_major_formatter(PiFormatter())

        # Vertically center tick labels since PGF has an issue with
        # vertical alignment)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_verticalalignment('center')

        return
    else:
        raise ValueError("Option 'whichAxis' passed to PiMultipleTicks must be one of {'x','y'}")

## Plot Profile
if(plot_profile):

    JPress = np.transpose(np.array(
        [FullProfile('jeffreys', xs, P=P, depth=profileDepth,
            eps=eps, t=time, m=m) for time in profile_times]))

    # Calculate skewness numerically (note: value differs slightly since
    # analytic expression truncates to order(epsilon), but numerical
    # expression doesn't (ie, it retains *some* higher order terms)
    SkJ = (np.trapz(JPress**3,axis=0,dx=360/xs.size)/360)/\
            (np.trapz(JPress**2,axis=0,dx=360/xs.size)/360)**(1.5)
    print("Numerical Skewness Jeffreys: " + str(SkJ))
    # Calculate asymmetry numerically (note: value differs slightly since
    # analytic expression truncates to order(epsilon), but numerical
    # expression doesn't (ie, it retains *some* higher order terms)
    JPressH = np.imag(sp.signal.hilbert(JPress,axis=0))
    AsJ = (np.trapz(JPressH**3,axis=0,dx=360/xs.size)/360)/\
            (np.trapz(JPressH**2,axis=0,dx=360/xs.size)/360)**(1.5)
    print("Numerical Asymmetry Generalized: " + str(AsJ))

    GPress = np.transpose(np.array(
        [FullProfile('generalized', xs, P=P, depth=depth, eps=eps,
            psi_P = psi_P, t=time, m=m) for time in profile_times]))

    # Calculate skewness numerically (note: value differs slightly since
    # analytic expression truncates to order(epsilon), but numerical
    # expression doesn't (ie, it retains *some* higher order terms)
    SkG = (np.trapz(GPress**3,axis=0,dx=360/xs.size)/360)/\
            (np.trapz(GPress**2,axis=0,dx=360/xs.size)/360)**(1.5)
    print("Numerical Skewness Generalized: " + str(SkG))
    # Calculate asymmetry numerically (note: value differs slightly since
    # analytic expression truncates to order(epsilon), but numerical
    # expression doesn't (ie, it retains *some* higher order terms)
    GPressH = np.imag(sp.signal.hilbert(GPress,axis=0))
    AsG = (np.trapz(GPressH**3,axis=0,dx=360/xs.size)/360)/\
            (np.trapz(GPressH**2,axis=0,dx=360/xs.size)/360)**(1.5)
    print("Numerical Asymmetry Generalized: " + str(AsG))

    # Initialize figure
    fig, ax = texplot.newfig(0.9,nrows=0,ncols=0)
    size = fig.get_size_inches()
    fig.set_size_inches([size[0],size[1]*1.2])
    fig.set_tight_layout(False)
    gs = gridspec.GridSpec(2, 1, left=0.14, right=0.88, top=0.82,
            bottom=0.14,hspace=0.3)
    ax = [None]*2

    ax[0] = fig.add_subplot(gs[0,0])
    ax[1] = fig.add_subplot(gs[1,0], sharex=ax[0])

    # Hide tick labels on upper plot (must manually do this when using
    # GridSpec)
    [label.set_visible(False) for label in ax[0].get_xticklabels()]

    ax[0].set_title(r'Jeffreys Profile vs Time (in co-moving frame): $\epsilon = {eps}$, $P = {PJ}$'.format(eps = eps, PJ=P))
    ax[1].set_title(r'Generalized ($\psi_P = \SI{{{psiP}}}{{\degree}}$) Profile vs Time (in co-moving frame): $\epsilon = {eps}$, $P = {PJ}$'.format(eps = eps, PJ=P, psiP=np.around(np.degrees(psi_P))))

    # Create fake axes around all subplots for axis labels
    gs = gridspec.GridSpec(1, 1, left=0.12, right=0.85, top=0.82,
            bottom=0.14,hspace=0.3)
    fakeAx = fig.add_subplot(gs[0,0],frameon=False)
    fakeAx.set_ylabel(r'Wave Amplitude $k \eta$')
    # Hide tick labels (must manually do this when using GridSpec)
    fakeAx.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    ax[1].set_xlabel(r'Distance $x$')

    linesPress = ax[0].plot(xs,JPress)
    linesDepth = ax[1].plot(xs,GPress)

    fig.legend(np.around(profile_times,decimals=2),
            title=r'\begin{tabular}{c} Time $t / k \sqrt{gh} $\end{tabular}',
            loc='right',
            bbox_to_anchor=(1.0,0.5))

    # Add line depicting wind direction
    ax[0].annotate('Wind', xy=(-3./4*np.pi, 0.1),va='top',ha='center')
    ax[0].annotate('', xy=(-0.55*np.pi,0.05), xytext=(-0.9*np.pi, 0.05), arrowprops=dict(arrowstyle='->'))

    # Add line depicting wind direction
    ax[1].annotate('Wind', xy=(-3./4*np.pi, 0.1),va='top',ha='center')
    ax[1].annotate('', xy=(-0.55*np.pi,0.05), xytext=(-0.9*np.pi, 0.05), arrowprops=dict(arrowstyle='->'))

    # Add plot labels
    ax[0].annotate('a)', xy=(0.05,0.9),xycoords='axes fraction',
            va='top',ha='center')
    ax[1].annotate('b)', xy=(0.05,0.9),xycoords='axes fraction',
            va='top',ha='center')

    texplot.savefig(fig,'../Figures/Cnoidal-Profile')
