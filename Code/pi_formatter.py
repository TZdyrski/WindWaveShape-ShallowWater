#!/usr/bin/env python3
# pi_formatter.py
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
import fractions
import re
import texplot

## Return tick labels in units of pi*multiple
def _pi_label_format(x,pos=0):
    # Convert ticklabel to fractions
    ticklabel =fractions.Fraction(x/sp.pi).limit_denominator()
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

## Return formatter for tick labels in units of pi*multiple
## Note: pi_formatter rounds to the nearest rational multiple of pi (with
## maximum denominator given by limit_denominator's max_denominator
## default); therefore, for most accurate labels, ensure ticks are
## placed at rational multiples of pi using the _pi_locator locator
def pi_formatter():

    formatter = mpl.ticker.FuncFormatter(_pi_label_format)

    return formatter

## Return locator giving ticks in units of pi*multiple
def _pi_locator(multiple):
    # Define unit as pi times multiple
    unit = sp.pi*multiple

    # Round to nearest fraction
    unit = fractions.Fraction(unit/sp.pi).limit_denominator()*sp.pi

    # Create locators
    locator = mpl.ticker.MultipleLocator(unit)

    return locator

## Set tick mark locations as multiples of $\pi$
def pi_multiple_ticks(ax,whichAxis,multiple,minorMultiple=0):

    majorLocator = _pi_locator(multiple)

    if minorMultiple != 0: minorLocator = _pi_locator(minorMultiple)

    if whichAxis == 'x':
        # Draw ticks
        ax.xaxis.set_major_locator(majorLocator)
        if minorMultiple != 0: ax.xaxis.set_minor_locator(minorLocator)

        # Label major tick
        ax.xaxis.set_major_formatter(pi_formatter())

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
        ax.yaxis.set_major_formatter(pi_formatter())

        # Vertically center tick labels since PGF has an issue with
        # vertical alignment)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_verticalalignment('center')

        return
    else:
        raise ValueError("Option 'whichAxis' passed to pi_multiple_ticks must be one of {'x','y'}")
