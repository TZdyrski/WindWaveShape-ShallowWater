#!/usr/bin/env python
## Source: https://github.com/nilsleiffischer/texfig
# texplot.py
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d.axes3d import Axes3D

#useBackend = 'pgf'
useBackend = 'mplcairoEps'

if useBackend == 'pgf':
    mpl.use('pgf')
elif useBackend == 'mplcairoEps':
    # Use mplcairoEps backend to correct eps text alignment and font
    # specification
    # Must be installed from (https://github.com/matplotlib/mplcairoEps)
    mpl.use("module://mplcairo.base")


    # Patch xpdf_distill to fix incorrect parameter
    def new_xpdf_distill(tmpfile, eps=False, ptype='letter', bbox=None, rotated=False):
        import glob
        import os
        import shutil
        import logging
        _log = logging.getLogger(__name__)

        # For some reason, mplcairoEps passes False to eps parameter, even when
        # is_eps = True; since I know I am only using eps files, set
        # eps=True
        eps = True
        """
        Use ghostscript's ps2pdf and xpdf's/poppler's pdftops to distill a file.
        This yields smaller files without illegal encapsulated postscript
        operators. This distiller is preferred, generating high-level postscript
        output that treats text as text.
        """
        pdffile = str(tmpfile) + '.pdf'
        psfile = str(tmpfile) + '.ps'

        # Pass options as `-foo#bar` instead of `-foo=bar` to keep Windows happy
        # (https://www.ghostscript.com/doc/9.22/Use.htm#MS_Windows).
        mpl.cbook._check_and_log_subprocess(
            ["ps2pdf",
             "-dAutoFilterColorImages#false",
             "-dAutoFilterGrayImages#false",
             "-dAutoRotatePages#/None", # This is the patch
             "-sGrayImageFilter#FlateEncode",
             "-sColorImageFilter#FlateEncode",
             "-dEPSCrop" if eps else "-sPAPERSIZE#%s" % ptype,
             tmpfile, pdffile], _log)
        mpl.cbook._check_and_log_subprocess(
            ["pdftops", "-paper", "match", "-level2", pdffile, psfile], _log)

        os.remove(tmpfile)
        shutil.move(psfile, tmpfile)

        if eps:
            mpl.backends.backend_ps.pstoeps(str(tmpfile))

        for fname in glob.glob(str(tmpfile)+'.*'):
            os.remove(fname)


    # Patch xpdf_distill to fix incorrect parameter
    mpl.backends.backend_ps.xpdf_distill = new_xpdf_distill

else:
    raise('Variable `useBackend` in texplot must be one of {pgf,mplcairoEps}')

# Return figure size [width,height] for a given scale
def figsize(figScale=1, golden=True, orientation='landscape', columnWidth=False):
    text_width_pt = 384.0                        # Get this from LaTeX
                                                 #   using \the\textwidth
                                                 #   or \the\contentwidth
                                                 #   (beamer)
    text_height_pt = 211.3                       # Get this from LaTeX
                                                 #   using \the\textheight
                                                 #   or \the\contentheight
                                                 #   (beamer)

    num_col = 2                                  # The number of columns
                                                 #   for calculating
                                                 #   column_width_pt
    column_sep_pt = 10                           # Get this from LaTex
                                                 #   using \the\columnsep
    column_width_pt = text_width_pt/num_col - column_sep_pt

    inches_per_pt = 1.0/72.27                    # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0         # Aesthetic ratio (you
                                                 #   could change this)

    if columnWidth:
        fig_width = column_width_pt*inches_per_pt*figScale # width in inches
    else:
        fig_width = text_width_pt*inches_per_pt*figScale # width in inches

    if golden:
        if orientation == 'landscape':
            fig_height = fig_width*golden_mean
        elif orientation == 'portrait':
            fig_height = fig_width/golden_mean
        else:
            raise ValueError('Option `orientation` must be either `landscape` (default) or `portrait`')
    else:
        fig_height = text_height_pt*inches_per_pt*figScale   # height in inches

    fig_size = [fig_width,fig_height]

    return fig_size

# Set matplotlibrc params
pgf_with_latex = {                      # Setup matplotlib to use latex
                                        #   for output
    "text.usetex": True,                # Use LaTeX to write all text
    "font.family": "serif",
    "font.size": 10,                    # Text size must be specified
                                        #   (for spacing)
    "figure.figsize": figsize(0.9),     # Default fig size of 0.9 textwidth
# Either 1) enable autolayout
    "figure.autolayout": True,          # Allow matplotlib to re-arrange
                                        #   objects
# Or 2) adjust subplot layout
#    "figure.subplot.left": 0.1,        # Make left and right margins
#    "figure.subplot.right": 0.9,       #   symmetric
    "text.latex.preamble":
#        r"\usepackage[utf8x]{inputenc}", # Use utf8 fonts
        r"\usepackage[T1]{fontenc}"      # Plots will be generated using
                                         #   this preamble
        +r"\usepackage{siunitx}"         # Typsetting units
        +r'\sisetup{detect-all}'         # Force siunitx to actually use
                                         #   your fonts
        +r"\usepackage{physics}"         # Use physics package for symbols
        +r'\usepackage{amsmath}'         # Use amsmath to declare new math symbols
        +r'\newcommand{\im}{\mathrm{i}}' # Define \im as Roman i
        +r'\renewcommand*{\epsilon}{\varepsilon}' # Replace epsilon with varepsilon
        +r'\DeclareMathOperator{\Sk}{Sk}' # Define skewness
        +r'\DeclareMathOperator{\As}{As}' # Define asymmetry
    ,
    }

if useBackend == 'pgf':
    pgf_with_latex.update({
        "pgf.texsystem": "pdflatex",         # Change this if using xetex or
                                             #   lautex
        "pgf.rcfonts": False,                # Disable automatic choosing of
                                             #   fonts
        "font.serif": [],                    # Blank entries should cause
        "font.sans-serif": [],               #   plots to inherit fonts from
        "font.monospace": [],                #   the document
        "pgf.preamble":
            r"\usepackage[utf8x]{inputenc}" # Use utf8 input encoding
            +r"\usepackage[T1]{fontenc}"     # Use Type1 fonts
            +r"\usepackage{siunitx}"         # Typsetting units
            +r'\sisetup{detect-all}'         # Force siunitx to actually use
                                             #  your fonts
            +r'\usepackage{physics}'         # Use physics package for symbols
            +r'\usepackage{amsmath}'         # Use amsmath to declare new math symbols
            +r'\newcommand{\im}{\mathrm{i}}' # Define \im as Roman i
            +r'\renewcommand*{\epsilon}{\varepsilon}' # Replace epsilon with varepsilon
            +r'\DeclareMathOperator{\Sk}{Sk}' # Define skewness
            +r'\DeclareMathOperator{\As}{As}' # Define asymmetry
        ,
    })
elif useBackend == 'mplcairoEps':
    pgf_with_latex.update({
        "font.serif": 'cmr10',              # Must set font for eps since it's not being typset in latex document
        "ps.usedistiller": 'xpdf',          # Don't rasterize text
        "legend.framealpha": 1.0,           # Remove transparency so ps/eps
                                            #   doesn't rasterize
    })

mpl.rcParams.update(pgf_with_latex)
import matplotlib.pyplot as plt

# Create a new figure
def newfig(figScale, pad=0.3, orientation=None, columnWidth=None,
        golden=None, *args, **kwargs):
    plt.close()
    figsize_args = {'figScale' : figScale, 'orientation' : orientation,
            'columnWidth' : columnWidth, 'golden' : golden}
    fig, ax = plt.subplots(figsize=figsize(
        **{k : v for k,v in figsize_args.items() if v != None}
        ), *args,**kwargs)
    # Set padding
    fig.set_tight_layout({'pad':pad})
    return fig, ax

# Create a new 3D figure
def new3dfig(pad=0.3, figScale=None, orientation=None, columnWidth=None,
        golden=None, *args, **kwargs):
    plt.close()
    figsize_args = {'figScale' : figscale, 'orientation' : orientation,
            'columnWidth' : columnWidth, 'golden' : golden}
    fig, ax = plt.subplots(figsize=figsize(
        **{k : v for k,v in figsize_args.items() if v != None}
        ), subplot_kw={'projection': '3d'}, *args, **kwargs)
    # Set padding
    fig.set_tight_layout({'pad':pad})
    return fig, ax

# Save a pdf and pgf version of the figure
def savefig(fig,filename,*args,**kwargs):
    if useBackend == 'pgf':
        fig.savefig('{}.pgf'.format(filename),facecolor=fig.get_facecolor(),*args,**kwargs)
    elif useBackend == 'mplcairoEps':
        fig.savefig('{}.eps'.format(filename),facecolor=fig.get_facecolor(),*args,**kwargs)
    fig.savefig('{}.pdf'.format(filename),facecolor=fig.get_facecolor(),*args,**kwargs)
