import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3d plotting
import matplotlib as mpl
from cycler import cycler

####################
# plotting lines and color
####################
mpl.rcParams['lines.linewidth'] = 1.8
mpl.rcParams['lines.markersize'] = 9
mpl.rcParams['axes.prop_cycle'] = cycler('color',
                                         ['#004BA8', '#0072B2', '#56B4E9',
                                          '#7A68A6', '#990000', '#CC79A7',
                                          '#009E73', '#EDD3BF', '#5EE5A2',
                                          '#E76333'])

####################
# latex
####################
mpl.rcParams['mathtext.fontset'] = 'cm'

####################
# title
####################
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['axes.titlepad'] = 15
mpl.rcParams['text.color'] = '#2c2c2c'

####################
# axes labels
####################
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.labelcolor'] = '#EDD3BF'

####################
# axes ticks
####################
mpl.rcParams['xtick.color'] = '#EDD3BF'
mpl.rcParams['ytick.color'] = '#EDD3BF'
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['xtick.major.pad'] = 3.5
mpl.rcParams['ytick.major.pad'] = 3.5
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'


####################
# dimensions
####################
# axes.xmargin        : .5  # x margin.  See `axes.Axes.margins`
# axes.ymargin        : .5  # y margin See `axes.Axes.margins`
mpl.rcParams['figure.subplot.left'] = 0.1
mpl.rcParams['figure.subplot.right'] = 0.95
mpl.rcParams['figure.subplot.bottom'] = 0.15
mpl.rcParams['figure.subplot.top'] = 0.9
mpl.rcParams['figure.subplot.wspace'] = 0.3

####################
# grid
####################
mpl.rcParams['text.hinting_factor'] = 8
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.color'] = 'b2b2b2'         # grid color
mpl.rcParams['grid.linestyle'] = '--'            # solid
mpl.rcParams['grid.linewidth'] = 0.8             # in points
mpl.rcParams['grid.alpha'] = 0.3

####################
# color
####################
# graph color
mpl.rcParams['axes.facecolor'] = "#EDD3BF"
mpl.rcParams['axes.edgecolor'] = "#EDD3BF"

# background color around the graphs
mpl.rcParams['figure.facecolor'] = "#323232"
mpl.rcParams['figure.edgecolor'] = "#8080ff"

####################
# saving
####################
# figure dots per inch or 'figure'
mpl.rcParams['savefig.dpi'] = 'figure'
# figure facecolor when saving
mpl.rcParams['savefig.facecolor'] = 'white'
# figure edgecolor when saving
mpl.rcParams['savefig.edgecolor'] = (0.29, 0.33, 0.38)
# png, ps, pdf, svg
mpl.rcParams['savefig.format'] = 'pdf'

####################
# figure size
####################
mpl.rcParams['figure.figsize'] = 4, 3  # figure size in inches
mpl.rcParams['figure.dpi'] = 300       # figure dots per inch
mpl.rcParams['figure.frameon'] = True   # enable figure frame

print("⦿⦿⦿⦿⦿⦿⦿⦿⦿⦿⦿⦿ Setup Ilya's matplotlib style ⦿⦿⦿⦿⦿⦿⦿⦿⦿⦿⦿⦿")
