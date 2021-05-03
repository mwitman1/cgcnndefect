#! /usr/bin/env python

import numpy as np
import os, sys
import argparse
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import cycler
prop_cycle = plt.rcParams['axes.prop_cycle']
DEFAULT_COLORS = prop_cycle.by_key()['color']
plt.rc('text', usetex=True)
plt.rc('font', **{'size':9})
DEFCOLS = prop_cycle.by_key()['color']

def plot(files, labels=None):

    datalist = [pd.read_csv(f,names=['struct','actual','predicted']) for f in files]

    # in case we want to plot them separately
    # https://matplotlib.org/3.1.1/gallery/subplots_axes_and_figures/gridspec_and_subplots.html
    fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(3.3,2.0),constrained_layout=True)


    # Figure 1: Parity Plots
    min_act_energy = np.min([d['actual'] for d in datalist])
    max_act_energy = np.max([d['actual'] for d in datalist])

    xdata = []
    ydata = []
    colors = []
    
    for i, df in enumerate(datalist):
        xdata += list(df['actual'])
        ydata += list(df['predicted'])
        colors += [DEFCOLS[i] for _ in range(len(df['actual']))]

    xdata = np.array(xdata)
    ydata = np.array(ydata)
    colors = np.array(colors)

    p = np.random.permutation(len(xdata))

    # all data
    ax[0].scatter(xdata[p], ydata[p], color=colors[p],
                  edgecolor='black',linewidth=0.3,s=15)
    # dummy data for legend
    if labels is not None:
        for i, lab in enumerate(labels):
            ax[0].scatter([],[],color=DEFCOLS[i],label=lab,
                          edgecolor='black',linewidth=0.3,s=15)

    # parity line
    print(min_act_energy)
    print(max_act_energy)
    ax[0].plot([min_act_energy,max_act_energy],[min_act_energy,max_act_energy],c='black') 

    ax[0].legend(loc="lower left",borderpad=0.15,handletextpad=0.25,bbox_to_anchor=(0,1.02),
                 columnspacing=0.15, ncol=2)
    ax[0].set_xlabel(r'$E_{\textrm{DFT}}\textrm{ [eV]}$')
    ax[0].set_ylabel(r'$E_{\textrm{CGCNN}}\textrm{ [eV]}$')



    # Figure 2: histograms of errors
    errors = [df['actual']-df['predicted'] for df in datalist]
    minerror = np.min(errors)
    maxerror = np.max(errors)

    for i, error in enumerate(errors):
        MAE = np.mean(np.abs(error))
        hist, bin_edges = np.histogram(error, bins=np.linspace(minerror,maxerror,40))
        print(hist.shape,bin_edges.shape)
        print(hist,bin_edges)
        print(MAE)
        print(np.ones_like(hist))
        print(matplotlib.colors.to_rgb(DEFCOLS[i]))
        # https://stackoverflow.com/questions/28398200/matplotlib-plotting-transparent-histogram-with-non-transparent-edge
        ax[1].hist(bin_edges[:-1],bins=bin_edges,weights=hist,
                   lw=.5, fc = matplotlib.colors.to_rgb(DEFCOLS[i])+(0.5,),
                   edgecolor = matplotlib.colors.to_rgb(DEFCOLS[i]),#+(0.5,),
                   label=r'$\textrm{MAE=%.3f}$'%MAE)

    ax[1].legend(loc="lower left",borderpad=0.15,handletextpad=0.25,bbox_to_anchor=(0,1.02),
                 columnspacing=0.15, ncol=2)

    #ax[1].set_yscale('log')
    ax[1].set_ylabel(r'$\textrm{frequency}$')
    ax[1].set_xlabel(r'$|\textrm{error}|$')


    plt.show()

    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-files', nargs='*', help='list of all parity plot data files')
    parser.add_argument('-labels', nargs='*', help='label for each plot')

    args = parser.parse_args()

    
    plot(**vars(args))

    
