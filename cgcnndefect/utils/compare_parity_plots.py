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
import plotly.express as px
import plotly.graph_objects as go

prop_cycle = plt.rcParams['axes.prop_cycle']
DEFAULT_COLORS = prop_cycle.by_key()['color']
#plt.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
# plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# plt.rc('text', usetex=True)
# plt.rc('font', **{'size':8})
DEFCOLS = prop_cycle.by_key()['color']
# https://stackoverflow.com/questions/2537868/sans-serif-math-with-latex-in-matplotlib
# https://tex.stackexchange.com/questions/314190/upright-sans-serif-greek-in-math-mode
plt.rcParams['text.latex.preamble'] = [
       r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
       r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
       r'\usepackage{helvet}',    # set the normal font here
       r'\usepackage{sansmathfonts}',  # load up the sansmath so that math -> helvet
       #r'\sansmath',              # <- tricky! -- gotta actually tell tex to use!
]
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

def plot(files, labels=None, predtype=('E','eV')):

    datalist = [pd.read_csv(f,names=['struct','actual','predicted']) for f in files]

    # in case we want to plot them separately
    # https://matplotlib.org/3.1.1/gallery/subplots_axes_and_figures/gridspec_and_subplots.html
    fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(4,1.6),constrained_layout=True)

    print(datalist)

    xdata = []
    ydata = []
    colors = []
    structs = []
    batch = []
    ns = []
    
    for i, df in enumerate(datalist):
        xdata += list(df['actual'])
        ydata += list(df['predicted'])
        colors += [DEFCOLS[i] for _ in range(len(df['actual']))]
        structs += list(df['struct'])
        batch += [i for _ in range(len(df['actual']))]
        ns += [struct.split('-')[2][1:] for struct in list(df['struct'])]

    xdata = np.array(xdata,dtype=float)
    ydata = np.array(ydata,dtype=float)
    colors = np.array(colors)

    df_all = pd.DataFrame()
    df_all['struct'] = structs
    df_all['ns'] = ns
    df_all['actual'] = xdata
    df_all['predicted'] = ydata
    df_all['batch'] = batch
    df_all['error'] = ydata-xdata

    min_act_energy = np.min(df_all['actual'])
    max_act_energy = np.max(df_all['actual'])
    df_yeqx = pd.DataFrame()
    df_yeqx['x']=[min_act_energy,max_act_energy]
    df_yeqx['y']=[min_act_energy,max_act_energy]
    df_yeqx['color']=['black','black']

    df_O = df_all[df_all['struct'].str.contains('O')]

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
    print('min', min_act_energy)
    print('max', max_act_energy)
    ax[0].plot([min_act_energy,max_act_energy],[min_act_energy,max_act_energy],c='black') 

    ax[0].set_xlabel(r'$%s_{\mathrm{DFT}}$ [%s]'%(predtype[0],predtype[1]))
    ax[0].set_ylabel(r'$%s_{\mathrm{Model}}$ [%s]'%(predtype[0],predtype[1]))



    # Figure 2: histograms of errors
    errors = [df['actual']-df['predicted'] for df in datalist]
    minerror = np.min(df_all['error'])
    maxerror = np.max(df_all['error'])

    for i, error in enumerate(errors):
        MAE = np.mean(np.abs(error))
        hist, bin_edges = np.histogram(error, bins=np.linspace(minerror,maxerror,40))
        #print(hist.shape,bin_edges.shape)
        #print(hist,bin_edges)
        print(i, MAE)
        #print(np.ones_like(hist))
        #print(matplotlib.colors.to_rgb(DEFCOLS[i]))
        # https://stackoverflow.com/questions/28398200/matplotlib-plotting-transparent-histogram-with-non-transparent-edge
        ax[1].hist(bin_edges[:-1],bins=bin_edges,weights=hist,
                   lw=.5, fc = matplotlib.colors.to_rgb(DEFCOLS[i])+(0.5,),
                   edgecolor = matplotlib.colors.to_rgb(DEFCOLS[i]),#+(0.5,),
                   label=r'MAE=%.3f'%MAE)

    ax[1].legend(loc="upper left",borderpad=0.15,handletextpad=0.25,bbox_to_anchor=(1.,1.02),
                 handlelength=1,columnspacing=0.15, ncol=1,prop={'size':6})

    #ax[1].set_yscale('log')
    ax[1].set_ylabel(r'frequency')
    ax[1].set_xlabel(r'error [%s]'%predtype[1])
    #ax[1].set_xlim((-2,2))

    print(np.mean(np.abs(np.array(df_O['actual'],dtype=float)-\
                         np.array(df_O['predicted'],dtype=float))))
    print(np.mean(np.abs(np.array(df_all['actual'],dtype=float)-\
                         np.array(df_all['predicted'],dtype=float))))

    print(np.mean(np.abs(np.array(df_O['actual'],dtype=float)-\
                         np.array(df_O['predicted'],dtype=float))/\
                  np.array(df_O['predicted'],dtype=float)))
    print(np.mean(np.abs(np.array(df_all['actual'],dtype=float)-\
                         np.array(df_all['predicted'],dtype=float))/\
                  np.array(df_all['predicted'],dtype=float)))

    plt.show()
    plt.close()

  
    # https://plotly.com/python/line-and-scatter/ 
    # https://stackoverflow.com/questions/65124833/plotly-how-to-combine-scatter-and-line-plots-using-plotly-express
    fig1 = px.scatter(df_all, x="actual", y="predicted", hover_data=['struct'],color='ns')
    fig2 = px.line(df_yeqx, x='x', y='y')
    fig3 = go.Figure(data=fig1.data + fig2.data)

    #fig = go.Figure()
    #fig.add_trace(go.Scatter(
    #    x=df_all["actual"], 
    #    y=df_all["predicted"], 
    #    mode='markers',
    #    marker=dict(color=df_all['batch']),
    #    text=df_all['batch']
    #))
    #fig.add_trace(go.Scatter(
    #    x=[min_act_energy,max_act_energy],
    #    y=[min_act_energy,max_act_energy],
    #    marker=dict(color='black'),
    #    mode='lines'
    #))
    fig3.update_layout(
         #title=r'Defect-CGCNN vs. DFT predictions of $\Delta H_{f,d}$',
         xaxis_title = r'%s (DFT) [%s]'%(predtype[0],predtype[1]),
         yaxis_title = r'%s (Model) [%s]'%(predtype[0],predtype[1]))

    fig3.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-files', nargs='*', help='list of all parity plot data files')
    parser.add_argument('-labels', nargs='*', help='label for each plot')
    parser.add_argument('-predtype', nargs=2, default=['E','eV'],
                        help='label for each plot')

    args = parser.parse_args()

    
    plot(**vars(args))

    
