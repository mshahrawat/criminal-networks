# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 13:06:33 2017

@author: Eghosa
"""

import numpy as np
import pandas as pd
import sklearn as skl
from sklearn.datasets import make_spd_matrix as make_cov
import matplotlib.pyplot as plt
import scipy as sp
import os
import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
import math
import pdb
import networkx as nx
import plotly.plotly as py_o
from plotly.grid_objs import Grid, Column
import plotly.tools as tls
import time
from functools import partial
import pickle
from functools import reduce
import sqlite3 as sql
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA,ARIMA
from sklearn.metrics import mean_squared_error
import seaborn



def plotGraph(G,org,phase,plot=False,pos=None):
    if pos == None:
        pos = nx.fruchterman_reingold_layout(G)
    print(pos)
    dmin=1
#    ncenter=0
    for n in pos:
        x,y=pos[n]
        d=(x-0.5)**2+(y-0.5)**2
        if d<dmin:
#            ncenter=n
            dmin=d
    edge_trace = go.Scatter(
            x=[],
            y=[],
            text = [],
            visible = True,
            line=go.Line(width=0.5,color='#888'),
                         hoverinfo='text',
                         mode='lines+markers',
                         marker= go.Marker(
            showscale=True,
            # colorscale options
            # 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' |
            # Jet' | 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'
            colorscale='YIGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)))

    def node_info(node):
        in_adj = G.in_degree(node)
        out_adj = G.out_degree(node)

        node_info = org.get(node,str(node+1)) + "("+str(node+1)+")"+' <br># of outgoing connections: '+str(out_adj) + ' <br># of incoming connections: '+str(in_adj)
#        elif node <= 82:
#            node_info = "Trafficer # "+str(node) +' <br># of outgoing connections: '+str(out_adj) + ' <br># of incoming connections: '+str(in_adj)
#        else:
#            node_info = "Non-Trafficer # "+str(node) +' <br># of outgoing connections: '+str(out_adj) + ' <br># of incoming connections: '+str(in_adj)
        return node_info

    for edge in G.edges():
        x0, y0 = pos[edge[0]+1]
        x1, y1 = pos[edge[1]+1]
        adj0 = G.neighbors(edge[0])
        adj1 = G.neighbors(edge[1])
        edge_trace['x'] += [x0, x1, None]
        edge_trace['y'] += [y0, y1, None]
        edge_trace['marker']['color'] += [len(adj0),len(adj1),None]
        info0 = node_info(edge[0])
        info1 = node_info(edge[1])
        edge_trace['text'] += [info0, info1, None]
    
    
    fig = go.Figure(data=go.Data([edge_trace]),
             layout=go.Layout(
                title='<br>Network graph of ' + phase,
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=go.XAxis(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=go.YAxis(showgrid=False, zeroline=False, showticklabels=False)))
    if plot:
        py.plot(fig, filename=phase+'.html') 
    return edge_trace
def plotRegions(G,backg):
    trace1= G
    layout= go.Layout(images= [dict(
                  source= "https://upload.wikimedia.org/wikipedia/commons/8/82/Regions_administratives_du_Quebec.png",
                  xref= "x",
                  yref= "y",
                  x= 300,
                  y= 1000,
                  sizex= 400,
                  sizey= 1000,
                  sizing= "stretch",
                  opacity= 0.5,
                  layer= "below")])
    fig=go.Figure(data=[trace1],layout=layout)
    py.plot(fig)
mat = pd.read_csv("region_adj.csv",header=None)
print(mat)
G = nx.from_numpy_matrix(mat.values)
G = G.to_directed()
print(G.edges())
org = dict()

pos = {17:[448,100],16:[432,60],15:[395,130],14:[416,130],13:[420,75],12:[471,115],11:[601,220],10:[426,575],9:[536,390],8:[354,210],7:[372,120],6:[423,65],5:[462,70],4:[431,190],3:[464,175],2:[456,290],1:[508,200]}
trace = plotGraph(G,org,"regions",plot=False,pos=pos)
plotRegions(trace,None)