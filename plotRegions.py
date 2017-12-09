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



def plotGraph(G,org,phase,plot=False,center_node=None):
    pos = nx.fruchterman_reingold_layout(G)
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
            visible = False,
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

    def node_info(node,freq=None):
        if freq==None:
            in_adj = G.in_degree(node)
            out_adj = G.out_degree(node)
        else:
            out_adj = G.get_edge_data(node,freq,default=0)
            in_adj = G.get_edge_data(freq,node,default=0)
            if isinstance(out_adj,dict):
                out_adj = out_adj["weight"]
            if isinstance(in_adj,dict):
                in_adj = in_adj["weight"]
        node_info = org.get(node,str(node)) + "("+str(node)+")"+' <br># of outgoing connections: '+str(out_adj) + ' <br># of incoming connections: '+str(in_adj)
#        elif node <= 82:
#            node_info = "Trafficer # "+str(node) +' <br># of outgoing connections: '+str(out_adj) + ' <br># of incoming connections: '+str(in_adj)
#        else:
#            node_info = "Non-Trafficer # "+str(node) +' <br># of outgoing connections: '+str(out_adj) + ' <br># of incoming connections: '+str(in_adj)
        return node_info
    if center_node != None:
        neighbors = G.neighbors(center_node)
        total_in = 0
        total_out = 0
        for n in neighbors:
            x0, y0 = pos[center_node]
            x1, y1 = pos[n]
            adj0 = G.neighbors(center_node)
            adj1 = G.neighbors(n)
            edge_trace['x'] += [x0, x1, None]
            edge_trace['y'] += [y0, y1, None]
            edge_trace['marker']['color'] += [len(adj0),len(adj1),None]
            weightOut =  G.get_edge_data(center_node,n,default=0)
            weightIn = G.get_edge_data(n,center_node,default=0)
            if isinstance(weightOut,dict):
                total_out += weightOut["weight"]
            if isinstance(weightIn,dict):
                total_in += weightIn["weight"]
            info0 = None
            info1 = node_info(n,center_node)
            edge_trace['text'] += [info0, info1, None]
        total_info = org[center_node] + "("+str(center_node)+")"+' <br># of outgoing connections: '+str(total_out) + ' <br># of incoming connections: '+str(total_in)
        edge_trace['text'][0] = total_info
    else:
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
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
    print(fig)
    if plot:
        py.plot(fig, filename=phase+'.html')
    else:
        print(edge_trace)
        return edge_trace  

def plotRegions(G,backg):
    trace1= go.Scatter(x=np.arange(1000),y=np.arange(1000))
    layout= go.Layout(images= [dict(
                  source= "https://upload.wikimedia.org/wikipedia/commons/8/82/Regions_administratives_du_Quebec.png",
                  xref= "x",
                  yref= "y",
                  x= 300,
                  y= 950,
                  sizex= 400,
                  sizey= 900,
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
plotGraph(G,{},"regions",plot=True)
plotRegions(None,None)