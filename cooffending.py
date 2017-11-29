# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 18:21:21 2017

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
import time
from functools import partial
import pickle
from functools import reduce
import sqlite3 as sql




load_from_file = False

def read_vectors():
    all_vectors = {}
    all_actors = set()
    for subdir, dirs, files in os.walk(os.getcwd()):
#        con = sql.connect("cooffending.db")
        for f in files:
            if f.endswith(".csv"):
                fname = f.split(".")[0]
                vec = pd.read_csv(f,header=0)
#                vec.to_sql("cooffending",con)
                all_actors.union(set(vec.index.values))
#                vec[vec==0] = np.nan
                all_vectors[fname] = vec
    return all_vectors

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
            in_adj = 0
            out_adj = G.degree(node)
        else:
            out_adj = G.get_edge_data(node,freq,default=0)
            in_adj = G.get_edge_data(freq,node,default=0)
            if isinstance(out_adj,dict):
                out_adj = out_adj["weight"]
            if isinstance(in_adj,dict):
                in_adj = in_adj["weight"]
        if node in org:
            node_info = org[node] + "("+str(node)+")"+' <br># of outgoing connections: '+str(out_adj) + ' <br># of incoming connections: '+str(in_adj)
        elif node <= 82:
            node_info = "Criminal # "+str(node) +' <br># of outgoing connections: '+str(out_adj) + ' <br># of incoming connections: '+str(in_adj)
        else:
            node_info = "Criminal # "+str(node) +' <br># of outgoing connections: '+str(out_adj) + ' <br># of incoming connections: '+str(in_adj)
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
                title='<br>Network graph of Criminal Activity in Quebec from ' + phase,
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=go.XAxis(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=go.YAxis(showgrid=False, zeroline=False, showticklabels=False)))
    
    if plot:
        py.plot(fig, filename=phase+'.html')
    else:
        print(edge_trace)
        return edge_trace       

class Offender(object):
    
    def __init__(self,id_num,DOB,sex):
        self.id_num = id_num
        self.DOB = DOB
        self.sex = sex
        self.crimes = {}
        self.cooffenders = {}
        
    def __repr__(self):
        return str(self.id_num)
    
#    def __eq__(self,other):
#        return self.id_num == other.id_num
    
    def add_crime(self,year,date,event,crime):
        crime_year = self.crimes.setdefault(year,{})
        crime_date = crime_year.setdefault(date,{})
        crime_event = crime_date.setdefault(event,{})
        crime = crime_event.setdefault("crimes",[])
        crime.append(crime)
    
    def add_crimes(self,year,date,event,crimes):
        for crime in crimes:
            self.add_crime(year,date,event,crime)
   
    def get_crimes(self,year=None):
        return self.crimes.get(year,self.crimes)
    
    def add_cooffender(self,year,date,event,offender):
        crime_year = self.crimes.setdefault(year,{})
        crime_date = crime_year.setdefault(date,{})
        crime_event = crime_date.setdefault(event,{})
        cooffenders = crime_event.setdefault("cooffenders",[])
        cooffenders.append(offender)
    
    def add_cooffenders(self,year,date,event,offenders):
        ooff_year = self.cooffenders.setdefault(year,set())
            
        for offender in offenders:
            self.add_cooffender(year,date,event,offender)
            ooff_year.add(offender)
    
    def get_cooffenders_for_year(self,year=None):
        val = self.cooffenders.get(year,[])
        if len(val) > 0:
            return val
        all_co = self.cooffenders.values()
        if len(all_co) > 0:
            ans = reduce(set.union,all_co,set())
#            print(ans)
            return list(ans)
        return []
        

    @property
    def num_crimes(self):
        return sum(len(crimes) for crimes in self.crimes.values())
    
class CrimeEvent(object):
    
    def __init__(self,event,year,date,crime_types,cooffenders,location,municipalites,youths,adults):
        self.event = event
        self.year = year
        self.date = date
        self.crime_types = crime_types
        self.cooffenders = cooffenders
        self.location = location
        self.mun = municipalites
        self.youths = youths
        self.adults = adults
    
def create_data_dict(data):
    crime_dict = {}
    offenders = {}
    dict_order = ["annee","Date", "SeqE"]
    for index,row in data.iterrows():
        
        current_dict = crime_dict
        for key in dict_order:
#            current_dict = current_dict.setdefault(key,{})
            current_dict = current_dict.setdefault(row[key],{})
        current_dict["Location"] = row["ED1"]
        current_dict["MUN"] = row["MUN"]
        offender = offenders.get(row['NoUnique'],Offender(row['NoUnique'],row['Naissance'],row['SEXE']))
        crimes = list(filter(lambda x: x !='',map(lambda x: x.strip(),row[['NCD1','NCD2','NCD3','NCD4']])))
        offender.add_crimes(row["annee"],row['Date'],row["SeqE"],crimes)
        current_dict = current_dict.setdefault("offenders",{})
        current_dict[offender] = offender.num_crimes
        offenders[row['NoUnique']] = offender
    return crime_dict,offenders
    

if not load_from_file:
    cooffend = read_vectors()
#    conn = sql.connect("cooffending.db")
#    cur = conn.cursor()
#    cur.execute('SELECT * FROM cooffending WHERE annee=?',(2003,))
#    for row in cur.fetchall():
#        print(row)
#    assert True == False
    crime_dict,offenders = create_data_dict(cooffend["Cooffending"])
    pickle.dump(cooffend,open('cooffend.p',"wb"))
    pickle.dump(crime_dict,open("crimes.p","wb"))
    pickle.dump(offenders,open('offenders.p', "wb"))
else:
    cooffend = pickle.load(open("cooffend.p","rb"))
    crime_dict = pickle.load(open("crimes.p","rb"))
    offenders = pickle.load(open("offenders.p","rb"))
#graph = create_graph(cooeffend["Cooffending"])
#print(graph)


"""
Problem 5.1
"""
#num_offenders = len(offenders)
#num_cases = len(cooffend["Cooffending"]["SeqE"])
#def num_crimes(year):
#    num = 0
#    for date in year:
#        num += len(year[date])
#    return num
#        
#total_crimes = 0
#crimes_per_year = {}
#for year,dates in crime_dict.items():
#    crimes = num_crimes(dates)
#    total_crimes += crimes
#    crimes_per_year[year] = crimes
#    
#all_crimes = []
#for year,dates in crime_dict.items():
#    for date,events in dates.items():
#        for event,info in events.items():
#            crime_data = (event,len(info["offenders"]),info["MUN"])
#            all_crimes.append(crime_data)
#
#
#sort_crimes = sorted(all_crimes,key=lambda x: x[1],reverse=True)
#top_five = sort_crimes[:5]
#print("Number of ooffenders:",num_offenders)
#print("Number of cases:",num_cases)
#print("Total num of crimes:",total_crimes)
#print("crimes per yer:",crimes_per_year)
#print("top five crimes:",top_five)

"""
Problem 5.2
"""

def update_cooffenders(crime_dict,cooffenders):
    
    cooff_count = {}
    for year,dates in crime_dict.items():
        for date,events in dates.items():
            for event,info in events.items():
                offenders = info["offenders"]
                for offender in offenders:
                    others = list(set(offenders).difference(set([offender])))
                    if others:
                        cooffenders[offender.id_num].add_cooffenders(year,date,event,others)
                    count = cooff_count.setdefault(offender,set())
                    count.update(others)
    return cooff_count
def crimes_committed_together(off1,off2,year=None):
    
    off1_crimes = off1.get_crimes(year)
    off2_crimes = off2.get_crimes(year)
    num_matches = 0
    if year != None:
        for date,events in off1_crimes.items():
            off1_events = set(events.keys())
            off2_events = set(off2_crimes.get(date,{}).keys())
            num_matches += len(off1_events.intersection(off2_events))
    else: 
        for year,dates in off1_crimes.items():
            for date,events in dates.items():
                off1_events = set(events.keys())
                off2_events = set(off2_crimes.get(year,{}).get(date,{}).keys())
                num_matches += len(off1_events.intersection(off2_events))
    return num_matches

def create_cooffending_network(offenders):
    years = {k:nx.Graph() for k in range(2003,2011)}
    overall = nx.Graph()
    for key,offender in offenders.items():
#        for year in years:
#            cooffenders = offender.get_cooffenders_for_year(year)
##            print("cooff",offender.cooffenders.values())
#            edges = []
#     
##            print(cooffenders)
#            for cooff in cooffenders:
#                crimes_toget = crimes_committed_together(offender,cooff,year)
#                old_weight = years[year].get_edge_data(offender,cooff,default=0)
#                
#                if isinstance(old_weight,dict):
#                    old_weight = old_weight["weight"]
#                
#                crimes_toget += old_weight
#                
#                edges.append((offender,cooff,crimes_toget))
#                
#            years[year].add_weighted_edges_from(edges,weight="weight")
        
        o_edges = []
        all_cooffenders = offender.get_cooffenders_for_year()
        for cooff in all_cooffenders:
            all_toget = crimes_committed_together(offender,cooff)
            old_weight_o = overall.get_edge_data(offender,cooff,default=0)
            if isinstance(old_weight_o,dict):
                old_weight_o = old_weight_o["weight"]
            new_weight = old_weight_o + all_toget
            o_edges.append((offender,cooff,new_weight))
        overall.add_weighted_edges_from(o_edges,weight="weight")
    
    years["overall"] = overall
#    num_individual_offenders = 0
#    for key,graph in years.items():
#        deg = graph.degree()
#        to_remove = [i for i in deg if deg[i] < 1]
#        if key=="overall":
#            print("removing nodes",len(to_remove))
#            num_individual_offenders += len(to_remove)
#        graph.remove_nodes_from(to_remove)

    return years

def plotDist(x,y,x_label,y_label,title):
    trace1 = go.Scatter(
                x = x,
                y = y,
                mode = 'lines+markers')
    data = [trace1]
    layout = go.Layout(
            title = title,
            xaxis = dict(title=x_label),
            yaxis = dict(title=y_label))
    fig = go.Figure(data=data,layout=layout)
    py.plot(fig,filename= title +".html")
    
def plotHist(x,bin_size,x_label,y_label,title):
    trace1 = go.Histogram(
                x = x,
                xbins = dict(start=np.min(x), size=bin_size, end=np.max(x)))
    data = [trace1]
    layout = go.Layout(
            title = title,
            xaxis = dict(title=x_label),
            yaxis = dict(title=y_label))
    fig = go.Figure(data=data,layout=layout)
    py.plot(fig,filename= title +".html")
    

counts =  update_cooffenders(crime_dict,offenders)
num_lone = len(list(filter(lambda x: len(counts[x])==0,counts)))
#if load_from_file:
#    print(len(offenders))
#    overall = nx.read_gpickle("graph.p")
#    print(len(offenders))
#else:
#    print(len(offenders))
overall = create_cooffending_network(offenders)["overall"]
print(len(offenders))
#nx.write_gpickle(overall,"graph.p")

print("number of nodes:",nx.number_of_nodes(overall))
print("number of lone_wolves:",num_lone)
print("number of edges:",nx.number_of_edges(overall))
deg_freq = nx.degree_histogram(overall)
degrees = np.arange(len(deg_freq))
connected_components = [c for c in sorted(nx.connected_components(overall), key=len, reverse=True)]
largest_component = connected_components[0]
print("number of connected components:",len(connected_components))
print("num of nodes in largest cc:",len(largest_component))
plotDist(degrees,deg_freq,"Number of Degrees","Frequency of Degrees","Frequency of nodes with degrees")

"""
Problem 5.3
"""

def sort_dict_by_value(data_dict,reverse=False):
    return sorted([(k,v) for k,v in data_dict.items()],key=lambda tup: tup[1],reverse=reverse)

lcc_graph = overall.subgraph(largest_component)
#larg_deg_freq = nx.degree_histogram(lcc_graph)
#lcc_degrees = np.arange(len(larg_deg_freq))
#plotDist(lcc_degrees,larg_deg_freq,"Number of Degrees","Frequency of Degrees","Frequency of nodes with degrees in Largest Connected Component")
#
#node_degrees = sort_dict_by_value(lcc_graph.degree(),True)
#top5_deg = node_degrees[:5]
#print(top5_deg)
btwCent = nx.betweenness_centrality(lcc_graph,weight="weight")
sort_btw = sort_dict_by_value(btwCent,True)
top5_btw = sort_btw[:5]
print(top5_btw)
#eigCent = nx.eigenvector_centrality(lcc_graph,max_iter = 2000, tol = 1e-2, weight="weight")
#sort_eig = sort_dict_by_value(eigCent,True)
#top5_eig = sort_eig[:5]
#print(top5_eig)
#clustering = nx.clustering(lcc_graph,weight="weight")
#sort_clust = sort_dict_by_value(clustering,True)
#top5_clust = sort_clust[:5]
#print(top5_clust)
plotHist([round(val,4) for val in btwCent.values()],0.002,"Betweeness Centraliity","Frequency of nodes","Betweeness Centrality Distribution for nodes in LCC")
#plotHist([round(val,4) for val in eigCent.values()],0.002,"Eigenvector Centraliity","Frequency of nodes","Eigenvector Centrality Distribution for nodes in LCC")
plotGraph(lcc_graph,{},"2003-2010",plot=True)

            
"""
Bonus
"""
def create_crime_events_network(crime_dict):
    pass
                       