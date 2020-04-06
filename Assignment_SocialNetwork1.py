
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-social-network-analysis/resources/yPcBs) course resource._
# 
# ---

# # Assignment 1 - Creating and Manipulating Graphs
# 
# Eight employees at a small company were asked to choose 3 movies that they would most enjoy watching for the upcoming company movie night. These choices are stored in the file `Employee_Movie_Choices.txt`.
# 
# A second file, `Employee_Relationships.txt`, has data on the relationships between different coworkers. 
# 
# The relationship score has value of `-100` (Enemies) to `+100` (Best Friends). A value of zero means the two employees haven't interacted or are indifferent.
# 
# Both files are tab delimited.

# In[47]:


import networkx as nx
import pandas as pd
import numpy as np
from networkx.algorithms import bipartite


# This is the set of employees
employees = set(['Pablo',
                 'Lee',
                 'Georgia',
                 'Vincent',
                 'Andy',
                 'Frida',
                 'Joan',
                 'Claude'])

# This is the set of movies
movies = set(['The Shawshank Redemption',
              'Forrest Gump',
              'The Matrix',
              'Anaconda',
              'The Social Network',
              'The Godfather',
              'Monty Python and the Holy Grail',
              'Snakes on a Plane',
              'Kung Fu Panda',
              'The Dark Knight',
              'Mean Girls'])


# you can use the following function to plot graphs
# make sure to comment it out before submitting to the autograder
def plot_graph(G, weight_name=None):
    '''
    G: a networkx G
    weight_name: name of the attribute for plotting edge weights (if G is weighted)
    '''
    get_ipython().magic('matplotlib notebook')
    import matplotlib.pyplot as plt
    
    plt.figure()
    pos = nx.spring_layout(G)
    edges = G.edges()
    weights = None
    
    if weight_name:
        weights = [int(G[u][v][weight_name]) for u,v in edges]
        labels = nx.get_edge_attributes(G,weight_name)
        nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
        nx.draw_networkx(G, pos, edges=edges, width=weights);
    else:
        nx.draw_networkx(G, pos, edges=edges);


# ### Question 1
# 
# Using NetworkX, load in the bipartite graph from `Employee_Movie_Choices.txt` and return that graph.
# 
# *This function should return a networkx graph with 19 nodes and 24 edges*

# In[48]:


def answer_one():
        
    G = nx.read_edgelist('Employee_Movie_Choices.txt', delimiter='\t')# Your Code Here
    return G


# ### Question 2
# 
# Using the graph from the previous question, add nodes attributes named `'type'` where movies have the value `'movie'` and employees have the value `'employee'` and return that graph.
# 
# *This function should return a networkx graph with node attributes `{'type': 'movie'}` or `{'type': 'employee'}`*

# In[49]:


def answer_two():
    
    # Your Code Here
    G = answer_one()
    for node in G.nodes():
        if node in employees:
            G.add_node(node,type='employee')
        elif node in movies:
            G.add_node(node,type='movie')
            
    return G# Your Answer Here


# ### Question 3
# 
# Find a weighted projection of the graph from `answer_two` which tells us how many movies different pairs of employees have in common.
# 
# *This function should return a weighted projected graph.*

# In[50]:


def answer_three():
    
        
    G = answer_two()
    weighted_projected_graph = bipartite.weighted_projected_graph(G, employees)
    return weighted_projected_graph# Your Answer Here


# ### Question 4
# 
# Suppose you'd like to find out if people that have a high relationship score also like the same types of movies.
# 
# Find the Pearson correlation ( using `DataFrame.corr()` ) between employee relationship scores and the number of movies they have in common. If two employees have no movies in common it should be treated as a 0, not a missing value, and should be included in the correlation calculation.
# 
# *This function should return a float.*

# In[69]:


def answer_four():
        
    # Your Code Here
    G = answer_three()
    relationship = nx.read_edgelist('Employee_Relationships.txt',data=[('relationship_score',int)])
    
    G_df = pd.DataFrame(G.edges(data=True),columns=['From','To','movies_score'])
    relationship_df = pd.DataFrame(relationship.edges(data=True),columns=['From','To','relationship_score'])
    
    ## without concat, df['movies_score'] only has one direction information, we need from 'from' to 'to' and also from 'to' to 'from' 
    G_df1 = G_df.copy()
    G_df1.rename(columns={'From':'temp','To':'From'},inplace=True)
    G_df1.rename(columns={'temp':'To'},inplace=True)
    df_concat = pd.concat([G_df,G_df1])
    df = pd.merge(df_concat,relationship_df,on = ['From','To'],how='right')
    
    ## if df['movies_score'] is nan, we fill it as 0
    def fillNan(val):
        if val is np.nan:
            return {'weight':0}
        else:
            return val
    df['movies_score'] = df['movies_score'].map(fillNan)
    
    ## get the score from dictiorary
    df['movies_score'] = df['movies_score'].map(lambda x: x['weight'])
    df['relationship_score'] = df['relationship_score'].map(lambda x: x['relationship_score'])
    value = df['movies_score'].corr(df['relationship_score'])
    
    
    return value# Your Answer Here


# In[70]:


answer_four()


# In[ ]:





# In[ ]:




