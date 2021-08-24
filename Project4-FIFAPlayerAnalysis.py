#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#import eli5
#from eli5.sklearn import PermutationImportance
from collections import Counter
#import missingno as msno

import warnings
warnings.filterwarnings('ignore')
#import plotly
sns.set_style('darkgrid')


# In[2]:


df=pd.read_csv('/Users/cesarvenzor/Documents/Projects/JupNotebook/data.csv')


# In[3]:


df.head()


# In[4]:


df.head().T


# In[5]:


df.columns


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.set_index('Name')


# In[253]:


#df.drop(['Unnamed: 0','Photo','Flag','Club Logo'],axis=1,inplace=True)


# In[9]:


df.head()


# In[255]:


df.columns


# In[276]:


df.isnull().sum()


# In[257]:


missing_height = df[df['Height'].isnull()].index.tolist()
missing_weight = df[df['Weight'].isnull()].index.tolist()


# In[260]:


missing_height


# In[261]:


df[df['Height'].isnull()]


# In[263]:


df['Height']


# In[243]:


df.drop(df.index[missing_height],inplace =True)


# In[244]:


df.isnull().sum()


# In[153]:


print('Total number of countries : {0}'.format(df['Nationality'].nunique()))
print(df['Nationality'].value_counts().head(5))
print('--'*40)
print("\nEuropean Countries have most players")


# In[154]:


print('Total number of clubs : {0}'.format(df['Club'].nunique()))
print(df['Club'].value_counts())


# In[155]:


a = df['Club'].value_counts().tolist()


# In[156]:


t = 0
for n in a:
    t+= n


# In[157]:


t


# In[221]:


t/651

a.mean()


# In[159]:


print('Maximum Potential : '+str(df.loc[df['Potential'].idxmax()][1]))
print('Maximum Overall Perforamnce : '+str(df.loc[df['Overall'].idxmax()][1]))


# In[160]:


#dfl = df.columns.tolist()
#dfl


# In[161]:


pr_cols=['Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',
       'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',
       'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance',
       'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',
       'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
       'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving',
       'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']
print('BEST IN DIFFERENT ASPECTS :')
print('_________________________\n\n')
i=0
while i < len(pr_cols):
    print('Best {0} : {1}'.format(pr_cols[i],df.loc[df[pr_cols[i]].idxmax()][1]))
    i += 1


# In[162]:


for i in pr_cols:
    print('Best {0} : {1}'.format(pr_cols[pr_cols.index(i)],df.loc[df[pr_cols[pr_cols.index(i)]].idxmax()][1]))


# In[191]:


def value_to_int(df_value):
    try:
        value = float(df_value[1:-1])
        suffix = df_value[-1:]

        if suffix == 'M':
            value = value * 1000000
        elif suffix == 'K':
            value = value * 1000
    except ValueError:
        value = 0
    return value


# In[197]:


#df['Value'] = df['Value'].apply(value_to_int)
#df['Release Clause']= df['Release Clause'].apply(value_to_int)


# In[183]:


df['Value']


# In[185]:



df['Wage']


# In[193]:


df['Release Clause']


# In[177]:


df.head(10) #


# In[204]:


print('Most valued player : '+str(df.loc[df['Value'].idxmax()][1]))
print('Highest earner : '+str(df.loc[df['Wage'].idxmax()][1]))
print("--"*40)
print("\nTop Earners")


# In[211]:


df = df.set_index(df['Name'])


# In[212]:


df


# In[214]:


df.columns


# In[217]:


player_features = (
    'Acceleration', 'Aggression', 'Agility', 
    'Balance', 'BallControl', 'Composure', 
    'Crossing', 'Dribbling', 'FKAccuracy', 
    'Finishing', 'GKDiving', 'GKHandling', 
    'GKKicking', 'GKPositioning', 'GKReflexes', 
    'HeadingAccuracy', 'Interceptions', 'Jumping', 
    'LongPassing', 'LongShots', 'Marking', 'Penalties'
)

from math import pi
idx = 1
plt.figure(figsize=(15,45))
for position_name, features in df.groupby(df['Position'])[player_features].mean().iterrows():
    top_features = dict(features.nlargest(5))
    
    # number of variable
    categories=top_features.keys()
    N = len(categories)

    # We are going to plot the first line of the data frame.
    # But we need to repeat the first value to close the circular graph:
    values = list(top_features.values())
    values += values[:1]

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(10, 3, idx, polar=True)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=8)
 # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([25,50,75], ["25","50","75"], color="grey", size=7)
    plt.ylim(0,100)
    
    plt.subplots_adjust(hspace = 0.5)
    
    # Plot data
    ax.plot(angles, values, linewidth=1, linestyle='solid')

    # Fill area
    ax.fill(angles, values, 'b', alpha=0.1)
    
    plt.title(position_name, size=11, y=1.1)
    
    idx += 1


# In[235]:


df['Position'].nunique()


# In[234]:


df.groupby(df['Position'])[player_features].mean()


# In[215]:


sns.jointplot(x=df['Age'],y=df['Aggression'],
              joint_kws={'alpha':0.1,'s':5,'color':'red'},
              marginal_kws={'color':'red'})


# In[13]:


sns.jointplot(x=df['Age'],y=df['Potential'],
              joint_kws={'alpha':0.1,'s':5,'color':'red'},
              marginal_kws={'color':'red'})


# In[236]:


sns.lmplot(data = df, x = 'Age', y = 'SprintSpeed',lowess=True,scatter_kws={'alpha':0.01, 's':5,'color':'green'}, 
           line_kws={'color':'red'})


# In[22]:


sns.lmplot(x = 'BallControl', y = 'Dribbling', data = df,col = 'Preferred Foot', scatter_kws = {'alpha':0.1,'color':'orange'},
           line_kws={'color':'red'})


# In[23]:


sns.jointplot(x=df['Dribbling'], y=df['Crossing'], kind="hex", color="#4CB391");


# In[16]:


sns.jointplot(x=df['Marking'], y=df['Interceptions'], kind="hex", color="#4CB391")


# In[18]:


sns.jointplot(x=df['Aggression'], y=df['Agility'], kind="hex", color="#4CB391");


# In[19]:


sns.jointplot(x=df['Composure'], y=df['Finishing'], kind="hex", color="#4CB391");


# In[20]:


sns.jointplot(x=df['Composure'], y=df['Penalties'], kind="hex", color="#4CB391");


# In[ ]:




