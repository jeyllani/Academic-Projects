#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd 
import numpy as np
import os
script_dir = os.path.dirname(__file__)
base_path = os.path.join(script_dir, 'initial_cleaning', 'final_datasets/')
base_path2 = os.path.join(script_dir)
pd.set_option('display.max_columns', None)

# In[48]:


df = pd.read_csv(base_path + '/did_with_all.csv')
df2 = pd.read_csv(base_path + 'merged.csv')
df3 = pd.read_csv(base_path + 'interpoled.csv')
df4 = pd.read_csv(base_path + 'merged_short.csv')
dep = pd.read_csv(base_path + 'depreciation_raw.csv')
ide = pd.read_csv(base_path + 'ide.csv')
ide = ide.sort_values(by=['country', 'year'], ascending=True)
df2 = df2.dropna(axis=1)
df2 = df2.drop(columns=['brevets_pct', 'brevets_tri', 'dem_pct'])

df2['country'].unique()


# In[49]:



df2[['NX', 'salary', 'gdp', 'cho', 'ppa', 'public_spending']] = df2[[ 'NX', 'salary', 'gdp', 'cho', 'ppa', 'public_spending']].pct_change()
df2['pop'] = np.log(df2['pop'])
df2['brevets'] = np.log(df2['brevets'])
# ∆∆ -> Economic acceleration


# In[50]:


df2 = df2[(df2['year'] >= 1996) & (df2['year'] < 2019)]


# In[51]:


df2[df2['country'] == 'CHE'].describe()


# In[52]:
# Code was changed specifically to review the outputs 

# print(ide.head(1))
# print(dep.head(1))


# In[53]:


ide['IDE'] = ide['IDE'].pct_change()    
ide = ide[(ide['year'] >= 1996) & (ide['year'] < 2019)]
dep['depreciation'] = dep['depreciation'].pct_change()
dep['ratio_dbrut'] = dep['ratio_dbrut'].pct_change()
dep = dep[(dep['year'] >= 1996) & (dep['year'] < 2019)]


# In[54]:


dep.rename(columns={'id': 'country'}, inplace=True)


# In[55]:


df2 = pd.merge(df2, ide, on=['country', 'year'], how='left')


# In[56]:


# For patents
# df2.to_csv('~/final_files/final_clean/patents_dataset.csv', index=False)
# Code was changed specifically to review the outputs 

# In[57]:


# dep.country.unique()


# In[58]:


df2 = df2[(df2['country'] != 'SWE') & (df2['country'] != 'CHE')] ## Missing Values for SWE / CHE


# In[59]:


df2 = pd.merge(df2, dep, on=['country', 'year'], how='left')


# In[60]:
# Code was changed specifically to review the outputs 

# For Depreciation
# df2.to_csv('~/final_files/final_clean/depreciation_dataset.csv', index=False)


# In[61]:

df = pd.read_csv(os.path.join(base_path2, 'final_clean', 'depreciation_dataset.csv'))
zf = pd.read_csv(os.path.join(base_path2, 'final_clean', 'patents_dataset.csv'))

# In[63]:


df.country.unique()


# In[64]:

def summary(df=pd.DataFrame(), country=str, subject=str):

    lat_c = df[(df['country'] == country)] # Change here in case to CHE
    lat_c = lat_c[['year', 'country', 'brevets', 'gdp', 'pop', 'NX', 'public_spending']]
    lat_c = lat_c.describe()
    lat_c = lat_c.iloc[1:,1:]
    lat_c.loc['skewness'] = lat_c.skew()
    lat_c.loc['kurtosis'] = lat_c.kurt()
    lat_c.loc['median'] = lat_c.median()
    lat_c = lat_c.T
    lat_c = lat_c[['mean', 'median', 'std', 'skewness', 'kurtosis', '25%', '75%']]
    

    return print("\n", "\n", "\n", "Country :" + country, "\n",  "Subject : " + subject, "\n",  lat_c)


## Summaries Patents

summary(zf, 'IRL', 'Patents')
summary(zf, 'CHE', 'Patents')
summary(zf, 'AUT', 'Patents')
summary(zf, 'FIN', 'Patents')
summary(zf, 'SWE', 'Patents')


## Summaries Depreciation 
summary(df, 'IRL', 'Depreciation')
summary(df, 'FIN', 'Depreciation')
summary(df, 'AUT', 'Depreciation')


# In[65]:

# Code was changed specifically to review the outputs 

# lat_b.to_latex('~/final_files/final_LaTeX/summary/brevets_AUT_summary.tex')
# lat_c.to_latex('~/final_files/final_LaTeX/summary/brevets_AUT_summary.tex')




# In[67]:

# Code was changed specifically to review the outputs 

# Save summary, country name can be changed in the previous
# df.to_latex('~/final_files/final_LaTeX/summary/depreciation_FIN_summary.tex')
