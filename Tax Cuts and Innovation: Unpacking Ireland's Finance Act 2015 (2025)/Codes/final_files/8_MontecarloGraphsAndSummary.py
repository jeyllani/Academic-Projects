#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys 

script_dir = os.path.dirname(__file__)

path = os.path.abspath(os.path.join(script_dir, 'simulations_ds'))
df = pd.read_csv(path + '/1000_hetero_ds')
df2 = pd.read_csv(path + '/1000_correlated_ds')
df3 = pd.read_csv(path + '/1000_normal_ds')


path2 = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir, os.pardir, 'final_LaTeX'))

print(path2)


# In[ ]:


df = df[['SC Score', 'RMSPE']]
df2 = df2[['SC Score', 'RMSPE']]
df3 = df3[['SC Score', 'RMSPE']]

df = df.rename(columns={'SC Score': 'hetero_sc', 'RMSPE': 'hetero_rmspe'})
df2 = df2.rename(columns={'SC Score': 'correlated_sc', 'RMSPE': 'correlated_rmspe'})
df3 = df3.rename(columns={'SC Score': 'normal_sc', 'RMSPE': 'normal_rmspe'})


# In[ ]:


df = pd.concat([df, df2, df3], axis=1)
df.head()
df.columns


# In[ ]:


# Tracer les distributions
plt.figure(figsize=(30, 10))

plt.subplot(1, 3, 1)
sns.histplot(df['hetero_sc'], kde=True, color='lightblue', bins=30)
plt.title(r'$\hat{\tau}$ Score (with noise-on-Exo)')
plt.xlabel('Scores SC')
plt.legend(['1000 sample size'])
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
sns.histplot(df['correlated_sc'], kde=True, color='salmon', bins=30)
plt.title(r'$\hat{\tau}$ Score (with correlation-btw-exo)')
plt.xlabel('Scores SC')
plt.legend(['1000 sample size'])
plt.ylabel('Frequency')

plt.subplot(1, 3, 3)
sns.histplot(df['normal_sc'], kde=True, color='orange', bins=30, alpha=0.4)
plt.title(r'$\hat{\tau}$ Score')
plt.xlabel('Scores SC')
plt.legend(['1000 sample size'])
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# In[ ]:


plt.figure(figsize=(30, 10))

plt.subplot(1, 3, 1)
sns.histplot(df['hetero_rmspe'], kde=True, color='lightblue', bins=30)
plt.title(r'RMSPE Score (with noise-on-Exo)')
plt.xlabel('RMSPE SC')
plt.legend(['1000 sample size'])
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
sns.histplot(df['correlated_rmspe'], kde=True, color='salmon', bins=30)
plt.title(r'RMSPE Score (with correlation-btw-exo)')
plt.xlabel('RMSPE SC')
plt.legend(['1000 sample size'])
plt.ylabel('Frequency')

plt.subplot(1, 3, 3)
sns.histplot(df['normal_rmspe'], kde=True, color='orange', bins=30, alpha=0.4)
plt.title(r'RMSPE Score')
plt.xlabel('RMSPE SC')
plt.legend(['1000 sample size'])
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# In[ ]:


s = df[['hetero_sc', 'correlated_sc', 'normal_sc']]
r = df[['hetero_rmspe', 'correlated_rmspe', 'normal_rmspe']]

s = s.rename(columns={'hetero_sc': 'noise', 'correlated_sc': 'correlation', 'normal_sc': 'normal'})
r = r.rename(columns={'hetero_rmspe': 'noise', 'correlated_rmspe': 'correlation', 'normal_rmspe': 'normal'})

r = r.describe()
s = s.describe()

r.loc['skew'] = r.skew()
r.loc['kurt'] = r.kurt()
r = r.T.round(2)

s.loc['skew'] = s.skew()
s.loc['kurt'] = s.kurt()
s = s.T.round(2)

# r.to_latex(path2 + 'rmspe.tex')
print('\n', 'RMSPE', '\n', r)


# In[ ]:


# s.to_latex(path2 + 'sc.tex')
print('\n', 'tau_hat', '\n', s)

