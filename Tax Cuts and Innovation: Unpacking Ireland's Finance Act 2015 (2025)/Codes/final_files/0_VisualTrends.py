


# In[111]:


import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np
from linearmodels.panel import PanelOLS
from scipy.stats import skew, kurtosis
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scienceplots
plt.style.use('seaborn-v0_8-deep')


import os 
script_dir = os.path.dirname(__file__)
path = os.path.abspath(os.path.join(script_dir,'final_clean'))
path2 = os.path.abspath(os.path.join(script_dir,'final_LaTeX', 'pt_test'))

pd.set_option('display.max_columns', None)


# In[93]:


d1 = pd.read_csv(f"{path}/patents_dataset.csv")
d2 = pd.read_csv(f"{path}/depreciation_dataset.csv")


# In[95]:


pre_bv = d1[d1['year'] <= 2015][['year', 'country', 'brevets']]
post_bv = d1[d1['year'] >= 2015][['year', 'country', 'brevets']]

plt.figure(figsize=(13, 6))
plt.rc('font', family='serif', serif='Latin Modern Roman')
plt.rc('text', usetex=True)
plt.title(r'\textbf{Patents over time}')    
sns.regplot(data =pre_bv[pre_bv['country']    == 'IRL'], x ='year', y ='brevets', scatter=False, label=r'\textit{Ireland}', color='green', ci=0)
sns.regplot(data =post_bv[post_bv['country']  == 'IRL'], x ='year', y ='brevets', scatter =False, color ='green', ci=0)
sns.regplot(data =pre_bv[pre_bv['country']    == 'CHE'], x ='year', y ='brevets', scatter =False, label =r'\textit{Switzerland}', color ='blue', ci=0)
sns.regplot(data =post_bv[post_bv['country']  == 'CHE'], x ='year', y ='brevets', scatter =False, color ='blue', ci=0)
sns.regplot(data =pre_bv[pre_bv['country']    == 'SWE'], x ='year', y ='brevets', scatter =False, label =r'\textit{Sweden', color       ='orange', ci=0)
sns.regplot(data =post_bv[post_bv['country']  == 'SWE'], x ='year', y ='brevets', scatter =False, color ='orange', ci=0)
sns.regplot(data =pre_bv[pre_bv['country']    == 'FIN'], x ='year', y ='brevets', scatter =False, label =r'\textit{Finland}', color     ='red', ci=0)
sns.regplot(data =post_bv[post_bv['country']  == 'FIN'], x ='year', y ='brevets', scatter =False, color ='red', ci=0)
sns.regplot(data =pre_bv[pre_bv['country']    == 'AUT'], x ='year', y ='brevets', scatter =False, label =r'\textit{Austria}', color     ='black', ci=0)
sns.regplot(data =post_bv[post_bv['country']  == 'AUT'], x ='year', y ='brevets', scatter =False, color ='black', ci=0)
plt.axvline(x=2015, color='red', linestyle='--', label=r'\textit{Treatment Year}')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6, fontsize='small')
plt.ylabel(r'\textit{log Patents}')
plt.grid(True)
plt.show()




# In[97]:


pre_nx = d1[d1['year'] <= 2015][['year', 'country', 'NX']]
post_nx = d1[d1['year'] >= 2015][['year', 'country', 'NX']]

plt.figure(figsize=(13, 6))
plt.rc('font', family='serif', serif='Latin Modern Roman')
plt.rc('text', usetex=True)
plt.title(r'\textbf{\% NX over time}')    
sns.regplot(data =pre_nx[pre_nx['country']    == 'IRL'], x ='year', y ='NX', scatter=False, label=r'\textit{Ireland}', color='green', ci=0)
sns.regplot(data =post_nx[post_nx['country']  == 'IRL'], x ='year', y ='NX', scatter =False, color ='green', ci=0)
sns.regplot(data =pre_nx[pre_nx['country']    == 'CHE'], x ='year', y ='NX', scatter =False, label =r'\textit{Switzerland}', color ='blue', ci=0)
sns.regplot(data =post_nx[post_nx['country']  == 'CHE'], x ='year', y ='NX', scatter =False, color ='blue', ci=0)
sns.regplot(data =pre_nx[pre_nx['country']    == 'SWE'], x ='year', y ='NX', scatter =False, label =r'\textit{Sweden', color       ='orange', ci=0)
sns.regplot(data =post_nx[post_nx['country']  == 'SWE'], x ='year', y ='NX', scatter =False, color ='orange', ci=0)
sns.regplot(data =pre_nx[pre_nx['country']    == 'FIN'], x ='year', y ='NX', scatter =False, label =r'\textit{Finland}', color     ='red', ci=0)
sns.regplot(data =post_nx[post_nx['country']  == 'FIN'], x ='year', y ='NX', scatter =False, color ='red', ci=0)
sns.regplot(data =pre_nx[pre_nx['country']    == 'AUT'], x ='year', y ='NX', scatter =False, label =r'\textit{Austria}', color     ='black', ci=0)
sns.regplot(data =post_nx[post_nx['country']  == 'AUT'], x ='year', y ='NX', scatter =False, color ='black', ci=0)
plt.axvline(x=2015, color='red', linestyle='--', label=r'\textit{Treatment Year}')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6, fontsize='small')
plt.ylabel(r'\textit{\% Net Exchanges}')
plt.grid(True)
plt.show()


# In[98]:


pre_gdp = d1[d1['year'] <= 2015][['year', 'country', 'gdp']]
post_gdp = d1[d1['year'] >= 2015][['year', 'country', 'gdp']]

plt.figure(figsize=(13, 6))
plt.rc('font', family='serif', serif='Latin Modern Roman')
plt.rc('text', usetex=True)
plt.title(r'\textbf{gdp acceleration over time}')    
sns.regplot(data =pre_gdp[pre_gdp['country']    == 'IRL'], x ='year', y ='gdp', scatter=False, label=r'\textit{Ireland}', color='green', ci=0)
sns.regplot(data =post_gdp[post_gdp['country']  == 'IRL'], x ='year', y ='gdp', scatter =False, color ='green', ci=0)
sns.regplot(data =pre_gdp[pre_gdp['country']    == 'CHE'], x ='year', y ='gdp', scatter =False, label =r'\textit{Switzerland}', color ='blue', ci=0)
sns.regplot(data =post_gdp[post_gdp['country']  == 'CHE'], x ='year', y ='gdp', scatter =False, color ='blue', ci=0)
sns.regplot(data =pre_gdp[pre_gdp['country']    == 'SWE'], x ='year', y ='gdp', scatter =False, label =r'\textit{Sweden', color       ='orange', ci=0)
sns.regplot(data =post_gdp[post_gdp['country']  == 'SWE'], x ='year', y ='gdp', scatter =False, color ='orange', ci=0)
sns.regplot(data =pre_gdp[pre_gdp['country']    == 'FIN'], x ='year', y ='gdp', scatter =False, label =r'\textit{Finland}', color     ='red', ci=0)
sns.regplot(data =post_gdp[post_gdp['country']  == 'FIN'], x ='year', y ='gdp', scatter =False, color ='red', ci=0)
sns.regplot(data =pre_gdp[pre_gdp['country']    == 'AUT'], x ='year', y ='gdp', scatter =False, label =r'\textit{Austria}', color     ='black', ci=0)
sns.regplot(data =post_gdp[post_gdp['country']  == 'AUT'], x ='year', y ='gdp', scatter =False, color ='black', ci=0)
plt.axvline(x=2015, color='red', linestyle='--', label=r'\textit{Treatment Year}')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6, fontsize='small')
plt.ylabel(r'$\Delta^2$\textit{GDP}')
plt.grid(True)
plt.show()


# In[99]:


pre_pop = d1[d1['year'] <= 2015][['year', 'country', 'pop']]
post_pop = d1[d1['year'] >= 2015][['year', 'country', 'pop']]

plt.figure(figsize=(13, 6))
plt.title(r'\textbf{log population over time}')    
sns.regplot(data =pre_pop[pre_pop['country']    == 'IRL'], x ='year', y ='pop', scatter=False, label=r'\textit{Ireland}', color='green', ci=0)
sns.regplot(data =post_pop[post_pop['country']  == 'IRL'], x ='year', y ='pop', scatter =False, color ='green', ci=0)
sns.regplot(data =pre_pop[pre_pop['country']    == 'CHE'], x ='year', y ='pop', scatter =False, label =r'\textit{Switzerland}', color ='blue', ci=0)
sns.regplot(data =post_pop[post_pop['country']  == 'CHE'], x ='year', y ='pop', scatter =False, color ='blue', ci=0)
sns.regplot(data =pre_pop[pre_pop['country']    == 'SWE'], x ='year', y ='pop', scatter =False, label =r'\textit{Sweden', color       ='orange', ci=0)
sns.regplot(data =post_pop[post_pop['country']  == 'SWE'], x ='year', y ='pop', scatter =False, color ='orange', ci=0)
sns.regplot(data =pre_pop[pre_pop['country']    == 'FIN'], x ='year', y ='pop', scatter =False, label =r'\textit{Finland}', color     ='red', ci=0)
sns.regplot(data =post_pop[post_pop['country']  == 'FIN'], x ='year', y ='pop', scatter =False, color ='red', ci=0)
sns.regplot(data =pre_pop[pre_pop['country']    == 'AUT'], x ='year', y ='pop', scatter =False, label =r'\textit{Austria}', color     ='black', ci=0)
sns.regplot(data =post_pop[post_pop['country']  == 'AUT'], x ='year', y ='pop', scatter =False, color ='black', ci=0)
plt.axvline(x=2015, color='red', linestyle='--', label=r'\textit{Treatment Year}')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6, fontsize='small')
plt.ylabel(r'\textit{log(population)}')
plt.grid(True)
plt.show()


# In[110]:


pre_sal = d1[d1['year'] <= 2015][['year', 'country', 'salary']]
post_sal = d1[d1['year'] >= 2015][['year', 'country', 'salary']]

plt.figure(figsize=(13, 6))
plt.title(r'\textbf{\% salary over time}')    
sns.regplot(data =pre_sal[pre_sal['country']    == 'IRL'], x ='year', y ='salary', scatter=False, label=r'\textit{Ireland}', color='green', ci=0)
sns.regplot(data =post_sal[post_sal['country']  == 'IRL'], x ='year', y ='salary', scatter =False, color ='green', ci=0)
sns.regplot(data =pre_sal[pre_sal['country']    == 'FIN'], x ='year', y ='salary', scatter =False, label =r'\textit{Finland}', color     ='red', ci=0)
sns.regplot(data =post_sal[post_sal['country']  == 'FIN'], x ='year', y ='salary', scatter =False, color ='red', ci=0)
sns.regplot(data =pre_sal[pre_sal['country']    == 'AUT'], x ='year', y ='salary', scatter =False, label =r'\textit{Austria}', color     ='black', ci=0)
sns.regplot(data =post_sal[post_sal['country']  == 'AUT'], x ='year', y ='salary', scatter =False, color ='black', ci=0)
plt.axvline(x=2015, color='red', linestyle='--', label=r'\textit{Treatment Year}')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6, fontsize='small')
plt.ylabel(r'\textit{\% Salary}')
plt.grid(True)
plt.show()


# In[109]:


pre_cho = d1[d1['year'] <= 2015][['year', 'country', 'cho']]
post_cho = d1[d1['year'] >= 2015][['year', 'country', 'cho']]

plt.figure(figsize=(13, 6))
plt.title(r'\textbf{\% Unemployment (cho) over time}')    
sns.regplot(data =pre_cho[pre_cho['country']    == 'IRL'], x ='year', y ='cho', scatter=False, label=r'\textit{Ireland}', color='green', ci=0)
sns.regplot(data =post_cho[post_cho['country']  == 'IRL'], x ='year', y ='cho', scatter =False, color ='green', ci=0)
sns.regplot(data =pre_cho[pre_cho['country']    == 'FIN'], x ='year', y ='cho', scatter =False, label =r'\textit{Finland}', color     ='red', ci=0)
sns.regplot(data =post_cho[post_cho['country']  == 'FIN'], x ='year', y ='cho', scatter =False, color ='red', ci=0)
sns.regplot(data =pre_cho[pre_cho['country']    == 'AUT'], x ='year', y ='cho', scatter =False, label =r'\textit{Austria}', color     ='black', ci=0)
sns.regplot(data =post_cho[post_cho['country']  == 'AUT'], x ='year', y ='cho', scatter =False, color ='black', ci=0)
plt.axvline(x=2015, color='red', linestyle='--', label=r'\textit{Treatment Year}')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6, fontsize='small')
plt.ylabel(r'\textit{\% Unemployment}')
plt.grid(True)
plt.show()


# In[108]:


pre_ppa = d1[d1['year'] <= 2015][['year', 'country', 'ppa']]
post_ppa = d1[d1['year'] >= 2015][['year', 'country', 'ppa']]

plt.figure(figsize=(13, 6))
plt.title(r'\textbf{\% PPP (ppa) over time}')    
sns.regplot(data =pre_ppa[pre_ppa['country']    == 'IRL'], x ='year', y ='ppa', scatter=False, label=r'\textit{Ireland}', color='green', ci=0)
sns.regplot(data =post_ppa[post_ppa['country']  == 'IRL'], x ='year', y ='ppa', scatter =False, color ='green', ci=0)
sns.regplot(data =pre_ppa[pre_ppa['country']    == 'FIN'], x ='year', y ='ppa', scatter =False, label =r'\textit{Finland}', color     ='red', ci=0)
sns.regplot(data =post_ppa[post_ppa['country']  == 'FIN'], x ='year', y ='ppa', scatter =False, color ='red', ci=0)
sns.regplot(data =pre_ppa[pre_ppa['country']    == 'AUT'], x ='year', y ='ppa', scatter =False, label =r'\textit{Austria}', color     ='black', ci=0)
sns.regplot(data =post_ppa[post_ppa['country']  == 'AUT'], x ='year', y ='ppa', scatter =False, color ='black', ci=0)
plt.axvline(x=2015, color='red', linestyle='--', label=r'\textit{Treatment Year}')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6, fontsize='small')
plt.ylabel(r'\textit{\% PPP}')
plt.grid(True)
plt.show()


# In[103]:


pre_spe = d1[d1['year'] <= 2015][['year', 'country', 'public_spending']]
post_spe = d1[d1['year'] >= 2015][['year', 'country', 'public_spending']]

plt.figure(figsize=(13, 6))
plt.title(r'\textbf{\% public_spending over time}')    
sns.regplot(data=pre_spe[pre_spe['country'] == 'IRL'], x='year', y='public_spending',       scatter=False, label=r'\textit{Ireland}', color='green', ci=0)
sns.regplot(data =post_spe[post_spe['country']  == 'IRL'], x ='year', y ='public_spending', scatter =False, color ='green', ci=0)
sns.regplot(data =pre_spe[pre_spe['country']    == 'CHE'], x ='year', y ='public_spending', scatter =False, label =r'\textit{Switzerland}', color ='blue', ci=0)
sns.regplot(data =post_spe[post_spe['country']  == 'CHE'], x ='year', y ='public_spending', scatter =False, color ='blue', ci=0)
sns.regplot(data =pre_spe[pre_spe['country']    == 'SWE'], x ='year', y ='public_spending', scatter =False, label =r'\textit{Sweden', color       ='orange', ci=0)
sns.regplot(data =post_spe[post_spe['country']  == 'SWE'], x ='year', y ='public_spending', scatter =False, color ='orange', ci=0)
sns.regplot(data =pre_spe[pre_spe['country']    == 'FIN'], x ='year', y ='public_spending', scatter =False, label =r'\textit{Finland}', color     ='red', ci=0)
sns.regplot(data =post_spe[post_spe['country']  == 'FIN'], x ='year', y ='public_spending', scatter =False, color ='red', ci=0)
sns.regplot(data =pre_spe[pre_spe['country']    == 'AUT'], x ='year', y ='public_spending', scatter =False, label =r'\textit{Austria}', color     ='black', ci=0)
sns.regplot(data =post_spe[post_spe['country']  == 'AUT'], x ='year', y ='public_spending', scatter =False, color ='black', ci=0)
plt.axvline(x=2015, color='red', linestyle='--', label=r'\textit{Treatment Year}')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6, fontsize='small')
plt.ylabel(r'\textit{\% Public Spending}')
plt.grid(True)
plt.show()


# In[107]:


pre_ide = d1[d1['year'] <= 2015][['year', 'country', 'IDE']]
post_ide = d1[d1['year'] >= 2015][['year', 'country', 'IDE']]

plt.figure(figsize=(13, 6))
plt.title(r'\textbf{\% FDI (ide) over time}')    
sns.regplot(data =pre_ide[pre_ide['country']    == 'IRL'], x ='year', y ='IDE', scatter =False, label =r'\textit{Ireland}', color     ='green', ci=0)
sns.regplot(data =post_ide[post_ide['country']  == 'IRL'], x ='year', y ='IDE', scatter =False, color ='green', ci=0)
sns.regplot(data =pre_ide[pre_ide['country']    == 'FIN'], x ='year', y ='IDE', scatter =False, label =r'\textit{Finland}', color     ='red', ci=0)
sns.regplot(data =post_ide[post_ide['country']  == 'FIN'], x ='year', y ='IDE', scatter =False, color ='red', ci=0)
sns.regplot(data =pre_ide[pre_ide['country']    == 'AUT'], x ='year', y ='IDE', scatter =False, label =r'\textit{Austria}', color     ='black', ci=0)
sns.regplot(data =post_ide[post_ide['country']  == 'AUT'], x ='year', y ='IDE', scatter =False, color ='black', ci=0)
plt.axvline(x=2015, color='red', linestyle='--', label=r'\textit{Treatment Year}')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6, fontsize='small')
plt.ylabel(r'\textit{\% FDI}')
plt.grid(True)
plt.show()


# In[ ]:


pre_dep = d2[d2['year'] <= 2015][['year', 'country', 'depreciation']]
post_dep = d2[d2['year'] >= 2015][['year', 'country', 'depreciation']]

plt.figure(figsize=(13, 6))
plt.title(r'\textbf{\% depreciation (PI assets) over time}')    
sns.regplot(data =pre_dep[pre_dep['country']    == 'IRL'], x ='year', y ='depreciation', scatter =False, label =r'\textit{Ireland}', color     ='green', ci=0)
sns.regplot(data =post_dep[post_dep['country']  == 'IRL'], x ='year', y ='depreciation', scatter =False, color ='green', ci=0)
sns.regplot(data =pre_dep[pre_dep['country']    == 'FIN'], x ='year', y ='depreciation', scatter =False, label =r'\textit{Finland}', color     ='red', ci=0)
sns.regplot(data =post_dep[post_dep['country']  == 'FIN'], x ='year', y ='depreciation', scatter =False, color ='red', ci=0)
sns.regplot(data =pre_dep[pre_dep['country']    == 'AUT'], x ='year', y ='depreciation', scatter =False, label =r'\textit{Austria}', color     ='black', ci=0)
sns.regplot(data =post_dep[post_dep['country']  == 'AUT'], x ='year', y ='depreciation', scatter =False, color ='black', ci=0)
plt.axvline(x=2015, color='red', linestyle='--', label=r'\textit{Treatment Year}')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6, fontsize='small')
plt.ylabel(r'\textit{\% Depreciation}')
plt.grid(True)
plt.show()

