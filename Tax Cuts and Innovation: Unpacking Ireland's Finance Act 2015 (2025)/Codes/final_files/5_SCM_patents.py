#!/usr/bin/env python
## !!!! Line 84 
## !!!! To have the right graph change years (2015 - 2014) and (2016 - 2015) 
## !!!! -> miss match on the visual treatement year in the package -> correction should be done
# coding: utf-8

# In[ ]:


import warnings

warnings.filterwarnings("ignore")

import os  
import sys
script_dir = os.path.dirname(__file__)
parent =  os.path.join(script_dir, 'package', 'synthdid')
dir = os.path.dirname(parent)
sys.path.append(dir)

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

from tqdm import tqdm

from synthdid.model import SynthDID

path = os.path.join(script_dir, 'final_clean')
path2 = os.path.join(script_dir, 'final_LaTeX')

pd.set_option('display.max_columns', None)


# In[4]:


df = pd.read_csv(path + "/patents_dataset.csv")
df.head()


# In[6]:


df = df[['year', 'country', 'brevets', 'gdp', 'pop', 'NX', 'public_spending']]


# In[7]:


df['country_code'] = pd.factorize(df['country'])[0] 

codes_df = df.copy()

def code(codes=codes_df) -> None:
    """
    Print country codes for AUT, FIN, and IRL, and list unique countries.
    """
    print("AUT code is = ", df[df['country'] == 'AUT']['country_code'].head(1).to_list(), "\n")
    print("FIN code is = ", df[df['country'] == 'FIN']['country_code'].head(1).to_list(), "\n")
    print("IRL code is = ", df[df['country'] == 'IRL']['country_code'].head(1).to_list(), "\n")
    print("CHE code is = ", df[df['country'] == 'CHE']['country_code'].head(1).to_list(), "\n")
    print("SWE code is = ", df[df['country'] == 'SWE']['country_code'].head(1).to_list(), "\n")
    
code()


# In[8]:


df =df.rename(columns={'country_code': 'state'})
df


# In[9]:


df_ = df.pivot(index='state', columns="year")['brevets'].T
df_.head()


# In[10]:


PRE_TEREM = [1999, 2015]  ## For the graphs change year 2015 -> 2014 and 2016 to 2015 -> miss match on the visual treatement year in the package -> correction must be done
POST_TEREM = [2016, 2019]
TREATMENT = [2]  # "Treated country

sdid = SynthDID(df_, PRE_TEREM, POST_TEREM, TREATMENT)


# In[11]:


sdid.fit(zeta_type="base", sparce_estimation=True)
hat_omega_simple = sdid.estimated_params(model="sc")
# hat_omega_elastic = sdid.estimated_params(model="ElasticNet")


# In[14]:


sdid.plot(model="sc", title='Log Patents')
# sdid.plot(model="did")

summary = sdid.summary(model="sc")
print(summary)


# In[15]:


predicted = sdid.sdid_trajectory()
observed = sdid.target_y()
rmspe = ((observed - predicted) ** 2).mean() ** 0.5
print(f"RMSPE : {rmspe}")


# In[16]:


# hat_omega_sdid, _ = sdid.estimated_params()
# print(hat_omega_sdid)
# Appeler la méthode comparison_plot
# sdid.comparison_plot(model="all", figsize=(10, 7))
weights_scm = sdid.hat_omega_ADH
weights_scm_df = pd.DataFrame(weights_scm, columns=['weights'])
weights_scm_df['weights'] = weights_scm_df['weights'].apply(lambda x: f"{x:.2e}")
# weights_scm_df.T.to_latex("~/Latex/sdid/Brevets_weights_scm.tex")
# print("\n", "weights_scm", weights_scm)
# print(df_)


# In[21]:


weights_sc = sdid.hat_omega_ADH  # Poids SC (Synthetic Control)
control_units = sdid.control     # Liste des colonnes des unités de contrôle

weights_sc_dict = {unit: weight for unit, weight in zip(control_units, weights_sc)}

weights_sc_df = pd.DataFrame({'state': control_units, 'weight_sc': weights_sc})
# print(weights_sc_df)


# In[19]:


tau_hat = sdid.hat_tau(model="sc")

b = {
    "RMSPE" : rmspe,
    "tau_hat" : tau_hat
}
b = pd.DataFrame(b, index=[0])

weights_sc_df.iloc[0,0:1] = 'AUT'
weights_sc_df.iloc[1,0:1] = 'FIN'
weights_sc_df.iloc[2,0:1] = 'CHE'
weights_sc_df.iloc[3,0:1] = 'SWE'

print(weights_sc_df)


# In[ ]:


# b.to_latex(path2 + "/tau_hat_patents.tex", float_format="%.2e")
# weights_sc_df.to_latex(path2 + "/weights_patents.tex", float_format="%.2e")

