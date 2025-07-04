import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt     
import matplotlib.dates as mdates
import seaborn as sns   
from datetime import datetime
from scipy import stats
from scipy.stats import stats, skew, kurtosis
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import coint
sns.set_theme(style="whitegrid", context="talk")
plt.style.use('seaborn-v0_8-darkgrid')



df = pd.ExcelFile('Data.xlsx') 
df = df.parse('OilandGas')



df = df.iloc[1:].reset_index(drop=True)

df['Name'] = pd.to_datetime(df['Name'])
df.set_index('Name', inplace=True)
df.index.name = None


df = df.apply(pd.to_numeric, errors='coerce')

df.rename(columns={
    'EXXON MOBIL (~U$) (#T) - EXXON MOBIL (~U$) (#T)': 'EXXON_MOBIL',
    'CONOCOPHILLIPS (~U$) (#T) - CONOCOPHILLIPS (~U$) (#T)': 'CONOCOPHILLIPS',
    'CHEVRON (~U$) (#T) - CHEVRON (~U$) (#T)': 'CHEVRON',
    'TOTALENERGIES (LON) (~U$) (#T) - TOTALENERGIES (LON) (~U$) (#T)': 'TOTALENERGIES',
    'OCCIDENTAL PTL. (~U$) (#T) - OCCIDENTAL PTL. (~U$) (#T)': 'OCCIDENTAL_PTL'
}, inplace=True)



if df.isnull().sum().sum() == 0:
    print('There are no missing values in the dataset')



weekly_prices = df.resample('W-MON').first()



weekly_simple       = weekly_prices.pct_change(fill_method=None).dropna()
weekly_log          = np.log(weekly_prices / weekly_prices.shift(1)).dropna()

daily_simple        = df.pct_change(fill_method=None).dropna()
daily_log           = np.log(df / df.shift(1)).dropna()





def compute_stats(returns, annual_factor):
    
    stats = pd.DataFrame(index=returns.columns,
                         columns=['annualized_mean', 'annualized_variance', 'skewness', 'kurtosis', 'min', 'max'])
    for col in returns.columns:
        
        mean_annual=returns[col].mean() * annual_factor
        
        var_annual = returns[col].var() * annual_factor
        
        skewness_val = skew(returns[col].dropna())
        kurtosis_val = kurtosis(returns[col].dropna(), fisher=False)
        
        min_val = returns[col].min()
        max_val = returns[col].max()
        stats.loc[col] = [mean_annual, var_annual, skewness_val, kurtosis_val, min_val, max_val]
    return stats



trading_days    = 252 
weeks_per_year  = 52  



daily_stats_simple  = compute_stats(daily_simple, trading_days)
daily_stats_log     = compute_stats(daily_log, trading_days)



weekly_stats_log     = compute_stats(weekly_log, weeks_per_year)
weekly_stats_simple  = compute_stats(weekly_simple, weeks_per_year)



print('\n', "# ================================== Q 1.1 - 1.2 ======================================= #", '\n')


comparison_daily = pd.concat([daily_stats_simple, daily_stats_log], axis=1, keys=['Daily Simple Returns', 'Daily Log Returns']).apply(pd.to_numeric).round(4).T





comparison_weekly = pd.concat([weekly_stats_simple, weekly_stats_log], axis=1, keys=['Weekly Simple Returns', 'Weekly Log Returns']).apply(pd.to_numeric).round(4).T






concat_ = pd.concat([comparison_daily, comparison_weekly], axis=0)
print(concat_)



print('\n', "# ================================== Q 2.1.1 - 2.1.4 ======================================= #", '\n')



T = df.shape[0]


N = 10000

np.random.seed(123)  




epsilon_rep = np.random.normal(loc=0, scale=1, size=(N, T))







p_sim = np.cumsum(epsilon_rep, axis=1)









t_stats = np.empty(N)  

for i in range(N):
    
    series = p_sim[i, :]
    
    
    y = series[1:]
    
    
    X = series[:-1]
    
    
    X = sm.add_constant(X)
    
    
    model = sm.OLS(y, X)
    results = model.fit()
    
    
    phi_hat = results.params[1]
    phi_se = results.bse[1]
    
    
    
    
    
    t_stat = (phi_hat - 1) / phi_se  
    t_stats[i] = t_stat



crit_10 = np.percentile(t_stats, 10)  
crit_5  = np.percentile(t_stats, 5)   
crit_1  = np.percentile(t_stats, 1)   

print("Critical value at 10%:", np.round(crit_10, 4))
print("Critical value at 5% :", np.round(crit_5, 4))
print("Critical value at 1% :", np.round(crit_1, 4))


print('\n', "# ================================== Q 2.1.5 - 2.1.7 ======================================= #", '\n')

fig, ax = plt.subplots(figsize=(12, 7))
sns.histplot(t_stats, bins=50, kde=True, color="teal", edgecolor="black", ax=ax, linewidth=1.2)
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax.axhline(y=0, color='lightgray', linestyle='--', linewidth=2)  
ax.set_title(r"Distribution of $t(\hat{\phi}-1)$ under $H_0$: $\phi=1$", fontsize=16, pad=15)
ax.set_xlabel(r"$t(\hat{\phi}-1)$", fontsize=14)
ax.set_ylabel(r"$\mathrm{Frequency}$", fontsize=14)
ax.tick_params(axis='both', labelsize=12)
plt.tight_layout()
plt.show()





crit_10 = np.percentile(t_stats, 10)
crit_5  = np.percentile(t_stats, 5)
crit_1  = np.percentile(t_stats, 1)

print("Critical value at 10% level:", np.round(crit_10, 4))
print("Critical value at 5% level :", np.round(crit_5, 4))
print("Critical value at 1% level :", np.round(crit_1, 4))

crit_to_latex = pd.DataFrame({
    '10%': np.round(crit_10, 4),
    '5%': np.round(crit_5, 4),
    '1%': np.round(crit_1, 4)
}, index=['Critical Value'])




phi_true = 0.2
epsilon_rep_alt = np.random.normal(loc=0, scale=1, size=(N, T))
p_sim_alt = np.empty((N, T))
p_sim_alt[:, 0] = 0  


for t in range(1, T):
    p_sim_alt[:, t] = phi_true * p_sim_alt[:, t-1] + epsilon_rep_alt[:, t]


t_stats_alt = np.empty(N)
for i in range(N):
    series_alt = p_sim_alt[i, :]
    y_alt = series_alt[1:]
    X_alt = series_alt[:-1]
    X_alt = sm.add_constant(X_alt)
    model_alt = sm.OLS(y_alt, X_alt).fit()
    phi_hat_alt = model_alt.params[1]
    phi_se_alt = model_alt.bse[1]
    t_stats_alt[i] = (phi_hat_alt - 1) / phi_se_alt


fig, ax = plt.subplots(figsize=(12, 7))
sns.histplot(t_stats_alt, bins=50, kde=True, color="salmon", ax=ax, edgecolor='black', linewidth=1.2)
ax.set_title(r"Distribution of $t(\hat{\phi}-1)$ under AR(1) with $\phi=0.2$", fontsize=16, pad=15)
ax.set_xlabel(r"$t(\hat{\phi}-1)$", fontsize=14)
ax.set_ylabel("Frequency", fontsize=14)
ax.tick_params(axis='both', labelsize=12)
plt.tight_layout()
plt.show()






print('\n', "# ================================== End Section 2.1 ======================================= #", '\n')

T500 = 500         
N500 = 10000       

np.random.seed(123)  

epsilon_rep_500 = np.random.normal(loc=0, scale=1, size=(N500, T500))


p_sim_500 = np.cumsum(epsilon_rep_500, axis=1)  


t_stats_500 = np.empty(N500)
for i in range(N500):
    series = p_sim_500[i, :]
    y = series[1:]                
    X = series[:-1]               
    X = sm.add_constant(X)        
    model = sm.OLS(y, X).fit()    
    phi_hat = model.params[1]     
    phi_se = model.bse[1]         
    t_stats_500[i] = (phi_hat - 1) / phi_se  

crit_10_500 = np.percentile(t_stats_500, 10)
crit_5_500  = np.percentile(t_stats_500, 5)
crit_1_500  = np.percentile(t_stats_500, 1)


crit_df_500 = pd.DataFrame({
    '10%': [np.round(crit_10_500, 4)],
    '5%': [np.round(crit_5_500, 4)],
    '1%': [np.round(crit_1_500, 4)]
}, index=['Critical Value for T=500'])




print("Critical value at 10% level for T = 500:", np.round(crit_10_500, 4))
print("Critical value at 5% level for T = 500 :", np.round(crit_5_500, 4))
print("Critical value at 1% level for T = 500 :", np.round(crit_1_500, 4))


fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(t_stats_500, bins=50, kde=True, color="steelblue", edgecolor="black", linewidth=1.2, ax=ax)
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax.set_title(r"Distribution of $t(\hat{\phi}-1)$ for $T=500$", fontsize=16, pad=15)
ax.set_xlabel(r"$t(\hat{\phi}-1)$", fontsize=14)
ax.set_ylabel("Frequency", fontsize=14)
ax.tick_params(axis='both', labelsize=12)
plt.tight_layout()
plt.show()




print('\n', "# ================================== Q 2.8 - 2.9 ======================================= #", '\n')




results_list_10 = []
results_list_5  = []
results_list_1  = []


for asset in df.columns:
    
    p = np.log(df[asset].dropna().values)
    
    
    y = p[1:]           
    X = p[:-1]          
    X = sm.add_constant(X)  
    model = sm.OLS(y, X).fit()
    
    
    phi_hat = model.params[1]
    phi_se = model.bse[1]
    
    
    DF_stat = (phi_hat - 1) / phi_se
    
    
    p_value = np.mean(t_stats < DF_stat)
    
    
    
    if DF_stat < crit_10:
        conclusion_10 = "Reject $H_0$ : Series is stationary"
    else:
        conclusion_10 = "Fail to reject $H_0$ : Unit root present"
    
    
    if DF_stat < crit_5:
        conclusion_5 = "Reject $H_0$ : Series is stationary"
    else:
        conclusion_5 = "Fail to reject $H_0$ : Unit root present"
    
    
    if DF_stat < crit_1:
        conclusion_1 = "Reject $H_0$ : Series is stationary"
    else:
        conclusion_1 = "Fail to reject $H_0$ : Unit root present"
    
    
    results_list_10.append({
        "Asset": asset,
        "DF_statistic": DF_stat,
        "Critical_value 10%": crit_10,
        "p_value": p_value,
        "Conclusion": conclusion_10
    })
    
    results_list_5.append({
        "Asset": asset,
        "DF_statistic": DF_stat,
        "Critical_value 5%": crit_5,
        "p_value": p_value,
        "Conclusion": conclusion_5
    })
    
    results_list_1.append({
        "Asset": asset,
        "DF_statistic": DF_stat,
        "Critical_value 1%": crit_1,
        "p_value": p_value,
        "Conclusion": conclusion_1
    })


results_df_10 = pd.DataFrame(results_list_10)
results_df_5  = pd.DataFrame(results_list_5)
results_df_1  = pd.DataFrame(results_list_1)

results_df_10.set_index('Asset', inplace=True)
results_df_5.set_index('Asset', inplace=True)
results_df_1.set_index('Asset', inplace=True)

results_df_10.index.name = None
results_df_5.index.name = None
results_df_1.index.name = None









print(results_df_10, '\n')
print(results_df_5, '\n')
print(results_df_1, '\n')





print('\n', "# ================================== Subection 3.1 ======================================= #", '\n')

T_cointegration = 500
N_cointegration = 10000

np.random.seed(123)  

epsilon_A = np.random.normal(loc=0, scale=1, size=(N_cointegration, T_cointegration))
epsilon_B = np.random.normal(loc=0, scale=1, size=(N_cointegration, T_cointegration))


pA_sim = np.cumsum(epsilon_A, axis=1)
pB_sim = np.cumsum(epsilon_B, axis=1)


  
t_stats_cointegration = np.empty(N_cointegration)

for i in range(N_cointegration):
    
    pA = pA_sim[i, :]
    pB = pB_sim[i, :]
    
    
    X = sm.add_constant(pB)
    cointegration_model = sm.OLS(pA, X).fit()
    z = cointegration_model.resid  
    
    
    dz = np.diff(z)            
    z_lag = z[:-1]             
    X_ar = sm.add_constant(z_lag)
    ar_model = sm.OLS(dz, X_ar).fit()
    
    phi_hat = ar_model.params[1]   
    phi_se = ar_model.bse[1]         
    
    t_stats_cointegration[i] = phi_hat / phi_se


crit_10_coin = np.percentile(t_stats_cointegration, 10)
crit_5_coin = np.percentile(t_stats_cointegration, 5)
crit_1_coin = np.percentile(t_stats_cointegration, 1)

critical_coin = pd.DataFrame({
    '10%': np.round(crit_10_coin, 4),
    '5%': np.round(crit_5_coin, 4),
    '1%': np.round(crit_1_coin, 4)
}, index=['Critical Values for T=500'])





print("Cointegration DF Test Critical Values for T = 500:")
print("10% level:", np.round(crit_10_coin, 4))
print("5% level :", np.round(crit_5_coin, 4))
print("1% level :", np.round(crit_1_coin, 4))


print('\n', "# ================================== Q 3.1 ======================================= #", '\n')

fig, ax = plt.subplots(figsize=(12, 7))
sns.histplot(t_stats_cointegration, bins=50, kde=True, color="mediumseagreen", ax=ax, edgecolor="black", linewidth=1.2)
ax.set_title(r"Distribution of $t(\hat{\phi})$ from the cointegration DF test for $T=500$", fontsize=16, pad=15)
ax.set_xlabel(r"$t(\hat{\phi})$", fontsize=14)
ax.set_ylabel("Frequency", fontsize=14)
ax.tick_params(axis="both", labelsize=12)
ax.axhline(y=0, color='lightgray', linestyle='--', linewidth=2)  
plt.tight_layout()
plt.show()




print('\n', "# ================================== Subsection 3.2 ======================================= #", '\n')




prices_ = df


assets = [asset for asset in df.columns if asset != "TOTALENERGIES"]





def cointegration_test(asset_A, asset_B, prices_, sim_tstats, log_price=True):
    """
    Performs a cointegration test between asset_A and asset_B.
    
    Parameters:
      - asset_A, asset_B: column names in prices_.
      - prices_: DataFrame containing raw price series.
      - sim_tstats: simulated distribution of t-statistics to compare with the obtained statistic.
      - log_price: boolean, if True, uses log-prices; otherwise, uses raw prices.
    
    Steps:
      1. Extract and, if necessary, convert the series (using log if log_price=True) and align them.
      2. Estimate the cointegration regression:
         pA_t = α + β * pB_t + z_t
      3. On the residuals z_t, estimate an AR(1) model in differences:
         Δz_t = μ + φ * z_{t-1} + ε_t,
       and compute the t-statistic to test H₀: φ = 0 (unit root in z_t).
      4. Compute the p-value by comparing with the simulated distribution sim_tstats.
    
    Returns a dictionary containing:
      - Alpha, Beta
      - DF Statistic (t_stat)
      - p_value
      - Conclusion
    """
    
    
    if log_price:
        series_A = np.log(prices_[asset_A].dropna().values)
        series_B = np.log(prices_[asset_B].dropna().values)
    else:
        series_A = prices_[asset_A].dropna().values
        series_B = prices_[asset_B].dropna().values
        
    
    T = min(len(series_A), len(series_B))
    pA = series_A[:T]
    pB = series_B[:T]

    
    X = sm.add_constant(pB)
    cointegration_model = sm.OLS(pA, X).fit()
    alpha_hat = cointegration_model.params[0]
    beta_hat  = cointegration_model.params[1]
    z = cointegration_model.resid  

    
    dz        = np.diff(z) 
    z_lag     = z[:-1] 
    X_ar      = sm.add_constant(z_lag)
    ar_model  = sm.OLS(dz, X_ar).fit()
    phi_hat   = ar_model.params[1] 
    phi_se    = ar_model.bse[1] 
    t_stat    = phi_hat / phi_se 

    
    p_value = np.mean(sim_tstats < t_stat)

    
    crit_5 = np.percentile(sim_tstats, 5)
    if t_stat < crit_5:
        conclusion = "Reject H₀ (cointegration)"
    else:
        conclusion = "Fail to reject H₀ (no cointegration)"

    return {
        "Asset_A"      : asset_A,
        "Asset_B"      : asset_B,
        "Direction"    : f"{asset_A} on {asset_B}",
        "Alpha"        : alpha_hat,
        "Beta"         : beta_hat,
        "DF_Statistic" : t_stat,
        "Critical_5%"  : crit_5,
        "p_value"      : p_value,
        "Conclusion"   : conclusion
    }




results = []
for i in range(len(assets)):
    for j in range(len(assets)):
        if i != j:
            test_res = cointegration_test(assets[i], assets[j], prices_, t_stats_cointegration, log_price=True)
            results.append(test_res)


results_df = pd.DataFrame(results)
results_df_sorted = results_df.sort_values(by="p_value").reset_index(drop=True)



print('\n', "# ================================== Q 3.2 - 3.3 - 3.4 ======================================= #", '\n')

print("\nCointegration Test Results (Sorted by p-value):")
print(results_df_sorted)




strongest = results_df_sorted.iloc[0]
print("\nMost strongly cointegrated pair (lowest p-value):")
print(strongest)


print("\nThis pair has the lowest p-value, suggesting a long-term relationship (cointegration).\n")


asset_A = strongest["Asset_A"]
asset_B = strongest["Asset_B"]


print('\n', "# ================================== Q 3.5 ======================================= #", '\n')
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

sns.lineplot(data=df[asset_A], label=asset_A, color="teal", lw=0.9)
sns.lineplot(data=df[asset_B], label=asset_B, color="salmon", lw=0.9)

plt.xlabel("Time", fontsize=12)
plt.ylabel("Price", fontsize=12)
plt.title(f"Price Series of {asset_A} and {asset_B}", fontsize=16, pad=15)


plt.legend(title="Assets", fontsize=12, title_fontsize=12, loc='best')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(title='Assets',fontsize=12, title_fontsize=12, loc='best')
plt.tight_layout()
plt.show()




print('\n', "# ================================== Section 4 ======================================= #", '\n')
 

strongest_raw_prices = [cointegration_test(strongest['Asset_A'], strongest['Asset_B'], df, t_stats_cointegration, log_price=False),
                        cointegration_test(strongest['Asset_B'], strongest['Asset_A'], df, t_stats_cointegration, log_price=False)]
strongest_raw_results = pd.DataFrame(strongest_raw_prices)
strongest_raw_results = strongest_raw_results.sort_values(by="p_value").reset_index(drop=True)

print('Pair Winners by log(P_t) tested in raw P_t results :\n', strongest_raw_results)

alpha_hat = strongest_raw_results.loc[0, 'Alpha']   
beta_hat  = strongest_raw_results.loc[0, 'Beta']    




print('\n', "# ================================== Q 4.2 ======================================= #", '\n')

spread = df["CHEVRON"] - alpha_hat - beta_hat * df["CONOCOPHILLIPS"]


spread_normalized = spread / spread.std()


plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid", context="talk")
sns.lineplot(data=spread_normalized, label=r'$\tilde{z}_t$', lw=0.8, color='purple')
plt.axhline(0, color='black', linestyle='--', lw=0.8)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Normalized Spread", fontsize=12)
plt.title("Normalized Spread Signal for the Pair: CHEVRON and CONOCOPHILLIPS", fontsize=15, pad=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()



print('\n', "# ================================== Q 4.3 ======================================= #", '\n')






lags = range(1, 11)  
acf = sm.tsa.acf(spread_normalized, nlags=10)  


acf_df = pd.DataFrame({'Lag': range(0, 11), 'Autocorrelation': acf})


plt.figure(figsize=(12, 6))
sns.barplot(x='Lag', y='Autocorrelation', data=acf_df, color='royalblue', alpha=0.8)
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8)


n = len(spread_normalized)
conf_level = 1.96/np.sqrt(n)
plt.axhline(y=conf_level, color='red', linestyle='-', alpha=0.7, 
                     label=f'95% Confidence Band (±{conf_level:.4f})')
plt.axhline(y=-conf_level, color='red', linestyle='-', alpha=0.7)

plt.title(r"Autocorrelation Function of Normalized Spread $\tilde{z}_t$ (up to 10 lags)", fontsize=14)
plt.xlabel("Lag", fontsize=12)
plt.ylabel("Autocorrelation", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(fontsize=10, loc='best')
plt.tight_layout()
plt.show()


acf_df = acf_df.set_index('Lag', drop=True)

print("Autocorrelations of normalized spread:")
print(acf_df.T)




lb_test = acorr_ljungbox(spread_normalized, lags=[10], return_df=True)

print('\n\n')
print("Ljung-Box test results (10 lags):")
print(lb_test)



print('\n', "# ================================== AUDIT Stationarity of TotalEnergies ======================================= #", '\n')


def adf_test_all_columns(dataframe, use_log=True):
    """
    Perform Augmented Dickey-Fuller test on all columns in the dataframe.
    
    Parameters:
    -----------
    dataframe : pandas.DataFrame
        DataFrame containing time series data
    use_log : bool, default=True
        If True, applies natural log transformation before testing
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with ADF test results for each column
    """
    results = {}
    
    for column in dataframe.columns:
        
        series = dataframe[column].dropna()
        
        
        if use_log:
            
            if (series <= 0).any():
                print(f"Warning: {column} contains non-positive values, skipping log transformation")
                test_series = series
            else:
                test_series = np.log(series)
        else:
            test_series = series
        
        
        result = adfuller(test_series, autolag=None, maxlag=0) 
        
        
        results[column] = {
            'ADF Statistic': result[0],
            'p-value': result[1],
            'Used Lags': result[2],
            'Number of Observations': result[3],
            'Critical Value (1%)': result[4]['1%'],
            'Critical Value (5%)': result[4]['5%'], 
            'Critical Value (10%)': result[4]['10%']
        }
    
    
    results_df = pd.DataFrame(results).T
    
    
    results_df['Conclusion'] = results_df.apply(
        lambda x: 'Stationary' if x['p-value'] < 0.05 else 'Non-stationary', 
        axis=1
    )
    
    return results_df


adf_results = adf_test_all_columns(df).sort_values(by='p-value')

print('AUDIT: Confirmation of the exclusion of TOTALENERGIES because of stationarity :')

print(adf_results)



print('\n', "# ================================== AUDIT Q 3.5 ======================================= #", '\n')



def test_cointegration_all_pairs(data):
    """
    Test cointegration for all possible pairs of columns in the input dataframe.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing price series (columns are assets, rows are time periods)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with results of cointegration tests for each pair of assets
    """
    n = len(data.columns)
    asset_names = data.columns
    results = []
    
    
    for i in range(n):
        for j in range(i+1, n):  
            asset1 = asset_names[i]
            asset2 = asset_names[j]
            
            
            series1 = data[asset1].dropna()
            series2 = data[asset2].dropna()
            
            
            common_idx = series1.index.intersection(series2.index)
            y1 = series1.loc[common_idx]
            y2 = series2.loc[common_idx]
            
            
            score, pvalue, _ = coint(y1, y2)
            
            
            results.append({
                'Asset1': asset1,
                'Asset2': asset2,
                'Test Statistic': score,
                'p-value': pvalue,
                'Conclusion': 'Cointegrated' if pvalue < 0.05 else 'Not cointegrated'
            })
            
            
            score_rev, pvalue_rev, _ = coint(y2, y1, trend='c', maxlag=0) 
            
            results.append({
                'Asset1': asset2,
                'Asset2': asset1,
                'Test Statistic': score_rev,
                'p-value': pvalue_rev,
                'Conclusion': 'Cointegrated' if pvalue_rev < 0.05 else 'Not cointegrated'
            })
    
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('p-value').reset_index(drop=True)
    
    return results_df



audit_coin  = np.log(df[assets].dropna())
results_coin_test = test_cointegration_all_pairs(audit_coin)

print('AUDIT: Confirmation of our selection for the winning pair CHEVRON -> CONOCOPHILLIPS are the best: ')

print(results_coin_test)



print('\n', "# ================================== AUDIT Q 4.3 ======================================= #", '\n')



z = spread.values  
dz = np.diff(z)    


z_lag = z[:-1]     


T_actual = len(z)
print(f"Actual length of spread: {T_actual}")


np.random.seed(123)  
n_simulations = 10000
t_stats_correct_size = np.zeros(n_simulations)

for i in range(n_simulations):
    
    e = np.random.normal(0, 1, T_actual)
    y = np.cumsum(e)
    
    
    dy = np.diff(y)
    y_lag = y[:-1]
    X = sm.add_constant(y_lag)
    
    model_sim = sm.OLS(dy, X).fit()
    phi_sim = model_sim.params[1]
    se_sim = model_sim.bse[1]
    t_stats_correct_size[i] = phi_sim / se_sim


X_ar = sm.add_constant(z_lag)


model = sm.OLS(dz, X_ar).fit()
phi_hat = model.params[1]       
phi_se = model.bse[1]           
t_stat = phi_hat / phi_se       


p_value = np.mean(t_stats_correct_size < t_stat)  


crit_10 = np.percentile(t_stats_correct_size, 10)
crit_5  = np.percentile(t_stats_correct_size, 5)
crit_1  = np.percentile(t_stats_correct_size, 1)


def decision(ts, crit):
    return "Reject H₀ (Stationary)" if ts < crit else "Fail to reject H₀ (Non-stationary)"

conclusion_10 = decision(t_stat, crit_10)
conclusion_5  = decision(t_stat, crit_5)
conclusion_1  = decision(t_stat, crit_1)


results_data = {
    "DF_Statistic": [t_stat],
    "Critical_10%": [crit_10],
    "Conclusion_10%": [conclusion_10],
    "Critical_5%": [crit_5],
    "Conclusion_5%": [conclusion_5],
    "Critical_1%": [crit_1],
    "Conclusion_1%": [conclusion_1],
    "p_value": [p_value],
    "Sample_Size": [T_actual]
}
results_test_spread_1 = pd.DataFrame(results_data)

print(results_test_spread_1)



print('\n', "# ================================== AUDIT Q 4.3 ======================================= #", '\n')

result_st_2 = adfuller(spread, maxlag=0)
adf_stat, p_value, usedlag, nobs, crit_values, icbest = result_st_2


def decision(stat, crit):
    return "Reject H₀ (Stationary)" if stat < crit else "Fail to reject H₀ (Non-stationary)"


results_data = {
    "ADF_Statistic": [adf_stat],
    "Critical_10%": [crit_values['10%']],
    "Conclusion_10%": [decision(adf_stat, crit_values['10%'])],
    "Critical_5%": [crit_values['5%']],
    "Conclusion_5%": [decision(adf_stat, crit_values['5%'])],
    "Critical_1%": [crit_values['1%']],
    "Conclusion_1%": [decision(adf_stat, crit_values['1%'])],
    "p_value": [p_value]
}
results_spread_test_df = pd.DataFrame(results_data)

print(results_spread_test_df)




print('\n', "# ================================== Complmentary Analysis ======================================= #", '\n')









spread_mean = np.mean(spread)
spread_var = np.var(spread, ddof=0)  

spread_ = spread.copy()
spread_ = spread.asfreq('B')  

model = sm.tsa.AutoReg(spread_.dropna(), lags=1).fit()
phi = model.params.iloc[1]  


half_life = -np.log(2) / np.log(phi) if phi > 0 else np.inf




horizon = int(5 * half_life)  
irf = [phi**h for h in range(horizon)]



ar1_summary = pd.DataFrame({
    'Value': [
        f"{spread.count()}",
        f"{spread_mean:.4f}",
        f"{spread_var:.4f}",
        f"{phi:.4f}",
        f"{half_life:.1f} days"
    ]
}, index=['Number of observations', 'Mean of spread', 'Variance of spread', 'Phi coefficient (AR1)', 'Half-life'])


print("\n====== AR(1) Model Summary ======")
print(ar1_summary)







plt.figure(figsize=(12, 6))
plt.plot(irf, color='darkred', lw=2, label=r'IRF ($\phi = {:.4f}$)'.format(phi))
plt.axhline(0.5, ls='--', color='grey', label='50% of initial shock')
plt.title(r"Impulse Response Function (IRF) - AR(1) Model" + "\n" + 
          r"Half-life = {:.1f} days".format(half_life), fontsize=14, pad=20)
plt.xlabel("Horizon (days)", fontsize=12)
plt.ylabel("Residual shock amplitude", fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)

plt.show()



print('\n', "# ================================== Q 4.4 ======================================= #", '\n')
 




z_in = 1.0





spread = df["CHEVRON"] - alpha_hat - beta_hat * df["CONOCOPHILLIPS"]


spread_normalized = spread / spread.std()












signals = []  


for t in df.index:
    
    zt_norm = spread_normalized.loc[t]
    price_A = df.loc[t, "CHEVRON"]
    price_B = df.loc[t, "CONOCOPHILLIPS"]
    
    
    if zt_norm > z_in:
        
        port_val = -price_A + beta_hat * price_B
        signals.append({"Date": t, "Signal": "Signal 1", "Spread_normalized": zt_norm, "Portfolio_Value": port_val})
        
    
    elif zt_norm < -z_in:
        
        port_val = price_A - beta_hat * price_B
        signals.append({"Date": t, "Signal": "Signal 2", "Spread_normalized": zt_norm, "Portfolio_Value": port_val})


signals_df = pd.DataFrame(signals)
signals_df.sort_values(by="Date", inplace=True)
signals_df.reset_index(drop=True, inplace=True)











plt.figure(figsize=(12, 6))
sns.lineplot(x=spread_normalized.index, 
             y=spread_normalized.values, 
             label="Normalized Spread", 
             color="purple", 
             linewidth=0.8)

plt.axhline(z_in, color="red", linestyle="--", label=r"$\tilde{z}^{in}$")
plt.axhline(-z_in, color="red", linestyle="--")


signal1_dates = []
signal1_values = []
signal2_dates = []
signal2_values = []

for idx, row in signals_df.iterrows():
    if row["Signal"] == "Signal 1":
        signal1_dates.append(row["Date"])
        signal1_values.append(row["Spread_normalized"])
    elif row["Signal"] == "Signal 2":
        signal2_dates.append(row["Date"])
        signal2_values.append(row["Spread_normalized"])


plt.scatter(signal1_dates, signal1_values, 
            color="blue", 
            marker="v", 
            s=20, 
            label="Signal 1 (Short Spread)", 
            alpha=0.7)

plt.scatter(signal2_dates, signal2_values, 
            color="green", 
            marker="^", 
            s=20, 
            label="Signal 2 (Long Spread)", 
            alpha=0.7)

plt.xlabel("Date", fontsize=10)
plt.ylabel(r"Normalized Spread $\tilde{z}_t$", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.title("Normalized Spread Series with Entry Signals", fontsize=14)
plt.legend(fontsize=10, loc='best')
plt.tight_layout()
plt.show()




print('\n', "# ================================== Q 4.5 ======================================= #", '\n')


z_in = 1.0  


entry_indices = spread_normalized[spread_normalized > z_in].index
if len(entry_indices) > 0:
    t_entry = entry_indices[0]
    
    z_entry = spread.loc[t_entry]  
    
    
    exit_candidates = spread_normalized.loc[t_entry:][spread_normalized <= 0].index
    if len(exit_candidates) > 0:
        t_exit = exit_candidates[0]
        
        profit = z_entry - 0  
        print("Entry Time:", t_entry)
        print("Exit Time:", t_exit)
        print("Profit (in levels):", np.round(profit,2))
        print("Expected profit = z_in * std(spread):", np.round(z_in * spread.std(),2))
    else:
        print("No exit signal found after entry.")
else:
    print("No entry signal found (spread_normalized never exceeds z_in).")



exit_position = spread_normalized.index.get_loc(t_exit)

previous_day = spread_normalized.index[exit_position - 1]

print(f"\n\nSpread normalized on the day before exit ({previous_day}): {spread_normalized.loc[previous_day]}")
print(f"Spread normalized on the exit day ({t_exit}): {spread_normalized.loc[t_exit]}")
print('\n\nDiscrepency between the previous day and the day of the \n exit caused the Profit in levels to not be equal to the theoritical Profit:\n the tick == 0 is not in sample')








expected_profit = z_in * spread.std()


if spread_normalized[spread_normalized == 0].empty:
    print("\nNo exit signal (tick = 0) found after entry.")
    
print('\n \n The theorical expected profit is : ', expected_profit)




print('\n', "# ================================== Subsection 4.2 ======================================= #", '\n')



class PairTradingStrategy:
    """
    Classe pour implémenter la stratégie de pair-trading (Direct Strategy). 
    Gère le calcul du spread et du signal, le contrôle du levier, 
    l'entrée et la sortie de position, ainsi que la conservation de l'historique.
    """

    def __init__(self, df, alpha_hat, beta_hat, z_in, L, W0, z_stop=None):
        """
        Paramètres initiaux :
        :param df: DataFrame contenant les prix (close, open, etc.)
        :param alpha_hat: Estimation de alpha (décalage)
        :param beta_hat: Estimation de beta (coefficient)
        :param z_in: Seuil d'entrée pour z_t normalisé
        :param L: Levier maximal autorisé (ex: 2)
        :param W0: Richesse initiale (ex: 1000)
        :param z_stop: Niveau de spread normalisé pour le stop-loss (ex: 2.5)
                     Pour Signal 1: sortir si z_t > z_stop
                     Pour Signal 2: sortir si z_t < -z_stop
        """
        self.df = df
        self.alpha_hat = alpha_hat
        self.beta_hat = beta_hat
        self.z_in = z_in
        self.L = L
        self.W0 = W0
        self.z_stop = z_stop  
        
        
        self.in_position = False
        self.current_position = (0.0, 0.0)  
        self.current_wealth = W0
        self.trades = []
        
        
        self.wealth_history = []
        self.leverage_history = []
        self.date_history = []
        self.position_history = []  
        
        
        self.spread = None
        self.spread_normalized = None
        self.logger = None  
        
        self.leverage_safety = 0.98
        self.setup_logging()

    def compute_spread(self):
        """
        Calcule le spread et la version normalisée à partir de alpha_hat, beta_hat 
        et des colonnes 'CHEVRON', 'CONOCOPHILLIPS' de self.df.
        """
        prices_A = self.df["CHEVRON"]
        prices_B = self.df["CONOCOPHILLIPS"]
        self.spread = prices_A - self.alpha_hat - self.beta_hat * prices_B
        
        std_spread = self.spread.std()
        if std_spread is not None and std_spread > 0:
            self.spread_normalized = self.spread / std_spread
        else:
            self.spread_normalized = self.spread * 0.0  

    def detect_signal(self, date_idx):
        """
        Détermine s'il y a un Signal 1 (short A / long B) ou Signal 2 (long A / short B)
        à la date correspondante.
        Renvoie 'Signal 1', 'Signal 2', ou None.
        """
        z_t = self.spread_normalized.iloc[date_idx]
        if z_t > self.z_in:
            return "Signal 1"
        elif z_t < -self.z_in:
            return "Signal 2"
        else:
            return None

    def compute_position_size(self, signal_type, alpha_sign, P_A, P_B, W_current):
        """
        Calcule les quantités Q_A et Q_B (selon la direct strategy). 
        Adaptation en fonction du signe de alpha (alpha_sign > 0 ou < 0),
        et du signal (1 ou 2).
        """
        if signal_type == "Signal 1":
            
            if alpha_sign > 0:
                
                factor = (self.L * self.leverage_safety * W_current) / P_A
                Q_A = -1.0 * factor
                Q_B = self.beta_hat * factor
            else:
                
                denom = P_A - self.L * self.leverage_safety * (P_A - self.beta_hat * P_B)
                factor = (self.L * self.leverage_safety * W_current) / denom if denom != 0 else 0
                Q_A = -1.0 * factor
                Q_B = self.beta_hat * factor

        elif signal_type == "Signal 2":
            
            if alpha_sign > 0:
                
                denom = self.beta_hat * P_B + self.L * self.leverage_safety * (P_A - self.beta_hat * P_B)
                factor = (self.L * self.leverage_safety * W_current) / denom if denom != 0 else 0
                Q_A = 1.0 * factor
                Q_B = -self.beta_hat * factor
            else:
                
                denom = self.beta_hat * P_B
                factor = (self.L * self.leverage_safety * W_current) / denom if denom != 0 else 0
                Q_A = 1.0 * factor
                Q_B = -self.beta_hat * factor
        else:
            Q_A, Q_B = 0.0, 0.0
        
        return Q_A, Q_B

    def compute_exposure_and_margin(self, Q_A, Q_B, P_A, P_B, W_current):
        """
        Calcule l'exposition totale et le levier. 
        Exposition = |Q_A * P_A| + |Q_B * P_B|.
        Levier = exposition / W_current.
        """
        exposure = abs(Q_A) * P_A + abs(Q_B) * P_B
        leverage = exposure / W_current if W_current != 0 else 0.0
        return exposure, leverage

    def enter_position(self, date, signal_type, Q_A, Q_B, P_A, P_B):
        """
        Enregistre l'entrée en position, stocke l'état in_position, 
        sauvegarde la date d'entrée et la taille de position.
        """
        self.in_position = True
        self.current_position = (Q_A, Q_B)
        
        
        if self.trades and 'Exit_Wealth' in self.trades[-1]:
            
            entry_wealth = self.trades[-1]['Exit_Wealth']
        else:
            
            entry_wealth = self.current_wealth
        
        entry_wealth = self.current_wealth
        self.trades.append({
            "Entry_Date": date,
            "Entry_Wealth": entry_wealth,  
            "Signal": signal_type,
            "Q_A": Q_A,
            "Q_B": Q_B,
            "P_A_entry": P_A,
            "P_B_entry": P_B
        })
        
        if self.logger:
            self.logger.info(f"ENTER POSITION | {date.date()} | {signal_type} | Q_A: {Q_A:.2f} | Q_B: {Q_B:.2f} | P_A: {P_A:.2f} | P_B: {P_B:.2f} | Wealth: {entry_wealth:.2f}")

    def get_trades_summary(self):
        """
        Generate a DataFrame summarizing all trades with correct wealth accounting.
        """
        import pandas as pd
        import numpy as np
        
        
        if not self.trades:
            columns = ['Signal', 'Entry Date', 'Exit Date', 'Exit Reason', 'Profit', 
                    'Return (%)', 'Entry Wealth', 'Exit Wealth']
            return pd.DataFrame(columns=columns)
        
        
        trades_data = []
        running_wealth = self.W0  
        
        for trade in self.trades:
            
            if 'Entry_Date' not in trade or 'Exit_Date' not in trade:
                continue
                
            
            entry_wealth = running_wealth
            
            
            profit = trade.get('Profit', 0)
            
            
            exit_wealth = entry_wealth + profit
            
            
            running_wealth = exit_wealth
            
            
            if entry_wealth > 0:
                
                return_pct = (profit / entry_wealth) * 100
            elif entry_wealth < 0:
                if exit_wealth >= 0:
                    
                    
                    return_pct = ((exit_wealth - entry_wealth) / abs(entry_wealth)) * 100
                else:
                    
                    
                    return_pct = (profit / abs(entry_wealth)) * 100
            else:  
                
                return_pct = float('inf') if profit > 0 else float('-inf') if profit < 0 else 0.0
            
            trades_data.append({
                'Signal': trade.get('Signal', ''),
                'Entry Date': trade.get('Entry_Date', ''),
                'Exit Date': trade.get('Exit_Date', ''),
                'Exit Reason': trade.get('Exit_Reason', ''),
                'Profit': profit,
                'Return (%)': return_pct,
                'Entry Wealth': entry_wealth,
                'Exit Wealth': exit_wealth
            })
        
        
        trades_df = pd.DataFrame(trades_data)
        
        
        if 'Return (%)' in trades_df.columns:
            trades_df['Return (%)'] = trades_df['Return (%)'].round(2)
        
        return trades_df

    def close_position(self, date, P_A_exit, P_B_exit, reason="Signal crossover"):
        """
        Calcule le profit/pertes sur la position, met à jour la richesse,
        enregistre la sortie de position dans trades.
        
        :param reason: Raison de la sortie ("Signal crossover", "Stop-loss", etc.)
        """
        (Q_A, Q_B) = self.current_position
        last_trade = self.trades[-1] if self.trades else None

        if last_trade and self.in_position:
            signal_type = last_trade["Signal"]
            P_A_entry = last_trade["P_A_entry"]
            P_B_entry = last_trade["P_B_entry"]

            
            if signal_type == "Signal 1":
                
                pnl = Q_A * (P_A_entry - P_A_exit) + Q_B * (P_B_exit - P_B_entry)
            else:
                
                pnl = Q_A * (P_A_exit - P_A_entry) + Q_B * (P_B_entry - P_B_exit)
            
            self.current_wealth += pnl
            self.trades[-1].update({
                "Exit_Date": date,
                "P_A_exit": P_A_exit,
                "P_B_exit": P_B_exit,
                "Profit": pnl,
                "Exit_Reason": reason
            })
            if self.logger:
                self.logger.info(f"EXIT POSITION | {date.date()} | Reason: {reason} | P_A: {P_A_exit:.2f} | P_B: {P_B_exit:.2f} | Profit: {pnl:.2f} | Wealth: {self.current_wealth:.2f}")
        self.in_position = False
        self.current_position = (0.0, 0.0)

    def check_stop_loss(self, date_idx):
        """
        Vérifie si le spread normalisé atteint le niveau de stop-loss défini (z_stop).
        - Pour Signal 1 (Short A, Long B): sortir si z_t > z_stop
        - Pour Signal 2 (Long A, Short B): sortir si z_t < -z_stop
        
        :param date_idx: Index de la date courante dans le DataFrame
        :return: True si le stop-loss est déclenché, False sinon
        """
        if self.z_stop is None or not self.in_position:
            return False
            
        current_date = self.df.index[date_idx]
        z_t = self.spread_normalized.iloc[date_idx]
        
        last_trade = self.trades[-1] if self.trades else None
        if not last_trade:
            return False
            
        
        if last_trade["Signal"] == "Signal 1" and z_t > self.z_stop:
            
            P_A = self.df.loc[current_date, "CHEVRON"]
            P_B = self.df.loc[current_date, "CONOCOPHILLIPS"]
            self.close_position(date=current_date, P_A_exit=P_A, P_B_exit=P_B, reason="Stop-loss")
            return True
            
        elif last_trade["Signal"] == "Signal 2" and z_t < -self.z_stop:
            
            P_A = self.df.loc[current_date, "CHEVRON"]
            P_B = self.df.loc[current_date, "CONOCOPHILLIPS"]
            self.close_position(date=current_date, P_A_exit=P_A, P_B_exit=P_B, reason="Stop-loss")
            return True
            
        return False

    def update_for_day(self, date_idx):
        """
        Routine quotidienne : détecter un signal, éventuellement entrer en position,
        ou détecter une sortie si le spread recroise 0, 
        puis mettre à jour wealth_history et leverage_history.
        """
        current_date = self.df.index[date_idx]

        
        if self.in_position and date_idx > 0:
            
            if self.check_stop_loss(date_idx):
                
                
                pass
            else:
                
                P_A_curr = self.df.loc[current_date, "CHEVRON"]
                P_B_curr = self.df.loc[current_date, "CONOCOPHILLIPS"]
                Q_A, Q_B = self.current_position
                exposure, leverage = self.compute_exposure_and_margin(
                    Q_A, Q_B, P_A_curr, P_B_curr, self.current_wealth
                )
                
                
                if leverage > self.L:
                    reduction_ratio = self.L * self.leverage_safety / leverage
                    Q_A *= reduction_ratio
                    Q_B *= reduction_ratio
                    self.current_position = (Q_A, Q_B)
                
                
                z_prev = self.spread_normalized.iloc[date_idx - 1]
                z_curr = self.spread_normalized.iloc[date_idx]
                if z_prev * z_curr < 0:  
                    
                    if date_idx + 1 < len(self.df.index):
                        close_date = self.df.index[date_idx + 1]
                        P_A_exit = self.df.loc[close_date, "CHEVRON"]
                        P_B_exit = self.df.loc[close_date, "CONOCOPHILLIPS"]
                        self.close_position(close_date, P_A_exit, P_B_exit, reason="Signal crossover")
                        

        
        if self.in_position:
            mtm_value = self.calculate_current_portfolio_value(date_idx)
            self.wealth_history.append(mtm_value)
        else:
            self.wealth_history.append(self.current_wealth)
            
        
        if not self.in_position:
            signal_type = self.detect_signal(date_idx)
            if signal_type in ["Signal 1", "Signal 2"]:
                
                if date_idx + 1 < len(self.df.index):
                    open_date = self.df.index[date_idx + 1]
                    P_A_entry = self.df.loc[open_date, "CHEVRON"]
                    P_B_entry = self.df.loc[open_date, "CONOCOPHILLIPS"]
                    
                    alpha_sign = 1 if self.alpha_hat > 0 else -1
                    Q_A, Q_B = self.compute_position_size(
                        signal_type, alpha_sign, P_A_entry, P_B_entry, self.current_wealth
                    )
                    
                    
                    exposure, leverage = self.compute_exposure_and_margin(
                        Q_A, Q_B, P_A_entry, P_B_entry, self.current_wealth
                    )
                    if leverage > self.L:
                        
                        ratio = self.L * self.leverage_safety / leverage
                        Q_A *= ratio
                        Q_B *= ratio
                    
                    self.enter_position(open_date, signal_type, Q_A, Q_B, P_A_entry, P_B_entry)

        
        P_A_close = self.df.loc[self.df.index[date_idx], "CHEVRON"]
        P_B_close = self.df.loc[self.df.index[date_idx], "CONOCOPHILLIPS"]
        (Q_A, Q_B) = self.current_position
        exposure, leverage = self.compute_exposure_and_margin(
            Q_A, Q_B, P_A_close, P_B_close, self.current_wealth
        )
        
        self.leverage_history.append(leverage)
        self.date_history.append(current_date)
        self.position_history.append((Q_A, Q_B))  
        
    def run_backtest(self):
        """
        Lance la boucle sur toutes les dates. 
        À la fin, la position restante est fermée (optionnel si on veut forcer la liquidation).
        """
        self.compute_spread()
        for i in range(len(self.df.index)):
            self.update_for_day(i)
            
        
        if self.in_position and len(self.df.index) > 0:
            last_date = self.df.index[-1]
            P_A_last = self.df.loc[last_date, "CHEVRON"]
            P_B_last = self.df.loc[last_date, "CONOCOPHILLIPS"]
            self.close_position(last_date, P_A_last, P_B_last, reason="End of backtest")

    def calculate_stop_loss_probability(self):
        """
        Calcule la probabilité conditionnelle d'atteindre le niveau de stop-loss
        le jour suivant l'entrée en position.
        
        Utilise un modèle AR(1) pour estimer la dynamique du spread normalisé
        et calcule Pr(z_{t+1} > z_stop | z_t = z_in).
        """
        import statsmodels.api as sm
        import scipy.stats as stats
        import numpy as np
        
        
        z_series = self.spread_normalized.copy()
    
        
        z_series = z_series.asfreq('B')  
        
        
        model = sm.tsa.AutoReg(z_series.dropna(), lags=1)
        results = model.fit()
        
        
        intercept = np.round(results.params.iloc[0], 1)
        phi = results.params.iloc[1]  
        sigma = np.sqrt(results.sigma2)  
        
        
        z_in = self.z_in
        z_stop = self.z_stop
        
        
        conditional_mean = intercept + phi * z_in
        
        
        prob_hit_stop_signal1 = 1 - stats.norm.cdf(z_stop, loc=conditional_mean, scale=sigma)
        
        
        conditional_mean_signal2 = intercept + phi * (-z_in)
        prob_hit_stop_signal2 = stats.norm.cdf(-z_stop, loc=conditional_mean_signal2, scale=sigma)
        
        return {
            "params": {
                "intercept": intercept,
                "phi": phi,
                "sigma": sigma
            },
            "probabilities": {
                "signal1": prob_hit_stop_signal1,
                "signal2": prob_hit_stop_signal2
            },
            "z_values": {
                "z_in": z_in,
                "z_stop": z_stop
            }
        }

    def calculate_current_portfolio_value(self, date_idx):
        """
        Calcule la valeur mark-to-market du portefeuille à la date donnée
        """
        current_date = self.df.index[date_idx]
        P_A = self.df.loc[current_date, "CHEVRON"]
        P_B = self.df.loc[current_date, "CONOCOPHILLIPS"]
        Q_A, Q_B = self.current_position
        
        
        if self.in_position:
            
            last_trade = self.trades[-1]
            signal_type = last_trade["Signal"]
            P_A_entry = last_trade["P_A_entry"]
            P_B_entry = last_trade["P_B_entry"]
            
            
            if signal_type == "Signal 1":  
                pnl = Q_A * (P_A_entry - P_A) + Q_B * (P_B - P_B_entry)
            else:  
                pnl = Q_A * (P_A - P_A_entry) + Q_B * (P_B_entry - P_B)
                
            return self.trades[-1]["Entry_Wealth"] + pnl
        else:
            return self.current_wealth

    def plot_results(self):
        """
        Trace l'évolution de la richesse, du leverage et du spread normalisé au fil du temps.
        Indique également les périodes de position et le type de signal.
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import numpy as np
        
        
        plt.style.use('seaborn-v0_8-darkgrid')
        
        
                
        if len(self.date_history) != len(self.wealth_history):
            print(f"Attention: Les longueurs ne correspondent pas! dates: {len(self.date_history)}, wealth: {len(self.wealth_history)}")
            
            min_len = min(len(self.date_history), len(self.wealth_history))
            dates = self.date_history[:min_len]
            wealth = self.wealth_history[:min_len]
        else:
            dates = self.date_history
            wealth = self.wealth_history
    
        
        
        
        wealth_array = np.array(wealth)
        max_wealth = np.maximum.accumulate(wealth_array)
        drawdown = (wealth_array - max_wealth) / max_wealth * 100
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        max_dd_idx = np.argmin(drawdown) if len(drawdown) > 0 else 0

        
        if max_drawdown < 0 and max_dd_idx > 0:
            
            dd_start_idx = 0
            for i in range(max_dd_idx, 0, -1):
                if drawdown[i-1] >= 0 and drawdown[i] < 0:
                    dd_start_idx = i
                    break
            
            
            dd_end_idx = len(drawdown) - 1
            for i in range(max_dd_idx, len(drawdown)-1):
                if drawdown[i] < 0 and drawdown[i+1] >= 0:
                    dd_end_idx = i
                    break
            
            
            dd_duration_days = (dates[dd_end_idx] - dates[dd_start_idx]).days
        else:
            dd_duration_days = 0
    
            
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(dates, wealth, label='Wealth', color='#2f77e4', linewidth=1.1)

        ax.set_title("Evolution of Wealth ($)", fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Value ($)', fontsize=12)
        ax.axhline(y=self.W0, color='black', linestyle='--', linewidth=0.95, alpha=0.7, label='Initial Wealth')
        ax.tick_params(axis='x', labelsize=10, labelcolor='gray')
        ax.tick_params(axis='y', labelsize=10, labelcolor='gray')

        
        if max_dd_idx > 0 and max_drawdown < 0:
            max_dd_date = dates[max_dd_idx]
            
            ax.axvline(x=max_dd_date, color='red', linestyle='-', linewidth=0.4, 
                    label= r'max$^{DD} =$' f' {max_drawdown:.2f}%', alpha=0.7)
            
            
            ax.scatter(max_dd_date, wealth[max_dd_idx], color='red', marker='^', s=50, zorder=5)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=2, frameon=False)
        ax.text(0.92, 0.02, f'Final Wealth : $ {np.round(self.wealth_history[-1], 2)}', ha='center', va='center', transform=ax.transAxes, fontsize=10)
        plt.tight_layout()
        plt.show()
        

        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(self.date_history, self.leverage_history, label='Leverage', color='#d62728', linewidth=1.1)
        ax.set_title("Evolution of the Leverage", fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Leverage', fontsize=12)
        ax.axhline(y=self.L, color='black', linestyle='--', alpha=0.7, linewidth=0.95, label=f'Max : {self.L}')
        ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=2, frameon=False)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.tick_params(axis='x', labelsize=10, labelcolor='gray')
        ax.tick_params(axis='y', labelsize=10, labelcolor='gray')
        plt.tight_layout()
        plt.show()
        
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(self.df.index, self.spread_normalized, label=r'$\tilde{z}_t$', color='#9467bd', linewidth=0.95)
        ax.axhline(y=self.z_in, color='black', linestyle='--', alpha=0.7, label=r'$\tilde{z}^{\text{in}}$' f' = {self.z_in}', linewidth=1.1)
        ax.axhline(y=-self.z_in, color='black', linestyle='--', alpha=0.7, linewidth=1.1)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.set_title(r'$\tilde{z}_t \text{ with } \tilde{z}^{\text{in}} \text{ and } \tilde{z}^{\text{out}}$', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(r'$\tilde{z}_t$', fontsize=12)
        ax.tick_params(axis='x', labelsize=10, labelcolor='gray')
        ax.tick_params(axis='y', labelsize=10, labelcolor='gray')

        
        legend_elements = [
            plt.Line2D([0], [0], color='#9467bd', lw=0.95, label=r'$\tilde{z}_t$'),
            plt.Line2D([0], [0], color='black', lw=1.1, linestyle='--', label=r'$|\tilde{z}^{\text{in}}|$' f' = {self.z_in}'),
            plt.scatter([], [], marker='v', color='red', s=45, label=r'Signal 1 (A$\downarrow$, B $\uparrow$)'),
            plt.scatter([], [], marker='^', color='green', s=45, label=r'Signal 2 (A$\uparrow$, B $\downarrow$)'),
            plt.scatter([], [], marker='x', color='black', s=45, label='Exit (Crossover 0)')
        ]

        
        if self.z_stop is not None:
            ax.axhline(y=self.z_stop, color='blue', linestyle='-.', alpha=0.7, 
                    label=r'$\tilde{z}^{\text{stop}}$' f' = {self.z_stop}', linewidth=1.1)
            ax.axhline(y=-self.z_stop, color='blue', linestyle='-.', alpha=0.7, linewidth=1.1)
            
            legend_elements.append(plt.Line2D([0], [0], color='blue', lw=1.1, linestyle='-.', 
                                    label=r'$\tilde{z}^{\text{stop}}$' f' = {self.z_stop}'))
            legend_elements.append(plt.scatter([], [], marker='x', color='blue', s=25, linewidth=1, label='Exit (Stop-loss)'))

        
        for trade in self.trades:
            if 'Entry_Date' in trade and trade['Entry_Date'] in self.spread_normalized.index:
                if trade['Signal'] == 'Signal 1':
                    ax.scatter(trade['Entry_Date'], self.spread_normalized.loc[trade['Entry_Date']], 
                            marker='v', color='red', s=45)
                else:
                    ax.scatter(trade['Entry_Date'], self.spread_normalized.loc[trade['Entry_Date']], 
                            marker='^', color='green', s=45)
                        
            if 'Exit_Date' in trade and trade['Exit_Date'] in self.spread_normalized.index:
                if trade['Exit_Reason'] == 'Signal crossover':
                    
                    ax.scatter(trade['Exit_Date'], self.spread_normalized.loc[trade['Exit_Date']], 
                            marker='x', color='black', s=45)
                else:
                    
                    ax.scatter(trade['Exit_Date'], self.spread_normalized.loc[trade['Exit_Date']], 
                            marker='x', color='blue', s=25, linewidth=1)
                    
        
        ax.legend(handles=legend_elements, fontsize=8, loc='upper left', bbox_to_anchor=(1, 1), 
                borderaxespad=2, frameon=False)
        plt.show()
        plt.tight_layout()

    def setup_logging(self, log_level='INFO', log_file=None, console_output=False):
        """
        Configure le système de logging pour la stratégie de pair trading.
        
        Parameters:
        -----------
        log_level : str
            Niveau de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file : str, optional
            Chemin du fichier de log. Si None, pas d'enregistrement dans un fichier.
        console_output : bool
            Si True, affiche les logs dans la console.
        """
        import logging
        import os
        
        
        self.logger = logging.getLogger(f'PairTrading_CHEVRON_CONOCOPHILLIPS')
        self.logger.setLevel(getattr(logging, log_level))
        self.logger.handlers = []  
        
        
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', 
                                    datefmt='%Y-%m-%d %H:%M:%S')
        
        
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        
        if log_file:
            
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            file_handler = logging.FileHandler(log_file, mode='w')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        self.logger.info(f"Logging initialized for pair trading strategy CHEVRON/CONOCOPHILLIPS")
        self.logger.info(f"Parameters: alpha={self.alpha_hat:.4f}, beta={self.beta_hat:.4f}, z_in={self.z_in}, L={self.L}, W0={self.W0}, z_stop={self.z_stop}")
        
        
        self.logger.propagate = False
        


def audit_pair_trading_strategy(df, alpha_hat, beta_hat, z_in=1.5, L=2, W0=1000.0, stop__loss=None):
    """
    Fonction de test (audit basique) pour la classe PairTradingStrategy.
    On crée une instance de la classe, on exécute run_backtest(),
    puis on fait quelques vérifications sur les résultats.
    """
    import pandas as pd
    
    strategy = PairTradingStrategy(
        df=df,
        alpha_hat=alpha_hat,
        beta_hat=beta_hat,
        z_in=z_in,
        L=L,
        W0=W0,
        z_stop=stop__loss
    )
    
    
    strategy.run_backtest()
    
    
    
    total_enters = len([t for t in strategy.trades if "Entry_Date" in t])
    total_exits = len([t for t in strategy.trades if "Exit_Date" in t])
    print("===== Audit =====")
    print(f"Number of detected signals (Entry positions): {total_enters}")
    print(f"Number of closed trades: {total_exits}")

    
    final_wealth = strategy.wealth_history[-1] if strategy.wealth_history else None
    print(f"Final wealth: {final_wealth:.2f}" if final_wealth else "No wealth value recorded.")

    
    max_leverage = max(strategy.leverage_history) if strategy.leverage_history else None
    if max_leverage is not None:
        print(f"Maximum observed leverage: {max_leverage:.2f}")
        if max_leverage > L + 1e-6:
            print("Alert: leverage has exceeded the allowed limit!")
    else:
        print("No leverage value recorded.")
     
    
    wealth_array = np.array(strategy.wealth_history)
    max_wealth = np.maximum.accumulate(wealth_array)
    drawdown = (wealth_array - max_wealth) / max_wealth * 100
    max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
    print(f"Drawdown maximum : {max_drawdown:.2f}%")
    
    
    
    

    
    strategy.plot_results()
    
    
    stop_loss_df = None
    if stop__loss is not None:
        prob_results = strategy.calculate_stop_loss_probability()
        import pandas as pd
        stop_loss_df = pd.DataFrame({
            'Paramètre': ['Intercept', 'Phi', 'Sigma', 'z_in', 'z_stop', 
                          'Prob_StopLoss_Signal1', 'Prob_StopLoss_Signal2'],
            'Valeur': [prob_results['params']['intercept'], 
                      prob_results['params']['phi'],
                      prob_results['params']['sigma'],
                      prob_results['z_values']['z_in'],
                      prob_results['z_values']['z_stop'],
                      prob_results['probabilities']['signal1'],
                      prob_results['probabilities']['signal2']]
        })
    
    
  

    
    closed_trades = [t for t in strategy.trades if "Exit_Date" in t]
    profits = [t["Profit"] for t in closed_trades] if closed_trades else [0]
    durations = [(t["Exit_Date"] - t["Entry_Date"]).days for t in closed_trades] if closed_trades else [0]
    win_trades = len([p for p in profits if p > 0])
    
    
    wealth_array = np.array(strategy.wealth_history)
    max_wealth = np.maximum.accumulate(wealth_array)
    drawdown = (wealth_array - max_wealth) / max_wealth * 100
    max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
    
    
    strategy_data = {
        "configuration": {
            "alpha_hat": alpha_hat,
            "beta_hat": beta_hat,
            "z_in": z_in,
            "L": L,
            "W0": W0,
            "z_stop": stop__loss,
            "date_execution": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "trades_stats": {
            "total_trades": total_exits,
            "successful_trades": win_trades,
            "win_rate": win_trades / total_exits if total_exits > 0 else 0,
            "avg_profit": np.mean(profits) if profits else 0,
            "median_profit": np.median(profits) if profits else 0,
            "max_profit": max(profits) if profits else 0,
            "max_loss": min(profits) if profits else 0,
            "profit_std": np.std(profits) if profits else 0,
            "avg_duration_days": np.mean(durations) if durations else 0,
            "total_pnl": sum(profits) if profits else 0
        },
        "performance": {
            "final_wealth": final_wealth if final_wealth else 0,
            "total_return": (final_wealth / W0 - 1) * 100 if final_wealth else 0,
            "max_drawdown": max_drawdown,
            "max_leverage": max_leverage if max_leverage else 0,
            "volatility": np.std(np.diff(wealth_array) / wealth_array[:-1]) * 100 if len(wealth_array) > 1 else 0
        },
        "histories": {
            "wealth_history": strategy.wealth_history,
            "leverage_history": strategy.leverage_history,
            "date_history": [d.strftime("%Y-%m-%d") if isinstance(d, pd.Timestamp) else str(d) for d in strategy.date_history],
            "position_history": strategy.position_history if hasattr(strategy, "position_history") else []
        },
        "trades": strategy.trades,
        "stop_loss_analysis": prob_results if stop__loss is not None else None
    }
    trades_df = strategy.get_trades_summary()

    return strategy, strategy_data, stop_loss_df, trades_df


def strategy_summary_table(strategy_data):
    """
    Crée un DataFrame résumant les résultats de la stratégie de pairs trading,
    incluant les dates importantes de la session de trading et des trades.
    """
    
    wealth_history = np.array(strategy_data["histories"]["wealth_history"])
    max_wealth = np.max(wealth_history) if len(wealth_history) > 0 else 0
    min_wealth = np.min(wealth_history) if len(wealth_history) > 0 else 0
    
    
    first_bankruptcy_date = "No Bankruptcy"
    if min_wealth <= 0:
        bankruptcy_indices = np.where(wealth_history <= 0)[0]
        if len(bankruptcy_indices) > 0:
            first_bankruptcy_index = bankruptcy_indices[0]
            first_bankruptcy_date = strategy_data["histories"]["date_history"][first_bankruptcy_index]
    
    
    session_start_date = "Data Unavailable"
    session_end_date = "Data Unavailable"
    if strategy_data["histories"]["date_history"] and len(strategy_data["histories"]["date_history"]) > 0:
        session_start_date = strategy_data["histories"]["date_history"][0]
        session_end_date = strategy_data["histories"]["date_history"][-1]
    
    
    first_trade_date = "No Trades"
    last_trade_date = "No Trades"
    
    
    if strategy_data["trades"] and len(strategy_data["trades"]) > 0:
        first_trade_date = strategy_data["trades"][0].get("Entry_Date", "Unknown")
    
    
    if strategy_data["trades"] and len(strategy_data["trades"]) > 0:
        
        for trade in reversed(strategy_data["trades"]):
            if "Exit_Date" in trade:
                last_trade_date = trade["Exit_Date"]
                break
    
    
    results = {
        "Metrics": [
            
            "z_in",
            "W_0",
            
            
            "W_final",
            "W_max",
            "W_min",
            "Net Profit",
            "Return (%)",
            "Nb trades",
            "Max Drawdown (%)",
            "Max Leverage",
            
            
            "Trading Session Start Date",
            "Trading Session End Date",
            "First Trade Date",
            "Last Trade Date",
            "First Bankruptcy Date"
        ],
        "Value": [
            
            strategy_data["configuration"]["z_in"],
            strategy_data["configuration"]["W0"],
            
            
            strategy_data["performance"]["final_wealth"],
            max_wealth,
            min_wealth,
            strategy_data["trades_stats"]["total_pnl"],
            strategy_data["performance"]["total_return"],
            strategy_data["trades_stats"]["total_trades"],
            strategy_data["performance"]["max_drawdown"],
            strategy_data["performance"]["max_leverage"],
            
            
            session_start_date,
            session_end_date,
            first_trade_date,
            last_trade_date,
            first_bankruptcy_date
        ]
    }
    
    return pd.DataFrame(results)




print('\n', "# ---------------------------------- Q 4.7 --------------------------------------- #", '\n')

strategy_instance, strategy_data, stop_loss_df, trades_df = audit_pair_trading_strategy(df, alpha_hat, beta_hat, z_in=1.5, L=2, W0=1000.0, stop__loss=None)

strategy_summary_df  = np.round(strategy_summary_table(strategy_data), 2)
trades_df            = np.round(trades_df)



print(strategy_summary_df)
print(trades_df)



def analyze_trade_leverage(strategy):
    """
    Analyze leverage values at trade entry and exit points
    
    Parameters:
    -----------
    strategy : PairTradingStrategy instance
        An instance of PairTradingStrategy after running backtest
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing trade details with leverage at entry and exit
    """
    import pandas as pd
    import numpy as np
    
    
    if not strategy.trades or not strategy.date_history or not strategy.leverage_history:
        return pd.DataFrame(columns=['Trade', 'Entry_Date', 'Exit_Date', 'Signal', 'Entry_Leverage', 'Exit_Leverage'])
    
    
    leverage_series = pd.Series(
        strategy.leverage_history, 
        index=pd.DatetimeIndex(strategy.date_history)
    )
    
    
    trade_leverage = []
    for i, trade in enumerate(strategy.trades):
        
        if 'Entry_Date' not in trade or 'Exit_Date' not in trade:
            continue
            
        entry_date = trade['Entry_Date']
        exit_date = trade['Exit_Date']
        
        
        try:
            entry_leverage = leverage_series.loc[entry_date]
        except KeyError:
            
            closest_entry_idx = np.abs(leverage_series.index - entry_date).argmin()
            entry_leverage = leverage_series.iloc[closest_entry_idx]
            
        
        try:
            exit_leverage = leverage_series.loc[exit_date]
        except KeyError:
            
            closest_exit_idx = np.abs(leverage_series.index - exit_date).argmin()
            exit_leverage = leverage_series.iloc[closest_exit_idx]
        
        
        trade_leverage.append({
            'Trade': i+1,
            'Entry_Date': entry_date,
            'Exit_Date': exit_date,
            'Signal': trade['Signal'],
            'Entry_Leverage': entry_leverage,
            'Exit_Leverage': exit_leverage,
            'Max_Leverage': leverage_series[entry_date:exit_date].max() if exit_date > entry_date else entry_leverage,
            'Profit': trade.get('Profit', None),
            'Return(%)': (trade.get('Profit', 0) / trade.get('Entry_Wealth', 1)) * 100 if trade.get('Entry_Wealth') else None,
            'Duration(days)': (exit_date - entry_date).days
        })
    
    
    leverage_df = pd.DataFrame(trade_leverage)
    
    
    for col in ['Entry_Leverage', 'Exit_Leverage', 'Max_Leverage', 'Return(%)']:
        if col in leverage_df.columns:
            leverage_df[col] = leverage_df[col].round(2)
    
    return leverage_df









print('\n', "# ====================================  Q 4.8 ========================================= #", '\n')
 
strategy_instance2, strategy_data2, stop_loss_df2, trades_df2 = audit_pair_trading_strategy(df, alpha_hat, beta_hat, z_in=1.5, L=20, W0=1000.0, stop__loss=None)

strategy_summary_df2 = np.round(strategy_summary_table(strategy_data2), 2)
trade_leverage_analysis = analyze_trade_leverage(strategy_instance2)


print(strategy_summary_df2)

print(trades_df2)




print('\n', "# ====================================  Q 4.9 ========================================= #", '\n')
 
strategy_instance3, strategy_data3, stop_loss_df3, trades_df3 = audit_pair_trading_strategy(df,
                                                                                alpha_hat,
                                                                                beta_hat,
                                                                                z_in=1.5,
                                                                                L=2,
                                                                                W0=1000.0,
                                                                                stop__loss=1.75)
strategy_summary_df3 = np.round(strategy_summary_table(strategy_data3), 2)

strategy_summary_df3



stop_loss_df3


z_stat_data = strategy_instance3.calculate_stop_loss_probability()



phi = z_stat_data['params']['phi']




adjusted_threshold = 1.75 - 1.5 * phi  


print(f"AR(1) coefficient phi = {phi:.4f}")
print(f"Effective threshold (1.75 - 1.5*phi) = {adjusted_threshold:.4f}")
print(f"Persistence component (1.5*phi) = {1.5*phi:.4f}")




from scipy import stats   


hitting_probability = 1 - stats.norm.cdf(
    adjusted_threshold,
    loc=0,
    scale=z_stat_data['params']['sigma']
)

print(f"Probability of hitting stop-loss: {hitting_probability:.4f}")



print('\n', "# ====================================  Q 4.10 ========================================= #", '\n')
 
strategy_instance4, strategy_data4, stop_loss_df4, trades_df4 = audit_pair_trading_strategy(df,
                                                                                alpha_hat,
                                                                                beta_hat,
                                                                                z_in=1.5,
                                                                                L=2,
                                                                                W0=1000.0,
                                                                                stop__loss=2.75)
strategy_summary_df4 = np.round(strategy_summary_table(strategy_data4), 2)
print(strategy_summary_df4)
print(trades_df4)




class RollingWindowAnalysis:
    """
    Classe complète et indépendante pour l'analyse par fenêtre glissante et le trading
    basé sur la cointegration entre deux actifs financiers.
    """
    
    def __init__(self, df, window_size=500, step_size=20, asset_A='CHEVRON', asset_B='CONOCOPHILLIPS', audit=False):
        self.df = df.copy()
        self.window_size = window_size
        self.step_size = step_size
        self.asset_A = asset_A
        self.asset_B = asset_B
        self.audit = audit  
        
        self.window_dates = []
        self.alphas = []
        self.betas = []
        self.price_correlations = []
        self.return_correlations = []
        self.p_values = []
        self.is_cointegrated = []
        
        
        self.out_of_sample_spread = pd.DataFrame(index=df.index)
        self.out_of_sample_spread['normalized_spread'] = np.nan
        self.out_of_sample_spread['estimation_window'] = pd.Series(dtype='datetime64[ns]')  
        self.out_of_sample_spread['is_cointegrated'] = pd.Series(dtype='bool')  
        
        
        self.full_sample_params = self.compute_full_sample_parameters()
        
        
        self.trades = []
        self.wealth_history = []
        self.leverage_history = []
        self.date_history = []
        
    def compute_full_sample_parameters(self):
        """
        Calcule les paramètres alpha et beta sur l'échantillon complet.
        """
        print("Calcul des paramètres sur échantillon complet...")
        
        X = self.df[self.asset_B].values.reshape(-1, 1)
        X = sm.add_constant(X)
        y = self.df[self.asset_A].values
        
        model = sm.OLS(y, X).fit()
        
        
        print(f"Type des paramètres: {type(model.params)}")
        
        if isinstance(model.params, np.ndarray):
            alpha = model.params[0]  
            beta = model.params[1]
            print(f"Utilisation de l'indexation NumPy: alpha={alpha:.4f}, beta={beta:.4f}")
        else:
            alpha = model.params.iloc[0]  
            beta = model.params.iloc[1]
            print(f"Utilisation de .iloc: alpha={alpha:.4f}, beta={beta:.4f}")
        
        
        spread = self.df[self.asset_A] - alpha - beta * self.df[self.asset_B]
        spread_std = spread.std()
        
        
        print(f"Full sample - Moyennes: {self.asset_A}={self.df[self.asset_A].mean():.2f}, {self.asset_B}={self.df[self.asset_B].mean():.2f}")
        print(f"Full sample - Écart-types: {self.asset_A}={self.df[self.asset_A].std():.2f}, {self.asset_B}={self.df[self.asset_B].std():.2f}")
        print(f"Full sample - Corrélation: {self.df[self.asset_A].corr(self.df[self.asset_B]):.4f}")
        print(f"Full sample - Spread std: {spread_std:.4f}")
        
        return {
            'alpha': alpha,
            'beta': beta,
            'spread': spread,
            'spread_normalized': spread / spread_std,
            'spread_std': spread_std
        }
        
    def estimate_parameters(self, window_data):
        """
        Estime les paramètres alpha et beta pour une fenêtre donnée.
        """
        X = window_data[self.asset_B].values.reshape(-1, 1)
        X = sm.add_constant(X)
        y = window_data[self.asset_A].values
        
        model = sm.OLS(y, X).fit()
        
        if isinstance(model.params, np.ndarray):
            return model.params[0], model.params[1]  
        else:
            return model.params.iloc[0], model.params.iloc[1]  
    
    def compute_correlations(self, window_data):
        """
        Calcule les corrélations des prix et des rendements pour une fenêtre.
        """
        
        price_corr = window_data[self.asset_A].corr(window_data[self.asset_B])
        
        
        returns_A = window_data[self.asset_A].pct_change().dropna()
        returns_B = window_data[self.asset_B].pct_change().dropna()
        return_corr = returns_A.corr(returns_B)
        
        return price_corr, return_corr
    
    def test_cointegration(self, window_data, pvalue_threshold=0.05):
        """
        Teste la cointégration avec un test DF simple
        """
        
        alpha, beta = self.estimate_parameters(window_data)
        spread = window_data[self.asset_A] - alpha - beta * window_data[self.asset_B]
        
        
        adf_result = adfuller(spread, regression='c', maxlag=0)
        pvalue = adf_result[1]
        is_cointegrated = pvalue < pvalue_threshold
        
        
        return pvalue, is_cointegrated

    def test_cointegration_advanced(self, window_data, pvalue_threshold=0.05, max_lag=None, 
                                regression_type='c', autolag='AIC', verbose=False, 
                                check_residuals_ljungbox=False, ljungbox_lags=None, ljungbox_threshold=0.05):
        """
        Teste la cointégration avec un test ADF avancé et des options configurables.
        Peut également effectuer un test de Ljung-Box sur les résidus pour vérifier l'absence d'autocorrélation.
        
        Cette méthode calcule d'abord le spread entre les deux actifs en utilisant les paramètres alpha et beta
        estimés, puis applique un test de Dickey-Fuller augmenté (ADF) pour déterminer si le spread est stationnaire,
        ce qui indiquerait une relation de cointégration.
        
        Args:
            window_data (pandas.DataFrame): DataFrame contenant les données de la fenêtre avec les colonnes
                                        correspondant aux actifs A et B
            pvalue_threshold (float): Seuil pour déterminer la cointégration (défaut: 0.05)
            max_lag (int, optional): Nombre maximum de retards pour le test ADF
                                - None = sélection automatique basée sur autolag
                                - int = nombre maximum spécifique de retards à utiliser
            regression_type (str): Type de régression pour le test ADF:
                                - 'c' = inclut une constante (défaut)
                                - 'ct' = inclut une constante et une tendance linéaire
                                - 'ctt' = inclut une constante et une tendance quadratique
                                - 'nc' = pas de termes déterministes
            autolag (str): Méthode pour la sélection du nombre de retards:
                        - 'AIC' = Critère d'information d'Akaike (défaut)
                        - 'BIC' = Critère d'information bayésien
                        - 't-stat' = t-statistique
            verbose (bool): Afficher des informations supplémentaires sur le processus de test
            check_residuals_ljungbox (bool): Si True, effectue un test de Ljung-Box sur les résidus
            ljungbox_lags (int or list, optional): Retards à utiliser pour le test de Ljung-Box
                                            - None = [5, 10, 15]
                                            - int = valeur unique de retard
                                            - list = liste de retards
            ljungbox_threshold (float): Seuil de p-value pour le test de Ljung-Box (défaut: 0.05)
            
        Returns:
            tuple: (p-value, is_cointegrated, adf_stats)
                - p-value: valeur p du test ADF
                - is_cointegrated: bool indiquant si la cointégration est détectée
                - adf_stats: dictionnaire avec les statistiques détaillées du test
                    (adf_statistic, pvalue, used_lag, nobs, critical_values, ljungbox_results si demandé)
        
        Raises:
            ValueError: Si les données d'entrée ne sont pas suffisantes ou si les paramètres sont invalides
            RuntimeWarning: Si le test présente des anomalies potentielles
        """
        
        if window_data is None or len(window_data) < 20:  
            raise ValueError(f"Données insuffisantes pour le test ADF: {len(window_data) if window_data is not None else 0} observations")
        
        if self.asset_A not in window_data.columns or self.asset_B not in window_data.columns:
            raise ValueError(f"Les colonnes {self.asset_A} et/ou {self.asset_B} ne sont pas présentes dans les données")
        
        
        missing_values_A = window_data[self.asset_A].isna().sum()
        missing_values_B = window_data[self.asset_B].isna().sum()
        missing_ratio = (missing_values_A + missing_values_B) / (len(window_data) * 2)
        
        if missing_ratio > 0.05:  
            import warnings
            warnings.warn(f"Les données contiennent {missing_ratio:.2%} de valeurs manquantes, ce qui peut affecter la fiabilité du test", 
                        RuntimeWarning)
        
        
        clean_data = window_data[[self.asset_A, self.asset_B]].dropna()
        
        if len(clean_data) < 20:  
            raise ValueError(f"Données insuffisantes après suppression des valeurs manquantes: {len(clean_data)} observations")
        
        try:
            
            alpha, beta = self.estimate_parameters(clean_data)
            spread = clean_data[self.asset_A] - alpha - beta * clean_data[self.asset_B]
            
            if verbose:
                print(f"Spread calculé pour {len(spread)} observations")
                print(f"Paramètres: alpha={alpha:.6f}, beta={beta:.6f}")
                print(f"Statistiques du spread: moyenne={spread.mean():.6f}, écart-type={spread.std():.6f}")
            
            
            from statsmodels.tsa.stattools import adfuller
            import numpy as np
            
            
            if np.isnan(spread).any() or np.isinf(spread).any():
                
                spread = spread.replace([np.inf, -np.inf], np.nan).dropna()
                if len(spread) < 20:
                    raise ValueError("Trop de valeurs problématiques dans le spread après nettoyage")
            
            
            adf_result = adfuller(spread, regression=regression_type, maxlag=max_lag, autolag=autolag)
            
            
            adf_statistic = adf_result[0]
            pvalue = adf_result[1]
            used_lag = adf_result[2]
            nobs = adf_result[3]
            critical_values = adf_result[4]
            is_cointegrated = pvalue < pvalue_threshold
            
            
            adf_stats = {
                'adf_statistic': adf_statistic,
                'pvalue': pvalue,
                'used_lag': used_lag,
                'nobs': nobs,
                'critical_values': critical_values,
                'spread_mean': spread.mean(),
                'spread_std': spread.std(),
                'spread_skew': spread.skew(),
                'spread_kurtosis': spread.kurtosis(),
                'test_parameters': {
                    'regression_type': regression_type,
                    'max_lag': max_lag,
                    'autolag': autolag
                }
            }
            
            
            if check_residuals_ljungbox:
                from statsmodels.stats.diagnostic import acorr_ljungbox
                
                
                if ljungbox_lags is None:
                    ljungbox_lags = [5, 10, 15]
                elif isinstance(ljungbox_lags, int):
                    ljungbox_lags = [ljungbox_lags]
                
                
                try:
                    lb_result = acorr_ljungbox(spread, lags=ljungbox_lags, return_df=True)
                    
                    
                    adf_stats['ljungbox_results'] = {
                        'statistics': lb_result['lb_stat'].to_dict(),
                        'pvalues': lb_result['lb_pvalue'].to_dict(),
                        'no_autocorr': (lb_result['lb_pvalue'] > ljungbox_threshold).to_dict()
                    }
                    
                    
                    
                    no_autocorr = any(p > ljungbox_threshold for p in lb_result['lb_pvalue'])
                    adf_stats['ljungbox_no_autocorr'] = no_autocorr
                    
                    
                    
                    is_cointegrated_combined = is_cointegrated and no_autocorr
                    
                    
                    adf_stats['is_cointegrated_adf'] = is_cointegrated
                    adf_stats['is_cointegrated_combined'] = is_cointegrated_combined
                    
                    if verbose:
                        print(f"Test de Ljung-Box complété pour les retards {ljungbox_lags}")
                        print(f"Résultats de Ljung-Box: {lb_result['lb_pvalue'].to_dict()}")
                        print(f"Absence d'autocorrélation: {no_autocorr}")
                        print(f"Résultat combiné: {'Cointegrated' if is_cointegrated_combined else 'Not cointegrated'}")
                    
                    
                    is_cointegrated = is_cointegrated_combined
                
                except Exception as e:
                    import warnings
                    warnings.warn(f"Échec du test de Ljung-Box: {str(e)}. Utilisation du résultat ADF uniquement.")
                    adf_stats['ljungbox_error'] = str(e)
            
            if verbose:
                print(f"Test ADF complété: statistique={adf_statistic:.6f}, p-value={pvalue:.6f}")
                print(f"Nombre de retards utilisés: {used_lag}")
                print(f"Résultat: {'Cointegrated' if is_cointegrated else 'Not cointegrated'} (seuil={pvalue_threshold})")
            
            
            
            
            
            
            
            return pvalue, is_cointegrated, adf_stats
        
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            
            
            error_stats = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'error_details': error_details
            }
            
            if verbose:
                print(f"Erreur lors du test de cointégration: {str(e)}")
            
            
            
            raise ValueError(f"Échec du test de cointégration: {str(e)}") from e
            
    def run_rolling_analysis(self, pvalue_threshold=0.05):
        """
        Exécute l'analyse par fenêtre glissante sur l'ensemble des données.
        """
        
        self.out_of_sample_spread['is_cointegrated'] = self.out_of_sample_spread['is_cointegrated'].astype('bool')
        
        """
        Exécute l'analyse par fenêtre glissante sur l'ensemble des données.
        """
        if self.audit:
            print(f"Démarrage de l'analyse glissante avec {len(self.df)} observations")
            print(f"Taille de la fenêtre: {self.window_size}, pas: {self.step_size}")
        
        window_count = 0
        
        for start_idx in range(0, len(self.df) - self.window_size, self.step_size):
            end_idx = start_idx + self.window_size
            forecast_end = min(end_idx + self.step_size, len(self.df))
            
            
            window_count += 1
            if self.audit and window_count % 10 == 0:  
                print(f"Traitement de la fenêtre {window_count}: indices {start_idx}-{end_idx}")
            
            
            estimation_window = self.df.iloc[start_idx:end_idx]
            forecast_window = self.df.iloc[end_idx:forecast_end]
            
            if forecast_window.empty:
                continue
            
            
            window_date = estimation_window.index[len(estimation_window) // 2]
            self.window_dates.append(window_date)
            
            
            alpha, beta = self.estimate_parameters(estimation_window)
            self.alphas.append(alpha)
            self.betas.append(beta)
            
            
            price_corr, return_corr = self.compute_correlations(estimation_window)
            self.price_correlations.append(price_corr)
            self.return_correlations.append(return_corr)
            
            
            pvalue, is_cointegrated = self.test_cointegration(estimation_window, pvalue_threshold)
            self.p_values.append(pvalue)
            self.is_cointegrated.append(is_cointegrated)
            
            
            spread_in = estimation_window[self.asset_A] - alpha - beta * estimation_window[self.asset_B]
            spread_std = spread_in.std()
            
            
            spread_out = forecast_window[self.asset_A] - alpha - beta * forecast_window[self.asset_B]
            spread_out_norm = spread_out / spread_std
            
            
            self.out_of_sample_spread.loc[forecast_window.index, 'normalized_spread'] = spread_out_norm.values
            self.out_of_sample_spread.loc[forecast_window.index, 'estimation_window'] = window_date
            self.out_of_sample_spread.loc[forecast_window.index, 'is_cointegrated'] = is_cointegrated
            
        
        self.alphas = np.array(self.alphas)
        self.betas = np.array(self.betas)
        self.price_correlations = np.array(self.price_correlations)
        self.return_correlations = np.array(self.return_correlations)
        self.p_values = np.array(self.p_values)
        
    def get_detailed_window_analysis(self):
        """
        Fournit une analyse détaillée de chaque fenêtre pour faciliter l'audit indépendant des résultats.
        
        Returns:
            dict: Dictionnaire contenant plusieurs DataFrames d'analyse et statistiques
        """
        
        if not hasattr(self, 'window_dates') or len(self.window_dates) == 0:
            print("Aucune analyse par fenêtre disponible. Exécutez d'abord run_rolling_analysis().")
            
            return {
                'window_details': pd.DataFrame(),
                'period_summary': pd.DataFrame(),
                'param_stability': pd.DataFrame(),
                'trades_with_windows': pd.DataFrame(),
                'performance_by_window': pd.DataFrame(),
                'spread_by_window': pd.DataFrame() if hasattr(self, 'out_of_sample_spread') else pd.DataFrame(),
                'rolling_stability': pd.DataFrame(),
                'full_sample_params': self.full_sample_params if hasattr(self, 'full_sample_params') else {}
            }
        
        
        window_details = pd.DataFrame({
            'window_date': self.window_dates,
            'start_date': [self.df.index[max(0, self.df.index.get_loc(date) - self.window_size//2)] for date in self.window_dates],
            'end_date': [self.df.index[min(len(self.df)-1, self.df.index.get_loc(date) + self.window_size//2)] for date in self.window_dates],
            'alpha': self.alphas,
            'beta': self.betas,
            'p_value': self.p_values,
            'is_cointegrated': self.is_cointegrated,
            'price_correlation': self.price_correlations,
            'return_correlation': self.return_correlations
        })
        
        
        if hasattr(self, 'full_sample_params'):
            window_details['alpha_full_deviation'] = window_details['alpha'] - self.full_sample_params['alpha']
            window_details['beta_full_deviation'] = window_details['beta'] - self.full_sample_params['beta']
            window_details['alpha_full_deviation_pct'] = (window_details['alpha'] / self.full_sample_params['alpha'] - 1) * 100
            window_details['beta_full_deviation_pct'] = (window_details['beta'] / self.full_sample_params['beta'] - 1) * 100
        
        
        window_count = len(self.window_dates)
        period_summary = pd.DataFrame()
        if window_count > 0:
            
            num_bins = max(1, min(10, window_count))
            periods = pd.cut(range(window_count), bins=num_bins, labels=False)
            period_summary = window_details.groupby(periods).agg({
                'window_date': lambda x: f"{min(x).date()} - {max(x).date()}",
                'alpha': ['mean', 'std', 'min', 'max'],
                'beta': ['mean', 'std', 'min', 'max'],
                'p_value': ['mean', 'min', 'count', lambda x: sum(x < 0.05)/len(x) * 100],
                'is_cointegrated': ['mean', 'sum'],
                'price_correlation': ['mean', 'min', 'max'],
                'return_correlation': ['mean', 'min', 'max']
            })
            
            period_summary.columns = ['_'.join(col).strip() for col in period_summary.columns.values]
            period_summary = period_summary.rename(columns={
                'p_value_<lambda_0>': 'cointegration_pct',
                'is_cointegrated_mean': 'cointegration_ratio',
                'is_cointegrated_sum': 'cointegrated_windows'
            })
        
        
        trades_with_windows = []
        if hasattr(self, 'trades') and self.trades and hasattr(self, 'out_of_sample_spread'):
            for trade in self.trades:
                if 'Entry_Date' in trade and 'Exit_Date' in trade:
                    
                    if ('estimation_window' in self.out_of_sample_spread.columns and 
                        trade['Entry_Date'] in self.out_of_sample_spread.index):
                        entry_window_idx = self.out_of_sample_spread.loc[trade['Entry_Date'], 'estimation_window']
                        
                        if entry_window_idx in self.window_dates:
                            trade_copy = trade.copy()
                            trade_copy['Estimation_Window'] = entry_window_idx
                            
                            window_idx = self.window_dates.index(entry_window_idx)
                            trade_copy['Window_Alpha'] = self.alphas[window_idx]
                            trade_copy['Window_Beta'] = self.betas[window_idx]
                            trade_copy['Window_P_Value'] = self.p_values[window_idx]
                            trade_copy['Window_Is_Cointegrated'] = self.is_cointegrated[window_idx]
                            trades_with_windows.append(trade_copy)
        
        
        spread_by_window = pd.DataFrame()
        if hasattr(self, 'out_of_sample_spread'):
            spread_by_window = self.out_of_sample_spread.copy()
        
        
        performance_by_window = pd.DataFrame()
        if trades_with_windows:
            trades_df = pd.DataFrame(trades_with_windows)
            if 'Estimation_Window' in trades_df.columns:
                performance_by_window = trades_df.groupby('Estimation_Window').agg({
                    'Profit': ['sum', 'mean', 'count'] if 'Profit' in trades_df.columns else [],
                    'Return': ['mean', 'min', 'max'] if 'Return' in trades_df.columns else [],
                    'Signal': 'count' if 'Signal' in trades_df.columns else []
                })
                if not performance_by_window.empty:
                    performance_by_window.columns = ['_'.join(col).strip() for col in performance_by_window.columns.values]
                    
                    
                    if 'Profit' in trades_df.columns:
                        winning_trades = trades_df[trades_df['Profit'] > 0].groupby('Estimation_Window').size()
                        total_trades = trades_df.groupby('Estimation_Window').size()
                        win_ratio = winning_trades / total_trades
                        performance_by_window['win_ratio'] = win_ratio
        
        
        rolling_stability = pd.DataFrame()
        if hasattr(self, 'window_dates') and window_count > 0:
            rolling_stability = pd.DataFrame({
                'date': self.window_dates,
                'alpha': self.alphas,
                'beta': self.betas,
                'is_cointegrated': self.is_cointegrated
            })
            
            if hasattr(self, 'full_sample_params'):
                rolling_stability['alpha_deviation'] = rolling_stability['alpha'] - self.full_sample_params['alpha']
                rolling_stability['beta_deviation'] = rolling_stability['beta'] - self.full_sample_params['beta']
                
                
                if window_count >= 5:
                    rolling_stability['alpha_mean_deviation'] = rolling_stability['alpha_deviation'].rolling(5, min_periods=1).mean()
                    rolling_stability['beta_mean_deviation'] = rolling_stability['beta_deviation'].rolling(5, min_periods=1).mean()
                    rolling_stability['cointegration_ratio_5w'] = rolling_stability['is_cointegrated'].rolling(5, min_periods=1).mean()
        
        
        return {
            'window_details': window_details,
            'period_summary': period_summary,
            'param_stability': window_details if not window_details.empty else pd.DataFrame(),
            'trades_with_windows': pd.DataFrame(trades_with_windows) if trades_with_windows else pd.DataFrame(),
            'performance_by_window': performance_by_window,
            'spread_by_window': spread_by_window,
            'rolling_stability': rolling_stability,
            'full_sample_params': self.full_sample_params if hasattr(self, 'full_sample_params') else {}
        }
        
    def plot_rolling_correlations(self):
        """
        Plots the rolling correlations between price levels and returns.
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        
        ax.plot(self.window_dates, self.price_correlations, 
                label=r'$\rho_{t}^{\text{ price}}$', 
                linewidth=1.1, color='#2f77e4')
        
        
        ax.plot(self.window_dates, self.return_correlations, 
                label=r'$\rho_{t}^{\text{ returns}}$', 
                linewidth=1.1, color='#d62728')
        
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.95, alpha=0.7)
        
        
        ax.set_title(f"Rolling Correlations between {self.asset_A} and {self.asset_B} (Window Size: {self.window_size} days)", fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Correlation', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        
        
        ax.tick_params(axis='x', labelsize=10, labelcolor='gray')
        ax.tick_params(axis='y', labelsize=10, labelcolor='gray')
        
        
        ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=2, frameon=False)
        
        
        price_corr_stats = f"Price Corr: Mean = {np.mean(self.price_correlations):.3f}, Min = {np.min(self.price_correlations):.3f}, Max = {np.max(self.price_correlations):.3f}"
        return_corr_stats = f"Return Corr: Mean = {np.mean(self.return_correlations):.3f}, Min = {np.min(self.return_correlations):.3f}, Max = {np.max(self.return_correlations):.3f}"
        ax.text(0.02, 0.12, price_corr_stats + '\n' + return_corr_stats, transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        return None
        
    def plot_parameter_dynamics(self):
        """
        Plots the evolution of alpha and beta parameters over rolling windows.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        
        ax1.plot(self.window_dates, self.alphas, 
                label=r'$\alpha_t$ (dynamic)', linewidth=1.1, color='#2f77e4')
        ax1.axhline(y=self.full_sample_params['alpha'], linestyle='--', linewidth=0.95, 
                alpha=0.7, color='#d62728', 
                label=f'$\\alpha$ (static) = {self.full_sample_params["alpha"]:.4f}')
        
        
        alpha_mean = np.mean(self.alphas)
        alpha_std = np.std(self.alphas)
        alpha_min = np.min(self.alphas)
        alpha_max = np.max(self.alphas)
        
        
        alpha_stats = f"Mean = {alpha_mean:.4f}, Std = {alpha_std:.4f}\nMin = {alpha_min:.4f}, Max = {alpha_max:.4f}"
        ax1.text(0.83, 0.05, alpha_stats, transform=ax1.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        ax1.set_title(r"Evolution of Parameter $\alpha$ over Time", fontsize=14)
        ax1.set_ylabel(r'$\alpha$', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', labelsize=10, labelcolor='gray')
        ax1.tick_params(axis='y', labelsize=10, labelcolor='gray')
        ax1.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=2, frameon=False)
        
        
        ax2.plot(self.window_dates, self.betas, 
                label=r'$\beta_t$ (dynamic)', linewidth=1.1, color='#2f77e4')
        ax2.axhline(y=self.full_sample_params['beta'], linestyle='--', linewidth=0.95, 
                alpha=0.7, color='#d62728', 
                label=f'$\\beta$ (static) = {self.full_sample_params["beta"]:.4f}')
        
        
        beta_mean = np.mean(self.betas)
        beta_std = np.std(self.betas)
        beta_min = np.min(self.betas)
        beta_max = np.max(self.betas)
        
        
        beta_stats = f"Mean = {beta_mean:.4f}, Std = {beta_std:.4f}\nMin = {beta_min:.4f}, Max = {beta_max:.4f}"
        ax2.text(0.83, 0.90, beta_stats, transform=ax2.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        ax2.set_title(r"Evolution of Parameter $\beta$ over Time", fontsize=14)
        ax2.set_ylabel(r'$\beta$', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', labelsize=10, labelcolor='gray')
        ax2.tick_params(axis='y', labelsize=10, labelcolor='gray')
        ax2.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=2, frameon=False)
        
        
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax2.xaxis.set_major_locator(mdates.YearLocator())
        
        plt.tight_layout()
        plt.show()
        
        return None
        
    def plot_spread_comparison(self):
        """
        Compare les spreads normalisés dynamique et statique (sans le spread brut).
        Utilise des notations LaTeX cohérentes et une grille d'arrière-plan standard.
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import numpy as np
        
        
        if not hasattr(self, 'out_of_sample_spread') or self.out_of_sample_spread.empty:
            print("Exécutez d'abord compute_out_of_sample_spread() pour générer les données.")
            return
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.grid(True, alpha=0.4)
        
        valid_data = self.out_of_sample_spread.dropna()
        
        
        if 'normalized_spread' in valid_data.columns:
            ax.plot(valid_data.index, valid_data['normalized_spread'], color='#fc6a1c', 
                linewidth=1.2, label=r'Dynamic $\tilde{z}_t$ (rolling windows)', alpha=0.9)
        
        
        if hasattr(self, 'full_sample_params') and 'spread_normalized' in self.full_sample_params:
            
            static_spread = self.full_sample_params['spread_normalized'].loc[valid_data.index]
            ax.plot(valid_data.index, static_spread, color='#371b66', linewidth=1.1,
                linestyle='-', label=r'Static $\tilde{z}_t$ (full sample)', alpha=0.8)
            
            
            if 'normalized_spread' in valid_data.columns:
                corr = valid_data['normalized_spread'].corr(static_spread)
        
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.axhline(y=1, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axhline(y=-1, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        
        
        
        ax.text(valid_data.index[0], 1.1, r'$\tilde{z}_t = 1$', fontsize=9, color='gray')
        ax.text(valid_data.index[0], -1.1, r'$\tilde{z}_t = -1$', fontsize=9, color='gray')
    
        
        ax.set_title('Comparison between Dynamic and Static Normalized Spreads', fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(r'$\tilde{z}_t$', fontsize=12)
        
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        
        
        ax.tick_params(axis='x', labelsize=10, labelcolor='gray')
        ax.tick_params(axis='y', labelsize=10, labelcolor='gray')
        
        
        ax.grid(True, linestyle='-', alpha=0.3)
        
        
        legend_elements = []
        
        if 'normalized_spread' in valid_data.columns:
            legend_elements.append(plt.Line2D([0], [0], color='#fc6a1c', lw=1.2, 
                                        label=r'Dynamic $\tilde{z}_t$ (rolling windows)'))
        
        if hasattr(self, 'full_sample_params') and 'spread_normalized' in self.full_sample_params:
            legend_elements.append(plt.Line2D([0], [0], color='#9467bd', lw=1.0, 
                                        label=r'Static $\tilde{z}_t$ (full sample)'))
        
        
        legend_elements.append(plt.Line2D([0], [0], color='gray', linestyle='--', linewidth=0.8, 
                                    label=r'$\tilde{z}_t = \pm 1$'))
        
        
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), 
                fontsize=9, frameon=False, borderaxespad=0)
        
        
        stats_text = []
        
        if 'normalized_spread' in valid_data.columns:
            norm_mean = valid_data['normalized_spread'].mean()
            norm_std = valid_data['normalized_spread'].std()
            stats_text.append(f"Dynamic $\\tilde{{z}}_t$: $\\mu = {norm_mean:.2f}$, $\\sigma = {norm_std:.2f}$")
            
            
            z_above_1 = (np.abs(valid_data['normalized_spread']) > 1).mean() * 100
            stats_text.append(r'$|\tilde{z}_t| > 1$: ' + f"{z_above_1:.1f}% of time")
        
        if hasattr(self, 'full_sample_params') and 'spread_normalized' in self.full_sample_params:
            static_spread = self.full_sample_params['spread_normalized'].loc[valid_data.index]
            static_mean = static_spread.mean()
            static_std = static_spread.std()
            stats_text.append(f"Static $\\tilde{{z}}_t$: $\\mu = {static_mean:.2f}$, $\\sigma = {static_std:.2f}$")
            
            
            if 'normalized_spread' in valid_data.columns:
                stats_text.append(r'Correlation: $\rho = $' + f"{corr:.2f}")
        
        
        fig.text(0.91, 0.05, '\n'.join(stats_text), fontsize=9,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.7, edgecolor='lightgray'))
        
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.87)  
        plt.show()
        
        
        if 'normalized_spread' in valid_data.columns and hasattr(self, 'full_sample_params') and 'spread_normalized' in self.full_sample_params:
            static_spread = self.full_sample_params['spread_normalized'].loc[valid_data.index]
            corr = valid_data['normalized_spread'].corr(static_spread)
            print(f"\nCorrelation between dynamic and static normalized spread: {corr:.4f}")
            
            
            from statsmodels.tsa.stattools import adfuller
            adf_dynamic = adfuller(valid_data['normalized_spread'].dropna())
            adf_static = adfuller(static_spread.dropna())
            
            print("\nAugmented Dickey-Fuller Test:")
            print(f"Dynamic spread - p-value: {adf_dynamic[1]:.4f} {'(stationary)' if adf_dynamic[1] < 0.05 else '(non-stationary)'}")
            print(f"Static spread - p-value: {adf_static[1]:.4f} {'(stationary)' if adf_static[1] < 0.05 else '(non-stationary)'}")
            
            
            print("\nDistribution statistics:")
            print(f"Dynamic spread - Skewness: {valid_data['normalized_spread'].skew():.4f}, Kurtosis: {valid_data['normalized_spread'].kurtosis():.4f}")
            print(f"Static spread - Skewness: {static_spread.skew():.4f}, Kurtosis: {static_spread.kurtosis():.4f}")
        
    def plot_cointegration_pvalues(self, use_advanced=False, advanced_test_params=None):
        """
        Trace un graphique en tiges des p-values du test de cointegration avec un style cohérent.
        
        Args:
            use_advanced (bool): If True, uses p-values from advanced cointegration test
                                instead of basic DF test
            advanced_test_params (dict, optional): Parameters for the advanced test if use_advanced=True
                                                Default settings will be used if None
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import numpy as np
        
        
        if use_advanced:
            
            if advanced_test_params is None:
                advanced_test_params = {
                    'max_lag': None,
                    'regression_type': 'c',
                    'autolag': 'AIC',
                    'verbose': False,
                    'check_residuals_ljungbox': False
                }
            
            
            advanced_pvalues = []
            advanced_is_cointegrated = []
            
            print(f"Computing advanced p-values for {len(self.window_dates)} windows...")
            for i, window_date in enumerate(self.window_dates):
                
                if i % 20 == 0:
                    print(f"Processing window {i}/{len(self.window_dates)}...")
                    
                
                window_start = max(0, self.df.index.get_loc(window_date) - self.window_size//2)
                window_end = min(len(self.df)-1, self.df.index.get_loc(window_date) + self.window_size//2)
                window_data = self.df.iloc[window_start:window_end+1]
                
                
                p_value, is_coint, _ = self.test_cointegration_advanced(
                    window_data,
                    pvalue_threshold=0.05,
                    **advanced_test_params
                )
                advanced_pvalues.append(p_value)
                advanced_is_cointegrated.append(is_coint)
            
            
            p_values = np.array(advanced_pvalues)
            is_cointegrated = np.array(advanced_is_cointegrated)
            title_suffix = "(ADF test)"
        else:
            
            p_values = self.p_values
            is_cointegrated = np.array(self.is_cointegrated)
            title_suffix = "(Basic DF test)"
        
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        
        markerline, stemlines, baseline = ax.stem(self.window_dates, p_values, 
                                                linefmt='darkblue', markerfmt='bo', 
                                                basefmt='-', label='p-value')
        plt.setp(stemlines, linewidth=1, alpha=0.7)
        plt.setp(markerline, markersize=3)
        
        
        mask_significant = p_values < 0.05
        ax.scatter(np.array(self.window_dates)[mask_significant], 
                p_values[mask_significant], 
                color='red', s=40, label='p < 0.05')
        
        
        ax.axhline(y=0.05, linestyle='--', color='red', alpha=0.7, label='Threshold p=0.05')
        
        
        ax.set_title(f"P-values of the cointegration test per window {title_suffix}", fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('p-value', fontsize=12)
        ax.set_ylim(0, min(1.0, max(p_values) * 1.1))
        ax.grid(True, axis='y', alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.tick_params(axis='x', labelsize=10, labelcolor='gray')
        ax.tick_params(axis='y', labelsize=10, labelcolor='gray')
        
        
        ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=2, frameon=False)
        
        
        pct_coint = np.mean(is_cointegrated) * 100
        ax.text(0.22, 0.93, f"{pct_coint:.1f}% of the windows shows cointegration",
            transform=ax.transAxes, fontsize=9, ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.7, edgecolor='gray'))
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.87)  
        plt.show()
        
        
        return 

    def plot_trading_session_pvalues(self):
        """
        Trace un graphique des p-values du test de cointegration uniquement pour la période effective de trading.
        Exclut la première fenêtre qui est utilisée uniquement pour l'estimation initiale.
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import matplotlib.patches as mpatches
        import numpy as np
        
        
        if len(self.window_dates) == 0:
            print("Exécutez d'abord run_rolling_analysis() pour générer des résultats.")
            return
        
        
        trading_start_date = None
        if hasattr(self, 'out_of_sample_spread') and not self.out_of_sample_spread.dropna().empty:
            trading_start_date = self.out_of_sample_spread.dropna().index[0]
        else:
            trading_start_date = self.window_dates[1] if len(self.window_dates) > 1 else self.window_dates[0]
        
        
        window_indices = [i for i, date in enumerate(self.window_dates) if date >= trading_start_date]
        trading_window_dates = [self.window_dates[i] for i in window_indices]
        trading_p_values = self.p_values[window_indices]
        trading_is_cointegrated = [self.is_cointegrated[i] for i in window_indices]
        
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        
        markerline, stemlines, baseline = ax.stem(trading_window_dates, trading_p_values, 
                                                linefmt='darkblue', markerfmt='bo', 
                                                basefmt='-', label='p-value')
        plt.setp(stemlines, linewidth=1, alpha=0.7)
        plt.setp(markerline, markersize=4)
        
        
        mask_significant = trading_p_values < 0.05
        ax.scatter(np.array(trading_window_dates)[mask_significant], 
                trading_p_values[mask_significant], 
                color='red', s=40, label='p < 0.05')
        
        
        ax.axhline(y=0.05, linestyle='--', color='red', alpha=0.7, label='Threshold p=0.05')
        
        
        ax.set_title("Cointegration p-values during Trading Session", fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('p-value', fontsize=12)
        ax.set_ylim(0, min(1.0, max(trading_p_values) * 1.1) if len(trading_p_values) > 0 else 1.0)
        ax.grid(True, axis='y', alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.tick_params(axis='x', labelsize=10, labelcolor='gray')
        ax.tick_params(axis='y', labelsize=10, labelcolor='gray')
        
        
        ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=2, frameon=False)
        
        
        if trading_is_cointegrated:
            pct_coint_windows = np.mean(trading_is_cointegrated) * 100
            
            
            pct_coint_days = "N/A"
            if hasattr(self, 'cointegration_history') and not self.cointegration_history.empty:
                
                trading_coint_history = self.cointegration_history.loc[trading_start_date:]
                pct_coint_days = (sum(trading_coint_history['is_cointegrated']) / len(trading_coint_history) * 100)
                pct_coint_days = f"{pct_coint_days:.1f}%"
            
            
            trading_stats = (f"Trading session start: {trading_start_date.date()}\n"
                            f"Windows with cointegration: {pct_coint_windows:.1f}%\n"
                            f"Trading days with cointegration: {pct_coint_days}")
            
            
            ax.text(0.98, 0.05, trading_stats, transform=ax.transAxes, fontsize=9, ha='right',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.7, edgecolor='gray'))
        
        
        if hasattr(self, 'trades') and self.trades:
            trade_windows = set()
            for trade in self.trades:
                if 'Entry_Date' in trade and trade['Entry_Date'] in self.out_of_sample_spread.index:
                    entry_window = self.out_of_sample_spread.loc[trade['Entry_Date'], 'estimation_window']
                    if entry_window in self.window_dates:
                        window_idx = self.window_dates.index(entry_window)
                        if window_idx in window_indices:  
                            trade_windows.add(window_idx)
            
            
            for idx in trade_windows:
                if idx < len(self.window_dates):
                    date = self.window_dates[idx]
                    if date in trading_window_dates:
                        p_value = self.p_values[idx]
                        ax.scatter(date, p_value, marker='*', color='orange', s=100, 
                                zorder=5, label='Window used for trading' if len(trade_windows) == 1 else None)
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.87)  
        plt.show()
        
        
        if len(trading_p_values) > 0:
            print(f"\n===== Cointegration Statistics (Trading Session Only) =====")
            print(f"Trading session start: {trading_start_date.date()}")
            print(f"Number of estimation windows during trading: {len(trading_window_dates)}")
            print(f"Windows showing cointegration: {sum(trading_is_cointegrated)} ({pct_coint_windows:.1f}%)")
            print(f"Mean p-value: {np.mean(trading_p_values):.4f}")
            print(f"Median p-value: {np.median(trading_p_values):.4f}")
            
            
            ranges = [(0, 0.01), (0.01, 0.05), (0.05, 0.1), (0.1, 0.5), (0.5, 1.0)]
            for low, high in ranges:
                count = sum((low <= p <= high) for p in trading_p_values)
                pct = (count / len(trading_p_values)) * 100
                print(f"P-values between {low:.2f} and {high:.2f}: {count} windows ({pct:.1f}%)")
      
    def run_out_of_sample_strategy(self, z_in=1.5, L=2, W0=1000.0, z_stop=None):
        """
        Exécute la stratégie de trading sur le spread out-of-sample.
        
        Args:
            z_in (float): Seuil d'entrée pour le spread normalisé
            L (float): Levier maximum autorisé
            W0 (float): Richesse initiale
            z_stop (float, optional): Seuil de stop-loss
                
        Returns:
            dict: Résultats de la stratégie
        """
        
        self.z_in = z_in
        self.L = L
        self.W0 = W0
        self.z_stop = z_stop  
        
        
        self.trades = []
        self.wealth_history = [W0]
        self.leverage_history = [0]
        self.date_history = [self.df.index[0]]
        
        current_wealth = W0
        in_position = False
        current_position = (0, 0)  
        current_signal = None
        
        
        for i in range(1, len(self.df)):
            date = self.df.index[i]
            
            
            P_A = self.df.loc[date, self.asset_A]
            P_B = self.df.loc[date, self.asset_B]
            
            
            if pd.isna(self.out_of_sample_spread.loc[date, 'normalized_spread']):
                
                self.date_history.append(date)
                self.wealth_history.append(current_wealth)
                self.leverage_history.append(0 if not in_position else 
                                        self._calculate_leverage(current_position, date, current_wealth))
                continue
                    
            
            z_t = self.out_of_sample_spread.loc[date, 'normalized_spread']
            
            
            
            
            if in_position:
                
                if self.check_stop_loss(z_t, current_signal):
                    
                    self._close_position(date, P_A, P_B, "Stop-loss", current_position, current_wealth)
                    
                    current_wealth = self.trades[-1]["Entry_Wealth"] + self.trades[-1]["Profit"]  
                    in_position = False
                    current_position = (0, 0)
                    current_signal = None
                
                
                elif (current_signal == 'Signal 1' and z_t < 0) or \
                    (current_signal == 'Signal 2' and z_t > 0):
                    
                    self._close_position(date, P_A, P_B, "Signal crossover", current_position, current_wealth)
                    
                    current_wealth = self.trades[-1]["Entry_Wealth"] + self.trades[-1]["Profit"]  
                    in_position = False
                    current_position = (0, 0)
                    current_signal = None
            
            
            if not in_position:
                
                if z_t > z_in:
                    
                    Q_A, Q_B = self._calculate_position_size('Signal 1', P_A, P_B, current_wealth, L)
                    
                    
                    if Q_A < 0 and Q_B > 0:  
                        
                        current_position = (Q_A, Q_B)
                        in_position = True
                        current_signal = 'Signal 1'
                        
                        
                        self.trades.append({
                            'Signal': 'Signal 1',
                            'Entry_Date': date,
                            'Entry_z': z_t,
                            'P_A_entry': P_A,
                            'P_B_entry': P_B,
                            'Q_A': Q_A,
                            'Q_B': Q_B,
                            'Entry_Wealth': current_wealth  
                        })
                
                
                elif z_t < -z_in:
                    
                    Q_A, Q_B = self._calculate_position_size('Signal 2', P_A, P_B, current_wealth, L)
                    
                    
                    if Q_A > 0 and Q_B < 0:  
                        
                        current_position = (Q_A, Q_B)
                        in_position = True
                        current_signal = 'Signal 2'
                        
                        
                        self.trades.append({
                            'Signal': 'Signal 2',
                            'Entry_Date': date,
                            'Entry_z': z_t,
                            'P_A_entry': P_A,
                            'P_B_entry': P_B,
                            'Q_A': Q_A,
                            'Q_B': Q_B,
                            'Entry_Wealth': current_wealth
                        })
            
            
            if in_position:
                
                last_trade = self.trades[-1]
                P_A_entry = last_trade['P_A_entry']
                P_B_entry = last_trade['P_B_entry']
                Q_A, Q_B = current_position
                
                
                if current_signal == 'Signal 1':  
                    pnl = Q_A * (P_A_entry - P_A) + Q_B * (P_B - P_B_entry)
                else:  
                    pnl = Q_A * (P_A - P_A_entry) + Q_B * (P_B_entry - P_B)
                
                current_wealth = last_trade['Entry_Wealth'] + pnl
            
            
            self.date_history.append(date)
            self.wealth_history.append(current_wealth)
            
            
            current_leverage = 0
            if in_position:
                current_leverage = self._calculate_leverage(current_position, date, current_wealth)
                
                if current_leverage > L:
                    current_leverage = L * 0.98  
            
            self.leverage_history.append(current_leverage)
        
        
        if in_position:
            last_date = self.df.index[-1]
            P_A_last = self.df.loc[last_date, self.asset_A]
            P_B_last = self.df.loc[last_date, self.asset_B]
            self._close_position(last_date, P_A_last, P_B_last, "End of sample", current_position, current_wealth)
            current_wealth = self.trades[-1]["Entry_Wealth"] + self.trades[-1]["Profit"]  
            self.wealth_history[-1] = current_wealth  
            
        
        total_trades = len([t for t in self.trades if 'Exit_Date' in t])
        profitable_trades = len([t for t in self.trades if 'Profit' in t and t['Profit'] > 0])
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        
        self.out_of_sample_wealth = self.wealth_history
        self.out_of_sample_dates = self.date_history
        self.out_of_sample_leverage = self.leverage_history
        
        
        if 'normalized_spread' in self.out_of_sample_spread.columns:
            self.out_of_sample_spread['spread_normalized'] = self.out_of_sample_spread['normalized_spread']
        
        
        return {
            'final_wealth': self.wealth_history[-1],
            'total_return': (self.wealth_history[-1] / W0 - 1) * 100,
            'max_wealth': max(self.wealth_history),
            'min_wealth': min(self.wealth_history),
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'win_rate': win_rate,
            'max_leverage': max(self.leverage_history),
            'trades': self.trades,
            'wealth_history': self.wealth_history,
            'leverage_history': self.leverage_history,
            'date_history': self.date_history
        }

    def plot_out_of_sample_results(self):
        """
        Affiche les résultats de la stratégie out-of-sample avec un style visuel amélioré.
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import numpy as np
        import pandas as pd
        
        
        if (not hasattr(self, 'out_of_sample_wealth') or len(self.out_of_sample_wealth) == 0) and \
        (not hasattr(self, 'wealth_history') or len(self.wealth_history) == 0):
            print("Exécutez d'abord run_out_of_sample_strategy() ou run_cointegration_aware_strategy() pour générer des résultats.")
            return
        
        
        dates = getattr(self, 'out_of_sample_dates', None) or self.date_history
        wealth = getattr(self, 'out_of_sample_wealth', None) or self.wealth_history
        leverage = getattr(self, 'out_of_sample_leverage', None) or self.leverage_history
        
    
        if not hasattr(self, 'z_in'):
            print("Attention: z_in non défini, utilisation de la valeur par défaut 1.5")
            self.z_in = 1.5
        
        if not hasattr(self, 'W0'):
            print("Attention: W0 non défini, utilisation de la valeur par défaut 1000.0")
            self.W0 = 1000.0
            
        if not hasattr(self, 'L'):
            print("Attention: L non défini, utilisation de la valeur par défaut 2.0")
            self.L = 2.0
        
        
        if 'normalized_spread' in self.out_of_sample_spread.columns and 'spread_normalized' not in self.out_of_sample_spread.columns:
            self.out_of_sample_spread['spread_normalized'] = self.out_of_sample_spread['normalized_spread']
        
        
        
        
        wealth_array = np.array(wealth)
        max_wealth = np.maximum.accumulate(wealth_array)
        drawdown = (wealth_array - max_wealth) / max_wealth * 100
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        max_dd_idx = np.argmin(drawdown) if len(drawdown) > 0 else 0

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(dates, wealth, label='Wealth', color='#2f77e4', linewidth=1.1)

        ax.set_title("Evolution of Wealth ($)", fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Value ($)', fontsize=12)
        ax.axhline(y=self.W0, color='black', linestyle='--', linewidth=0.95, alpha=0.7, label='Initial Wealth')
        ax.tick_params(axis='x', labelsize=10, labelcolor='gray')
        ax.tick_params(axis='y', labelsize=10, labelcolor='gray')

        
        if max_dd_idx > 0 and max_drawdown < 0:
            max_dd_date = dates[max_dd_idx]
            
            ax.axvline(x=max_dd_date, color='red', linestyle='-', linewidth=0.4, 
                    label= r'max$^{DD} =$' f' {max_drawdown:.2f}%', alpha=0.7)
            
            
            ax.scatter(max_dd_date, wealth[max_dd_idx], color='red', marker='^', s=50, zorder=5)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=2, frameon=False)
        ax.text(0.92, 0.02, f'Final Wealth: $ {np.round(wealth[-1], 2)}', ha='center', va='center', transform=ax.transAxes, fontsize=10)
        plt.tight_layout()
        plt.show()
        
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(dates, leverage, label='Leverage', color='#d62728', linewidth=1.1)
        ax.set_title("Evolution of the Leverage", fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Leverage', fontsize=12)
        ax.axhline(y=self.L, color='black', linestyle='--', alpha=0.7, linewidth=0.95, label=f'Max: {self.L}')
        ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=2, frameon=False)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.tick_params(axis='x', labelsize=10, labelcolor='gray')
        ax.tick_params(axis='y', labelsize=10, labelcolor='gray')
        plt.tight_layout()
        plt.show()
        
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(self.out_of_sample_spread.index, self.out_of_sample_spread['spread_normalized'], 
                label=r'$\tilde{z}_t$', color='#9467bd', linewidth=0.95)
        ax.axhline(y=self.z_in, color='black', linestyle='--', alpha=0.7, 
                label=r'$\tilde{z}^{\text{in}}$' f' = {self.z_in}', linewidth=1.1)
        ax.axhline(y=-self.z_in, color='black', linestyle='--', alpha=0.7, linewidth=1.1)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.set_title(r'$\tilde{z}_t \text{ with } \tilde{z}^{\text{in}} \text{ and } \tilde{z}^{\text{out}}$', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(r'$\tilde{z}_t$', fontsize=12)
        ax.tick_params(axis='x', labelsize=10, labelcolor='gray')
        ax.tick_params(axis='y', labelsize=10, labelcolor='gray')

        
        legend_elements = [
            plt.Line2D([0], [0], color='#9467bd', lw=0.95, label=r'$\tilde{z}_t$'),
            plt.Line2D([0], [0], color='black', lw=1.1, linestyle='--', label=r'$|\tilde{z}^{\text{in}}|$' f' = {self.z_in}'),
            plt.scatter([], [], marker='v', color='red', s=45, label=r'Signal 1 (A$\downarrow$, B $\uparrow$)'),
            plt.scatter([], [], marker='^', color='green', s=45, label=r'Signal 2 (A$\uparrow$, B $\downarrow$)'),
            plt.scatter([], [], marker='x', color='black', s=45, label='Exit (Crossover 0)')
        ]

        
        if hasattr(self, 'z_stop') and self.z_stop is not None and self.z_stop != float('inf'):
            ax.axhline(y=self.z_stop, color='blue', linestyle='-.', alpha=0.7, 
                    label=r'$\tilde{z}^{\text{stop}}$' f' = {self.z_stop}', linewidth=1.1)
            ax.axhline(y=-self.z_stop, color='blue', linestyle='-.', alpha=0.7, linewidth=1.1)
            legend_elements.append(plt.Line2D([0], [0], color='blue', lw=1.1, linestyle='-.', 
                                            label=r'$\tilde{z}^{\text{stop}}$' f' = {self.z_stop}'))
            legend_elements.append(plt.scatter([], [], marker='x', color='blue', s=25, linewidth=1, label='Exit (Stop-loss)'))

        
        for trade in self.trades:
            if 'Entry_Date' in trade and trade['Entry_Date'] in self.out_of_sample_spread.index:
                if trade['Signal'] == 'Signal 1':
                    ax.scatter(trade['Entry_Date'], self.out_of_sample_spread.loc[trade['Entry_Date'], 'spread_normalized'], 
                            marker='v', color='red', s=45)
                else:
                    ax.scatter(trade['Entry_Date'], self.out_of_sample_spread.loc[trade['Entry_Date'], 'spread_normalized'], 
                            marker='^', color='green', s=45)
                        
            

            if 'Exit_Date' in trade and trade['Exit_Date'] in self.out_of_sample_spread.index:
                exit_reason = trade.get('Exit_Reason', '')
                if exit_reason == 'Signal crossover':
                    
                    ax.scatter(trade['Exit_Date'], self.out_of_sample_spread.loc[trade['Exit_Date'], 'spread_normalized'], 
                            marker='x', color='black', s=45)
                elif exit_reason == 'End of sample':
                    
                    ax.scatter(trade['Exit_Date'], self.out_of_sample_spread.loc[trade['Exit_Date'], 'spread_normalized'], 
                            marker='D', color='purple', s=45)
                else:
                    
                    ax.scatter(trade['Exit_Date'], self.out_of_sample_spread.loc[trade['Exit_Date'], 'spread_normalized'], 
                            marker='x', color='blue', s=25, linewidth=1)
                                

                    
        
        ax.legend(handles=legend_elements, fontsize=8, loc='upper left', bbox_to_anchor=(1, 1), 
                borderaxespad=2, frameon=False)

        legend_elements.append(plt.scatter([], [], marker='D', color='purple', s=45, 
                                            label='Exit (End of sample)'))
        
        plt.tight_layout()
        plt.show()
        
        bankruptcy_check = next((idx for idx, w in enumerate(wealth) if w <= 0), None)
        first_bankruptcy = 'No Bankruptcy' if bankruptcy_check is None else dates[bankruptcy_check].date()
        
        
        
        
        metrics = {
            
            'Trading Session Start': self.out_of_sample_spread.dropna().index[0].date() if not self.out_of_sample_spread.dropna().empty else 'Unknown',
            'First Trade Date': min([t['Entry_Date'].date() for t in self.trades if 'Entry_Date' in t], default="No trades"),
            'Last Trade Date': max([t['Exit_Date'].date() for t in self.trades if 'Exit_Date' in t], default="No trades"),
            'Trading Session End': self.df.index[-1].date(),
            
            
            'z_in': self.z_in,
            'z_stop': self.z_stop if hasattr(self, 'z_stop') and self.z_stop is not None else 'Not used',
            'W_0': self.W0,
            'Maximum Leverage': self.L,
            'Window Size (days)': self.window_size,
            'Step Size (days)': self.step_size,
            'Number of Windows': len(self.window_dates),
            
            
            'W_final': wealth[-1],
            'Net Profit': wealth[-1] - self.W0,
            'Return (%)': (wealth[-1] / self.W0 - 1) * 100,
            'W_max': np.max(wealth),
            'W_min': np.min(wealth),
            
            
            'Nb trades': len(self.trades),
            'Max Drawdown (%)': max_drawdown,
            'Max Leverage Used': np.max(leverage),
            'First Bankruptcy Date': first_bankruptcy
        }
        
        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])

        
        print("\n======= Performance Summary =======")
        print(metrics_df)

        
        return metrics_df

    def run_cointegration_aware_strategy(self, z_in=None, L=2, W0=1000.0, z_stop=None, p_threshold=0.05, use_advanced_test=None):
        """
        Exécute une stratégie qui trade uniquement pendant les périodes où la cointegration est détectée.
        Si la cointégration se rompt, ferme la position si ouverte et n'en ouvre pas de nouvelles.
        
        Args:
            z_in (float, optional): Seuil d'entrée pour le spread normalisé (None pour utiliser seulement le signe du spread)
            L (float): Levier maximum autorisé
            W0 (float): Richesse initiale
            z_stop (float, optional): Seuil de stop-loss (None pour désactiver)
            p_threshold (float): Seuil de p-value pour considérer la cointégration (0.01, 0.05 ou 0.10)
            use_advanced_test (dict, optional): Configuration pour utiliser le test avancé (None pour test simple)
                - max_lag (int): Nombre maximum de retards pour le test ADF
                - regression_type (str): Type de régression pour le test ADF
                
        Returns:
            dict: Résultats de la stratégie
        """
        
        self.z_in               = z_in
        self.L                  = L
        self.W0                 = W0
        self.z_stop             = z_stop
        self.p_threshold        = p_threshold
        self.use_advanced_test  = use_advanced_test
        
        
        if p_threshold not in [0.01, 0.05, 0.10]:
            print(f"Attention: p_threshold={p_threshold} non standard. Valeurs recommandées: 0.01, 0.05 ou 0.10")
        
        
        self.trades            = []
        self.wealth_history    = [W0]
        self.leverage_history  = [0]
        self.date_history      = [self.df.index[0]]
        
        
        self.cointegration_history = pd.DataFrame(index=self.df.index)
        self.cointegration_history['is_cointegrated'] = False
        self.cointegration_history['p_value'] = None
        
        current_wealth = W0
        in_position = False
        current_position = (0, 0)  
        current_signal = None
        
        
        for i in range(1, len(self.df)):
            date = self.df.index[i]
            
            
            P_A = self.df.loc[date, self.asset_A]
            P_B = self.df.loc[date, self.asset_B]
            
            
            if pd.isna(self.out_of_sample_spread.loc[date, 'normalized_spread']):
                
                self.date_history.append(date)
                self.wealth_history.append(current_wealth)
                self.leverage_history.append(0 if not in_position else 
                                        self._calculate_leverage(current_position, date, current_wealth))
                self.cointegration_history.loc[date, 'is_cointegrated'] = False
                continue
                    
            
            z_t = self.out_of_sample_spread.loc[date, 'normalized_spread']
            window_date = self.out_of_sample_spread.loc[date, 'estimation_window']
            
            
            if window_date in self.window_dates:
                
                window_idx = self.window_dates.index(window_date)
                
                if self.use_advanced_test is not None:
                    
                    window_start = max(0, self.df.index.get_loc(window_date) - self.window_size//2)
                    window_end = min(len(self.df)-1, self.df.index.get_loc(window_date) + self.window_size//2)
                    window_data = self.df.iloc[window_start:window_end+1]
                    
                    
                    max_lag = self.use_advanced_test.get('max_lag', None)
                    regression_type = self.use_advanced_test.get('regression_type', 'c')
                    
                    
                    p_value, is_cointegrated, _ = self.test_cointegration_advanced(
                        window_data, 
                        pvalue_threshold=p_threshold,
                        **self.use_advanced_test
                    )
                else:
                    
                    p_value = self.p_values[window_idx]
                    is_cointegrated = p_value < p_threshold
            else:
                
                p_value = 1.0
                is_cointegrated = False



            
            self.cointegration_history.loc[date, 'is_cointegrated'] = is_cointegrated
            self.cointegration_history.loc[date, 'p_value'] = p_value
            
            
            if in_position:
                
                if not is_cointegrated:
                    
                    self._close_position(date, P_A, P_B, "Cointegration break", current_position, current_wealth)
                    current_wealth = self.trades[-1]["Entry_Wealth"] + self.trades[-1]["Profit"]
                    in_position = False
                    current_position = (0, 0)
                    current_signal = None
                
                
                elif self.z_stop is not None and self.check_stop_loss(z_t, current_signal):
                    
                    self._close_position(date, P_A, P_B, "Stop-loss", current_position, current_wealth)
                    current_wealth = self.trades[-1]["Entry_Wealth"] + self.trades[-1]["Profit"]
                    in_position = False
                    current_position = (0, 0)
                    current_signal = None
                
                
                elif self.z_in is not None:
                    if (current_signal == 'Signal 1' and z_t < 0) or \
                    (current_signal == 'Signal 2' and z_t > 0):
                        
                        self._close_position(date, P_A, P_B, "Signal crossover", current_position, current_wealth)
                        current_wealth = self.trades[-1]["Entry_Wealth"] + self.trades[-1]["Profit"]
                        in_position = False
                        current_position = (0, 0)
                        current_signal = None
            
            
            if not in_position and is_cointegrated:
                
                signal_to_use = None
                
                
                if self.z_in is not None:
                    if z_t > self.z_in:
                        signal_to_use = 'Signal 1'  
                    elif z_t < -self.z_in:
                        signal_to_use = 'Signal 2'  
                
                
                else:
                    if z_t > 0:
                        signal_to_use = 'Signal 1'  
                    elif z_t < 0:
                        signal_to_use = 'Signal 2'  
                
                
                if signal_to_use:
                    
                    Q_A, Q_B = self._calculate_position_size(signal_to_use, P_A, P_B, current_wealth, L)
                    
                    
                    valid_position = False
                    if signal_to_use == 'Signal 1' and Q_A < 0 and Q_B > 0:
                        valid_position = True
                    elif signal_to_use == 'Signal 2' and Q_A > 0 and Q_B < 0:
                        valid_position = True
                        
                    if valid_position:
                        
                        current_position = (Q_A, Q_B)
                        in_position = True
                        current_signal = signal_to_use
                        
                        
                        self.trades.append({
                            'Signal': signal_to_use,
                            'Entry_Date': date,
                            'Entry_z': z_t,
                            'P_A_entry': P_A,
                            'P_B_entry': P_B,
                            'Q_A': Q_A,
                            'Q_B': Q_B,
                            'Entry_Wealth': current_wealth,
                            'Cointegration_pvalue': p_value
                        })
            
            
            if in_position:
                
                last_trade = self.trades[-1]
                P_A_entry = last_trade['P_A_entry']
                P_B_entry = last_trade['P_B_entry']
                Q_A, Q_B = current_position
                
                
                if current_signal == 'Signal 1':  
                    pnl = Q_A * (P_A_entry - P_A) + Q_B * (P_B - P_B_entry)
                else:  
                    pnl = Q_A * (P_A - P_A_entry) + Q_B * (P_B_entry - P_B)
                
                current_wealth = last_trade['Entry_Wealth'] + pnl
            
            
            self.date_history.append(date)
            self.wealth_history.append(current_wealth)
            
            
            current_leverage = 0
            if in_position:
                current_leverage = self._calculate_leverage(current_position, date, current_wealth)
                
                if current_leverage > L:
                    current_leverage = L * 0.98  
            
            self.leverage_history.append(current_leverage)
        
        
        if in_position:
            last_date = self.df.index[-1]
            P_A_last = self.df.loc[last_date, self.asset_A]
            P_B_last = self.df.loc[last_date, self.asset_B]
            self._close_position(last_date, P_A_last, P_B_last, "End of sample", current_position, current_wealth)
            current_wealth = self.trades[-1]["Entry_Wealth"] + self.trades[-1]["Profit"]
            
        
        self.out_of_sample_wealth = self.wealth_history
        self.out_of_sample_dates = self.date_history
        self.out_of_sample_leverage = self.leverage_history
        
        
        if 'normalized_spread' in self.out_of_sample_spread.columns:
            self.out_of_sample_spread['spread_normalized'] = self.out_of_sample_spread['normalized_spread']
        
        
        total_trades = len([t for t in self.trades if 'Exit_Date' in t])
        profitable_trades = len([t for t in self.trades if 'Profit' in t and t['Profit'] > 0])
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        
        cointegration_days = sum(self.cointegration_history['is_cointegrated'])
        total_days = len(self.cointegration_history)
        cointegration_pct = (cointegration_days / total_days * 100) if total_days > 0 else 0
        
        
        return {
            'final_wealth': self.wealth_history[-1],
            'total_return': (self.wealth_history[-1] / W0 - 1) * 100,
            'max_wealth': max(self.wealth_history),
            'min_wealth': min(self.wealth_history),
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'win_rate': win_rate,
            'max_leverage': max(self.leverage_history),
            'cointegration_days': cointegration_days,
            'cointegration_percentage': cointegration_pct,
            'trades': self.trades,
            'wealth_history': self.wealth_history,
            'leverage_history': self.leverage_history,
            'date_history': self.date_history
        }

    def plot_cointegration_aware_results(self):
        """
        Affiche les résultats de la stratégie basée sur la cointégration avec un style visuel amélioré.
        Inclut un graphique spécifique montrant les périodes de cointégration.
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import matplotlib.patches as mpatches
        import numpy as np
        
        
        if not hasattr(self, 'cointegration_history') or self.cointegration_history.empty:
            print("Exécutez d'abord run_cointegration_aware_strategy() pour générer des résultats.")
            return
        
        
        dates = self.out_of_sample_dates
        wealth = self.out_of_sample_wealth
        leverage = self.out_of_sample_leverage
        
        
        used_advanced_test = hasattr(self, 'use_advanced_test') and self.use_advanced_test is not None
        test_type_label = "ADF" if used_advanced_test else "DF"
        
        
        
        wealth_array = np.array(wealth)
        max_wealth = np.maximum.accumulate(wealth_array)
        drawdown = (wealth_array - max_wealth) / max_wealth * 100
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        max_dd_idx = np.argmin(drawdown) if len(drawdown) > 0 else 0

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(dates, wealth, label='Wealth', color='#2f77e4', linewidth=1.1)

        ax.set_title(f"Evolution of Wealth ($)", fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Value ($)', fontsize=12)
        ax.axhline(y=self.W0, color='black', linestyle='--', linewidth=0.95, alpha=0.7, label='Initial Wealth')
        ax.tick_params(axis='x', labelsize=10, labelcolor='gray')
        ax.tick_params(axis='y', labelsize=10, labelcolor='gray')

        if max_dd_idx > 0 and max_drawdown < 0:
            max_dd_date = dates[max_dd_idx]
            ax.axvline(x=max_dd_date, color='red', linestyle='-', linewidth=0.4, 
                    label= r'max$^{DD} =$' f' {max_drawdown:.2f}%', alpha=0.7)
            ax.scatter(max_dd_date, wealth[max_dd_idx], color='red', marker='^', s=50, zorder=5)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=2, frameon=False)
        ax.text(0.92, 0.02, f'Final Wealth: $ {np.round(wealth[-1], 2)}', ha='center', va='center', transform=ax.transAxes, fontsize=10)
        plt.tight_layout()
        plt.show()
        
        
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(dates, leverage, label='Leverage', color='#d62728', linewidth=1.1)
        ax.set_title(f"Evolution of the Leverage ", fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Leverage', fontsize=12)
        ax.axhline(y=self.L, color='black', linestyle='--', alpha=0.7, linewidth=0.95, label=f'Max: {self.L}')
        ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=2, frameon=False)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.tick_params(axis='x', labelsize=10, labelcolor='gray')
        ax.tick_params(axis='y', labelsize=10, labelcolor='gray')
        plt.tight_layout()
        plt.show()
        
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        
        ax1.plot(self.out_of_sample_spread.index, self.out_of_sample_spread['spread_normalized'], 
                label=r'$\tilde{z}_t$', color='#9467bd', linewidth=0.95, alpha=0.8)
        
        
        if hasattr(self, 'z_in') and self.z_in is not None:
            
            ax1.axhline(y=self.z_in, color='red', linestyle='--', alpha=0.8, 
                    label=r'$\tilde{z}^{\text{in}}$' f' = {self.z_in}', linewidth=1.5)
            ax1.axhline(y=-self.z_in, color='red', linestyle='--', alpha=0.8, linewidth=1.5)
            
            
            ax1.text(self.out_of_sample_spread.index[10], self.z_in + 0.2, f'z_in = {self.z_in}', 
                    fontsize=9, color='red', fontweight='bold')
            ax1.text(self.out_of_sample_spread.index[10], -self.z_in - 0.3, f'z_in = -{self.z_in}', 
                    fontsize=9, color='red', fontweight='bold')
        else:
            
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)
            
            
            ax1.text(0.05, 0.95, 'Trading based on cointegration status only', 
                    fontsize=10, color='black', fontweight='bold', 
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.7, edgecolor='gray'),
                    horizontalalignment='left', verticalalignment='top',
                    transform=ax1.transAxes)
        
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        ax1.set_title(r'Normalized Spread $\tilde{z}_t$ and Cointegration Status', fontsize=14)
        ax1.set_ylabel(r'$\tilde{z}_t$', fontsize=12)
        ax1.tick_params(axis='x', labelsize=10, labelcolor='gray')
        ax1.tick_params(axis='y', labelsize=10, labelcolor='gray')
        
        
        
        coint_status_series = self.cointegration_history['is_cointegrated'].astype(int)
        
        
        for i in range(len(self.cointegration_history)-1):
            date = self.cointegration_history.index[i]
            next_date = self.cointegration_history.index[i+1]
            is_coint = self.cointegration_history.loc[date, 'is_cointegrated']
            
            
            if is_coint:
                ax2.fill_between([date, next_date], 0, 1, color='green', alpha=0.03)  
            else:
                ax2.fill_between([date, next_date], 0, 1, color='red', alpha=0.03)    
        
        
        ax2.step(self.cointegration_history.index, coint_status_series, 
                where='post', color='gray', alpha=0.5, linewidth=0.1)
        
        
        ax2.set_yticks([])
        
        
        ax2_twin = ax2.twinx()
        valid_pvalues = self.cointegration_history[~self.cointegration_history['p_value'].isna()]
        ax2_twin.plot(valid_pvalues.index, valid_pvalues['p_value'], color='blue', linewidth=1.1, alpha=0.6, 
                    label=f'p-value ({test_type_label} test)')  
        ax2_twin.axhline(y=self.p_threshold, color='black', linestyle='--', alpha=0.7, linewidth=0.8, 
                        label=f'Threshold: {self.p_threshold}')
        
        
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Cointegration Status', fontsize=12)
        ax2_twin.set_ylabel('p-value', fontsize=10, color='blue')
        ax2_twin.tick_params(axis='y', labelcolor='blue')
        
        
        for trade in self.trades:
            if 'Entry_Date' in trade and trade['Entry_Date'] in self.out_of_sample_spread.index:
                if trade['Signal'] == 'Signal 1':
                    ax1.scatter(trade['Entry_Date'], self.out_of_sample_spread.loc[trade['Entry_Date'], 'spread_normalized'], 
                            marker='v', color='red', s=45)
                else:
                    ax1.scatter(trade['Entry_Date'], self.out_of_sample_spread.loc[trade['Entry_Date'], 'spread_normalized'], 
                            marker='^', color='green', s=45)

            if 'Exit_Date' in trade and trade['Exit_Date'] in self.out_of_sample_spread.index:
                exit_reason = trade.get('Exit_Reason', '')
                if exit_reason == 'Signal crossover':
                    ax1.scatter(trade['Exit_Date'], self.out_of_sample_spread.loc[trade['Exit_Date'], 'spread_normalized'], 
                            marker='x', color='black', s=45)
                elif exit_reason == 'Stop-loss':
                    ax1.scatter(trade['Exit_Date'], self.out_of_sample_spread.loc[trade['Exit_Date'], 'spread_normalized'], 
                            marker='x', color='blue', s=25, linewidth=1)
                elif exit_reason == 'Cointegration break':
                    ax1.scatter(trade['Exit_Date'], self.out_of_sample_spread.loc[trade['Exit_Date'], 'spread_normalized'], 
                            marker='s', color='purple', s=45, linewidth=1)
                elif exit_reason == 'End of sample':
                    
                    ax1.scatter(trade['Exit_Date'], self.out_of_sample_spread.loc[trade['Exit_Date'], 'spread_normalized'], 
                            marker='D', color='orange', s=45)
                else:
                    
                    ax1.scatter(trade['Exit_Date'], self.out_of_sample_spread.loc[trade['Exit_Date'], 'spread_normalized'], 
                            marker='s', color='gray', s=35)

                            
        
        ax1_legend = [
            plt.Line2D([0], [0], color='#9467bd', lw=0.95, label=r'$\tilde{z}_t$'),
            plt.scatter([], [], marker='v', color='red', s=45, label=r'Signal 1 (A$\downarrow$, B $\uparrow$)'),
            plt.scatter([], [], marker='^', color='green', s=45, label=r'Signal 2 (A$\uparrow$, B $\downarrow$)'),
            plt.scatter([], [], marker='s', color='purple', s=45, label='Exit (Cointegration Break)')
        ]
        
        ax1_legend.append(plt.scatter([], [], marker='D', color='orange', s=45, 
                                        label='Exit (End of sample)'))
        
        
        if hasattr(self, 'z_in') and self.z_in is not None:
            ax1_legend.append(plt.scatter([], [], marker='x', color='black', s=45, label='Exit (Crossover 0)'))
            ax1_legend.append(plt.Line2D([0], [0], color='red', lw=1.5, linestyle='--', 
                                        label=r'$|\tilde{z}^{\text{in}}|$' f' = {self.z_in}'))
        
        
        if hasattr(self, 'z_stop') and self.z_stop is not None and self.z_stop != float('inf'):
            ax1.axhline(y=self.z_stop, color='blue', linestyle='-.', alpha=0.7, 
                    label=r'$\tilde{z}^{\text{stop}}$' f' = {self.z_stop}', linewidth=1.1)
            ax1.axhline(y=-self.z_stop, color='blue', linestyle='-.', alpha=0.7, linewidth=1.1)
            ax1_legend.append(plt.Line2D([0], [0], color='blue', lw=1.1, linestyle='-.', 
                                        label=r'$\tilde{z}^{\text{stop}}$' f' = {self.z_stop}'))
            ax1_legend.append(plt.scatter([], [], marker='x', color='blue', s=25, linewidth=1, label='Exit (Stop-loss)'))
        
        
        ax1.legend(handles=ax1_legend, fontsize=8, loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=2, frameon=False)
        
        
        
        legend_elements = [
            mpatches.Patch(color='green', alpha=0.15, label='Cointegrated'),
            mpatches.Patch(color='red', alpha=0.15, label='Not Cointegrated'),
            plt.Line2D([0], [0], color='blue', lw=0.8, alpha=0.6, label=f'p-value ({test_type_label} test)'),
            plt.Line2D([0], [0], color='black', lw=0.8, linestyle='--', alpha=0.7, label=f'Threshold: {self.p_threshold}')
        ]
        
        
        ax2.legend(handles=legend_elements, fontsize=8, loc='upper left', 
                bbox_to_anchor=(1.05, 0.45), borderaxespad=0, frameon=False)
        
        
        cointegration_days = self.cointegration_history['is_cointegrated'].sum()
        total_days = len(self.cointegration_history)
        cointegration_pct = (cointegration_days / total_days * 100) if total_days > 0 else 0
        
        
        fig.text(0.089, 0.248, f"Cointegration: {cointegration_pct:.1f}% of days\np-threshold: {self.p_threshold}\nTest type: {test_type_label}",
                fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.4))
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.87)  
        plt.show()
        
        
        bankruptcy_check = next((idx for idx, w in enumerate(wealth) if w <= 0), None)
        first_bankruptcy = 'No Bankruptcy' if bankruptcy_check is None else dates[bankruptcy_check].date()
        
        
        cointegration_changes = (self.cointegration_history['is_cointegrated'].diff() != 0).sum()
        cointegration_days = self.cointegration_history['is_cointegrated'].sum()
        total_days = len(self.cointegration_history)
        cointegration_pct = (cointegration_days / total_days * 100) if total_days > 0 else 0
        cointegration_break_exits = sum(1 for trade in self.trades 
                                    if 'Exit_Reason' in trade and trade['Exit_Reason'] == 'Cointegration break')
        
        
        metrics = {
            
            'Trading Session Start': self.out_of_sample_spread.dropna().index[0].date() if not self.out_of_sample_spread.dropna().empty else 'Unknown',
            'First Trade Date': min([t['Entry_Date'].date() for t in self.trades if 'Entry_Date' in t], default="No trades"),
            'Last Trade Date': max([t['Exit_Date'].date() for t in self.trades if 'Exit_Date' in t], default="No trades"),
            'Trading Session End': self.df.index[-1].date(),
            
            
            'p_threshold': self.p_threshold,
            'z_in': self.z_in if hasattr(self, 'z_in') else None,
            'z_stop': self.z_stop if hasattr(self, 'z_stop') and self.z_stop is not None else 'Not used',
            'W_0': self.W0,
            'Maximum Leverage': self.L,
            'Window Size (days)': self.window_size,
            'Step Size (days)': self.step_size,
            'Number of Windows': len(self.window_dates),
            
            
            'W_final': wealth[-1],
            'Net Profit': wealth[-1] - self.W0,
            'Return (%)': (wealth[-1] / self.W0 - 1) * 100,
            'W_max': np.max(wealth),
            'W_min': np.min(wealth),
            
            
            'Nb trades': len(self.trades),
            'Max Drawdown (%)': max_drawdown,
            'Max Leverage Used': np.max(leverage),
            'First Bankruptcy Date': first_bankruptcy,
            
            
            'Coint. Days': cointegration_days,
            'Coint. % of trading days': cointegration_pct,
            'Coint. regime Changes': cointegration_changes,
            'Exits due to Coint. Break': cointegration_break_exits,
            'Test Type': test_type_label
        }
        
        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        
        
        print("\n===== Cointegration-Aware Strategy Performance =====")
        print(f"p-value threshold: {self.p_threshold}")
        print(f"Test type: {test_type_label}")

        print("\nCointegration Statistics:")
        print(f"- Cointegrated days: {cointegration_days} ({cointegration_pct:.2f}% of trading days)")
        print(f"- Cointegration regime changes: {cointegration_changes}")
        print(f"- Positions closed due to cointegration breaks: {cointegration_break_exits}")

        print("\n===== Performance Summary =====")
        print(metrics_df)

        return metrics_df

    def plot_trading_results(self, strategy_type="out-of-sample"):
        """
        Affiche les résultats de la stratégie de trading.
        
        Args:
            strategy_type (str): Type de stratégie ('out-of-sample' ou 'cointegration-aware')
        """
        if not self.wealth_history:
            print("Aucune stratégie exécutée. Utilisez d'abord run_out_of_sample_strategy() ou run_cointegration_aware_strategy().")
            return
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 16), sharex=True)
        
        
        ax1.plot(self.date_history, self.wealth_history, color='blue')
        ax1.set_title(f"Évolution de la richesse ({strategy_type})", fontsize=14)
        ax1.set_ylabel('Richesse ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        
        for trade in self.trades:
            if 'Entry_Date' in trade and 'Exit_Date' in trade:
                
                entry_idx = self.date_history.index(trade['Entry_Date']) if trade['Entry_Date'] in self.date_history else None
                exit_idx = self.date_history.index(trade['Exit_Date']) if trade['Exit_Date'] in self.date_history else None
                
                if entry_idx is not None:
                    
                    color = 'green' if trade.get('Profit', 0) > 0 else 'red'
                    ax1.scatter(trade['Entry_Date'], self.wealth_history[entry_idx], 
                            marker='^', color=color, s=50)
                
                if exit_idx is not None:
                    
                    color = 'green' if trade.get('Profit', 0) > 0 else 'red'
                    ax1.scatter(trade['Exit_Date'], self.wealth_history[exit_idx], 
                            marker='v', color=color, s=50)
        
        
        ax2.plot(self.date_history, self.leverage_history, color='purple')
        ax2.set_title("Évolution du levier", fontsize=14)
        ax2.set_ylabel('Levier', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        
        mask = ~self.out_of_sample_spread['normalized_spread'].isna()
        ax3.plot(self.out_of_sample_spread[mask].index, 
                self.out_of_sample_spread.loc[mask, 'normalized_spread'],
                color='black', label='Spread normalisé')
        
        
        ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax3.axhline(y=1.5, color='blue', linestyle='--', alpha=0.7, label='z_in')
        ax3.axhline(y=-1.5, color='blue', linestyle='--', alpha=0.7)
        
        
        if strategy_type == 'cointegration-aware':
            
            coint_mask = self.out_of_sample_spread['is_cointegrated'] & mask
            
            
            dates = self.out_of_sample_spread.index[coint_mask]
            if len(dates) > 0:
                for i in range(0, len(dates), 10):
                    block = dates[i:i+10]
                    if len(block) > 1:
                        ax3.axvspan(block[0], block[-1], alpha=0.2, color='green', label='Cointegrated' if i == 0 else '')
        
        ax3.set_title("Spread normalisé out-of-sample", fontsize=14)
        ax3.set_ylabel('Spread normalisé', fontsize=12)
        ax3.set_xlabel('Date', fontsize=12)
        ax3.legend(fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        
        plt.gcf().autofmt_xdate()
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        
        plt.tight_layout()
        plt.show()

    def calculate_and_plot_ljung_box_evolution(self, lags=[5, 10, 15], alpha=0.05):
        """
        Calcule et affiche l'évolution des statistiques du test de Ljung-Box pour chaque fenêtre.
        
        Parameters:
        -----------
        lags : list
            Liste des retards pour lesquels calculer les statistiques de Ljung-Box
        alpha : float
            Niveau de significativité pour le test (par défaut: 0.05)
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame contenant les statistiques de Ljung-Box pour chaque fenêtre
        """
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from statsmodels.stats.diagnostic import acorr_ljungbox
        
        
        if not hasattr(self, 'window_dates') or len(self.window_dates) == 0:
            print("Exécutez d'abord run_rolling_analysis() pour générer des résultats.")
            return
        
        
        lb_statistics = pd.DataFrame(index=range(len(self.window_dates)))
        lb_statistics['window_date'] = self.window_dates
        
        for lag in lags:
            lb_statistics[f'lb_stat_lag{lag}'] = np.nan
            lb_statistics[f'lb_pvalue_lag{lag}'] = np.nan
            lb_statistics[f'lb_significant_lag{lag}'] = False
        
        
        for i, window_date in enumerate(self.window_dates):
            
            start_idx = max(0, self.df.index.get_loc(window_date) - self.window_size//2)
            end_idx = min(len(self.df)-1, self.df.index.get_loc(window_date) + self.window_size//2)
            start_date = self.df.index[start_idx]
            end_date = self.df.index[end_idx]
            
            
            window_data = self.df.loc[start_date:end_date].iloc[:self.window_size]
            
            
            alpha, beta = self.estimate_parameters(window_data)
            
            
            spread = window_data[self.asset_A] - alpha - beta * window_data[self.asset_B]
            
            
            for lag in lags:
                try:
                    
                    if len(spread) > lag:
                        lb_result = acorr_ljungbox(spread, lags=[lag], return_df=True)
                        lb_statistics.loc[i, f'lb_stat_lag{lag}'] = lb_result['lb_stat'].iloc[0]
                        lb_statistics.loc[i, f'lb_pvalue_lag{lag}'] = lb_result['lb_pvalue'].iloc[0]
                        lb_statistics.loc[i, f'lb_significant_lag{lag}'] = lb_result['lb_pvalue'].iloc[0] < alpha
                except Exception as e:
                    print(f"Erreur lors du calcul du test de Ljung-Box pour la fenêtre {i}, lag {lag}: {e}")
        
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
        
        
        for lag in lags:
            ax1.plot(lb_statistics['window_date'], lb_statistics[f'lb_stat_lag{lag}'],
                    label=f'Lag {lag}', linewidth=1.1)
        
        ax1.set_title(f"Évolution des statistiques de Ljung-Box (Fenêtre: {self.window_size} jours)", fontsize=14)
        ax1.set_ylabel("Statistique Q", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        
        for lag in lags:
            ax2.plot(lb_statistics['window_date'], lb_statistics[f'lb_pvalue_lag{lag}'],
                    label=f'Lag {lag}', linewidth=1.1)
        
        ax2.axhline(y=alpha, color='red', linestyle='--', label=f'Seuil α={alpha}')
        ax2.set_title("P-valeurs du test de Ljung-Box", fontsize=14)
        ax2.set_xlabel("Date", fontsize=12)
        ax2.set_ylabel("P-valeur", fontsize=12)
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        
        
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax2.xaxis.set_major_locator(mdates.YearLocator())
        
        
        fig.text(0.5, 0.01, 
                "L'hypothèse nulle du test de Ljung-Box est l'absence d'autocorrélation jusqu'au lag spécifié.\n"
                f"P-valeur < {alpha} indique le rejet de H0 et la présence d'autocorrélation significative.",
                ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        plt.show()
        
        
        for lag in lags:
            significant_count = lb_statistics[f'lb_significant_lag{lag}'].sum()
            significant_pct = (significant_count / len(lb_statistics)) * 100
            print(f"Lag {lag}: {significant_pct:.1f}% des fenêtres montrent une autocorrélation significative")
        
        return lb_statistics
        
    def _calculate_position_size(self, signal_type, P_A, P_B, W_current, L):
        """
        Calcule la taille de position optimale en s'assurant que le levier respecte la limite.
        Continue trading even with negative wealth.
        """
        
        
        W_for_calculation = abs(W_current)
        
        alpha_sign = 1 if self.full_sample_params['alpha'] > 0 else -1
        beta = self.full_sample_params['beta']
        
        
        leverage_safety = 0.98
        
        if signal_type == "Signal 1":  
            if alpha_sign > 0:
                
                factor = (L * leverage_safety * W_for_calculation) / P_A
                Q_A = -1.0 * factor
                Q_B = beta * factor
            else:
                
                denom = P_A - L * leverage_safety * (P_A - beta * P_B)
                factor = (L * leverage_safety * W_for_calculation) / denom if denom != 0 else 0
                Q_A = -1.0 * factor
                Q_B = beta * factor

        elif signal_type == "Signal 2":  
            if alpha_sign > 0:
                
                denom = beta * P_B + L * leverage_safety * (P_A - beta * P_B)
                factor = (L * leverage_safety * W_for_calculation) / denom if denom != 0 else 0
                Q_A = 1.0 * factor
                Q_B = -beta * factor
            else:
                
                denom = beta * P_B
                factor = (L * leverage_safety * W_for_calculation) / denom if denom != 0 else 0
                Q_A = 1.0 * factor
                Q_B = -beta * factor
        else:
            Q_A, Q_B = 0.0, 0.0
        
        
        exposure = abs(Q_A * P_A) + abs(Q_B * P_B)
        current_leverage = exposure / W_for_calculation if W_for_calculation > 0 else 0
        
        if current_leverage > L:
            
            adjustment_ratio = (L * leverage_safety) / current_leverage
            Q_A *= adjustment_ratio
            Q_B *= adjustment_ratio
        
        return Q_A, Q_B
    
    def _enter_position(self, date, signal_type, Q_A, Q_B, P_A, P_B, W_current):
        """
        Enregistre l'entrée en position et met à jour l'état.
        """
        self.in_position = True
        self.current_position = (Q_A, Q_B)
        self.current_signal = signal_type
        
        
        self.trades.append({
            "Entry_Date": date,
            "Entry_Wealth": W_current,
            "Signal": signal_type,
            "Q_A": Q_A,
            "Q_B": Q_B,
            "P_A_entry": P_A,
            "P_B_entry": P_B
        })

    def _close_position(self, date, P_A_exit, P_B_exit, reason="Signal crossover", current_position=None, current_wealth=None):
        """
        Ferme la position actuelle, calcule le profit et met à jour la richesse.
        
        Args:
            date: Date de sortie
            P_A_exit, P_B_exit: Prix de sortie
            reason: Raison de la sortie ("Signal crossover", "Stop-loss", etc.)
            current_position: Position actuelle (Q_A, Q_B)
            current_wealth: Richesse actuelle
        """
        
        if not self.trades:
            return
        
        
        last_trade = self.trades[-1]
        
        
        if 'Exit_Date' not in last_trade:
            (Q_A, Q_B) = current_position
            signal_type = last_trade["Signal"]
            P_A_entry = last_trade["P_A_entry"]
            P_B_entry = last_trade["P_B_entry"]
            
            
            if signal_type == "Signal 1":  
                pnl = Q_A * (P_A_entry - P_A_exit) + Q_B * (P_B_exit - P_B_entry)
            else:  
                pnl = Q_A * (P_A_exit - P_A_entry) + Q_B * (P_B_entry - P_B_exit)
            
            
            self.trades[-1].update({
                "Exit_Date": date,
                "P_A_exit": P_A_exit,
                "P_B_exit": P_B_exit,
                "Profit": pnl,
                "Return": (pnl / last_trade["Entry_Wealth"]) * 100 if last_trade["Entry_Wealth"] > 0 else 0,
                "Exit_Reason": reason
            })
        
    def check_stop_loss(self, z_t, signal_type):
        """
        Vérifie si le spread normalisé atteint le niveau de stop-loss défini.
        
        Args:
            z_t (float): Valeur actuelle du spread normalisé
            signal_type (str): Type de signal ('Signal 1' ou 'Signal 2')
            
        Returns:
            bool: True si le stop-loss est déclenché, False sinon
        """
        
        if self.z_stop is None:
            return False
        
        
        if signal_type == "Signal 1" and z_t > self.z_stop:
            return True
        
        
        if signal_type == "Signal 2" and z_t < -self.z_stop:
            return True
        
        return False

    def _calculate_leverage(self, position, date, W_current):
        """
        Calcule le levier actuel de la position.
        
        Args:
            position (tuple): Position actuelle (Q_A, Q_B)
            date (datetime): Date courante
            W_current (float): Richesse actuelle
            
        Returns:
            float: Levier actuel
        """
        Q_A, Q_B = position
        P_A = self.df.loc[date, self.asset_A]
        P_B = self.df.loc[date, self.asset_B]
        
        
        exposure = abs(Q_A * P_A) + abs(Q_B * P_B)
        
        
        leverage = exposure / W_current if W_current > 0 else 0
        
        return leverage

    def strategy_data(self):
        """
        Retourne un dictionnaire contenant toutes les données importantes pour l'audit et l'analyse approfondie,
        avec une séparation claire entre les deux stratégies possibles.
        
        Returns:
            dict: Données complètes de l'analyse et des stratégies
        """
        
        
        window_df = pd.DataFrame({
            'window_date': self.window_dates,
            'alpha': self.alphas,
            'beta': self.betas,
            'price_correlation': self.price_correlations,
            'return_correlation': self.return_correlations,
            'p_value': self.p_values,
            'is_cointegrated': self.is_cointegrated
        })
        
        
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        
        wealth_history_df = pd.DataFrame({
            'date': self.date_history,
            'wealth': self.wealth_history,
            'leverage': self.leverage_history
        }) if self.date_history else pd.DataFrame()
        
        
        if len(self.wealth_history) > 0:
            wealth_array = np.array(self.wealth_history)
            max_wealth = np.maximum.accumulate(wealth_array)
            drawdown = (wealth_array - max_wealth) / max_wealth * 100
            wealth_history_df['drawdown'] = drawdown
            wealth_history_df['max_drawdown_so_far'] = np.minimum.accumulate(drawdown)
        
        
        spreads_df = pd.DataFrame({
            'date': self.out_of_sample_spread.index,
            'out_of_sample_spread': self.out_of_sample_spread['normalized_spread'],
        })
        
        
        if 'is_cointegrated' in self.out_of_sample_spread.columns:
            spreads_df['is_cointegrated'] = self.out_of_sample_spread['is_cointegrated']
        
        if 'estimation_window' in self.out_of_sample_spread.columns:
            spreads_df['estimation_window'] = self.out_of_sample_spread['estimation_window']
        
        
        if hasattr(self, 'full_sample_params') and 'spread_normalized' in self.full_sample_params:
            spreads_df['in_sample_spread'] = self.full_sample_params['spread_normalized']
        
        
        window_stats = {
            'window_size': self.window_size,
            'step_size': self.step_size,
            'nb_windows': len(self.window_dates),
            'window_coverage': f"{self.df.index[0].date()} - {self.df.index[-1].date()}",
            'cointegrated_pct': np.mean(self.is_cointegrated) * 100 if len(self.is_cointegrated) > 0 else 0,
            'min_pvalue': np.min(self.p_values) if len(self.p_values) > 0 else None,
            'max_pvalue': np.max(self.p_values) if len(self.p_values) > 0 else None,
            'mean_pvalue': np.mean(self.p_values) if len(self.p_values) > 0 else None,
            'median_pvalue': np.median(self.p_values) if len(self.p_values) > 0 else None,
            'mean_alpha': np.mean(self.alphas) if len(self.alphas) > 0 else None,
            'mean_beta': np.mean(self.betas) if len(self.betas) > 0 else None,
            'std_alpha': np.std(self.alphas) if len(self.alphas) > 0 else None,
            'std_beta': np.std(self.betas) if len(self.betas) > 0 else None,
        }
        
        
        out_of_sample_stats = {}
        
        
        has_run_out_of_sample = hasattr(self, 'out_of_sample_wealth') and len(getattr(self, 'out_of_sample_wealth', [])) > 0
        
        if has_run_out_of_sample and self.trades:
            
            os_trades = [t for t in self.trades if 'Cointegration_pvalue' not in t]
            
            if os_trades:  
                total_trades = len([t for t in os_trades if 'Exit_Date' in t])
                profitable_trades = len([t for t in os_trades if 'Profit' in t and t['Profit'] > 0])
                win_rate = profitable_trades / total_trades if total_trades > 0 else 0
                
                
                if len(self.wealth_history) > 0:
                    wealth_array = np.array(self.wealth_history)
                    max_wealth = np.maximum.accumulate(wealth_array)
                    drawdown = (wealth_array - max_wealth) / max_wealth * 100
                    max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
                    max_dd_idx = np.argmin(drawdown) if len(drawdown) > 0 else 0
                    max_dd_date = self.date_history[max_dd_idx] if max_dd_idx < len(self.date_history) else None
                else:
                    max_drawdown = 0
                    max_dd_date = None
                
                out_of_sample_stats = {
                    'z_in': self.z_in if hasattr(self, 'z_in') else None,
                    'W0': self.W0 if hasattr(self, 'W0') else None,
                    'L': self.L if hasattr(self, 'L') else None,
                    'z_stop': self.z_stop if hasattr(self, 'z_stop') else None,
                    'total_trades': total_trades,
                    'profitable_trades': profitable_trades,
                    'win_rate': win_rate * 100,
                    'final_wealth': self.wealth_history[-1] if self.wealth_history else None,
                    'total_return': ((self.wealth_history[-1] / self.W0) - 1) * 100 if self.wealth_history and hasattr(self, 'W0') and self.W0 != 0 else None,
                    'max_wealth': max(self.wealth_history) if self.wealth_history else None,
                    'min_wealth': min(self.wealth_history) if self.wealth_history else None,
                    'max_leverage': max(self.leverage_history) if self.leverage_history else None,
                    'max_drawdown': max_drawdown,
                    'max_drawdown_date': max_dd_date,
                    'bankruptcy': next((idx for idx, w in enumerate(self.wealth_history) if w <= 0), None) if self.wealth_history else None
                }
        
        
        cointegration_aware_stats = {}
        
        
        has_run_cointegration_aware = (hasattr(self, 'cointegration_history') and 
                                    not getattr(self, 'cointegration_history', pd.DataFrame()).empty)
        
        if has_run_cointegration_aware:
            
            ca_trades = [t for t in self.trades if 'Cointegration_pvalue' in t] if self.trades else []
            
            
            total_trades = len([t for t in ca_trades if 'Exit_Date' in t])
            profitable_trades = len([t for t in ca_trades if 'Profit' in t and t['Profit'] > 0])
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0
            
            
            if len(self.wealth_history) > 0:
                wealth_array = np.array(self.wealth_history)
                max_wealth = np.maximum.accumulate(wealth_array)
                drawdown = (wealth_array - max_wealth) / max_wealth * 100
                max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
                max_dd_idx = np.argmin(drawdown) if len(drawdown) > 0 else 0
                max_dd_date = self.date_history[max_dd_idx] if max_dd_idx < len(self.date_history) else None
            else:
                max_drawdown = 0
                max_dd_date = None
            
            
            cointegration_days = self.cointegration_history['is_cointegrated'].sum()
            total_days = len(self.cointegration_history)
            cointegration_pct = (cointegration_days / total_days * 100) if total_days > 0 else 0
            cointegration_changes = (self.cointegration_history['is_cointegrated'].diff() != 0).sum()
            
            cointegration_aware_stats = {
                'z_in': self.z_in if hasattr(self, 'z_in') else None,
                'W0': self.W0 if hasattr(self, 'W0') else None,
                'L': self.L if hasattr(self, 'L') else None,
                'z_stop': self.z_stop if hasattr(self, 'z_stop') else None,
                'p_threshold': self.p_threshold if hasattr(self, 'p_threshold') else None,
                'total_trades': total_trades,
                'profitable_trades': profitable_trades,
                'win_rate': win_rate * 100,
                'final_wealth': self.wealth_history[-1] if self.wealth_history else None,
                'total_return': ((self.wealth_history[-1] / self.W0) - 1) * 100 if self.wealth_history and hasattr(self, 'W0') and self.W0 != 0 else None,
                'max_wealth': max(self.wealth_history) if self.wealth_history else None,
                'min_wealth': min(self.wealth_history) if self.wealth_history else None,
                'max_leverage': max(self.leverage_history) if self.leverage_history else None,
                'max_drawdown': max_drawdown,
                'max_drawdown_date': max_dd_date,
                'bankruptcy': next((idx for idx, w in enumerate(self.wealth_history) if w <= 0), None) if self.wealth_history else None,
                'cointegration_days': cointegration_days,
                'cointegration_percentage': cointegration_pct,
                'cointegration_changes': cointegration_changes,
                'exits_due_to_coint_break': sum(1 for t in ca_trades if t.get('Exit_Reason') == 'Cointegration break')
            }
        
        
        result = {
            'window_df': window_df,
            'trades_df': trades_df,
            'wealth_history_df': wealth_history_df,
            'spreads_df': spreads_df,
            'full_sample_params': self.full_sample_params if hasattr(self, 'full_sample_params') else {},
            'window_stats': window_stats,
            'out_of_sample_stats': out_of_sample_stats,
            'cointegration_aware_stats': cointegration_aware_stats
        }
        
        
        if has_run_cointegration_aware:
            result['cointegration_history_df'] = self.cointegration_history.copy()
        
        return result
    
    def display_strategy_data(self, show_all=False, max_rows=10, show_plots=False):
        """
        Affiche les données de stratégie sans utiliser de codes de couleur ANSI.
        
        Parameters:
        -----------
        show_all : bool, default=False
            Si True, affiche tous les dataframes en entier, sinon montre seulement les premières lignes
        max_rows : int, default=10
            Nombre de lignes à afficher quand show_all=False
        show_plots : bool, default=False
            Si True, génère des visualisations des données principales
        """
        import pandas as pd
        import numpy as np
        
        def format_table(df, max_rows=10, show_all=False):
            
            if df.empty:
                return "Empty DataFrame"
            
            original_max_rows = pd.get_option('display.max_rows')
            pd.set_option('display.max_columns', None)
            
            if show_all:
                pd.set_option('display.max_rows', None)
            else:
                pd.set_option('display.max_rows', max_rows)
            
            df_str = df.__str__()
            
            pd.set_option('display.max_rows', original_max_rows)
            
            return df_str
        
        strategy_data_dict = self.strategy_data()
        
        has_oos_strategy = hasattr(self, 'out_of_sample_wealth') and hasattr(self, 'wealth_history')
        has_coint_strategy = hasattr(self, 'cointegration_history') and not getattr(self, 'cointegration_history', pd.DataFrame()).empty
        
        print("\n" + "=" * 100)
        print("TRADING STRATEGY ANALYSIS REPORT")
        print("=" * 100)
        print(f"Asset Pair: {self.asset_A}/{self.asset_B}")
        print(f"Data Range: {min(self.df.index).date()} to {max(self.df.index).date()}")
        print("=" * 100)
        
        print("\n" + "=" * 100)
        print("ANALYSIS CONFIGURATION")
        print("=" * 100)
        
        if 'window_stats' in strategy_data_dict:
            ws = strategy_data_dict['window_stats']
            print("\nROLLING WINDOW PARAMETERS:")
            print(f"  Window Size: {ws.get('window_size', 'N/A')} days")
            print(f"  Step Size: {ws.get('step_size', 'N/A')} days")
            print(f"  Number of Windows: {ws.get('nb_windows', 0):,}")
            print(f"  Analysis Period: {ws.get('window_coverage', 'N/A')}")

        print("\n" + "=" * 100)
        print("COINTEGRATION ANALYSIS")
        print("=" * 100)

        if 'window_stats' in strategy_data_dict:
            ws = strategy_data_dict['window_stats']
            print("\nPARAMETERS ESTIMATION:")
            
            mean_alpha = ws.get("mean_alpha")
            std_alpha = ws.get("std_alpha")
            mean_beta = ws.get("mean_beta")
            std_beta = ws.get("std_beta")
            
            # Correction : utiliser des variables intermédiaires pour le formatage conditionnel
            alpha_str = f"{mean_alpha:.4f}" if mean_alpha is not None else "N/A"
            alpha_std_str = f"{std_alpha:.4f}" if std_alpha is not None else "N/A"
            print(f"  Mean Alpha (α): {alpha_str} (std: {alpha_std_str})")
            
            beta_str = f"{mean_beta:.4f}" if mean_beta is not None else "N/A"
            beta_std_str = f"{std_beta:.4f}" if std_beta is not None else "N/A"
            print(f"  Mean Beta (β): {beta_str} (std: {beta_std_str})")
        
            print("\nCOINTEGRATION STATISTICS:")
            coint_pct = ws.get('cointegrated_pct', 0)
            print(f"  Cointegrated Windows: {coint_pct:.2f}%")
            
            mean_pvalue = ws.get("mean_pvalue")
            # Correction : utiliser une variable intermédiaire pour le formatage conditionnel
            pvalue_str = f"{mean_pvalue:.4f}" if mean_pvalue is not None else "N/A"
            print(f"  Mean P-value: {pvalue_str}")
            
            if 'full_sample_params' in strategy_data_dict:
                fs_params = strategy_data_dict['full_sample_params']
                print("\nFULL SAMPLE PARAMETERS:")
                if 'alpha' in fs_params:
                    print(f"  Alpha (α): {fs_params.get('alpha'):.4f}")
                if 'beta' in fs_params:
                    print(f"  Beta (β): {fs_params.get('beta'):.4f}")
        
        if 'window_df' in strategy_data_dict and not strategy_data_dict['window_df'].empty:
            print("\nESTIMATION WINDOWS SUMMARY:")
            window_summary = strategy_data_dict['window_df'][['alpha', 'beta', 'p_value', 'is_cointegrated']].describe()
            print(format_table(window_summary, max_rows=max_rows, show_all=show_all))
        
        if has_oos_strategy:
            print("\n\n" + "=" * 100)
            print("STANDARD OUT-OF-SAMPLE STRATEGY DETAILS")
            print("=" * 100)
            
            if 'out_of_sample_stats' in strategy_data_dict and strategy_data_dict['out_of_sample_stats']:
                stats = strategy_data_dict['out_of_sample_stats']
            else:
                stats = {
                    'z_in': self.z_in if hasattr(self, 'z_in') else None,
                    'z_stop': self.z_stop if hasattr(self, 'z_stop') else None,
                    'W0': self.W0 if hasattr(self, 'W0') else None,
                    'L': self.L if hasattr(self, 'L') else None,
                    'final_wealth': self.wealth_history[-1] if hasattr(self, 'wealth_history') and self.wealth_history else None,
                    'total_return': ((self.wealth_history[-1] / self.W0) - 1) * 100 if hasattr(self, 'wealth_history') and self.wealth_history and hasattr(self, 'W0') else None,
                    'max_leverage': max(self.leverage_history) if hasattr(self, 'leverage_history') and self.leverage_history else None,
                }
                
                if hasattr(self, 'trades') and self.trades:
                    os_trades = [t for t in self.trades if 'Cointegration_pvalue' not in t]
                    stats['total_trades'] = len(os_trades)
                    stats['profitable_trades'] = len([t for t in os_trades if 'Profit' in t and t['Profit'] > 0])
                    stats['win_rate'] = (stats['profitable_trades'] / stats['total_trades']) * 100 if stats['total_trades'] > 0 else 0
                
                if hasattr(self, 'wealth_history') and self.wealth_history:
                    wealth_array = np.array(self.wealth_history)
                    max_wealth = np.maximum.accumulate(wealth_array)
                    drawdown = (wealth_array - max_wealth) / max_wealth * 100
                    stats['max_drawdown'] = np.min(drawdown) if len(drawdown) > 0 else 0
            
            print("\nSTRATEGY PARAMETERS:")
            print(f"  Entry Threshold (z_in): {stats.get('z_in')}")
            print(f"  Stop-Loss Threshold (z_stop): {stats.get('z_stop') if stats.get('z_stop') is not None else 'Not used'}")
            print(f"  Maximum Leverage: {stats.get('L')}")
            print(f"  Initial Capital: ${stats.get('W0'):,.2f}")
            
            print("\nPERFORMANCE METRICS:")
            print(f"  Final Capital: ${stats.get('final_wealth'):,.2f}")
            total_return = stats.get('total_return', 0)
            print(f"  Total Return: {total_return:.2f}%")
            
            print("\nTRADING ACTIVITY:")
            print(f"  Total Trades Executed: {stats.get('total_trades', 0)}")
            print(f"  Profitable Trades: {stats.get('profitable_trades', 0)} ({stats.get('win_rate', 0):.2f}%)")
            print(f"  Maximum Drawdown: {stats.get('max_drawdown', 0):.2f}%")
            print(f"  Maximum Leverage Used: {stats.get('max_leverage', 0):.2f}")
            
            if stats.get('bankruptcy') is not None:
                print(f"\nBANKRUPTCY ALERT: Strategy went bankrupt on day {stats.get('bankruptcy')}")
        
        # Le reste de la fonction reste inchangé...

        if has_oos_strategy and has_coint_strategy:
            print("\n\n" + "=" * 100)
            print("TRADING STRATEGIES COMPARISON")
            print("=" * 100)
            
            oos_stats = strategy_data_dict.get('out_of_sample_stats', {})
            coint_stats = strategy_data_dict.get('cointegration_aware_stats', {})
            
            print("\nBoth strategies have been executed. Performance comparison:")
            
            comparison_data = {
                'Metric': [
                    'Initial Capital', 'Final Capital', 'Total Return', 
                    'Max Drawdown', 'Win Rate', 'Total Trades'
                ],
                'Standard Strategy': [
                    f"${oos_stats.get('W0', 1000):,.2f}",
                    f"${oos_stats.get('final_wealth', 0):,.2f}",
                    f"{oos_stats.get('total_return', 0):.2f}%",
                    f"{oos_stats.get('max_drawdown', 0):.2f}%",
                    f"{oos_stats.get('win_rate', 0):.2f}%",
                    f"{oos_stats.get('total_trades', 0)}"
                ],
                'Cointegration-Aware': [
                    f"${coint_stats.get('W0', 1000):,.2f}",
                    f"${coint_stats.get('final_wealth', 0):,.2f}",
                    f"{coint_stats.get('total_return', 0):.2f}%",
                    f"{coint_stats.get('max_drawdown', 0):.2f}%",
                    f"{coint_stats.get('win_rate', 0):.2f}%",
                    f"{coint_stats.get('total_trades', 0)}"
                ]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            print("\n" + format_table(comparison_df, max_rows=None, show_all=True))
            
            oos_return = oos_stats.get('total_return', 0)
            coint_return = coint_stats.get('total_return', 0)
            better_strategy = "Standard" if oos_return > coint_return else "Cointegration-Aware"
            print(f"\nBetter Performing Strategy: {better_strategy}")

        if 'trades_df' in strategy_data_dict and not strategy_data_dict['trades_df'].empty:
            print("\n\n" + "=" * 100)
            print("TRADES ANALYSIS")
            print("=" * 100)
            
            trades_df = strategy_data_dict['trades_df']
            print("\nTRADE STATISTICS:")
            
            if 'Profit' in trades_df.columns:
                profit_stats = trades_df['Profit'].describe()
                print("\nProfit Distribution:")
                print(format_table(profit_stats, max_rows=10, show_all=True))
                
                profitable = (trades_df['Profit'] > 0).sum()
                unprofitable = (trades_df['Profit'] <= 0).sum()
                total = profitable + unprofitable
                
                print("\nTrade Outcomes:")
                print(f"  Profitable Trades: {profitable} ({profitable/total*100:.2f}%)")
                print(f"  Unprofitable Trades: {unprofitable} ({unprofitable/total*100:.2f}%)")
                
                if 'Entry_Date' in trades_df.columns and 'Exit_Date' in trades_df.columns:
                    try:
                        trades_df['holding_period'] = (trades_df['Exit_Date'] - trades_df['Entry_Date']).dt.days
                        avg_holding = trades_df['holding_period'].mean()
                        print(f"\nAverage Holding Period: {avg_holding:.2f} days")
                    except:
                        pass
            
            print("\nSAMPLE OF TRADES:")
            
            display_cols = ['Signal', 'Entry_Date', 'Exit_Date', 'Profit', 'Return']
            display_cols = [col for col in display_cols if col in trades_df.columns]
            
            if display_cols:
                sample_trades = trades_df[display_cols].head(max_rows)
                print(format_table(sample_trades, max_rows=max_rows, show_all=False))
            else:
                print("No detailed trade information available")
        
        print("\n\n" + "=" * 100)
        print("CONCLUSION")
        print("=" * 100)
        
        if has_oos_strategy or has_coint_strategy:
            print("\nSUMMARY OF FINDINGS:")
            
            oos_profitable = has_oos_strategy and (strategy_data_dict.get('out_of_sample_stats', {}).get('total_return', 0) > 0)
            coint_profitable = has_coint_strategy and (strategy_data_dict.get('cointegration_aware_stats', {}).get('total_return', 0) > 0)
            
            if oos_profitable or coint_profitable:
                print("\n✓ At least one trading strategy showed positive returns.")
                
                if has_oos_strategy and has_coint_strategy:
                    oos_return = strategy_data_dict['out_of_sample_stats'].get('total_return', 0)
                    coint_return = strategy_data_dict['cointegration_aware_stats'].get('total_return', 0)
                    
                    if oos_return > coint_return:
                        print(f"The Standard Strategy outperformed with {oos_return:.2f}% return vs {coint_return:.2f}% for the Cointegration-Aware Strategy.")
                    else:
                        print(f"The Cointegration-Aware Strategy outperformed with {coint_return:.2f}% return vs {oos_return:.2f}% for the Standard Strategy.")
            else:
                print("\n⚠ None of the trading strategies showed positive returns.")
            
            if 'window_stats' in strategy_data_dict:
                coint_pct = strategy_data_dict['window_stats'].get('cointegrated_pct', 0)
                if coint_pct > 70:
                    print(f"\n✓ Strong cointegration relationship detected ({coint_pct:.2f}% of windows).")
                elif coint_pct > 30:
                    print(f"\n⚠ Moderate cointegration relationship detected ({coint_pct:.2f}% of windows).")
                else:
                    print(f"\n✗ Weak cointegration relationship detected ({coint_pct:.2f}% of windows).")
        else:
            print("\nNo trading strategies have been executed yet.")
        
        print("\n" + "=" * 100)
        return
    
    def display_trade_logs(self, filter_type=None, max_trades=None):
        """
        Returns a DataFrame with trade logs including wealth evolution.
        
        Parameters:
        -----------
        filter_type : str, optional
            Type de filtre: "cointegration" pour les trades cointegration-aware,
                        "standard" pour les trades standard,
                        None pour tous les trades
        max_trades : int or bool, optional
            Nombre maximum de trades à afficher (True = tous les trades)
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame contenant les détails des trades avec évolution de richesse
        """
        import pandas as pd
        
        
        if filter_type == "cointegration":
            filtered_trades = [trade for trade in self.trades if 'Cointegration_pvalue' in trade]
        elif filter_type == "standard":
            filtered_trades = [trade for trade in self.trades if 'Cointegration_pvalue' not in trade]
        else:
            filtered_trades = self.trades
        
        
        completed_trades = [t for t in filtered_trades if 'Exit_Date' in t]
        
        
        if max_trades is True or max_trades is None:
            display_trades = completed_trades
        else:
            display_trades = completed_trades[:max_trades]
        
        
        df_data = []
        running_wealth = self.W0  
        
        for i, trade in enumerate(display_trades):
            entry_wealth = trade.get('Entry_Wealth', running_wealth)
            profit = trade.get('Profit', 0)
            exit_wealth = entry_wealth + profit  
            running_wealth = exit_wealth  
            
            row_dict = {
                "Signal": trade.get('Signal'),
                "Entry Date": trade.get('Entry_Date').date() if hasattr(trade.get('Entry_Date'), 'date') else trade.get('Entry_Date'),
                "Exit Date": trade.get('Exit_Date').date() if hasattr(trade.get('Exit_Date'), 'date') else trade.get('Exit_Date'),
                "Exit Reason": trade.get('Exit_Reason'),
                "Profit": profit,
                "Return (%)": trade.get('Return', 0),
                "Entry Wealth": entry_wealth,
                "Exit Wealth": exit_wealth
            }
            
            
            if filter_type == "cointegration":
                row_dict["P-value"] = trade.get('Cointegration_pvalue', None)
            
            df_data.append(row_dict)
        
        
        if df_data:
            result_df = pd.DataFrame(df_data)
            
            result_df.index = range(1, len(result_df) + 1)
            return result_df
        else:
            
            columns = ["Signal", "Entry Date", "Exit Date", "Exit Reason", "Profit", "Return (%)", "Entry Wealth", "Exit Wealth"]
            if filter_type == "cointegration":
                columns.insert(5, "P-value")
            empty_df = pd.DataFrame(columns=columns)
            empty_df.index.name = None
            return empty_df



print('\n', "# ====================================  4.3.1 ========================================= #", '\n')

rolling_analyzer = RollingWindowAnalysis(
    df=df,
    window_size=500,
    step_size=20,
    asset_A='CHEVRON',
    asset_B='CONOCOPHILLIPS',
    audit=False  
)
rolling_analyzer.run_rolling_analysis()

detailed_analysis = rolling_analyzer.get_detailed_window_analysis()



print('\n', "# ====================================  4.3.1 ========================================= #", '\n')

def display_window_parameters(detailed_analysis, window_index=0):
    """
    Affiche les paramètres estimés pour une fenêtre spécifique.
    
    Parameters:
    -----------
    detailed_analysis : dict
        Résultat de la méthode get_detailed_window_analysis()
    window_index : int
        Indice de la fenêtre à afficher (0 pour la première, 1 pour la deuxième, etc.)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame contenant les paramètres de la fenêtre spécifiée
    """
    
    selected_window = detailed_analysis['window_details'].iloc[window_index]
    
    
    window_df = pd.DataFrame({
        'Parameter'  : ['$\\alpha$ ', '$\\beta$'],
        'Estimate'   : [selected_window['alpha'], selected_window['beta']],
        'Start Date' : [selected_window['start_date'], selected_window['start_date']], 
        'End Date'   : [selected_window['end_date'], selected_window['end_date']]
    })
    
    
    window_df['Period Length'] = [(selected_window['end_date'] - selected_window['start_date']).days] * 2
    window_df['Cointegration p-value'] = [selected_window['p_value'], '']
    window_df['Is Cointegrated'] = [selected_window['is_cointegrated'], '']
    
    
    if 'price_correlation' in selected_window:
        window_df['Price Correlation'] = [selected_window['price_correlation'], '']
    
    if 'return_correlation' in selected_window:
        window_df['Return Correlation'] = [selected_window['return_correlation'], '']
    
    if 'window_size' in selected_window:
        window_df['Window Size'] = [selected_window['window_size'], '']
    
    if 'step_size' in selected_window:
        window_df['Step Size'] = [selected_window['step_size'], '']
    
    if 'window_index' in selected_window:
        window_df['Window Index'] = [window_index, '']
    else:
        window_df['Window Index'] = [window_index, '']
    
    
    window_df = window_df.set_index('Parameter')
    window_df.index.name = 'Parameter'
    window_df.columns.name = 'Value'
    window_df = window_df.reset_index()
    
    return window_df


windows_0 = display_window_parameters(detailed_analysis, window_index=0)



print(windows_0)









 
def validate_rolling_window(detailed_analysis, df, window_index=0, asset_A='CHEVRON', asset_B='CONOCOPHILLIPS', 
                           visualize=True):
    """
    Validates the results of a specific rolling window using an independent analysis.
    
    Parameters:
    -----------
    detailed_analysis : dict
        Result of the get_detailed_window_analysis() method.
    df : pandas.DataFrame
        Original DataFrame containing the data.
    window_index : int
        Index of the window to validate (0 for the first, 1 for the second, etc.).
    asset_A, asset_B : str
        Names of the columns for the assets to be analyzed.
    visualize : bool
        If True, generates visualization plots.
        
    Returns:
    --------
    dict
        Dictionary containing the validation results.
    """
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller
    from scipy import stats
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import pandas as pd
    import numpy as np

    
    window = detailed_analysis['window_details'].iloc[window_index]
    
    
    start_date = window['start_date']
    end_date = window['end_date']
    
    
    window_data = df.loc[start_date:end_date].iloc[:500]
    
    
    trading_days_count = len(window_data)
    calendar_days_count = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    
    
    X = window_data[asset_B]
    y = window_data[asset_A]
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()
    
    alpha_hat = model.params.iloc[0]
    beta_hat = model.params.iloc[1]
    
    
    spread = y - (alpha_hat + beta_hat * X)
    adf_result = adfuller(spread, regression='c', maxlag=0)  
    p_value = adf_result[1]
    is_cointegrated = p_value < 0.05
    
    
    spread_stats = spread.describe()
    k2, p_normal = stats.normaltest(spread)
    is_normal = p_normal > 0.05
    
    
    results = {
        "window_info": {
            "window_index": window_index,
            "start_date": start_date,
            "end_date": end_date,
            "trading_days": trading_days_count,
            "calendar_days": calendar_days_count
        },
        "parameters": {
            "alpha": {
                "independent": alpha_hat,
                "class": window['alpha'],
                "diff": alpha_hat - window['alpha'],
                "diff_pct": (alpha_hat / window['alpha'] - 1) * 100 if window['alpha'] != 0 else np.nan
            },
            "beta": {
                "independent": beta_hat,
                "class": window['beta'],
                "diff": beta_hat - window['beta'],
                "diff_pct": (beta_hat / window['beta'] - 1) * 100 if window['beta'] != 0 else np.nan
            }
        },
        "cointegration": {
            "p_value": {
                "independent": p_value,
                "class": window['p_value'],
                "diff": p_value - window['p_value']
            },
            "is_cointegrated": {
                "independent": is_cointegrated,
                "class": window['is_cointegrated'],
                "match": is_cointegrated == window['is_cointegrated']
            }
        },
        "spread_stats": spread_stats,
        "spread_normality": {
            "p_value": p_normal,
            "is_normal": is_normal
        }
    }
    
    
    print(f"\n=== Validation of window {window_index} ({start_date.date()} to {end_date.date()}) ===")
    print(f"Number of observations: {trading_days_count} trading days ({calendar_days_count} calendar days)")
    
    print("\nRegression parameters:")
    print(f"Alpha: {alpha_hat:.6f} (class: {window['alpha']:.6f}, diff: {alpha_hat - window['alpha']:.6f})")
    print(f"Beta: {beta_hat:.6f} (class: {window['beta']:.6f}, diff: {beta_hat - window['beta']:.6f})")
    
    print("\nCointegration test:")
    print(f"P-value: {p_value:.6f} (class: {window['p_value']:.6f}, diff: {p_value - window['p_value']:.6f})")
    cointegration_status = "identical" if is_cointegrated == window['is_cointegrated'] else "different"
    print(f"Conclusion: {'Cointegrated' if is_cointegrated else 'Not cointegrated'} (class: {'Cointegrated' if window['is_cointegrated'] else 'Not cointegrated'}, {cointegration_status})")
    
    
    if visualize:
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.scatter(X, y, alpha=0.5, color='#2f77e4')
        ax.plot(X, alpha_hat + beta_hat * X, color='#d62728', linewidth=1.1)
        
        ax.set_xlabel(asset_B, fontsize=12)
        ax.set_ylabel(asset_A, fontsize=12)
        ax.set_title(f'Window {window_index}: Regression {asset_A} = α + β×{asset_B} ({start_date.date()} to {end_date.date()})', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', labelsize=10, labelcolor='gray')
        ax.tick_params(axis='y', labelsize=10, labelcolor='gray')
        
        
        text_info = (
            f'α = {alpha_hat:.4f}\n'
            f'β = {beta_hat:.4f}\n'
            f'p-value (DF) = {p_value:.4f}\n'
            f'Cointegrated: {"Yes" if is_cointegrated else "No"}'
        )
        
        ax.annotate(text_info, xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                    va='top', fontsize=10)
        
        
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2f77e4', 
                      markersize=5, alpha=0.5, label=f'Price points'),
            plt.Line2D([0], [0], color='#d62728', linewidth=1.1, label=f'Regression fit')
        ]
        
        ax.legend(handles=legend_elements, fontsize=8, loc='upper left', 
                 bbox_to_anchor=(1, 1), borderaxespad=2, frameon=False)
        
        plt.tight_layout()
        plt.show()
        
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(spread.index, spread, color='#9467bd', linewidth=0.95, label='Spread')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.95, alpha=0.7, label='Mean')
        
        
        std = spread.std()
        ax.axhline(y=std, color='#d62728', linestyle='--', alpha=0.4, linewidth=0.8, label=f'+1σ = {std:.2f}')
        ax.axhline(y=-std, color='#d62728', linestyle='--', alpha=0.4, linewidth=0.8)
        ax.axhline(y=2*std, color='red', linestyle=':', alpha=0.4, linewidth=0.8, label=f'+2σ = {2*std:.2f}')
        ax.axhline(y=-2*std, color='red', linestyle=':', alpha=0.4, linewidth=0.8)
        
        ax.set_title(f'Window {window_index}: Spread {asset_A} - (α + β×{asset_B})', fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Spread', fontsize=12)
        ax.tick_params(axis='x', labelsize=10, labelcolor='gray')
        ax.tick_params(axis='y', labelsize=10, labelcolor='gray')
        
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=0)
        
        
        stats_text = (
            f'Mean: {spread.mean():.4f}\n'
            f'Std: {spread.std():.4f}\n'
            f'Skew: {spread.skew():.4f}\n'
            f'Kurt: {spread.kurtosis():.4f}\n'
            f'DF p-value: {p_value:.4f}'
        )
        
        ax.annotate(stats_text, xy=(0.05, 0.05), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                    va='bottom', fontsize=9)
        
        ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=2, frameon=False)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    results = pd.DataFrame(results)
    
    return None






window_0_validation = validate_rolling_window(detailed_analysis, 
                                              df, 
                                              window_index=0,
                                              visualize=True)



print('\n', "# ====================================  4.3.2 ========================================= #", '\n')


window__1 = display_window_parameters(detailed_analysis, window_index=1)





window_1 = validate_rolling_window(detailed_analysis,
                                   df, 
                                   window_index=1,
                                   visualize=True)



print('\n', "# ====================================  Q 4.11 ========================================= #", '\n')

rolling_analyzer.plot_rolling_correlations()



 
print('\n', "# ====================================  Q 4.12 ========================================= #", '\n')
rolling_analyzer.plot_parameter_dynamics()
rolling_analyzer.plot_spread_comparison()



print('\n', "# ====================================  Q 4.13 ========================================= #", '\n')


rolling_analyzer.run_out_of_sample_strategy(z_in=1.5, 
                                            L=2, 
                                            W0=1000.0, 
                                            z_stop=None)

os_sample_strategy = rolling_analyzer.plot_out_of_sample_results() 

trades_df = rolling_analyzer.display_trade_logs(filter_type="standard", max_trades=True)
print(trades_df)  








print('\n', "# ====================================  Q 4.14 ========================================= #", '\n')

rolling_analyzer.plot_cointegration_pvalues(
    use_advanced=True,
    advanced_test_params={
        'max_lag': None,
        'regression_type': 'c',  
        'autolag': 'AIC',
        'check_residuals_ljungbox': False,
        'ljungbox_lags': False,
        'ljungbox_threshold': False
    }
)



print('\n', "# ====================================  Q 4.15 ========================================= #", '\n')






rolling_analyzer.run_cointegration_aware_strategy(z_in=None,                
                                                  L=2,                      
                                                  W0=1000.0,                
                                                  z_stop=None,              
                                                  p_threshold=0.05,         
                                                  use_advanced_test={       
                      'max_lag'                  : None,                    
                      'regression_type'          : 'c',                     
                      'autolag'                  : 'AIC',                   
                      'verbose'                  : False,                   
                      'check_residuals_ljungbox' : None,                    
                      'ljungbox_lags'            : None,                    
                      'ljungbox_threshold'       : None                     
                      }
                                                  )

cointegration_aware_strategy = rolling_analyzer.plot_cointegration_aware_results()


coint_trades_df = rolling_analyzer.display_trade_logs(filter_type="cointegration")
print(coint_trades_df)

