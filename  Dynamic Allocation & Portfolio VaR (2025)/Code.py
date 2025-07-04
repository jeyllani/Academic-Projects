import pandas as pd
import numpy as np  
from datetime import datetime
import matplotlib.pyplot as plt     
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.cm as cm
import seaborn as sns   
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.stats import kstest, skew, kurtosis
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from scipy.stats import t
from scipy.stats import norm 
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import genextreme
import os
import warnings
from arch import arch_model
from arch.utility.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=ConvergenceWarning)
sns.set_theme(style="whitegrid", context="talk")
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (22, 8)
print("Actual directory:", os.listdir())





file_path = 'DATA.xlsx'




ftse_100_raw = pd.read_excel(file_path, sheet_name='FTSE100')
bond_data_raw = pd.read_excel(file_path, sheet_name='BMUK10Y')
riskfree_data_raw = pd.read_excel(file_path, sheet_name='ECUKPST')


ftse_100_daily = pd.read_excel(file_path, sheet_name='FTSE100')
bond_data_daily = pd.read_excel(file_path, sheet_name='BMUK10Y')
riskfree_data_daily = pd.read_excel(file_path, sheet_name='ECUKPST')



def prepare_dataframe(df, col_name):
    
    df_clean = df.iloc[1:, :].copy()
    
    
    if col_name == 'rf':
        df_clean.columns = ['Date', col_name]
    else:
        df_clean.columns = ['Date', col_name, '-']
    
    
    df_clean['Date'] = pd.to_datetime(df_clean['Date'])
    
    
    df_clean.set_index('Date', inplace=True)
    
    
    if col_name != 'rf':
        df_clean = df_clean[[col_name]].astype(float)
    else:
        df_clean = df_clean[[col_name]].astype(float)
        
    
    df_clean.index.name = None
    
    return df_clean


ftse_100 = prepare_dataframe(ftse_100_raw, 'stock')
bond_data = prepare_dataframe(bond_data_raw, 'bond')
riskfree_data = prepare_dataframe(riskfree_data_raw, 'rf')


print("Initial dates in raw data:")
print(f"FTSE 100: {ftse_100_raw.iloc[1, 0]}")
print(f"Bonds: {bond_data_raw.iloc[1, 0]}")
print(f"Risk-free rate: {riskfree_data_raw.iloc[1, 0]}")

print("\nFirst date after cleaning:")
print(f"FTSE 100: {ftse_100.index.min()}")
print(f"Bonds: {bond_data.index.min()}")
print(f"Risk-free rate: {riskfree_data.index.min()}")



ftse_100_weekly = ftse_100.resample('W-MON').last()  # Last price of the week
bond_data_weekly = bond_data.resample('W-MON').last()
riskfree_data_weekly = riskfree_data.resample('W-MON').last()


ftse_100_weekly['stock_returns'] = ftse_100_weekly['stock'].pct_change()
bond_data_weekly['bond_returns'] = bond_data_weekly['bond'].pct_change()


riskfree_data_weekly['rf_weekly'] = riskfree_data_weekly['rf'] / 100 / 52




df_weekly = pd.merge(ftse_100_weekly[['stock_returns']], 
                     bond_data_weekly[['bond_returns']], 
                     left_index=True, right_index=True,
                     how='outer')
df_weekly = pd.merge(df_weekly, 
                     riskfree_data_weekly[['rf_weekly']], 
                     left_index=True, right_index=True,
                     how='outer')



print("Lines containing NaNs:")
nas = df_weekly[df_weekly.isna().any(axis=1)]
print(f"Number of lines with NaN: {len(nas)}")
print(nas)



df_weekly.rename(columns={
    'stock_returns': 'stock', 
    'bond_returns': 'bond', 
    'rf_weekly': 'rf'
}, inplace=True)


start_date = '2001-01-01'
end_date = '2024-12-31'
df_weekly = df_weekly[(df_weekly.index >= start_date) & (df_weekly.index <= end_date)]
print(f"\nNumber of observations before deleting NAs: {len(df_weekly)}")


df_weekly = df_weekly.dropna()
print(f"Number of observations after deleting NAs: {len(df_weekly)}")
print(f"First date kept: {df_weekly.index.min()}")
print(f"Last date kept: {df_weekly.index.max()}")


print("\nSummary of weekly returns:")
print(df_weekly.describe())


print("\nAnnualized risk premiums:")
print(f"Stocks: {(df_weekly['stock'].mean() - df_weekly['rf'].mean()) * 52:.2%}")
print(f"Bonds: {(df_weekly['bond'].mean() - df_weekly['rf'].mean()) * 52:.2%}")


print("\nFirst weeks of the DataFrame:")
print(df_weekly.head())











print('# ---------------------- 1 Static Allocation ---------------------------------------- #')





print('# ---------------------- Q 1.2  ---------------------------------------- #')






returns = df_weekly[['stock', 'bond']]
mu = returns.mean().values.reshape(-1, 1)
Rf = df_weekly['rf'].mean()
e = np.ones((2, 1))
Sigma = returns.cov().values


def optimal_weights(lmbda):
    inv_Sigma = np.linalg.inv(Sigma)
    alpha = (1 / lmbda) * inv_Sigma @ (mu - Rf * e)
    return alpha






alpha_2 = optimal_weights(2)
alpha_10 = optimal_weights(10)



df_abs_weights = pd.DataFrame({
    'λ (risk aversion)' : [2, 10],
    'Weight Stocks'     : [alpha_2[0, 0], alpha_10[0, 0]],
    'Weight Bonds'      : [alpha_2[1, 0], alpha_10[1, 0]],
    'Weight Cash'       : [1 - alpha_2.sum(), 1 - alpha_10.sum()]
})


df_rel_weights = pd.DataFrame({
    'λ (risk aversion)' : [2, 10],
})


for lambda_idx in range(2):
    
    stock_weight = df_abs_weights.loc[lambda_idx, 'Weight Stocks']
    bond_weight = df_abs_weights.loc[lambda_idx, 'Weight Bonds']
    cash_weight = df_abs_weights.loc[lambda_idx, 'Weight Cash']
    
    
    total_abs_weight = abs(stock_weight) + abs(bond_weight) + abs(cash_weight)
    
    
    df_rel_weights.loc[lambda_idx, 'Prop. Stocks'] = abs(stock_weight) / total_abs_weight
    df_rel_weights.loc[lambda_idx, 'Prop. Bonds'] = abs(bond_weight) / total_abs_weight
    df_rel_weights.loc[lambda_idx, 'Prop. Cash'] = abs(cash_weight) / total_abs_weight



df_abs_weights.set_index('λ (risk aversion)', inplace=True)
df_rel_weights.set_index('λ (risk aversion)', inplace=True)
print("\nPoids optimaux du portefeuille selon λ (valeurs absolues):\n")
print(df_abs_weights.round(4))

print("\nComposition relative du portefeuille selon λ (proportions):\n")
print(df_rel_weights.round(4))
print("(Somme des proportions = 1)")







print('# ---------------------- 2 Estimation of a GARCH model -------------------------- #')





print('# ---------------------- Q 2.1  ---------------------------------------- #')






plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (22, 8)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

def calculer_rendements_excedentaires(df, actifs, rf_col='rf'):
    """
    Calcule les rendements excédentaires (rendement - taux sans risque)
    """
    resultat = pd.DataFrame(index=df.index)
    
    for actif in actifs:
        resultat[f"{actif}_excess"] = df[actif] - df[rf_col]
    
    return resultat

def test_normalite_ks(series_dict):
    """
    Test de normalité de Kolmogorov-Smirnov pour plusieurs series
    
    Parameters:
    -----------
    series_dict : dict
        Dictionnaire avec clés = noms des series et valeurs = series pandas
    
    Returns:
    --------
    DataFrame
        Résultats des tests de normalité
    """
    resultats = []
    
    for nom, serie in series_dict.items():
        serie_clean = serie.dropna()
        
        
        statistic, p_valeur = kstest(
            serie_clean, 
            'norm',
            args=(serie_clean.mean(), serie_clean.std())
        )
        
        est_normal = p_valeur >= 0.05
        skewness = skew(serie_clean)
        kurt = kurtosis(serie_clean)
        
        resultats.append({
            'Serie': nom,
            'Statistic KS': statistic,
            'P-val': p_valeur,
            'Est normal': est_normal,
            'Skewness': skewness,
            'Kurtosis': kurt
        })
    
    return pd.DataFrame(resultats)

def test_autocorrelation_ljungbox(series_dict, lags=4):
    """
    Test d'autocorrélation de Ljung-Box pour plusieurs series
    
    Parameters:
    -----------
    series_dict : dict
        Dictionnaire avec clés = noms des series et valeurs = series pandas
    lags : int
        Nombre de décalages à utiliser pour le test
    
    Returns:
    --------
    DataFrame
        Résultats des tests d'autocorrélation
    """
    resultats = []
    
    for nom, serie in series_dict.items():
        serie_clean = pd.to_numeric(serie, errors='coerce').dropna()
        
        if len(serie_clean) < 2:
            resultats.append({
                'Serie': nom,
                'Lags': lags,
                'Statistic LB': np.nan,
                'P-val': np.nan,
                'Is autocorrelated': False,
                'Is stationary': False
            })
            continue
        
        try:
            
            adf_result = adfuller(serie_clean)
            is_stationary = adf_result[1] < 0.05
            
            
            result = acorr_ljungbox(serie_clean, lags=[lags], return_df=True)
            
            lb_stat_float = float(result['lb_stat'].iloc[0])
            p_valeur_float = float(result['lb_pvalue'].iloc[0])
            est_autocorrele = p_valeur_float < 0.05
            
            resultats.append({
                'Serie': nom,
                'Lags': lags,
                'Statistic LB': lb_stat_float,
                'P-val': p_valeur_float,
                'Is autocorrelated': est_autocorrele,
                'Is stationary': is_stationary
            })
            
        except Exception as e:
            resultats.append({
                'Serie': nom,
                'Lags': lags,
                'Statistic LB': np.nan,
                'P-val': np.nan,
                'Is autocorrelated': False,
                'Is stationary': False
            })
    
    return pd.DataFrame(resultats)

def visualiser_distributions_comparees(series_dict):
    """
    Visualise la distribution de plusieurs series et les compare à une Normal Law
    
    Parameters:
    -----------
    series_dict : dict
        Dictionnaire avec clés = noms des series et valeurs = series pandas
    """
    plt.figure(figsize=(22, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    linestyles = ['-', '--', '-.', ':']
    alphas = [0.7, 0.6]
    
    for i, (nom, serie) in enumerate(series_dict.items()):
        serie_clean = serie.dropna()
        
        
        sns.kdeplot(
            serie_clean, 
            label=f'Observed Distribution - {nom}',
            color=colors[i],
            alpha=alphas[0]
        )
        
        
        x = np.linspace(serie_clean.min(), serie_clean.max(), 100)
        y = stats.norm.pdf(x, serie_clean.mean(), serie_clean.std())
        plt.plot(
            x, y, 
            linestyle=linestyles[i],
            color=colors[i], 
            linewidth=2, 
            alpha=alphas[1],
            label=f'Normal Law - {nom}'
        )
    
    
    plt.title("Distribution of Excess Returns vs Normal Law", fontweight='bold')
    plt.xlabel("Excess Returns")
    plt.ylabel("Probability Density")
    plt.legend(loc='best', frameon=True)
    plt.grid(True, linestyle=':', alpha=0.4)
    
    
    stats_text = []
    for nom, serie in series_dict.items():
        serie_clean = serie.dropna()
        skewness = skew(serie_clean)
        kurt = kurtosis(serie_clean)
        stats_text.append(f"{nom}: Skewness={skewness:.4f}, Kurtosis={kurt:.4f}")
    
    plt.annotate(
        '\n'.join(stats_text),
        xy=(0.02, 0.96),
        xycoords='axes fraction',
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
        va='top'
    )
    
    plt.tight_layout()
    plt.show()

def visualiser_autocorrelation(series_dict, lags=20):
    """
    Visualise les fonctions d'autocorrélation pour plusieurs series
    
    Parameters:
    -----------
    series_dict : dict
        Dictionnaire avec clés = noms des series et valeurs = series pandas
    lags : int
        Nombre de décalages à afficher
    """
    num_series = len(series_dict)
    fig, axes = plt.subplots(num_series, 1, figsize=(22, 8))
    
    if num_series == 1:
        axes = [axes]
    
    for i, (nom, serie) in enumerate(series_dict.items()):
        serie_clean = serie.dropna()
        
        
        plot_acf(serie_clean, lags=lags, ax=axes[i], alpha=0.05, title=f"Fonction d'autocorrélation - {nom}")
        
        
        axes[i].set_xlabel("Lag")
        axes[i].set_ylabel("Autocorrélation")
        axes[i].grid(True, linestyle=':', alpha=0.4)
        axes[i].set_facecolor('white')
    
    plt.tight_layout()
    plt.show()

def analyser_rendements(df):
    """
    Fonction principale qui analyse les rendements et leurs carrés
    """
    
    actifs = ['stock', 'bond']
    excess_returns = calculer_rendements_excedentaires(df, actifs)
    
    
    series_dict = {
        'Stock': excess_returns['stock_excess'],
        'Bond': excess_returns['bond_excess'],
        'Stock²': excess_returns['stock_excess'] ** 2,
        'Bond²': excess_returns['bond_excess'] ** 2
    }
    
    
    table_normalite = test_normalite_ks(series_dict)
    
    
    table_autocorr = test_autocorrelation_ljungbox(series_dict)
    
    
    print("\n=== Normality test of excess returns ===")
    print(table_normalite.round(4))

    print("\n=== Autocorrelation test of excess returns ===")
    print(table_autocorr.round(4))
    
    
    visualiser_distributions_comparees({
        'Stock': excess_returns['stock_excess'],
        'Bond': excess_returns['bond_excess']
    })
    
    
    
    
    
    
    
    
    
    
    
    
    
    return table_normalite, table_autocorr, excess_returns


df_weekly_clean = df_weekly.apply(pd.to_numeric, errors='coerce')
table_normalite, table_autocorr, excess_returns_df = analyser_rendements(df_weekly_clean)












print('# ---------------------- Q 2.2  ---------------------------------------- #')













def estimer_ar1(series, nom_serie):
    """
    Estime un modèle AR(1) pour une série temporelle
    """
    
    model = sm.tsa.AutoReg(series, lags=1)
    result = model.fit()
    
    
    residus = result.resid
    
    
    a_i = result.params.iloc[0]  
    rho_i = result.params.iloc[1]  
    
    
    y_mean = series.mean()
    ss_total = ((series - y_mean) ** 2).sum()
    ss_residual = (residus ** 2).sum()
    r_squared = 1 - (ss_residual / ss_total)
    
    
    print(f"\n=== Modèle AR(1) pour {nom_serie} ===")
    print(f"a_i (constante) = {a_i:.6f}")
    print(f"rho_i (coefficient AR(1)) = {rho_i:.6f}")
    print(f"R² = {r_squared:.6f}")
    print(f"AIC = {result.aic:.6f}")
    print(f"BIC = {result.bic:.6f}")
    print(f"Écart-type des résidus = {residus.std():.6f}")
    
    
    lb_result = acorr_ljungbox(residus, lags=[4], return_df=True)
    lb_stat = float(lb_result['lb_stat'].iloc[0])
    lb_pval = float(lb_result['lb_pvalue'].iloc[0])
    
    print(f"Test Ljung-Box sur les résidus (4 lags): statistique = {lb_stat:.4f}, p-valeur = {lb_pval:.4f}")
    print(f"Les résidus sont {'autocorrélés' if lb_pval < 0.05 else 'non autocorrélés'} (seuil 5%)")
    
    
    ks_stat, ks_pval = stats.kstest(residus, 'norm', args=(residus.mean(), residus.std()))
    print(f"Test KS de normalité des résidus: statistique = {ks_stat:.4f}, p-valeur = {ks_pval:.4f}")
    print(f"Les résidus sont {'non normaux' if ks_pval < 0.05 else 'normaux'} (seuil 5%)")
    
    
    print("\nRésumé complet du modèle:")
    print(result.summary())
    
    return result, residus, {'a_i': a_i, 'rho_i': rho_i, 'r_squared': r_squared}

def plot_ar1_residuals_distribution(residuals_dict, figsize=(22, 8)):
    """
    Plots the distribution of AR(1) model residuals against normal distribution
    with a clean, minimalist design
    
    Parameters:
    -----------
    residuals_dict : dict
        Dictionary with asset names as keys and their AR(1) residuals as values
    figsize : tuple
        Figure size (width, height)
    """
    
    plt.figure(figsize=figsize, dpi=50)
    
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    
    for i, (asset_name, residuals) in enumerate(residuals_dict.items()):
        
        residuals_clean = residuals.dropna()
        
        
        sns.kdeplot(
            residuals_clean,
            color=colors[i],
            alpha=0.8,
            label=f'Residuals distribution - {asset_name}',
            linewidth=2
        )
        
        
        x = np.linspace(residuals_clean.min(), residuals_clean.max(), 100)
        y = stats.norm.pdf(x, residuals_clean.mean(), residuals_clean.std())
        plt.plot(
            x, y, 
            linestyle='--',
            color=colors[i],
            linewidth=2,
            alpha=0.7,
            label=f'Normal distribution - {asset_name}'
        )
        
        
        skewness = stats.skew(residuals_clean)
        kurtosis = stats.kurtosis(residuals_clean)
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    plt.title("AR(1) Model Residuals Distribution vs. Normal Distribution", fontsize=16, fontweight='bold')
    plt.xlabel("Residuals", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, linestyle=':', alpha=0.3)
    
    
    plt.tight_layout()
    plt.show()

def analyser_modeles_ar1(df_weekly):
    """
    Analyse complète des modèles AR(1) pour les rendements des actions et des obligations
    """
    
    stock_returns = df_weekly['stock']
    bond_returns = df_weekly['bond']
    
    
    stock_result, stock_residus, stock_params = estimer_ar1(stock_returns, "Stocks")
    bond_result, bond_residus, bond_params = estimer_ar1(bond_returns, "Bonds")
    
    
    residus_df = pd.DataFrame(index=df_weekly.index)
    residus_df['stock_residus'] = stock_residus
    residus_df['bond_residus'] = bond_residus
    
    
    plot_ar1_residuals_distribution({
        'Stocks': stock_residus,
        'Bonds': bond_residus
    })
    
    
    params_df = pd.DataFrame({
        'Asset': ['Stocks', 'Bonds'],
        'a_i (constant)': [stock_params['a_i'], bond_params['a_i']],
        'ρ_i (AR coefficient)': [stock_params['rho_i'], bond_params['rho_i']],
        'R²': [stock_params['r_squared'], bond_params['r_squared']]
    })
    
    print("\n=== AR(1) Models Estimation Summary ===")
    print(params_df.round(6))
    
    return residus_df, params_df


residus_df, params_df = analyser_modeles_ar1(df_weekly)





residus_df.dropna()











print('# ---------------------- Q 2.3  ---------------------------------------- #')








resid_stock = residus_df['stock_residus'].dropna()
resid_bond = residus_df['bond_residus'].dropna()


print(f"Valeurs NaN dans resid_stock: {resid_stock.isna().sum()}")
print(f"Valeurs NaN dans resid_bond: {resid_bond.isna().sum()}")
print(f"Valeurs infinies dans resid_stock: {np.isinf(resid_stock).sum()}")
print(f"Valeurs infinies dans resid_bond: {np.isinf(resid_bond).sum()}")


try:
    garch_stock = arch_model(resid_stock, vol='GARCH', p=1, q=1, rescale=False)
    garch_bond = arch_model(resid_bond, vol='GARCH', p=1, q=1, rescale=False)
    
    res_garch_stock = garch_stock.fit(disp='off')
    res_garch_bond = garch_bond.fit(disp='off')
    print("Modèles GARCH estimés avec succès")
except Exception as e:
    print(f"Erreur lors de l'estimation du modèle GARCH: {e}")
    
    def filter_outliers(series, n_std=3):
        mean = series.mean()
        std = series.std()
        return series[(series >= mean - n_std*std) & (series <= mean + n_std*std)]
    
    print("Tentative avec filtrage des valeurs extrêmes...")
    resid_stock = filter_outliers(resid_stock)
    resid_bond = filter_outliers(resid_bond)
    
    garch_stock = arch_model(resid_stock, vol='GARCH', p=1, q=1)
    garch_bond = arch_model(resid_bond, vol='GARCH', p=1, q=1)
    
    res_garch_stock = garch_stock.fit(disp='off')
    res_garch_bond = garch_bond.fit(disp='off')


def summarize_garch(result, name):
    params = result.params
    alpha = params['alpha[1]']
    beta = params['beta[1]']
    persistence = alpha + beta

    
    se_alpha = result.std_err.get('alpha[1]', 0)
    se_beta = result.std_err.get('beta[1]', 0)
    
    if se_alpha > 0 and se_beta > 0:
        var_sum = se_alpha**2 + se_beta**2
        t_stat = (persistence - 1) / np.sqrt(var_sum)
        p_value = norm.cdf(t_stat)  
    else:
        t_stat = np.nan
        p_value = np.nan

    return {
        'Asset': name,
        'omega': params['omega'],
        'alpha': alpha,
        'beta': beta,
        'α + β': persistence,
        't-stat (α+β=1)': t_stat,
        'p-value': p_value
    }


results_23 = pd.DataFrame([
    summarize_garch(res_garch_stock, 'Stocks'),
    summarize_garch(res_garch_bond, 'Bonds')
])




    


print("\nRésultats GARCH(1,1) :")
print(results_23.round(6))


vol_stock = res_garch_stock.conditional_volatility
vol_bond = res_garch_bond.conditional_volatility


std_resid_stock = res_garch_stock.resid / res_garch_stock.conditional_volatility
std_resid_bond = res_garch_bond.resid / res_garch_bond.conditional_volatility

print(f"Moyenne des résidus standardisés (stocks): {std_resid_stock.mean():.4f}")
print(f"Écart-type des résidus standardisés (stocks): {std_resid_stock.std():.4f}")
print(f"Moyenne des résidus standardisés (bonds): {std_resid_bond.mean():.4f}")
print(f"Écart-type des résidus standardisés (bonds): {std_resid_bond.std():.4f}")







print('Count standardized residuals (stocks):', std_resid_stock.count())
print('Count standardized residuals (bonds):', std_resid_bond.count())












def configure_plot_style():
    """Configure un style cohérent pour tous les graphiques"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (22, 8)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.grid'] = True


def plot_conditional_volatility(vol_stock, vol_bond):
    """Visualise les volatilités conditionnelles avec style amélioré"""
    configure_plot_style()

    fig, axes = plt.subplots(2, 1, figsize=(22, 8), dpi=50, sharex=True)

    
    colors = {'stock': '#1f77b4', 'bond': '#ff7f0e'}
    
    
    axes[0].plot(vol_stock, label='Stocks', color=colors['stock'], linewidth=1.5)
    axes[0].set_title("Conditional Volatility - Stocks (GARCH(1,1))", fontweight='bold')
    axes[0].set_ylabel("Volatility")
    axes[0].legend(frameon=True, fancybox=True, shadow=True)
    axes[0].grid(True, linestyle=':', alpha=0.4)
    
    
    axes[1].plot(vol_bond, label='Bonds', color=colors['bond'], linewidth=1.5)
    axes[1].set_title("Conditional Volatility - Bonds (GARCH(1,1))", fontweight='bold')
    axes[1].set_ylabel("Volatility")
    axes[1].set_xlabel("")  # Pas besoin de libellé car dates affichées
    axes[1].legend(frameon=True, fancybox=True, shadow=True)
    axes[1].grid(True, linestyle=':', alpha=0.4)
    
    
    axes[0].xaxis.set_ticklabels([])  
    
    
    import matplotlib.dates as mdates
    years_fmt = mdates.DateFormatter('%Y')
    axes[1].xaxis.set_major_formatter(years_fmt)
    axes[1].xaxis.set_major_locator(mdates.YearLocator(2))  
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)  
    plt.show()


def plot_residual_diagnostics(std_resid_stock, std_resid_bond):
    """Visualisation améliorée des diagnostics de résidus GARCH sur une figure unique"""
    configure_plot_style()
    
    
    color_stock = '#1f77b4'  # bleu pour stocks
    color_bond = '#ff7f0e'   # orange pour bonds
    
    
    fig = plt.figure(figsize=(22, 10), dpi=50)
    
    
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)
    
    
    titles = {
        'qq_stock': "QQ-Plot (Stocks)",
        'qq_bond': "QQ-Plot (Bonds)",
        'acf_stock': "ACF - Residuals (Stocks)",
        'acf_bond': "ACF - Residuals (Bonds)",
        'acf_stock_squared': "ACF - Squared Residuals (Stocks)",
        'acf_bond_squared': "ACF - Squared Residuals (Bonds)"
    }
    
    
    ax0 = fig.add_subplot(gs[0, 0])
    stats.probplot(std_resid_stock, dist="norm", plot=ax0)
    ax0.set_title(titles['qq_stock'], fontweight='bold')
    ax0.grid(True, linestyle=':', alpha=0.4)
    ax0.get_lines()[0].set_markerfacecolor(color_stock)
    ax0.get_lines()[0].set_markersize(3)
    
    
    ax1 = fig.add_subplot(gs[0, 1])
    stats.probplot(std_resid_bond, dist="norm", plot=ax1)
    ax1.set_title(titles['qq_bond'], fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.4)
    ax1.get_lines()[0].set_markerfacecolor(color_bond)
    ax1.get_lines()[0].set_markersize(3)
    
    
    ax2 = fig.add_subplot(gs[0, 2])
    plot_acf(std_resid_stock, lags=20, ax=ax2, alpha=0.05, title="", markersize=4, color=color_stock)
    ax2.set_title(titles['acf_stock'], fontweight='bold')
    ax2.set_xlabel("Lag")
    ax2.set_ylabel("Autocorrelation")
    ax2.grid(True, linestyle=':', alpha=0.4)
    
    
    ax3 = fig.add_subplot(gs[1, 0])
    plot_acf(std_resid_bond, lags=20, ax=ax3, alpha=0.05, title="", markersize=4, color=color_bond)
    ax3.set_title(titles['acf_bond'], fontweight='bold')
    ax3.set_xlabel("Lag")
    ax3.set_ylabel("Autocorrelation")
    ax3.grid(True, linestyle=':', alpha=0.4)
    
    
    ax4 = fig.add_subplot(gs[1, 1])
    plot_acf(std_resid_stock**2, lags=20, ax=ax4, alpha=0.05, title="", markersize=4, color=color_stock)
    ax4.set_title(titles['acf_stock_squared'], fontweight='bold')
    ax4.set_xlabel("Lag")
    ax4.set_ylabel("Autocorrelation")
    ax4.grid(True, linestyle=':', alpha=0.4)
    
    
    ax5 = fig.add_subplot(gs[1, 2])
    plot_acf(std_resid_bond**2, lags=20, ax=ax5, alpha=0.05, title="", markersize=4, color=color_bond)
    ax5.set_title(titles['acf_bond_squared'], fontweight='bold')
    ax5.set_xlabel("Lag")
    ax5.set_ylabel("Autocorrelation")
    ax5.grid(True, linestyle=':', alpha=0.4)
    
    
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_stock, markersize=10, label='Stocks'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_bond, markersize=10, label='Bonds')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.95, 0.98), frameon=True)
    
    
    fig.suptitle("GARCH(1,1) Residual Diagnostics", fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    plt.show()


def plot_volatility_comparison(rolling_vol_stock, garch_vol_stock, rolling_vol_bond, garch_vol_bond):
    """Visualisation des comparaisons de volatilités sur une seule image"""
    configure_plot_style()
    
    
    fig, axes = plt.subplots(2, 1, figsize=(22, 8), sharex=True, dpi=50)
    
    
    colors = {
        'rolling': '#ba001c',
        'stock_garch': '#1f77b4',
        'bond_garch': '#ff7f0e'
    }
    
    
    axes[0].plot(
        rolling_vol_stock.index, rolling_vol_stock, 
        label='Rolling Std (20 weeks)', 
        color=colors['rolling'], 
        linewidth=1.5,
        alpha=0.8
    )
    axes[0].plot(
        garch_vol_stock.index, garch_vol_stock, 
        label='GARCH Volatility', 
        color=colors['stock_garch'], 
        linewidth=1.5
    )
    axes[0].set_title("Stock Volatility: Rolling Standard Deviation vs GARCH(1,1)", fontweight='bold')
    axes[0].set_ylabel("Volatility")
    axes[0].grid(True, linestyle=':', alpha=0.4)
    axes[0].legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    
    axes[1].plot(
        rolling_vol_bond.index, rolling_vol_bond, 
        label='Rolling Std (20 weeks)', 
        color=colors['rolling'], 
        linewidth=1,
        alpha=0.8
    )
    axes[1].plot(
        garch_vol_bond.index, garch_vol_bond, 
        label='GARCH Volatility', 
        color=colors['bond_garch'], 
        linewidth=1.5
    )
    axes[1].set_title("Bond Volatility: Rolling Standard Deviation vs GARCH(1,1)", fontweight='bold')
    axes[1].set_ylabel("Volatility")
    axes[1].grid(True, linestyle=':', alpha=0.4)
    axes[1].legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    
    axes[0].xaxis.set_ticklabels([])
    
    
    import matplotlib.dates as mdates
    years_fmt = mdates.DateFormatter('%Y')
    axes[1].xaxis.set_major_formatter(years_fmt)
    axes[1].xaxis.set_major_locator(mdates.YearLocator(2))  
    
    
    fig.suptitle("Comparison of Volatility Estimation Methods", fontsize=18, fontweight='bold', y=0.98)
    
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.15, top=0.92)  
    
    plt.show()
    


plot_conditional_volatility(vol_stock, vol_bond)


plot_residual_diagnostics(std_resid_stock, std_resid_bond)


try:
    
    rolling_vol_stock = df_weekly['stock'].rolling(window=20).std()
    rolling_vol_bond = df_weekly['bond'].rolling(window=20).std()
    
    common_index_stock = rolling_vol_stock.dropna().index.intersection(vol_stock.index)
    common_index_bond = rolling_vol_bond.dropna().index.intersection(vol_bond.index)
    
    
    if len(common_index_stock) > 0 and len(common_index_bond) > 0:
        
        rolling_vol_stock = rolling_vol_stock.loc[common_index_stock]
        vol_stock_aligned = vol_stock.loc[common_index_stock]
        
        rolling_vol_bond = rolling_vol_bond.loc[common_index_bond]
        vol_bond_aligned = vol_bond.loc[common_index_bond]
        
        
        plot_volatility_comparison(
            rolling_vol_stock, 
            vol_stock_aligned, 
            rolling_vol_bond, 
            vol_bond_aligned
        )
    else:
        print("Pas assez d'index communs pour la visualisation des volatilités")
except Exception as e:
    print(f"Erreur lors de la création du graphique de comparaison: {e}")
    





print('# ---------------------- Section 3  ---------------------------------------- #')





print('# ---------------------- Q 3.1  ---------------------------------------- #')






a_stock = params_df.loc[0, 'a_i (constant)']
rho_stock = params_df.loc[0, 'ρ_i (AR coefficient)']
a_bond = params_df.loc[1, 'a_i (constant)']
rho_bond = params_df.loc[1, 'ρ_i (AR coefficient)']

omega_stock = results_23.loc[0, 'omega']
alpha_stock = results_23.loc[0, 'alpha']
beta_stock = results_23.loc[0, 'beta']
omega_bond = results_23.loc[1, 'omega']
alpha_bond = results_23.loc[1, 'alpha']
beta_bond = results_23.loc[1, 'beta']


print("DIAGNOSTIC - Paramètres GARCH:")
print(f"omega_stock: {omega_stock}, alpha_stock: {alpha_stock}, beta_stock: {beta_stock}")
print(f"Somme α+β pour stock: {alpha_stock + beta_stock}")
print(f"omega_bond: {omega_bond}, alpha_bond: {alpha_bond}, beta_bond: {beta_bond}")
print(f"Somme α+β pour bond: {alpha_bond + beta_bond}")


aligned_data = pd.DataFrame(index=df_weekly.index)
aligned_data['stock_returns'] = df_weekly['stock'].copy()
aligned_data['bond_returns'] = df_weekly['bond'].copy()
aligned_data['rf'] = df_weekly['rf'].copy()


residus_df_clean = residus_df.dropna()
aligned_data = pd.merge(
    aligned_data,
    residus_df_clean,
    left_index=True, 
    right_index=True,
    how='inner'
)

print("\nDIAGNOSTIC - Après alignement des données:")
print(f"Taille du DataFrame aligné: {len(aligned_data)}")
print(f"Colonnes disponibles: {aligned_data.columns.tolist()}")
print(f"NaNs dans aligned_data: {aligned_data.isna().sum().sum()}")


rho_sb = np.corrcoef(aligned_data['stock_residus'], aligned_data['bond_residus'])[0, 1]
print(f"Corrélation constante entre résidus stock et bond: {rho_sb:.4f}")


aligned_data['exp_stock_return'] = a_stock + rho_stock * aligned_data['stock_returns'].shift(1)
aligned_data['exp_bond_return'] = a_bond + rho_bond * aligned_data['bond_returns'].shift(1)


aligned_data['vol_stock'] = 0.0  # Initialisation plus sûre
aligned_data['vol_bond'] = 0.0   # Initialisation plus sûre



denom_stock = 1 - alpha_stock - beta_stock
denom_bond = 1 - alpha_bond - beta_bond

print("\nDIAGNOSTIC - Dénominateurs pour volatilités initiales:")
print(f"1 - alpha_stock - beta_stock = {denom_stock}")
print(f"1 - alpha_bond - beta_bond = {denom_bond}")


if abs(denom_stock) < 1e-10:
    print("ATTENTION: Dénominateur proche de 0 pour stock - utilisation d'une valeur par défaut")
    aligned_data.loc[aligned_data.index[0], 'vol_stock'] = aligned_data['stock_residus'].std()
else:
    aligned_data.loc[aligned_data.index[0], 'vol_stock'] = np.sqrt(omega_stock / denom_stock)
    
if abs(denom_bond) < 1e-10:
    print("ATTENTION: Dénominateur proche de 0 pour bond - utilisation d'une valeur par défaut")
    aligned_data.loc[aligned_data.index[0], 'vol_bond'] = aligned_data['bond_residus'].std()
else:
    aligned_data.loc[aligned_data.index[0], 'vol_bond'] = np.sqrt(omega_bond / denom_bond)


print(f"Vol. initiale stock: {aligned_data['vol_stock'].iloc[0]}")
print(f"Vol. initiale bond: {aligned_data['vol_bond'].iloc[0]}")


for t in range(1, len(aligned_data)):
    
    prev_vol_stock = aligned_data['vol_stock'].iloc[t-1]
    prev_resid_stock = aligned_data['stock_residus'].iloc[t-1]
    vol_stock_squared = omega_stock + alpha_stock * (prev_resid_stock**2) + beta_stock * (prev_vol_stock**2)
    
    
    prev_vol_bond = aligned_data['vol_bond'].iloc[t-1]
    prev_resid_bond = aligned_data['bond_residus'].iloc[t-1]
    vol_bond_squared = omega_bond + alpha_bond * (prev_resid_bond**2) + beta_bond * (prev_vol_bond**2)
    
    
    if np.isnan(vol_stock_squared) or vol_stock_squared <= 0:
        vol_stock_squared = omega_stock  
    if np.isnan(vol_bond_squared) or vol_bond_squared <= 0:
        vol_bond_squared = omega_bond    
    
    
    aligned_data.loc[aligned_data.index[t], 'vol_stock'] = np.sqrt(vol_stock_squared)
    aligned_data.loc[aligned_data.index[t], 'vol_bond'] = np.sqrt(vol_bond_squared)


aligned_data['cov_stock_bond'] = rho_sb * aligned_data['vol_stock'] * aligned_data['vol_bond']


print("\nDIAGNOSTIC - Après calcul des volatilités:")
print(f"NaNs dans vol_stock: {aligned_data['vol_stock'].isna().sum()}")
print(f"NaNs dans vol_bond: {aligned_data['vol_bond'].isna().sum()}")
print(f"NaNs dans cov_stock_bond: {aligned_data['cov_stock_bond'].isna().sum()}")
print(f"Valeurs infinies dans vol_stock: {np.isinf(aligned_data['vol_stock']).sum()}")
print(f"Valeurs infinies dans vol_bond: {np.isinf(aligned_data['vol_bond']).sum()}")


print("\nStatistiques des volatilités calculées:")
print(aligned_data[['vol_stock', 'vol_bond']].describe())


def optimal_weights_dynamic(exp_returns, rf, sigma_matrix, lmbda):
    """Calcule les poids optimaux du portefeuille avec gestion d'erreurs améliorée"""
    try:
        
        if np.isnan(exp_returns).any() or np.isnan(rf) or np.isnan(sigma_matrix).any():
            print(f"ATTENTION: NaN détecté - exp_returns={exp_returns}, rf={rf}, sigma_matrix={sigma_matrix}")
            return np.array([0, 0])
            
        mu = exp_returns.reshape(-1, 1)
        e = np.ones((2, 1))
        
        
        det_sigma = np.linalg.det(sigma_matrix)
        if abs(det_sigma) < 1e-10:
            print(f"ATTENTION: Matrice singulière détectée (det={det_sigma}) - régularisation appliquée")
            sigma_matrix = sigma_matrix + np.eye(2) * max(1e-6, np.abs(sigma_matrix).max() * 1e-3)
        
        
        inv_sigma = np.linalg.inv(sigma_matrix)
        alpha = (1 / lmbda) * inv_sigma @ (mu - rf * e)
        
        
        if np.isnan(alpha).any() or np.isinf(alpha).any():
            print(f"ATTENTION: Poids invalides calculés: {alpha}")
            return np.array([0, 0])
            
        return alpha.flatten()
        
    except Exception as e:
        print(f"ERREUR dans optimal_weights_dynamic: {e}")
        return np.array([0, 0])



aligned_data['w_stock_lambda2'] = np.nan
aligned_data['w_bond_lambda2'] = np.nan
aligned_data['w_rf_lambda2'] = np.nan
aligned_data['w_stock_lambda10'] = np.nan
aligned_data['w_bond_lambda10'] = np.nan
aligned_data['w_rf_lambda10'] = np.nan


error_count_lambda2 = 0
error_count_lambda10 = 0


for t in range(1, len(aligned_data)):
    try:
        
        exp_returns = np.array([
            aligned_data['exp_stock_return'].iloc[t],
            aligned_data['exp_bond_return'].iloc[t]
        ])
        
        
        if np.isnan(exp_returns).any():
            error_count_lambda2 += 1
            error_count_lambda10 += 1
            continue
            
        rf = aligned_data['rf'].iloc[t]
        
        sigma_matrix = np.array([
            [aligned_data['vol_stock'].iloc[t]**2, aligned_data['cov_stock_bond'].iloc[t]],
            [aligned_data['cov_stock_bond'].iloc[t], aligned_data['vol_bond'].iloc[t]**2]
        ])
        
        
        weights_lambda2 = optimal_weights_dynamic(exp_returns, rf, sigma_matrix, 2)
        if not np.isnan(weights_lambda2).any():
            aligned_data.loc[aligned_data.index[t], 'w_stock_lambda2'] = weights_lambda2[0]
            aligned_data.loc[aligned_data.index[t], 'w_bond_lambda2'] = weights_lambda2[1]
            aligned_data.loc[aligned_data.index[t], 'w_rf_lambda2'] = 1 - weights_lambda2.sum()
        else:
            error_count_lambda2 += 1
        
        
        weights_lambda10 = optimal_weights_dynamic(exp_returns, rf, sigma_matrix, 10)
        if not np.isnan(weights_lambda10).any():
            aligned_data.loc[aligned_data.index[t], 'w_stock_lambda10'] = weights_lambda10[0]
            aligned_data.loc[aligned_data.index[t], 'w_bond_lambda10'] = weights_lambda10[1]
            aligned_data.loc[aligned_data.index[t], 'w_rf_lambda10'] = 1 - weights_lambda10.sum()
        else:
            error_count_lambda10 += 1
            
    except Exception as e:
        print(f"ERREUR à la date {aligned_data.index[t]}: {e}")
        error_count_lambda2 += 1
        error_count_lambda10 += 1

print(f"\nDIAGNOSTIC - Nombre d'erreurs dans le calcul des poids:")
print(f"λ=2: {error_count_lambda2}/{len(aligned_data)-1} périodes")
print(f"λ=10: {error_count_lambda10}/{len(aligned_data)-1} périodes")



print("\nDIAGNOSTIC - NaNs dans les poids calculés:")
for col in ['w_stock_lambda2', 'w_bond_lambda2', 'w_rf_lambda2', 
           'w_stock_lambda10', 'w_bond_lambda10', 'w_rf_lambda10']:
    print(f"{col}: {aligned_data[col].isna().sum()}/{len(aligned_data)} NaNs")


weights_cols_lambda2 = ['w_stock_lambda2', 'w_bond_lambda2', 'w_rf_lambda2']
weights_cols_lambda10 = ['w_stock_lambda10', 'w_bond_lambda10', 'w_rf_lambda10']

dynamic_allocation_clean = aligned_data.dropna(subset=weights_cols_lambda2 + weights_cols_lambda10)
print(f"\nNombre de périodes avec des poids dynamiques valides: {len(dynamic_allocation_clean)}")


if len(dynamic_allocation_clean) == 0:
    print("ATTENTION: Aucune période avec des poids complets - tentative de récupération partielle")
    
    dynamic_allocation_clean = aligned_data.dropna(subset=weights_cols_lambda2)
    print(f"Périodes récupérées pour λ=2: {len(dynamic_allocation_clean)}")
    
    
    if len(dynamic_allocation_clean) == 0:
        dynamic_allocation_clean = aligned_data.dropna(subset=weights_cols_lambda10)
        print(f"Périodes récupérées pour λ=10: {len(dynamic_allocation_clean)}")


if len(dynamic_allocation_clean) > 0:
    
    lambda2_cols = [col for col in weights_cols_lambda2 if col in dynamic_allocation_clean.columns]
    if lambda2_cols:
        print("\nStatistiques des poids dynamiques (λ=2):")
        print(dynamic_allocation_clean[lambda2_cols].describe().round(4))
    
    lambda10_cols = [col for col in weights_cols_lambda10 if col in dynamic_allocation_clean.columns]
    if lambda10_cols:
        print("\nStatistiques des poids dynamiques (λ=10):")
        print(dynamic_allocation_clean[lambda10_cols].describe().round(4))
else:
    print("\nAUCUNE DONNÉE VALIDE POUR AFFICHER DES STATISTIQUES")



def plot_dynamic_weights_corrected(dynamic_df, static_weights_lambda2, static_weights_lambda10):
    """
    Visualise les poids dynamiques vs statiques pour les actions et obligations
    en regroupant les différentes valeurs de lambda sur les mêmes graphiques
    """
    if len(dynamic_df) == 0:
        print("ERREUR: Aucune donnée valide pour créer les graphiques")
        return

    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (22, 10)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'
    
    
    static_w_stock_lambda2 = float(static_weights_lambda2[0, 0])
    static_w_bond_lambda2 = float(static_weights_lambda2[1, 0])
    static_w_stock_lambda10 = float(static_weights_lambda10[0, 0])
    static_w_bond_lambda10 = float(static_weights_lambda10[1, 0])
    
    
    fig, axes = plt.subplots(2, 1, figsize=(22, 10), sharex=True, dpi=50)
    
    
    colors = {
        'stock_lambda2': '#1f77b4',   # Bleu foncé
        'stock_lambda10': '#fc0303',  # Bleu clair
        'bond_lambda2': '#ff7f0e',    # Orange foncé
        'bond_lambda10': '#00b52d'    # Orange clair
    }
    
    
    has_lambda2_data = all(col in dynamic_df.columns for col in ['w_stock_lambda2', 'w_bond_lambda2'])
    has_lambda10_data = all(col in dynamic_df.columns for col in ['w_stock_lambda10', 'w_bond_lambda10'])
    
    
    ax_stock = axes[0]
    
    if has_lambda2_data:
        ax_stock.plot(dynamic_df.index, dynamic_df['w_stock_lambda2'], 
                     label=f'λ=2 (Dynamic)', color=colors['stock_lambda2'], linewidth=1)
        ax_stock.axhline(y=static_w_stock_lambda2, color=colors['stock_lambda2'], linestyle='--', 
                        label=f'λ=2 (Static): {static_w_stock_lambda2:.2f}', alpha=0.6, linewidth=1)
    
    if has_lambda10_data:
        ax_stock.plot(dynamic_df.index, dynamic_df['w_stock_lambda10'], 
                     label=f'λ=10 (Dynamic)', color=colors['stock_lambda10'], linewidth=1)
        ax_stock.axhline(y=static_w_stock_lambda10, color=colors['stock_lambda10'], linestyle='--', 
                        label=f'λ=10 (Static): {static_w_stock_lambda10:.2f}', alpha=0.6, linewidth=1)
    
    
    ax_stock.set_title("Stock Weight Evolution for Different Risk Aversion Levels", fontweight='bold', fontsize=14)
    ax_stock.set_ylabel("Weight", fontsize=12)
    ax_stock.legend(loc='upper left', frameon=True, fancybox=True, fontsize=12)
    ax_stock.grid(True, linestyle=':', alpha=0.3)
    
    
    if has_lambda2_data and has_lambda10_data:
        ax_stock.fill_between(dynamic_df.index, 
                           dynamic_df['w_stock_lambda2'], 
                           dynamic_df['w_stock_lambda10'],
                           color='#1f77b4', alpha=0.1)
    
    
    ax_bond = axes[1]
    
    if has_lambda2_data:
        ax_bond.plot(dynamic_df.index, dynamic_df['w_bond_lambda2'], 
                    label=f'λ=2 (Dynamic)', color=colors['bond_lambda2'], linewidth=1)
        ax_bond.axhline(y=static_w_bond_lambda2, color=colors['bond_lambda2'], linestyle='--', 
                       label=f'λ=2 (Static): {static_w_bond_lambda2:.2f}', alpha=0.6, linewidth=1)
    
    if has_lambda10_data:
        ax_bond.plot(dynamic_df.index, dynamic_df['w_bond_lambda10'], 
                    label=f'λ=10 (Dynamic)', color=colors['bond_lambda10'], linewidth=2)
        ax_bond.axhline(y=static_w_bond_lambda10, color=colors['bond_lambda10'], linestyle='--', 
                       label=f'λ=10 (Static): {static_w_bond_lambda10:.2f}', alpha=0.6, linewidth=1)
    
    
    ax_bond.set_title("Bond Weight Evolution for Different Risk Aversion Levels", fontweight='bold', fontsize=14)
    ax_bond.set_ylabel("Weight", fontsize=12)
    ax_bond.set_xlabel("Date", fontsize=12)
    ax_bond.legend(loc='upper left', frameon=True, fancybox=True, fontsize=12)
    ax_bond.grid(True, linestyle=':', alpha=0.3)
    
    
    if has_lambda2_data and has_lambda10_data:
        ax_bond.fill_between(dynamic_df.index, 
                          dynamic_df['w_bond_lambda2'], 
                          dynamic_df['w_bond_lambda10'],
                          color='#ff7f0e', alpha=0.1)
    
    
    
    import matplotlib.dates as mdates
    years_fmt = mdates.DateFormatter('%Y')
    ax_bond.xaxis.set_major_formatter(years_fmt)
    ax_bond.xaxis.set_major_locator(mdates.YearLocator(2))  
    
    
    
    important_dates = {
        pd.Timestamp('2008-09-15'): 'Lehman Brothers',
        pd.Timestamp('2020-03-11'): 'COVID-19'
    }
    
    for date, label in important_dates.items():
        if date >= dynamic_df.index.min() and date <= dynamic_df.index.max():
            for ax in axes:
                ax.axvline(x=date, color='black', linestyle='-.', alpha=0.5, linewidth=1)
                ax.text(date, ax.get_ylim()[1]*0.95, label, rotation=90, 
                       verticalalignment='top', fontsize=12, alpha=0.7)
    
    
    fig.suptitle("Time Series of Optimal Weights: Static vs Dynamic Allocation", 
                fontsize=16, fontweight='bold')
    
    
    
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.25, top=0.92)
    plt.show()


if len(dynamic_allocation_clean) > 0:
    plot_dynamic_weights_corrected(dynamic_allocation_clean, alpha_2, alpha_10)
else:
    print("Pas de données valides pour tracer les graphiques.")





dyna_count = dynamic_allocation_clean.count()
print_directory = dynamic_allocation_clean.describe().round(4)
print("\nDIAGNOSTIC - Number of periods with valid dynamic weights:")
print(f"λ=2: {dyna_count['w_stock_lambda2']} periods, λ=10: {dyna_count['w_stock_lambda10']} periods")






print('# ---------------------- Q 3.1 Dynamic Allocation with Constrains -------------- #')





def optimal_weights_dynamic_constrained(exp_returns, rf, sigma_matrix, lmbda, lower_bound=-1, upper_bound=1.5):
    """
    Calcule les poids optimaux du portefeuille avec contraintes et gestion d'erreurs
    
    Parameters:
    -----------
    exp_returns : array
        Rendements espérés des actifs
    rf : float
        Taux sans risque
    sigma_matrix : array (2x2) 
        Matrice de variance-covariance
    lmbda : float
        Coefficient d'aversion au risque
    lower_bound : float
        Borne inférieure pour les poids
    upper_bound : float
        Borne supérieure pour les poids
        
    Returns:
    --------
    array
        Poids optimaux contraints
    """
    try:
        
        if np.isnan(exp_returns).any() or np.isnan(rf) or np.isnan(sigma_matrix).any():
            return np.array([0, 0])
            
        mu = exp_returns.reshape(-1, 1)
        e = np.ones((2, 1))
        
        
        det_sigma = np.linalg.det(sigma_matrix)
        if abs(det_sigma) < 1e-10:
            sigma_matrix = sigma_matrix + np.eye(2) * max(1e-6, np.abs(sigma_matrix).max() * 1e-3)
        
        
        inv_sigma = np.linalg.inv(sigma_matrix)
        alpha = (1 / lmbda) * inv_sigma @ (mu - rf * e)
        alpha = alpha.flatten()
        
        
        alpha_constrained = np.clip(alpha, lower_bound, upper_bound)
        
        if np.isnan(alpha_constrained).any() or np.isinf(alpha_constrained).any():
            return np.array([0, 0])
            
        return alpha_constrained
        
    except Exception as e:
        print(f"ERREUR dans optimal_weights_dynamic_constrained: {e}")
        return np.array([0, 0])
    




lower_bound = -1.0
upper_bound = 1.5


aligned_data['w_stock_lambda2_constrained'] = np.nan
aligned_data['w_bond_lambda2_constrained'] = np.nan
aligned_data['w_rf_lambda2_constrained'] = np.nan
aligned_data['w_stock_lambda10_constrained'] = np.nan
aligned_data['w_bond_lambda10_constrained'] = np.nan
aligned_data['w_rf_lambda10_constrained'] = np.nan


error_count_constrained = 0


for t in range(1, len(aligned_data)):
    try:
        
        exp_returns = np.array([
            aligned_data['exp_stock_return'].iloc[t],
            aligned_data['exp_bond_return'].iloc[t]
        ])
        
        if np.isnan(exp_returns).any():
            error_count_constrained += 1
            continue
            
        rf = aligned_data['rf'].iloc[t]
        
        sigma_matrix = np.array([
            [aligned_data['vol_stock'].iloc[t]**2, aligned_data['cov_stock_bond'].iloc[t]],
            [aligned_data['cov_stock_bond'].iloc[t], aligned_data['vol_bond'].iloc[t]**2]
        ])
        
        
        weights_lambda2 = optimal_weights_dynamic_constrained(
            exp_returns, rf, sigma_matrix, 2, lower_bound, upper_bound
        )
        if not np.isnan(weights_lambda2).any():
            aligned_data.loc[aligned_data.index[t], 'w_stock_lambda2_constrained'] = weights_lambda2[0]
            aligned_data.loc[aligned_data.index[t], 'w_bond_lambda2_constrained'] = weights_lambda2[1]
            aligned_data.loc[aligned_data.index[t], 'w_rf_lambda2_constrained'] = 1 - weights_lambda2.sum()
        
        
        weights_lambda10 = optimal_weights_dynamic_constrained(
            exp_returns, rf, sigma_matrix, 10, lower_bound, upper_bound
        )
        if not np.isnan(weights_lambda10).any():
            aligned_data.loc[aligned_data.index[t], 'w_stock_lambda10_constrained'] = weights_lambda10[0]
            aligned_data.loc[aligned_data.index[t], 'w_bond_lambda10_constrained'] = weights_lambda10[1]
            aligned_data.loc[aligned_data.index[t], 'w_rf_lambda10_constrained'] = 1 - weights_lambda10.sum()
            
    except Exception as e:
        print(f"ERREUR à la date {aligned_data.index[t]}: {e}")
        error_count_constrained += 1

print(f"\nDIAGNOSTIC - Nombre d'erreurs dans le calcul des poids contraints: {error_count_constrained}/{len(aligned_data)-1} périodes")


weights_cols_constrained = [
    'w_stock_lambda2_constrained', 'w_bond_lambda2_constrained', 'w_rf_lambda2_constrained',
    'w_stock_lambda10_constrained', 'w_bond_lambda10_constrained', 'w_rf_lambda10_constrained'
]


dynamic_allocation_with_constraints = aligned_data.dropna(subset=weights_cols_lambda2 + weights_cols_lambda10 + weights_cols_constrained)
print(f"\nNombre de périodes avec tous les poids valides: {len(dynamic_allocation_with_constraints)}")


if len(dynamic_allocation_with_constraints) > 0:
    print("\nStatistiques des poids non contraints vs contraints (λ=2):")
    comparison_lambda2 = pd.DataFrame({
        'Non contraint - Stock': dynamic_allocation_with_constraints['w_stock_lambda2'],
        'Contraint - Stock': dynamic_allocation_with_constraints['w_stock_lambda2_constrained'],
        'Non contraint - Bond': dynamic_allocation_with_constraints['w_bond_lambda2'],
        'Contraint - Bond': dynamic_allocation_with_constraints['w_bond_lambda2_constrained']
    })
    print(comparison_lambda2.describe().round(4))
    
    print("\nStatistiques des poids non contraints vs contraints (λ=10):")
    comparison_lambda10 = pd.DataFrame({
        'Non contraint - Stock': dynamic_allocation_with_constraints['w_stock_lambda10'],
        'Contraint - Stock': dynamic_allocation_with_constraints['w_stock_lambda10_constrained'],
        'Non contraint - Bond': dynamic_allocation_with_constraints['w_bond_lambda10'],
        'Contraint - Bond': dynamic_allocation_with_constraints['w_bond_lambda10_constrained']
    })
    print(comparison_lambda10.describe().round(4))

def plot_dynamic_weights_constrained_improved(dynamic_df, static_weights_lambda2, static_weights_lambda10, 
                                            lower_bound=-1, upper_bound=1.5):
    """
    Professional visualization of optimal allocations with and without constraints.
    Shows stock and bond weights together for each risk aversion level.
    """
    if len(dynamic_df) == 0:
        print("ERROR: No valid data to create charts")
        return

    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (22, 12)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['legend.fontsize'] = 12
    
    
    static_w_stock_lambda2 = float(static_weights_lambda2[0, 0])
    static_w_bond_lambda2 = float(static_weights_lambda2[1, 0])
    static_w_stock_lambda10 = float(static_weights_lambda10[0, 0])
    static_w_bond_lambda10 = float(static_weights_lambda10[1, 0])
    
    
    fig, axes = plt.subplots(2, 1, figsize=(22, 12), sharex=True, dpi=50)
    
    
    colors = {
        'stock_unconst': '#1f77b4',     # Dark blue
        'stock_const': '#63a8e2',       # Medium blue
        'bond_unconst': '#d62728',      # Red
        'bond_const': '#ff9896',        # Light red
        'stock_static': '#2ca02c',      # Green
        'bond_static': '#ff7f0e'        # Orange
    }
    
    
    y_min = min(lower_bound - 0.2, -2)
    y_max = max(upper_bound + 0.2, 2)
    
    
    ax_lambda2 = axes[0]
    
    
    ax_lambda2.plot(dynamic_df.index, dynamic_df['w_stock_lambda2'], 
                 label='Stocks (unconstrained)', color=colors['stock_unconst'], linewidth=1)
    ax_lambda2.plot(dynamic_df.index, dynamic_df['w_stock_lambda2_constrained'], 
                 label='Stocks (constrained)', color=colors['stock_const'], linewidth=1, linestyle='-')
    
    
    ax_lambda2.plot(dynamic_df.index, dynamic_df['w_bond_lambda2'], 
                 label='Bonds (unconstrained)', color=colors['bond_unconst'], linewidth=1)
    ax_lambda2.plot(dynamic_df.index, dynamic_df['w_bond_lambda2_constrained'], 
                 label='Bonds (constrained)', color=colors['bond_const'], linewidth=1, linestyle='-')
    
    
    ax_lambda2.axhline(y=static_w_stock_lambda2, color=colors['stock_static'], linestyle='--', 
                     label=f'Stocks (static): {static_w_stock_lambda2:.2f}', linewidth=1)
    ax_lambda2.axhline(y=static_w_bond_lambda2, color=colors['bond_static'], linestyle='--', 
                     label=f'Bonds (static): {static_w_bond_lambda2:.2f}', linewidth=1)
    
    
    ax_lambda2.axhline(y=lower_bound, color='gray', linestyle=':', alpha=0.7)
    ax_lambda2.axhline(y=upper_bound, color='gray', linestyle=':', alpha=0.7)
    ax_lambda2.text(dynamic_df.index[0], lower_bound - 0.05, f"Lower limit: {lower_bound}", 
                  fontsize=10, color='gray', ha='left', va='top')
    ax_lambda2.text(dynamic_df.index[0], upper_bound + 0.05, f"Upper limit: {upper_bound}", 
                  fontsize=10, color='gray', ha='left', va='bottom')
    
    
    ax_lambda2.set_title("Optimal Allocation for λ=2 (Low Risk Aversion)", 
                       fontweight='bold', fontsize=16)
    ax_lambda2.set_ylabel("Portfolio Weight", fontsize=14)
    ax_lambda2.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, 
                    fancybox=True, shadow=True, fontsize=12)
    ax_lambda2.grid(True, linestyle=':', alpha=0.3)
    ax_lambda2.set_ylim(y_min, y_max)  
    
    
    ax_lambda10 = axes[1]
    
    
    ax_lambda10.plot(dynamic_df.index, dynamic_df['w_stock_lambda10'], 
                  label='Stocks (unconstrained)', color=colors['stock_unconst'], linewidth=1)
    ax_lambda10.plot(dynamic_df.index, dynamic_df['w_stock_lambda10_constrained'], 
                  label='Stocks (constrained)', color=colors['stock_const'], linewidth=1, linestyle='-')
    
    
    ax_lambda10.plot(dynamic_df.index, dynamic_df['w_bond_lambda10'], 
                  label='Bonds (unconstrained)', color=colors['bond_unconst'], linewidth=1)
    ax_lambda10.plot(dynamic_df.index, dynamic_df['w_bond_lambda10_constrained'], 
                  label='Bonds (constrained)', color=colors['bond_const'], linewidth=1, linestyle='-')
    
    
    ax_lambda10.axhline(y=static_w_stock_lambda10, color=colors['stock_static'], linestyle='--', 
                      label=f'Stocks (static): {static_w_stock_lambda10:.2f}', linewidth=1)
    ax_lambda10.axhline(y=static_w_bond_lambda10, color=colors['bond_static'], linestyle='--', 
                      label=f'Bonds (static): {static_w_bond_lambda10:.2f}', linewidth=1)
    
    
    ax_lambda10.axhline(y=lower_bound, color='gray', linestyle=':', alpha=0.7)
    ax_lambda10.axhline(y=upper_bound, color='gray', linestyle=':', alpha=0.7)
    ax_lambda10.text(dynamic_df.index[0], lower_bound - 0.05, f"Lower limit: {lower_bound}", 
                   fontsize=10, color='gray', ha='left', va='top')
    ax_lambda10.text(dynamic_df.index[0], upper_bound + 0.05, f"Upper limit: {upper_bound}", 
                   fontsize=10, color='gray', ha='left', va='bottom')
    
    
    ax_lambda10.set_title("Optimal Allocation for λ=10 (High Risk Aversion)", 
                        fontweight='bold', fontsize=16)
    ax_lambda10.set_ylabel("Portfolio Weight", fontsize=14)
    ax_lambda10.set_xlabel("Date", fontsize=14, fontweight='bold')
    ax_lambda10.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, 
                     fancybox=True, shadow=True, fontsize=12)
    ax_lambda10.grid(True, linestyle=':', alpha=0.3)
    ax_lambda10.set_ylim(y_min, y_max)  
    
    
    for ax in axes:
        years_fmt = mdates.DateFormatter('%Y')
        ax.xaxis.set_major_formatter(years_fmt)
        ax.xaxis.set_major_locator(mdates.YearLocator(1))  
    
    
    important_dates = {
        pd.Timestamp('2008-09-15'): 'Lehman Brothers',
        pd.Timestamp('2020-03-11'): 'COVID-19'
    }
    
    
    crisis_periods = {
        'Financial Crisis': (pd.Timestamp('2008-09-01'), pd.Timestamp('2009-06-30')),
        'COVID-19': (pd.Timestamp('2020-02-15'), pd.Timestamp('2020-06-30'))
    }
    
    crisis_colors = {
        'Financial Crisis': 'red',
        'COVID-19': 'purple'
    }
    
    
    for crisis_name, (start_date, end_date) in crisis_periods.items():
        if (start_date >= dynamic_df.index.min() and start_date <= dynamic_df.index.max() and
            end_date >= dynamic_df.index.min() and end_date <= dynamic_df.index.max()):
            for ax in axes:
                ax.axvspan(start_date, end_date, alpha=0.15, color=crisis_colors[crisis_name], 
                          label=crisis_name)
                
                
                middle_date = start_date + (end_date - start_date) / 2
                ax.text(middle_date, y_max*0.9, crisis_name, 
                       ha='center', fontsize=10, fontweight='bold',
                       bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    
    
    for date, label in important_dates.items():
        if date >= dynamic_df.index.min() and date <= dynamic_df.index.max():
            for ax in axes:
                ax.axvline(x=date, color='black', linestyle='-.', alpha=0.4, linewidth=1)
                
                if label == 'Lehman Brothers':
                    y_pos = y_max * 0.8
                    va = 'top'
                else:
                    y_pos = y_min * 0.8
                    va = 'bottom'
                ax.text(date, y_pos, label, rotation=90, 
                    verticalalignment=va, horizontalalignment='right',
                    fontsize=10, fontweight='bold', alpha=0.7,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', 
                                boxstyle='round,pad=0.3'))
    
    
    fig.text(0.01, 0.5, "Constraints limit positions to values between -1 and 1.5", 
            fontsize=11, rotation=90, va='center', ha='left', color='gray', alpha=0.8)
    
    
    fig.suptitle("Impact of Constraints on Dynamic Asset Allocation", 
                fontsize=20, fontweight='bold', y=0.98)
    
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, right=0.85, hspace=0.2)  
    
    plt.show()


if len(dynamic_allocation_with_constraints) > 0:
    print("\nAffichage des allocations dynamiques avec et sans contraintes:")
    plot_dynamic_weights_constrained_improved(
        dynamic_allocation_with_constraints, 
        alpha_2, alpha_10,
        lower_bound=lower_bound,
        upper_bound=upper_bound
    )
else:
    print("\nPas de données valides pour afficher les allocations contraintes.")





def plot_expected_return_variance_ratio_and_weights(dynamic_df, lower_bound=-1, upper_bound=1.5):
    """
    Visualise la relation entre le ratio rendement excédentaire/variance et les poids optimaux des actions
    """
    
    required_cols = ['exp_stock_return', 'rf', 'vol_stock', 'w_stock_lambda2']
    if not all(col in dynamic_df.columns for col in required_cols):
        print("ERROR: Missing required columns for this visualization")
        return
        
    
    df_plot = dynamic_df.copy()
    df_plot.loc[:, 'excess_return'] = df_plot['exp_stock_return'] - df_plot['rf']
    df_plot.loc[:, 'variance'] = df_plot['vol_stock']**2
    df_plot.loc[:, 'return_variance_ratio'] = df_plot['excess_return'] / df_plot['variance']
    
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (22, 10)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['legend.fontsize'] = 12
    
    
    fig, ax1 = plt.subplots(figsize=(22, 10), dpi=50)
    
    
    color_weights = '#1f77b4'  # Bleu
    ax1.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Stock Weight (λ=2)', fontsize=14, color=color_weights)
    ax1.plot(dynamic_df.index, dynamic_df['w_stock_lambda2'], color=color_weights, linewidth=1, 
             label='Unconstrained Stock Weight (λ=2)')
    ax1.tick_params(axis='y', labelcolor=color_weights)
    ax1.grid(True, linestyle=':', alpha=0.3)
    
    
    ax1.axhline(y=lower_bound, color='gray', linestyle=':', alpha=0.7, 
               label=f'Lower Constraint: {lower_bound}')
    ax1.axhline(y=upper_bound, color='gray', linestyle=':', alpha=0.7, 
               label=f'Upper Constraint: {upper_bound}')
    
    
    ax2 = ax1.twinx()
    color_ratio = 'black'
    ax2.set_ylabel('Expected Excess Return-to-Variance Ratio (Stocks)', fontsize=14, color=color_ratio)  # Précision ajoutée
    ax2.plot(dynamic_df.index, df_plot['return_variance_ratio'], color=color_ratio, linewidth=1,
             label='Stock Return-to-Variance Ratio (μ-Rf)/σ²')  # Légende améliorée
    ax2.tick_params(axis='y', labelcolor=color_ratio)
    
    
    threshold = 20
    ax2.axhline(y=threshold, color='red', linestyle='--', alpha=0.7,
               label=f'Critical Threshold: {threshold}')
    
    
    low_vol_start = pd.Timestamp('2015-01-01')
    low_vol_end = pd.Timestamp('2019-12-31')
    
    
    if (low_vol_start >= dynamic_df.index.min() and low_vol_start <= dynamic_df.index.max() and
        low_vol_end >= dynamic_df.index.min() and low_vol_end <= dynamic_df.index.max()):
        ax1.axvspan(low_vol_start, low_vol_end, alpha=0.2, color='lightgreen', 
                  label='Low Volatility Period (2015-2019)')
        
        
        mid_date = low_vol_start + (low_vol_end - low_vol_start) / 2
        ax1.text(mid_date, ax1.get_ylim()[1]*0.9, "Low Volatility Period", 
               ha='center', fontsize=12, fontweight='bold',
               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    
    
    important_dates = {
        pd.Timestamp('2008-09-15'): 'Lehman Brothers',
        pd.Timestamp('2020-03-11'): 'COVID-19'
    }
    
    
    crisis_periods = {
        'Financial Crisis': (pd.Timestamp('2008-09-01'), pd.Timestamp('2009-06-30')),
        'COVID-19': (pd.Timestamp('2020-02-15'), pd.Timestamp('2020-06-30'))
    }
    
    crisis_colors = {
        'Financial Crisis': 'red',
        'COVID-19': 'purple'
    }
    
    
    for crisis_name, (start_date, end_date) in crisis_periods.items():
        if (start_date >= dynamic_df.index.min() and start_date <= dynamic_df.index.max() and
            end_date >= dynamic_df.index.min() and end_date <= dynamic_df.index.max()):
            ax1.axvspan(start_date, end_date, alpha=0.15, color=crisis_colors[crisis_name], 
                      label=crisis_name)
    
    
    for date, label in important_dates.items():
        if date >= dynamic_df.index.min() and date <= dynamic_df.index.max():
            ax1.axvline(x=date, color='black', linestyle='-.', alpha=0.4, linewidth=1)
            
            y_pos = ax1.get_ylim()[1] * 0.95
            ax1.text(date, y_pos, label, rotation=90, 
                   verticalalignment='top', horizontalalignment='right',
                   fontsize=10, fontweight='bold', alpha=0.7, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', 
                                boxstyle='round,pad=0.3'))
    
    
    import matplotlib.dates as mdates
    years_fmt = mdates.DateFormatter('%Y')
    ax1.xaxis.set_major_formatter(years_fmt)
    ax1.xaxis.set_major_locator(mdates.YearLocator(1))  
    
    
    y2_min, y2_max = ax2.get_ylim()
    if y2_max < threshold * 1.5:
        ax2.set_ylim(y2_min, threshold * 1.5)
    
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(0.01, 0.99),
              frameon=True, fancybox=True, shadow=True)
    
    
    plt.title("Stock Expected Excess Return-to-Variance Ratio and Optimal Weight Inflation (Jan-2001–Dec-2024)",
             fontsize=18, fontweight='bold')
    
    
    
    
    
    
    
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  
    
    plt.show()


if len(dynamic_allocation_with_constraints) > 0:
    plot_expected_return_variance_ratio_and_weights(
        dynamic_allocation_with_constraints,
        lower_bound=lower_bound, 
        upper_bound=upper_bound
    )
else:
    print("Pas de données valides pour visualiser le ratio rendement/variance.")





print(f'Number of periods with valid constrained weights: {len(dynamic_allocation_with_constraints)}')





print('# ---------------------- Q 3.2  --------------------------- #')





def calculate_portfolio_returns(weights_df, returns_df, rf_series):
    """
    Calcule les rendements du portefeuille à partir des poids optimaux et des rendements réalisés
    
    Parameters:
    -----------
    weights_df : DataFrame
        DataFrame contenant les poids optimaux (w_stock, w_bond, etc.)
    returns_df : DataFrame
        DataFrame contenant les rendements réalisés des actifs (stock, bond)
    rf_series : Series
        Série contenant le taux sans risque
        
    Returns:
    --------
    DataFrame
        DataFrame contenant les rendements du portefeuille pour chaque stratégie
    """
    
    common_dates = weights_df.index.intersection(returns_df.index)
    weights = weights_df.loc[common_dates]
    returns = returns_df.loc[common_dates]
    rf = rf_series.loc[common_dates]
    
    
    portfolio_returns = pd.DataFrame(index=common_dates)
    
    
    
    if 'w_stock_lambda2' in weights.columns and 'w_bond_lambda2' in weights.columns:
        portfolio_returns['Static_λ2'] = (
            weights['w_stock_lambda2'] * returns['stock'] + 
            weights['w_bond_lambda2'] * returns['bond'] + 
            (1 - weights['w_stock_lambda2'] - weights['w_bond_lambda2']) * rf
        )
    
    
    if 'w_stock_lambda10' in weights.columns and 'w_bond_lambda10' in weights.columns:
        portfolio_returns['Static_λ10'] = (
            weights['w_stock_lambda10'] * returns['stock'] + 
            weights['w_bond_lambda10'] * returns['bond'] + 
            (1 - weights['w_stock_lambda10'] - weights['w_bond_lambda10']) * rf
        )
    
    
    if 'w_stock_lambda2_constrained' in weights.columns and 'w_bond_lambda2_constrained' in weights.columns:
        portfolio_returns['Dynamic_Constrained_λ2'] = (
            weights['w_stock_lambda2_constrained'] * returns['stock'] + 
            weights['w_bond_lambda2_constrained'] * returns['bond'] + 
            (1 - weights['w_stock_lambda2_constrained'] - weights['w_bond_lambda2_constrained']) * rf
        )
    
    
    if 'w_stock_lambda10_constrained' in weights.columns and 'w_bond_lambda10_constrained' in weights.columns:
        portfolio_returns['Dynamic_Constrained_λ10'] = (
            weights['w_stock_lambda10_constrained'] * returns['stock'] + 
            weights['w_bond_lambda10_constrained'] * returns['bond'] + 
            (1 - weights['w_stock_lambda10_constrained'] - weights['w_bond_lambda10_constrained']) * rf
        )
    
    return portfolio_returns

def calculate_cumulative_returns(returns_df, use_log=False):
    """
    Calcule les rendements cumulés à partir des rendements
    
    Parameters:
    -----------
    returns_df : DataFrame
        DataFrame contenant les rendements des portefeuilles
    use_log : bool, optional
        Si True, utilise le log des rendements (pour éviter l'explosion)
        
    Returns:
    --------
    DataFrame
        DataFrame contenant les rendements cumulés
    """
    cumulative_returns = pd.DataFrame(index=returns_df.index)
    
    if use_log:
        
        log_returns = np.log(1 + returns_df)
        
        for col in log_returns.columns:
            cumulative_returns[col] = np.exp(log_returns[col].cumsum())
    else:
        
        for col in returns_df.columns:
            cumulative_returns[col] = (1 + returns_df[col]).cumprod()
    
    return cumulative_returns

def calculate_performance_metrics(returns_df, rf_series=None):
    """
    Calcule les métriques de performance complètes pour chaque stratégie
    
    Parameters:
    -----------
    returns_df : DataFrame
        DataFrame contenant les rendements hebdomadaires des portefeuilles
    rf_series : Series, optional
        Série contenant le taux sans risque hebdomadaire
        
    Returns:
    --------
    DataFrame
        DataFrame contenant les métriques de performance détaillées
    """
    
    if rf_series is None:
        rf_mean = 0
    else:
        
        common_dates = returns_df.index.intersection(rf_series.index)
        rf_aligned = rf_series.loc[common_dates]
        rf_mean = rf_aligned.mean()
    
    
    WEEKS_PER_YEAR = 52
    WEEKS_PER_MONTH = 4.33
    
    
    metrics = {}
    
    for strategy in returns_df.columns:
        
        returns = returns_df[strategy].dropna()
        
        
        if rf_series is not None:
            excess_returns = returns - rf_aligned.loc[returns.index]
        else:
            excess_returns = returns
        
        
        monthly_returns = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        if rf_series is not None:
            monthly_rf = rf_aligned.resample('ME').apply(lambda x: (1 + x).prod() - 1)
            monthly_excess = monthly_returns - monthly_rf
        else:
            monthly_excess = monthly_returns
        
        
        
        annualized_avg_return = ((1 + returns.mean()) ** WEEKS_PER_YEAR - 1) * 100
        
        
        annualized_volatility = returns.std() * np.sqrt(WEEKS_PER_YEAR) * 100
        
        
        sharpe_ratio_1 = (annualized_avg_return - rf_mean * WEEKS_PER_YEAR * 100) / annualized_volatility
        
        
        annualized_excess_mean = ((1 + excess_returns.mean()) ** WEEKS_PER_YEAR - 1) * 100
        
        
        total_return = (1 + returns).prod() - 1
        n_years = len(returns) / WEEKS_PER_YEAR
        annualized_cum_return = ((1 + total_return) ** (1 / n_years) - 1) * 100
        
        
        
        monthly_mean_return = monthly_returns.mean() * 100
        
        
        monthly_volatility = monthly_returns.std() * 100
        
        
        rf_monthly_mean = monthly_rf.mean() * 100 if rf_series is not None else 0
        monthly_sharpe = (monthly_mean_return - rf_monthly_mean) / monthly_volatility if monthly_volatility > 0 else 0
        
        
        monthly_excess_mean = monthly_excess.mean() * 100
        
        
        max_monthly_return = monthly_returns.max() * 100
        min_monthly_return = monthly_returns.min() * 100
        
        
        
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns / running_max) - 1
        
        
        max_drawdown = drawdowns.min() * 100
        max_drawdown_date = drawdowns.idxmin().strftime('%Y-%m-%d')
        
        
        try:
            max_dd_idx = drawdowns.idxmin()
            max_dd_idx_pos = drawdowns.index.get_loc(max_dd_idx)
            
            recovery_idx = None
            for i in range(max_dd_idx_pos + 1, len(drawdowns)):
                if drawdowns.iloc[i] >= 0:
                    recovery_idx = drawdowns.index[i]
                    break
            
            if recovery_idx is not None:
                recovery_period = (recovery_idx - max_dd_idx).days / 30.44  
            else:
                recovery_period = float('nan')
        except:
            recovery_period = float('nan')
        
        
        metrics[strategy] = {
            "Annualized Average Return (%)": round(annualized_avg_return, 2),
            "Annualized Volatility (%)": round(annualized_volatility, 2),
            "Annualized Sharpe Ratio": round(sharpe_ratio_1, 2),
            "Annualized Excess Return (%)": round(annualized_excess_mean, 2),
            "Annualized Cumulative Return (%)": round(annualized_cum_return, 2),
            "Monthly Average Return (%)": round(monthly_mean_return, 2),
            "Monthly Volatility (%)": round(monthly_volatility, 2),
            "Monthly Sharpe Ratio": round(monthly_sharpe, 2),
            "Monthly Excess Return (%)": round(monthly_excess_mean, 2),
            "Maximum Monthly Return (%)": round(max_monthly_return, 2),
            "Minimum Monthly Return (%)": round(min_monthly_return, 2),
            "Maximum Drawdown (%)": round(max_drawdown, 2),
            "Maximum Drawdown Date": max_drawdown_date,
            "Recovery Period (months)": round(recovery_period, 1) if not np.isnan(recovery_period) else "Not recovered"
        }
    
    
    return pd.DataFrame.from_dict(metrics)


def align_portfolio_data(static_weights, dynamic_allocation_with_constraints, realized_returns, risk_free_rate):
    """
    Aligne toutes les données sur un index commun pour assurer une comparaison équitable.
    
    Cette fonction est conçue pour être utilisée avant la concaténation des poids et
    le calcul des rendements, sans modifier le pipeline principal.
    
    Parameters:
    -----------
    static_weights : DataFrame
        DataFrame des poids statiques
    dynamic_allocation_with_constraints : DataFrame
        DataFrame des poids dynamiques contraints
    realized_returns : DataFrame
        DataFrame des rendements réalisés des actifs
    risk_free_rate : Series
        Série du taux sans risque
        
    Returns:
    --------
    tuple
        (static_weights_aligned, dynamic_weights_aligned, returns_aligned, rf_aligned)
    """
    
    dynamic_weights = dynamic_allocation_with_constraints[[
        'w_stock_lambda2_constrained', 'w_bond_lambda2_constrained',
        'w_stock_lambda10_constrained', 'w_bond_lambda10_constrained'
    ]]
    
    
    common_index = dynamic_weights.index.intersection(
                    realized_returns.index.intersection(
                        risk_free_rate.index))
    
    print(f"Number of common observations after aligning Static and Dynamic Returns: {len(common_index)}")
    
    
    static_weights_aligned = pd.DataFrame(index=common_index)
    for col in static_weights.columns:
        
        static_weights_aligned[col] = static_weights[col].iloc[0]
    
    
    dynamic_weights_aligned = dynamic_weights.loc[common_index]
    returns_aligned = realized_returns.loc[common_index]
    rf_aligned = risk_free_rate.loc[common_index]
    
    return static_weights_aligned, dynamic_weights_aligned, returns_aligned, rf_aligned



print("# ---------------------- Q 3.2 - CumReturns -------------- #")



static_weights = pd.DataFrame(index=dynamic_allocation_with_constraints.index)
static_weights['w_stock_lambda2'] = alpha_2[0, 0]  # Poids constant des actions pour λ=2
static_weights['w_bond_lambda2'] = alpha_2[1, 0]   # Poids constant des obligations pour λ=2
static_weights['w_stock_lambda10'] = alpha_10[0, 0]  # Poids constant des actions pour λ=10
static_weights['w_bond_lambda10'] = alpha_10[1, 0]   # Poids constant des obligations pour λ=10


dynamic_weights = dynamic_allocation_with_constraints[[
    'w_stock_lambda2_constrained', 'w_bond_lambda2_constrained',
    'w_stock_lambda10_constrained', 'w_bond_lambda10_constrained'
]]


realized_returns = df_weekly[['stock', 'bond']]
risk_free_rate = df_weekly['rf']


static_weights_aligned, dynamic_weights_aligned, returns_aligned, rf_aligned = align_portfolio_data(
    static_weights, dynamic_allocation_with_constraints, realized_returns, risk_free_rate
)


all_weights = pd.concat([static_weights_aligned, dynamic_weights_aligned], axis=1)



portfolio_returns = calculate_portfolio_returns(all_weights, realized_returns, risk_free_rate)


use_log_returns = True  
cumulative_returns = calculate_cumulative_returns(portfolio_returns, use_log=use_log_returns)


performance_metrics = calculate_performance_metrics(portfolio_returns, risk_free_rate)


print("\nPerformance Metrics:")
print(performance_metrics.round(2).style.background_gradient(cmap='RdYlGn', axis=0))












def create_portfolio_comparison_dashboard(
    portfolio1_series, 
    portfolio2_series, 
    portfolio3_series=None,
    portfolio4_series=None,  
    rf_series=None, 
    portfolio1_name="Portfolio 1", 
    portfolio2_name="Portfolio 2", 
    portfolio3_name="Portfolio 3",
    portfolio4_name="Portfolio 4",  # Ajout du nom pour le quatrième portefeuille
    colors=None, 
    figsize=(16, 12), 
    save_path=None,
    title="Portfolio Comparison",
    
    cumulative_return_horizontal_shift=5,  
    cumulative_return_vertical_spacing=5,  
    drawdown_horizontal_shift=30,          
    drawdown_arc_radius=0.3               
):
    """
    Crée un tableau de bord simplifié comparant jusqu'à quatre portefeuilles avec:
    1. Rendements cumulés (en haut, plus grand)
    2. Drawdowns (en bas, plus petit)
    
    Paramètres:
    -----------
    portfolio1_series, portfolio2_series, portfolio3_series, portfolio4_series: pd.Series
        Séries temporelles des rendements des portefeuilles
    rf_series: pd.Series, optional
        Série temporelle du taux sans risque
    portfolio1_name, portfolio2_name, portfolio3_name, portfolio4_name: str
        Noms des portefeuilles pour la légende
    colors: dict, optional
        Dictionnaire associant les noms de portefeuilles à leurs couleurs
    figsize: tuple, optional
        Dimensions du graphique (largeur, hauteur)
    save_path: str, optional
        Chemin pour sauvegarder l'image
    title: str, optional
        Titre du graphique
    cumulative_return_horizontal_shift: int
        Contrôle le shift horizontal des annotations de valeur finale
    cumulative_return_vertical_spacing: int
        Contrôle l'espacement vertical entre les annotations de valeur finale
    drawdown_horizontal_shift: int
        Contrôle le shift horizontal des annotations de drawdown
    drawdown_arc_radius: float
        Contrôle la courbure des flèches de drawdown (0=droit, >0=courbe)
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch
    
    
    all_series = [portfolio1_series, portfolio2_series]
    if portfolio3_series is not None:
        all_series.append(portfolio3_series)
    if portfolio4_series is not None:
        all_series.append(portfolio4_series)
    
    
    common_index = all_series[0].index
    for series in all_series[1:]:
        common_index = common_index.intersection(series.index)
    
    if len(common_index) == 0:
        raise ValueError("Les séries de portefeuilles n'ont pas d'index commun")
    
    
    portfolio1_series = portfolio1_series.loc[common_index]
    portfolio2_series = portfolio2_series.loc[common_index]
    if portfolio3_series is not None:
        portfolio3_series = portfolio3_series.loc[common_index]
    if portfolio4_series is not None:
        portfolio4_series = portfolio4_series.loc[common_index]

    print(f"⚠️ Indexes have been aligned on {len(common_index)} common points retained.")

    
    if colors is None:
        colors = {
            portfolio1_name: '#1f77b4',  # Bleu
            portfolio2_name: '#ff7f0e',  # Orange
            portfolio3_name: '#2ca02c',   # Vert
            portfolio4_name: '#d62728'    # Rouge
        }
    
    
    all_series_processed = []
    for series in all_series:
        series.index = pd.to_datetime(series.index)
        series.index = series.index.normalize()
        all_series_processed.append(series)
    
    portfolio1_series = all_series_processed[0]
    portfolio2_series = all_series_processed[1]
    if portfolio3_series is not None:
        portfolio3_series = all_series_processed[2]
    if portfolio4_series is not None and len(all_series_processed) > 3:
        portfolio4_series = all_series_processed[3]
    
    
    portfolio1_cumulative_returns = (1 + portfolio1_series).cumprod()
    portfolio2_cumulative_returns = (1 + portfolio2_series).cumprod()
    
    
    cumulative_returns_comparison = {
        portfolio1_name: portfolio1_cumulative_returns,
        portfolio2_name: portfolio2_cumulative_returns
    }
    
    
    if portfolio3_series is not None:
        portfolio3_cumulative_returns = (1 + portfolio3_series).cumprod()
        cumulative_returns_comparison[portfolio3_name] = portfolio3_cumulative_returns
    
    
    if portfolio4_series is not None:
        portfolio4_cumulative_returns = (1 + portfolio4_series).cumprod()
        cumulative_returns_comparison[portfolio4_name] = portfolio4_cumulative_returns
    
    cumulative_returns_df = pd.DataFrame(cumulative_returns_comparison)

    
    
    portfolio1_drawdown = pd.Series(0.0, index=portfolio1_cumulative_returns.index, dtype=float)
    portfolio1_cummax = portfolio1_cumulative_returns.cummax()
    mask1 = portfolio1_cummax != 0
    portfolio1_drawdown.loc[mask1] = ((portfolio1_cummax.loc[mask1] - portfolio1_cumulative_returns.loc[mask1]) / 
                                    portfolio1_cummax.loc[mask1]).astype(float)

    
    portfolio2_drawdown = pd.Series(0.0, index=portfolio2_cumulative_returns.index, dtype=float)
    portfolio2_cummax = portfolio2_cumulative_returns.cummax()
    mask2 = portfolio2_cummax != 0
    portfolio2_drawdown.loc[mask2] = ((portfolio2_cummax.loc[mask2] - portfolio2_cumulative_returns.loc[mask2]) / 
                                    portfolio2_cummax.loc[mask2]).astype(float)
    
    
    portfolio3_drawdown = None
    if portfolio3_series is not None:
        portfolio3_drawdown = pd.Series(0.0, index=portfolio3_cumulative_returns.index, dtype=float)
        portfolio3_cummax = portfolio3_cumulative_returns.cummax()
        mask3 = portfolio3_cummax != 0
        portfolio3_drawdown.loc[mask3] = ((portfolio3_cummax.loc[mask3] - portfolio3_cumulative_returns.loc[mask3]) / 
                                        portfolio3_cummax.loc[mask3]).astype(float)
    
    
    portfolio4_drawdown = None
    if portfolio4_series is not None:
        portfolio4_drawdown = pd.Series(0.0, index=portfolio4_cumulative_returns.index, dtype=float)
        portfolio4_cummax = portfolio4_cumulative_returns.cummax()
        mask4 = portfolio4_cummax != 0
        portfolio4_drawdown.loc[mask4] = ((portfolio4_cummax.loc[mask4] - portfolio4_cumulative_returns.loc[mask4]) / 
                                        portfolio4_cummax.loc[mask4]).astype(float)

    
    portfolio1_max_dd = portfolio1_drawdown.max() 
    portfolio2_max_dd = portfolio2_drawdown.max()
    portfolio3_max_dd = None
    if portfolio3_drawdown is not None:
        portfolio3_max_dd = portfolio3_drawdown.max()
    portfolio4_max_dd = None
    if portfolio4_drawdown is not None:
        portfolio4_max_dd = portfolio4_drawdown.max()

    
    plt.style.use('seaborn-v0_8-whitegrid')

    
    def apply_custom_style(ax, show_grid=True):
        """Apply custom styling to a matplotlib axis"""
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if show_grid:
            ax.grid(True, linestyle='--', alpha=0.4, color='#cccccc')
        else:
            ax.grid(False)  
        ax.xaxis.label.set_fontsize(12)
        ax.yaxis.label.set_fontsize(12)
        ax.tick_params(axis='both', which='major', labelsize=10, colors='#333333')
        return ax

    
    fig = plt.figure(figsize=figsize, facecolor='white', dpi=50)
    
    
    
    gs = GridSpec(2, 1, figure=fig, height_ratios=[0.80, 0.20])
    
    
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(cumulative_returns_df.index, cumulative_returns_df[portfolio1_name], 
            color=colors[portfolio1_name], linewidth=2, label=portfolio1_name)
    ax1.plot(cumulative_returns_df.index, cumulative_returns_df[portfolio2_name], 
            color=colors[portfolio2_name], linewidth=2, label=portfolio2_name)
    
    
    if portfolio3_name in cumulative_returns_df.columns:
        ax1.plot(cumulative_returns_df.index, cumulative_returns_df[portfolio3_name], 
                color=colors[portfolio3_name], linewidth=2, label=portfolio3_name)
    
    
    if portfolio4_name in cumulative_returns_df.columns:
        ax1.plot(cumulative_returns_df.index, cumulative_returns_df[portfolio4_name], 
                color=colors[portfolio4_name], linewidth=2, label=portfolio4_name)

    
    
    final_values = {
        portfolio1_name: cumulative_returns_df[portfolio1_name].iloc[-1],
        portfolio2_name: cumulative_returns_df[portfolio2_name].iloc[-1],
    }
    
    if portfolio3_name in cumulative_returns_df.columns:
        final_values[portfolio3_name] = cumulative_returns_df[portfolio3_name].iloc[-1]
    
    if portfolio4_name in cumulative_returns_df.columns:
        final_values[portfolio4_name] = cumulative_returns_df[portfolio4_name].iloc[-1]
    
    
    sorted_portfolios = sorted(final_values.items(), key=lambda x: x[1])
    
    
    vertical_spacing = cumulative_return_vertical_spacing  
    
    for i, (name, value) in enumerate(sorted_portfolios):
        
        horizontal_shift = cumulative_return_horizontal_shift
        
        
        if i == 0:  
            vertical_shift = 0
        else:
            
            prev_value = sorted_portfolios[i-1][1]
            diff = value - prev_value
            
            
            if diff < 0.2:  
                vertical_shift = -vertical_spacing * (len(sorted_portfolios) - i)
            else:
                vertical_shift = 0
        
        ax1.annotate(f'{value:.2f}x', 
                    xy=(cumulative_returns_df.index[-1], value),
                    xytext=(horizontal_shift, vertical_shift),  
                    textcoords='offset points',
                    ha='left', va='center', 
                    fontweight='bold', 
                    color=colors[name])

    
    ax1.annotate('Cumulative Return Growth of 1 USD', 
                xy=(0.95, 0.02),  
                xycoords='axes fraction',  # Coordonnées relatives à l'axe
                ha='right', va='bottom',  # Alignement
                fontsize=10, color='#333333',  # Gris discret
                style='italic')  # Style italique pour plus de discrétion
                
    ax1.set_ylabel('Cumulative Return')
    
    
    ax1.tick_params(axis='x', labelbottom=False)
    
    
    ax1.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=10)
    apply_custom_style(ax1)

    
    ax2 = fig.add_subplot(gs[1])

    
    ax2.fill_between(portfolio1_drawdown.index, 0, -portfolio1_drawdown.values, 
                    alpha=0.3, color=colors[portfolio1_name], 
                    label=portfolio1_name, step=None)

    ax2.fill_between(portfolio2_drawdown.index, 0, -portfolio2_drawdown.values, 
                    alpha=0.3, color=colors[portfolio2_name], 
                    label=portfolio2_name, step=None)
    
    
    if portfolio3_drawdown is not None:
        ax2.fill_between(portfolio3_drawdown.index, 0, -portfolio3_drawdown.values, 
                        alpha=0.3, color=colors[portfolio3_name], 
                        label=portfolio3_name, step=None)
    
    
    if portfolio4_drawdown is not None:
        ax2.fill_between(portfolio4_drawdown.index, 0, -portfolio4_drawdown.values, 
                        alpha=0.3, color=colors[portfolio4_name], 
                        label=portfolio4_name, step=None)

    ax2.plot(portfolio1_drawdown.index, -portfolio1_drawdown.values, alpha=0.7, color=colors[portfolio1_name], linewidth=1)
    ax2.plot(portfolio2_drawdown.index, -portfolio2_drawdown.values, alpha=0.7, color=colors[portfolio2_name], linewidth=1)
    
    
    if portfolio3_drawdown is not None:
        ax2.plot(portfolio3_drawdown.index, -portfolio3_drawdown.values, alpha=0.7, color=colors[portfolio3_name], linewidth=1)
    
    if portfolio4_drawdown is not None:
        ax2.plot(portfolio4_drawdown.index, -portfolio4_drawdown.values, alpha=0.7, color=colors[portfolio4_name], linewidth=1)

    
    portfolio1_valley_date = portfolio1_drawdown.idxmax()
    portfolio2_valley_date = portfolio2_drawdown.idxmax()
    portfolio3_valley_date = None
    if portfolio3_drawdown is not None:
        portfolio3_valley_date = portfolio3_drawdown.idxmax()
    portfolio4_valley_date = None
    if portfolio4_drawdown is not None:
        portfolio4_valley_date = portfolio4_drawdown.idxmax()

    
    
    drawdowns = [
        (portfolio1_name, portfolio1_max_dd, portfolio1_valley_date, colors[portfolio1_name]),
        (portfolio2_name, portfolio2_max_dd, portfolio2_valley_date, colors[portfolio2_name])
    ]
    
    if portfolio3_drawdown is not None:
        drawdowns.append((portfolio3_name, portfolio3_max_dd, portfolio3_valley_date, colors[portfolio3_name]))
    
    if portfolio4_drawdown is not None:
        drawdowns.append((portfolio4_name, portfolio4_max_dd, portfolio4_valley_date, colors[portfolio4_name]))
    
    
    drawdowns.sort(key=lambda x: x[2])
    
    
    max_dd_value = max([dd[1] for dd in drawdowns])
    
    
    for i, (name, max_dd, valley_date, color) in enumerate(drawdowns):
        
        
        horizontal_shift = drawdown_horizontal_shift * (-1 if i % 2 == 0 else 1)
        
        
        arc_rad = drawdown_arc_radius * (-1 if i % 2 == 0 else 1)
        
        ax2.annotate(f'Max DD: -{max_dd:.2%}',
                    xy=(valley_date, -max_dd),  
                    xytext=(horizontal_shift, 0),  
                    textcoords='offset points',
                    arrowprops=dict(
                        arrowstyle='->',
                        color=color,
                        connectionstyle=f'arc3,rad={arc_rad}',  # Arc contrôlé par paramètre
                        shrinkA=0,
                        shrinkB=5
                    ),
                    ha='right' if i % 2 == 0 else 'left',  # Alignement alterné
                    va='center',
                    fontsize=10, 
                    fontweight='bold', 
                    color=color,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7, edgecolor='none'))

    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.xaxis.set_major_locator(mdates.YearLocator(1))
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    
    
    
    all_max_dd = max([dd[1] for dd in drawdowns])
    ax2.set_ylim(-all_max_dd * 1.3, all_max_dd * 0.4)  
    
    
    ax2.legend(loc='lower left', frameon=True, framealpha=0.9, fontsize=8)
    apply_custom_style(ax2, show_grid=False)  

    
    plt.tight_layout()
    
    
    fig.suptitle(title, fontsize=18, fontweight='bold', x=0.5, y=0.98, ha='center')
    
    
    plt.subplots_adjust(top=0.92, hspace=0.0)
    
    if save_path:
        plt.savefig(save_path, dpi=50, bbox_inches='tight')
        print(f"Dashboard saved to {save_path}")
    
    return fig, [ax1, ax2]






fig5, axes5 = create_portfolio_comparison_dashboard(
    portfolio_returns['Static_λ2'],
    portfolio_returns['Static_λ10'],
    portfolio_returns['Dynamic_Constrained_λ2'],
    portfolio_returns['Dynamic_Constrained_λ10'],
    rf_series=risk_free_rate,
    portfolio1_name="Static (λ=2)",
    portfolio2_name="Static (λ=10)",
    portfolio3_name="Dynamic Constrained (λ=2)",
    portfolio4_name="Dynamic Constrained (λ=10)",
    colors={
        "Static (λ=2)": '#1f77b4',    # Bleu
        "Static (λ=10)": '#ff7f0e',    # Orange
        "Dynamic Constrained (λ=2)": '#2ca02c',    # Vert
        "Dynamic Constrained (λ=10)": '#d62728'    # Rouge
    },
    figsize=(22, 8),
    title="Performance Comparison: Static vs. Dynamic Constrained Portfolio Strategies (λ=2 and λ=10)",
    cumulative_return_horizontal_shift=10,
    cumulative_return_vertical_spacing=-5,
    drawdown_horizontal_shift=40,
    drawdown_arc_radius=0.3
)

plt.show()





print('# ---------------------- Q 3.3 TC  ---------------------------------------- #')










dynamic_weights_lambda2 = dynamic_allocation_with_constraints[['w_stock_lambda2_constrained', 'w_bond_lambda2_constrained']]
dynamic_weights_lambda10 = dynamic_allocation_with_constraints[['w_stock_lambda10_constrained', 'w_bond_lambda10_constrained']]

dynamic_returns_lambda2 = portfolio_returns['Dynamic_Constrained_λ2']
static_returns_lambda2 = portfolio_returns['Static_λ2']
dynamic_returns_lambda10 = portfolio_returns['Dynamic_Constrained_λ10']
static_returns_lambda10 = portfolio_returns['Static_λ10']


turnover_lambda2 = dynamic_weights_lambda2.diff().abs().sum(axis=1).fillna(0)

turnover_lambda10 = dynamic_weights_lambda10.diff().abs().sum(axis=1).fillna(0)



def net_cumulative_return(rp_dynamic, turnover, rp_static):
    def cumulative_net_return(f):
        """Calcule la série de rendements cumulés nets après coûts de transaction"""
        rp_net = rp_dynamic - turnover * f
        log_rp_net = np.log1p(rp_net)
        return np.exp(log_rp_net.cumsum())  

    def loss(f):
        """Fonction objectif: minimise la différence au carré entre les rendements finaux"""
        rp_net = rp_dynamic - turnover * f
        cum_net = np.exp(np.log1p(rp_net).cumsum()).iloc[-1]  
        cum_static = np.exp(np.log1p(rp_static).cumsum()).iloc[-1]  
        return (cum_net - cum_static) ** 2  

    return cumulative_net_return, loss


cum_return_lambda2, loss_lambda2 = net_cumulative_return(dynamic_returns_lambda2, turnover_lambda2, static_returns_lambda2)
cum_return_lambda10, loss_lambda10 = net_cumulative_return(dynamic_returns_lambda10, turnover_lambda10, static_returns_lambda10)


f_values = np.linspace(0, 0.05, 100)
results_df_lambda2 = pd.DataFrame({
    'f': f_values,
    'Dynamic_Cumulative_Return': [cum_return_lambda2(f).iloc[-1] for f in f_values],  # Utiliser .iloc[-1]
    'Static_Cumulative_Return': np.exp(np.log1p(static_returns_lambda2).sum())
})

results_df_lambda10 = pd.DataFrame({
    'f': f_values,
    'Dynamic_Cumulative_Return': [cum_return_lambda10(f).iloc[-1] for f in f_values],  # Utiliser .iloc[-1]
    'Static_Cumulative_Return': np.exp(np.log1p(static_returns_lambda10).sum())
})




def audit_critical_f(rp_dynamic, rp_static, weights_dynamic, f_value):
    """
    Vérifie que la valeur critique de f trouvée est correcte
    en comparant les rendements cumulés finaux
    """
    turnover = weights_dynamic.diff().abs().sum(axis=1).fillna(0)
    rp_net = rp_dynamic - turnover * f_value
    
    
    cum_net = np.exp(np.log1p(rp_net).cumsum())
    cum_static = np.exp(np.log1p(rp_static).cumsum())
    
    
    final_net = cum_net.iloc[-1]
    final_static = cum_static.iloc[-1]
    
    print(f"Rendement cumulé final (Dynamic avec coûts f={f_value*100:.4f}%): {final_net:.6f}")
    print(f"Rendement cumulé final (Static): {final_static:.6f}")
    print(f"Différence absolue: {abs(final_net - final_static):.8f}")
    print(f"Différence relative: {abs((final_net - final_static)/final_static)*100:.8f}%")
    
    
    rel_diff = abs((final_net - final_static)/final_static)
    is_valid = rel_diff < 1e-5  
    print(f"La valeur f={f_value*100:.4f}% est {'valide' if is_valid else 'non valide'}")
    
    return is_valid














def analyze_transaction_costs(dynamic_returns, static_returns, dynamic_weights, name=""):
    """
    Analyse complète des coûts de transaction pour comparer stratégies dynamique et statique
    suivant la formulation académique précise: 
    ∑log(1+R^net_p,t(f*)) = ∑log(1+R^static_p,t)
    
    Parameters:
    -----------
    dynamic_returns : Series - Rendements du portefeuille dynamique
    static_returns : Series - Rendements du portefeuille statique
    dynamic_weights : DataFrame - Poids du portefeuille dynamique (pour calculer turnover)
    name : str - Nom pour l'identification (ex: "λ=2")
    
    Returns:
    --------
    dict - Résultats complets de l'analyse
    """
    print(f"\n=== Analyse des coûts de transaction pour {name} ===")
    
    
    turnover = dynamic_weights.diff().abs().sum(axis=1).fillna(0)
    
    
    def calc_net_return(f):
        """Calcule les rendements nets après coûts de transaction"""
        net_returns = dynamic_returns - turnover * f
        return net_returns
    
    
    def objective_function(f):
        """
        Calcule la différence entre la SOMME des log-rendements (pas les rendements cumulés finaux)
        Formulation académique exacte: ∑log(1+R^net_p,t(f*)) = ∑log(1+R^static_p,t)
        """
        net_returns = calc_net_return(f)
        
        
        sum_log_net = np.sum(np.log1p(net_returns))
        sum_log_static = np.sum(np.log1p(static_returns))
        
        
        return (sum_log_net - sum_log_static) ** 2
    
    
    result = minimize_scalar(
        objective_function, 
        bounds=(0, 0.01),  
        method='bounded',
        options={'xatol': 1e-10}
    )
    
    
    critical_f = result.x
    
    
    net_returns = calc_net_return(critical_f)
    
    
    sum_log_net = np.sum(np.log1p(net_returns))
    sum_log_static = np.sum(np.log1p(static_returns))
    
    
    cum_net = np.exp(np.log1p(net_returns).cumsum())
    cum_static = np.exp(np.log1p(static_returns).cumsum())
    
    
    def generate_plot_data(f_values):
        """Génère un DataFrame avec les rendements cumulatifs pour différentes valeurs de f"""
        data = []
        for f in f_values:
            net_ret = calc_net_return(f)
            cum_ret = np.exp(np.log1p(net_ret).cumsum()).iloc[-1]
            data.append(cum_ret)
        
        return pd.DataFrame({
            'f': f_values,
            'Dynamic_Cumulative_Return': data,
            'Static_Cumulative_Return': cum_static.iloc[-1]
        })
    
    
    abs_diff_logs = abs(sum_log_net - sum_log_static)
    
    print(f"Valeur critique de f: {critical_f:.8f} ({critical_f*100:.6f}%)")
    print(f"L'algorithme a convergé: {'Oui' if result.success else 'Non'}")
    print(f"Somme des log-rendements (Dynamic avec coûts): {sum_log_net:.6f}")
    print(f"Somme des log-rendements (Static): {sum_log_static:.6f}")
    print(f"Différence absolue (logs): {abs_diff_logs:.8f}")
    
    
    f_values = np.linspace(0, 0.005, 100)  
    plot_data = generate_plot_data(f_values)
    
    return {
        'critical_f': critical_f,
        'cum_net': cum_net,
        'cum_static': cum_static,
        'turnover': turnover,
        'plot_data': plot_data
    }




results_lambda2 = analyze_transaction_costs(
    dynamic_returns_lambda2,
    static_returns_lambda2,
    dynamic_weights_lambda2,
    name="λ=2"
)


results_lambda10 = analyze_transaction_costs(
    dynamic_returns_lambda10,
    static_returns_lambda10,
    dynamic_weights_lambda10,
    name="λ=10"
)




critical_f_lambda2 = results_lambda2['critical_f'] 
critical_f_lambda10 = results_lambda10['critical_f']





def plot_comparative_transaction_costs(results_lambda2, results_lambda10, separate=False):
    """
    Crée un ou deux graphiques comparatifs des coûts de transaction pour λ=2 et λ=10
    avec une apparence professionnelle et des annotations claires.
    
    Parameters:
    -----------
    results_lambda2 : dict - Résultats d'analyse pour λ=2
    results_lambda10 : dict - Résultats d'analyse pour λ=10
    separate : bool - Si True, crée deux figures séparées au lieu d'une figure avec deux sous-graphiques
    
    Returns:
    --------
    list - Liste de figures matplotlib
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['figure.figsize'] = (22, 8) if separate else (22, 8)
    
    
    colors = {
        'dynamic': '#2ca02c',      # vert
        'static': '#1f77b4',       # bleu
        'critical': '#d62728',     # rouge
        'annotation': '#7f7f7f',   # gris
        'background': '#f5f5f5'    # gris très clair pour les zones
    }
    
    figures = []
    
    
    def create_plot(ax, results, lambda_value):
        df = results['plot_data']
        critical_f = results['critical_f']
        static_return = df['Static_Cumulative_Return'].iloc[0]
        
        
        lower_bound = max(0, critical_f - 0.0005)
        upper_bound = min(df['f'].max(), critical_f + 0.0005)
        mask = (df['f'] >= lower_bound) & (df['f'] <= upper_bound)
        ax.fill_between(df.loc[mask, 'f'] * 100, 0, df.loc[mask, 'Dynamic_Cumulative_Return'], 
                        color=colors['background'], alpha=0.3)
        
        
        ax.plot(df['f'] * 100, df['Dynamic_Cumulative_Return'], 
                color=colors['dynamic'], linewidth=3, label='Dynamic Portfolio')
        
        
        ax.axhline(y=static_return, color=colors['static'], linestyle='--', linewidth=2.5, 
                   label='Static Portfolio')
        
        
        ax.axvline(x=critical_f * 100, color=colors['critical'], linestyle='-', linewidth=1.5, 
                   label=f'Critical f = {critical_f*100:.4f}%')
        
        
        ax.annotate(f'Critical f = {critical_f*100:.4f}%',
                    xy=(critical_f * 100, static_return),
                    xytext=(critical_f * 100 + 0.03, static_return * 1.1),  
                    arrowprops=dict(arrowstyle='->',
                                   color=colors['critical'],
                                   connectionstyle='arc3,rad=.2'),  # Flèche courbée
                    fontsize=12,
                    fontweight='bold',
                    color=colors['critical'])
        
        
        
        
        
        
        
        
        
        
        
        
        ax.set_title(f'Impact of Transaction Costs (λ={lambda_value})', 
                     fontsize=16, fontweight='bold')
        ax.set_xlabel('Transaction Cost f (%)', fontsize=14)
        ax.set_ylabel('Cumulative Return', fontsize=14)
        ax.grid(True, linestyle=':', alpha=0.7)
        
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=12)
        
        
        legend = ax.legend(loc='upper right', fontsize=10, framealpha=0.95, 
                          edgecolor=colors['annotation'], shadow=True)
        legend.get_frame().set_facecolor('white')
        
    
    if not separate:
        fig, axes = plt.subplots(2, 1, figsize=(22, 8), sharex=True, dpi=50)
        
        create_plot(axes[0], results_lambda2, 2)
        create_plot(axes[1], results_lambda10, 10)
        
        
        fig.suptitle('Transaction Costs Analysis - Static vs Dynamic Allocation', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        
        axes[0].set_xlabel('')
        
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, hspace=0.25)
        figures.append(fig)
    
    
    else:
        
        fig1 = plt.figure(figsize=(22, 8), dpi=50)
        ax1 = fig1.add_subplot(111)
        create_plot(ax1, results_lambda2, 2)
        ax1.set_title('Transaction Costs Analysis - Static vs Dynamic Allocation (λ=2)', 
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        figures.append(fig1)
        plt.show()
        
        
        fig2 = plt.figure(figsize=(22, 8), dpi=50)
        ax2 = fig2.add_subplot(111)
        create_plot(ax2, results_lambda10, 6)
        ax2.set_title('Transaction Costs Analysis - Static vs Dynamic Allocation (λ=10)', 
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        figures.append(fig2)
    
    
    plt.show()
    
    return figures


figs = plot_comparative_transaction_costs(results_lambda2, results_lambda10)








def calculate_transaction_costs(weights_df, f):
    """
    Calcule les coûts de transaction comme le produit du turnover et du facteur de coût f
    
    Parameters:
    -----------
    weights_df : DataFrame
        DataFrame contenant les poids du portefeuille
    f : float
        Facteur de coût de transaction
        
    Returns:
    --------
    Series
        Série temporelle des coûts de transaction
    """
    
    turnover = weights_df.diff().abs().sum(axis=1).fillna(0)
    
    
    transaction_costs = turnover * f
    
    return transaction_costs





def calculate_returns_with_costs(dynamic_returns, weights_df, f):
    """Calcule les rendements du portefeuille dynamique après coûts de transaction"""
    transaction_costs = calculate_transaction_costs(weights_df, f)
    return dynamic_returns - transaction_costs

def plot_comparison_with_critical_costs(static_returns, dynamic_returns, weights_df, critical_f, lambda_value):
    """Visualise la comparaison entre portefeuille statique et dynamique avec coûts critiques"""
    
    
    adjusted_returns = calculate_returns_with_costs(dynamic_returns, weights_df, critical_f)
    
    
    cumul_static = (1 + static_returns).cumprod()
    cumul_dynamic_no_costs = (1 + dynamic_returns).cumprod()
    cumul_dynamic_with_costs = (1 + adjusted_returns).cumprod()
    
    
    fig, axes = create_portfolio_comparison_dashboard(
        static_returns,
        dynamic_returns,
        adjusted_returns,
        rf_series=None,
        portfolio1_name=f"Static (λ={lambda_value})",
        portfolio2_name=f"Dynamic without costs (λ={lambda_value})",
        portfolio3_name=f"Dynamic with costs f={critical_f*100:.2f}% (λ={lambda_value})",
        colors={
            f"Static (λ={lambda_value})": '#1f77b4',  # Bleu
            f"Dynamic without costs (λ={lambda_value})": '#2ca02c',  # Vert
            f"Dynamic with costs f={critical_f*100:.2f}% (λ={lambda_value})": '#ff7f0e'  # Orange
        },
        figsize=(22, 8),
        title=f"Performance Comparison with Critical Transaction Cost (λ={lambda_value}, f={critical_f*100:.2f}%)",
        cumulative_return_horizontal_shift=10,
        cumulative_return_vertical_spacing=6,
        drawdown_horizontal_shift=40,
        drawdown_arc_radius=0.3
    )
    
    return fig, axes


fig, axes = plot_comparison_with_critical_costs(
    static_returns_lambda2, 
    dynamic_returns_lambda2, 
    dynamic_weights_lambda2, 
    critical_f_lambda2, 
    2
)

fig2, axes2 = plot_comparison_with_critical_costs(
    static_returns_lambda10, 
    dynamic_returns_lambda10, 
    dynamic_weights_lambda10, 
    critical_f_lambda10, 
    10
)






adjusted_returns_lambda2 = calculate_returns_with_costs(dynamic_returns_lambda2, dynamic_weights_lambda2, critical_f_lambda2)
adjusted_returns_lambda10 = calculate_returns_with_costs(dynamic_returns_lambda10, dynamic_weights_lambda10, critical_f_lambda10)


comparison_returns = pd.DataFrame({
    'Static_λ2': static_returns_lambda2,
    'Dynamic_λ2_no_costs': dynamic_returns_lambda2,
    'Dynamic_λ2_with_costs': adjusted_returns_lambda2,
    'Static_λ10': static_returns_lambda10,
    'Dynamic_λ10_no_costs': dynamic_returns_lambda10,
    'Dynamic_λ10_with_costs': adjusted_returns_lambda10
})


performance_metrics_with_costs = calculate_performance_metrics(comparison_returns, df_weekly['rf'])


print("\nMétriques de performance avec et sans coûts de transaction:")
print(performance_metrics_with_costs.round(2))








print('### Diagnostic of the number of observations in each series ###')

print(f"Static λ2: {len(static_returns_lambda2)}")
print(f"Dynamic λ2 (without costs): {len(dynamic_returns_lambda2)}")
print(f"Dynamic λ2 (with costs): {len(adjusted_returns_lambda2)}")
print(f"Static λ10: {len(static_returns_lambda10)}")
print(f"Dynamic λ10 (without costs): {len(dynamic_returns_lambda10)}")
print(f"Dynamic λ10 (with costs): {len(adjusted_returns_lambda10)}")


common_index = static_returns_lambda2.index
for series in [dynamic_returns_lambda2, adjusted_returns_lambda2, 
               static_returns_lambda10, dynamic_returns_lambda10, adjusted_returns_lambda10]:
    common_index = common_index.intersection(series.index)

print(f"Number of common indices: {len(common_index)}")





print('# ---------------------- Section 4  ---------------------------------------- #')





print('# ---------------------- Q 4.1  ---------------------------------------- #')






def prepare_daily_data(df, col_name):
    df_clean = df.iloc[1:, :].copy()
    
    if col_name == 'rf':
        df_clean.columns = ['Date', col_name]
    else:
        df_clean.columns = ['Date', col_name, '-']
    
    df_clean['Date'] = pd.to_datetime(df_clean['Date'])
    df_clean.set_index('Date', inplace=True)
    
    if col_name != 'rf':
        df_clean = df_clean[[col_name]].astype(float)
    else:
        df_clean = df_clean[[col_name]].astype(float)
        
        df_clean[col_name] = df_clean[col_name] / 100 / 252  
    
    df_clean.index.name = None
    return df_clean




ftse_100_daily = prepare_daily_data(ftse_100_daily, 'stock')
bond_data_daily = prepare_daily_data(bond_data_daily, 'bond')
riskfree_data_daily = prepare_daily_data(riskfree_data_daily, 'rf')


ftse_100_daily['stock_returns'] = ftse_100_daily['stock'].pct_change()
bond_data_daily['bond_returns'] = bond_data_daily['bond'].pct_change()


df_daily = pd.merge(
    ftse_100_daily[['stock_returns']],
    bond_data_daily[['bond_returns']],
    left_index=True, right_index=True,
    how='outer'
)
df_daily = pd.merge(
    df_daily,
    riskfree_data_daily[['rf']],
    left_index=True, right_index=True,
    how='outer'
)


df_daily.rename(columns={
    'stock_returns': 'stock',
    'bond_returns': 'bond'
}, inplace=True)


start_date = '2001-01-01'
end_date = '2024-12-31'  # Ajusté à 2023 pour cohérence
df_daily = df_daily[(df_daily.index >= start_date) & (df_daily.index <= end_date)]
df_daily = df_daily.dropna()

print(f"Nombre d'observations journalières: {len(df_daily)}")


def calculate_daily_portfolio_losses(daily_returns, weights, rf_daily):
    """
    Calcule les pertes journalières du portefeuille
    
    Parameters:
    -----------
    daily_returns : DataFrame
        DataFrame contenant les rendements journaliers des actifs
    weights : array ou DataFrame
        Poids des actifs (statiques ou dynamiques)
    rf_daily : Series
        Taux sans risque journalier
        
    Returns:
    --------
    Series
        Série temporelle des pertes journalières du portefeuille
    """
    
    if isinstance(weights, np.ndarray):
        portfolio_returns = (
            weights[0] * daily_returns['stock'] + 
            weights[1] * daily_returns['bond'] + 
            (1 - weights[0] - weights[1]) * rf_daily
        )
        losses = -portfolio_returns
        return losses
    
    
    
    daily_weights = pd.DataFrame(index=daily_returns.index, columns=['w_stock', 'w_bond'])
    
    
    for week_start, row in weights.iterrows():
        
        week_end = week_start + pd.Timedelta(days=4)
        
        
        mask = (daily_returns.index >= week_start) & (daily_returns.index <= week_end)
        
        
        if 'w_stock_lambda2_constrained' in weights.columns:
            w_stock = row['w_stock_lambda2_constrained']
            w_bond = row['w_bond_lambda2_constrained']
        elif 'w_stock_lambda10_constrained' in weights.columns:
            w_stock = row['w_stock_lambda10_constrained']
            w_bond = row['w_bond_lambda10_constrained']
        else:
            raise ValueError("Colonnes de poids non trouvées")
        
        
        daily_weights.loc[mask, 'w_stock'] = w_stock
        daily_weights.loc[mask, 'w_bond'] = w_bond
    
    
    common_dates = daily_returns.index.intersection(daily_weights.dropna().index)
    portfolio_returns = pd.Series(index=common_dates, dtype=float)
    
    for date in common_dates:
        w_stock = daily_weights.loc[date, 'w_stock']
        w_bond = daily_weights.loc[date, 'w_bond']
        w_rf = 1 - w_stock - w_bond
        
        portfolio_returns[date] = (
            w_stock * daily_returns.loc[date, 'stock'] +
            w_bond * daily_returns.loc[date, 'bond'] +
            w_rf * rf_daily.loc[date]
        )
    
    
    losses = -portfolio_returns
    return losses


static_weights_lambda2 = np.array([alpha_2[0, 0], alpha_2[1, 0]])
static_weights_lambda10 = np.array([alpha_10[0, 0], alpha_10[1, 0]])

static_losses_lambda2 = calculate_daily_portfolio_losses(df_daily, static_weights_lambda2, df_daily['rf'])
static_losses_lambda10 = calculate_daily_portfolio_losses(df_daily, static_weights_lambda10, df_daily['rf'])


dynamic_weights_lambda2 = dynamic_allocation_with_constraints[['w_stock_lambda2_constrained', 'w_bond_lambda2_constrained']]
dynamic_weights_lambda10 = dynamic_allocation_with_constraints[['w_stock_lambda10_constrained', 'w_bond_lambda10_constrained']]

dynamic_losses_lambda2 = calculate_daily_portfolio_losses(df_daily, dynamic_weights_lambda2, df_daily['rf'])
dynamic_losses_lambda10 = calculate_daily_portfolio_losses(df_daily, dynamic_weights_lambda10, df_daily['rf'])


def calculate_unconditional_var(losses, confidence_level=0.99):
    """
    Calcule la VaR inconditionnelle pour une distribution normale
    """
    
    mean_loss = losses.mean()
    var_loss = losses.var()
    std_loss = np.sqrt(var_loss)
    
    
    z_score = stats.norm.ppf(confidence_level)
    var_unconditional = mean_loss + std_loss * z_score
    
    return {
        'mean_loss': mean_loss,
        'variance_loss': var_loss,
        'std_loss': std_loss,
        'var_unconditional': var_unconditional
    }


var_static_lambda2 = calculate_unconditional_var(static_losses_lambda2)
var_static_lambda10 = calculate_unconditional_var(static_losses_lambda10)
var_dynamic_lambda2 = calculate_unconditional_var(dynamic_losses_lambda2)
var_dynamic_lambda10 = calculate_unconditional_var(dynamic_losses_lambda10)


print("\n === Unconditional VaR Analysis (99% confidence level) === ")
print("\nStatic Portfolio (λ=2):")
print(f"Unconditional Mean Loss (μ): {var_static_lambda2['mean_loss']:.6f}")
print(f"Unconditional Variance (σ²): {var_static_lambda2['variance_loss']:.6f}")
print(f"Unconditional Standard Deviation (σ): {var_static_lambda2['std_loss']:.6f}")
print(f"Unconditional VaR (99%): {var_static_lambda2['var_unconditional']:.6f}")

print("\nStatic Portfolio (λ=10):")
print(f"Unconditional Mean Loss (μ): {var_static_lambda10['mean_loss']:.6f}")
print(f"Unconditional Variance (σ²): {var_static_lambda10['variance_loss']:.6f}")
print(f"Unconditional Standard Deviation (σ): {var_static_lambda10['std_loss']:.6f}")
print(f"Unconditional VaR (99%): {var_static_lambda10['var_unconditional']:.6f}")

print("\nDynamic Portfolio (λ=2):")
print(f"Unconditional Mean Loss (μ): {var_dynamic_lambda2['mean_loss']:.6f}")
print(f"Unconditional Variance (σ²): {var_dynamic_lambda2['variance_loss']:.6f}")
print(f"Unconditional Standard Deviation (σ): {var_dynamic_lambda2['std_loss']:.6f}")
print(f"Unconditional VaR (99%): {var_dynamic_lambda2['var_unconditional']:.6f}")

print("\nDynamic Portfolio (λ=10):")
print(f"Unconditional Mean Loss (μ): {var_dynamic_lambda10['mean_loss']:.6f}")
print(f"Unconditional Variance (σ²): {var_dynamic_lambda10['variance_loss']:.6f}")
print(f"Unconditional Standard Deviation (σ): {var_dynamic_lambda10['std_loss']:.6f}")
print(f"Unconditional VaR (99%): {var_dynamic_lambda10['var_unconditional']:.6f}")


var_table = pd.DataFrame({
    'Portfolio': ['Static (λ=2)', 'Static (λ=10)', 'Dynamic (λ=2)', 'Dynamic (λ=10)'],
    'Mean Loss (μ)': [
        var_static_lambda2['mean_loss'],
        var_static_lambda10['mean_loss'],
        var_dynamic_lambda2['mean_loss'],
        var_dynamic_lambda10['mean_loss']
    ],
    'Variance (σ²)': [
        var_static_lambda2['variance_loss'],
        var_static_lambda10['variance_loss'],
        var_dynamic_lambda2['variance_loss'],
        var_dynamic_lambda10['variance_loss']
    ],
    'Standard Deviation (σ)': [
        var_static_lambda2['std_loss'],
        var_static_lambda10['std_loss'],
        var_dynamic_lambda2['std_loss'],
        var_dynamic_lambda10['std_loss']
    ],
    'VaR (99%)': [
        var_static_lambda2['var_unconditional'],
        var_static_lambda10['var_unconditional'],
        var_dynamic_lambda2['var_unconditional'],
        var_dynamic_lambda10['var_unconditional']
    ]
})

var_table = var_table.set_index('Portfolio')
print("\nComparaison de la VaR inconditionnelle à 99%:\n")
print(var_table)






var_table_formatted = var_table.copy()


var_table_formatted = var_table_formatted * 100


var_table_formatted.columns = [
    'Mean Loss ($\\mu$) \\%',
    'Variance ($\\sigma^2$) \\%',
    'Standard Deviation ($\\sigma$) \\%',
    'VaR (99\\%) \\%'
]


var_table_formatted.index = [
    'Static ($\\lambda=2$)',
    'Static ($\\lambda=10$)',
    'Dynamic ($\\lambda=2$)',
    'Dynamic ($\\lambda=10$)'
]


print(var_table_formatted.round(4))








print("=== DIAGNOSTIC OF VAR TABLE OBSERVATIONS ===")
print(f"Total number of rows in var_table: {len(var_table)}")


for portfolio in var_table.index:
    print(f"Portfolio: {portfolio}")
    
    
    if 'Total Observations' in var_table.columns:
        print(f"  Observations: {var_table.loc[portfolio, 'Total Observations']}")
    elif hasattr(var_table, 'Total_Observations'):  # Changé ici
        print(f"  Observations: {var_table.Total_Observations[portfolio]}")
    else:
        
        if portfolio == "Static (λ=2)":
            print(f"  Observations: {len(static_losses_lambda2)}")
        elif portfolio == "Static (λ=10)":
            print(f"  Observations: {len(static_losses_lambda10)}")
        elif portfolio == "Dynamic (λ=2)":
            print(f"  Observations: {len(dynamic_losses_lambda2)}")
        elif portfolio == "Dynamic (λ=10)":
            print(f"  Observations: {len(dynamic_losses_lambda10)}")
            
print("\n=== TOTAL AVAILABLE DAILY DATA ===")
print(f"Daily data points: {len(df_daily)}")
print(f"Period covered: {df_daily.index.min()} to {df_daily.index.max()}")






plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.dpi'] = 50


fig = plt.figure(figsize=(18, 8))
gs = GridSpec(1, 2, figure=fig)


colors = {
    'Static (λ=2)': '#1f77b4',    # bleu
    'Static (λ=10)': '#ff7f0e',   # orange
    'Dynamic (λ=2)': '#2ca02c',   # vert
    'Dynamic (λ=10)': '#d62728',  # rouge
}


ax1 = fig.add_subplot(gs[0, 0])
sns.histplot(static_losses_lambda2, kde=True, ax=ax1, color=colors['Static (λ=2)'], 
             stat='density', alpha=0.5, label='Static (λ=2)')
sns.histplot(static_losses_lambda10, kde=True, ax=ax1, color=colors['Static (λ=10)'], 
             stat='density', alpha=0.5, label='Static (λ=10)')


ax1.axvline(var_static_lambda2['var_unconditional'], color=colors['Static (λ=2)'], 
            linestyle='--', linewidth=2)
ax1.axvline(var_static_lambda10['var_unconditional'], color=colors['Static (λ=10)'], 
            linestyle='--', linewidth=2)

ax1.set_title('Panel A: Loss Distribution and VaR (99%) - Static Strategies', fontweight='bold')
ax1.set_xlabel('Daily Losses')
ax1.set_ylabel('Density')
ax1.grid(False)
ax1.legend()


y_max = ax1.get_ylim()[1]


ax1.annotate(f'VaR: {var_static_lambda2["var_unconditional"]*100:.3f}%', 
             xy=(var_static_lambda2['var_unconditional'], y_max*0.8),
             xytext=(var_static_lambda2['var_unconditional']*1.2, y_max*0.8), # Déplacé plus loin à droite
             arrowprops=dict(arrowstyle='->', color=colors['Static (λ=2)'], 
                            connectionstyle='arc3,rad=0.3'), # Arc plus prononcé
             color=colors['Static (λ=2)'], fontsize=10, fontweight='bold')


ax1.annotate(f'VaR: {var_static_lambda10["var_unconditional"]*100:.3f}%', 
             xy=(var_static_lambda10['var_unconditional'], y_max*0.7),
             xytext=(var_static_lambda10['var_unconditional']*3.1, y_max*0.7), # Déplacé beaucoup plus à gauche
             arrowprops=dict(arrowstyle='->', color=colors['Static (λ=10)'], 
                            connectionstyle='arc3,rad=0.3'), # Arc plus prononcé
             color=colors['Static (λ=10)'], fontsize=10, fontweight='bold',
             ha='right') # Alignement du texte à droite


ax2 = fig.add_subplot(gs[0, 1])
sns.histplot(dynamic_losses_lambda2, kde=True, ax=ax2, color=colors['Dynamic (λ=2)'], 
             stat='density', alpha=0.5, label='Dynamic (λ=2)')
sns.histplot(dynamic_losses_lambda10, kde=True, ax=ax2, color=colors['Dynamic (λ=10)'], 
             stat='density', alpha=0.5, label='Dynamic (λ=10)')


ax2.axvline(var_dynamic_lambda2['var_unconditional'], color=colors['Dynamic (λ=2)'], 
            linestyle='--', linewidth=2)
ax2.axvline(var_dynamic_lambda10['var_unconditional'], color=colors['Dynamic (λ=10)'], 
            linestyle='--', linewidth=2)

ax2.set_title('Panel B: Loss Distribution and VaR (99%) - Dynamic Strategies', fontweight='bold')
ax2.set_xlabel('Daily Losses')
ax2.set_ylabel('Density')
ax2.legend()


y_max = ax2.get_ylim()[1]


ax2.annotate(f'VaR: {var_dynamic_lambda2["var_unconditional"]*100:.3f}%', 
             xy=(var_dynamic_lambda2['var_unconditional'], y_max*0.8),
             xytext=(var_dynamic_lambda2['var_unconditional']*1.2, y_max*0.8), # Déplacé plus loin à droite
             arrowprops=dict(arrowstyle='->', color=colors['Dynamic (λ=2)'], 
                            connectionstyle='arc3,rad=0.3'), # Arc plus prononcé
             color=colors['Dynamic (λ=2)'], fontsize=10, fontweight='bold')


ax2.annotate(f'VaR: {var_dynamic_lambda10["var_unconditional"]*100:.3f}%', 
             xy=(var_dynamic_lambda10['var_unconditional'], y_max*0.7),
             xytext=(var_dynamic_lambda10['var_unconditional']*2.1, y_max*0.7), # Déplacé beaucoup plus à gauche
             arrowprops=dict(arrowstyle='->', color=colors['Dynamic (λ=10)'], 
                            connectionstyle='arc3,rad=0.3'), # Arc plus prononcé
             color=colors['Dynamic (λ=10)'], fontsize=10, fontweight='bold',
             ha='right') # Alignement du texte à droite




x_min = min(var_static_lambda2['var_unconditional'], var_static_lambda10['var_unconditional'], 
            var_dynamic_lambda2['var_unconditional'], var_dynamic_lambda10['var_unconditional']) - 0.02
x_max = max(var_static_lambda2['var_unconditional'], var_static_lambda10['var_unconditional'], 
            var_dynamic_lambda2['var_unconditional'], var_dynamic_lambda10['var_unconditional']) + 0.02

ax1.set_xlim(x_min, x_max)
ax2.set_xlim(x_min, x_max)
ax2.grid(False)


fig.suptitle('Unconditional Value-at-Risk (99%) Analysis: Static vs. Dynamic Allocation Strategies', 
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
plt.subplots_adjust(top=0.9, bottom=0.1)  
plt.show()






print("\n=== DETAILED DIAGNOSTIC OF OBSERVATIONS ===")


print(f"Observations static allocation λ=2: {len(static_losses_lambda2)}")
print(f"Observations static allocation λ=10: {len(static_losses_lambda10)}")
print(f"Observations dynamic allocation λ=2: {len(dynamic_losses_lambda2)}")
print(f"Observations dynamic allocation λ=10: {len(dynamic_losses_lambda10)}")


common_index_all = static_losses_lambda2.index.intersection(
    static_returns_lambda10.index.intersection(
        dynamic_returns_lambda2.index.intersection(
            dynamic_returns_lambda10.index
        )
    )
)

print(f"\nNumber of observations common to all series: {len(common_index_all)}")


coverage_static_2 = len(common_index_all) / len(static_losses_lambda2) * 100
coverage_static_10 = len(common_index_all) / len(static_losses_lambda10) * 100
coverage_dynamic_2 = len(common_index_all) / len(dynamic_losses_lambda2) * 100
coverage_dynamic_10 = len(common_index_all) / len(dynamic_losses_lambda10) * 100

print(f"Coverage of static data λ=2: {coverage_static_2:.2f}%")
print(f"Coverage of static data λ=10: {coverage_static_10:.2f}%")
print(f"Coverage of dynamic data λ=2: {coverage_dynamic_2:.2f}%")
print(f"Coverage of dynamic data λ=10: {coverage_dynamic_10:.2f}%")


print("\nPeriod covered by each series:")
print(f"Static λ=2: {static_losses_lambda2.index.min()} to {static_losses_lambda2.index.max()}")
print(f"Static λ=10: {static_losses_lambda10.index.min()} to {static_losses_lambda10.index.max()}")
print(f"Dynamic λ=2: {dynamic_losses_lambda2.index.min()} to {dynamic_losses_lambda2.index.max()}")
print(f"Dynamic λ=10: {dynamic_losses_lambda10.index.min()} to {dynamic_losses_lambda10.index.max()}")


if len(common_index_all) > 0:
    print(f"\nCommon period: {common_index_all.min()} to {common_index_all.max()}")
else:
    print("\nNo common period found!")





def plot_combined_var_violations(losses_dict, var_values_dict, violation_table=None, sample_period=None):
    """
    Visualise les violations de VaR de plusieurs portefeuilles sur une seule figure,
    en utilisant les statistiques du tableau de violations pour assurer la cohérence.
    
    Parameters:
    -----------
    losses_dict : dict
        Dictionnaire des séries de pertes {nom_portfolio: série_pertes}
    var_values_dict : dict
        Dictionnaire des valeurs VaR {nom_portfolio: valeur_var}
    violation_table : DataFrame, optional
        Tableau contenant les statistiques de violation (pour cohérence)
    sample_period : tuple, optional
        (date_début, date_fin) pour limiter l'affichage à une période
    """
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (22, 12)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    
    
    colors = {
        'Static Portfolio (λ=2)': '#1f77b4',    # bleu
        'Static Portfolio (λ=10)': '#ff7f0e',   # orange
        'Dynamic Portfolio (λ=2)': '#2ca02c',   # vert
        'Dynamic Portfolio (λ=10)': '#d62728'   # rouge
    }
    
    
    fig, axes = plt.subplots(2, 2, figsize=(22, 12), dpi=50, sharex=True)
    fig.suptitle('Value-at-Risk Violations (99%): Static vs. Dynamic Allocations', 
                fontsize=18, fontweight='bold')
    
    
    axes = axes.flatten()
    
    
    portfolio_mapping = {
        'Static Portfolio (λ=2)': 'Static (λ=2)',
        'Static Portfolio (λ=10)': 'Static (λ=10)', 
        'Dynamic Portfolio (λ=2)': 'Dynamic (λ=2)',
        'Dynamic Portfolio (λ=10)': 'Dynamic (λ=10)'
    }
    
    
    for i, (name, losses) in enumerate(losses_dict.items()):
        ax = axes[i]
        color = colors.get(name, 'steelblue')
        
        
        if sample_period:
            start_date, end_date = sample_period
            plot_losses = losses.loc[start_date:end_date]
        else:
            plot_losses = losses
        
        var_value = var_values_dict[name]
        
        
        ax.plot(plot_losses.index, plot_losses, color=color, alpha=0.5, 
                linewidth=0.8, label='Daily Losses')
        
        
        ax.axhline(y=var_value, color='red', linestyle='--', linewidth=1.5,
                  label=f'Unconditional VaR 99% = {var_value:.4f}')
        
        
        violations = plot_losses[plot_losses > var_value]
        
        
        ax.scatter(violations.index, violations, color='red', s=40, 
                   label=f'Violations ({len(violations)} occurrences)')
        
        
        if violation_table is not None and portfolio_mapping[name] in violation_table['Portfolio'].values:
            table_row = violation_table[violation_table['Portfolio'] == portfolio_mapping[name]].iloc[0]
            total_obs = table_row['Total Observations']
            num_violations = table_row['Violations']
            violation_rate = table_row['Violation Rate (%)']
            deviation = table_row['Deviation from 1%']
        else:
            
            total_obs = len(plot_losses)
            num_violations = len(violations)
            violation_rate = (num_violations / total_obs * 100) if total_obs > 0 else 0
            expected_rate = 1.0  
            deviation = violation_rate - expected_rate
        
        
        stats_text = (f"Total observations: {total_obs}\n"
                     f"Number of violations: {num_violations}\n"
                     f"Rate of violation: {violation_rate:.2f}%\n"
                     f"Deviation from 1%: {deviation:.2f}%")
        
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.02, 0.97, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=props)
        
        
        ax.set_title(name, fontweight='bold', fontsize=12)
        if i >= 2:  
            ax.set_xlabel('Date')
        if i % 2 == 0:  
            ax.set_ylabel('Losses')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower left', fontsize=8)
    
    for ax in axes.flatten():
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format année uniquement
        ax.xaxis.set_major_locator(mdates.YearLocator(2))  
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)  
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  
    
    return fig


var_static_lambda2_uncond = var_static_lambda2
var_static_lambda10_uncond = var_static_lambda10
var_dynamic_lambda2_uncond = var_dynamic_lambda2
var_dynamic_lambda10_uncond = var_dynamic_lambda10


sample_period = ('2001-01-01', '2024-12-31')  # Période incluant la crise financière


losses_dict = {
    'Static Portfolio (λ=2)': static_losses_lambda2,
    'Static Portfolio (λ=10)': static_losses_lambda10,
    'Dynamic Portfolio (λ=2)': dynamic_losses_lambda2, 
    'Dynamic Portfolio (λ=10)': dynamic_losses_lambda10
}

var_values_dict = {
    'Static Portfolio (λ=2)': var_static_lambda2_uncond['var_unconditional'],
    'Static Portfolio (λ=10)': var_static_lambda10_uncond['var_unconditional'],
    'Dynamic Portfolio (λ=2)': var_dynamic_lambda2_uncond['var_unconditional'],
    'Dynamic Portfolio (λ=10)': var_dynamic_lambda10_uncond['var_unconditional']
}


violation_table = pd.DataFrame({
    'Portfolio': ['Static (λ=2)', 'Static (λ=10)', 'Dynamic (λ=2)', 'Dynamic (λ=10)'],
    'VaR 99%': [
        var_static_lambda2['var_unconditional'],
        var_static_lambda10['var_unconditional'],
        var_dynamic_lambda2['var_unconditional'],
        var_dynamic_lambda10['var_unconditional']
    ],
    'Violations': [
        (static_losses_lambda2 > var_static_lambda2['var_unconditional']).sum(),
        (static_losses_lambda10 > var_static_lambda10['var_unconditional']).sum(),
        (dynamic_losses_lambda2 > var_dynamic_lambda2['var_unconditional']).sum(),
        (dynamic_losses_lambda10 > var_dynamic_lambda10['var_unconditional']).sum()
    ]
})


violation_table['Total Observations'] = [
    len(static_losses_lambda2),
    len(static_losses_lambda10),
    len(dynamic_losses_lambda2),
    len(dynamic_losses_lambda10)
]
violation_table['Violation Rate (%)'] = (violation_table['Violations'] / violation_table['Total Observations'] * 100).round(2)
violation_table['Deviation from 1%'] = (violation_table['Violation Rate (%)'] - 1.0).round(2)


fig = plot_combined_var_violations(losses_dict, var_values_dict, violation_table, sample_period)
plt.show()


print(violation_table)





print("=== DIAGNOSTIC OF OBSERVATIONS IN VAR VIOLATION ANALYSIS ===")
print(f"Static Portfolio (λ=2): {len(static_losses_lambda2)} days")
print(f"Static Portfolio (λ=10): {len(static_losses_lambda10)} days")
print(f"Dynamic Portfolio (λ=2): {len(dynamic_losses_lambda2)} days")
print(f"Dynamic Portfolio (λ=10): {len(dynamic_losses_lambda10)} days")


static_avg = (len(static_losses_lambda2) + len(static_losses_lambda10)) / 2
dynamic_avg = (len(dynamic_losses_lambda2) + len(dynamic_losses_lambda10)) / 2
ratio = dynamic_avg / static_avg if static_avg > 0 else 0

print("\n=== COMPARISON SUMMARY ===")
print(f"Average observations in static strategies: {static_avg:.0f} days")
print(f"Average observations in dynamic strategies: {dynamic_avg:.0f} days")
print(f"Ratio of dynamic to static observations: {ratio:.2%}")





violation_table.set_index('Portfolio', inplace=True)
violation_table.T





print('# ---------------------- Q 4.2  ---------------------------------------- #')






def estimate_ar1(returns):
    """Estime un AR(1) sur les retours."""
    model = AutoReg(returns, lags=1).fit()
    a, rho = model.params
    resid = model.resid
    return a, rho, resid

def estimate_garch11(resid):
    """Estime un GARCH(1,1) sur les résidus AR(1), avec rescale."""
    
    scaling_factor = 100
    scaled_resid = resid * scaling_factor
    gm = arch_model(scaled_resid, vol='GARCH', rescale=True, dist='StudentsT', p=1, q=1).fit(disp='off')

    
    ω = gm.params['omega']
    α = gm.params['alpha[1]']
    β = gm.params['beta[1]']
    
    σ = pd.Series(gm.conditional_volatility / scaling_factor, index=resid.index)
    
    return ω / (scaling_factor**2), α, β, σ

def conditional_mean(returns, a, rho):
    """Calcule μ_t = a + ρ·r_{t-1} pour un AR(1)."""
    m = pd.Series(index=returns.index, dtype=float)
    m.iloc[0] = a
    for t in range(1, len(returns)):
        m.iloc[t] = a + rho * returns.iloc[t-1]
    return m

def compute_var_conditional(returns, level=0.99):
    """
    Calcule la VaR conditionnelle 1-day à niveau 'level' et back-test.
    Inclut les statistiques d'inférence pour chaque paramètre.
    """
    if returns.index.freq is None:
        returns = returns.asfreq('B')
        
    
    model_ar = AutoReg(returns, lags=1).fit()
    a = model_ar.params.iloc[0]        
    rho = model_ar.params.iloc[1]      
    
    
    a_se = model_ar.bse.iloc[0]        
    rho_se = model_ar.bse.iloc[1]      
    a_tstat = model_ar.tvalues.iloc[0] 
    rho_tstat = model_ar.tvalues.iloc[1] 
    a_pval = model_ar.pvalues.iloc[0]  
    rho_pval = model_ar.pvalues.iloc[1] 
    resid = model_ar.resid
    
    
    scaling_factor = 100
    scaled_resid = resid * scaling_factor
    gm = arch_model(scaled_resid, vol='GARCH', rescale=False, p=1, q=1).fit(disp='off')
    
    
    ω = gm.params['omega'] / (scaling_factor**2)
    α = gm.params['alpha[1]']
    β = gm.params['beta[1]']
    
    
    ω_se = gm.std_err['omega'] / (scaling_factor**2)
    α_se = gm.std_err['alpha[1]']
    β_se = gm.std_err['beta[1]']
    
    ω_tstat = gm.tvalues['omega'] 
    α_tstat = gm.tvalues['alpha[1]']
    β_tstat = gm.tvalues['beta[1]']
    
    ω_pval = gm.pvalues['omega']
    α_pval = gm.pvalues['alpha[1]']
    β_pval = gm.pvalues['beta[1]']
    
    
    σ = pd.Series(gm.conditional_volatility / scaling_factor, index=resid.index)
    
    
    μ_ret = pd.Series(index=returns.index, dtype=float)
    μ_ret.iloc[0] = a
    for t in range(1, len(returns)):
        μ_ret.iloc[t] = a + rho * returns.iloc[t-1]
    
    
    μ_ret_f = μ_ret.shift(1).dropna()
    σ_f = σ.shift(1).dropna()
    
    
    μ_loss_f = -μ_ret_f
    
    
    z = norm.ppf(level)
    var_t = μ_loss_f + z * σ_f
    
    
    losses = -returns
    losses_aligned = losses.loc[var_t.index]
    
    n_viol = (losses_aligned > var_t).sum()
    rate = n_viol / len(losses_aligned)
    z_stat = (rate - (1-level)) / np.sqrt((1-level)*level/len(losses_aligned))
    p_val = 2 * (1 - norm.cdf(abs(z_stat)))
    
    
    return {
        'a': a, 'rho': rho,
        'omega': ω, 'alpha': α, 'beta': β,
        'a_se': a_se, 'rho_se': rho_se, 
        'omega_se': ω_se, 'alpha_se': α_se, 'beta_se': β_se,
        'a_tstat': a_tstat, 'rho_tstat': rho_tstat,
        'omega_tstat': ω_tstat, 'alpha_tstat': α_tstat, 'beta_tstat': β_tstat,
        'a_pval': a_pval, 'rho_pval': rho_pval,
        'omega_pval': ω_pval, 'alpha_pval': α_pval, 'beta_pval': β_pval,
        'vol_cond_mean': σ_f.mean(),
        'var': var_t,
        'n_viol': n_viol,
        'rate': rate,
        'z_stat': z_stat,
        'p_val': p_val
    }






resS2 = compute_var_conditional(-static_losses_lambda2,  level=0.99)
resD2 = compute_var_conditional(-dynamic_losses_lambda2, level=0.99)


print("=== Static (λ=2) ===")
print(f"Vol cond moyenne   : {resS2['vol_cond_mean']:.4f}")
print(f"Violations         : {resS2['n_viol']} / {len(static_losses_lambda2)}")
print(f"Violation rate     : {resS2['rate']:.2%}")
print(f"Z-stat / p-value   : {resS2['z_stat']:.2f} / {resS2['p_val']:.4f}")
print(f"AR(1) params       : a={resS2['a']:.6f}, ρ={resS2['rho']:.6f}")
print(f"GARCH params       : ω={resS2['omega']:.6f}, α={resS2['alpha']:.6f}, β={resS2['beta']:.6f}")

print("\n=== Dynamic (λ=2) ===")
print(f"Vol cond moyenne   : {resD2['vol_cond_mean']:.4f}")
print(f"Violations         : {resD2['n_viol']} / {len(dynamic_losses_lambda2)}")
print(f"Violation rate     : {resD2['rate']:.2%}")
print(f"Z-stat / p-value   : {resD2['z_stat']:.2f} / {resD2['p_val']:.4f}")






resS10 = compute_var_conditional(-static_losses_lambda10, level=0.99)
resD10 = compute_var_conditional(-dynamic_losses_lambda10, level=0.99)


print("\n=== Static (λ=10) ===")
print(f"Vol cond moyenne   : {resS10['vol_cond_mean']:.4f}")
print(f"Violations         : {resS10['n_viol']} / {len(static_losses_lambda10)}")
print(f"Violation rate     : {resS10['rate']:.2%}")
print(f"Z-stat / p-value   : {resS10['z_stat']:.2f} / {resS10['p_val']:.4f}")

print("\n=== Dynamic (λ=10) ===")
print(f"Vol cond moyenne   : {resD10['vol_cond_mean']:.4f}")
print(f"Violations         : {resD10['n_viol']} / {len(dynamic_losses_lambda10)}")
print(f"Violation rate     : {resD10['rate']:.2%}")
print(f"Z-stat / p-value   : {resD10['z_stat']:.2f} / {resD10['p_val']:.4f}")





def plot_conditional_var_comparison(static_losses_lambda2, static_losses_lambda10,
                                   dynamic_losses_lambda2, dynamic_losses_lambda10,
                                   resS2, resS10, resD2, resD10,
                                   sample_start=None, sample_end=None):
    """
    Creates a professional comparison chart of conditional VaR for all 4 strategies
    with highlighted violations and summary statistics
    """
    
    plt.close('all')
    
    
    if sample_start is not None and sample_end is not None:
        static_losses_lambda2 = static_losses_lambda2[sample_start:sample_end]
        static_losses_lambda10 = static_losses_lambda10[sample_start:sample_end]
        dynamic_losses_lambda2 = dynamic_losses_lambda2[sample_start:sample_end]
        dynamic_losses_lambda10 = dynamic_losses_lambda10[sample_start:sample_end]
        
        var_S2 = resS2['var'][sample_start:sample_end]
        var_S10 = resS10['var'][sample_start:sample_end]
        var_D2 = resD2['var'][sample_start:sample_end]
        var_D10 = resD10['var'][sample_start:sample_end]
    else:
        var_S2 = resS2['var']
        var_S10 = resS10['var']
        var_D2 = resD2['var']
        var_D10 = resD10['var']
    
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['figure.figsize'] = (22, 12)
    
    
    colors = {
        'Static (λ=2)': '#1f77b4',     # blue
        'Static (λ=10)': '#ff7f0e',    # orange
        'Dynamic (λ=2)': '#2ca02c',    # green
        'Dynamic (λ=10)': '#d62728',   # red
        'VaR': '#424242',              # gray
        'Violations': '#e41a1c'        # bright red
    }
    
    
    fig, axes = plt.subplots(2, 2, figsize=(22, 12), sharex=True)
    
    
    axes[0, 0].plot(static_losses_lambda2.index, static_losses_lambda2, 
                    color=colors['Static (λ=2)'], alpha=0.7, label='Actual Losses')
    axes[0, 0].plot(var_S2.index, var_S2, 
                    color=colors['VaR'], linestyle='--', linewidth=0.8, label='Conditional VaR (99%)')
    
    
    common_idx_S2 = static_losses_lambda2.index.intersection(var_S2.index)
    violations_S2 = static_losses_lambda2.loc[common_idx_S2][static_losses_lambda2.loc[common_idx_S2] > var_S2.loc[common_idx_S2]]
    axes[0, 0].scatter(violations_S2.index, violations_S2, color=colors['Violations'], 
                       s=50, alpha=0.7, marker='o', edgecolors='red', linewidths=0.3,
                       label=f'Violations ({len(violations_S2)})')
    
    axes[0, 0].set_title(f'Static (λ=2): Losses vs. Conditional VaR\nViolations: {resS2["n_viol"]} ({resS2["rate"]:.2%})', 
                        fontweight='bold')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend(loc='upper left',fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    
    axes[0, 1].plot(static_losses_lambda10.index, static_losses_lambda10, 
                    color=colors['Static (λ=10)'], alpha=0.7, label='Actual Losses')
    axes[0, 1].plot(var_S10.index, var_S10, 
                    color=colors['VaR'], linestyle='--', linewidth=0.8, label='Conditional VaR (99%)')
    
    
    common_idx_S10 = static_losses_lambda10.index.intersection(var_S10.index)
    violations_S10 = static_losses_lambda10.loc[common_idx_S10][static_losses_lambda10.loc[common_idx_S10] > var_S10.loc[common_idx_S10]]
    axes[0, 1].scatter(violations_S10.index, violations_S10, color=colors['Violations'], 
                       s=50, alpha=0.7, marker='o', edgecolors='black', linewidths=0.3,
                       label=f'Violations ({len(violations_S10)})')
    
    axes[0, 1].set_title(f'Static (λ=10): Losses vs. Conditional VaR\nViolations: {resS10["n_viol"]} ({resS10["rate"]:.2%})', 
                         fontweight='bold')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend(loc='upper left',fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    
    axes[1, 0].plot(dynamic_losses_lambda2.index, dynamic_losses_lambda2, 
                    color=colors['Dynamic (λ=2)'], alpha=0.7, label='Actual Losses')
    axes[1, 0].plot(var_D2.index, var_D2, 
                    color=colors['VaR'], linestyle='--', linewidth=0.8, label='Conditional VaR (99%)')
    
    
    common_idx_D2 = dynamic_losses_lambda2.index.intersection(var_D2.index)
    violations_D2 = dynamic_losses_lambda2.loc[common_idx_D2][dynamic_losses_lambda2.loc[common_idx_D2] > var_D2.loc[common_idx_D2]]
    axes[1, 0].scatter(violations_D2.index, violations_D2, color=colors['Violations'], 
                       s=50, alpha=0.7, marker='o', edgecolors='black', linewidths=0.3,
                       label=f'Violations ({len(violations_D2)})')
    
    axes[1, 0].set_title(f'Dynamic (λ=2): Losses vs. Conditional VaR\nViolations: {resD2["n_viol"]} ({resD2["rate"]:.2%})', 
                         fontweight='bold')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend(loc='upper left',fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    
    axes[1, 1].plot(dynamic_losses_lambda10.index, dynamic_losses_lambda10, 
                    color=colors['Dynamic (λ=10)'], alpha=0.7, label='Actual Losses')
    axes[1, 1].plot(var_D10.index, var_D10, 
                    color=colors['VaR'], linestyle='--', linewidth=0.8, label='Conditional VaR (99%)')
    
    
    common_idx_D10 = dynamic_losses_lambda10.index.intersection(var_D10.index)
    violations_D10 = dynamic_losses_lambda10.loc[common_idx_D10][dynamic_losses_lambda10.loc[common_idx_D10] > var_D10.loc[common_idx_D10]]
    axes[1, 1].scatter(violations_D10.index, violations_D10, color=colors['Violations'], 
                       s=50, alpha=0.7, marker='o', edgecolors='black', linewidths=0.3,
                       label=f'Violations ({len(violations_D10)})')
    
    axes[1, 1].set_title(f'Dynamic (λ=10): Losses vs. Conditional VaR\nViolations: {resD10["n_viol"]} ({resD10["rate"]:.2%})', 
                         fontweight='bold')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend(loc='upper left',fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    
    for ax in axes.flatten():
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
    
    
    fig.suptitle('Conditional 99% Value-at-Risk (AR(1)-GARCH(1,1)): Static vs. Dynamic Allocations', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.10)
    
    return fig, axes


fig, axes = plot_conditional_var_comparison(
    static_losses_lambda2, static_losses_lambda10,
    dynamic_losses_lambda2, dynamic_losses_lambda10,
    resS2, resS10, resD2, resD10
)

plt.show()



sample_start = '2019-01-01'
sample_end = '2020-12-31'


fig2, axes2 = plot_conditional_var_comparison(
    static_losses_lambda2, static_losses_lambda10,
    dynamic_losses_lambda2, dynamic_losses_lambda10,
    resS2, resS10, resD2, resD10,
    sample_start=sample_start, sample_end=sample_end
)

plt.show()








portfolios = {
    'Static (λ=2)':  resS2,
    'Static (λ=10)': resS10,
    'Dynamic (λ=2)': resD2,
    'Dynamic (λ=10)':resD10
}


metrics = [
    'a', 'a_se', 'a_tstat', 'a_pval',
    'rho', 'rho_se', 'rho_tstat', 'rho_pval',
    'omega', 'omega_se', 'omega_tstat', 'omega_pval',
    'alpha', 'alpha_se', 'alpha_tstat', 'alpha_pval',
    'beta', 'beta_se', 'beta_tstat', 'beta_pval',
    'n_viol', 'rate', 'z_stat', 'p_val'
]


row_labels = [
    'AR(1): $a$',    'SE$(a)$',       '$t$-stat$(a)$',      '$p$-value$(a)$',
    'AR(1): $\\rho$',    'SE$(\\rho)$',       '$t$-stat$(\\rho)$',      '$p$-value$(\\rho)$',
    'GARCH: $\\omega$',    'SE$(\\omega)$',       '$t$-stat$(\\omega)$',      '$p$-value$(\\omega)$',
    'GARCH: $\\alpha$',    'SE$(\\alpha)$',       '$t$-stat$(\\alpha)$',      '$p$-value$(\\alpha)$',
    'GARCH: $\\beta$',    'SE$(\\beta)$',       '$t$-stat$(\\beta)$',      '$p$-value$(\\beta)$',
    'Violations',  'Violation rate', 'Kupiec $Z$-stat', 'Kupiec $p$-value'
]


table = pd.DataFrame({
    name: pd.Series({label: res[key] for label, key in zip(row_labels, metrics)})
    for name, res in portfolios.items()
})
table.index.name = 'Metric'


table = table.rename(columns={
    'Static (λ=2)': 'Static ($\\lambda=2$)',
    'Static (λ=10)': 'Static ($\\lambda=10$)',
    'Dynamic (λ=2)': 'Dynamic ($\\lambda=2$)',
    'Dynamic (λ=10)': 'Dynamic ($\\lambda=10$)'
})










print("=== DIAGNOSTIC: Number of days in conditional VaR analysis datasets ===")
print(f"Static strategy (λ=2): {len(static_losses_lambda2)} days")
print(f"Static strategy (λ=10): {len(static_losses_lambda10)} days")
print(f"Dynamic strategy (λ=2): {len(dynamic_losses_lambda2)} days")
print(f"Dynamic strategy (λ=10): {len(dynamic_losses_lambda10)} days")


common_days = static_losses_lambda2.index.intersection(
    static_losses_lambda10.index.intersection(
        dynamic_losses_lambda2.index.intersection(
            dynamic_losses_lambda10.index
        )
    )
)
print(f"Common period across all strategies: {len(common_days)} days")


print("\n=== DIAGNOSTIC: Aligned periods in conditional VaR model results ===")
print(f"Model results - Static (λ=2): {len(resS2['var'])} days")
print(f"Model results - Static (λ=10): {len(resS10['var'])} days")
print(f"Model results - Dynamic (λ=2): {len(resD2['var'])} days")
print(f"Model results - Dynamic (λ=10): {len(resD10['var'])} days")


if sample_start is not None and sample_end is not None:
    sample_period = pd.date_range(start=sample_start, end=sample_end)
    print(f"\n=== DIAGNOSTIC: Sample period ({sample_start} to {sample_end}) ===")
    print(f"Number of days in sample period: {len(sample_period)}")
    
    
    print(f"Static (λ=2) days in sample: {len(static_losses_lambda2.loc[sample_start:sample_end])}")
    print(f"Static (λ=10) days in sample: {len(static_losses_lambda10.loc[sample_start:sample_end])}")
    print(f"Dynamic (λ=2) days in sample: {len(dynamic_losses_lambda2.loc[sample_start:sample_end])}")
    print(f"Dynamic (λ=10) days in sample: {len(dynamic_losses_lambda10.loc[sample_start:sample_end])}")





print('# ---------------------- Q 4.3  ---------------------------------------- #\n\n')

print('# ------------------------------------------------------------------ #')
print('# ------------------------------------------------------------------ #')
print('# Please wait bootstraping, this may take a little moment, ... #')
print('# ------------------------------------------------------------------ #')
print('# ------------------------------------------------------------------ #')



def extract_std_residuals_from_model(model_dict, returns):
    """Extrait correctement les résidus standardisés d'un modèle renvoyé par compute_var_conditional"""
    
    a = model_dict['a']
    rho = model_dict['rho']
    
    
    omega = model_dict['omega'] 
    alpha = model_dict['alpha']
    beta = model_dict['beta']
    
    
    mu = pd.Series(index=returns.index, dtype=float)
    mu.iloc[0] = a
    for t in range(1, len(returns)):
        mu.iloc[t] = a + rho * returns.iloc[t-1]
    
    resid = returns - mu
    
    
    sigma = pd.Series(index=returns.index, dtype=float)
    sigma.iloc[0] = np.sqrt(omega / (1 - alpha - beta))
    
    for t in range(1, len(returns)):
        sigma.iloc[t] = np.sqrt(omega + alpha * resid.iloc[t-1]**2 + beta * sigma.iloc[t-1]**2)
    
    
    std_resid = resid / sigma
    
    return std_resid



def compute_block_maxima(standardized_residuals, block_size=60):
    
    n_obs = len(standardized_residuals)
    n_blocks = n_obs // block_size
    
    maxima = []
    block_dates = []
    
    for i in range(n_blocks):
        start_idx = i * block_size
        end_idx = (i + 1) * block_size
        
        if end_idx <= n_obs:  
            block = standardized_residuals.iloc[start_idx:end_idx]
            block_max = block.max()
            maxima.append(block_max)
            block_dates.append(block.index[-1])  
    
    remaining = n_obs % block_size
    if remaining > block_size // 2:  
        last_block = standardized_residuals.iloc[-remaining:]
        maxima.append(last_block.max())
        block_dates.append(last_block.index[-1])
    
    return pd.Series(maxima, index=block_dates, name='BlockMaxima')

def fit_gev(block_maxima, method='mle'):
    
    shape, loc, scale = genextreme.fit(-block_maxima, method=method)
    
    xi = -shape
    varpi = loc
    psi = scale
    
    try:
        np.random.seed(123)  
        n_bootstrap = 5000
        bootstrap_estimates = np.zeros((n_bootstrap, 3))
        
        for i in range(n_bootstrap):
            bootstrap_sample = genextreme.rvs(shape, loc=loc, scale=scale, size=len(block_maxima))
            bootstrap_shape, bootstrap_loc, bootstrap_scale = genextreme.fit(-bootstrap_sample, method=method)
            bootstrap_estimates[i] = [-bootstrap_shape, bootstrap_loc, bootstrap_scale]
        
        xi_se, varpi_se, psi_se = np.std(bootstrap_estimates, axis=0)
    except:
        xi_se = np.abs(xi * 0.2)
        varpi_se = np.abs(varpi * 0.1)
        psi_se = np.abs(psi * 0.1)
    
    return {
        'xi': xi,
        'varpi': varpi,
        'psi': psi,
        'xi_se': xi_se,
        'varpi_se': varpi_se,
        'psi_se': psi_se
    }






def calculate_return_level(params, return_period, block_size=60, n_trading_days=252):
    
    xi = params['xi']
    varpi = params['varpi']
    psi = params['psi']
    
    blocks_per_year = n_trading_days / block_size
    T_blocks = return_period * blocks_per_year
    
    if xi == 0:  
        return_level = varpi - psi * np.log(-np.log(1 - 1/T_blocks))
    else:
        return_level = varpi + (psi/xi) * ((T_blocks)**xi - 1)
    
    return -return_level





def analyze_extreme_values(standardized_residuals, block_size=60, portfolio_name="Portfolio"):
    
    print(f"\n=== Analyse GEV pour {portfolio_name} ===")
    
    block_maxima = compute_block_maxima(standardized_residuals, block_size)
    print(f"Nombre de blocs: {len(block_maxima)}")
    
    gev_params = fit_gev(block_maxima)
    print("\nParamètres estimés:")
    print(f"ξ (xi) = {gev_params['xi']:.4f} ± {gev_params['xi_se']:.4f}")
    print(f"ϖ (varpi) = {gev_params['varpi']:.4f} ± {gev_params['varpi_se']:.4f}")
    print(f"ψ (psi) = {gev_params['psi']:.4f} ± {gev_params['psi_se']:.4f}")
    
    if gev_params['xi'] > 0:
        tail_type = "Heavy tail (Fréchet type)"
    elif gev_params['xi'] < 0:
        tail_type = "Bounded tail (Weibull type)"
    else:
        tail_type = "Light tail (Gumbel type)"
    
    print(f"Type de queue: {tail_type}")
    
    
    
    
    
    
    
    results = pd.DataFrame({
        'Parameter': ['ξ (xi)', 'ϖ (varpi)', 'ψ (psi)'],
        'Estimate': [gev_params['xi'], gev_params['varpi'], gev_params['psi']],
        'Std. Error': [gev_params['xi_se'], gev_params['varpi_se'], gev_params['psi_se']],
        't-stat': [gev_params['xi']/gev_params['xi_se'], 
                  gev_params['varpi']/gev_params['varpi_se'],
                  gev_params['psi']/gev_params['psi_se']]
    })
    
    print("\nTableau des estimations:")
    print(results)
    
    return_periods = [0.25, 0.5, 1, 2, 5, 10]
    return_levels = calculate_return_level(gev_params, np.array(return_periods), block_size)
    
    return_level_table = pd.DataFrame({
        'Return Period (years)': return_periods,
        'Return Level': return_levels
    })
    
    print("\nNiveaux de retour:")
    print(return_level_table)
    
    
    
    
    
    
    
    
    
    
    return block_maxima, gev_params, results, return_level_table


def get_standardized_residuals(returns=None, model_results=None):
    """
    Obtient les résidus standardisés soit à partir d'un modèle existant,
    soit en estimant un modèle AR(1)-GARCH(1,1) sur les rendements fournis.
    
    Parameters:
    -----------
    returns : Series, optional
        Série des rendements (utilisée si model_results n'est pas fourni)
    model_results : object, optional
        Résultats d'un modèle GARCH déjà estimé
        
    Returns:
    --------
    Series
        Résidus standardisés
    """
    
    if model_results is not None and hasattr(model_results, 'resid') and hasattr(model_results, 'conditional_volatility'):
        std_residuals = model_results.resid / model_results.conditional_volatility
        return pd.Series(std_residuals, index=model_results.resid.index)
    
    
    elif model_results is not None and isinstance(model_results, dict) and 'var' in model_results:
        std_residuals = model_results.get('standardized_residuals', None)
        if std_residuals is not None:
            return std_residuals
    
    
    elif returns is not None:
        print("Estimation d'un modèle AR(1)-GARCH(1,1)...")
        
        
        ar_model = sm.tsa.ARIMA(returns, order=(1, 0, 0)).fit()
        ar_resid = ar_model.resid
        
        
        garch_model = arch_model(ar_resid, vol='GARCH', p=1, q=1).fit(disp='off')
        std_residuals = garch_model.resid / garch_model.conditional_volatility
        
        return pd.Series(std_residuals, index=returns.index, name="standardized_residuals")
    
    
    else:
        print("ATTENTION: Ni modèle ni rendements fournis. Utilisation de données simulées.")
        
        index = pd.date_range(start='2020-01-01', periods=1000, freq='D')
        return pd.Series(np.random.normal(0, 1, 1000), index=index, name="Simulated_Residuals")


def compare_portfolios_gev(portfolios_data, block_size=60, verbose=False):
    """
    Analyse et compare plusieurs portefeuilles avec GEV.
    
    Parameters:
    -----------
    portfolios_data : dict
        Dictionnaire avec structure {nom_portfolio: (returns ou model_results, losses)}
    block_size : int
        Taille des blocs pour GEV
        
    Returns:
    --------
    DataFrame
        Tableau comparatif des paramètres GEV
    """
    all_results = {}
    all_maxima = {}
    
    for name, (model_or_returns, losses) in portfolios_data.items():
        if verbose:
            print(f"\nTraitement de {name}...")
        
        
        std_residuals = get_standardized_residuals(
            returns=-losses if losses is not None else None,
            model_results=model_or_returns
        )
        
        
        block_maxima, gev_params, params_table, return_levels = analyze_extreme_values(
            std_residuals, block_size=block_size, portfolio_name=name)
        
        
        all_results[name] = {
            'gev_params': gev_params,
            'params_table': params_table,
            'return_levels': return_levels
        }
        all_maxima[name] = block_maxima
        
        
    
    
    compare_params = pd.DataFrame(
        {name: [all_results[name]['gev_params']['xi'], 
                all_results[name]['gev_params']['varpi'], 
                all_results[name]['gev_params']['psi']] 
         for name in all_results.keys()},
        index=['ξ (xi)', 'ϖ (varpi)', 'ψ (psi)']
    )
    
    print("\n=== GEV Params comparison between portfolios ===")
    print(compare_params)
    

    
    
    return compare_params, all_results, all_maxima













def create_combined_return_level_plots(results_dict, colors_dict, block_size=60, max_years=10):
    """
    Crée une figure unique contenant tous les graphiques de niveaux de retour avec un schéma de couleurs cohérent.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=50)
    axes = axes.flatten()
    
    
    return_periods = np.linspace(0.1, max_years, 100)
    
    for i, (name, results) in enumerate(results_dict.items()):
        params = results['gev_params']
        color = colors_dict.get(name, 'blue')
        
        
        return_levels = calculate_return_level(params, return_periods, block_size)
        
        
        xi = params['xi']
        psi = params['psi']
        xi_se = params['xi_se']
        psi_se = params['psi_se']
        z = stats.norm.ppf(0.975)  
        
        blocks_per_year = 252 / block_size
        T_blocks = return_periods * blocks_per_year
        
        
        if xi != 0:
            term1 = (psi_se/xi) * ((T_blocks)**xi - 1)
            term2 = (psi*xi_se/xi**2) * ((T_blocks)**xi - 1)
            term3 = (psi*xi_se/xi) * ((T_blocks)**xi * np.log(T_blocks))
            se_return_level = np.sqrt(term1**2 + term2**2 + term3**2)
        else:
            se_return_level = psi_se * np.log(T_blocks)
        
        
        upper_bound = return_levels + z * se_return_level
        lower_bound = return_levels - z * se_return_level
        
        
        axes[i].plot(return_periods, return_levels, '-', color=color, linewidth=2, label='Expected')
        axes[i].fill_between(return_periods, lower_bound, upper_bound, color=color, alpha=0.2, 
                           label='95% CI')
        
        
        highlight_periods = [0.25, 0.5, 1, 2, 5, 10]
        highlight_periods = [p for p in highlight_periods if p <= max_years]
        highlight_levels = calculate_return_level(params, np.array(highlight_periods), block_size)
        
        axes[i].scatter(highlight_periods, highlight_levels, color='red', s=50, zorder=5)
        
        
        for period, level in zip(highlight_periods, highlight_levels):
            if period >= 1:
                axes[i].annotate(f"{period:.0f}y: {level:.2f}", 
                              xy=(period, level), xytext=(5, 0),
                              textcoords='offset points', fontsize=8)
        
        axes[i].set_title(f"Return Level Plot: {name}", fontweight='bold')
        axes[i].set_xlabel('Return Period (years)')
        axes[i].set_ylabel('Return Level')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc='upper left')
    
    plt.tight_layout()
    plt.suptitle("Return Level Plots by Portfolio", y=1.02, fontsize=16, fontweight='bold')
    return fig


portfolio_colors = {
    "Static (λ=2)": "#1f77b4",     # bleu
    "Static (λ=10)": "#ff7f0e",    # orange
    "Dynamic (λ=2)": "#2ca02c",    # vert
    "Dynamic (λ=10)": "#d62728"    # rouge
}


def compare_portfolios_gev_silent(portfolios_data, block_size=60):
    """Version silencieuse de compare_portfolios_gev qui n'affiche pas les graphiques intermédiaires"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        
        import sys
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        compare_params, results, maxima = compare_portfolios_gev(portfolios_data, block_size)
        
        
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        
        print(output)
        
        
        
        
        
        
        
        
        
        
        
        
        
        return compare_params, results, maxima




portfolios_data = {
    "Static (λ=2)": (extract_std_residuals_from_model(resS2, -static_losses_lambda2), static_losses_lambda2),
    "Static (λ=10)": (extract_std_residuals_from_model(resS10, -static_losses_lambda10), static_losses_lambda10),
    "Dynamic (λ=2)": (extract_std_residuals_from_model(resD2, -dynamic_losses_lambda2), dynamic_losses_lambda2),
    "Dynamic (λ=10)": (extract_std_residuals_from_model(resD10, -dynamic_losses_lambda10), dynamic_losses_lambda10)
}

compare_params, results, maxima = compare_portfolios_gev_silent(portfolios_data)






compare_params






print("=== DIAGNOSTIC: Observations and blocks for GEV analysis ===")
print(f"Static strategy (λ=2): {len(maxima['Static (λ=2)'])} blocks")
print(f"Static strategy (λ=10): {len(maxima['Static (λ=10)'])} blocks")
print(f"Dynamic strategy (λ=2): {len(maxima['Dynamic (λ=2)'])} blocks")
print(f"Dynamic strategy (λ=10): {len(maxima['Dynamic (λ=10)'])} blocks")


block_size = 60  
print("\n=== Estimated number of days (based on block size) ===")
print(f"Static strategy (λ=2): {len(maxima['Static (λ=2)']) * block_size} days")
print(f"Static strategy (λ=10): {len(maxima['Static (λ=10)']) * block_size} days") 
print(f"Dynamic strategy (λ=2): {len(maxima['Dynamic (λ=2)']) * block_size} days")
print(f"Dynamic strategy (λ=10): {len(maxima['Dynamic (λ=10)']) * block_size} days")


print("\n=== Date range covered by each strategy ===")
for strategy in maxima.keys():
    if len(maxima[strategy]) > 0:
        start_date = maxima[strategy].index.min()
        end_date = maxima[strategy].index.max()
        print(f"{strategy}: {start_date} to {end_date}")


static_blocks = (len(maxima['Static (λ=2)']) + len(maxima['Static (λ=10)'])) / 2
dynamic_blocks = (len(maxima['Dynamic (λ=2)']) + len(maxima['Dynamic (λ=10)'])) / 2

print("\n=== Static vs Dynamic comparison ===")
print(f"Average blocks in static strategies: {static_blocks:.1f}")
print(f"Average blocks in dynamic strategies: {dynamic_blocks:.1f}")
print(f"Ratio of dynamic to static blocks: {dynamic_blocks/static_blocks:.2%}")





def plot_block_maxima_distributions(maxima_dict, results_dict=None, colors_dict=None, 
                                    figsize=(22, 12), show_gev_pdf=False):
    """
    prints block maxima distributions for all portfolios on 4 subplots,
    with option to show theoretical GEV distributions.
    
    Parameters:
    -----------
    maxima_dict : dict
        Dictionary of block maxima series for each portfolio
    results_dict : dict, optional
        Dictionary containing GEV results (required if show_gev_pdf=True)
    colors_dict : dict, optional
        Dictionary mapping each portfolio to a color
    figsize : tuple, optional
        Figure size
    show_gev_pdf : bool, optional
        If True, shows corresponding theoretical GEV densities
        
    Returns:
    --------
    matplotlib.figure.Figure
        Matplotlib figure containing the plot
    """
    from scipy.stats import genextreme
    import numpy as np
    
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    
    if colors_dict is None:
        colors_dict = {
            "Static (λ=2)": "#1f77b4",     # blue
            "Static (λ=10)": "#ff7f0e",    # orange
            "Dynamic (λ=2)": "#2ca02c",    # green
            "Dynamic (λ=10)": "#d62728"    # red
        }
    
    
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=50, sharex=False, sharey=False)
    axes = axes.flatten()  
    
    all_dates = []
    for series in maxima_dict.values():
        all_dates.extend(series.index.tolist())
    
    min_date = min(all_dates)
    max_date = max(all_dates)
    date_range_years = (max_date - min_date).days / 365.25
    
    
    
    
    
    
    fig.suptitle("Block Maxima Distribution", fontsize=18, fontweight='bold', y=0.98)
    
    
    for i, (name, maxima_series) in enumerate(maxima_dict.items()):
        color = colors_dict.get(name, 'blue')
        ax = axes[i]
        
        
        mean_val = maxima_series.mean()
        std_dev = maxima_series.std()
        
        
        sns.histplot(
            maxima_series, 
            kde=True,
            bins = 20,
            stat="density",
            alpha=0.4, 
            color=color,
            ax=ax,
            label=f"Empirical distribution (μ={mean_val:.2f}, σ={std_dev:.2f})"
        )
        
        if show_gev_pdf and results_dict is not None:
            
            params = results_dict[name]['gev_params']
            xi = params['xi']       # paramètre de forme
            varpi = params['varpi'] # paramètre de position
            psi = params['psi']     # paramètre d'échelle
            
            
            x_min, x_max = ax.get_xlim()
            x = np.linspace(x_min, x_max, 1000)
            
            
            y = genextreme.pdf(x, -xi, loc=varpi, scale=psi)
            
            
            ax.plot(x, y, '--', color='black', linewidth=1.5, 
                   label=f"GEV théorique (ξ={xi:.3f})")
            
            
            if xi < 0:
                law_type = "Weibull (queue bornée)"
            elif xi > 0:
                law_type = "Fréchet (queue lourde)"
            else:
                law_type = "Gumbel (queue légère)"
                
            ax.text(
                0.95, 0.95, 
                f"Type: {law_type}", 
                transform=ax.transAxes, 
                fontsize=10, 
                ha='right',
                va='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
        
        
        ax.set_title(f"{name}", fontweight='bold')
        ax.set_xlabel("Maximum value of the block", fontsize=9)
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3, linestyle='--')
        
        
        ax.legend(loc='upper right', frameon=True, fancybox=True)
    
    
    
    
    
    
    
    
    
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.08)  
    
    return fig

    
def create_combined_qq_plots(maxima_dict, results_dict, colors_dict):
    """
    Crée une figure unique contenant tous les QQ-plots avec un schéma de couleurs cohérent.
    
    Parameters:
    -----------
    maxima_dict : dict
        Dictionnaire des maxima par bloc pour chaque portefeuille
    results_dict : dict
        Dictionnaire des résultats GEV pour chaque portefeuille
    colors_dict : dict
        Dictionnaire associant chaque portefeuille à une couleur
    """
    fig, axes = plt.subplots(2, 2, figsize=(22, 12), dpi=50)
    axes = axes.flatten()
    
    for i, (name, maxima) in enumerate(maxima_dict.items()):
        
        params = results_dict[name]['gev_params']
        shape = -params['xi']  # Conversion pour scipy
        loc = params['varpi']
        scale = params['psi']
        
        
        sorted_data = np.sort(-maxima.values)
        n = len(sorted_data)
        p = np.arange(1, n + 1) / (n + 1)
        theoretical_quantiles = genextreme.ppf(p, shape, loc=loc, scale=scale)
        
        
        axes[i].scatter(theoretical_quantiles, sorted_data, alpha=0.7, 
                       color=colors_dict.get(name, 'blue'), s=40)
        
        
        min_val = min(theoretical_quantiles.min(), sorted_data.min())
        max_val = max(theoretical_quantiles.max(), sorted_data.max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.6)
        
        
        param_text = f"ξ = {params['xi']:.4f} (±{params['xi_se']:.4f})\n"
        param_text += f"ϖ = {params['varpi']:.4f} (±{params['varpi_se']:.4f})\n"
        param_text += f"ψ = {params['psi']:.4f} (±{params['psi_se']:.4f})"
        
        axes[i].annotate(param_text, xy=(0.05, 0.95), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                        ha='left', va='top', fontsize=9)
        
        axes[i].set_title(f"QQ-Plot: {name}", fontweight='bold')
        axes[i].set_xlabel('Theoretical Quantiles (GEV)')
        axes[i].set_ylabel('Sample Quantiles')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle("QQ-Plots for Generalized Extreme Value Distribution", y=1.02, fontsize=16, fontweight='bold')
    return fig


def create_combined_return_level_plots(results_dict, colors_dict, block_size=60, max_years=10):
    """
    Crée une figure unique contenant tous les graphiques de niveaux de retour avec un schéma de couleurs cohérent.
    """
    fig, axes = plt.subplots(2, 2, figsize=(22, 12), dpi=50)
    axes = axes.flatten()
    
    
    return_periods = np.linspace(0.1, max_years, 100)
    
    for i, (name, results) in enumerate(results_dict.items()):
        params = results['gev_params']
        color = colors_dict.get(name, 'blue')
        
        
        return_levels = calculate_return_level(params, return_periods, block_size)
        
        
        xi = params['xi']
        psi = params['psi']
        xi_se = params['xi_se']
        psi_se = params['psi_se']
        z = stats.norm.ppf(0.975)  
        
        blocks_per_year = 252 / block_size
        T_blocks = return_periods * blocks_per_year
        
        
        if xi != 0:
            term1 = (psi_se/xi) * ((T_blocks)**xi - 1)
            term2 = (psi*xi_se/xi**2) * ((T_blocks)**xi - 1)
            term3 = (psi*xi_se/xi) * ((T_blocks)**xi * np.log(T_blocks))
            se_return_level = np.sqrt(term1**2 + term2**2 + term3**2)
        else:
            se_return_level = psi_se * np.log(T_blocks)
        
        
        upper_bound = return_levels + z * se_return_level
        lower_bound = return_levels - z * se_return_level
        
        
        axes[i].plot(return_periods, return_levels, '-', color=color, linewidth=2, label='Expected')
        axes[i].fill_between(return_periods, lower_bound, upper_bound, color=color, alpha=0.2, 
                           label='95% CI')
        
        
        highlight_periods = [0.25, 0.5, 1, 2, 5, 10]
        highlight_periods = [p for p in highlight_periods if p <= max_years]
        highlight_levels = calculate_return_level(params, np.array(highlight_periods), block_size)
        
        axes[i].scatter(highlight_periods, highlight_levels, color='red', s=50, zorder=5)
        
        
        for period, level in zip(highlight_periods, highlight_levels):
            if period >= 1:
                axes[i].annotate(f"{period:.0f}y: {level:.2f}", 
                              xy=(period, level), xytext=(5, 0),
                              textcoords='offset points', fontsize=8)
        
        axes[i].set_title(f"Return Level Plot: {name}", fontweight='bold')
        axes[i].set_xlabel('Return Period (years)')
        axes[i].set_ylabel('Return Level')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc='upper right')
    
    plt.tight_layout()
    plt.suptitle("Return Level Plots by Portfolio", y=1.02, fontsize=16, fontweight='bold')
    return fig





fig = plot_block_maxima_distributions(
    maxima, 
    results_dict=results, 
    colors_dict=portfolio_colors, 
    show_gev_pdf=False
)
plt.show()

fig_qq = create_combined_qq_plots(maxima, results, portfolio_colors)
plt.show()


fig_return = create_combined_return_level_plots(results, portfolio_colors)
plt.show()






def create_enhanced_comparative_table(results_dict):
    """
    Crée un tableau comparatif amélioré des paramètres GEV avec intervalles de confiance et p-values.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionnaire des résultats de l'analyse GEV pour chaque portefeuille
        
    Returns:
    --------
    pd.DataFrame
        Tableau comparatif des paramètres avec intervalles de confiance et p-values
    """
    
    params_data = {}
    
    for name, results in results_dict.items():
        params = results['gev_params']
        
        
        z_value = 1.96  
        
        xi = params['xi']
        xi_se = params['xi_se']
        xi_lower = xi - z_value * xi_se
        xi_upper = xi + z_value * xi_se
        xi_tstat = xi / xi_se
        xi_pval = 2 * (1 - stats.norm.cdf(abs(xi_tstat)))  
        
        varpi = params['varpi']
        varpi_se = params['varpi_se']
        varpi_lower = varpi - z_value * varpi_se
        varpi_upper = varpi + z_value * varpi_se
        varpi_tstat = varpi / varpi_se
        varpi_pval = 2 * (1 - stats.norm.cdf(abs(varpi_tstat)))  
        
        psi = params['psi']
        psi_se = params['psi_se']
        psi_lower = psi - z_value * psi_se
        psi_upper = psi + z_value * psi_se
        psi_tstat = psi / psi_se
        psi_pval = 2 * (1 - stats.norm.cdf(abs(psi_tstat)))  
        
        
        params_data[name] = {
            'ξ': f"{xi:.4f}",
            'ξ_IC': f"[{xi_lower:.4f}, {xi_upper:.4f}]",
            'ϖ': f"{varpi:.4f}",
            'ϖ_IC': f"[{varpi_lower:.4f}, {varpi_upper:.4f}]",
            'ψ': f"{psi:.4f}",
            'ψ_IC': f"[{psi_lower:.4f}, {psi_upper:.4f}]",
        }
    
    
    df = pd.DataFrame(params_data).T
    df.index.name = 'Portfolio'

    
    df = df[['ξ', 'ξ_IC', 'ϖ', 'ϖ_IC', 'ψ', 'ψ_IC']]
    
    return df


def create_comprehensive_gev_table(results_dict, return_periods=[0.25, 0.5, 1, 2, 5, 10, 15, 20], block_size=60):
    """
    Crée un tableau complet des paramètres et résultats GEV pour les annexes.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionnaire des résultats de l'analyse GEV pour chaque portefeuille
    return_periods : list
        Liste des périodes de retour (en années) à inclure
    block_size : int
        Taille des blocs utilisés
        
    Returns:
    --------
    pd.DataFrame
        Tableau complet des paramètres et résultats
    """
    
    portfolios = list(results_dict.keys())
    
    
    
    all_dates = []
    for name in maxima.keys():
        all_dates.extend(maxima[name].index.tolist())
    
    min_date = min(all_dates)
    max_date = max(all_dates)
    date_range_years = (max_date - min_date).days / 365.25
    
    
    data = []
    
    for name in portfolios:
        params = results_dict[name]['gev_params']
        
        
        xi_tstat = params['xi'] / params['xi_se']
        varpi_tstat = params['varpi'] / params['varpi_se']
        psi_tstat = params['psi'] / params['psi_se']
        
        xi_pval = 2 * (1 - stats.norm.cdf(abs(xi_tstat)))
        varpi_pval = 2 * (1 - stats.norm.cdf(abs(varpi_tstat)))
        psi_pval = 2 * (1 - stats.norm.cdf(abs(psi_tstat)))
        
        
        if params['xi'] > 0:
            tail_type = "Heavy tail (Fréchet)"
        elif params['xi'] < 0:
            tail_type = "Bounded tail (Weibull)"
        else:
            tail_type = "Light tail (Gumbel)"
        
        
        n_blocks = len(maxima[name])
        
        
        row = {
            'Portfolio': name,
            'Period covered (years)': f"{date_range_years:.2f}",  # Utiliser la période commune
            'Number of blocks': n_blocks,
            'ξ (xi)': params['xi'],
            'Standard error ξ': params['xi_se'],
            't-stat ξ': xi_tstat,
            'ϖ (varpi)': params['varpi'],
            'Standard error ϖ': params['varpi_se'],
            't-stat ϖ': varpi_tstat,
            'ψ (psi)': params['psi'],
            'Standard error ψ': params['psi_se'],
            't-stat ψ': psi_tstat,
            'Tail type': tail_type
        }
        
        
        for period in return_periods:
            return_level = calculate_return_level(params, period, block_size)
            row[f"RL_{period}"] = return_level
        
        data.append(row)
    
    
    df = pd.DataFrame(data)
    
    return df

def interpret_gev_results(results_dict, maxima_dict, block_size=60):
    """
    Fournit une interprétation détaillée des résultats GEV en tenant compte de la dimension temporelle.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionnaire des résultats de l'analyse GEV pour chaque portefeuille
    maxima_dict : dict
        Dictionnaire des séries de maxima par bloc pour chaque portefeuille
    block_size : int
        Taille des blocs utilisés
        
    Returns:
    --------
    str
        Texte d'interprétation des résultats
    """
    interpretation = []
    interpretation.append("# Interprétation des résultats de l'analyse GEV")
    
    
    all_dates = []
    for series in maxima_dict.values():
        all_dates.extend(series.index.tolist())
    
    min_date = min(all_dates)
    max_date = max(all_dates)
    date_range_years = (max_date - min_date).days / 365.25

    interpretation.append(f"\n## Observation period")
    interpretation.append(f"- Start date: {min_date.strftime('%Y-%m-%d')}")
    interpretation.append(f"- End date: {max_date.strftime('%Y-%m-%d')}")
    interpretation.append(f"- Total duration: {date_range_years:.2f} years")
    interpretation.append(f"- Block size: {block_size} trading days")
    interpretation.append(f"- Approximate equivalent: {block_size/21:.1f} months per block")
    
    
    
    interpretation.append(f"\n## Blocks analyzed by portfolio")
    for name, series in maxima_dict.items():
        interpretation.append(f"- {name}: {len(series)} blocks")
    
    
    interpretation.append(f"\n## Shape parameter analysis (ξ)")
    interpretation.append("The parameter ξ determines the tail behavior of the distribution:")
    interpretation.append("- ξ > 0: Heavy tail (Fréchet type), extreme risks more likely")
    interpretation.append("- ξ = 0: Light tail (Gumbel type)")
    interpretation.append("- ξ < 0: Bounded tail (Weibull type), existence of an upper bound for losses")
    
    for name, results in results_dict.items():
        params = results['gev_params']
        xi = params['xi']
        xi_se = params['xi_se']
        
        interpretation.append(f"\n### {name}")
        interpretation.append(f"- ξ = {xi:.4f} ± {xi_se:.4f}")
        
        if xi > 0:
            if xi > 2*xi_se:  
                interpretation.append("- **Significant heavy tail**: Extreme events are more likely than expected under a normal distribution.")
                interpretation.append(f"- This portfolio presents a high risk of extreme events.")
            else:
                interpretation.append("- Tendency towards heavy tail, but not statistically significant.")
        elif xi < 0:
            if xi < -2*xi_se:  
                interpretation.append("- **Significant bounded tail**: Losses have a theoretical upper bound.")
                interpretation.append(f"- This portfolio presents a limited risk of very large extreme events.")
            else:
                interpretation.append("- Tendency towards bounded tail, but not statistically significant.")
        else:
            interpretation.append("- Light tail (Gumbel type).")
    
    
    interpretation.append(f"\n## Return period interpretation")
    interpretation.append("A return period of T years means that an event of this magnitude has a probability of 1/T of occurring in any given year.")
    interpretation.append("For example, an event with a 10-year return period has a probability of 1/10 = 0.1 (10%) of occurring in a year.")
    
    interpretation.append("\nImportant: Even though our sample covers only 23 years (2001-2024), extreme value theory allows us to extrapolate beyond this period.")
    interpretation.append("However, uncertainty increases for return period estimates much longer than our sample.")
    
    
    interpretation.append(f"\n## Return levels for a 10-year period")
    for name, results in results_dict.items():
        params = results['gev_params']
        return_level_10y = calculate_return_level(params, 10, block_size)
        interpretation.append(f"- {name}: {return_level_10y:.4f}")
        
        
        if 'Dynamic' in name:
            interp_text = "This level represents the maximum loss expected once every 10 years for the dynamic allocation strategy."
        else:
            interp_text = "This level represents the maximum loss expected once every 10 years for the static allocation strategy."
            
        interpretation.append(f"  - {interp_text}")
    
    return "\n".join(interpretation)







enhanced_table = create_enhanced_comparative_table(results)
print(enhanced_table.T)

comprehensive_table = create_comprehensive_gev_table(results)
comprehensive_table.set_index('Portfolio', inplace=True)




print(comprehensive_table.T)





def create_enhanced_comparative_table_latex(results_dict):
    """
    Crée un tableau comparatif amélioré des paramètres GEV avec intervalles de confiance,
    formaté pour LaTeX.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionnaire des résultats de l'analyse GEV pour chaque portefeuille
        
    Returns:
    --------
    pd.DataFrame
        Tableau comparatif des paramètres avec intervalles de confiance formaté pour LaTeX
    """
    
    params_data = {}
    
    for name, results in results_dict.items():
        params = results['gev_params']
        
        
        latex_name = name
        if "Static (λ=2)" in name:
            latex_name = "Static$_{\\lambda=2}$"
        elif "Static (λ=10)" in name:
            latex_name = "Static$_{\\lambda=10}$"
        elif "Dynamic (λ=2)" in name:
            latex_name = "Dynamic$_{\\lambda=2}$"
        elif "Dynamic (λ=10)" in name:
            latex_name = "Dynamic$_{\\lambda=10}$"
        
        
        z_value = 1.96
        
        xi = params['xi']
        xi_se = params['xi_se']
        xi_lower = xi - z_value * xi_se
        xi_upper = xi + z_value * xi_se
        
        varpi = params['varpi']
        varpi_se = params['varpi_se']
        varpi_lower = varpi - z_value * varpi_se
        varpi_upper = varpi + z_value * varpi_se
        
        psi = params['psi']
        psi_se = params['psi_se']
        psi_lower = psi - z_value * psi_se
        psi_upper = psi + z_value * psi_se
        
        
        params_data[latex_name] = {
            '$\\xi$': f"{xi:.4f}",
            '$\\xi$ 95\\% CI': f"[{xi_lower:.4f}, {xi_upper:.4f}]",
            '$\\varpi$': f"{varpi:.4f}",
            '$\\varpi$ 95\\% CI': f"[{varpi_lower:.4f}, {varpi_upper:.4f}]",
            '$\\psi$': f"{psi:.4f}",
            '$\\psi$ 95\\% CI': f"[{psi_lower:.4f}, {psi_upper:.4f}]"
        }
    
    
    df = pd.DataFrame(params_data).T
    df.index.name = 'Portfolio'
    
    
    df = df[['$\\xi$', '$\\xi$ 95\\% CI', '$\\varpi$', '$\\varpi$ 95\\% CI', '$\\psi$', '$\\psi$ 95\\% CI']]
    
    return df

def create_comprehensive_gev_table_latex(results_dict, return_periods=[0.25, 0.5, 1, 2, 5, 10, 15, 20], block_size=60):
    """
    Crée un tableau complet des paramètres et résultats GEV pour les annexes,
    formaté pour LaTeX.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionnaire des résultats de l'analyse GEV pour chaque portefeuille
    return_periods : list
        Liste des périodes de retour (en années) à inclure
    block_size : int
        Taille des blocs utilisés
        
    Returns:
    --------
    pd.DataFrame
        Tableau complet des paramètres et résultats formaté pour LaTeX
    """
    
    data = []
    
    for name in results_dict.keys():
        params = results_dict[name]['gev_params']
        
        
        latex_name = name
        if "Static (λ=2)" in name:
            latex_name = "Static$_{\\lambda=2}$"
        elif "Static (λ=10)" in name:
            latex_name = "Static$_{\\lambda=10}$"
        elif "Dynamic (λ=2)" in name:
            latex_name = "Dynamic$_{\\lambda=2}$"
        elif "Dynamic (λ=10)" in name:
            latex_name = "Dynamic$_{\\lambda=10}$"
        
        
        if params['xi'] > 0:
            tail_type = "Heavy tail (Fr\\'echet)"
        elif params['xi'] < 0:
            tail_type = "Bounded tail (Weibull)"
        else:
            tail_type = "Light tail (Gumbel)"
        
        
        n_blocks = len(maxima[name])
        years_covered = n_blocks * block_size / 252
        
        
        row = {
            'Portfolio': latex_name,
            'Period (years)': f"{years_covered:.2f}",
            'Blocks': n_blocks,
            '$\\xi$': f"{params['xi']:.4f}",
            'SE$_{\\xi}$': f"{params['xi_se']:.4f}",
            't-stat$_{\\xi}$': f"{params['xi'] / params['xi_se']:.4f}",
            '$\\varpi$': f"{params['varpi']:.4f}",
            'SE$_{\\varpi}$': f"{params['varpi_se']:.4f}",
            't-stat$_{\\varpi}$': f"{params['varpi'] / params['varpi_se']:.4f}",
            '$\\psi$': f"{params['psi']:.4f}",
            'SE$_{\\psi}$': f"{params['psi_se']:.4f}",
            't-stat$_{\\psi}$': f"{params['psi'] / params['psi_se']:.4f}",
            'Tail type': tail_type
        }
        
        
        for period in return_periods:
            return_level = calculate_return_level(params, period, block_size)
            row[f"RL$_{{{period}}}$"] = f"{return_level:.4f}"
        
        data.append(row)
    
    
    df = pd.DataFrame(data)
    
    
    column_mapping = {
        'Period (years)': 'Period (years)',
        'Blocks': 'Blocks',
        '$\\xi$': '$\\xi$',
        'SE$_{\\xi}$': 'SE$_{\\xi}$',
        't-stat$_{\\xi}$': 't-stat$_{\\xi}$',
        '$\\varpi$': '$\\varpi$',
        'SE$_{\\varpi}$': 'SE$_{\\varpi}$',
        't-stat$_{\\varpi}$': 't-stat$_{\\varpi}$',
        '$\\psi$': '$\\psi$',
        'SE$_{\\psi}$': 'SE$_{\\psi}$',
        't-stat$_{\\psi}$': 't-stat$_{\\psi}$',
        'Tail type': 'Tail type'
    }
    
    
    df.rename(columns=column_mapping, inplace=True)
    
    return df






enhanced_table_latex = create_enhanced_comparative_table_latex(results)
comprehensive_table_latex = create_comprehensive_gev_table_latex(results)


comprehensive_table_latex.set_index('Portfolio', inplace=True)













print("=== DIAGNOSTIC: Observations and blocks for GEV analysis ===")
print(f"Static strategy (λ=2): {len(maxima['Static (λ=2)'])} blocks")
print(f"Static strategy (λ=10): {len(maxima['Static (λ=10)'])} blocks")
print(f"Dynamic strategy (λ=2): {len(maxima['Dynamic (λ=2)'])} blocks")
print(f"Dynamic strategy (λ=10): {len(maxima['Dynamic (λ=10)'])} blocks")


block_size = 60  
print("\n=== Estimated number of days (based on block size) ===")
print(f"Static strategy (λ=2): {len(maxima['Static (λ=2)']) * block_size} days")
print(f"Static strategy (λ=10): {len(maxima['Static (λ=10)']) * block_size} days") 
print(f"Dynamic strategy (λ=2): {len(maxima['Dynamic (λ=2)']) * block_size} days")
print(f"Dynamic strategy (λ=10): {len(maxima['Dynamic (λ=10)']) * block_size} days")





print('# ---------------------- Q 4.4  ---------------------------------------- #')






def analyze_gev_quantiles(results_dict, portfolio_name, block_size=60, alpha=0.99):
    """
    Calcule les quantiles à 99% pour la distribution des maxima et en déduit
    l'information correspondante pour les résidus standardisés.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionnaire des résultats GEV (comme obtenu dans Q4.3)
    portfolio_name : str
        Nom du portefeuille à analyser
    block_size : int
        Taille des blocs utilisée dans l'analyse GEV
    alpha : float
        Niveau du quantile (par défaut 0.99)
    """
    
    params = results_dict[portfolio_name]['gev_params']
    xi = params['xi']
    varpi = params['varpi']
    psi = params['psi']
    
    
    gev_dist = genextreme(c=-xi, loc=varpi, scale=psi)
    
    
    q_max_99 = gev_dist.ppf(alpha)
    
    
    
    
    
    
    
    p_residual = alpha**(1/block_size)
    
    
    
    q_z_99 = gev_dist.ppf(alpha**block_size)  
    
    
    print(f"\nGEV Analysis for {portfolio_name}:")
    print(f"Quantile at {alpha*100:.0f}% for block maxima: {q_max_99:.4f}")
    print(f"This threshold corresponds to the {p_residual*100:.6f}% quantile of standardized residuals")
    print(f"In other words: P(Z ≤ {q_max_99:.4f}) = {p_residual:.6f}")
    print(f"Quantile at {alpha*100:.0f}% of standardized residuals: {q_z_99:.4f}")
    
    return {
        'q_max_99': q_max_99,
        'p_residual': p_residual,
        'q_z_99': q_z_99
    }


quantiles_results = {}
for name in results:
    quantiles_results[name] = analyze_gev_quantiles(results, name)



df_quantiles = pd.DataFrame({
    name: {
        'Quantile 99% of maxima': res['q_max_99'],
        'Corresponding probability in residuals': res['p_residual'],
        'Quantile 99% of standardized residuals': res['q_z_99']
    } 
    for name, res in quantiles_results.items()
}).T

print("\nComparative table of quantiles:")
print(df_quantiles)






df_quantiles_latex = df_quantiles.copy()


df_quantiles_latex.index = [
    'Static ($\\lambda=2$)',
    'Static ($\\lambda=10$)',
    'Dynamic ($\\lambda=2$)',
    'Dynamic ($\\lambda=10$)'
]


df_quantiles_latex = df_quantiles_latex.rename(columns={
    'Quantile 99% of maxima': '$q_{m_{\\tau}}^{99\\%}$',
    'Corresponding probability in residuals': '$p_{z}$',
    'Quantile 99% of standardized residuals': '$q_{z}^{99\\%}$'
})


df_formatted = pd.DataFrame(index=df_quantiles_latex.index)
df_formatted['$q_{m_{\\tau}}^{99\\%}$'] = df_quantiles_latex['$q_{m_{\\tau}}^{99\\%}$'].apply(lambda x: f"{x:.4f}")
df_formatted['$p_{z}$'] = df_quantiles_latex['$p_{z}$'].apply(lambda x: f"{x*100:.6f}\\%")
df_formatted['$q_{z}^{99\\%}$'] = df_quantiles_latex['$q_{z}^{99\\%}$'].apply(lambda x: f"{x:.4f}")


df_formatted_T = df_formatted.T


print(df_formatted_T)








print('# ---------------------- Q 4.5  ---------------------------------------- #')





def calculate_gev_var(model_results, gev_quantile_99, losses=None, portfolio_name="Portfolio"):
    """
    Calculate the GEV-based VaR using 1-day conditional forecasts and GEV quantiles
    
    Parameters:
    -----------
    model_results : dict
        Results from AR(1)-GARCH(1,1) model containing conditional mean and volatility
    gev_quantile_99 : float
        99% quantile of standardized residuals from GEV analysis
    losses : Series, optional
        Original loss series for comparison and plotting
    portfolio_name : str
        Portfolio name for plot labeling
        
    Returns:
    --------
    DataFrame
        Time series of GEV-based 99% VaR
    """
    mu_t = model_results['var']
    z_99 = stats.norm.ppf(0.99)
    sigma_t = (mu_t - model_results['a']) / z_99
    
    
    
    
    var_gev = model_results['a'] + sigma_t * abs(gev_quantile_99)
    
    
    var_comparison = pd.DataFrame({
        'GARCH_VaR_99%': mu_t,  
        'GEV_VaR_99%': var_gev,
        'Conditional_Mean': model_results['a'],
        'Conditional_Volatility': sigma_t
    })
    
    if losses is not None:
        common_dates = var_gev.index.intersection(losses.index)
        aligned_losses = losses.loc[common_dates]
        aligned_var = var_gev.loc[common_dates]
        
        exceedances = aligned_losses > aligned_var
        exceedance_rate = exceedances.mean()
        
        print(f"GEV VaR Backtest for {portfolio_name}:")
        print(f"Expected exceedance rate: 1.00%")
        print(f"Actual exceedance rate: {exceedance_rate:.2%}")
        print(f"Number of exceedances: {exceedances.sum()} out of {len(exceedances)}")
    
    return var_comparison


def plot_gev_var_comparison(var_comparison, losses=None, portfolio_name="Portfolio", 
                           sample_start=None, sample_end=None):
    """
    Create visualization of GEV-VaR compared with GARCH-VaR and actual losses
    
    Parameters:
    -----------
    var_comparison : DataFrame
        DataFrame containing GEV and GARCH VaR estimates
    losses : Series, optional
        Original loss series for comparison
    portfolio_name : str
        Portfolio name for plot labeling
    sample_start, sample_end : str, optional
        Date range for zoomed visualization
        
    Returns:
    --------
    Figure
        Matplotlib figure with the plot
    """
    
    plt.figure(figsize=(22, 10))
    
    
    if sample_start is not None and sample_end is not None:
        var_subset = var_comparison.loc[sample_start:sample_end].copy()
        if losses is not None:
            losses_subset = losses.loc[sample_start:sample_end].copy()
        else:
            losses_subset = None
    else:
        var_subset = var_comparison.copy()
        losses_subset = losses
    
    
    plt.fill_between(
        var_subset.index,
        0,
        var_subset['Conditional_Volatility'] * 5,  # Scaled for visibility
        color='lightgray', alpha=0.3, label='Conditional Volatility (scaled)'
    )
    
    
    plt.plot(var_subset.index, var_subset['GARCH_VaR_99%'], 
             color='blue', linestyle='--', linewidth=1.5, 
             label='GARCH VaR (99%)')
    
    plt.plot(var_subset.index, var_subset['GEV_VaR_99%'], 
             color='red', linewidth=2, 
             label='GEV VaR (99%)')
    
    
    if losses_subset is not None:
        plt.scatter(
            losses_subset.index, losses_subset, 
            color='gray', alpha=0.5, s=10, 
            label='Realized Losses'
        )
        
        
        common_dates = losses_subset.index.intersection(var_subset.index)
        gev_exceedances = losses_subset.loc[common_dates] > var_subset.loc[common_dates, 'GEV_VaR_99%']
        garch_exceedances = losses_subset.loc[common_dates] > var_subset.loc[common_dates, 'GARCH_VaR_99%']
        
        
        plt.scatter(
            common_dates[gev_exceedances], 
            losses_subset.loc[common_dates][gev_exceedances],
            color='red', s=50, marker='x',
            label='GEV VaR Exceedances'
        )
        
    
    plt.title(f"GEV-Based VaR vs GARCH VaR: {portfolio_name}", fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    
    
    plt.legend(loc='upper left', framealpha=0.7)
    
    plt.tight_layout()
    return plt.gcf()



var_static_lambda2 = calculate_gev_var(
    resS2, 
    quantiles_results['Static (λ=2)']['q_z_99'], 
    static_losses_lambda2,
    "Static (λ=2)"
)


var_static_lambda10 = calculate_gev_var(
    resS10, 
    quantiles_results['Static (λ=10)']['q_z_99'], 
    static_losses_lambda10,
    "Static (λ=10)"
)


var_dynamic_lambda2 = calculate_gev_var(
    resD2, 
    quantiles_results['Dynamic (λ=2)']['q_z_99'], 
    dynamic_losses_lambda2,
    "Dynamic (λ=2)"
)


var_dynamic_lambda10 = calculate_gev_var(
    resD10, 
    quantiles_results['Dynamic (λ=10)']['q_z_99'], 
    dynamic_losses_lambda10,
    "Dynamic (λ=10)"
)
















def create_var_comparison_table(portfolios):
    """
    Create a summary table comparing the three VaR approaches
    
    Parameters:
    -----------
    portfolios : list of dicts
        List of dictionaries containing portfolio name, unconditional VaR, and VaR comparison DataFrame
        
    Returns:
    --------
    DataFrame
        Summary comparison table
    """
    results = []
    
    for p in portfolios:
        name = p['name']
        uncond_var = p['uncond_var']
        var_comp = p['var_comp']
        losses = p['losses']
        
        
        avg_garch_var = var_comp['GARCH_VaR_99%'].mean()
        avg_gev_var = var_comp['GEV_VaR_99%'].mean()
        
        
        common_dates = losses.index.intersection(var_comp.index)
        uncond_exceedances = (losses.loc[common_dates] > uncond_var).mean()
        garch_exceedances = (losses.loc[common_dates] > var_comp.loc[common_dates, 'GARCH_VaR_99%']).mean()
        gev_exceedances = (losses.loc[common_dates] > var_comp.loc[common_dates, 'GEV_VaR_99%']).mean()
        
        results.append({
            'Portfolio': name,
            'Unconditional VaR': uncond_var,
            'Avg GARCH VaR': avg_garch_var,
            'Avg GEV VaR': avg_gev_var,
            'Uncond Exceedance %': uncond_exceedances * 100,
            'GARCH Exceedance %': garch_exceedances * 100,
            'GEV Exceedance %': gev_exceedances * 100
        })
    
    return pd.DataFrame(results)



var_static_lambda2_uncond = calculate_unconditional_var(static_losses_lambda2)
var_static_lambda10_uncond = calculate_unconditional_var(static_losses_lambda10)
var_dynamic_lambda2_uncond = calculate_unconditional_var(dynamic_losses_lambda2)
var_dynamic_lambda10_uncond = calculate_unconditional_var(dynamic_losses_lambda10)


portfolios_data = [
    {
        'name': 'Static (λ=2)',
        'uncond_var': var_static_lambda2_uncond['var_unconditional'],
        'var_comp': var_static_lambda2,
        'losses': static_losses_lambda2
    },
    {
        'name': 'Static (λ=10)',
        'uncond_var': var_static_lambda10_uncond['var_unconditional'],
        'var_comp': var_static_lambda10,
        'losses': static_losses_lambda10
    },
    {
        'name': 'Dynamic (λ=2)',
        'uncond_var': var_dynamic_lambda2_uncond['var_unconditional'],
        'var_comp': var_dynamic_lambda2,
        'losses': dynamic_losses_lambda2
    },
    {
        'name': 'Dynamic (λ=10)',
        'uncond_var': var_dynamic_lambda10_uncond['var_unconditional'],
        'var_comp': var_dynamic_lambda10,
        'losses': dynamic_losses_lambda10
    }
]
var_summary_table = create_var_comparison_table(portfolios_data)
print("\nVaR Approach Comparison:")
print(var_summary_table)




plt.show()





def create_combined_gev_var_plot(var_data_list, losses_list, names_list):
    """
    Crée un graphique combiné avec 4 sous-graphiques montrant la VaR GEV vs pertes réalisées
    """
    
    colors = {
        "Static (λ=2)": "#1f77b4",     # bleu
        "Static (λ=10)": "#ff7f0e",    # orange
        "Dynamic (λ=2)": "#2ca02c",    # vert
        "Dynamic (λ=10)": "#d62728"    # rouge
    }
    
    
    fig, axes = plt.subplots(2, 2, figsize=(22, 12), dpi=50)
    axes = axes.flatten()
    
    for i, (var_data, losses, name) in enumerate(zip(var_data_list, losses_list, names_list)):
        ax = axes[i]
        color = colors.get(name, 'blue')
        
        
        ax.plot(var_data.index, var_data['GEV_VaR_99%'], 
                color=color, linewidth=1.5, label='GEV VaR (99%)')
        
        
        common_dates = losses.index.intersection(var_data.index)
        ax.scatter(common_dates, losses.loc[common_dates], 
                  color='gray', alpha=0.4, s=8, label='Realized Losses')
        
        
        violations = losses.loc[common_dates] > var_data.loc[common_dates, 'GEV_VaR_99%']
        ax.scatter(common_dates[violations], losses.loc[common_dates][violations],
                  color='black', s=40, marker='x', label='VaR Violations')
        
        
        violation_rate = violations.mean() * 100
        violation_count = violations.sum()
        ax.text(0.02, 0.93, f'Violation rate: {violation_rate:.2f}%\n'
                         f'Violations: {violation_count} out of {len(common_dates)}',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        
        
        ax.set_title(f'{name}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Value')
        if i >= 2:  
            ax.set_xlabel('Date')
        
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        
        
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=10, framealpha=0.7)
    
    
    fig.suptitle('GEV-Based 99% VaR vs. Realized Losses', fontsize=16, fontweight='bold', y=0.98)
    
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    
    return fig


var_data_list = [var_static_lambda2, var_static_lambda10, var_dynamic_lambda2, var_dynamic_lambda10]
losses_list = [static_losses_lambda2, static_losses_lambda10, dynamic_losses_lambda2, dynamic_losses_lambda10]
names_list = ['Static (λ=2)', 'Static (λ=10)', 'Dynamic (λ=2)', 'Dynamic (λ=10)']


combined_var_fig = create_combined_gev_var_plot(var_data_list, losses_list, names_list)
plt.show()




print('# ---------------------- Q 4.6  ---------------------------------------- #\n')

def create_detailed_backtest_table(portfolios_data):
    """
    Crée un tableau de backtest détaillé en utilisant le même nombre de jours 
    pour toutes les stratégies afin d'assurer une comparaison équitable.
    """
    
    common_dates = None
    for p in portfolios_data:
        losses = p['losses']
        var_comp = p['var_comp']
        dates = losses.index.intersection(var_comp.index)
        
        if common_dates is None:
            common_dates = dates
        else:
            common_dates = common_dates.intersection(dates)

    print(f"Common days in all the strategies: {len(common_dates)}")

    
    rows = []
    
    for p in portfolios_data:
        name = p['name']
        uncond_var = p['uncond_var']
        var_comp = p['var_comp']
        losses = p['losses']
        
        
        losses_common = losses.loc[common_dates]
        var_comp_common = var_comp.loc[common_dates]
        
        
        uncond_exceedances = losses_common > uncond_var
        uncond_rate = uncond_exceedances.mean() * 100
        uncond_count = uncond_exceedances.sum()
        
        
        garch_exceedances = losses_common > var_comp_common['GARCH_VaR_99%']
        garch_rate = garch_exceedances.mean() * 100
        garch_count = garch_exceedances.sum()
        
        
        gev_exceedances = losses_common > var_comp_common['GEV_VaR_99%']
        gev_rate = gev_exceedances.mean() * 100
        gev_count = gev_exceedances.sum()
        
        
        rows.append({
            'Portfolio': name,
            'Method': 'Uncond',
            'Exceedance %': uncond_rate,
            '# Violations': uncond_count,
            'Deviation from 1%': uncond_rate - 1.0,
            'Total Observations': len(common_dates)
        })
        
        rows.append({
            'Portfolio': name,
            'Method': 'GARCH',
            'Exceedance %': garch_rate,
            '# Violations': garch_count,
            'Deviation from 1%': garch_rate - 1.0,
            'Total Observations': len(common_dates)
        })
        
        rows.append({
            'Portfolio': name,
            'Method': 'GEV',
            'Exceedance %': gev_rate,
            '# Violations': gev_count,
            'Deviation from 1%': gev_rate - 1.0,
            'Total Observations': len(common_dates)
        })
    
    
    backtest_df = pd.DataFrame(rows)
    
    
    backtest_df['Exceedance %'] = backtest_df['Exceedance %'].round(2)
    backtest_df['Deviation from 1%'] = backtest_df['Deviation from 1%'].round(2)
    
    return backtest_df


backtest_table = create_detailed_backtest_table(portfolios_data)
backtest_table.set_index(['Portfolio', 'Method'], inplace=True)


print("\nTableau de backtest VaR:")
print(backtest_table)






def create_var_comparison_barplot(backtest_table):
    """
    Creates a professional grouped bar plot comparing VaR exceedance rates
    of different methods by portfolio, with a 1% reference line.
    """
    
    if isinstance(backtest_table.index, pd.MultiIndex):
        df = backtest_table.reset_index()
    else:
        df = backtest_table.copy()
    
    
    plot_data = df.pivot(index='Portfolio', columns='Method', values='Exceedance %')
    
    
    if all(method in plot_data.columns for method in ['Uncond', 'GARCH', 'GEV']):
        plot_data = plot_data[['Uncond', 'GARCH', 'GEV']]
    
    
    plt.figure(figsize=(22, 10), dpi=50)
    
    
    colors = {
        'Uncond': '#34495e',  # Dark blue-gray
        'GARCH': '#c0392b',   # Darker red
        'GEV': '#27ae60'      # Emerald green
    }
    
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.2
    
    
    portfolios = plot_data.index
    n_portfolios = len(portfolios)
    width = 0.22  
    indices = np.arange(n_portfolios)
    
    
    for i, method in enumerate(plot_data.columns):
        positions = indices + (i - 1) * width
        plt.bar(positions, plot_data[method], width, 
                label=method, color=colors.get(method, f'C{i}'),
                alpha=0.85, edgecolor='#333333', linewidth=0.8,
                zorder=3)
    
    
    plt.axhline(y=1.0, color='#333333', linestyle='--', linewidth=1.8, 
                label='Target (1%)', zorder=2)
    
    
    arrow_color = '#27ae60'  # Green color for arrow
    for i, portfolio in enumerate(portfolios):
        
        deviations = np.abs(plot_data.loc[portfolio] - 1.0)
        best_method = deviations.idxmin()
        best_method_idx = list(plot_data.columns).index(best_method)
        
        
        position = i + (best_method_idx - 1) * width
        value = plot_data.loc[portfolio, best_method]
        
        
        plt.annotate('',
            xy=(position, value+0.20),  
            xytext=(position, value + 0.75),  
            arrowprops=dict(
                facecolor=arrow_color,
                shrink=0.35,
                width=1,
                headwidth=10,
                headlength=5,
                edgecolor='#333333',
                linewidth=1.0
            ),
            zorder=5
        )
    
    plt.xticks(indices, portfolios, rotation=0, fontsize=20, fontweight='bold', y=-0.02)
    
    
    for i, method in enumerate(plot_data.columns):
        positions = indices + (i - 1) * width
        for j, value in enumerate(plot_data[method]):
            deviation = value - 1.0
            deviation_text = f"+{deviation:.2f}" if deviation > 0 else f"{deviation:.2f}"
            
            
            if abs(deviation) <= 0.2:  
                color = 'darkgreen'
            elif abs(deviation) <= 0.5:  
                color = 'darkorange'
            else:  
                color = 'darkred'
            
            
            plt.annotate(
                f"{value:.2f}%\n({deviation_text})",
                xy=(positions[j], value + 0.05),
                xytext=(0, 5),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=16,
                fontweight='bold',
                color=color,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85)
            )
    
    
    plt.ylabel('Exceedance Rate (%)', fontsize=14, fontweight='bold')
    plt.title('Comparison of Value-at-Risk Methods: Exceedance Rates (99% Confidence Level)', 
              fontsize=18, fontweight='bold', pad=20)
    
    
    legend = plt.legend(title='VaR Method', loc='upper right', 
               bbox_to_anchor=(0.99, 0.99), frameon=True,
               fancybox=True, shadow=True, fontsize=12)
    legend.get_title().set_fontweight('bold')
    
    
    plt.grid(axis='y', linestyle='--', alpha=0.3, zorder=1)
    
    
    explanation = "Green arrow indicates the method closest to the target rate of 1%\n" + \
                 "Numbers in parentheses show deviation from ideal 1% rate"
    
    plt.figtext(0.05, 0.85, explanation, fontsize=16, 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='#333333', 
                         boxstyle='round,pad=0.5'),
                ha='left', va='bottom')
    
    
    plt.ylim(0, plot_data.values.max() + 0.7)
    
    plt.tight_layout()
    
    return plt.gcf()


fig = create_var_comparison_barplot(backtest_table)




plt.show()