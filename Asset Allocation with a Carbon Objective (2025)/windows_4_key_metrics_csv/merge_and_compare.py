#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


m48 = pd.read_csv('48.csv')
m60 = pd.read_csv('60.csv')
m72 = pd.read_csv('72.csv')
m84 = pd.read_csv('84.csv')


# In[3]:


m48 = pd.DataFrame(m48)
m60 = pd.DataFrame(m60)
m72 = pd.DataFrame(m72)
m84 = pd.DataFrame(m84)


# In[5]:


print(m48, m60, m72, m84)


# In[ ]:


def merge_and_format_dataframes(df_dict):
    """
    Fusionne plusieurs DataFrames en un seul avec une structure multi-index en colonnes.
    Les portefeuilles sont le premier niveau et les fenêtres temporelles le second niveau.
    
    Parameters:
    -----------
    df_dict : dict
        Dictionnaire avec les noms des fenêtres comme clés et les DataFrames comme valeurs
        
    Returns:
    --------
    pd.DataFrame
        DataFrame fusionné avec structure multi-index
    """
    # Liste pour stocker les DataFrames traités
    processed_dfs = []
    
    # Traitement de chaque DataFrame
    for window, df in df_dict.items():
        # Copie pour éviter de modifier l'original
        temp_df = df.copy()
        
        # Définir la colonne 'Unnamed: 0' comme index et la renommer 'Metric'
        temp_df.set_index('Unnamed: 0', inplace=True)
        temp_df.index.name = 'Metric'
        
        # Créer un multi-index en colonnes avec le type de portefeuille et la fenêtre temporelle
        # Cette fois avec le portefeuille comme premier niveau et la fenêtre comme second niveau
        temp_df.columns = pd.MultiIndex.from_product([temp_df.columns, [window]])
        
        # Ajouter à la liste
        processed_dfs.append(temp_df)
    
    # Concaténer les DataFrames
    result = pd.concat(processed_dfs, axis=1)
    
    # Réorganiser pour avoir les colonnes regroupées par portefeuille
    # Les portefeuilles seront le premier niveau et les fenêtres le second niveau
    result = result.sort_index(axis=1, level=0)
    
    return result

def export_to_latex(df, filename=None, caption=None, label=None):
    """
    Exporte le DataFrame en format LaTeX avec des options de formatage améliorées.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame à exporter
    filename : str, optional
        Nom du fichier de sortie
    caption : str, optional
        Légende pour le tableau LaTeX
    label : str, optional
        Étiquette pour référencement dans LaTeX
    
    Returns:
    --------
    str
        Code LaTeX formaté
    """
    latex_code = df.to_latex(
        float_format="%.2f",  # Format à 2 décimales
        multicolumn=True,     # Activer les multi-colonnes
        multicolumn_format='c',  # Centrer les en-têtes de multi-colonnes
        bold_rows=False,      # Ne pas mettre les lignes en gras
        longtable=False,      # Utiliser table standard au lieu de longtable
        escape=False,         # Ne pas échapper les caractères spéciaux LaTeX
        na_rep="-"            # Représenter NaN par un tiret
    )
    
    # Amélioration du formatage LaTeX
    if caption:
        caption_line = f"\\caption{{{caption}}}\n"
        latex_code = latex_code.replace("\\begin{tabular}", f"\\begin{{table}}\n\\centering\n{caption_line}\\begin{{tabular}}")
        latex_code = latex_code.replace("\\end{tabular}", "\\end{tabular}\n\\end{table}")
    
    if label:
        label_line = f"\\label{{{label}}}\n"
        if "\\caption" in latex_code:
            latex_code = latex_code.replace("\\caption", f"\\caption{{{caption}}}\n{label_line}")
        else:
            latex_code = latex_code.replace("\\begin{tabular}", f"\\begin{{table}}\n\\centering\n{label_line}\\begin{{tabular}}")
            latex_code = latex_code.replace("\\end{tabular}", "\\end{tabular}\n\\end{table}")
    
    # Ajouter des lignes horizontales entre les sections
    latex_code = latex_code.replace("\\bottomrule", "\\midrule\n\\bottomrule")
    
    # Écrire dans un fichier si demandé
    if filename:
        with open(filename, 'w') as f:
            f.write(latex_code)
        print(f"Tableau exporté vers {filename}")
    
    return latex_code


window_dfs = {
    '48m': m48,
    '60m': m60,
    '72m': m72,
    '84m': m84
}


merged_df = merge_and_format_dataframes(window_dfs)


print("DataFrame fusionné avec structure multi-index:")
print(merged_df)


# latex_output = export_to_latex(
#     merged_df,
#     filename='Article_Latex/comparison_table_4window.tex',
#     caption='Comparaison des métriques de portefeuille selon différentes fenêtres temporelles',
#     label='tab:portfolio_comparison'
# )


# In[15]:


portfolio_extremes = {}
portfolios = ['MVP', 'VW', 'MVP50', 'TE', 'NZ']
windows = ['48m', '60m', '72m', '84m']

metrics_mapping = {
    'Return': 'Rendement (%)',
    'Volatility': 'Volatilité (%)',
    'Sharpe': 'Ratio de Sharpe Annualisé',
    'Footprint': 'Réduction Empreinte Carbone (%)',
    'TE': 'TE Annualisé (%)'  # Ajout du tracking error
}

for portfolio in portfolios:
    portfolio_data = {}
    for metric_name, metric_df in metrics_mapping.items():
        values = []
        for window in windows:
            try:
                # Extraire la valeur pour ce portefeuille/fenêtre/métrique
                value = merged_df.loc[metric_df, (portfolio, window)]
                if pd.notna(value):  # Ignorer les NaN
                    values.append(value)
            except (KeyError, ValueError):
                continue
        
        if values:  # Si nous avons des valeurs valides
            # Calculer (min + max)/2 au lieu de la moyenne arithmétique
            portfolio_data[metric_name] = (max(values) - min(values))
        else:
            portfolio_data[metric_name] = None
    
    portfolio_extremes[portfolio] = portfolio_data

# Créer le DataFrame de comparaison par rapport à VW avec des différences absolues
vw_reference = portfolio_extremes['VW']
portfolio_names = {
    'MVP': '$P_{oos}^{(mv)}$',
    'MVP50': '$P_{oos}^{(mv)}(0.5)$', 
    'NZ': '$P_{oos}^{(vw)}(\\mathrm{NZ})$',
    'TE': '$P_{oos}^{(vw)}(0.5)$',
    'VW': '$P^{(vw)}$'
}


comparison_data = []
for portfolio in portfolios:
    row = {
        'Portfolio': portfolio_names[portfolio],
        '$\\Delta$ Return (pp)': abs(portfolio_extremes[portfolio]['Return'] - vw_reference['Return']) if (portfolio_extremes[portfolio]['Return'] is not None and vw_reference['Return'] is not None) else '-',
        '$\\Delta$ Volatility (pp)': abs(portfolio_extremes[portfolio]['Volatility'] - vw_reference['Volatility']) if (portfolio_extremes[portfolio]['Volatility'] is not None and vw_reference['Volatility'] is not None) else '-',
        '$\\Delta$ Sharpe': abs(portfolio_extremes[portfolio]['Sharpe'] - vw_reference['Sharpe']) if (portfolio_extremes[portfolio]['Sharpe'] is not None and vw_reference['Sharpe'] is not None) else '-',
        '$\\Delta$ Footprint (\\%)': portfolio_extremes[portfolio]['Footprint'] if portfolio_extremes[portfolio]['Footprint'] is not None else '-',
        'TE (pp)': portfolio_extremes[portfolio]['TE'] if portfolio_extremes[portfolio]['TE'] is not None else '-'
    }
    comparison_data.append(row)


comparison_df = pd.DataFrame(comparison_data)

latex_output = comparison_df.to_latex(
    index=False,
    escape=False,
    float_format="%.2f",  # Format à 2 décimales
    column_format="lrrrrr",  # 5 colonnes maintenant au lieu de 4
    na_rep="-"
)


# with open('Article_Latex/portfolio_extremes_comparison.tex', 'w') as f:
#     f.write(latex_output)


print(comparison_df)

