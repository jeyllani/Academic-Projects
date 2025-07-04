# To convert this script to a Jupyter notebook, install jupytext and run:
# jupytext --to notebook GP_script.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import seaborn as sns
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
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import minimize
import sys 
import os 
import pickle
import tqdm as tqdm
plt.rcParams['figure.figsize'] = (22, 8)
pd.set_option('display.max_columns', None)
sns.set_theme(style="darkgrid") 



use_pickle = True                   # <<<- Set to True to import and save portfolio weights from disk (recommended).
                                    # If True and no folder exists: the script will create a directory called 'pickle_weights', 
                                    # with a subfolder for each lookback window (e.g., 'pickle_weights/72/').
                                    # This allows the script to load precomputed weights if available, or save new ones as you run the pipeline.
                                    # If False: the script recalculates all weights from scratch and does not save them.
                                    #
                                    # Precomputed weights for L = (48, 60, 72, 84) are provided to avoid lengthy computations.
                                    # (All were obtained on a Mac M2 Pro, using VSCode/CPU.)
                                    #
                                    # (We faced and solved a reproducibility issue with the weights: 
                                    # the optimiser was run on unsorted lists, leading to non-deterministic results.
                                    # All lists are now sorted before optimisation, ensuring full reproducibility.)
                                    #
                                    # Note: All objective function values (for the main case L=72) are multiplied by 10,000x for numerical stability.
                                    # On shorter windows (L=48, 60), 10,000x may cause the optimiser to fail; try 1,000x instead if needed.
                                    #
                                    # Recommendation: keep use_pickle = True to save time and preserve results.

                                    
# Choose lookback window for weight storage/loading ('48', '60', '72', '84')
pickle_weights_window = '72'        # Must correspond to subfolder in 'pickle_weights/'


save_all_metrics_in_current_directory_csv = False  # <<<- Set to True to save all metrics in the current directory as CSV files.


save_images = None


# In[136]:


# Get the current directory
current_directory = os.getcwd()
print(f"Current directory: {current_directory}")

if use_pickle:
    # Define the paths of the folders to create
    pickle_dir = os.path.join(current_directory, "pickle_weights")
    window_dir = os.path.join(pickle_dir, pickle_weights_window)
    
    # Create the main folder pickle_weights if it does not exist
    if not os.path.exists(pickle_dir):
        os.makedirs(pickle_dir)
        print(f"Folder created: {pickle_dir}")
    else:
        print(f"The folder {pickle_dir} already exists")
    
    # Create the subfolder with the name of the time window
    if not os.path.exists(window_dir):
        os.makedirs(window_dir)
        print(f"Subfolder created: {window_dir}")
    else:
        print(f"The subfolder {window_dir} already exists")
    
    print(f"Complete folder structure: {window_dir}")
else:
    print("Folder creation is not enabled (use_pickle = False)")


# In[137]:


data_DS_MV_T_USD_M  = 'Data/DS_MV_T_USD_M.xlsx'
data_DS_MV_T_USD_Y  = 'Data/DS_MV_T_USD_Y.xlsx'
data_DS_REV_USD_Y   = 'Data/DS_REV_USD_Y.xlsx'
data_DS_RI_T_USD_M  = 'Data/DS_RI_T_USD_M.xlsx'
data_DS_RI_T_USD_Y  = 'Data/DS_RI_T_USD_Y.xlsx'
data_rf             = 'Data/Risk_Free_Rate.xlsx'
data_Scope_1        = 'Data/Scope_1.xlsx'
data_Scope_2        = 'Data/Scope_2.xlsx'
data_Static         = 'Data/Static.xlsx'


mcap                = pd.read_excel(data_DS_MV_T_USD_M)
ycap                = pd.read_excel(data_DS_MV_T_USD_Y)
yrevenue            = pd.read_excel(data_DS_REV_USD_Y)
mprice              = pd.read_excel(data_DS_RI_T_USD_M)
yprice              = pd.read_excel(data_DS_RI_T_USD_Y)
rf                  = pd.read_excel(data_rf)
scope1              = pd.read_excel(data_Scope_1)
scope2              = pd.read_excel(data_Scope_2)
src                 = pd.read_excel(data_Static)

# src['Region'].unique() -> array(['AMER', 'EM', 'EUR', 'PAC', 'Other', 'FRON', nan], dtype=object)
# (Group P = North America = 'AMER') + SCOPE1 and SCOPE2
src = src[src['Region'] == 'AMER']



rf['Date']      = pd.to_datetime(rf['Unnamed: 0'].astype(str), format ='%Y%m')
rf              = rf.set_index('Date')
rf.drop(columns =['Unnamed: 0'], inplace=True)
rf.index.name   = None



mcap                  = mcap[mcap['ISIN'].isin(src['ISIN'])]
ycap                  = ycap[ycap['ISIN'].isin(src['ISIN'])]
yrevenue              = yrevenue[yrevenue['ISIN'].isin(src['ISIN'])]
mprice                = mprice[mprice['ISIN'].isin(src['ISIN'])]
yprice                = yprice[yprice['ISIN'].isin(src['ISIN'])]
scope1                = scope1[scope1['ISIN'].isin(src['ISIN'])]
scope2                = scope2[scope2['ISIN'].isin(src['ISIN'])]


mcap.drop(columns     =['NAME'], inplace =True)
ycap.drop(columns     =['NAME'], inplace =True)
yrevenue.drop(columns =['NAME'], inplace =True)
mprice.drop(columns   =['NAME'], inplace =True)
yprice.drop(columns   =['NAME'], inplace =True)
scope1.drop(columns   =['NAME'], inplace =True)
scope2.drop(columns   =['NAME'], inplace =True)

# Set the index to 'ISIN' for all DataFrames
mcap.set_index(['ISIN']     , inplace =True)
ycap.set_index(['ISIN']     , inplace =True)
yrevenue.set_index(['ISIN'] , inplace =True)
mprice.set_index(['ISIN']   , inplace =True)
yprice.set_index(['ISIN']   , inplace =True)
scope1.set_index(['ISIN']   , inplace =True)
scope2.set_index(['ISIN']   , inplace =True)


mcap      = mcap.T.sort_index(ascending     =True)
ycap      = ycap.T.sort_index(ascending     =True)
yrevenue  = yrevenue.T.sort_index(ascending =True)
mprice    = mprice.T.sort_index(ascending   =True)
yprice    = yprice.T.sort_index(ascending   =True)
scope1    = scope1.T.sort_index(ascending   =True)
scope2    = scope2.T.sort_index(ascending   =True)


mcap      = mcap.apply(pd.to_numeric, errors='coerce')
ycap      = ycap.apply(pd.to_numeric, errors='coerce')
yrevenue  = yrevenue.apply(pd.to_numeric, errors='coerce')
mprice    = mprice.apply(pd.to_numeric, errors='coerce')
yprice    = yprice.apply(pd.to_numeric, errors='coerce')
scope1    = scope1.apply(pd.to_numeric, errors='coerce')
scope2    = scope2.apply(pd.to_numeric, errors='coerce')

mcap.index      = pd.to_datetime(mcap.index)
mprice.index    = pd.to_datetime(mprice.index)
yrevenue.index  = pd.to_datetime(yrevenue.index , format    ='%Y').to_period('Y')
ycap.index      = pd.to_datetime(ycap.index ,     format    ='%Y').to_period('Y')
yprice.index    = pd.to_datetime(yprice.index ,   format    ='%Y').to_period('Y')
scope1.index    = pd.to_datetime(scope1.index ,   format    ='%Y').to_period('Y')
scope2.index    = pd.to_datetime(scope2.index ,   format    ='%Y').to_period('Y')
rf.index        = pd.to_datetime(rf.index)


# In[138]:


print('# ------------------------ 0.0  Data processing begin -------------------------- #')


# In[139]:


def process_interior_missing_values(df: pd.DataFrame) -> dict[str]:
    """
    Processes only the missing values located between the first and last non-missing values of each column.

    For each column:
      - Identifies the first and last index with a non-missing value.
      - Examines the portion of the series between these two bounds.
      - Detects consecutive NaN sequences within this interval.
      - If a sequence contains more than one NaN, its details are added to 'targeted'.
      - If a sequence contains exactly one NaN, it interpolates that point.

    Returns a dictionary with:
      - "interpolated_df": The modified DataFrame with interpolation applied to isolated gaps in the interior.
      - "targeted": A dictionary where keys are column names and values are lists of tuples 
                    (start, end, length) for each sequence of consecutive NaNs (length > 1).
      - "interpolated_columns": A list of columns where interpolation was applied (isolated gaps).

    Example usage:
    --------------
    ```data = {
        'A': [5, None, 7, None, 9, 10],         # A series with two isolated gaps in the interior
        'B': [3, 4, None, None, 8, 9],          # A sequence of 2 consecutive NaNs in the interior -> targeted
        'C': [None, 2, 3, 4, 5, None],          # Missing values at the start and end are ignored
        'D': [1 , 2 , 3 , 4 , 5 , 6]            # No NaNs in the interior
    }
    df_example = pd.DataFrame(data)
    
    result = process_interior_missing_values(df_example)

    # Results: DataFrame after interior interpolation (only isolated gaps are interpolated):
          A    B    C  D
    0   5.0  3.0  NaN  1
    1   6.0  4.0  2.0  2
    2   7.0  NaN  3.0  3
    3   8.0  NaN  4.0  4
    4   9.0  8.0  5.0  5
    5  10.0  9.0  NaN  6

    # Targeted columns (with gaps of more than one NaN in the interior) and sequence details:
    {'B': [(2, 4, 2)]}

    # Columns where interpolation was applied (isolated gaps):
    ['A']
    ```
    """

    # Copie du DataFrame pour appliquer l'interpolation sur l'intérieur sans toucher aux extrémités
    df_interpolated = df.copy()
    
    targeted = {}           # Dictionnaire pour stocker les colonnes avec gap > 1 dans l'intérieur
    interpolated_columns = []  # Liste des colonnes où interpolation a été appliquée
    
    # Parcours de chaque colonne
    for col in df.columns:
        series = df[col]
        
        # Trouver le premier et le dernier index où la valeur n'est pas NaN
        first_valid = series.first_valid_index()
        last_valid = series.last_valid_index()
        
        # Si la colonne est entièrement NaN ou ne possède qu'une seule valeur valide, on passe à la suivante
        if first_valid is None or last_valid is None or first_valid == last_valid:
            continue
        
        # On se restreint à la portion intérieure (entre la première et la dernière valeur valide)
        interior_series = series.loc[first_valid:last_valid]
        
        # Analyse des séquences de NaN dans l'intérieur
        sequences = []  # Liste des séquences sous forme de tuple (start_index, end_index, length)
        current_seq_start = None
        current_seq_length = 0
        
        for idx, val in interior_series.items():
            if pd.isna(val):
                if current_seq_start is None:
                    current_seq_start = idx
                    current_seq_length = 1
                else:
                    current_seq_length += 1
            else:
                # Si une séquence était en cours, on la sauvegarde
                if current_seq_start is not None:
                    sequences.append((current_seq_start, idx, current_seq_length))
                    current_seq_start = None
                    current_seq_length = 0
        # Au cas où la séquence se terminerait à la fin de l'intervalle (ce qui ne devrait pas arriver puisque last_valid est non NaN)
        if current_seq_start is not None:
            sequences.append((current_seq_start, interior_series.index[-1], current_seq_length))
        
        # Variables de contrôle pour savoir s'il y a des gaps multiples ou seulement des gaps isolés
        gap_multiple = False
        
        # Parcours de chaque séquence identifiée
        for seq in sequences:
            start_seq, end_seq, length = seq
            if length > 1:
                gap_multiple = True
                # Enregistrer la séquence pour le traitement ciblé
                if col not in targeted:
                    targeted[col] = []
                targeted[col].append(seq)
        
        # Si dans cette colonne, on n'a trouvé que des gaps isolés (au moins un gap présent) dans l'intérieur
        if not gap_multiple and len(sequences) > 0:
            # Interpolation dans la tranche intérieure uniquement
            df_interpolated.loc[first_valid:last_valid, col] = \
                df_interpolated.loc[first_valid:last_valid, col].ffill()
            interpolated_columns.append(col)
    if not targeted.keys():
        print("   1. No multiple gaps (NaN>1) found in the interior of the time series.", '\n')
        print("   2. ffill() applied to isolated gaps in the interior of the time series.")
        print("")
    else:
        print("   Multiple gaps (NaN>1) found in the interior:")
        print(targeted)
    return {
        "interpolated_df": df_interpolated,
        "targeted": targeted,
        "interpolated_columns": interpolated_columns
    }

# Ensure df is assigned a valid DataFrame before calling the function
print('mprice')
missings = process_interior_missing_values(mprice)
print('yprice')
missings_ = process_interior_missing_values(yprice)
# ---------------------------------------------------------------------------- #
mprice = missings['interpolated_df'].copy()
yprice = missings_['interpolated_df'].copy()


# In[140]:


def find_columns_with_data_then_nan(df, head_years=5, tail_years=3):
# This function answer the question can we use yrevenue, yrpice and ycap in the impute missing data without afffecting it 
    """
    Find columns that have data at the beginning but have NaN values toward the end.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with time series data
    head_years : int
        Number of initial years to check for data presence
    tail_years : int
        Number of final years to check for NaN presence
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the columns that match the criteria
    """
    selected_cols = []
    
    for col in df.columns:
        # Check if there's at least one valid value in the first few years
        has_data_at_start = df[col].head(head_years).notna().any()
        
        # Check if there's at least one NaN in the last few years
        has_nan_at_end = df[col].tail(tail_years).isna().any()
        
        if has_data_at_start and has_nan_at_end:
            selected_cols.append(col)
    
    return df[selected_cols]

# Find columns with data at the beginning but NaN values at the end
cols_with_pattern_ycap = find_columns_with_data_then_nan(ycap)
cols_with_pattern_yrevenue = find_columns_with_data_then_nan(yrevenue)
cols_with_pattern_yprice = find_columns_with_data_then_nan(yprice)

# Display the number of columns found and show a sample
print('Inspection')
print(f" Found {len(cols_with_pattern_ycap.columns)} columns with - ycap - data at beginning and NaN values at end")
print(f" Found {len(cols_with_pattern_yrevenue.columns)} columns with - yrevenue - data at beginning and NaN values at end")
print(f" Found {len(cols_with_pattern_yprice.columns)} columns with - yprice - data at beginning and NaN values at end")


# In[141]:


print(' -> yprice excluded for now')
dataframes_to_impute = {
    'scope1': scope1,
    'scope2': scope2,
    'yrevenue': yrevenue,
    'ycap': ycap
}


# In[142]:


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remplace les valeurs manquantes selon des règles spécifiques:
    
    1. Pour les valeurs NaN entre deux valeurs valides (intérieures):
       - Utiliser la fonction process_interior_missing_values (ffill)
       
    2. Pour les séquences à la fin de l'index:
       - Si la dernière valeur est NaN et précédée d'une valeur non-NaN,
         imputer avec la dernière valeur non-NaN connue
    
    Args:
        df: DataFrame source avec potentiellement des valeurs manquantes
        
    Returns:
        DataFrame avec les valeurs manquantes imputées selon les règles spécifiées
    """
    # Étape 1: Traiter les valeurs NaN intérieures avec process_interior_missing_values
    result = process_interior_missing_values(df)
    df_imputed = result['interpolated_df']
    
    # Étape 2: Traiter les valeurs NaN à la fin de l'index
    for column in df.columns:
        series = df_imputed[column]
        
        # Ignorer les colonnes sans NaN
        if not series.isna().any():
            continue
            
        # Identifier le dernier indice valide
        last_valid_idx = series.last_valid_index()
        
        # Ignorer les colonnes entièrement NaN
        if last_valid_idx is None:
            continue
        
        # Vérifier s'il y a des NaN à la fin (après le dernier indice valide)
        if last_valid_idx < series.index[-1]:
            # Extraire la séquence à la fin
            end_sequence = series.loc[series.index > last_valid_idx]
            
            # Si toutes les valeurs de fin sont NaN, les remplacer par la dernière valeur valide
            if end_sequence.isna().all():
                last_valid_value = series[last_valid_idx]
                df_imputed.loc[series.index > last_valid_idx, column] = last_valid_value
    
    return df_imputed

def process_yearly_data(filtered_dataframes: dict) -> dict:
    """
    Traite les DataFrames filtrés pour chaque année en imputant les valeurs manquantes.
    
    Parameters
    ----------
    filtered_dataframes : dict
        Dictionnaire de DataFrames filtrés, avec des clés descriptives comme 'scope1', 'scope2', etc.
    
    Returns
    -------
    dict
        Dictionnaire avec les mêmes clés que filtered_dataframes, mais où chaque DataFrame
        a été traité pour imputer les valeurs manquantes.
    """
    processed_dataframes = {}
    
    for df_name, df in filtered_dataframes.items():
        # Imputer les valeurs manquantes
        processed_df = impute_missing_values(df)
        
        # Stocker le DataFrame traité
        processed_dataframes[df_name] = processed_df
        
        # Afficher des informations sur le traitement
        print(f" Yearly DataFrame '{df_name}' processed: {len(df.columns)} columns -> {len(processed_df.columns)} columns")
    
    return processed_dataframes

print('Processing on yearly data following described criterion in the Project Preamble!\n')
processed = process_yearly_data(dataframes_to_impute)

processed['yprice'] = yprice


# In[143]:


def check_nan_values(dataframes: dict) -> list:
# Si nous n'avions pas supprimé les colonnes dans la cellule 2 elles apparaîteraient ici.
    """
    Checks for columns with all NaN values in multiple DataFrames and returns a list of these columns.

    Parameters
    ----------
    dataframes : dict
        Dictionary with DataFrame names as keys and pandas DataFrames as values.

    Returns
    -------
    list
        A list containing the names of columns that have all NaN values across all DataFrames.
    """
    all_nan_cols = []
    for key, df in dataframes.items():
        nan_cols = df.columns[df.isna().all()].tolist()  # Convert Index to list
        print(f" DataFrame: {key}")
        if nan_cols:
            print(" Columns with all NaN values:")
            print(nan_cols)
            all_nan_cols.extend(nan_cols)
        else:
            print(" No columns with all NaN values.")
        print("", "-" * 30)

    # Remove duplicates by converting to a set and back to a list
    unique_nan_cols = list(set(all_nan_cols))
    return unique_nan_cols
print(' Specific columns containing NaN values in Yearly Dataframes \n\n')
# Example usage:
nan_columns = check_nan_values(processed)


# In[144]:


def drop_nan_columns(processed_dict, nan_columns, additional_dfs=None):
    """
    Supprime les colonnes contenant uniquement des NaN de tous les DataFrames.
    
    Parameters
    ----------
    processed_dict : dict
        Dictionnaire contenant les DataFrames à traiter (scope1, scope2, etc.)
    nan_columns : list
        Liste des colonnes à supprimer (généralement obtenue via check_nan_values)
    additional_dfs : dict, optional
        Dictionnaire supplémentaire contenant d'autres DataFrames à traiter (mcap, mprice, etc.)
        
    Returns
    -------
    tuple
        (processed_dict mis à jour, additional_dfs mis à jour, summary_df)
        summary_df est un DataFrame résumant les modifications avec les colonnes:
        - Variable: nom du DataFrame
        - Colonnes initiales: nombre de colonnes avant traitement
        - Colonnes finales: nombre de colonnes après traitement
        - Colonnes supprimées: nombre de colonnes supprimées
    """
    # Traitement des DataFrames dans le dictionnaire principal
    cleaned_dict = {}
    
    # Liste pour stocker les informations sur les modifications
    summary_data = []
    
    for key, df in processed_dict.items():
        # Filtrer pour ne garder que les colonnes qui existent dans ce DataFrame
        cols_to_drop = [col for col in nan_columns if col in df.columns]
        if cols_to_drop:
            cleaned_df = df.drop(columns=cols_to_drop)
            print(f"DataFrame '{key}': {len(df.columns)} colonnes -> {len(cleaned_df.columns)} colonnes (-{len(cols_to_drop)})")
        else:
            cleaned_df = df
            print(f"DataFrame '{key}': Aucune colonne à supprimer")
        
        cleaned_dict[key] = cleaned_df
        
        # Ajouter les informations de modification à notre liste pour le résumé
        summary_data.append({
            'Variable': key,
            'Initial Companies': len(df.columns),
            'Final Companies': len(cleaned_df.columns),
            'Excluded Companies': len(df.columns) - len(cleaned_df.columns)
        })
    
    # Traitement des DataFrames supplémentaires
    cleaned_additional = {}
    if additional_dfs:
        for key, df in additional_dfs.items():
            cols_to_drop = [col for col in nan_columns if col in df.columns]
            if cols_to_drop:
                cleaned_df = df.drop(columns=cols_to_drop)
                print(f"DataFrame '{key}': {len(df.columns)} colonnes -> {len(cleaned_df.columns)} colonnes (-{len(cols_to_drop)})")
            else:
                cleaned_df = df
                print(f"DataFrame '{key}': Aucune colonne à supprimer")
                
            cleaned_additional[key] = cleaned_df
            
            # Ajouter également ces DataFrames au résumé
            summary_data.append({
                'Variable': key,
                'Initial Companies': len(df.columns),
                'Final Companies': len(cleaned_df.columns),
                'Excluded Companies': len(df.columns) - len(cleaned_df.columns)
            })
    
    # Créer le DataFrame résumant les modifications
    import pandas as pd
    summary_df = pd.DataFrame(summary_data)
    
    # Calculer les totaux
    total_row = {
        'Variable': 'Total',
        'Initial Companies': summary_df['Initial Companies'].sum(),
        'Final Companies': summary_df['Final Companies'].sum(),
        'Excluded Companies': summary_df['Excluded Companies'].sum()
    }
    summary_df = pd.concat([summary_df, pd.DataFrame([total_row])], ignore_index=True)
    
    return cleaned_dict, cleaned_additional, summary_df


# In[145]:


# Créer un dictionnaire pour les DataFrames additionnels
additional_dataframes = {
    'mcap': mcap,
    'mprice': mprice
}

print(" Exclusion of th columns containing only NaN from all Dataframes\n")
# Appliquer la fonction
cleaned_processed, cleaned_additional, summary_cleaning0 = drop_nan_columns(processed, nan_columns, additional_dataframes)

# Récupérer les DataFrames nettoyés
scope1_cleaned    = cleaned_processed['scope1']
scope2_cleaned    = cleaned_processed['scope2']
yrevenue_cleaned  = cleaned_processed['yrevenue']
ycap_cleaned      = cleaned_processed['ycap']
yprice_cleaned    = cleaned_processed['yprice']
mcap_cleaned      = cleaned_additional['mcap']
mprice_cleaned    = cleaned_additional['mprice']

# Mise à jour des variables originales si nécessaire
scope1    = scope1_cleaned
scope2    = scope2_cleaned
yrevenue  = yrevenue_cleaned
ycap      = ycap_cleaned
yprice    = yprice_cleaned
mcap      = mcap_cleaned
mprice    = mprice_cleaned

# Mise à jour du dictionnaire processed
processed = cleaned_processed


# In[146]:


raw_companies = summary_cleaning0['Initial Companies']
first_process = companies = summary_cleaning0['Final Companies']


# In[147]:


def filter_complete_data_companies(scope1, scope2, yrevenue, ycap, years=None):
    """
    Filters companies with complete data (non-NaN values) for scope1, scope2, revenue, 
    and market capitalization across specified years ycap.
    
    This function identifies companies that have valid carbon emission data (scope1 and scope2), 
    revenue data, and market capitalization data for each specified year. Companies are excluded 
    if they have any missing values or zero market capitalization (to avoid division by zero 
    in subsequent calculations like carbon intensity metrics).
    
    Parameters
    ----------
    scope1 : pandas.DataFrame
        DataFrame of scope1 carbon emissions with years as index (Period objects) 
        and companies (identified by ISIN) as columns.
    
    scope2 : pandas.DataFrame
        DataFrame of scope2 carbon emissions with years as index (Period objects)
        and companies (identified by ISIN) as columns.
    
    yrevenue : pandas.DataFrame
        DataFrame of annual company revenues with years as index (Period objects)
        and companies (identified by ISIN) as columns.
    
    ycap : pandas.DataFrame
        DataFrame of annual market capitalizations with years as index (Period objects)
        and companies (identified by ISIN) as columns.
    
    years : list, optional
        Specific list of years to process (format: Period objects or strings).
        If None, uses all years common to all DataFrames.
    
    Returns
    -------
    dict
        Dictionary with years (Period objects) as keys and lists of valid company ISINs as values.
        Example: {Period('2015'): ['US0378331005', 'US5949181045', ...], ...}
    
    dict
        Dictionary detailing excluded companies and reasons for each year.
        Structure: {
            Period('2015'): {
                'US1234567890': ['scope1_nan', 'scope2_nan'],  # Company excluded due to missing scope1 and scope2 data
                'US0987654321': ['ycap_zero'],                  # Company excluded due to zero market cap
                ...
            },
            ...
        }
        
    Examples
    --------
    >>> valid_companies, eliminated_info = filter_complete_data_companies(
    ...     scope1_df, scope2_df, revenue_df, market_cap_df, 
    ...     years=[pd.Period('2015'), pd.Period('2016')]
    ... )
    >>> print(f"Number of valid companies in 2015: {len(valid_companies[pd.Period('2015')])}")
    Number of valid companies in 2015: 320
    >>> print(f"Main reasons for elimination in 2016: {set(reason for reasons in eliminated_info[pd.Period('2016')].values() for reason in reasons)}")
    Main reasons for elimination in 2016: {'scope1_nan', 'scope2_nan', 'yrevenue_nan'}
    
    Notes
    -----
    - Companies are considered valid only if they have non-NaN values for all four data types
      (scope1, scope2, revenue, and market capitalization) and non-zero market capitalization.
    - The function tracks the specific reasons for exclusion for each company and year.
    - This function is particularly useful for preparing data for carbon intensity calculations
      and portfolio construction where complete data is required.
    """
    # Convertir les années en Period si elles sont fournies en str
    if years is not None:
        years = [pd.Period(year, freq='Y') if isinstance(year, str) else year for year in years]
    else:
        # Trouver les années communes à tous les DataFrames
        common_years = set(scope1.index) & set(scope2.index) & set(yrevenue.index) & set(ycap.index)
        years = sorted(common_years)
    
    valid_companies_by_year = {}
    eliminated_companies = {}
    
    for year in years:
        # S'assurer que l'année existe dans tous les DataFrames
        if year not in scope1.index or year not in scope2.index or \
           year not in yrevenue.index or year not in ycap.index:
            print(f"⚠️ Année {year}: Données manquantes dans au moins un DataFrame")
            continue
        
        # Obtenir toutes les entreprises disponibles pour cette année
        all_companies = set(scope1.columns) & set(scope2.columns) & set(yrevenue.columns) & set(ycap.columns)
        
        # Initialiser le dictionnaire pour suivre les raisons d'élimination
        eliminated_reasons = {company: [] for company in all_companies}
        
        # Vérifier les entreprises avec des NaN dans scope1 et scope2
        scope1_nan = set(scope1.loc[year][scope1.loc[year].isna()].index)
        scope2_nan = set(scope2.loc[year][scope2.loc[year].isna()].index)
        ycap_nan = set(ycap.loc[year][ycap.loc[year].isna()].index)
        yrevenue_nan = set(yrevenue.loc[year][yrevenue.loc[year].isna()].index)
        
        # Vérifier les entreprises avec capitalisation nulle (pour éviter division par zéro)
        zero_cap = set(ycap.loc[year][ycap.loc[year] == 0].index)
        
        # Ajouter les raisons d'élimination
        for company in scope1_nan:
            eliminated_reasons[company].append("scope1_nan")
        for company in scope2_nan:
            eliminated_reasons[company].append("scope2_nan")
        for company in ycap_nan:
            eliminated_reasons[company].append("ycap_nan")
        for company in yrevenue_nan:
            eliminated_reasons[company].append("yrevenue_nan")
        for company in zero_cap:
            eliminated_reasons[company].append("ycap_zero")
        
        # Identifier les entreprises valides (pas de NaN et pas de capitalisation nulle)
        invalid_companies = scope1_nan | scope2_nan | ycap_nan | yrevenue_nan | zero_cap
        valid_companies = all_companies - invalid_companies
        
        # Nettoyer les raisons pour ne garder que les entreprises éliminées
        eliminated_companies[year] = {
            company: reasons for company, reasons in eliminated_reasons.items() 
            if company in invalid_companies
        }
        
        valid_companies_by_year[year] = sorted(list(valid_companies))
        
        print(f"Year {year}: {len(valid_companies)} companies valid on {len(all_companies)} available")
        
    return valid_companies_by_year, eliminated_companies


def filter_and_organize_by_year(scope1, scope2, yrevenue, ycap, years=None, verbose=None):
    """
    Filters companies with complete data for each year and
    organizes data into a year-based structure.
    
    Parameters
    ----------
    scope1, scope2, yrevenue, ycap : pd.DataFrame
        DataFrames with years as index (Period) and companies (ISIN) as columns
    years : list, optional
        List of years to process (Format: Period, str or int)
        
    Returns
    -------
    dict
        Hierarchical structure:
        {
            '2010': {
                'scope1': Filtered DataFrame,
                'scope2': Filtered DataFrame,
                'yrevenue': Filtered DataFrame,
                'ycap': Filtered DataFrame,
                'valid_companies': list of valid companies
            },
            '2011': {
                ...
            },
            ...
        }
    dict
        Information about eliminated companies by year
    pd.DataFrame
        Summary statistics of filtering with columns:
        'Year', 'Valid Companies', 'Total Companies', 'Percentage'
    """
    # Get valid companies for each year
    valid_companies, eliminated_reasons = filter_complete_data_companies(
        scope1=scope1, scope2=scope2, yrevenue=yrevenue, ycap=ycap, years=years
    )
    
    # Initialize result structure
    annual_data = {}
    
    # Create DataFrame for summary statistics
    stats_data = []
    
    # For each year with valid data
    for year, companies in valid_companies.items():
        # Convert year to string for easier access
        year_str = str(year)
        
        # Get all available companies for this year (intersection of all datasets)
        # This matches the calculation in filter_complete_data_companies
        all_companies = set(scope1.columns) & set(scope2.columns) & set(yrevenue.columns) & set(ycap.columns)
        total_companies = len(all_companies)
        
        # Calculate percentage
        valid_count = len(companies)
        percentage = (valid_count / total_companies * 100) if total_companies > 0 else 0
        
        # Add stats to DataFrame
        stats_data.append({
            'Year': year_str,
            'Total': total_companies,
            'Valid': valid_count,
            'Percentage': round(percentage, 2)
        })
        
        # Select data for this year for valid companies
        annual_data[year_str] = {
            'scope1': scope1.loc[[year], companies],
            'scope2': scope2.loc[[year], companies],
            'yrevenue': yrevenue.loc[[year], companies],
            'ycap': ycap.loc[[year], companies],
            'valid_companies': companies
        }
    

    stats_df = pd.DataFrame(stats_data).set_index("Year")
    
    
    if verbose is not None:
        # Display detailed information about eliminated companies
        print("\n===== Elimination Reasons Analysis =====")
        for year in eliminated_reasons:
            if not eliminated_reasons[year]:  # Skip if no eliminations for this year
                continue
                
            year_str = str(year)
            total_eliminated = len(eliminated_reasons[year])
            
            print(f"\nYear {year_str}: {total_eliminated} companies excluded")
            
            # Count occurrences of each reason
            reason_counts = {}
            for company, reasons in eliminated_reasons[year].items():
                for reason in reasons:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
            
            # Display counts by reason
            print("Elimination reasons:")
            for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
                reason_desc = {
                    "scope1_nan": "Missing direct CO2 emissions data (Scope 1)",
                    "scope2_nan": "Missing indirect CO2 emissions data (Scope 2)",
                    "ycap_nan": "Missing market capitalization data",
                    "yrevenue_nan": "Missing revenue data",
                    "ycap_zero": "Zero market capitalization"
                }.get(reason, reason)
                
                percentage = (count / total_eliminated) * 100
                print(f"  • {reason_desc}: {count} companies ({percentage:.1f}%)")
    
    return annual_data, eliminated_reasons, stats_df


# In[148]:


# Appliquer le filtrage et organiser par année
print('Initial filtering based on annual variables\n')
annual_filtered_data, eliminated_info, filtering_stats = filter_and_organize_by_year(
    scope1=scope1_cleaned,
    scope2=scope2_cleaned,
    yrevenue=yrevenue_cleaned,
    ycap=ycap_cleaned,
    verbose=None,
)



# In[149]:


def generate_windows_with_filtered_companies(
    df: pd.DataFrame, 
    annual_filtered_data: dict,
    start_date: str = "2004-01-01", 
    end_date: str = "2024-12-31", 
    window_years: int = 11, 
    test_years: int = 1,
    min_months_available: int = 36,
    volatility_threshold: float = 3.0,
    ensure_complete_months: bool = True
) -> dict:
    """
    Génère des fenêtres glissantes pour le backtesting en appliquant un double filtrage :
    
    1. Filtre les entreprises disponibles dans annual_filtered_data pour l'année PRÉCÉDANT
       l'année de test (Y-1)
    2. Filtre supplémentaire basé sur l'historique de données et la volatilité

    Cette approche est plus réaliste pour un backtesting car elle n'utilise que les
    informations disponibles à la fin de la période d'estimation.
    
    Pour chaque année de test Y, cette fonction définit :
      - une fenêtre d'estimation allant de (Y - (window_years - test_years))-01-01 à (Y-1)-12-31
      - une période de test allant de Y-01-01 à Y-12-31
    
    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame des prix mensuels avec dates en index et entreprises (ISIN) en colonnes.
    annual_filtered_data : dict
        Dictionnaire avec les années comme clés et les données filtrées comme valeurs.
        Format: {'2014': {'scope1': pd.DataFrame, 'scope2': pd.DataFrame, ...}, ...}
    start_date : str, optional
        Date de début pour la génération des fenêtres, format 'YYYY-MM-DD'.
    end_date : str, optional
        Date de fin pour la génération des fenêtres, format 'YYYY-MM-DD'.
    window_years : int, optional
        Nombre total d'années pour chaque fenêtre (estimation + test).
    test_years : int, optional
        Nombre d'années pour la période de test.
    min_months_available : int, optional
        Nombre minimum de mois de données continues requis à la fin de la période d'estimation.
    volatility_threshold : float, optional
        Facteur k pour le filtrage par volatilité.
    ensure_complete_months : bool, optional
        Si True, s'assure que la fenêtre d'estimation commence au 1er jour du mois.
    
    Retourne
    -------
    dict
        Dictionnaire avec les années de test comme clés et les données de fenêtre comme valeurs.
    """
    # S'assurer que l'index est en datetime et trié
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    # Déduire le nombre d'années d'estimation
    estimation_years = window_years - test_years
    windows = {}
    
    # Définir la plage d'années de test
    first_test_year = pd.to_datetime(start_date).year + estimation_years
    last_test_year = pd.to_datetime(end_date).year
    
    for Y in range(first_test_year, last_test_year + 1):
        # Convertir Y en chaîne pour correspondre aux clés de annual_filtered_data
        test_year_str = str(Y)
        
        # L'année précédente (année d'estimation finale) pour le filtrage
        previous_year_str = str(Y - 1)
        
        # Vérifier si l'année précédente est disponible dans annual_filtered_data
        if previous_year_str not in annual_filtered_data:
            print(f"Année {previous_year_str} (précédant l'année de test {test_year_str}) non disponible dans annual_filtered_data, ignorée.")
            continue
        
        # Récupérer les entreprises éligibles depuis l'année PRÉCÉDANT l'année de test
        # Cela garantit que nous n'utilisons que les informations disponibles à la fin de l'année d'estimation
        sample_key = next(iter(annual_filtered_data[previous_year_str]))
        eligible_annual_companies = annual_filtered_data[previous_year_str]['valid_companies']
        
        if not eligible_annual_companies:
            print(f"Aucune entreprise éligible dans annual_filtered_data pour l'année {previous_year_str}, année de test {test_year_str} ignorée.")
            continue
        
        # Filtrer df pour ne contenir que les entreprises éligibles selon annual_filtered_data
        available_companies = [comp for comp in eligible_annual_companies if comp in df.columns]
        
        if not available_companies:
            print(f"Aucune entreprise de annual_filtered_data pour {previous_year_str} n'existe dans df, année de test {test_year_str} ignorée.")
            continue
            
        # Restreindre df aux entreprises disponibles
        df_eligible = df[available_companies]
        
        # Définition des bornes avec pd.Timestamp
        estimation_start = pd.Timestamp(f"{Y - estimation_years}-01-01")
        window_slice = pd.Timestamp(f"{Y - 1}-12-31")
        test_start = pd.Timestamp(f"{Y}-01-01")
        test_end = pd.Timestamp(f"{Y}-12-31")
        full_window_end = test_end
        
        # Si ensure_complete_months est True, s'assurer de commencer au 1er du mois
        if ensure_complete_months:
            available_dates = df_eligible.loc[df_eligible.index >= estimation_start].index
            if len(available_dates) > 0 and available_dates[0].day > 1:
                estimation_start = pd.Timestamp(f"{Y - estimation_years - 1}-12-01")
        
        # Extraire la tranche de données pour la période complète
        df_window = df_eligible.loc[(df_eligible.index >= estimation_start) & (df_eligible.index <= full_window_end)]
        if df_window.empty:
            print(f"Fenêtre vide pour l'année de test {test_year_str}, ignorée.")
            continue
        
        # Calculer les rendements sur cette tranche et retirer la première ligne NaN
        window_returns = df_window.pct_change(fill_method=None).iloc[1:]
        
        # Découper en deux DataFrames
        estimation_returns = window_returns.loc[window_returns.index <= window_slice]
        test_returns = window_returns.loc[(window_returns.index >= test_start) & (window_returns.index <= test_end)]
        
        if estimation_returns.empty or test_returns.empty:
            print(f"Période d'estimation ou de test vide pour l'année {test_year_str}, ignorée.")
            continue
        
        # Application des filtres supplémentaires sur les données
        final_eligible_companies = []
        excluded_companies = []
        bankruptcy_companies = []
        volatility_excluded_companies = []
        
        # Vérifier les contraintes pour chaque compagnie
        for col in estimation_returns.columns:
            column_data = estimation_returns[col]
            
            # Vérifier si l'entreprise a des données manquantes à la fin de la période d'estimation
            if pd.isna(column_data.iloc[-1]):
                bankruptcy_companies.append(col)
                excluded_companies.append(col)
                continue
            
            # Vérifier si les derniers 'min_months_available' mois sont disponibles en continu
            last_values = column_data.iloc[-min_months_available:] if len(column_data) >= min_months_available else column_data
            has_sufficient_continuous_data = not last_values.isna().any()
            
            if not has_sufficient_continuous_data or len(last_values) < min_months_available:
                excluded_companies.append(col)
                continue
            
            # Vérifier la volatilité
            clean_data = column_data.dropna()
            if len(clean_data) > 0:
                mean_return = clean_data.mean()
                std_return = clean_data.std()
                threshold = volatility_threshold * std_return
                
                if ((clean_data - mean_return).abs() > threshold).any():
                    volatility_excluded_companies.append(col)
                else:
                    final_eligible_companies.append(col)
            else:
                excluded_companies.append(col)
        
        # Filtrer les rendements sur les entreprises éligibles finales
        if not final_eligible_companies:
            print(f"Aucune entreprise ne satisfait tous les critères pour l'année de test {test_year_str}, ignorée.")
            continue
            
        estimation_returns = estimation_returns[final_eligible_companies]
        test_returns = test_returns[final_eligible_companies]
        
        # Calcul des entreprises avec au moins un NaN dans la période de test
        faillite_test_period = []
        for col in test_returns.columns:
            if test_returns[col].isna().any():
                faillite_test_period.append(col)
        
        # Combiner les faillites
        all_bankruptcy = list(set(bankruptcy_companies + faillite_test_period))
        
        # Ajouter les données de cette année au dictionnaire des fenêtres
        windows[test_year_str] = {
            'alpha_period': estimation_returns,
            'test_period': test_returns,
            'eligible_companies': final_eligible_companies,
            'excluded_companies': excluded_companies,
            'bankruptcy_companies': bankruptcy_companies,
            'faillite_test_period': faillite_test_period,
            'all_bankruptcy': all_bankruptcy,
            'volatility_excluded_companies': volatility_excluded_companies,
            'faillite_count': len(all_bankruptcy),
            'initial_eligible_annual': eligible_annual_companies,
            'final_vs_annual_ratio': len(final_eligible_companies) / len(eligible_annual_companies) if eligible_annual_companies else 0,
            'estimation_period': f"{estimation_start.strftime('%Y-%m-%d')} à {window_slice.strftime('%Y-%m-%d')}",
            'test_period_date': f"{test_start.strftime('%Y-%m-%d')} à {test_end.strftime('%Y-%m-%d')}",
            'annual_filter_year': previous_year_str  # On conserve l'année utilisée pour le filtrage
        }
    
    return windows


# In[150]:


def print_windows_summary(windows: dict, show_details: bool = False):
    """
    Displays a detailed and formatted summary for each test window generated with
    the generate_windows_with_filtered_companies function.
    
    Parameters
    ----------
    windows : dict
        Dictionary with test years as keys and window data as values.
    show_details : bool, optional
        Indicates whether detailed statistics should be displayed for each window.
            
    Returns
    -------
    None
        The function displays statistics but does not return any value.
    """
    import numpy as np
    
    if not windows:
        print("No windows found in the provided dictionary.")
        return
    
    # Prepare sorted test years
    test_years = sorted(windows.keys())
    total_years = len(test_years)
    
    # Prepare summary statistics
    all_stats = {
        'eligible_companies': [],
        'excluded_companies': [],
        'volatility_excluded_companies': [],
        'bankruptcy_companies': [],
        'faillite_test_period': [],
        'alpha_period_months': [],
        'test_period_months': [],
        'faillite_rates': [],
        'annual_filtered_companies': [],
        'final_vs_annual_ratio': []
    }
    
    # Header
    print("\n" + "="*85)
    print(" BACKTEST WINDOWS SUMMARY WITH FILTERING BASED ON HISTORIC AVAIlABILITY - {} PERIODS ".format(total_years).center(85))
    print("="*85 + "\n")
    
    # Process each year and display detailed information
    for i, test_year in enumerate(test_years):
        window = windows[test_year]
        alpha_period = window["alpha_period"]
        test_period = window["test_period"]
        eligible_count = len(window["eligible_companies"])
        excluded_count = len(window["excluded_companies"])
        volatility_excluded_count = len(window["volatility_excluded_companies"])
        bankruptcy_count = len(window.get("bankruptcy_companies", []))
        faillite_test_count = len(window.get("faillite_test_period", []))
        all_faillite_count = window.get("faillite_count", 0)
        annual_filter_year = window.get("annual_filter_year", "N/A")  # Year used for filtering
        
        # Annual filtering statistics
        annual_filtered_count = len(window.get("initial_eligible_annual", []))
        final_vs_annual_ratio = window.get("final_vs_annual_ratio", 0)
        
        # Collect statistics for the summary
        all_stats['eligible_companies'].append(eligible_count)
        all_stats['excluded_companies'].append(excluded_count)
        all_stats['volatility_excluded_companies'].append(volatility_excluded_count)
        all_stats['bankruptcy_companies'].append(bankruptcy_count)
        all_stats['faillite_test_period'].append(faillite_test_count)
        all_stats['annual_filtered_companies'].append(annual_filtered_count)
        all_stats['final_vs_annual_ratio'].append(final_vs_annual_ratio)
        
        # Format dates
        if not alpha_period.empty:
            alpha_start = alpha_period.index.min().strftime("%Y-%m-%d")
            alpha_end = alpha_period.index.max().strftime("%Y-%m-%d")
            alpha_months = len(alpha_period.index)
            all_stats['alpha_period_months'].append(alpha_months)
        else:
            alpha_start, alpha_end, alpha_months = "N/A", "N/A", 0
            all_stats['alpha_period_months'].append(0)
        
        if not test_period.empty:
            test_start = test_period.index.min().strftime("%Y-%m-%d")
            test_end = test_period.index.max().strftime("%Y-%m-%d")
            test_months = len(test_period.index)
            all_stats['test_period_months'].append(test_months)
        else:
            test_start, test_end, test_months = "N/A", "N/A", 0
            all_stats['test_period_months'].append(0)
        
        # Calculate bankruptcy rate
        if eligible_count > 0:
            faillite_rate = all_faillite_count / eligible_count
            all_stats['faillite_rates'].append(faillite_rate)
        else:
            faillite_rate = float('nan')
        
        # Display year header with progress
        progress = f"({i+1}/{total_years})"
        print("┌" + "─"*83 + "┐")
        header_text = f" TEST YEAR {test_year} (Filtering based on {annual_filter_year}) {progress:>{48-len(test_year)-len(annual_filter_year)}}"
        print("│" + header_text.ljust(83) + "│")
        print("└" + "─"*83 + "┘")
        
        # Basic information section
        print("\n● Time periods:")
        print(f"  Estimation window:  {alpha_start} → {alpha_end} ({alpha_months} months)")
        print(f"  Test period:        {test_start} → {test_end} ({test_months} months)")
        
        # Company statistics section
        print("\n● Company statistics:")
        print(f"  Companies filtered by {annual_filter_year} data:           {annual_filtered_count:,}")
        print(f"  Final eligible companies:                  {eligible_count:,}")
        print(f"  Eligible/annual filtered ratio:            {final_vs_annual_ratio:.2%}")
        print(f"  Excluded for missing data:                 {excluded_count:,}")
        print(f"  Excluded for high volatility:              {volatility_excluded_count:,}")
        print(f"  Bankruptcies pre-test period:              {bankruptcy_count:,}")
        print(f"  Bankruptcies during test period:           {faillite_test_count:,}")
        print(f"  Total bankruptcy events:                   {all_faillite_count:,}")
        print(f"  Bankruptcy rate:                           {faillite_rate:.2%}")
        
        # Additional details if requested
        if show_details:
            print("\n● Filtering details:")
            total_initial = eligible_count + excluded_count + volatility_excluded_count
            if total_initial > 0:
                print(f"  Initial universe after annual filtering:  {total_initial:,} companies")
                print(f"  Data quality filter:                      -{excluded_count:,} ({excluded_count/total_initial:.1%} of initial)")
                print(f"  Volatility filter:                        -{volatility_excluded_count:,} ({volatility_excluded_count/total_initial:.1%} of initial)")
                print(f"  Final universe:                           {eligible_count:,} ({eligible_count/total_initial:.1%} of initial)")
            
            if alpha_period.shape[1] > 0:
                nan_count = alpha_period.isna().sum().sum()
                total_cells = alpha_period.shape[0] * alpha_period.shape[1]
                if total_cells > 0:
                    print(f"  Missing values (alpha period):            {nan_count:,} ({nan_count/total_cells:.2%} of total points)")
        
        # Spacing between years
        print("\n" + "─"*85 + "\n")
    
    # Global summary section
    if total_years > 1:
        print("="*85)
        print(" GLOBAL SUMMARY ACROSS {} TEST PERIODS ".format(total_years).center(85))
        print("="*85 + "\n")
        
        # Calculate averages
        avg_eligible = sum(all_stats['eligible_companies']) / total_years
        avg_excluded = sum(all_stats['excluded_companies']) / total_years
        avg_vol_excluded = sum(all_stats['volatility_excluded_companies']) / total_years
        avg_annual_filtered = sum(all_stats['annual_filtered_companies']) / total_years
        avg_final_ratio = sum(all_stats['final_vs_annual_ratio']) / total_years
        avg_faillite_rate = sum(all_stats['faillite_rates']) / len(all_stats['faillite_rates']) if all_stats['faillite_rates'] else 0
        
        print("● Average statistics:")
        print(f"  Avg. companies filtered annually:      {avg_annual_filtered:.1f}")
        print(f"  Avg. final eligible companies:         {avg_eligible:.1f}")
        print(f"  Avg. eligible/filtered ratio:          {avg_final_ratio:.2%}")
        print(f"  Avg. excluded companies:               {avg_excluded:.1f}")
        print(f"  Avg. volatility exclusions:            {avg_vol_excluded:.1f}")
        print(f"  Avg. bankruptcy rate:                  {avg_faillite_rate:.2%}")
        
        print("\n● Extremes:")
        print(f"  Min/Max eligible companies:           {min(all_stats['eligible_companies']):,} / {max(all_stats['eligible_companies']):,}")
        if all_stats['faillite_rates']:
            print(f"  Min/Max bankruptcy rate:              {min(all_stats['faillite_rates']):.2%} / {max(all_stats['faillite_rates']):.2%}")
        
        # Calculate consistency metrics
        print("\n● Consistency metrics:")
        std_eligible = np.std(all_stats['eligible_companies']) if len(all_stats['eligible_companies']) > 1 else 0
        cv_eligible = std_eligible / avg_eligible if avg_eligible > 0 else float('nan')
        print(f"  Std. deviation of company count:       {std_eligible:.1f} (CV: {cv_eligible:.2f})")
    
    return


# In[151]:


windows_filtered = generate_windows_with_filtered_companies(
    df                     = mprice,  # DataFrame des prix mensuels
    annual_filtered_data   = annual_filtered_data, # Données annuelles filtrées
    start_date             = "2004-01-01",
    end_date               = "2024-12-31",
    window_years           = 11,
    test_years             = 1,
    min_months_available   = int(pickle_weights_window),
    volatility_threshold   = 15.0,
    ensure_complete_months = True
)

# # Afficher le résumé des fenêtres générées
print_windows_summary(windows_filtered, show_details=False)


# In[152]:


# print_windows_summary(windows_filtered, show_details=True)


# In[153]:


def process_bankruptcy_in_windows(windows: dict, verbose=None):
    """
    Traite les périodes de test (test_period) de chaque fenêtre en gérant les faillites 
    selon des règles spécifiques.
    
    Pour chaque entreprise dans la période de test, applique ces règles:
    1. Cas série uniquement de zéros : premier zéro devient -1 (faillite)
    2. Cas série uniquement de NaN : premier NaN devient -1, les autres deviennent 0
    3. Cas valeurs avec -1 : toutes les valeurs après le premier -1 deviennent 0
    4. Cas valeurs suivies de NaNs : premier NaN devient -1, les autres deviennent 0
    5. Cas mixte (-1 et NaNs) : première occurrence détermine la date de faillite
    
    Parameters
    ----------
    windows : dict
        Dictionnaire des fenêtres tel que généré par generate_windows_with_filtered_companies
    
    Returns
    -------
    tuple
        - processed_windows (dict): Dictionnaire avec les périodes de test modifiées
        - bankruptcy_dict (dict): Dictionnaire des dates de faillite par entreprise et par année
    """
    import pandas as pd
    import numpy as np
    
    # Dictionnaire qui contiendra les fenêtres traitées
    processed_windows = {}
    # Dictionnaire pour enregistrer les faillites par année
    bankruptcy_dict = {}
    
    # Valider l'entrée
    if not isinstance(windows, dict) or not windows:
        print("Erreur: windows doit être un dictionnaire non vide")
        return None, None
    
    # print("\n==== Processing Bakruptcy in the test period ====\n")
    
    # Parcourir chaque année de test dans le dictionnaire windows
    for test_year, window in windows.items():
        # Valider la structure de la fenêtre
        if "test_period" not in window:
            print(f"Avertissement: 'test_period' non trouvé dans la fenêtre pour l'année {test_year}. Ignoré.")
            continue
        
        if not isinstance(window["test_period"], pd.DataFrame):
            print(f"Avertissement: 'test_period' n'est pas un DataFrame pour l'année {test_year}. Ignoré.")
            continue
        
        # Copier la fenêtre complète pour la traiter
        processed_window = window.copy()
        
        # Copier le DataFrame de la période de test
        test_period = processed_window["test_period"].copy()
        
        # Dictionnaire pour enregistrer les faillites dans cette fenêtre
        window_bankruptcy = {}
        
        # Statistiques pour vérification
        stats = {"zeros_only": 0, "nans_only": 0, "has_minus_one": 0, "has_nans": 0, "mixed": 0}
        
        # Parcourir chaque société dans la période de test
        for company in test_period.columns:
            series = test_period[company]
            
            # Ignorer les séries entièrement non-NaN et sans valeur zéro ou -1
            if not series.isna().any() and not ((series == 0) | (series == -1)).any():
                continue
            
            # Cas 1: La série contient uniquement des 0
            if not series.isna().any() and (series == 0).all():
                test_period.at[series.index[0], company] = -1
                window_bankruptcy[company] = series.index[0]
                stats["zeros_only"] += 1
                continue
            
            # Cas 2: La série contient uniquement des NaN
            if series.isna().all():
                test_period.at[series.index[0], company] = -1
                for idx in series.index[1:]:
                    test_period.at[idx, company] = 0
                window_bankruptcy[company] = series.index[0]
                stats["nans_only"] += 1
                continue
            
            # Cas mixte: La série contient à la fois des -1 et des NaN
            if (series == -1).any() and series.isna().any():
                # Identifier la première occurrence de chaque type
                minus_one_indices = series[series == -1].index
                nan_indices = series[series.isna()].index
                
                first_minus_one = minus_one_indices[0] if len(minus_one_indices) > 0 else None
                first_nan = nan_indices[0] if len(nan_indices) > 0 else None
                
                # Prendre la première occurrence comme date de faillite
                if first_minus_one is not None and first_nan is not None:
                    bankruptcy_date = min(first_minus_one, first_nan)
                elif first_minus_one is not None:
                    bankruptcy_date = first_minus_one
                else:
                    bankruptcy_date = first_nan
                
                # Si la date de faillite est un NaN, la remplacer par -1
                if pd.isna(series.loc[bankruptcy_date]) or series.loc[bankruptcy_date] != -1:
                    test_period.at[bankruptcy_date, company] = -1
                
                # Mettre à 0 toutes les valeurs après la date de faillite
                for idx in series.loc[series.index > bankruptcy_date].index:
                    test_period.at[idx, company] = 0
                
                window_bankruptcy[company] = bankruptcy_date
                stats["mixed"] += 1
                continue
            
            # Cas 3: La série contient des valeurs et comporte un -1
            if (series == -1).any():
                first_bankruptcy_idx = series[series == -1].index[0]
                window_bankruptcy[company] = first_bankruptcy_idx
                # Mettre à 0 toutes les valeurs après first_bankruptcy_idx
                for idx in series.loc[series.index > first_bankruptcy_idx].index:
                    test_period.at[idx, company] = 0
                stats["has_minus_one"] += 1
                continue
            
            # Cas 4: La série est continue puis a des NaN
            if series.isna().any():
                # Trouver le premier NaN
                first_nan_index = series[series.isna()].index[0]
                test_period.at[first_nan_index, company] = -1
                window_bankruptcy[company] = first_nan_index
                # Mettre à 0 toutes les valeurs après first_nan_index
                for idx in series.loc[series.index > first_nan_index].index:
                    if pd.isna(test_period.at[idx, company]):
                        test_period.at[idx, company] = 0
                stats["has_nans"] += 1
                continue
        
        # Mettre à jour la période de test dans la fenêtre traitée
        processed_window["test_period"] = test_period
        processed_windows[test_year] = processed_window
        bankruptcy_dict[test_year] = window_bankruptcy
        
    
        # Afficher des statistiques de traitement
        total_bankruptcies = sum(stats.values())
        if verbose is not None:
            print(f"Year {test_year}: {total_bankruptcies} bankruptcies processed")
            if total_bankruptcies > 0:
                print(f"  - Series with only zeros: {stats['zeros_only']}")
                print(f"  - Series with only NaNs: {stats['nans_only']}")
                print(f"  - Series with a -1: {stats['has_minus_one']}")
                print(f"  - Series with NaNs after values: {stats['has_nans']}")
                print(f"  - Series with both -1 and NaNs: {stats['mixed']}")
                
                # Calculate the percentage of bankruptcies relative to the total number of companies
                bankruptcy_rate = total_bankruptcies / len(test_period.columns)
                print(f"  - Bankruptcy rate for this period: {bankruptcy_rate:.2%}")
        
        # print(f"\nProcessing complete: {len(processed_windows)} test periods processed.")
    return processed_windows, bankruptcy_dict


# In[154]:


# Traiter les périodes de test pour gérer les faillites
windows_, bankruptcy_dict = process_bankruptcy_in_windows(windows_filtered,
                                                          verbose=None)


# In[155]:


def bankruptcy_summary(windows_filtered):
    """
    Traite les données de faillite et crée un résumé synthétique.
    
    Parameters:
    -----------
    windows_filtered: dict
        Dictionnaire des fenêtres tel que généré par generate_windows_with_filtered_companies
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame avec les années en index et le nombre de faillites en colonne
    """
    import pandas as pd
    

    
    # Exécuter le traitement sans affichage verbeux
    processed_windows, bankruptcy_dict = process_bankruptcy_in_windows(windows_filtered, verbose=None)
    
    # Créer les données pour le DataFrame
    bankruptcy_data = []
    
    # Traiter chaque année
    for year in sorted(bankruptcy_dict.keys()):
        bankruptcies_count = len(bankruptcy_dict[year])
        bankruptcy_data.append({"Year": year, "Bankruptcies": bankruptcies_count})
        # print(f"\nYear {year}: {bankruptcies_count} bankruptcies detected")
        
        # Afficher un exemple si des faillites existent
        if bankruptcies_count > 0:
            sample = list(bankruptcy_dict[year].items())[0]
            # print(f"  Example: {sample[0]} (Date: {sample[1].strftime('%Y-%m-%d')})")
    
    # Créer le DataFrame final
    df_summary = pd.DataFrame(bankruptcy_data)
    df_summary.set_index("Year", inplace=True)
    
    # Afficher le résumé global
    total = df_summary["Bankruptcies"].sum()
    print("\n" + "="*50)
    print(f"SUMMARY: Total of {total} bankruptcies processed")
    print("="*50)
    
    return df_summary

# Générer le résumé et obtenir le DataFrame
bankruptcy_df_info = bankruptcy_summary(windows_filtered)

# Afficher le DataFrame
print("\nBankruptcy Summary DataFrame:")
print(bankruptcy_df_info)

# bankruptcy_df_info.to_latex('Article_Latex/bakrupcies.tex', index=None)


# In[156]:


def realign_annual_data_independent(annual_filtered_data):
    """
    Réaligne et filtre annual_filtered_data indépendamment :
    1. Shift des clés de +1 an (par ex: 2013 -> 2014)
    2. Shift également les index des DataFrames de +1 an
    3. Conserve uniquement 'scope1', 'scope2', 'yrevenue', 'ycap'
    4. Filtre pour ne garder que les années 2014-2024
    
    Parameters
    ----------
    annual_filtered_data : dict
        Dictionnaire original des données annuelles
        
    Returns
    -------
    dict
        Nouveau dictionnaire avec années décalées et données filtrées
    """
    realigned_data = {}
    
    # Dataframes à conserver
    keep_keys = ['scope1', 'scope2', 'yrevenue', 'ycap', 'valid_companies']
    
    # Parcourir chaque année
    for year_str, data_dict in annual_filtered_data.items():
        # Convertir l'année en entier, ajouter 1, et reconvertir en string
        new_year = str(int(year_str) + 1)
        
        # Ne conserver que les années entre 2014 et 2024 après le shift
        if 2014 <= int(new_year) <= 2024:
            # Créer une nouvelle entrée pour l'année shiftée
            realigned_data[new_year] = {}
            
            # Traiter chaque dataframe dans cette entrée
            for key, data in data_dict.items():
                # Ne conserver que les dataframes spécifiés
                if key not in keep_keys:
                    continue
                    
                if key == 'valid_companies':
                    # Pour la liste des entreprises valides, simplement la copier
                    realigned_data[new_year][key] = data.copy()
                else:
                    # Pour les DataFrames, shifter leur index d'une année
                    df = data.copy()
                    
                    # Shift l'index d'une année (les index sont de type Period)
                    # Créer un nouvel index décalé
                    new_index = df.index.map(lambda x: pd.Period(str(x.year + 1), freq='Y'))
                    df.index = new_index
                    
                    # Stocker le DataFrame avec l'index shifté
                    realigned_data[new_year][key] = df
    
    return realigned_data

# Créer une version réalignée indépendante
annual_filtered_data_realigned = realign_annual_data_independent(annual_filtered_data)


# In[157]:


def update_annual_data_with_backtest_companies(annual_filtered_data, windows_processed):
    """
    Filtre annual_filtered_data pour ne conserver que les entreprises présentes dans les fenêtres
    de backtest (alpha_period et test_period) pour chaque année correspondante.
    
    Parameters
    ----------
    annual_filtered_data : dict
        Dictionnaire des données annuelles filtrées.
        Format: {'2014': {'scope1': DataFrame, 'scope2': DataFrame, ..., 'valid_companies': list}, ...}
    windows_processed : dict
        Dictionnaire des fenêtres de backtest après traitement des faillites.
        Format: {'2014': {'alpha_period': DataFrame, 'test_period': DataFrame, ...}, ...}
    
    Returns
    -------
    dict
        Dictionnaire des données annuelles filtrées, ne contenant que les entreprises présentes
        dans les fenêtres de backtest.
    dict
        Statistiques de filtrage montrant le nombre d'entreprises avant/après filtrage pour chaque année.
    """
    updated_annual_data = {}
    filtration_stats = {}
    
    print("\n==== UPDATING ANNUAL DATAS WITH THE ELIGIBLE COMPANIES FROM THE BACKTEST CRITERION APPLIED ====\n")
    
    # Parcourir chaque année dans windows_processed
    for year, window_data in windows_processed.items():
        # Vérifier si l'année existe dans annual_filtered_data
        if year not in annual_filtered_data:
            print(f"L'année {year} n'existe pas dans annual_filtered_data, ignorée.")
            continue
        
        # Extraire les entreprises de test_period
        if 'test_period' in window_data:
            backtest_companies = list(window_data['test_period'].columns)
            
            # Statistiques de filtrage
            before_count = len(annual_filtered_data[year]['valid_companies'])
            after_count = len(backtest_companies)
            filtration_stats[year] = {
                'before': before_count,
                'after': after_count,
                'ratio': after_count / before_count if before_count > 0 else 0
            }
            
            # Filtrer les DataFrames de cette année
            updated_year_data = {}
            for key, df in annual_filtered_data[year].items():
                if key == 'valid_companies':
                    updated_year_data[key] = backtest_companies
                else:
                    # Pour les DataFrames, ne garder que les colonnes correspondant aux entreprises de backtest
                    columns_to_keep = [col for col in backtest_companies if col in df.columns]
                    updated_year_data[key] = df[columns_to_keep]
                    
                    # Vérifier si toutes les entreprises ont été retrouvées
                    missing = set(backtest_companies) - set(columns_to_keep)
                    if missing:
                        print(f"Attention: {len(missing)} entreprises non trouvées dans le DataFrame '{key}' pour l'année {year}")
            
            print(f"Year  {year}: {after_count} companies kept on {before_count} ({after_count/before_count:.2%})")
            updated_annual_data[year] = updated_year_data
        else:
            print(f"La clé 'test_period' manque dans window_data pour l'année {year}, ignorée.")
    
    return updated_annual_data, filtration_stats

def print_filtration_stats(filtration_stats):
    """
    Affiche les statistiques détaillées du filtrage des données annuelles et retourne un DataFrame.
    
    Parameters
    ----------
    filtration_stats : dict
        Dictionnaire contenant les statistiques de filtrage par année.
        Format: {'2014': {'before': 100, 'after': 80, 'ratio': 0.8}, ...}
    
    Returns
    -------
    pandas.DataFrame
        DataFrame contenant les statistiques de filtrage par année.
    """
    
    print("\n==== FILTERING STATISTICS ON THE ANNUAL DATA ====\n")
    
    # Trier les années
    years = sorted(filtration_stats.keys())
    
    if not years:
        print("Aucune statistique à afficher.")
        return pd.DataFrame()
    
    # Préparer les données pour le DataFrame
    data = []
    for year in years:
        stats = filtration_stats[year]
        before = stats['before']
        after = stats['after']
        difference = before - after
        ratio = stats['ratio']
        data.append([year, before, after, -difference, ratio])
    
    # Créer le DataFrame
    df = pd.DataFrame(data, columns=['YEAR', 'BEFORE', 'AFTER', 'KEPT', 'Ratio'])
    df.set_index('YEAR', inplace=True)
    
    # Calculer les statistiques globales
    total_before = df['BEFORE'].sum()
    total_after = df['AFTER'].sum()
    avg_ratio = total_after / total_before if total_before > 0 else 0
    
    # Ajouter la ligne de total au DataFrame
    total_row = pd.Series({
        'BEFORE': total_before,
        'AFTER': total_after,
        'KEPT': -(total_before - total_after),
        'Ratio': avg_ratio
    }, name='Total')
    
    df = pd.concat([df, pd.DataFrame(total_row).T])
    
    # Formater le DataFrame pour l'affichage
    df['Ratio'] = df['Ratio'].map('{:.2%}'.format)
    
    # Afficher le DataFrame formaté
    # print(df.to_string(index=True, justify='center'))
    
    # print("\n==== END OF STATISTICS ====\n")
    
    return df



updated_annual_data, filtration_stats = update_annual_data_with_backtest_companies(
    annual_filtered_data_realigned, 
    windows_
)

# Afficher les statistiques détaillées du filtrage
filtration_stats_ = round(print_filtration_stats(filtration_stats), 2)

# Remplacer annual_filtered_data par la version mise à jour
annual_filtered_data_ = updated_annual_data

# filtration_stats_.to_latex('Article_Latex/Elimination_companies_03_final.tex', index=True)
    


# In[158]:


# Création du DataFrame de base avec les années comme index
years = sorted([year for year in bankruptcy_df_info.index if year in filtration_stats_.index])
combined_df = pd.DataFrame(index=years)

# Ajout des données initiales
if 'Initial Companies' in summary_cleaning0.columns and 'Final Companies' in summary_cleaning0.columns:
    combined_df['Original'] = raw_companies[0]
    combined_df['Initial'] = first_process[0]
else:
    print("Attention: Les colonnes 'Initial Companies' ou 'Final Companies' sont manquantes dans summary_cleaning0")
    # Utiliser des valeurs alternatives si disponibles
    if 'Total Companies' in summary_cleaning0.columns:
        combined_df['Original'] = summary_cleaning0['Total Companies']

# Ajout des données de filtration_stats_
combined_df['Intermediary'] = filtration_stats_.loc[years, 'BEFORE']
combined_df['Final'] = filtration_stats_.loc[years, 'AFTER']
combined_df['Removed in Backtest'] = filtration_stats_.loc[years, 'KEPT']
combined_df['Backtest Filter Ratio (%)'] = (combined_df['Final'] / combined_df['Intermediary'] * 100).round(2)

# Ajout des données de faillites
combined_df['Bankruptcies'] = bankruptcy_df_info.loc[years, 'Bankruptcies']

# Ajout des données de filtering_stats
if isinstance(filtering_stats, dict):
    # Extraction des données du dictionnaire filtering_stats
    valid_before = []
    valid_after = []
    for year in years:
        if year in filtering_stats:
            valid_before.append(filtering_stats[year]['before'])
            valid_after.append(filtering_stats[year]['after'])
        else:
            valid_before.append(np.nan)
            valid_after.append(np.nan)
            
    combined_df['Available Before First Filter'] = valid_before
    combined_df['Available After First Filter'] = valid_after
    combined_df['First Filter Ratio (%)'] = (combined_df['Available After First Filter'] / combined_df['Available Before First Filter'] * 100).round(2)

# Calcul des métriques finales
combined_df['Bankruptcy Rate (%)'] = (combined_df['Bankruptcies'] / combined_df['Final'] * 100).round(2)
combined_df['Overall Retention Rate (%)'] = (combined_df['Final'] / combined_df['Intermediary'] * 100).round(2)

# Création d'une ligne de total
totals = {}
# Colonnes numériques à additionner
sum_columns = ['Original', 'Initial', 'Intermediary', 'Final', 
               'Removed in Backtest', 'Bankruptcies', 'Available Before First Filter', 'Available After First Filter']

# Calculer les sommes pour les colonnes numériques
for col in sum_columns:
    if col in combined_df.columns:
        totals[col] = combined_df[col].sum()

# Calcul des ratios pour la ligne Total
if 'Final' in totals and 'Intermediary' in totals and totals['Intermediary'] > 0:
    totals['Backtest Filter Ratio (%)'] = (totals['Final'] / totals['Intermediary'] * 100).round(2)
    totals['Overall Retention Rate (%)'] = (totals['Final'] / totals['Intermediary'] * 100).round(2)

if 'Bankruptcies' in totals and 'Final' in totals and totals['Final'] > 0:
    totals['Bankruptcy Rate (%)'] = (totals['Bankruptcies'] / totals['Final'] * 100).round(2)

if 'Available After First Filter' in totals and 'Available Before First Filter' in totals and totals['Available Before First Filter'] > 0:
    totals['First Filter Ratio (%)'] = (totals['Available After First Filter'] / totals['Available Before First Filter'] * 100).round(2)

# Ajout des totaux au DataFrame
totals_series = pd.Series(totals, name='Total')
combined_df__ = pd.concat([combined_df, totals_series.to_frame().T])

# Affichage du DataFrame final
round(combined_df__, 2)

combined_df__['Removed in Backtest'] = combined_df['Removed in Backtest'] * (-1) 
combined_df__ = combined_df__[['Original', 'Initial', 'Intermediary', 'Final', 'Removed in Backtest',
       'Backtest Filter Ratio (%)', 'Bankruptcies', 'Bankruptcy Rate (%)']]

# combined_df__ = round(combined_df__, 0)
# combined_df__.to_latex('Article_Latex/filtering_stats_general.tex', index=True, float_format="%.2f")
print(combined_df__)


# In[159]:


def cross_yprice_with_eligible_companies(yprice, windows_, bankruptcy_dict=None):
    """
    Croise les données de yprice (rendements annuels) avec les compagnies éligibles dans test_period 
    pour chaque année de 2014 à 2024, en gérant les valeurs manquantes selon les règles suivantes:
    - Si une compagnie est en faillite (selon bankruptcy_dict ou windows_), remplacer la valeur par -1
    - Si une compagnie a une valeur manquante (non faillite), remplacer par la moyenne annualisée
    
    Parameters:
    -----------
    yprice : pd.DataFrame
        DataFrame contenant les rendements annuels avec années en index (Period) et ISIN en colonnes
    windows_ : dict
        Dictionnaire contenant les fenêtres de test avec compagnies éligibles
    bankruptcy_dict : dict, optional
        Dictionnaire des faillites par année (keys=années, values=dict de compagnies et dates de faillite)
        
    Returns:
    --------
    tuple
        - Dict avec années en clés et DataFrames filtrés et complétés en valeurs
        - Dict avec statistiques sur les valeurs manquantes traitées par année
    """
    # Vérification des entrées
    if not isinstance(yprice, pd.DataFrame):
        raise TypeError("yprice doit être un DataFrame")
    if not isinstance(windows_, dict) or not windows_:
        raise TypeError("windows_ doit être un dictionnaire non vide")
        
    crossed_data = {}
    summary_stats = {}
    
    print("\n=== TRAITEMENT DES RENDEMENTS ANNUELS ===")
    print("Croisement avec les compagnies éligibles et traitement des valeurs manquantes\n")
    
    for year in sorted(windows_.keys()):
        # Vérifier que 'test_period' existe dans cette année
        if 'test_period' not in windows_[year]:
            print(f"⚠️ Année {year}: 'test_period' manquant dans windows_")
            continue
            
        # Extraction des compagnies éligibles pour l'année
        eligible_companies = windows_[year]['test_period'].columns.tolist()
        
        # Récupération des compagnies en faillite à partir de plusieurs sources
        bankruptcies = []
        
        # 1. Depuis windows_['faillite_test_period']
        if 'faillite_test_period' in windows_[year]:
            bankruptcies.extend(windows_[year]['faillite_test_period'])
        
        # 2. Depuis windows_['all_bankruptcy']
        if 'all_bankruptcy' in windows_[year]:
            bankruptcies.extend(windows_[year]['all_bankruptcy'])
            
        # 3. NOUVEAU: Depuis bankruptcy_dict
        if bankruptcy_dict is not None and year in bankruptcy_dict:
            bankruptcies.extend(bankruptcy_dict[year].keys())
            
        # Éliminer les doublons potentiels
        bankruptcies = list(set(bankruptcies))
        
        # Conversion de l'année en Period pour correspondre à l'index de yprice
        try:
            year_period = pd.Period(year)
        except ValueError:
            print(f"⚠️ Année {year}: Impossible de convertir en Period")
            continue
        
        # Initialiser les statistiques pour cette année
        summary_stats[year] = {
            'eligible_count': len(eligible_companies),
            'initial_missing': 0,
            'bankruptcies_filled': 0,
            'other_missing_filled': 0,
            'final_missing': 0,
            'available_count': 0
        }
        
        # Vérifier si l'année est dans l'index de yprice
        if year_period in yprice.index:
            # Filtrer yprice pour ne garder que les compagnies éligibles pour cette année
            common_companies = [comp for comp in eligible_companies if comp in yprice.columns]
            filtered_data = pd.DataFrame(index=[year_period], columns=common_companies)
            
            # Copier les données existantes
            for company in common_companies:
                filtered_data.loc[year_period, company] = yprice.loc[year_period, company]
            
            # Compter les valeurs manquantes initiales
            initial_missing = filtered_data.isna().sum().sum()
            summary_stats[year]['initial_missing'] = initial_missing
            summary_stats[year]['available_count'] = len(common_companies)
            
            # 1. Traiter les compagnies en faillite (remplacer par -1)
            # MODIFICATION: Traiter toutes les compagnies en faillite, qu'elles aient une valeur manquante ou non
            bankrupt_companies = [comp for comp in bankruptcies if comp in common_companies]
            bankruptcies_count = 0
            
            for company in bankrupt_companies:
                # Remplacer par -1 quelle que soit la valeur actuelle (NaN ou autre)
                filtered_data.loc[year_period, company] = -1.0
                bankruptcies_count += 1
            
            summary_stats[year]['bankruptcies_filled'] = bankruptcies_count
            
            # 2. Pour les autres compagnies avec NaN, utiliser la moyenne annualisée
            # Calculer la moyenne des rendements non-NaN et non-faillite
            valid_values = filtered_data.loc[year_period, :].copy()
            valid_values = valid_values[~valid_values.isin([-1.0])]  # Exclure les valeurs de faillite
            mean_return = valid_values.mean(skipna=True)
            
            # En cas de moyenne NaN (aucune valeur valide), utiliser 0.0 comme valeur par défaut
            if pd.isna(mean_return):
                mean_return = 0.0
                print(f"  ⚠️ Attention: Aucune valeur valide pour calculer la moyenne, utilisation de 0.0")
            
            # Remplacer les NaN restants par la moyenne
            missing_mask = filtered_data.isna()
            missing_count = missing_mask.sum().sum()
            
            if missing_count > 0:
                for company in common_companies:
                    if pd.isna(filtered_data.loc[year_period, company]):
                        filtered_data.loc[year_period, company] = mean_return
            
            summary_stats[year]['other_missing_filled'] = missing_count
            
            # Vérifier qu'il ne reste plus de NaN
            final_missing = filtered_data.isna().sum().sum()
            summary_stats[year]['final_missing'] = final_missing
            
            # Stocker les données filtrées et complétées
            crossed_data[year] = filtered_data
            
            print(f"Année {year}: {len(common_companies)} compagnies sur {len(eligible_companies)} trouvées dans yprice")
            print(f"  • Valeurs manquantes initiales: {initial_missing}")
            print(f"  • Compagnies en faillite (-1): {bankruptcies_count}")
            print(f"  • Autres manquantes remplacées par la moyenne ({mean_return:.4f}): {missing_count}")
            print(f"  • Valeurs manquantes restantes: {final_missing}")
            print("")
        else:
            print(f"⚠️ Année {year}: Aucune donnée disponible dans yprice")
            crossed_data[year] = pd.DataFrame()
    
    # Afficher un résumé global
    print("\n=== RÉSUMÉ DU TRAITEMENT ===")
    total_eligible = sum(stats['eligible_count'] for stats in summary_stats.values())
    total_available = sum(stats['available_count'] for stats in summary_stats.values())
    total_bankruptcies = sum(stats['bankruptcies_filled'] for stats in summary_stats.values())
    total_other_missing = sum(stats['other_missing_filled'] for stats in summary_stats.values())
    
    print(f"Total des compagnies éligibles: {total_eligible}")
    print(f"Total des compagnies disponibles dans yprice: {total_available} ({total_available/total_eligible:.1%})")
    print(f"Total des compagnies en faillite traitées: {total_bankruptcies}")
    print(f"Total des autres valeurs manquantes remplacées: {total_other_missing}")
    
    return crossed_data, summary_stats

# Exemple d'utilisation
yprice_processed, missing_by_year = cross_yprice_with_eligible_companies(yprice, windows_, bankruptcy_dict)

# Appel de la fonction
# yprice_by_test_period, missing_by_year = cross_yprice_with_eligible_companies(yprice, windows_)


# In[160]:


print('\n\n\n\n\n\n', '# ------------------------ 0.0  Data processing END -------------------------- #')


# In[161]:


# Not USED
# def frobenius_relative(windows, ridge=1e-6
#                        ):
#     """
#     % d’écart entre Σ_raw et Σ_psd   année par année.
#     """
#     import numpy as np, pandas as pd

#     out = {}
#     for year, win in windows.items():
#         R = win['alpha_period']
#         Σ_raw = R.cov().values

#         # projection PSD (même procédé que dans les portefeuilles)
#         eigval, eigvec = np.linalg.eigh(Σ_raw)
#         eigval = np.clip(eigval, ridge, None)
#         Σ_psd  = (eigvec * eigval) @ eigvec.T

#         diff_norm = np.linalg.norm(Σ_psd - Σ_raw, ord='fro')
#         raw_norm  = np.linalg.norm(Σ_raw          , ord='fro')

#         out[year] = 100 * diff_norm / raw_norm      #  en %
#     return pd.Series(out)


# In[162]:


# print('Frobenius relative distance')
# rel_gap = pd.DataFrame(frobenius_relative(windows_), columns=['%∆ Σ_raw - Σ_psd'])

# # Rename index to 'years'
# rel_gap.index.name = 'Years'

# rel_gap.loc['mean'] = rel_gap.mean()

# print(rel_gap)


# In[163]:


print('# ------------------------ 1.1  MVP Portfolio -------------------------- #')


# In[164]:


def _nearest_psd(cov: pd.DataFrame, ridge: float = 1e-6) -> np.ndarray:
    """
    Projection de Higham (2002) : force la PSD en
    clippant les valeurs propres négatives et en ajoutant un ridge minimal.
    """
    eigval, eigvec = np.linalg.eigh(cov.values)
    eigval = np.clip(eigval, ridge, None)        # >= ridge
    return (eigvec * eigval) @ eigvec.T          # Σ_psd

def min_variance_portfolio(returns: pd.DataFrame) -> pd.Series:
    """
    Portefeuille à variance minimale (long-only) avec matrice PSD sécurisée.
    """
    # 1) Matrice de covariance pair-wise
    Sigma_raw = returns.cov()                    # pair-wise, fenêtre 10 ans
    # 2) Projection PSD (même que pour TE / NZ)
    # Sigma = _nearest_psd(Sigma_raw)              # numpy array (n×n)

    n = len(returns.columns)

    # ----- optimisation SLSQP (long-only, somme = 1) ------------------------
    def objective(alpha):
        return 10000 * (alpha @ Sigma_raw @ alpha)     # facteur neutre pour le solveur

    constraints = [{'type': 'eq', 'fun': lambda a: a.sum() - 1}]
    bounds      = [(0.0, 1.0)] * n
    x0          = np.full(n, 1 / n)

    res = minimize(objective, x0, method='SLSQP',
                   bounds=bounds, constraints=constraints,
                   options={'ftol': 1e-8, 'maxiter': 500})

    if not res.success:
        raise ValueError(f"MVP optimisation failed → {res.message}")

    return pd.Series(res.x, index=returns.columns, name='MVP')


# Calcul des poids optimaux pour une fenêtre d'estimation donnée
# Itère sur les fenêtres de test pour afficher les poids optimaux et les résultats
def compute_optimal_weights(updated_tranches: dict) -> dict:
    """
    Applique la fonction d'optimisation (portefeuille à variance minimale) à la DataFrame 
    des rendements d'estimation de chaque tranche de backtest dans updated_tranches.
    
    Pour chaque année de test dans updated_tranches, la fonction :
      - Accède à la DataFrame 'alpha_period'
      - Calcule les poids optimaux via min_variance_portfolio
      - Affiche le résultat dans le format souhaité
      - Stocke les poids optimaux dans un dictionnaire, avec pour clé l'année de test
    
    Retourne
    --------
    optimal_weights_by_year : dict
        Dictionnaire avec pour chaque année de test (clé) la série des poids optimaux.
    """
    optimal_weights_by_year = {}
    
    # Créer un objet tqdm pour afficher la barre de progression
    years = list(updated_tranches.keys())
    with tqdm.tqdm(years, desc="Processing Optimisation", 
                   ncols=100,
                   leave=False,
                    smoothing=0.01 
                   ) as progress_bar:
        
        for year in progress_bar:
            est_returns = updated_tranches[year]['alpha_period']

            # Calcul des poids optimaux sur la période d'estimation
            opt_weights = min_variance_portfolio(est_returns)
            optimal_weights_by_year[year] = opt_weights
            
            # Récupérer les dates de début et de fin pour l'affichage
            if not est_returns.empty:
                est_start = est_returns.index[0].date()
                est_end = est_returns.index[-1].date()
            else:
                est_start, est_end = "N/A", "N/A"
            test_returns = updated_tranches[year]['test_period']
            if not test_returns.empty:
                test_start = test_returns.index[0].date()
                test_end = test_returns.index[-1].date()
            else:
                test_start, test_end = "N/A", "N/A"
            
            # Affichage formaté avec couleurs
            print("═" * 60)
            print("┌" + "─" * 58 + "┐")
            print("│ OPTIMIZATION FOR TEST YEAR: " + f"{year}".ljust(35) + "│")
            print("└" + "─" * 58 + "┘")
            print("")
            print("  ● Estimation window:    " + f"{est_start}  →  {est_end}")
            print("  ● Test period:          " + f"{test_start}  →  {test_end}")
            print("  ● Alpha Vector mean:    " + f"{opt_weights.mean():.6f}")
            print("")
            print("═" * 60)
    
    return optimal_weights_by_year


# In[165]:


# optimal_weights = compute_optimal_weights(windows_)


# In[166]:


# Après avoir calculé les poids optimaux

# Fonction qui calcule ou charge les poids optimaux
def get_optimal_weights(pickle_weights_window, windows_, force_recalculate=False):
    """
    Récupère les poids optimaux, soit depuis un fichier sauvegardé, soit en les recalculant.
  
    Parameters:
    -----------
    windows_ : dict
        Dictionnaire des fenêtres pour l'optimisation
    force_recalculate : bool, optional
        Si True, force le recalcul même si le fichier existe déjà
      
    Returns:
    --------
    dict
        Dictionnaire des poids optimaux
    """
    pickle_path = 'pickle_weights/' + f'{pickle_weights_window}' + '/optimal_weights.pkl'
  
    # Si le fichier existe et qu'on ne force pas le recalcul
    if os.path.exists(pickle_path) and not force_recalculate:
        print("📂 Chargement des poids optimaux depuis le fichier...")
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
    else:
        print("🔄 Calcul des poids optimaux en cours...")
        # Calcul des poids optimaux
        optimal_weights = compute_optimal_weights(windows_)
      
        # Sauvegarde pour une utilisation future
        with open(pickle_path, 'wb') as f:
            pickle.dump(optimal_weights, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("✅ Poids optimaux calculés et sauvegardés")
      
        return optimal_weights

# # Utilisation
optimal_weights = get_optimal_weights(pickle_weights_window, windows_, force_recalculate=False)


# In[167]:


def update_portfolio_weights(optimal_weights_by_year: dict, updated_tranches: dict) -> dict:
    """
    Pour chaque année de test présente dans optimal_weights_by_year et updated_tranches,
    met à jour de manière itérative les poids du portefeuille (alphas) sur la période de test.
    
    Pour chaque mois de la période de test, le calcul se fait ainsi :
       - R_p = sum(α_i * R_i)   (rendement du portefeuille pour le mois)
       - α_i(new) = α_i(old) * (1 + R_i) / (1 + R_p)
       
    L'output est un dictionnaire où chaque clé (année de test) est associée à un DataFrame
    qui donne, pour chaque mois de la période de test, l'évolution des poids.
    
    Parameters
    ----------
    optimal_weights_by_year : dict
        Dictionnaire avec, pour chaque année de test, une pd.Series contenant les poids optimaux 
        calculés sur la période d'estimation.
        
    updated_tranches : dict
        Dictionnaire avec, pour chaque année de test, notamment la clé 'test_returns' qui est
        un DataFrame des rendements mensuels sur la période de test (pour les compagnies éligibles).
        
    Returns
    -------
    updated_weights_by_year : dict
        Dictionnaire avec, pour chaque année de test, un DataFrame représentant l'évolution mensuelle
        des poids (les lignes correspondent aux dates de la période de test, les colonnes aux actifs).
    """
    updated_weights_by_year = {}
    
    # Parcourir chaque année de test
    for year in optimal_weights_by_year.keys():
        # Poids initiaux pour cette année (issus de l'optimisation sur la période d'estimation)
        current_weights = optimal_weights_by_year[year].copy()
        # DataFrame des rendements mensuels de la période de test pour cette année
        test_returns = updated_tranches[year]['test_period']
        
        # Créer une liste pour stocker l'évolution des poids
        evolution = []  # liste de tuples (date, weights_series)
        
        # Itérer sur chaque mois de la période de test (assurez-vous que l'index est trié)
        for date, returns in test_returns.iterrows():
            # Calculer le rendement du portefeuille pour ce mois : R_p = somme_i (α_i * R_i)
            R_p = (current_weights * returns).sum()
            # Mettre à jour les poids pour le mois suivant
            new_weights = current_weights * (1 + returns) / (1 + R_p)
            # (La formule garantit que la somme reste 1)
            evolution.append((date, new_weights))
            # Mettre à jour current_weights pour la prochaine itération
            current_weights = new_weights
        
        # Construire un DataFrame à partir de l'évolution, avec l'index = dates et les colonnes = actifs
        dates_list = [t[0] for t in evolution]
        weights_list = [t[1] for t in evolution]
        # On peut construire le DataFrame en combinant les poids pour chaque date
        weights_df = pd.DataFrame({col: [w[col] for w in weights_list] for col in current_weights.index},
                                  index=dates_list)
        
        updated_weights_by_year[year] = weights_df
        
        # # Affichage pour vérification
        # print(f"=== Evolution du poids moyen par actif pour l'année de test : {year} ===")
        # print(np.round(weights_df.mean().mean()*100, 3), '%',  "\n")
    
    return updated_weights_by_year


updated_weights = update_portfolio_weights(optimal_weights, windows_)


# In[168]:


print('Audit function update_weights - First Month')

product_df = windows_['2014']['test_period'].loc['2014-01-01':'2014-01-31'] * optimal_weights['2014']
# product_df
rp1 = product_df.sum(axis=1).to_list()[0]  # Initialisation trouver r_p dénominateur 
        # alpha_Y * r_i pour trouver r_p initial
        
rp_1 = 1 + rp1 # ajouter 1 à r_p 


r_i_1 = windows_['2014']['test_period'].loc['2014-01-01':'2014-01-31'] +1 # trouver (1 + r_i)
frac_ri = r_i_1 / rp_1 # fraction qui met à jour le premier mois des alphas

alpha_1 = optimal_weights['2014'] * frac_ri

print('Manual          :', alpha_1['CA0641491075'].loc['2014-01-31'])
print('Function        :', updated_weights['2014']['CA0641491075'].loc['2014-01-31'])
print('r_p first month :', rp1) 


# In[169]:


def compute_portfolio_returns(updated_weights: dict, updated_tranches: dict, optimal_weights: dict) -> pd.DataFrame:
    """
    Calcule les rendements du portefeuille selon la formule :
         R_p,t+k = α_{t+k-1}' * R_{t+k}
         
    Pour chaque année de test, on procède ainsi :
       - Pour le premier mois, on utilise le poids initial issu de l'estimation (optimal_weights)
       - Pour les mois suivants, on applique le poids mis à jour du mois précédent (donné dans updated_weights)
       
    Parameters
    ----------
    updated_weights : dict
        Dictionnaire avec, pour chaque année de test, un DataFrame des poids mis à jour
        pour chaque mois (ces poids ont été calculés en utilisant les rendements du mois courant
        pour anticiper le mois suivant).
    
    updated_tranches : dict
        Dictionnaire avec, pour chaque année de test, notamment la clé 'test_period' qui contient
        un DataFrame des rendements mensuels.
    
    optimal_weights : dict
        Dictionnaire avec, pour chaque année de test, une pd.Series contenant les poids optimaux calculés
        sur la période d'estimation (ces poids sont à appliquer pour le premier mois de la période de test).
        
    Returns
    -------
    portfolio_returns_all : pd.DataFrame
        DataFrame avec une colonne 'Portfolio_Return' contenant les rendements mensuels
        du portefeuille sur l'ensemble de la période de test, trié par date.
    """
    portfolio_returns_list = []

    # Parcourir les années de test par ordre chronologique
    for year in sorted(updated_tranches.keys()):
        # Récupérer les rendements mensuels des actifs pour l'année
        test_returns = updated_tranches[year]['test_period']
        # Récupérer les poids mis à jour pour cette année (ils correspondent aux poids calculés à la fin de chaque mois)
        weights_df = updated_weights[year]
        
        # On s'assure que les dates (index) et les actifs (colonnes) sont alignés entre test_returns et weights_df
        if not weights_df.index.equals(test_returns.index):
            print(f"⚠️ Attention : Les dates des poids et des rendements ne correspondent pas pour l'année {year}")
            common_dates = weights_df.index.intersection(test_returns.index)
            weights_df = weights_df.loc[common_dates]
            test_returns = test_returns.loc[common_dates]
            
        common_assets = weights_df.columns.intersection(test_returns.columns)
        if len(common_assets) != len(weights_df.columns) or len(common_assets) != len(test_returns.columns):
            print(f"⚠️ Attention : Certains actifs ne sont pas présents dans les deux DataFrames pour l'année {year}")
            weights_df = weights_df[common_assets]
            test_returns = test_returns[common_assets]
        
        # Construire un DataFrame qui contiendra les poids réellement utilisés pour calculer
        # le rendement de chaque mois. Pour le premier mois, on utilise le poids issu de l'estimation,
        # ensuite on décale les poids de updated_weights d'une période.
        weights_for_returns = pd.DataFrame(index=test_returns.index, columns=test_returns.columns)

        test_dates = test_returns.index.tolist()
        n = len(test_dates)
        
        # Pour le 1er mois : utiliser le poids initial issu de l'optimisation (optimal_weights)
        weights_for_returns.iloc[0] = optimal_weights[year]
        
        # Pour les mois suivants, utiliser le poids mis à jour du mois précédent
        # Attention : updated_weights[year] contient une ligne par mois de test, mais le premier
        # correspond au poids après le 1er mois (donc à appliquer pour le 2ème mois), etc.
        if n > 1:
            for i in range(1, n):
                # Le poids utilisé pour le mois i sera celui calculé lors du mois i-1
                weights_for_returns.iloc[i] = weights_df.iloc[i - 1]
        
        # Calculer le rendement du portefeuille pour chaque mois :
        # R_p = somme_i (α_i * R_i)
        portfolio_returns = (weights_for_returns * test_returns).sum(axis=1)
        
        portfolio_returns_df = portfolio_returns.to_frame(name='Portfolio_Return')
        portfolio_returns_list.append(portfolio_returns_df)
    
    if not portfolio_returns_list:
        print("⚠️ Aucun rendement de portefeuille n'a pu être calculé")
        return pd.DataFrame(columns=['Portfolio_Return'])
    
    portfolio_returns_all = pd.concat(portfolio_returns_list).sort_index()
    
    print(f"✓ Return of the portfolio calculated on {len(portfolio_returns_all)} months")
    print(f"  From                       : {portfolio_returns_all.index[0].strftime('%Y-%m-%d')} to {portfolio_returns_all.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Average monthly return     : {portfolio_returns_all['Portfolio_Return'].mean()*100:.2f}%")
    print(f"  Average monthly volatility : {portfolio_returns_all['Portfolio_Return'].std()*100:.2f}%")
    
    return portfolio_returns_all


portfolio_returns_all = compute_portfolio_returns(updated_weights, windows_, optimal_weights)


portfolio_returns_all.head()

# if use_pickle:
#     portfolio_returns_all.to_csv(f'pickle_weights/{pickle_weights_window}/mvp_{pickle_weights_window}.csv', index=True)
# portfolio_returns_all = pd.read_csv('portfolios/mvp.csv', index_col=0, parse_dates=True)

# portfolio_returns_all.index = pd.to_datetime(portfolio_returns_all.index)


# In[170]:


mvp_series = portfolio_returns_all['Portfolio_Return']


# In[171]:


def create_single_portfolio_dashboard(
    portfolio_series, 
    rf_series=None, 
    portfolio_name="Portfolio", 
    color=None, 
    figsize=(16, 12), 
    save_path=None,
    title="Portfolio Performance",
    # Paramètres de contrôle pour les shifts
    drawdown_horizontal_shift=30,
    drawdown_arc_radius=0.3
):
    """
    Crée un tableau de bord simplifié pour un seul portefeuille avec:
    1. Rendements cumulés (en haut, plus grand)
    2. Drawdowns (en bas, plus petit)
    
    Parameters
    ----------
    portfolio_series : pd.Series
        Série temporelle des rendements mensuels du portefeuille
    rf_series : pd.Series, optional
        Série temporelle du taux sans risque
    portfolio_name : str, optional
        Nom du portefeuille pour l'affichage
    color : str, optional
        Couleur pour le tracé du portefeuille
    figsize : tuple, optional
        Dimensions de la figure (largeur, hauteur)
    save_path : str, optional
        Chemin pour sauvegarder la figure
    title : str, optional
        Titre principal de la figure
    drawdown_horizontal_shift : int
        Contrôle le shift horizontal des annotations de drawdown
    drawdown_arc_radius : float
        Contrôle la courbure des flèches de drawdown
        
    Returns
    -------
    tuple
        Figure matplotlib et liste des axes
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker
    from matplotlib.gridspec import GridSpec
    
    # S'assurer que l'index est au format datetime
    portfolio_series.index = pd.to_datetime(portfolio_series.index)
    # Normaliser l'index pour supprimer toute information d'heure potentielle
    portfolio_series.index = portfolio_series.index.normalize()
    
    # Définir la couleur par défaut si non spécifiée
    if color is None:
        color = '#1f77b4'  # Bleu par défaut
    
    # Calculer les rendements cumulés
    portfolio_cumulative_returns = (1 + portfolio_series).cumprod()

    # Calculer les drawdowns
    portfolio_drawdown = pd.Series(0.0, index=portfolio_cumulative_returns.index, dtype=float)
    portfolio_cummax = portfolio_cumulative_returns.cummax()
    mask = portfolio_cummax != 0
    portfolio_drawdown.loc[mask] = ((portfolio_cummax.loc[mask] - portfolio_cumulative_returns.loc[mask]) / 
                                  portfolio_cummax.loc[mask]).astype(float)

    # Obtenir la valeur maximale de drawdown
    portfolio_max_dd = portfolio_drawdown.max() 

    # Set up a modern style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Create a function for custom styling
    def apply_custom_style(ax):
        """Apply custom styling to a matplotlib axis"""
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.7, color='#cccccc')
        ax.xaxis.label.set_fontsize(12)
        ax.yaxis.label.set_fontsize(12)
        ax.tick_params(axis='both', which='major', labelsize=10, colors='#333333')
        return ax

    # Create the dashboard with two main visualizations
    fig = plt.figure(figsize=figsize, facecolor='white', dpi=50)
    
    # Définir une GridSpec avec 2 lignes, une plus grande pour les rendements cumulés (80%)
    # et une plus petite pour les drawdowns (20%)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[0.80, 0.20])
    
    # 1. Cumulative Returns (en haut, plus grand)
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(portfolio_cumulative_returns.index, portfolio_cumulative_returns, 
            color=color, linewidth=2.5, label=portfolio_name)

    # Add ending value
    end_portfolio = portfolio_cumulative_returns.iloc[-1]
    ax1.annotate(f'{end_portfolio:.2f}x', 
               xy=(portfolio_cumulative_returns.index[-1], end_portfolio),
               xytext=(10, 0), textcoords='offset points',
               ha='left', va='center', fontweight='bold', color=color)

    # Ajouter le texte "Cumulative Return Growth of 1 USD" en bas à droite en gris discret
    ax1.annotate('Cumulative Return Growth of 1 USD', 
                xy=(0.95, 0.02),  # Position en bas à droite (coordonnées relatives)
                xycoords='axes fraction',  # Coordonnées relatives à l'axe
                ha='right', va='bottom',  # Alignement
                fontsize=10, color='#333333',  # Gris discret
                style='italic')  # Style italique pour plus de discrétion
                
    ax1.set_ylabel('Cumulative Return')
    
    # Hide x-axis labels for the top plot
    ax1.tick_params(axis='x', labelbottom=False)
    
    # Ajouter une ligne horizontale à y=1.0
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    
    ax1.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=12)
    apply_custom_style(ax1)

    # 2. Drawdowns (en bas, plus petit)
    ax2 = fig.add_subplot(gs[1])

    # Plot drawdowns
    ax2.fill_between(portfolio_drawdown.index, 0, -portfolio_drawdown.values, 
                    alpha=0.3, color=color, 
                    label=portfolio_name, step=None)

    ax2.plot(portfolio_drawdown.index, -portfolio_drawdown.values, 
             alpha=0.7, color=color, linewidth=1)

    # Find maximum drawdown date
    portfolio_valley_date = portfolio_drawdown.idxmax()

    # Highlight max drawdown period with improved arc annotation
    ax2.annotate(f'Max DD: -{portfolio_max_dd:.2%}',
                xy=(portfolio_valley_date, -portfolio_max_dd),
                xytext=(drawdown_horizontal_shift, 0), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color=color, 
                               connectionstyle=f'arc3,rad={drawdown_arc_radius}'),
                ha='left', fontsize=12, fontweight='bold', color=color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7, edgecolor='none'))

    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.xaxis.set_major_locator(mdates.YearLocator(1))
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    
    # Ajuster les limites pour avoir une marge en bas
    ax2.set_ylim(-portfolio_max_dd*1.3, 0.01)
    
    ax2.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=12)
    apply_custom_style(ax2)

    # Ajuster la mise en page avec peu d'espace entre les graphiques
    # plt.tight_layout()
    
    # Ajouter le titre principal tout en haut et en gras
    fig.suptitle(title, fontsize=18, fontweight='bold', x=0.5, y=0.98, ha='center')
    
    # Ajuster l'espace pour coller les graphiques et accommoder le titre
    plt.subplots_adjust(top=0.92, hspace=0.0)
    
    if save_path:
        plt.savefig(save_path, dpi=50, bbox_inches='tight')
        print(f"Dashboard saved to {save_path}")
    
    return fig, [ax1, ax2]

# Exemple d'utilisation pour visualiser le portefeuille MVP
fig, axes = create_single_portfolio_dashboard(
    portfolio_series=mvp_series,
    portfolio_name=r'$P_{oos}^{(mv)}$',
    color='#ff7f0e',  # Orange
    figsize=(22, 8),
    title=r'Minimum Variance Portfolio ($P_{oos}^{(mv)}$) Performance',
    save_path='MVP.png' if save_images else None
    )


# In[172]:


# First, combine the monthly returns of both portfolios into a single DataFrame
monthly_returns_mvp = pd.DataFrame({
    'MVP': portfolio_returns_all['Portfolio_Return']
})

# Make sure the index is sorted by date
# monthly_returns_mvp = monthly_returns_mvp.sort_index()

# Function to calculate portfolio performance metrics
def calculate_portfolio_metrics(returns_series, risk_free_series, market_returns_series=None):
    """Calculate key performance metrics for a portfolio, with detailed Sharpe ratio calculations"""
    # Match risk-free rates to the returns index and convert to decimal
    rf_matched = risk_free_series.reindex(returns_series.index, method='ffill')
    
    # Calculate excess returns
    excess_returns = returns_series - rf_matched
    
    # Calculate basic metrics
    monthly_mean_return = returns_series.mean()
    monthly_volatility = returns_series.std()
    annualized_avg_return = monthly_mean_return * 12
    annualized_volatility = monthly_volatility * np.sqrt(12)
    annualized_cum_return = ((1 + returns_series).prod()) ** (12/len(returns_series)) - 1
    
    # Nombre de mois dans la série
    num_months = len(returns_series)
    
    #----- Différentes méthodes de calcul du ratio de Sharpe -----#
    
    # Sharpe ratio mensuel (non-annualisé)
    monthly_sharpe = excess_returns.mean() / excess_returns.std()
    
    # Méthode 1: Annualisation directe du ratio de Sharpe mensuel
    sharpe_ratio_1 = monthly_sharpe * np.sqrt(12)
    
    # Méthode 2: Annualisation séparée des rendements et de la volatilité
    monthly_excess_mean = excess_returns.mean()
    annualized_excess_mean = monthly_excess_mean * 12
    sharpe_ratio_2 = annualized_excess_mean / annualized_volatility
    
    # Méthode 3: Utilisation des rendements composés (géométriques)
    portfolio_return_cumulative = (1 + returns_series).prod()
    annualized_portfolio_return = portfolio_return_cumulative ** (12/num_months) - 1
    
    rf_cumulative = (1 + rf_matched).prod()
    annualized_rf_geometric = rf_cumulative ** (12/num_months) - 1
    
    # Sharpe ratio basé sur les rendements géométriques
    sharpe_ratio_3 = (annualized_portfolio_return - annualized_rf_geometric) / annualized_volatility
    
    min_monthly_return = returns_series.min()
    max_monthly_return = returns_series.max()
    
    #----- Calcul du ratio de Treynor -----#
    treynor_ratio = None
    if market_returns_series is not None:
        # Match market returns to the portfolio returns index
        market_matched = market_returns_series.reindex(returns_series.index, method='ffill')
        
        # Calculate beta (covariance of portfolio and market / variance of market)
        covariance = np.cov(returns_series, market_matched)[0, 1]
        market_variance = market_matched.var()
        beta = covariance / market_variance
        
        # Calculate Treynor ratio (annualized)
        treynor_ratio = annualized_excess_mean / beta
    
    #----- Calcul du Maximum Drawdown et Recovery Period -----#
    
    # Calcul de l'indice de richesse cumulé
    wealth_index = (1 + returns_series).cumprod()
    
    # Calcul du maximum historique à chaque point
    previous_peaks = wealth_index.cummax()
    
    # Calcul du drawdown en pourcentage
    drawdowns = (wealth_index / previous_peaks - 1)
    
    # Maximum drawdown
    max_drawdown = drawdowns.min()
    
    # Date du maximum drawdown
    max_drawdown_date = drawdowns.idxmin()
    max_drawdown_date = max_drawdown_date.strftime('%Y-%m')
    
    # Période de récupération
    recovery_period = None
    max_drawdown_idx = drawdowns.idxmin()
    
    # Vérifier si nous avons récupéré après le max drawdown
    if max_drawdown_idx < wealth_index.index[-1]:
        # Valeur au pic précédent le max drawdown
        peak_before_max_dd = previous_peaks.loc[max_drawdown_idx]
        
        # Séries après le max drawdown
        post_drawdown = wealth_index.loc[max_drawdown_idx:]
        
        # Vérifier si nous avons récupéré
        if (post_drawdown >= peak_before_max_dd).any():
            recovery_idx = post_drawdown[post_drawdown >= peak_before_max_dd].index[0]
            recovery_period = len(returns_series.loc[max_drawdown_idx:recovery_idx])
    
    return {
        # Métriques standards
        "Annualized Average Return": annualized_avg_return,
        "Annualized Volatility": annualized_volatility,
        "Annualized Sharpe Ratio": sharpe_ratio_1,
        "Annualized Excess Return": annualized_excess_mean,
        "Annualized Cumulative Return": annualized_cum_return,
        "Monthly Average Return": monthly_mean_return,
        "Monthly Volatility": monthly_volatility,
        "Monthly Sharpe Ratio": monthly_sharpe,
        "Monthly Excess Return": monthly_excess_mean,
        "Maximum Monthly Return": max_monthly_return,
        "Minimum Monthly Return": min_monthly_return,
        "Maximum Drawdown": max_drawdown,
        "Maximum Drawdown Date": max_drawdown_date,
        "Recovery Period (months)": recovery_period,

    }
    
rf = rf/100
    
# Get column name for risk-free rate (assuming there's only one column)
rf_column = rf.columns[0]
    
mvp_metrics = calculate_portfolio_metrics(monthly_returns_mvp['MVP'], rf[rf_column])


mvp_metrics_df = pd.DataFrame({
    'MVP': pd.Series(mvp_metrics)
})


print(mvp_metrics_df)



# mvp_metrics_df.to_latex('Article_Latex/mvp_metrics.tex', index=True, float_format="%.5f")


# In[173]:


print('# ------------------------ END  MVP Portfolio -------------------------- #','\n\n\n\n\n\n')


# In[174]:


print('# ------------------------ 1.2  VW Portfolio -------------------------- #')


# $$\textcolor{yellow}{\text{1.2 ValueWeighted Portfolio} } $$
# 
# $$R_{t+1}^{(v w)}=\sum_{i=1}^N \textcolor{red}{w_{i, t}} \cdot R_{i, t+1}  \quad \textcolor{yellow}{avec} \quad \textcolor{red}{w_{i, t}}=\frac{C a p_{i, t} }{\sum_{j=1}^N C a p_{j, t}}$$
# 


# In[175]:


saved = mcap.copy()


# In[176]:


# On récupère les colonnes (ISIN) de test_period contenu dans la variable windows_ pour chaque année
# afin de récupérer leur marketCap à partir de 01.12.T jusqu'au 31.11.T+1 afin de calculer les w_i 
# qui multiplieront les rendements des actifs contenus dans test_period, aucun traitement de faillite n'est 
# nécessaire car traités dans test_period.


# Création d'un dictionnaire avec les années comme clés et les colonnes (ISIN) de test_period comme valeurs
def get_test_period_info(windows: dict) -> dict:
    """
    Extracts information about the test period for each year from the windows dictionary.

    For each year, retrieves:
      - The list of columns (ISIN) in the test_period.
      - The start date (begin period) of the test_period index.
      - The end date (end period) of the test_period index.

    Parameters
    ----------
    windows : dict
        Dictionary containing test_period data for each year.

    Returns
    -------
    test_period_info : dict
        Dictionary with years as keys and a dictionary as values containing:
          - 'columns': List of ISINs in the test_period.
          - 'begin_period': Start date of the test_period index.
          - 'end_period': End date of the test_period index.
    """
    test_period_info = {}
    for year, data in windows.items():
        test_period = data['test_period']
        test_period_info[year] = {
            'columns': test_period.columns.tolist(),
            'begin_period': test_period.index.min(),
            'end_period': test_period.index.max()
        }
    return test_period_info

# Appel de la fonction pour récupérer les informations sur les périodes de test
test_period_info = get_test_period_info(windows_)
test_period_info['2014']['begin_period']


# In[177]:


# On récupére les colonnes des sociétés de nos rendements existants dans mcap = leur marketCap
def extract_mcap_for_test_period(mcap: pd.DataFrame, test_period_info: dict) -> dict:
    """
    Extrait les données de capitalisation boursière (mcap) pour la période de test de chaque année.
    Gère les cas où les données peuvent être incomplètes en complétant par forward ou backward fill.
    """
    # S'assurer que l'index de mcap est au format datetime
    if not pd.api.types.is_datetime64_any_dtype(mcap.index):
        print("Conversion de l'index en datetime")
        mcap = mcap.copy()
        mcap.index = pd.to_datetime(mcap.index)
    
    mcap_by_year = {}
    
    for year, info in test_period_info.items():
        # Récupérer les ISINs pour la période de test
        requested_columns = info['columns']
        
        # Vérifier quels ISINs existent dans mcap
        available_columns = [col for col in requested_columns if col in mcap.columns]
        missing_columns = set(requested_columns) - set(available_columns)
        
        if missing_columns:
            print(f"Attention: {len(missing_columns)} ISINs non trouvés dans les données mcap pour l'année {year}")
            if len(missing_columns) < 10:  # Afficher seulement si le nombre est raisonnable
                print(f"ISINs manquants: {list(missing_columns)[:5]}...")
        
        if not available_columns:
            print(f"Erreur: Aucun ISIN trouvé dans les données mcap pour l'année {year}")
            mcap_by_year[year] = pd.DataFrame()
            continue
        
        try:
            # Pour s'assurer d'avoir des données pour l'année entière:
            # 1. Définir début et fin de la période de test complète
            year_int = int(year)
            test_start_date = pd.Timestamp(f"{year_int}-01-01")
            test_end_date = pd.Timestamp(f"{year_int}-12-31")
            
            # 2. Calculer la plage de dates pour mcap (mois précédent)
            mcap_start_date = test_start_date - pd.DateOffset(months=1)  # Décembre de l'année précédente
            mcap_end_date = test_end_date - pd.DateOffset(months=1)     # Novembre de l'année en cours
            
            print(f"Year {year}: Filtering mcap from {mcap_start_date.strftime('%Y-%m-%d')} to {mcap_end_date.strftime('%Y-%m-%d')}")
            
            # 3. Filtrer mcap pour les colonnes spécifiées et la plage de dates
            mask = (mcap.index >= mcap_start_date) & (mcap.index <= mcap_end_date)
            if not mask.any():
                print(f"Attention: Aucune date dans la plage pour l'année {year}")
                filtered_data = pd.DataFrame(columns=available_columns)
            else:
                filtered_data = mcap.loc[mask, available_columns]
            
            if filtered_data.empty:
                print(f"Attention: Aucune donnée de capitalisation trouvée pour l'année {year}")
                mcap_by_year[year] = pd.DataFrame()
                continue
                
            # 4. Vérifier si nous avons toutes les dates attendues (12 mois)
            expected_months = pd.date_range(
                start=mcap_start_date.replace(day=1),
                end=mcap_end_date.replace(day=1),
                freq='MS'  # Month Start
            )
            
            if len(filtered_data) < len(expected_months):
                print(f"Attention: Données incomplètes pour l'année {year}. "
                      f"Attendu: {len(expected_months)} mois, Trouvé: {len(filtered_data)} mois")
                
                # 5. Option: compléter les mois manquants par réplication des données les plus proches
                # Créer un DataFrame vide avec les dates attendues
                complete_data = pd.DataFrame(index=expected_months, columns=filtered_data.columns)
                
                # Copier les données existantes
                for date, row in filtered_data.iterrows():
                    closest_date = expected_months[np.abs((expected_months - date)).argmin()]
                    complete_data.loc[closest_date] = row
                
                # Remplir les valeurs manquantes par propagation en avant puis en arrière
                complete_data = complete_data.ffill().bfill()
                
                print(f"Données complétées pour l'année {year}: {len(complete_data)} mois")
                mcap_by_year[year] = complete_data
            else:
                print(f"Recovered data for year {year}: {len(filtered_data)} months and {filtered_data.shape[1]} companies\n")
                mcap_by_year[year] = filtered_data
            
        except Exception as e:
            print(f"Erreur lors du traitement de l'année {year}: {str(e)}")
            mcap_by_year[year] = pd.DataFrame()
    
    return mcap_by_year

# Evaluation d'une absence de donnée histoire de voir
mcap_by_year = extract_mcap_for_test_period(mcap, test_period_info)


# In[178]:


# Audit pour voir si w_i de mcap est bien calculé 
# print('Manual Audit for VW calculation (w_i)')
# print(mcap_by_year['2014'].iloc[1:2, 1:5])
m_2014 = saved[windows_['2014']['test_period'].columns.to_list()].loc['2013-12-01':'2014-11-30']
m_2014 = m_2014.apply(pd.to_numeric, errors='coerce')
# print(m_2014.iloc[1:2, 1:5])


# In[179]:


print('Manual Audit for VW calculation (first return)')
w_janvier = m_2014.loc['2013-12-01':'2013-12-31'] # Sélection que le mois de janvier
totaL_cap_janvier =  w_janvier.sum(axis=1).to_list()[0] # Total de la capitalisation boursière

w1 = w_janvier / totaL_cap_janvier # On a donc les w_i de janvier obtenus avec les cap de déc.embre

r1 = windows_['2014']['test_period'].loc['2014-01-01':'2014-01-31'].head(1) # Rendement de janvier

w1.index = r1.index

vwp1 = w1 * r1

print(vwp1.sum().sum())


# In[180]:


def check_nan_by_year(mcap_by_year: dict) -> dict:
    """
    Checks for NaN values in the market capitalization data for each year.

    For each year in mcap_by_year:
      - Counts the total number of NaN values in the DataFrame.

    Parameters
    ----------
    mcap_by_year : dict
        Dictionary with years as keys and DataFrames as values containing market capitalization data.

    Returns
    -------
    nan_count_by_year : dict
        Dictionary with years as keys and the total count of NaN values as values.
    """
    nan_count_by_year = {}
    for year, mcap_data in mcap_by_year.items():
        nan_count_by_year[year] = mcap_data.isnull().sum().sum()
        print(f'Year {year}','-> NaN  =', nan_count_by_year[year]  )
    return nan_count_by_year

print('Audit to see if there\'s any NaN in the in the Monthly Market Caps:')
# Appel de la fonction pour vérifier les NaN par année
nan_count_by_year = check_nan_by_year(mcap_by_year)


# In[181]:


def compute_weighted_mcap(mcap_by_year: dict) -> dict:
    """
    Computes the weighted market capitalization for each year.

    For each year in mcap_by_year:
      - Divides each value in the DataFrame by the sum of the values in the corresponding row.

    Parameters
    ----------
    mcap_by_year : dict
        Dictionary with years as keys and DataFrames as values containing market capitalization data.

    Returns
    -------
    weighted_mcap_by_year : dict
        Dictionary with years as keys and DataFrames as values containing the weighted market capitalization.
    """
    weighted_mcap_by_year = {}
    for year, mcap_data in mcap_by_year.items():
        # Diviser chaque valeur par la somme des valeurs de la ligne correspondante
        weighted_mcap = mcap_data.div(mcap_data.sum(axis=1), axis=0)
        weighted_mcap_by_year[year] = weighted_mcap
    return weighted_mcap_by_year

# Appel de la fonction pour calculer les pondérations
weighted_mcap_by_year = compute_weighted_mcap(mcap_by_year)


# In[182]:


# Dans l'idée de simplifier le produit matriciel avec les rendements
# On décale les poids de marketCap d'une année avec un décalage de +1 mois
def shift_weighted_mcap_by_year(weighted_mcap_by_year: dict) -> dict:
    """
    Shift les indices temporels des données de capitalisation boursière pondérée d'un mois vers l'avant.
    
    Cette fonction est utilisée pour aligner les poids de capitalisation boursière avec les rendements
    futurs, selon la formule R_{t+1}^{(vw)} = ∑_{i=1}^N w_{i,t} · R_{i,t+1}, où w_{i,t} est le poids
    de capitalisation boursière au temps t.
    
    La fonction garantit que chaque date est déplacée d'exactement un mois vers l'avant, en préservant
    le jour du mois quand c'est possible et en gérant correctement les cas spéciaux comme les fins de mois
    (ex: 31 janvier → 28/29 février).
    
    Parameters
    ----------
    weighted_mcap_by_year : dict
        Dictionnaire avec des années (str) comme clés et des DataFrames comme valeurs,
        contenant les capitalisations boursières pondérées. Chaque DataFrame doit avoir
        un index de type datetime.
    
    Returns
    -------
    shifted_weighted_mcap_by_year : dict
        Dictionnaire avec les mêmes années comme clés et des DataFrames comme valeurs,
        contenant les capitalisations boursières pondérées avec les indices décalés d'un mois.
    
    Raises
    ------
    TypeError
        Si l'entrée n'est pas un dictionnaire ou si une valeur n'est pas un DataFrame.
    ValueError
        Si un DataFrame a un index qui ne peut pas être converti en datetime.
    """
    # Validation de l'entrée
    if not isinstance(weighted_mcap_by_year, dict):
        raise TypeError("L'entrée doit être un dictionnaire")
    
    shifted_weighted_mcap_by_year = {}
    
    for year, weighted_mcap in weighted_mcap_by_year.items():
        # Validation du DataFrame pour cette année
        if not isinstance(weighted_mcap, pd.DataFrame):
            raise TypeError(f"La valeur pour l'année {year} n'est pas un DataFrame")
        
        # Traitement des DataFrames vides
        if weighted_mcap.empty:
            print(f"Avertissement: DataFrame vide pour l'année {year}")
            shifted_weighted_mcap_by_year[year] = weighted_mcap.copy()
            continue
        
        # S'assurer que l'index est de type datetime
        if not pd.api.types.is_datetime64_any_dtype(weighted_mcap.index):
            try:
                # Tenter de convertir en datetime si ce n'est pas déjà fait
                weighted_mcap = weighted_mcap.copy()
                weighted_mcap.index = pd.to_datetime(weighted_mcap.index)
                print(f"Index converti en datetime pour l'année {year}")
            except Exception as e:
                raise ValueError(f"Impossible de convertir l'index en datetime pour l'année {year}: {str(e)}")
        
        # Créer une copie pour éviter de modifier l'original
        shifted_weighted_mcap = weighted_mcap.copy()
        
        # Sauvegarder la longueur d'index et les dates originales pour vérification
        original_length = len(shifted_weighted_mcap.index)
        original_dates = shifted_weighted_mcap.index.copy()
        
        # Appliquer le décalage d'un mois
        shifted_weighted_mcap.index = shifted_weighted_mcap.index + pd.DateOffset(months=1)
        
        # Vérification: s'assurer que toutes les dates ont été décalées correctement
        if len(shifted_weighted_mcap.index) != original_length:
            raise ValueError(f"Le décalage des dates a modifié la longueur de l'index pour l'année {year}")
        
        # Vérifier que les dates ont été décalées d'exactement un mois
        for i, (old_date, new_date) in enumerate(zip(original_dates, shifted_weighted_mcap.index)):
            month_diff = (new_date.year * 12 + new_date.month) - (old_date.year * 12 + old_date.month)
            if month_diff != 1:
                print(f"Avertissement: Décalage incorrect pour la date à l'index {i} de l'année {year}: "
                      f"{old_date} → {new_date} (différence de {month_diff} mois)")
        
        # Stocker dans le dictionnaire de résultats
        shifted_weighted_mcap_by_year[year] = shifted_weighted_mcap
    
    print(f"Time shifting done:: {len(shifted_weighted_mcap_by_year)} years processed")
    return shifted_weighted_mcap_by_year

# Appel de la fonction pour décaler les poids
shifted_weighted_mcap_by_year = shift_weighted_mcap_by_year(weighted_mcap_by_year)

# Exemple pour vérifier le résultat pour l'année 2014
shifted_weighted_mcap_by_year['2014'].index = pd.to_datetime(shifted_weighted_mcap_by_year['2014'].index)
for year, data in shifted_weighted_mcap_by_year.items():
    print(f"Year: {year}, Shape: {data.shape}")


# In[183]:


# on applique maintenant la pondération des actifs par leur marketCap
# Appliquer la pondération pour chaque année
def compute_weighted_mcap(mcap_by_year: dict) -> dict:
    """
    Computes the weighted market capitalization for each year.

    For each year in mcap_by_year:
      - Divides each value in the DataFrame by the sum of the values in the corresponding row.

    Parameters
    ----------
    mcap_by_year : dict
        Dictionary with years as keys and DataFrames as values containing market capitalization data.

    Returns
    -------
    weighted_mcap_by_year : dict
        Dictionary with years as keys and DataFrames as values containing the weighted market capitalization.
    """
    weighted_mcap_by_year = {}
    for year, mcap_data in mcap_by_year.items():
        # Diviser chaque valeur par la somme des valeurs de la ligne correspondante
        weighted_mcap = mcap_data.div(mcap_data.sum(axis=1), axis=0)
        weighted_mcap_by_year[year] = weighted_mcap
    return weighted_mcap_by_year

# Appel de la fonction pour calculer les pondérations
weighted_mcap_by_year = compute_weighted_mcap(mcap_by_year)


# In[184]:


print('Audit Date Shifting')
print(w1.index[0]) # Ici on a les w_i de janvier calculés manuellements
print(weighted_mcap_by_year['2014'].index[0])
print(shifted_weighted_mcap_by_year['2014'].index[0])



# In[185]:


def compute_aligned_sorted_product(shifted_weighted_mcap_by_year: dict, windows_: dict) -> pd.DataFrame:
    """
    Calcule le produit élément par élément entre les poids décalés et les rendements de test,
    en alignant et en formatant correctement les dates. Pour chaque année de test, la fonction :
    
      - Vérifie et convertit l'index en datetime si nécessaire,
      - Trie les DataFrames par date,
      - Réindexe les poids (avec forward fill) pour correspondre à l'index du DataFrame des rendements,
      - Conserve l'intersection des colonnes communes (c'est-à-dire les compagnies disponibles dans les deux DataFrames),
      - Effectue le produit élément par élément (sans sommation),
      - Ajoute une colonne 'year' pour identifier l'année de test,
      - Concatène les résultats de toutes les années et trie le DataFrame final par date.
    
    Parameters
    ----------
    shifted_weighted_mcap_by_year : dict
        Dictionnaire avec pour clés les années (ex. "2014", "2015", etc.) et pour valeurs des DataFrames 
        contenant les poids décalés (index = dates, colonnes = compagnies).
    
    windows_ : dict
        Dictionnaire issu de generate_windows_with_returns. Pour chaque année (clé),
        windows_[year]['test_period'] est un DataFrame contenant les rendements mensuels 
        (index = dates, colonnes = compagnies).
    
    Returns
    -------
    pd.DataFrame
        DataFrame concaténé du produit élément par élément, trié par date.
    """
    
    results = []
    
    for year in sorted(windows_.keys()):
        # Récupération du DataFrame des rendements de test pour l'année
        test_returns_df = windows_[year]['test_period']
        # S'assurer que l'index est au format datetime et trié
        if not pd.api.types.is_datetime64_any_dtype(test_returns_df.index):
            test_returns_df.index = pd.to_datetime(test_returns_df.index)
        test_returns_df = test_returns_df.sort_index()
        
        # Récupération du DataFrame des poids décalés pour l'année (s'il existe)
        if year not in shifted_weighted_mcap_by_year:
            continue
        weights_df = shifted_weighted_mcap_by_year[year]
        if not pd.api.types.is_datetime64_any_dtype(weights_df.index):
            weights_df.index = pd.to_datetime(weights_df.index)
        weights_df = weights_df.sort_index()
        
        # Aligner l'index des poids sur celui des rendements (avec forward fill)
        weights_df = weights_df.reindex(test_returns_df.index, method='ffill')
        
        # Prendre l'intersection des colonnes disponibles dans les deux DataFrames
        common_cols = weights_df.columns.intersection(test_returns_df.columns)
        weights_df = weights_df[common_cols]
        test_returns_df = test_returns_df[common_cols]
        
        # Produit élément par élément (sans sommer)
        product_matrix = weights_df * test_returns_df
        
        # Ajout d'une colonne pour identifier l'année de test
        product_matrix['year'] = year
        
        results.append(product_matrix)
    
    # Concaténer les DataFrames issus de chaque année et trier par date
    if results:
        final_df = pd.concat(results, axis=0)
        final_df = final_df.sort_index()
        return final_df
    else:
        return pd.DataFrame()

product_df = compute_aligned_sorted_product(shifted_weighted_mcap_by_year, windows_)

product_df.head(1)


# In[186]:


# # Audit pour vérifier le premier élément
# print('Audit for the first monthly return by company')
# # print(vwp1) # Résultat effectué manuellement
# r1 = product_df.loc['2014-01-01':'2014-01-31'].head(1) # On vérifie que le produit est bien fait
# # print(r1)


# In[187]:


vw_portfolio_returns = product_df.drop(columns=['year']).sum(axis=1)
vw_portfolio_returns.name = 'VW_Portfolio_Return'

print('Audit for the first monthly return of the VWP')

print("Manual Audit :   return january 2014 -> ", vwp1.sum().sum())
print("Function     :   return january 2014 -> ", vw_portfolio_returns.loc['2014-01-01':'2014-01-31'].sum())

# if use_pickle:
#     vw_portfolio_returns.to_csv(f'pickle_weights/{pickle_weights_window}/vw_{pickle_weights_window}.csv', index=True)


# In[188]:


print('# ------------------------ 1.2  Metrics -------------------------- #')


# $$\textcolor{cyan}{\textbf{Metrics}}$$


# In[189]:


# First, combine the monthly returns of both portfolios into a single DataFrame
monthly_returns = pd.DataFrame({
    'MVP': portfolio_returns_all['Portfolio_Return'],
    'VW': vw_portfolio_returns
})

# Make sure the index is sorted by date
monthly_returns = monthly_returns.sort_index()

# Function to calculate portfolio performance metrics
def calculate_portfolio_metrics(returns_series, risk_free_series, market_returns_series=None):
    """Calculate key performance metrics for a portfolio, with detailed Sharpe ratio calculations"""
    # Match risk-free rates to the returns index and convert to decimal
    rf_matched = risk_free_series.reindex(returns_series.index, method='ffill')
    
    # Calculate excess returns
    excess_returns = returns_series - rf_matched
    
    # Calculate basic metrics
    monthly_mean_return = returns_series.mean()
    monthly_volatility = returns_series.std()
    annualized_avg_return = monthly_mean_return * 12
    annualized_volatility = monthly_volatility * np.sqrt(12)
    annualized_cum_return = ((1 + returns_series).prod()) ** (12/len(returns_series)) - 1
    
    # Nombre de mois dans la série
    num_months = len(returns_series)
    
    #----- Différentes méthodes de calcul du ratio de Sharpe -----#
    
    # Sharpe ratio mensuel (non-annualisé)
    monthly_sharpe = excess_returns.mean() / excess_returns.std()
    
    # Méthode 1: Annualisation directe du ratio de Sharpe mensuel
    sharpe_ratio_1 = monthly_sharpe * np.sqrt(12)
    
    # Méthode 2: Annualisation séparée des rendements et de la volatilité
    monthly_excess_mean = excess_returns.mean()
    annualized_excess_mean = monthly_excess_mean * 12
    sharpe_ratio_2 = annualized_excess_mean / annualized_volatility
    
    # Méthode 3: Utilisation des rendements composés (géométriques)
    portfolio_return_cumulative = (1 + returns_series).prod()
    annualized_portfolio_return = portfolio_return_cumulative ** (12/num_months) - 1
    
    rf_cumulative = (1 + rf_matched).prod()
    annualized_rf_geometric = rf_cumulative ** (12/num_months) - 1
    
    # Sharpe ratio basé sur les rendements géométriques
    sharpe_ratio_3 = (annualized_portfolio_return - annualized_rf_geometric) / annualized_volatility
    
    min_monthly_return = returns_series.min()
    max_monthly_return = returns_series.max()
    
    #----- Calcul du ratio de Treynor -----#
    treynor_ratio = None
    if market_returns_series is not None:
        # Match market returns to the portfolio returns index
        market_matched = market_returns_series.reindex(returns_series.index, method='ffill')
        
        # Calculate beta (covariance of portfolio and market / variance of market)
        covariance = np.cov(returns_series, market_matched)[0, 1]
        market_variance = market_matched.var()
        beta = covariance / market_variance
        
        # Calculate Treynor ratio (annualized)
        treynor_ratio = annualized_excess_mean / beta
    
    #----- Calcul du Maximum Drawdown et Recovery Period -----#
    
    # Calcul de l'indice de richesse cumulé
    wealth_index = (1 + returns_series).cumprod()
    
    # Calcul du maximum historique à chaque point
    previous_peaks = wealth_index.cummax()
    
    # Calcul du drawdown en pourcentage
    drawdowns = (wealth_index / previous_peaks - 1)
    
    # Maximum drawdown
    max_drawdown = drawdowns.min()
    
    # Date du maximum drawdown
    max_drawdown_date = drawdowns.idxmin()
    max_drawdown_date = max_drawdown_date.strftime('%Y-%m')
    
    # Période de récupération
    recovery_period = None
    max_drawdown_idx = drawdowns.idxmin()
    
    # Vérifier si nous avons récupéré après le max drawdown
    if max_drawdown_idx < wealth_index.index[-1]:
        # Valeur au pic précédent le max drawdown
        peak_before_max_dd = previous_peaks.loc[max_drawdown_idx]
        
        # Séries après le max drawdown
        post_drawdown = wealth_index.loc[max_drawdown_idx:]
        
        # Vérifier si nous avons récupéré
        if (post_drawdown >= peak_before_max_dd).any():
            recovery_idx = post_drawdown[post_drawdown >= peak_before_max_dd].index[0]
            recovery_period = len(returns_series.loc[max_drawdown_idx:recovery_idx])
    
    return {
        # Métriques standards
        "Annualized Average Return": annualized_avg_return,
        "Annualized Volatility": annualized_volatility,
        "Annualized Sharpe Ratio": sharpe_ratio_1,
        "Annualized Excess Return": annualized_excess_mean,
        "Annualized Cumulative Return": annualized_cum_return,
        "Monthly Average Return": monthly_mean_return,
        "Monthly Volatility": monthly_volatility,
        "Monthly Sharpe Ratio": monthly_sharpe,
        "Monthly Excess Return": monthly_excess_mean,
        "Maximum Monthly Return": max_monthly_return,
        "Minimum Monthly Return": min_monthly_return,
        "Maximum Drawdown": max_drawdown,
        "Maximum Drawdown Date": max_drawdown_date,
        "Recovery Period (months)": recovery_period,
        

    }
    
    
# Calculate metrics for each portfolio
mvp_metrics = calculate_portfolio_metrics(monthly_returns['MVP'], rf[rf_column])
vw_metrics = calculate_portfolio_metrics(monthly_returns['VW'], rf[rf_column])

# Create a DataFrame to display the results side by side
vw_mvp_metrics = pd.DataFrame({
    'MVP': pd.Series(mvp_metrics),
    'VW': pd.Series(vw_metrics)
})




# Afficher toutes les métriques
print(vw_mvp_metrics)



# vw_mvp_metrics.to_latex('Article_Latex/mvp_VS_vw.tex', index=True, float_format="%.5f")
# monthly_returns.to_excel('portfolio_monthly_returns.xlsx', index=True, sheet_name='Monthly Returns', float_format="%.35f")


# In[190]:


def extract_vw_weights(weighted_mcap_by_year):
    """
    Extrait les derniers poids VW pour chaque année à partir de weighted_mcap_by_year.
    
    Parameters
    ----------
    weighted_mcap_by_year : dict
        Dictionnaire contenant les poids VW pour chaque année sous forme de DataFrame
        
    Returns
    -------
    dict
        Dictionnaire avec les années comme clés et les poids VW (Series) comme valeurs
    """
    vw_weights = {}
    
    for year, weights_df in weighted_mcap_by_year.items():
        if not weights_df.empty:
            # Extraire les derniers poids VW pour cette année
            last_weights = weights_df.iloc[-1]
            vw_weights[year] = last_weights
    
    print(f"✅ Poids VW extraits pour {len(vw_weights)} années")
    return vw_weights

# Fonction qui calcule ou charge les poids optimaux
def get_optimal_weights_vw(pickle_weights_window, weighted_mcap_by_year, force_recalculate=False):
    """
    Récupère les poids optimaux, soit depuis un fichier sauvegardé, soit en les recalculant.
  
    Parameters:
    -----------
    windows_ : dict
        Dictionnaire des fenêtres pour l'optimisation
    force_recalculate : bool, optional
        Si True, force le recalcul même si le fichier existe déjà
      
    Returns:
    --------
    dict
        Dictionnaire des poids optimaux
    """
    pickle_path = 'pickle_weights/' + f'{pickle_weights_window}' + '/optimal_weights_VW.pkl'
  
    # Si le fichier existe et qu'on ne force pas le recalcul
    if os.path.exists(pickle_path) and not force_recalculate:
        print("📂 Chargement des poids optimaux depuis le fichier...")
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
    else:
        print("🔄 Calcul des poids optimaux en cours...")
        # Calcul des poids optimaux
        vw_weights = extract_vw_weights(weighted_mcap_by_year)
      
        # Sauvegarde pour une utilisation future
        with open(pickle_path, 'wb') as f:
            pickle.dump(vw_weights, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("✅ Poids optimaux calculés et sauvegardés")
      
        return vw_weights

# # Utilisation
vw_weights = get_optimal_weights_vw(pickle_weights_window, weighted_mcap_by_year, force_recalculate=False)


# $$\textcolor{yellow}{\text{Comparison Graphics MVP et VWP  }}$$


# In[191]:


vw_series = vw_portfolio_returns


# In[192]:


def create_dual_portfolio_dashboard(
    portfolio1_series, 
    portfolio2_series,
    rf_series=None, 
    portfolio1_name="Portfolio 1", 
    portfolio2_name="Portfolio 2",
    colors=None, 
    figsize=(16, 12), 
    save_path=None,
    title="Portfolio Comparison",
    # Paramètres de contrôle pour les shifts
    cumulative_return_horizontal_shift=5,
    cumulative_return_vertical_spacing=5,
    drawdown_horizontal_shift=30,
    drawdown_arc_radius=0.3
):
    """
    Crée un tableau de bord simplifié comparant deux portefeuilles avec:
    1. Rendements cumulés (en haut, plus grand)
    2. Drawdowns (en bas, plus petit)
    
    Parameters
    ----------
    portfolio1_series, portfolio2_series : pd.Series
        Séries temporelles des rendements des portefeuilles
    rf_series : pd.Series, optional
        Série temporelle du taux sans risque
    portfolio1_name, portfolio2_name : str, optional
        Noms des portefeuilles pour l'affichage
    colors : dict, optional
        Dictionnaire associant les noms de portefeuilles à leurs couleurs
    figsize : tuple, optional
        Dimensions de la figure (largeur, hauteur)
    save_path : str, optional
        Chemin pour sauvegarder la figure
    title : str, optional
        Titre principal de la figure
    cumulative_return_horizontal_shift : int
        Contrôle le shift horizontal des annotations de valeur finale
    cumulative_return_vertical_spacing : int
        Contrôle l'espacement vertical entre les annotations de valeur finale
    drawdown_horizontal_shift : int
        Contrôle le shift horizontal des annotations de drawdown
    drawdown_arc_radius : float
        Contrôle la courbure des flèches de drawdown
        
    Returns
    -------
    tuple
        Figure matplotlib et liste des axes
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker
    from matplotlib.gridspec import GridSpec
    
    # Vérifier que les index des séries sont alignés
    if not portfolio1_series.index.equals(portfolio2_series.index):
        common_index = portfolio1_series.index.intersection(portfolio2_series.index)
        if len(common_index) == 0:
            raise ValueError("Les deux séries de portefeuilles n'ont pas d'index commun")
        portfolio1_series = portfolio1_series.loc[common_index]
        portfolio2_series = portfolio2_series.loc[common_index]
        print(f"⚠️ Les index des séries ont été alignés. {len(common_index)} points communs conservés.")
    
    # Définir les couleurs par défaut si non spécifiées
    if colors is None:
        colors = {
            portfolio1_name: '#ff7f0e',  # Orange pour MVP
            portfolio2_name: '#2ca02c'   # Vert pour MVP50
        }
    
    # S'assurer que les index sont au format datetime
    portfolio1_series.index = pd.to_datetime(portfolio1_series.index)
    portfolio2_series.index = pd.to_datetime(portfolio2_series.index)
    # Normaliser les index pour supprimer toute information d'heure potentielle
    portfolio1_series.index = portfolio1_series.index.normalize()
    portfolio2_series.index = portfolio2_series.index.normalize()
    
    # Calculer les rendements cumulés
    portfolio1_cumulative_returns = (1 + portfolio1_series).cumprod()
    portfolio2_cumulative_returns = (1 + portfolio2_series).cumprod()

    # DataFrame pour la comparaison des rendements cumulatifs
    cumulative_returns_df = pd.DataFrame({
        portfolio1_name: portfolio1_cumulative_returns,
        portfolio2_name: portfolio2_cumulative_returns
    })

    # Calculer les drawdowns pour chaque portefeuille
    # Pour le premier portefeuille
    portfolio1_drawdown = pd.Series(0.0, index=portfolio1_cumulative_returns.index, dtype=float)
    portfolio1_cummax = portfolio1_cumulative_returns.cummax()
    mask1 = portfolio1_cummax != 0
    portfolio1_drawdown.loc[mask1] = ((portfolio1_cummax.loc[mask1] - portfolio1_cumulative_returns.loc[mask1]) / 
                                    portfolio1_cummax.loc[mask1]).astype(float)

    # Pour le deuxième portefeuille
    portfolio2_drawdown = pd.Series(0.0, index=portfolio2_cumulative_returns.index, dtype=float)
    portfolio2_cummax = portfolio2_cumulative_returns.cummax()
    mask2 = portfolio2_cummax != 0
    portfolio2_drawdown.loc[mask2] = ((portfolio2_cummax.loc[mask2] - portfolio2_cumulative_returns.loc[mask2]) / 
                                    portfolio2_cummax.loc[mask2]).astype(float)

    # Obtenir les valeurs maximales de drawdown
    portfolio1_max_dd = portfolio1_drawdown.max() 
    portfolio2_max_dd = portfolio2_drawdown.max()

    # Set up a modern style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Create a function for custom styling
    def apply_custom_style(ax):
        """Apply custom styling to a matplotlib axis"""
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.7, color='#cccccc')
        ax.xaxis.label.set_fontsize(12)
        ax.yaxis.label.set_fontsize(12)
        ax.tick_params(axis='both', which='major', labelsize=10, colors='#333333')
        return ax

    # Create the dashboard with two main visualizations
    fig = plt.figure(figsize=figsize, facecolor='white', dpi=50)
    
    # Définir une GridSpec avec 2 lignes, une plus grande pour les rendements cumulés (80%)
    # et une plus petite pour les drawdowns (20%)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[0.80, 0.20])
    
    # 1. Cumulative Returns (en haut, plus grand)
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(cumulative_returns_df.index, cumulative_returns_df[portfolio1_name], 
            color=colors[portfolio1_name], linewidth=2, label=portfolio1_name)
    ax1.plot(cumulative_returns_df.index, cumulative_returns_df[portfolio2_name], 
            color=colors[portfolio2_name], linewidth=2, label=portfolio2_name)

    # Amélioration du placement des valeurs finales à droite avec shift vertical et horizontal
    # Créer un dictionnaire des valeurs finales
    final_values = {
        portfolio1_name: cumulative_returns_df[portfolio1_name].iloc[-1],
        portfolio2_name: cumulative_returns_df[portfolio2_name].iloc[-1]
    }
    
    # Trier les portefeuilles par valeur finale
    sorted_portfolios = sorted(final_values.items(), key=lambda x: x[1])
    
    # Placer les annotations avec les shifts contrôlés
    vertical_spacing = cumulative_return_vertical_spacing
    
    for i, (name, value) in enumerate(sorted_portfolios):
        # Le shift horizontal est constant pour toutes les annotations
        horizontal_shift = cumulative_return_horizontal_shift
        
        # Calculer le shift vertical pour éviter les chevauchements
        if i == 0:  # Premier portefeuille (valeur la plus basse)
            vertical_shift = 0
        else:
            # Distance entre la valeur actuelle et celle du portefeuille précédent
            prev_value = sorted_portfolios[i-1][1]
            diff = value - prev_value
            
            # Si la différence est petite, augmenter le shift vertical
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

    # Ajouter le texte "Cumulative Return Growth of 1 USD" en bas à droite en gris discret
    ax1.annotate('Cumulative Return Growth of 1 USD', 
                xy=(0.95, 0.02),
                xycoords='axes fraction',
                ha='right', va='bottom',
                fontsize=10, color='#333333',
                style='italic')
                
    ax1.set_ylabel('Cumulative Return')
    
    # Hide x-axis labels for the top plot
    ax1.tick_params(axis='x', labelbottom=False)
    
    # Ajouter une ligne horizontale à y=1.0
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    
    ax1.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=10)
    apply_custom_style(ax1)

    # 2. Drawdowns (en bas, plus petit)
    ax2 = fig.add_subplot(gs[1])

    # Plot drawdowns with proper handling of values
    ax2.fill_between(portfolio1_drawdown.index, 0, -portfolio1_drawdown.values, 
                    alpha=0.3, color=colors[portfolio1_name], 
                    label=portfolio1_name, step=None)

    ax2.fill_between(portfolio2_drawdown.index, 0, -portfolio2_drawdown.values, 
                    alpha=0.3, color=colors[portfolio2_name], 
                    label=portfolio2_name, step=None)

    ax2.plot(portfolio1_drawdown.index, -portfolio1_drawdown.values, alpha=0.7, color=colors[portfolio1_name], linewidth=1)
    ax2.plot(portfolio2_drawdown.index, -portfolio2_drawdown.values, alpha=0.7, color=colors[portfolio2_name], linewidth=1)

    # Find maximum drawdowns dates
    portfolio1_valley_date = portfolio1_drawdown.idxmax()
    portfolio2_valley_date = portfolio2_drawdown.idxmax()

    # Collecter toutes les informations sur les drawdowns
    drawdowns = [
        (portfolio1_name, portfolio1_max_dd, portfolio1_valley_date, colors[portfolio1_name]),
        (portfolio2_name, portfolio2_max_dd, portfolio2_valley_date, colors[portfolio2_name])
    ]
    
    # Trier par date pour une meilleure organisation des annotations
    drawdowns.sort(key=lambda x: x[2])
    
    # Positionner les annotations à l'extérieur avec des flèches courbes
    for i, (name, max_dd, valley_date, color) in enumerate(drawdowns):
        # Décaler les annotations horizontalement pour éviter les chevauchements
        # Alternance droite/gauche pour les annotations
        horizontal_shift = drawdown_horizontal_shift * (-1 if i % 2 == 0 else 1)
        
        # Direction de l'arc basée sur la position (gauche ou droite)
        arc_rad = drawdown_arc_radius * (-1 if i % 2 == 0 else 1)
        
        ax2.annotate(f'Max DD: -{max_dd:.2%}',
                    xy=(valley_date, -max_dd),
                    xytext=(horizontal_shift, 0),
                    textcoords='offset points',
                    arrowprops=dict(
                        arrowstyle='->',
                        color=color,
                        connectionstyle=f'arc3,rad={arc_rad}',
                        shrinkA=0,
                        shrinkB=5
                    ),
                    ha='right' if i % 2 == 0 else 'left',
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
    
    # Ajuster les limites pour avoir une marge en bas et en haut
    all_max_dd = max(portfolio1_max_dd, portfolio2_max_dd)
    ax2.set_ylim(-all_max_dd * 1.3, all_max_dd * 0.4)
    
    ax2.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=10)
    apply_custom_style(ax2)

    # Ajuster la mise en page avec peu d'espace entre les graphiques
    # plt.tight_layout()
    
    # Ajouter le titre principal tout en haut et en gras
    fig.suptitle(title, fontsize=18, fontweight='bold', x=0.5, y=0.98, ha='center')
    
    # Ajuster l'espace pour coller les graphiques et accommoder le titre
    plt.subplots_adjust(top=0.92, hspace=0.0)
    
    if save_path:
        plt.savefig(save_path, dpi=50, bbox_inches='tight')
        print(f"Dashboard saved to {save_path}")
    
    return fig, [ax1, ax2]

# Exemple d'utilisation pour comparer MVP et MVP50
fig, axes = create_dual_portfolio_dashboard(
    portfolio1_series=mvp_series,
    portfolio2_series=vw_series,
    portfolio1_name=r'$P_{oos}^{(mv)}$',
    portfolio2_name=r'$P^{(mv)}$',
    colors={
        r'$P_{oos}^{(mv)}$': '#ff7f0e',
        r'$P^{(mv)}$': '#1f77b4'  # Vert pour VW
    },
    figsize=(22, 8),
    title=r'Comparison between $P_{oos}^{(mv)}$ and $P^{(vw)}$',
    save_path='MVP_vs_VW.png' if save_images else None,
    cumulative_return_horizontal_shift=10,
    cumulative_return_vertical_spacing=-5,
    drawdown_horizontal_shift=40,
    drawdown_arc_radius=0.3)



# In[193]:


print('# ------------------------ END 1.2  VW Portfolio -------------------------- #','\n\n\n\n\n\n')


# In[194]:


print('# ------------------------ 2.1  CF & WACI  -------------------------- #')


# In[195]:


def carbon_metrics_one_year(scope1, scope2, rev_kusd, cap_musd, weights):
    """
    Calcule le WACI et le Carbon Footprint pour un ensemble de poids donné à une date précise.
    
    Parameters:
    -----------
    scope1, scope2 : pd.Series 
        Émissions de scope 1 et scope 2 (en tonnes CO2e), index = ISIN
    rev_kusd : pd.Series
        Revenus des entreprises (en kUSD), index = ISIN
    cap_musd : pd.Series
        Capitalisations boursières (en MUSD), index = ISIN
    weights : pd.Series
        Poids des entreprises dans le portefeuille, index = ISIN
    
    Returns:
    --------
    tuple (waci, cf)
        WACI et Carbon Footprint en tCO2e/MUSD
    """
    # Identifier les entreprises présentes dans tous les datasets
    common = scope1.index.intersection(scope2.index).intersection(rev_kusd.index)\
             .intersection(cap_musd.index).intersection(weights.index)
    
    if len(common) == 0:
        raise ValueError("Aucune entreprise commune entre les datasets")

    # Restreindre toutes les séries aux entreprises communes
    emis = (scope1[common] + scope2[common]).astype(float)            # t
    rev_musd = (rev_kusd[common] / 1_000).astype(float)               # kUSD -> MUSD
    cap_musd = cap_musd[common].astype(float)                         # déjà en MUSD
    w = weights[common].astype(float)
    
    # Normaliser les poids pour s'assurer qu'ils somment à 1
    w = w / w.sum()
    
    # --- WACI: Weighted Average Carbon Intensity ---
    # CI_i = Emissions_i / Revenue_i, puis pondéré par les poids du portefeuille
    ci = emis / rev_musd                                              # t / MUSD
    waci = (w * ci).sum()                                             # t / MUSD

    # --- Carbon Footprint ---
    # CF = Σ(α_i × E_i) / Σ(α_i × Cap_i)
    weighted_emis = (w * emis).sum()                                  # t
    weighted_cap = (w * cap_musd).sum()                               # MUSD
    # --- Carbon Footprint (somme des ratios) ---------------------------
    cf = (w * emis / cap_musd).sum()        # t / MUSD

    # cf = weighted_emis / weighted_cap if weighted_cap > 0 else 0      # t / MUSD

    return waci, cf, len(common)


def carbon_metrics_all_years(annual_data, weights_by_year, verbose=False):
    """
    Calcule les métriques carbone (WACI et CF) pour toutes les années
    en utilisant les données annuelles et les poids du portefeuille.
    
    Parameters:
    -----------
    annual_data : dict
        Dictionnaire avec années en clés et sous-dictionnaires contenant
        'scope1', 'scope2', 'yrevenue', 'ycap' pour chaque année
    weights_by_year : dict
        Dictionnaire avec années en clés et séries de poids en valeurs
    verbose : bool, optional
        Si True, affiche les résultats pour chaque année
    
    Returns:
    --------
    pd.DataFrame
        DataFrame indexé par année avec colonnes 'WACI', 'CF', 'Firms'
    """
    rows = []
    
    for year, weights in sorted(weights_by_year.items()):
        if year not in annual_data:
            if verbose:
                print(f"Année {year}: Données annuelles non disponibles")
            continue
            
        year_data = annual_data[year]
        
        # Vérifier que toutes les données nécessaires sont disponibles
        if not all(key in year_data for key in ['scope1', 'scope2', 'yrevenue', 'ycap']):
            if verbose:
                print(f"Année {year}: Données incomplètes")
            continue
        
        try:
            # Extraire les dernières valeurs disponibles pour chaque série
            scope1_values = year_data['scope1'].iloc[-1] if len(year_data['scope1']) > 0 else pd.Series()
            scope2_values = year_data['scope2'].iloc[-1] if len(year_data['scope2']) > 0 else pd.Series()
            rev_values = year_data['yrevenue'].iloc[-1] if len(year_data['yrevenue']) > 0 else pd.Series()
            cap_values = year_data['ycap'].iloc[-1] if len(year_data['ycap']) > 0 else pd.Series()
            
            # Calculer WACI et CF
            waci, cf, n_firms = carbon_metrics_one_year(
                scope1_values,
                scope2_values,
                rev_values,
                cap_values,
                weights
            )
            
            # Enregistrer les résultats
            rows.append({
                'Year': int(year) if year.isdigit() else year,
                'WACI': waci,
                'CF': cf,
                'Firms': n_firms
            })
            
            if verbose:
                print(f"Année {year}: WACI={waci:.2f} tCO2e/MUSD, CF={cf:.2f} tCO2e/MUSD, {n_firms} entreprises")
                
        except Exception as e:
            if verbose:
                print(f"Erreur pour l'année {year}: {e}")
    
    # Créer et retourner le DataFrame final
    if not rows:
        return pd.DataFrame(columns=['Year', 'WACI', 'CF', 'Firms']).set_index('Year')
        
    return pd.DataFrame(rows).set_index('Year')


def compare_carbon_metrics(metrics1, metrics2, portfolio1_name="Portfolio1", portfolio2_name="Portfolio2"):
    """
    Compare les métriques carbone entre deux portefeuilles et calcule les différences relatives.
    
    Parameters:
    -----------
    metrics1, metrics2 : pd.DataFrame
        DataFrames des métriques carbone (sortie de carbon_metrics_all_years)
    portfolio1_name, portfolio2_name : str
        Noms des portefeuilles pour les colonnes du résultat
    
    Returns:
    --------
    pd.DataFrame
        DataFrame avec comparaison des métriques et différences relatives
    """
    # Joindre les deux DataFrames
    comparison = metrics1.join(metrics2, lsuffix=f'_{portfolio1_name}', rsuffix=f'_{portfolio2_name}')
    
    # Calculer les différences relatives en pourcentage
    comparison[f'WACI Δ%'] = 100 * (comparison[f'WACI_{portfolio1_name}']/comparison[f'WACI_{portfolio2_name}'] - 1)
    comparison[f'CF Δ%'] = 100 * (comparison[f'CF_{portfolio1_name}']/comparison[f'CF_{portfolio2_name}'] - 1)
    
    return comparison


# Calculer les métriques pour les deux portefeuilles
mvp_metrics_c = carbon_metrics_all_years(annual_filtered_data_, optimal_weights)
vw_metrics_c  = carbon_metrics_all_years(annual_filtered_data_, vw_weights)
comparison  = compare_carbon_metrics(mvp_metrics_c, vw_metrics_c, "MVP", "VW")
print(comparison.round(2))

# Vérifier que les ordres de grandeur sont cohérents:
# - VW WACI: environ 150t (2014) → 110t (2024)
# - VW CF: du même ordre que WACI (±20%)
# - MVP WACI/CF souvent > VW (secteurs "lourds" à variance minimale)
# - Δ% WACI entre -20% et +300% (si >1000% ou toujours négatif: anomalie)


# In[196]:


# mvp_metrics_c.to_latex('Article_Latex/mvp_waci_cf.tex', index=True, float_format="%.5f")


# In[197]:


comparison_fp = comparison[['CF_MVP', 'CF_VW', 'CF Δ%', 'Firms_MVP']]
comparison_fp = comparison_fp.rename(columns={
    'CF_MVP': 'CF$^{(P_{oos}^{mv})}_{Y}$',
    'CF_VW': 'CF$^{(P^{(vw)})}_{Y}$',
    'CF Δ%': '\\Delta (\\%)',
    'Firms_MVP': 'N$_{\\text{firms}}$'
})
# comparison_fp.to_latex('Article_Latex/mvp_vw_footprint.tex', index=True, float_format="%.2f")
print(comparison_fp)


# In[198]:


comparison_waci = comparison[['WACI_MVP', 'WACI_VW', 'WACI Δ%', 'Firms_MVP']]
comparison_waci = comparison_waci.rename(columns={
    'WACI_MVP': 'WACI$^{(P_{oos}^{mv})}_{Y}$',
    'WACI_VW': 'WACI$^{(P^{(vw)})}_{Y}$',
    'WACI Δ%': '\\Delta (\\%)',
    'Firms_MVP': 'N$_{\\text{firms}}$'
})
# comparison_waci.to_latex('Article_Latex/mvp_vw_waci.tex', index=True, float_format="%.2f")
print(comparison_waci)


# In[199]:


year = '2020'
scope1 = annual_filtered_data_[year]['scope1'].iloc[-1]
scope2 = annual_filtered_data_[year]['scope2'].iloc[-1]
cap    = annual_filtered_data_[year]['ycap'].iloc[-1]
w_mvp  = optimal_weights[year]

cf_contrib = w_mvp * (scope1+scope2) / cap          # t/M$
top = cf_contrib.sort_values(ascending=False).head(10)
print(top)


# In[200]:


def get_carbon_metrics(metrics_df, metric_type='WACI'):
    """
    Extrait une série temporelle spécifique depuis un DataFrame de métriques carbone.
    Retourne un dictionnaire avec les années en chaînes de caractères comme clés.
    
    Parameters:
    -----------
    metrics_df : pd.DataFrame
        DataFrame retourné par carbon_metrics_all_years
    metric_type : str
        Type de métrique à extraire ('WACI', 'CF', 'Firms')
        
    Returns:
    --------
    dict
        Dictionnaire avec les années (str) comme clés et les valeurs métriques comme valeurs
    """
    if metric_type not in metrics_df.columns:
        raise ValueError(f"Métrique {metric_type} non disponible. Options: {metrics_df.columns}")
    
    # Extraire la série et la convertir en dictionnaire avec années en str
    return {str(year): value for year, value in metrics_df[metric_type].items()}


# In[201]:


carbon_footprint_mvp = dict(get_carbon_metrics(mvp_metrics_c, 'CF'))
carbon_footprint_vw = dict(get_carbon_metrics(vw_metrics_c, 'CF'))

waci_mvp = dict(get_carbon_metrics(mvp_metrics_c, 'WACI'))
waci_vw = dict(get_carbon_metrics(vw_metrics_c, 'WACI'))


# In[202]:


# Obtenir le nombre d'entreprises pour chaque année
company_counts = []
for year in years:
    # Option 1: Utiliser les données de windows_ (si disponible)
    if year in windows_:
        # Le nombre d'entreprises est généralement le nombre de colonnes dans test_period ou alpha_period
        company_counts.append(len(windows_[year]['test_period'].columns))
    
    # Option 2: Utiliser les poids optimaux (si disponible)
    elif year in optimal_weights:
        company_counts.append(len(optimal_weights[year]))
    
    # Option 3: Utiliser annual_filtered_data_ (si disponible)
    elif year in annual_filtered_data_:
        # Utiliser le nombre d'entreprises dans scope1 ou scope2
        company_counts.append(len(annual_filtered_data_[year]['scope1'].columns))
    
    # Valeur par défaut si aucune source n'est disponible
    else:
        company_counts.append(0)


# In[203]:


# Couleurs pastel inspirées de viridis
colors_companies = [
    (0.98, 0.98, 0.98),  # Blanc légèrement gris
    (0.85, 0.86, 0.88),  # Gris très légèrement bleuté
    (0.75, 0.78, 0.85),  # Gris bleu clair
    (0.65, 0.7, 0.8)     # Gris bleu argenté
]
pastel_viridis = LinearSegmentedColormap.from_list('pastel_viridis', colors_companies)


plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['text.usetex'] = False  # Use LaTeX for text rendering

# Créer la figure avec une résolution plus élevée
fig = plt.figure(figsize=(22, 8), dpi=50)
# Réduire l'espace entre les graphes avec hspace=0.05
gs = GridSpec(2, 1, height_ratios=[5, 1], hspace=0.05)

# Graphique principal (WACI)
ax = fig.add_subplot(gs[0])

# Préparation des données
years = list(waci_mvp.keys())
waci_values_mvp = list(waci_mvp.values())
waci_values_vw = [waci_vw[year] for year in years]
x = np.arange(len(years))

# Calculer les moyennes pour les deux séries
avg_waci_mvp = np.mean(waci_values_mvp)
avg_waci_vw = np.mean(waci_values_vw)

# Largeur des barres pour le groupe
bar_width = 0.35

# Tracer les barres pour MVP
bars_mvp = ax.bar(x - bar_width/2, waci_values_mvp, width=bar_width, 
                 color='#ff7f0e', edgecolor='black', linewidth=0.5,
                 alpha=0.85, zorder=10, label=r'$P_{oos}^{(mv)}$ WACI')

# Tracer les barres pour VW
bars_vw = ax.bar(x + bar_width/2, waci_values_vw, width=bar_width,
                color='#1f77b4', edgecolor='black', linewidth=0.5,
                alpha=0.85, zorder=10, label=r'$P^{(vw)}$ WACI')

# Ajouter les lignes de moyenne
ax.axhline(avg_waci_mvp, color='#ff7f0e', linestyle='-', linewidth=1.2, 
           label=f'MVP Average: {avg_waci_mvp:.2f} tCO$_2$-eq/MUSD', zorder=15, alpha=0.7)
ax.axhline(avg_waci_vw, color='#1f77b4', linestyle='-', linewidth=1.2, 
           label=f'VW Average: {avg_waci_vw:.2f} tCO$_2$-eq/MUSD', zorder=15, alpha=0.7)

# Ajouter les annotations de valeur sur chaque barre
for i, bar in enumerate(bars_mvp):
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#ff7f0e', zorder=30)

for i, bar in enumerate(bars_vw):
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#1f77b4', zorder=30)

# Identifier les valeurs extrêmes pour MVP
min_idx_mvp = np.argmin(waci_values_mvp)
max_idx_mvp = np.argmax(waci_values_mvp)

# Ajouter une annotation pour la valeur minimale du MVP
ax.annotate(f'MVP Min: {waci_values_mvp[min_idx_mvp]:.2f}',
            xy=(min_idx_mvp - bar_width/2, waci_values_mvp[min_idx_mvp] + 5),
            xytext=(-40, 65),
            textcoords="offset points",
            ha='center', va='bottom',
            arrowprops=dict(
                arrowstyle='->',
                connectionstyle='arc3,rad=-.2',
                color='#ff7f0e',
                shrinkB=12,
                relpos=(0.5, 0.0)
            ),
            fontweight='bold', color='#ff7f0e', zorder=30,
            fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Ajouter une annotation pour la valeur maximale du MVP
ax.annotate(f'MVP Max: {waci_values_mvp[max_idx_mvp]:.2f}',
            xy=(max_idx_mvp - bar_width/2, waci_values_mvp[max_idx_mvp] + 8),
            xytext=(40, 80),
            textcoords="offset points",
            ha='center', va='bottom',
            arrowprops=dict(
                arrowstyle='->',
                connectionstyle='arc3,rad=.2',
                color='#ff7f0e',
                shrinkB=15,
                relpos=(0.5, 0.0)
            ),
            fontweight='bold', color='#ff7f0e', zorder=30,
            fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Identifier les valeurs extrêmes pour VW
min_idx_vw = np.argmin(waci_values_vw)
max_idx_vw = np.argmax(waci_values_vw)

# Ajouter une annotation pour la valeur minimale du VW
ax.annotate(f'VW Min: {waci_values_vw[min_idx_vw]:.2f}',
            xy=(min_idx_vw + bar_width/2, waci_values_vw[min_idx_vw] + 5),
            xytext=(40, 65),
            textcoords="offset points",
            ha='center', va='bottom',
            arrowprops=dict(
                arrowstyle='->',
                connectionstyle='arc3,rad=.2',
                color='#1f77b4',
                shrinkB=12,
                relpos=(0.5, 0.0)
            ),
            fontweight='bold', color='#1f77b4', zorder=30,
            fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Ajouter une annotation pour la valeur maximale du VW
ax.annotate(f'VW Max: {waci_values_vw[max_idx_vw]:.2f}',
            xy=(max_idx_vw + bar_width/2, waci_values_vw[max_idx_vw] + 8),
            xytext=(-30, 180),
            textcoords="offset points",
            ha='center', va='bottom',
            arrowprops=dict(
                arrowstyle='->',
                connectionstyle='arc3,rad=-.2',
                color='#1f77b4',
                shrinkB=12,
                relpos=(0.5, 0.0)
            ),
            fontweight='bold', color='#1f77b4', zorder=30,
            fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Personnaliser les axes du graphique supérieur
ax.set_xticks(x)
ax.set_xticklabels([])  # Masquer les étiquettes x sur le graphique du haut
ax.set_title(r'Weighted Average Carbon Intensity Comparison (2014-2024) - $\mathbf{WACI_Y^{(P_{oos}^{(mv)})}}$ vs $\mathbf{WACI_Y^{(P^{(vw)})}}$', pad=20)
ax.set_ylabel('WACI (tCO$_2$-eq/MUSD)', labelpad=10)

# Formater l'axe Y pour afficher les nombres avec des séparateurs de milliers
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))

# Améliorer l'apparence générale du graphique supérieur
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(False)
ax.set_axisbelow(True)

# Ajouter de l'espace au-dessus du graphique pour les annotations
ax.set_ylim(0, max(max(waci_values_mvp), max(waci_values_vw)) * 1.3)

# Graphique secondaire (Nombre d'entreprises)
ax2 = fig.add_subplot(gs[1])

# Ajouter une ligne de démarcation en haut du graphique inférieur
ax2.axhline(0, color='black', linewidth=0.5, zorder=5)

# Obtenir le nombre d'entreprises pour chaque année
company_counts = company_counts

# Créer un dégradé de couleur pour les barres d'entreprises basé sur le nombre
company_norm = plt.Normalize(min(company_counts) if company_counts else 0, 
                            max(company_counts) if company_counts else 1)
company_colors = pastel_viridis(company_norm(company_counts))

# Tracer les barres d'entreprises
bars2 = ax2.bar(x, company_counts, color=company_colors, alpha=0.8, width=0.7, 
               edgecolor='black', linewidth=0.5, zorder=10)

# Ajouter les annotations de valeur
for i, bar in enumerate(bars2):
    height = bar.get_height()
    ax2.annotate(f'{height}',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=9, fontweight='bold')

# Ajouter une annotation centrée pour remplacer le titre
fig.text(0.19, 0.24, 'Companies Included in Calculation', 
         ha='center', va='center', fontsize=8, fontweight='bold')

ax2.set_xlabel('Year', labelpad=10)
ax2.set_ylabel('Count', labelpad=10)
ax2.set_xticks(x)

# Utiliser les années originales pour les étiquettes
ax2.set_xticklabels(years, rotation=0)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(False)
ax2.set_axisbelow(True)

# Ajuster les dimensions pour s'assurer que les deux graphiques sont alignés
pos1 = ax.get_position()
pos2 = ax2.get_position()

# Légende pour le graphique principal
ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=10)

# Ajustement de la mise en page
plt.subplots_adjust(top=0.95, bottom=0.1, right=0.95)

plt.show()


# In[204]:


colors_companies = [
    (0.98, 0.98, 0.98),  # Blanc légèrement gris
    (0.85, 0.86, 0.88),  # Gris très légèrement bleuté
    (0.75, 0.78, 0.85),  # Gris bleu clair
    (0.65, 0.7, 0.8)     # Gris bleu argenté
]
pastel_viridis = LinearSegmentedColormap.from_list('pastel_viridis', colors_companies)


plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['text.usetex'] = False  # Use LaTeX for text rendering

# Créer la figure avec une résolution plus élevée
fig = plt.figure(figsize=(22, 8), dpi=50)
# Réduire l'espace entre les graphes avec hspace=0.05
gs = GridSpec(2, 1, height_ratios=[5, 1], hspace=0.05)

# Graphique principal (Carbon Intensity)
ax = fig.add_subplot(gs[0])

# Préparation des données
years = list(carbon_footprint_mvp.keys())
cf_values_mvp = list(carbon_footprint_mvp.values())
cf_values_vw = [carbon_footprint_vw[year] for year in years]
x = np.arange(len(years))

# Calculer les moyennes pour les deux séries
avg_cf_mvp = np.mean(cf_values_mvp)
avg_cf_vw = np.mean(cf_values_vw)

# Largeur des barres pour le groupe
bar_width = 0.35

# Tracer les barres pour MVP
bars_mvp = ax.bar(x - bar_width/2, cf_values_mvp, width=bar_width, 
                 color='#ff7f0e', edgecolor='black', linewidth=0.5,
                 alpha=0.85, zorder=10, label=r'$P_{oos}^{(mv)}$ Carbon Intensity')

# Tracer les barres pour VW
bars_vw = ax.bar(x + bar_width/2, cf_values_vw, width=bar_width,
                color='#1f77b4', edgecolor='black', linewidth=0.5,
                alpha=0.85, zorder=10, label=r'$P^{(vw)}$ Carbon Intensity')

# # Ajouter les lignes de tendance
# # Pour MVP
# z_mvp = np.polyfit(x, cf_values_mvp, 1)
# p_mvp = np.poly1d(z_mvp)
# ax.plot(x, p_mvp(x), linestyle='--', linewidth=1.2, color='#d62728', zorder=20, 
#         label=f"MVP CI Trend: {z_mvp[0]:.2f} tCO$_2$-eq/MUSD per year")

# # Pour VW
# z_vw = np.polyfit(x, cf_values_vw, 1)
# p_vw = np.poly1d(z_vw)
# ax.plot(x, p_vw(x), linestyle='--', linewidth=1.2, color='#2ca02c', zorder=20, 
#         label=f"VW CI Trend: {z_vw[0]:.2f} tCO$_2$-eq/MUSD per year")

# Ajouter les lignes de moyenne
ax.axhline(avg_cf_mvp, color='#ff7f0e', linestyle='-', linewidth=1.2, 
           label=f'MVP Average: {avg_cf_mvp:.2f} tCO$_2$-eq/MUSD', zorder=15, alpha=0.7)
ax.axhline(avg_cf_vw, color='#1f77b4', linestyle='-', linewidth=1.2, 
           label=f'VW Average: {avg_cf_vw:.2f} tCO$_2$-eq/MUSD', zorder=15, alpha=0.7)

# Ajouter les annotations de valeur sur chaque barre
for i, bar in enumerate(bars_mvp):
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#ff7f0e', zorder=30)

for i, bar in enumerate(bars_vw):
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#1f77b4', zorder=30)

# Identifier les valeurs extrêmes pour MVP
min_idx_mvp = np.argmin(cf_values_mvp)
max_idx_mvp = np.argmax(cf_values_mvp)

# Ajouter une annotation pour la valeur minimale du MVP
ax.annotate(f'MVP Min: {cf_values_mvp[min_idx_mvp]:.2f}',
            xy=(min_idx_mvp - bar_width/2, cf_values_mvp[min_idx_mvp] + 5),
            xytext=(-40, 65),
            textcoords="offset points",
            ha='center', va='bottom',
            arrowprops=dict(
                arrowstyle='->',
                connectionstyle='arc3,rad=-.2',
                color='#ff7f0e',
                shrinkB=12,
                relpos=(0.5, 0.0)
            ),
            fontweight='bold', color='#ff7f0e', zorder=30,
            fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Ajouter une annotation pour la valeur maximale du MVP
ax.annotate(f'MVP Max: {cf_values_mvp[max_idx_mvp]:.2f}',
            xy=(max_idx_mvp - bar_width/2, cf_values_mvp[max_idx_mvp] + 8),
            xytext=(40, 80),
            textcoords="offset points",
            ha='center', va='bottom',
            arrowprops=dict(
                arrowstyle='->',
                connectionstyle='arc3,rad=-.2',
                color='#ff7f0e',
                shrinkB=12,
                relpos=(0.5, 0.0)
            ),
            fontweight='bold', color='#ff7f0e', zorder=30,
            fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Identifier les valeurs extrêmes pour VW
min_idx_vw = np.argmin(cf_values_vw)
max_idx_vw = np.argmax(cf_values_vw)

# Ajouter une annotation pour la valeur minimale du VW
ax.annotate(f'VW Min: {cf_values_vw[min_idx_vw]:.2f}',
            xy=(min_idx_vw + bar_width/2, cf_values_vw[min_idx_vw] + 5),
            xytext=(40, 65),
            textcoords="offset points",
            ha='center', va='bottom',
            arrowprops=dict(
                arrowstyle='->',
                connectionstyle='arc3,rad=.2',
                color='#1f77b4',
                shrinkB=12,
                relpos=(0.5, 0.0)
            ),
            fontweight='bold', color='#1f77b4', zorder=30,
            fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Ajouter une annotation pour la valeur maximale du VW
ax.annotate(f'VW Max: {cf_values_vw[max_idx_vw]:.2f}',
            xy=(max_idx_vw + bar_width/2, cf_values_vw[max_idx_vw] + 8),
            xytext=(-30, 120),
            textcoords="offset points",
            ha='center', va='bottom',
            arrowprops=dict(
                arrowstyle='->',
                connectionstyle='arc3,rad=-.2',
                color='#1f77b4',
                shrinkB=12,
                relpos=(0.5, 0.0)
            ),
            fontweight='bold', color='#1f77b4', zorder=30,
            fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Personnaliser les axes du graphique supérieur
ax.set_xticks(x)
ax.set_xticklabels([])  # Masquer les étiquettes x sur le graphique du haut
ax.set_title(r'Carbon Footprint Comparison (2014-2024) - $\mathbf{C F_Y^{(P_{oos}^{(mv)})}}$ vs $\mathbf{C F_Y^{(P^{(vw)})}}$', pad=20)
ax.set_ylabel('Carbon Intensity (tCO$_2$-eq/MUSD)', labelpad=10)

# Formater l'axe Y pour afficher les nombres avec des séparateurs de milliers
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))

# Améliorer l'apparence générale du graphique supérieur
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(False)
ax.set_axisbelow(True)

# Ajouter de l'espace au-dessus du graphique pour les annotations
ax.set_ylim(0, max(max(cf_values_mvp), max(cf_values_vw)) * 1.3)

# Graphique secondaire (Nombre d'entreprises) - INCHANGÉ
ax2 = fig.add_subplot(gs[1])

# Ajouter une ligne de démarcation en haut du graphique inférieur
ax2.axhline(0, color='black', linewidth=0.5, zorder=5)

# Obtenir le nombre d'entreprises pour chaque année
company_counts = company_counts

# Créer un dégradé de couleur pour les barres d'entreprises basé sur le nombre
company_norm = plt.Normalize(min(company_counts) if company_counts else 0, 
                            max(company_counts) if company_counts else 1)
company_colors = pastel_viridis(company_norm(company_counts))

# Tracer les barres d'entreprises
bars2 = ax2.bar(x, company_counts, color=company_colors, alpha=0.8, width=0.7, 
               edgecolor='black', linewidth=0.5, zorder=10)

# Ajouter les annotations de valeur
for i, bar in enumerate(bars2):
    height = bar.get_height()
    ax2.annotate(f'{height}',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=9, fontweight='bold')

# Ajouter une annotation centrée pour remplacer le titre
fig.text(0.19, 0.24, 'Companies Included in Calculation', 
         ha='center', va='center', fontsize=8, fontweight='bold')

ax2.set_xlabel('Year', labelpad=10)
ax2.set_ylabel('Count', labelpad=10)
ax2.set_xticks(x)

# Utiliser les années originales sans décalage pour les étiquettes
ax2.set_xticklabels(years, rotation=0)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(False)
ax2.set_axisbelow(True)

# Ajuster les dimensions pour s'assurer que les deux graphiques sont alignés
pos1 = ax.get_position()
pos2 = ax2.get_position()

# Légende pour le graphique principal
ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=10)


# fig.text(0.5, 0.01, footnote, ha='center', fontsize=9, style='italic', alpha=0.7)

# Ajustement de la mise en page
plt.subplots_adjust(top=0.95, bottom=0.1, right=0.95)
# plt.tight_layout()
# plt.savefig('carbon_intensity_comparison_mvp_vw.png', dpi=50, bbox_inches='tight')
plt.show()


# In[205]:


def analyze_carbon_metrics(metrics1, metrics2, portfolio1_name="Portfolio1", portfolio2_name="Portfolio2"):
    """
    Compare les métriques carbone entre deux portefeuilles et calcule les différences relatives.
    
    Parameters:
    -----------
    metrics1, metrics2 : pd.DataFrame
        DataFrames des métriques carbone (sortie de carbon_metrics_all_years)
    portfolio1_name, portfolio2_name : str
        Noms des portefeuilles pour les colonnes du résultat
    
    Returns:
    --------
    pd.DataFrame
        DataFrame avec comparaison des métriques et différences relatives
    """
    # Vérifier que les entrées sont des DataFrames
    if not isinstance(metrics1, pd.DataFrame) or not isinstance(metrics2, pd.DataFrame):
        print("Erreur: les métriques doivent être des DataFrames")
        return pd.DataFrame()
    
    # Joindre les deux DataFrames
    comparison = metrics1.join(metrics2, lsuffix=f'_{portfolio1_name}', rsuffix=f'_{portfolio2_name}')
    
    # Calculer les différences relatives en pourcentage
    waci_col1 = f'WACI_{portfolio1_name}'
    waci_col2 = f'WACI_{portfolio2_name}'
    cf_col1 = f'CF_{portfolio1_name}'
    cf_col2 = f'CF_{portfolio2_name}'
    
    # Ajouter les colonnes de différence si les colonnes requises existent
    if waci_col1 in comparison.columns and waci_col2 in comparison.columns:
        comparison[f'WACI Δ%'] = 100 * (comparison[waci_col1]/comparison[waci_col2] - 1)
    
    if cf_col1 in comparison.columns and cf_col2 in comparison.columns:
        comparison[f'CF Δ%'] = 100 * (comparison[cf_col1]/comparison[cf_col2] - 1)
    
    # Afficher un résumé
    print("\n=== ANALYSE DES MÉTRIQUES CARBONE ===")
    print(f"Période analysée: {comparison.index.min()} à {comparison.index.max()}")
    print(f"Nombre d'années avec données: {len(comparison)}")
    
    if not comparison.empty:
        if f'WACI Δ%' in comparison.columns:
            avg_waci_reduction = comparison[f'WACI Δ%'].mean()
            print(f"\n1. Weighted Average Carbon Intensity (WACI):")
            print(f"  Réduction moyenne WACI ({portfolio1_name} vs {portfolio2_name}): {avg_waci_reduction:.2f}%")
            print(f"  WACI moyen par portefeuille:")
            print(f"    {portfolio1_name}: {comparison[waci_col1].mean():.2f} tCO2e/M$")
            print(f"    {portfolio2_name}: {comparison[waci_col2].mean():.2f} tCO2e/M$")
        
        if f'CF Δ%' in comparison.columns:
            avg_cf_reduction = comparison[f'CF Δ%'].mean()
            print(f"\n2. Carbon Footprint:")
            print(f"  Réduction moyenne CF ({portfolio1_name} vs {portfolio2_name}): {avg_cf_reduction:.2f}%")
            print(f"  Carbon Footprint moyen par portefeuille:")
            print(f"    {portfolio1_name}: {comparison[cf_col1].mean():.2f} tCO2e/M$")
            print(f"    {portfolio2_name}: {comparison[cf_col2].mean():.2f} tCO2e/M$")
    
    return comparison.round(2)


# Calculer les métriques pour les deux portefeuilles
mvp_metrics_c = carbon_metrics_all_years(annual_filtered_data_, optimal_weights)
vw_metrics_c = carbon_metrics_all_years(annual_filtered_data_, vw_weights)

# Comparer les métriques
comparison = analyze_carbon_metrics(mvp_metrics_c, vw_metrics_c, "MVP", "VW")

# comparison.to_latex('Article_Latex/waci_cf.tex', index=True)
print(comparison)



# In[206]:


print('\n\n\n\n\n\n')
print('# ------------------------ 2.2  P(mv)(0.5) -------------------------- #')


# In[207]:


def compute_carbon_constrained_portfolio(
    filtered_windows,
    filtered_annual_data,
    mvp_carbon_footprint,
    verbose=True
):
    """
    Calcule un portefeuille optimal sous contrainte carbone (Point 2.1).

    Problème :
        min_{α_Y}  σ²_p,Y  = α_Y' Σ_Y+1 α_Y
        s.t.        CF_Y^(p) ≤ 0.5 · CF_Y^(MVP)
                     α_i,Y ≥ 0  ∀i
                     Σ α_i,Y = 1

    ----------
    filtered_windows : dict
        {'YYYY': {'alpha_period': DataFrame de rendements mensuels}}
    filtered_annual_data : dict
        {'YYYY': {'scope1','scope2','ycap'}  – tous en fin d’année}
    mvp_carbon_footprint : dict
        CF_Y^(MVP) calculé avec ta nouvelle fonction « ratio des sommes »
    verbose : bool
        Affiche le diagnostic de chaque optimisation
    ----------
    Retour
        (weights_dict, cf_dict)
    """
    import numpy as np
    import pandas as pd
    from scipy.optimize import minimize

    opt_weights_by_year, cf_by_year = {}, {}

    if verbose:
        print("\nOptimisation sous contrainte carbone (schéma 2 .1)…")

    for year in sorted(filtered_windows):

        # ───────────────────────── Données disponibles ? ─────────────────────────
        if (
            year not in filtered_annual_data
            or year not in mvp_carbon_footprint
            or 'alpha_period' not in filtered_windows[year]
        ):
            if verbose:
                print(f"⚠️ {year}: données manquantes → année sautée")
            continue

        ann = filtered_annual_data[year]
        for k in ('scope1', 'scope2', 'ycap'):
            if k not in ann:
                if verbose:
                    print(f"⚠️ {year}: {k} absent → année sautée")
                break
        else:
            # ───────────────────────── Préparation des séries ─────────────────────────
            # scope1, scope2, ycap sont indexés par Period('YYYY', 'Y') ou par int/str
            scope1 = ann['scope1'].iloc[-1]  # Series (t)
            scope2 = ann['scope2'].iloc[-1]
            cap    = ann['ycap' ].iloc[-1]   # Series (MUSD)

            returns = filtered_windows[year]['alpha_period']

            # Entreprises communes et ordre unique
            common = (
                scope1.index
                .intersection(scope2.index)
                .intersection(cap.index)
                .intersection(returns.columns)
            )
            common = sorted(common)                      # fixe l’ordre une fois pour toutes
            if len(common) == 0:
                if verbose:
                    print(f"⚠️ {year}: aucune intersection d’entreprises")
                continue

            # ↳ filtrage
            emis   = (scope1[common] + scope2[common]).astype(float)   # t
            cap_m  = cap[common].astype(float)                         # MUSD
            R      = returns[common]                                   # DataFrame
            Σ      = R.cov().values                                    # matrice var-cov

            # Intensité carbone (t / MUSD) par titre
            ci = (emis / cap_m.replace(0, np.nan)).fillna(0).values

            # Contrainte : 50 % du CF_MVP (déjà en t/MUSD)
            cf_target = 0.5 * mvp_carbon_footprint[year]

            # ─────────────────────────  Optimisation SLSQP  ─────────────────────────
            n = len(common)
            def objective(x):           # variance du portefeuille
                return 10_000 * x @ Σ @ x

            cons = [
                {'type': 'eq',   'fun': lambda x: x.sum() - 1},
                {'type': 'ineq', 'fun': lambda x, ci=ci: cf_target - (x * ci).sum()}
            ]
            bounds = [(0, 1)] * n
            x0 = np.full(n, 1 / n)

            res = minimize(
                objective, x0, method='SLSQP',
                bounds=bounds, constraints=cons
            )

            if not res.success:
                if verbose:
                    print(f"❌ {year}: optimisation échouée – {res.message}")
                continue

            w_opt = pd.Series(res.x, index=common, name=year)
            cf_opt = (w_opt.values * ci).sum()          # t/MUSD

            # ──────────── stockage & diagnostics ────────────
            opt_weights_by_year[year] = w_opt
            cf_by_year[year] = cf_opt

            if verbose:
                print(
                    f"✅ {year}: n={n:3d}  CF_MVP={mvp_carbon_footprint[year]:6.2f}  "
                    f"cible={cf_target:6.2f}  CF_opt={cf_opt:6.2f}"
                )

    return opt_weights_by_year, cf_by_year


# In[208]:


# Calcul du portefeuille avec contrainte carbone P(0.5)
# constrained_weights, constrained_cf_values = compute_carbon_constrained_portfolio(
#     windows_,  # Vos fenêtres filtrées
#     annual_filtered_data_,  # Données annuelles filtrées
#     carbon_footprint_mvp  # Empreinte carbone du MVP
# )


# In[209]:


# Après avoir calculé les poids optimaux

# Fonction qui calcule ou charge les poids optimaux
def get_optimal_weights_p05(pickle_weights_window, 
                            windows_, 
                            annual_filtered_data_,
                            carbon_footprint_mvp,
                            force_recalculate=False):
    """
    Récupère les poids optimaux, soit depuis un fichier sauvegardé, soit en les recalculant.
  
    Parameters:
    -----------
    pickle_weights_window : str
        Nom du dossier pour sauvegarder les poids
    windows_ : dict
        Dictionnaire des fenêtres pour l'optimisation
    annual_filtered_data_ : dict
        Données annuelles filtrées
    carbon_footprint_mvp : float
        Empreinte carbone du MVP
    force_recalculate : bool, optional
        Si True, force le recalcul même si le fichier existe déjà
      
    Returns:
    --------
    tuple
        Tuple contenant (constrained_weights, constrained_cf_values)
    """
    pickle_path = 'pickle_weights/' + f'{pickle_weights_window}' + '/optimal_weights_MVP50.pkl'
  
    # Si le fichier existe et qu'on ne force pas le recalcul
    if os.path.exists(pickle_path) and not force_recalculate:
        print("📂 Chargement des poids optimaux depuis le fichier...")
        with open(pickle_path, 'rb') as f:
            optimal_weights = pickle.load(f)
            return optimal_weights['constrained_weights'], optimal_weights['constrained_cf_values']
    else:
        print("🔄 Calcul des poids optimaux en cours...")
        # Calcul des poids optimaux
        constrained_weights, constrained_cf_values = compute_carbon_constrained_portfolio(
                                    windows_,  # Vos fenêtres filtrées
                                    annual_filtered_data_,  # Données annuelles filtrées
                                    carbon_footprint_mvp  # Empreinte carbone du MVP
                                )
      
        # Création d'une structure pour stocker les deux dictionnaires
        optimal_weights = {
            'constrained_weights': constrained_weights,
            'constrained_cf_values': constrained_cf_values
        }
        
        # Sauvegarde pour une utilisation future
        with open(pickle_path, 'wb') as f:
            pickle.dump(optimal_weights, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print("✅ Poids optimaux calculés et sauvegardés")
      
        return constrained_weights, constrained_cf_values

# # Utilisation
constrained_weights, constrained_cf_values = get_optimal_weights_p05(pickle_weights_window, 
                                                                     windows_,
                                                                     annual_filtered_data_,
                                                                     carbon_footprint_mvp,
                                                                     force_recalculate=False)


# In[210]:


updated_constrained_weights = update_portfolio_weights(constrained_weights, windows_) # La fonction est définie plus haut pour le MVP


# In[211]:


# Audit manuel si updated_constrained_weights a bien fait la mise à jour
# Produit élément par élément: chaque colonne multiplie son poids associé

product_df_ = windows_['2014']['test_period'].loc['2014-01-01':'2014-01-31'] * constrained_weights['2014']

rp1_ = product_df_.sum(axis=1).to_list()[0]  # Initialisation trouver r_p dénominateur 
        # alpha_Y * r_i pour trouver r_p initial
        
rp_1_ = 1 + rp1_ # ajouter 1 à r_p 


r_i_1_ = windows_['2014']['test_period'].loc['2014-01-01':'2014-01-31'] +1 # trouver (1 + r_i)
frac_ri_ = r_i_1_ / rp_1_ # fraction qui met à jour le premier mois des alphas

alpha_1_ = constrained_weights['2014'] * frac_ri_
# print(alpha_1_)
# print(updated_constrained_weights['2014'].head(1))
print('r_p premier mois:', rp1_) 


# In[212]:


mvpc = compute_portfolio_returns(updated_constrained_weights, windows_, constrained_weights) # La fonction est définie plus haut pour le MVP
mvpc.index = pd.to_datetime(mvpc.index)
# mvpc.head()


# ### 2.2
# $$\textcolor{cyan}{\text{Metrics Graphics}}$$


# In[213]:


# Fonction pour calculer le tracking error annualisé entre deux portefeuilles
def calculate_tracking_error(returns, benchmark_returns):
    """
    Calcule le tracking error annualisé entre un portefeuille et son benchmark.
    
    Parameters:
    -----------
    returns : pd.Series
        Série de rendements du portefeuille
    benchmark_returns : pd.Series
        Série de rendements du benchmark
        
    Returns:
    --------
    float
        Tracking error annualisé
    """
    # Assurer l'alignement des deux séries
    common_index = returns.index.intersection(benchmark_returns.index)
    returns_aligned = returns.loc[common_index]
    benchmark_aligned = benchmark_returns.loc[common_index]
    
    # Calculer les différences de rendement
    diff_returns = returns_aligned - benchmark_aligned
    
    # Calculer l'écart-type des différences et annualiser
    tracking_error = diff_returns.std() * np.sqrt(12)  # Pour données mensuelles
    
    return tracking_error

# Fonction générique pour comparer plusieurs portefeuilles
def compare_portfolio_metrics(portfolios_dict, benchmark_name=None, rf_series=None, 
                             carbon_metrics_dict=None, latex_path=None, latex_column_names=None):
    """
    Compare les métriques de performance de plusieurs portefeuilles, y compris 
    le tracking error et l'empreinte carbone moyenne.
    
    Parameters:
    -----------
    portfolios_dict : dict
        Dictionnaire avec les séries de rendements de chaque portefeuille
    benchmark_name : str, optional
        Nom du portefeuille de référence pour le calcul du tracking error
    rf_series : pd.Series, optional
        Série du taux sans risque pour les ratios de Sharpe
    carbon_metrics_dict : dict, optional
        Dictionnaire d'empreinte carbone par portefeuille
    latex_path : str, optional
        Chemin pour sauvegarder en format LaTeX
    latex_column_names : dict, optional
        Noms LaTeX pour les colonnes du tableau
        
    Returns:
    --------
    pd.DataFrame
        Tableau des métriques comparatives
    """
    # Calculer les métriques standards
    portfolio_metrics = {}
    for name, returns in portfolios_dict.items():
        portfolio_metrics[name] = calculate_portfolio_metrics(returns, rf_series)
        
        # Ajouter l'empreinte carbone moyenne si disponible
        if carbon_metrics_dict and name in carbon_metrics_dict and carbon_metrics_dict[name]:
            portfolio_metrics[name]["Annual Footprint Average"] = np.mean(list(carbon_metrics_dict[name].values()))
        else:
            portfolio_metrics[name]["Annual Footprint Average"] = np.nan
    
    # Créer le DataFrame des métriques
    metrics_df = pd.DataFrame(portfolio_metrics)
    
    # Calculer le tracking error vs benchmark
    if benchmark_name and benchmark_name in portfolios_dict:
        benchmark_returns = portfolios_dict[benchmark_name]
        for name, returns in portfolios_dict.items():
            if name != benchmark_name:
                metrics_df.loc[f"Tracking Error vs {benchmark_name} (ann.)", name] = calculate_tracking_error(returns, benchmark_returns)
    
    # Exporter au format LaTeX si demandé
    if latex_path:
        latex_df = metrics_df.copy()
        if latex_column_names:
            latex_df = latex_df.rename(columns=latex_column_names)
        latex_df.to_latex(latex_path, index=True, float_format="%.5f")
    
    return metrics_df

# Extraire la série de rendements du DataFrame mvpc avant de la passer à la fonction
mvpc_series = mvpc['Portfolio_Return']

# Calculer les métriques standards
mvp_metrics = calculate_portfolio_metrics(monthly_returns['MVP'], rf[rf_column])
mvpc_metrics = calculate_portfolio_metrics(mvpc_series, rf[rf_column])
vw_metrics = calculate_portfolio_metrics(monthly_returns['VW'], rf[rf_column])

# Calculer le tracking error pour MVP50 vs MVP
tracking_error_mvpc_vs_mvp = calculate_tracking_error(mvpc_series, monthly_returns['MVP'])

# Calculer les moyennes d'empreinte carbone annuelle
annual_fp_mvp = np.mean(list(carbon_footprint_mvp.values())) if carbon_footprint_mvp else np.nan
annual_fp_mvpc = np.mean(list(constrained_cf_values.values())) if constrained_cf_values else np.nan
annual_fp_vw = np.mean(list(carbon_footprint_vw.values())) if carbon_footprint_vw else np.nan

# Create a DataFrame to display the results side by side
metrics_df = pd.DataFrame({
    'MVP': pd.Series(mvp_metrics),
    'VW': pd.Series(vw_metrics),
    'MVP50': pd.Series(mvpc_metrics)
})

# Ajouter le tracking error et l'empreinte carbone moyenne
metrics_df.loc["Annual Footprint Average", 'MVP'] = annual_fp_mvp
metrics_df.loc["Annual Footprint Average", 'VW'] = annual_fp_vw
metrics_df.loc["Annual Footprint Average", 'MVP50'] = annual_fp_mvpc
metrics_df.loc["TE (ann.) MVP-MVP50", 'MVP50'] = tracking_error_mvpc_vs_mvp

print(metrics_df)

metrics_df_to_latex = metrics_df.rename(columns={
    'MVP': r'$P_{oos}^{(mv)}$',
    'VW': r'$P_{oos}^{(vw)}$',
    'MVP50': r'$P_{oos}^{(mv)}(0.5)$'
})

# metrics_df_to_latex.to_latex('Article_Latex/mvp_vw_mvp05_metrics.tex', index=True, float_format="%.5f")


# In[214]:


def create_portfolio_comparison_dashboard(
    portfolio1_series, 
    portfolio2_series, 
    portfolio3_series=None,
    rf_series=None, 
    portfolio1_name="Portfolio 1", 
    portfolio2_name="Portfolio 2", 
    portfolio3_name="Portfolio 3",
    colors=None, 
    figsize=(16, 12), 
    save_path=None,
    title="Portfolio Comparison",
    # Nouveaux paramètres de contrôle pour les shifts
    cumulative_return_horizontal_shift=5,  # Contrôle le shift horizontal des valeurs finales
    cumulative_return_vertical_spacing=5,  # Contrôle l'espacement vertical entre les annotations
    drawdown_horizontal_shift=30,          # Contrôle le shift horizontal des annotations de drawdown
    drawdown_arc_radius=0.3               # Contrôle la courbure des flèches de drawdown
):
    """
    Crée un tableau de bord simplifié comparant deux ou trois portefeuilles avec:
    1. Rendements cumulés (en haut, plus grand)
    2. Drawdowns (en bas, plus petit)
    
    Paramètres:
    -----------
    portfolio1_series, portfolio2_series, portfolio3_series: pd.Series
        Séries temporelles des rendements des portefeuilles
    rf_series: pd.Series, optional
        Série temporelle du taux sans risque
    portfolio1_name, portfolio2_name, portfolio3_name: str
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
    
    # Vérifier que les index des séries sont alignés
    all_series = [portfolio1_series, portfolio2_series]
    if portfolio3_series is not None:
        all_series.append(portfolio3_series)
    
    # Trouver l'index commun à toutes les séries
    common_index = all_series[0].index
    for series in all_series[1:]:
        common_index = common_index.intersection(series.index)
    
    if len(common_index) == 0:
        raise ValueError("Les séries de portefeuilles n'ont pas d'index commun")
    
    # Filtrer chaque série pour ne conserver que l'index commun
    portfolio1_series = portfolio1_series.loc[common_index]
    portfolio2_series = portfolio2_series.loc[common_index]
    if portfolio3_series is not None:
        portfolio3_series = portfolio3_series.loc[common_index]
    
    print(f"⚠️ Les index des séries ont été alignés. {len(common_index)} points communs conservés.")
    
    # Définir les couleurs par défaut si non spécifiées
    if colors is None:
        colors = {
            portfolio1_name: '#1f77b4',  # Bleu
            portfolio2_name: '#ff7f0e',  # Orange
            portfolio3_name: '#2ca02c'   # Vert
        }
    
    # S'assurer que tous les index sont convertis au même format datetime
    all_series_processed = []
    for series in all_series:
        series.index = pd.to_datetime(series.index)
        series.index = series.index.normalize()
        all_series_processed.append(series)
    
    portfolio1_series = all_series_processed[0]
    portfolio2_series = all_series_processed[1]
    if portfolio3_series is not None:
        portfolio3_series = all_series_processed[2]
    
    # Calculate cumulative returns
    portfolio1_cumulative_returns = (1 + portfolio1_series).cumprod()
    portfolio2_cumulative_returns = (1 + portfolio2_series).cumprod()
    
    # DataFrame pour la comparaison des rendements cumulatifs
    cumulative_returns_comparison = {
        portfolio1_name: portfolio1_cumulative_returns,
        portfolio2_name: portfolio2_cumulative_returns
    }
    
    # Ajouter le troisième portefeuille s'il est fourni
    if portfolio3_series is not None:
        portfolio3_cumulative_returns = (1 + portfolio3_series).cumprod()
        cumulative_returns_comparison[portfolio3_name] = portfolio3_cumulative_returns
    
    cumulative_returns_df = pd.DataFrame(cumulative_returns_comparison)

    # Calculer les drawdowns pour chaque portefeuille
    # Pour le premier portefeuille - Correction du symbole de division manquant
    portfolio1_drawdown = pd.Series(0.0, index=portfolio1_cumulative_returns.index, dtype=float)
    portfolio1_cummax = portfolio1_cumulative_returns.cummax()
    mask1 = portfolio1_cummax != 0
    portfolio1_drawdown.loc[mask1] = ((portfolio1_cummax.loc[mask1] - portfolio1_cumulative_returns.loc[mask1]) / 
                                    portfolio1_cummax.loc[mask1]).astype(float)

    # Pour le deuxième portefeuille - Correction du symbole de division manquant
    portfolio2_drawdown = pd.Series(0.0, index=portfolio2_cumulative_returns.index, dtype=float)
    portfolio2_cummax = portfolio2_cumulative_returns.cummax()
    mask2 = portfolio2_cummax != 0
    portfolio2_drawdown.loc[mask2] = ((portfolio2_cummax.loc[mask2] - portfolio2_cumulative_returns.loc[mask2]) / 
                                    portfolio2_cummax.loc[mask2]).astype(float)
    
    # Pour le troisième portefeuille (si fourni) - Correction du symbole de division manquant
    portfolio3_drawdown = None
    if portfolio3_series is not None:
        portfolio3_drawdown = pd.Series(0.0, index=portfolio3_cumulative_returns.index, dtype=float)
        portfolio3_cummax = portfolio3_cumulative_returns.cummax()
        mask3 = portfolio3_cummax != 0
        portfolio3_drawdown.loc[mask3] = ((portfolio3_cummax.loc[mask3] - portfolio3_cumulative_returns.loc[mask3]) / 
                                        portfolio3_cummax.loc[mask3]).astype(float)

    # Obtenir les valeurs maximales de drawdown
    portfolio1_max_dd = portfolio1_drawdown.max() 
    portfolio2_max_dd = portfolio2_drawdown.max()
    portfolio3_max_dd = None
    if portfolio3_drawdown is not None:
        portfolio3_max_dd = portfolio3_drawdown.max()

    # Set up a modern style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Create a function for custom styling
    def apply_custom_style(ax):
        """Apply custom styling to a matplotlib axis"""
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.7, color='#cccccc')
        ax.xaxis.label.set_fontsize(12)
        ax.yaxis.label.set_fontsize(12)
        ax.tick_params(axis='both', which='major', labelsize=10, colors='#333333')
        return ax

    # Create the dashboard with two main visualizations
    fig = plt.figure(figsize=figsize, facecolor='white', dpi=50)
    
    # Définir une GridSpec avec 2 lignes, une plus grande pour les rendements cumulés (75%)
    # et une plus petite pour les drawdowns (25%)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[0.80, 0.20])
    
    # 1. Cumulative Returns (en haut, plus grand)
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(cumulative_returns_df.index, cumulative_returns_df[portfolio1_name], 
            color=colors[portfolio1_name], linewidth=2, label=portfolio1_name)
    ax1.plot(cumulative_returns_df.index, cumulative_returns_df[portfolio2_name], 
            color=colors[portfolio2_name], linewidth=2, label=portfolio2_name)
    
    # Ajouter le troisième portefeuille s'il existe
    if portfolio3_name in cumulative_returns_df.columns:
        ax1.plot(cumulative_returns_df.index, cumulative_returns_df[portfolio3_name], 
                color=colors[portfolio3_name], linewidth=2, label=portfolio3_name)

    # MODIFICATION: Amélioration du placement des valeurs finales à droite avec shift vertical et horizontal
    # Créer un dictionnaire des valeurs finales
    final_values = {
        portfolio1_name: cumulative_returns_df[portfolio1_name].iloc[-1],
        portfolio2_name: cumulative_returns_df[portfolio2_name].iloc[-1],
    }
    
    if portfolio3_name in cumulative_returns_df.columns:
        final_values[portfolio3_name] = cumulative_returns_df[portfolio3_name].iloc[-1]
    
    # Trier les portefeuilles par valeur finale
    sorted_portfolios = sorted(final_values.items(), key=lambda x: x[1])
    
    # Placer les annotations avec les shifts contrôlés
    vertical_spacing = cumulative_return_vertical_spacing  # Contrôle l'espacement vertical
    
    for i, (name, value) in enumerate(sorted_portfolios):
        # Le shift horizontal est désormais constant pour toutes les annotations
        horizontal_shift = cumulative_return_horizontal_shift
        
        # Calculer le shift vertical pour éviter les chevauchements
        if i == 0:  # Premier portefeuille (valeur la plus basse)
            vertical_shift = 0
        else:
            # Distance entre la valeur actuelle et celle du portefeuille précédent
            prev_value = sorted_portfolios[i-1][1]
            diff = value - prev_value
            
            # Si la différence est petite, augmenter le shift vertical pour éviter les chevauchements
            if diff < 0.2:  # Seuil ajustable
                vertical_shift = -vertical_spacing * (len(sorted_portfolios) - i)
            else:
                vertical_shift = 0
        
        ax1.annotate(f'{value:.2f}x', 
                    xy=(cumulative_returns_df.index[-1], value),
                    xytext=(horizontal_shift, vertical_shift),  # Shift horizontal et vertical
                    textcoords='offset points',
                    ha='left', va='center', 
                    fontweight='bold', 
                    color=colors[name])

    # Ajouter le texte "Cumulative Return Growth of 1 USD" en bas à droite en gris discret
    ax1.annotate('Cumulative Return Growth of 1 USD', 
                xy=(0.95, 0.02),  # Position en bas à droite (coordonnées relatives)
                xycoords='axes fraction',  # Coordonnées relatives à l'axe
                ha='right', va='bottom',  # Alignement
                fontsize=10, color='#333333',  # Gris discret
                style='italic')  # Style italique pour plus de discrétion
                
    ax1.set_ylabel('Cumulative Return')
    
    # Hide x-axis labels for the top plot
    ax1.tick_params(axis='x', labelbottom=False)
    
    # Créer la légende pour les courbes comme dans la deuxième fonction
    ax1.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=10)
    apply_custom_style(ax1)

    # 2. Drawdowns (en bas, plus petit)
    ax2 = fig.add_subplot(gs[1])

    # Plot drawdowns with proper handling of values
    ax2.fill_between(portfolio1_drawdown.index, 0, -portfolio1_drawdown.values, 
                    alpha=0.3, color=colors[portfolio1_name], 
                    label=portfolio1_name, step=None)

    ax2.fill_between(portfolio2_drawdown.index, 0, -portfolio2_drawdown.values, 
                    alpha=0.3, color=colors[portfolio2_name], 
                    label=portfolio2_name, step=None)
    
    # Ajouter le drawdown du troisième portefeuille s'il existe
    if portfolio3_drawdown is not None:
        ax2.fill_between(portfolio3_drawdown.index, 0, -portfolio3_drawdown.values, 
                        alpha=0.3, color=colors[portfolio3_name], 
                        label=portfolio3_name, step=None)

    ax2.plot(portfolio1_drawdown.index, -portfolio1_drawdown.values, alpha=0.7, color=colors[portfolio1_name], linewidth=1)
    ax2.plot(portfolio2_drawdown.index, -portfolio2_drawdown.values, alpha=0.7, color=colors[portfolio2_name], linewidth=1)
    
    # Ajouter la ligne de drawdown du troisième portefeuille s'il existe
    if portfolio3_drawdown is not None:
        ax2.plot(portfolio3_drawdown.index, -portfolio3_drawdown.values, alpha=0.7, color=colors[portfolio3_name], linewidth=1)

    # Find maximum drawdowns dates
    portfolio1_valley_date = portfolio1_drawdown.idxmax()
    portfolio2_valley_date = portfolio2_drawdown.idxmax()
    portfolio3_valley_date = None
    if portfolio3_drawdown is not None:
        portfolio3_valley_date = portfolio3_drawdown.idxmax()

    # MODIFICATION: Amélioration du placement des annotations de Max DD avec flèches externes et arcs
    # Collecter toutes les informations sur les drawdowns
    drawdowns = [
        (portfolio1_name, portfolio1_max_dd, portfolio1_valley_date, colors[portfolio1_name]),
        (portfolio2_name, portfolio2_max_dd, portfolio2_valley_date, colors[portfolio2_name])
    ]
    
    if portfolio3_drawdown is not None:
        drawdowns.append((portfolio3_name, portfolio3_max_dd, portfolio3_valley_date, colors[portfolio3_name]))
    
    # Trier par date pour une meilleure organisation des annotations
    drawdowns.sort(key=lambda x: x[2])
    
    # Calculer la valeur de drawdown maximale pour dimensionner la zone des annotations
    max_dd_value = max([dd[1] for dd in drawdowns])
    
    # Positionner les annotations à l'extérieur avec des flèches courbes
    for i, (name, max_dd, valley_date, color) in enumerate(drawdowns):
        # Décaler les annotations horizontalement pour éviter les chevauchements
        # Alternance droite/gauche pour les annotations
        horizontal_shift = drawdown_horizontal_shift * (-1 if i % 2 == 0 else 1)
        
        # Position de l'annotation (à l'extérieur du graphique avec shift horizontal)
        annotation_x = mdates.date2num(valley_date)
        annotation_y = -max_dd / 2  # Positionnement vertical à mi-hauteur du drawdown
        
        # Direction de l'arc basée sur la position (gauche ou droite)
        arc_rad = drawdown_arc_radius * (-1 if i % 2 == 0 else 1)
        
        ax2.annotate(f'Max DD: -{max_dd:.2%}',
                    xy=(valley_date, -max_dd),  # Point à annoter (drawdown max)
                    xytext=(horizontal_shift, 0),  # Shift horizontal
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
    
    # Ajuster les limites pour avoir une marge en bas et en haut
    # pour accommoder les annotations extérieures
    all_max_dd = max([dd[1] for dd in drawdowns])
    ax2.set_ylim(-all_max_dd * 1.3, all_max_dd * 0.4)  # Plus d'espace en haut pour les annotations
    
    # Légende pour le graphique de drawdowns
    ax2.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=10)
    apply_custom_style(ax2)

    # Ajuster la mise en page avec peu d'espace entre les graphiques
    # plt.tight_layout()
    
    # Ajouter le titre principal tout en haut et en gras
    fig.suptitle(title, fontsize=18, fontweight='bold', x=0.5, y=0.98, ha='center')
    
    # Ajuster l'espace pour coller les graphiques (hspace=0.0) et accommoder le titre
    plt.subplots_adjust(top=0.92, hspace=0.0)
    
    if save_path:
        plt.savefig(save_path, dpi=50, bbox_inches='tight')
        print(f"Dashboard saved to {save_path}")
    
    return fig, [ax1, ax2]

fig, axes = create_portfolio_comparison_dashboard(
    vw_series,           # Premier portefeuille (Value-Weighted)
    mvp_series,          # Deuxième portefeuille (Minimum-Variance)
    mvpc_series,         # Troisième portefeuille (MVP avec contrainte carbone 50%)
    portfolio1_name=r'$P^{(v w)}$',
    portfolio2_name=r'$P_{o o s}^{(m v)}$',
    portfolio3_name=r'$P_{o o s}^{(m v)}(0.5)$',
    save_path="Three_Portfolio_Comparison.png" if save_images else None,
    colors={
        r'$P^{(v w)}$': '#1f77b4',            # Bleu pour VW
        r'$P_{o o s}^{(m v)}$': '#ff7f0e',    # Orange pour MVP
        r'$P_{o o s}^{(m v)}(0.5)$': '#2ca02c' # Vert pour MVP05
    },
    figsize=(22, 8),
    title=r'Comparison between $P^{(v w)}$, $P_{o o s}^{(m v)}$ and $P_{o o s}^{(m v)}(0.5)$',
    # Paramètres de contrôle des shifts
    cumulative_return_horizontal_shift=10,
    cumulative_return_vertical_spacing=-5,
    drawdown_horizontal_shift=40,
    drawdown_arc_radius=0.3
)


# In[215]:


def plot_carbon_trajectory_and_tracking_error(mvpc_cf_values, mvp_cf_values, 
                                             mvpc_returns, mvp_returns,
                                             figsize=(14, 10), save_path=None,
                                             colors=None):
    """
    Creates a professional visualization of the carbon constrained portfolio (P005) showing:
    1. Carbon trajectory compared to MVP and constraint target (top panel)
    2. Tracking error versus the MVP portfolio (bottom panel)
    
    Parameters
    ----------
    mvpc_cf_values : dict
        Carbon footprint values for the constrained portfolio (P005) by year
    mvp_cf_values : dict
        Carbon footprint values for the standard MVP portfolio by year
    mvpc_returns : pd.Series
        Monthly returns of the constrained portfolio
    mvp_returns : pd.Series
        Monthly returns of the standard MVP portfolio
    figsize : tuple, optional
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure
    colors : dict, optional
        Custom colors for different elements
        
    Returns
    -------
    fig, axes : tuple
        Figure and axes objects for further customization if needed
    """
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch
    from scipy import stats
    # Palette de couleurs personnalisée pour le graphique inférieur
    te_c = [
        (0.90, 0.90, 0.90),  # Gris clair (plus visible)
        (0.75, 0.76, 0.80),  # Gris bleuté (plus foncé)
        (0.60, 0.65, 0.75),  # Gris bleu moyen
        (0.45, 0.50, 0.60)   # Gris bleu foncé
    ]
    pastel_viridis = LinearSegmentedColormap.from_list('pastel_viridis', te_c)
    
    # Set default colors if not provided
    if colors is None:
        colors = {
            'mvpc': '#026315',  # Green for carbon constrained portfolio
            'mvp': '#ff7f0e',   # Orange for standard MVP
            'target': '#d62728', # Red for target constraint
            'tracking_error': '#1f77b4',  # Blue for tracking error
            'positive_te': '#2ca02c',     # Dark green for positive tracking error
            'negative_te': '#d62728'      # Dark red for negative tracking error
        }
    
    # Set professional style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create figure with GridSpec for layout control with hspace=0 to make them connected
    fig = plt.figure(figsize=figsize, dpi=50, facecolor='white')
    gs = GridSpec(2, 1, height_ratios=[5, 1], hspace=0.05)
    
    # ====================== TOP PANEL: CARBON TRAJECTORY ====================== #
    ax1 = fig.add_subplot(gs[0])
    
    # Get common years and sort chronologically
    common_years = sorted(set(mvpc_cf_values) & set(mvp_cf_values))
    numeric_years = [int(y) for y in common_years]
    x_positions = np.arange(len(numeric_years))
    
    # Extract values
    mvpc_values = [mvpc_cf_values[y] for y in common_years]
    mvp_values = [mvp_cf_values[y] for y in common_years]
    carbon_constraints = [0.5 * mvp_cf_values[y] for y in common_years]
    
    # Calculate statistics
    avg_reduction = np.mean([1 - mvpc_values[i]/mvp_values[i] for i in range(len(mvpc_values))])
    min_reduction = min([1 - mvpc_values[i]/mvp_values[i] for i in range(len(mvpc_values))])
    max_reduction = max([1 - mvpc_values[i]/mvp_values[i] for i in range(len(mvpc_values))])
    
    # Create bar chart for carbon footprint comparison
    bar_width = 0.35
    bars1 = ax1.bar(x_positions - bar_width/2, mvp_values, bar_width, color=colors['mvp'], 
                    alpha=0.85, label=r'$P_{oos}^{(mv)}$ Carbon Footprint', edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x_positions + bar_width/2, mvpc_values, bar_width, color=colors['mvpc'], 
                    alpha=0.85, label=r'$P_{oos}^{(mv)}(0.5)$ Carbon Footprint', edgecolor='black', linewidth=0.5)
    
    # Add target constraint line
    ax1.plot(x_positions, carbon_constraints, '--', linewidth=1.8, color=colors['target'], 
             label='Carbon Constraint (50% of MVP)')
    
    # Fill area below constraint to highlight target zone
    ax1.fill_between(x_positions, [0]*len(numeric_years), carbon_constraints, 
                     color=colors['target'], alpha=0.1)
    
    # Add reduction percentage labels above the MVPC bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        reduction_pct = (1 - bar2.get_height()/bar1.get_height()) * 100
        ax1.annotate(f'{reduction_pct:.1f}%', 
                     xy=(bar2.get_x() + bar2.get_width()/2, bar2.get_height()),
                     xytext=(0, 5), textcoords='offset points',
                     ha='center', va='bottom', fontsize=9, fontweight='bold',
                     color=colors['mvpc'])
        
        # Add actual values on bars
        ax1.annotate(f'{bar1.get_height():.0f}',
                     xy=(bar1.get_x() + bar1.get_width()/2, bar1.get_height()),
                     xytext=(0, -15), textcoords='offset points', 
                     ha='center', va='top', fontsize=8,
                     color='white', fontweight='bold')
        
        ax1.annotate(f'{bar2.get_height():.0f}',
                     xy=(bar2.get_x() + bar2.get_width()/2, bar2.get_height()),
                     xytext=(0, -15), textcoords='offset points', 
                     ha='center', va='top', fontsize=8,
                     color='white', fontweight='bold')
    
    # Formatting top panel
    ax1.set_ylabel(r'Carbon Footprint (tCO$ _2$e/M\$)', fontsize=12)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([])  # Hide x-tick labels for the top panel
    ax1.grid(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=8)  # Smaller legend
    ax1.set_axisbelow(True)
    
    # Add annotation about the constraint area
    ax1.annotate('Target Constraint Zone', 
                xy=(x_positions[len(x_positions)//2], carbon_constraints[len(carbon_constraints)//2]/2),
                xytext=(0, -20), textcoords='offset points',
                ha='center', va='top', fontsize=8, style='italic',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec='#cccccc'),
                arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2", color='#555555'))
    
    # ====================== BOTTOM PANEL: TRACKING ERROR ====================== #
    ax2 = fig.add_subplot(gs[1])
    
    # Calculate tracking error (monthly return differences)
    mvpc_returns.index = pd.to_datetime(mvpc_returns.index)
    mvp_returns.index = pd.to_datetime(mvp_returns.index)
    
    # Align return series by index
    common_dates = mvpc_returns.index.intersection(mvp_returns.index)
    tracking_error = (mvpc_returns.loc[common_dates] - mvp_returns.loc[common_dates])
    
    # Calculate annualized tracking error
    annual_te = tracking_error.std() * np.sqrt(12) * 100  # Annualized and in percentage
    
    # Group tracking error by year for consistent x-axis with the carbon panel
    yearly_te = {}
    for year in numeric_years:
        year_mask = tracking_error.index.year == year
        if year_mask.any():
            yearly_te[str(year)] = tracking_error.loc[year_mask].values
    
    # Calculate average absolute TE by year
    avg_abs_te_by_year = [np.mean(np.abs(yearly_te[str(year)])) * 100 if str(year) in yearly_te else np.nan 
                         for year in numeric_years]
    
    # Utiliser la colormap personnalisée, normaliser les données entre 0-1 pour les couleurs
    abs_te_max = np.nanmax(avg_abs_te_by_year)
    if abs_te_max > 0:
        # Normaliser les valeurs entre 0 et 1
        color_idx = np.array(avg_abs_te_by_year) / abs_te_max
        # Remplacer les NaN par 0 pour éviter les erreurs
        color_idx = np.nan_to_num(color_idx)
        # Utiliser la colormap personnalisée pour chaque année
        bar_colors = [pastel_viridis(idx) for idx in color_idx]
    else:
        # Utiliser une couleur par défaut si toutes les valeurs sont NaN ou 0
        bar_colors = [pastel_viridis(0.5)] * len(avg_abs_te_by_year)
    
    # Tracer l'erreur de suivi absolue moyenne par année sous forme de barres avec le dégradé
    bars3 = ax2.bar(x_positions, avg_abs_te_by_year, color=bar_colors, alpha=0.7,
                   width=0.6, edgecolor='black', linewidth=0.5)
    
    # Add line for the annualized TE across the whole period
    ax2.axhline(annual_te, linestyle='--', color='#d62728', linewidth=1.5, 
               label=f'Annualized TE: {annual_te:.2f}%')
    
    
    # Add value labels on bars
    for i, bar in enumerate(bars3):
        if not np.isnan(bar.get_height()):
            ax2.annotate(f'{bar.get_height():.2f}%', 
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=9)
            
    ax2.text(0.02, 0.80, r'$\mathbf{P_{oos}^{(mv)}}$ - $\mathbf{P_{oos}^{(mv)}(0.5)}$ Tracking Error' , transform=ax2.transAxes, fontsize=8,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
            facecolor='white', alpha=0.9))

    # Format tracking error panel
    # ax2.set_title('Tracking Error vs MVP Portfolio', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel('Year', fontsize=12, labelpad=10)
    ax2.set_ylabel('Absolute TE (%)', fontsize=10, labelpad=10)
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(numeric_years, rotation=0)
    ax2.grid(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax2.set_axisbelow(True)

    # Add an overall title
    fig.suptitle(r'Carbon Constrained Portfolio $P_{oos}^{(mv)}(0.5)$ Footprint Trajectory', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.92, hspace=0.25)
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=50, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, (ax1, ax2)


# In[216]:


# Display the carbon trajectory and tracking error visualization
fig, axes = plot_carbon_trajectory_and_tracking_error(
    mvpc_cf_values=constrained_cf_values,  # Carbon footprint of constrained portfolio
    mvp_cf_values=carbon_footprint_mvp,        # Carbon footprint of standard MVP
    mvpc_returns=mvpc_series,              # Monthly returns of constrained portfolio
    mvp_returns=mvp_series,                # Monthly returns of standard MVP
    figsize=(22, 8),
    save_path="carbon_constrained_portfolio_analysis.png" if save_images else None,
    colors={
        'mvpc': '#026315',      # Green for carbon constrained portfolio
        'mvp': '#ff7f0e',       # Orange for standard MVP
        'target': '#d62728',    # Red for target constraint
        'tracking_error': '#657994'  # Blue for tracking error
    }
)


# ### Partie 2.3
# 
# Another interesting decarbonization strategy consists in designing the portfolio that is as close as possible to the benchmark, while reducing the carbon footprint by $25 \%$ (otherwise passive investor). This is done by solving the minimum variance criterion for the tracking error every year:
# 
# $$
# \begin{array}{ll}
# \min {{\alpha Y}} & \left(T E{p, Y}\right)^2=\left(\alpha_Y-\alpha_Y^{(v w)}\right)^{\prime} \Sigma_{Y+1}\left(\alpha_Y-\alpha_Y^{(v w)}\right) \
# \text { s.t. } & \left.C F_Y^{(p)} \leq 0.5 \times C F_Y^{(P(v w)}\right) \
# \text { s.t. } & \alpha_{i, Y} \geq 0 \quad \text { for all } i
# \end{array}
# $$
# 
# where $C F_Y^{\left(P^{(v w)}\right)}=\frac{1}{C a p_Y} \sum_{i=1}^N E_{i, Y}$ denotes the carbon footprint of the value-weighted portfolio, with $C a p_Y=\sum_{i=1}^N C a p_{i, Y}$ the total market value of the investment set.
# 
# We call this portfolio " $P_{o o s}^{(v w)}(0.5)$ ". Compute the characteristics of the portfolio over the sample. Again, plot the cumulative return series of both strategies and compare summary statistics


# In[217]:


print('\n\n\n\n\n\n')
print('# ------------------------ 2.3  P(vw)(0.5) -------------------------- #')


# In[218]:


def compute_tracking_error_portfolio(
        filtered_windows: dict,
        annual_filtered_data: dict,
        vw_weights_by_year: dict,
        carbon_cut: float = 0.50):
    """
    Portefeuille qui minimise la tracking-error vis-à-vis du VW
    sous contrainte carbone CF ≤ (1–cut) × CF_VW.

    Paramètres
    ----------
    filtered_windows        : rendements mensuels  (dict[year → DataFrame])
    annual_filtered_data    : scope1, scope2, ycap (dict[year → dict])
    vw_weights_by_year      : poids value-weighted (dict[year → Series])
    carbon_cut              : pourcentage de réduction (0.50 = –50 %)

    Retour
    ------
    te_weights      : dict[year → Series (poids optimisés)]
    te_cf_values    : dict[year → float  (CF obtenu)]
    """
    from scipy.optimize import minimize
    import numpy as np
    import pandas as pd

    te_weights, te_cf_values = {}, {}

    print("\n>>> Optimisation TE sous contrainte carbone (CF)")

    for year in sorted(filtered_windows):

        # 1) ---------------- Données et universe commun ----------------------
        if year not in annual_filtered_data or year not in vw_weights_by_year:
            print(f" {year}: données manquantes – ignoré")
            continue

        ret_m = filtered_windows[year]['alpha_period']          # mensuel
        vw_w  = vw_weights_by_year[year].iloc[-1]               # dernière obs. mensuelle

        d = annual_filtered_data[year]
        scope1, scope2, ycap = d['scope1'], d['scope2'], d['ycap']

        common = (set(ret_m.columns) & set(vw_w.index)
                  & set(scope1.columns) & set(scope2.columns) & set(ycap.columns))
        if not common:
            print(f" {year}: aucun titre commun – ignoré")
            continue
        common = sorted(common)

        ret_m      = ret_m[common]
        vw_w       = vw_w[common]
        emis       = (scope1[common] + scope2[common]).iloc[-1]   # tCO2
        cap_musd   = ycap[common].iloc[-1]                        # MUSD
        inten      = emis / cap_musd                              # t/MUSD
        inten      = inten.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # 2) ----------------- CF du benchmark et borne -----------------------
        cf_vw = float((vw_w * inten).sum())                       # Σ α^VW·E/Cap
        cf_max = (1 - carbon_cut) * cf_vw                         # ex : 0.5 × CF_VW

        # 3) ----------------- Covariance pairwise, projection PSD -----------

        Σ = ret_m.cov().values                                    # pairwise only 
        # eigval, eigvec = np.linalg.eigh(Σ)
        # eigval[eigval < 1e-6] = 1e-6                              # ridge
        # Σ_psd = (eigvec * eigval) @ eigvec.T

        vw_arr   = vw_w.values
        inten_arr = inten.values
        n = len(common)

        # 4) ----------------- Optimisation SLSQP ----------------------------

        def obj(x):
            d = x - vw_arr
            return 10000.0 * d @ Σ @ d                         # TE² × 1000

        cons = [{'type': 'eq',  'fun': lambda x: x.sum() - 1},
                {'type': 'ineq','fun': lambda x: cf_max - np.dot(x, inten_arr)}]

        x0 = np.full(n, 1/n)
        res = minimize(obj, x0, method='SLSQP',
                       bounds=[(0,1)]*n, constraints=cons,
                       options={'ftol':1e-6,'maxiter':1000})

        if not res.success:
            print(f" {year}: échec opti → {res.message}")
            continue

        w_opt = pd.Series(res.x, index=common)
        cf_opt = float(np.dot(w_opt, inten))
        
        # -------------------------------------------------------------------
        # Bloc ajouté : on s'assure que l'argument de sqrt est ≥ 0
        # -------------------------------------------------------------------
        
        te_arg = obj(res.x) / 10000.0        # on enlève le facteur 1 0000
        print(10*'=', ' Year ', year, 10*'=')
        
        print(f"Raw TE² before clipping: {te_arg:.18e}")  # 18 chiffres significatifs en notation scientifique
        
        if te_arg < 0:                      # peut arriver à cause des erreurs de
            te_arg = 0.0                    #   virgule flottante (ordre 1e-10)
            print(f"→ TE² was negative by {-te_arg:.18e}, resetting to 0")
        
        te = np.sqrt(te_arg)

        te_weights[year]   = w_opt
        te_cf_values[year] = cf_opt

        print(f" ✓ {year}: TE={te:.4f}  CF={cf_opt:.2f} (borne {cf_max:.2f}, VW {cf_vw:.2f})")

    return te_weights, te_cf_values


# In[219]:


# Calcul du portefeuille avec contrainte de tracking error
# te_weights, te_cf_values = compute_tracking_error_portfolio(
#     windows_,  # Fenêtres filtrées
#     annual_filtered_data_,  # Données annuelles filtrées
#     weighted_mcap_by_year  # Poids VW pour chaque année
# )

# te_w, te_cf = compute_tracking_error_portfolio(
#                  filtered_windows,
#                  annual_filtered_data,
#                  weighted_mcap_by_year,   # tes poids VW
#                  carbon_cut=0.50)         # –50 %


# In[220]:


# Fonction qui calcule ou charge les poids optimaux
def get_optimal_weights_te(pickle_weights_window, 
                           windows_, 
                           annual_filtered_data_,
                           weighted_mcap_by_year,
                           carbon_cut=0.50,
                           force_recalculate=False,
                           use_pickle=True):
    """
    Récupère les poids optimaux, soit depuis un fichier sauvegardé, soit en les recalculant.
  
    Parameters:
    -----------
    pickle_weights_window : str
        Nom du dossier pour sauvegarder les poids
    windows_ : dict
        Dictionnaire des fenêtres pour l'optimisation
    annual_filtered_data_ : dict
        Données annuelles filtrées
    weighted_mcap_by_year : dict
        Dictionnaire des poids de marché par année
    carbon_cut : float, optional
        Pourcentage de réduction de carbone souhaité (default: 0.50)
    force_recalculate : bool, optional
        Si True, force le recalcul même si le fichier existe déjà
    use_pickle : bool, optional
        Si True, utilise le mécanisme de sauvegarde/chargement pickle
      
    Returns:
    --------
    tuple
        Tuple contenant (te_weights, te_cf_values)
    """
    pickle_path = 'pickle_weights/' + f'{pickle_weights_window}' + '/optimal_weights_TE.pkl'
  
    # Si use_pickle est True et que le fichier existe et qu'on ne force pas le recalcul
    if use_pickle and os.path.exists(pickle_path) and not force_recalculate:
        print("📂 Chargement des poids optimaux depuis le fichier...")
        with open(pickle_path, 'rb') as f:
            optimal_weights = pickle.load(f)
            return optimal_weights['te_weights'], optimal_weights['te_cf_values']
    else:
        print("🔄 Calcul des poids optimaux en cours...")
        # Calcul des poids optimaux
        te_weights, te_cf_values = compute_tracking_error_portfolio(
                                    windows_,  # Vos fenêtres filtrées
                                    annual_filtered_data_,  # Données annuelles filtrées
                                    weighted_mcap_by_year,  # Poids de marché par année
                                    carbon_cut
                                )
      
        # Si use_pickle est True, sauvegarde pour une utilisation future
        if use_pickle:
            # Création d'une structure pour stocker les deux dictionnaires
            optimal_weights = {
                'te_weights': te_weights,
                'te_cf_values': te_cf_values
            }
            
            # Assurons-nous que le répertoire existe
            os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
            
            # Sauvegarde pour une utilisation future
            with open(pickle_path, 'wb') as f:
                pickle.dump(optimal_weights, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            print("✅ Poids optimaux calculés et sauvegardés")
        else:
            print("✅ Poids optimaux calculés (sans sauvegarde)")
      
        return te_weights, te_cf_values

# Utilisation
te_weights, te_cf_values = get_optimal_weights_te(pickle_weights_window, 
                                                 windows_,
                                                 annual_filtered_data_,
                                                 weighted_mcap_by_year,
                                                 carbon_cut=0.50,
                                                 force_recalculate=False,
                                                 use_pickle=True)  # Paramètre ajouté


# In[221]:


# Mise à jour des poids du portefeuille sur la période de test
te_updated_weights = update_portfolio_weights(te_weights, windows_)

# Calcul des rendements du portefeuille
te_portfolio_returns = compute_portfolio_returns(te_updated_weights, windows_, te_weights)

te_portfolio_returns.head()


# In[222]:


te_series = te_portfolio_returns['Portfolio_Return']
# Tableau récapitulatif des résultats
te_metrics = calculate_portfolio_metrics(te_series, rf[rf_column])
# Créer un DataFrame pour comparer les trois portefeuilles
metrics_comparison = pd.DataFrame({
    'MVP': pd.Series(mvp_metrics),  # Métriques du portefeuille MVP
    'MVP05': pd.Series(mvpc_metrics),
    'VW': pd.Series(vw_metrics),  # Métriques du portefeuille VW
    'TE': pd.Series(te_metrics)  # Métriques du portefeuille TE
})


# $$\textcolor{cyan}{\text{Metrics}}$$


# In[223]:


# Fonction pour calculer le tracking error annualisé entre deux portefeuilles
def calculate_tracking_error(returns, benchmark_returns):
    """
    Calcule le tracking error annualisé entre un portefeuille et son benchmark.
    
    Parameters:
    -----------
    returns : pd.Series
        Série de rendements du portefeuille
    benchmark_returns : pd.Series
        Série de rendements du benchmark
        
    Returns:
    --------
    float
        Tracking error annualisé
    """
    # Assurer l'alignement des deux séries
    common_index = returns.index.intersection(benchmark_returns.index)
    returns_aligned = returns.loc[common_index]
    benchmark_aligned = benchmark_returns.loc[common_index]
    
    # Calculer les différences de rendement
    diff_returns = returns_aligned - benchmark_aligned
    
    # Calculer l'écart-type des différences et annualiser
    tracking_error = diff_returns.std() * np.sqrt(12)  # Pour données mensuelles
    
    return tracking_error

# Fonction modifiée pour comparer plusieurs portefeuilles avec TE sélectif
def compare_portfolio_metrics_selective_te(portfolios_dict, benchmark_name=None, rf_series=None, 
                                           carbon_metrics_dict=None, latex_path=None, 
                                           latex_column_names=None, show_te_only_for=None):
    """
    Compare les métriques de performance de plusieurs portefeuilles, avec 
    tracking error affiché uniquement pour certains portefeuilles spécifiés.
    
    Parameters:
    -----------
    portfolios_dict : dict
        Dictionnaire avec les séries de rendements de chaque portefeuille
    benchmark_name : str, optional
        Nom du portefeuille de référence pour le calcul du tracking error
    rf_series : pd.Series, optional
        Série du taux sans risque pour les ratios de Sharpe
    carbon_metrics_dict : dict, optional
        Dictionnaire d'empreinte carbone par portefeuille
    latex_path : str, optional
        Chemin pour sauvegarder en format LaTeX
    latex_column_names : dict, optional
        Noms LaTeX pour les colonnes du tableau
    show_te_only_for : list, optional
        Liste des noms de portefeuilles pour lesquels afficher le tracking error
        
    Returns:
    --------
    pd.DataFrame
        Tableau des métriques comparatives
    """
    # Calculer les métriques standards
    portfolio_metrics = {}
    for name, returns in portfolios_dict.items():
        portfolio_metrics[name] = calculate_portfolio_metrics(returns, rf_series)
        
        # Ajouter l'empreinte carbone moyenne si disponible
        if carbon_metrics_dict and name in carbon_metrics_dict and carbon_metrics_dict[name]:
            portfolio_metrics[name]["Annual Footprint Average"] = np.mean(list(carbon_metrics_dict[name].values()))
        else:
            portfolio_metrics[name]["Annual Footprint Average"] = np.nan
    
    # Créer le DataFrame des métriques
    metrics_df = pd.DataFrame(portfolio_metrics)
    
    # Calculer le tracking error vs benchmark seulement pour les portefeuilles spécifiés
    if benchmark_name and benchmark_name in portfolios_dict:
        benchmark_returns = portfolios_dict[benchmark_name]
        for name, returns in portfolios_dict.items():
            if name != benchmark_name:
                te = calculate_tracking_error(returns, benchmark_returns)
                # N'ajouter le TE que pour les portefeuilles spécifiés
                if show_te_only_for is None or name in show_te_only_for:
                    metrics_df.loc[f"TE : {name} vs {benchmark_name} (ann.)", name] = te
    
    # Exporter au format LaTeX si demandé
    if latex_path:
        latex_df = metrics_df.copy()
        if latex_column_names:
            latex_df = latex_df.rename(columns=latex_column_names)
        latex_df.to_latex(latex_path, index=True, float_format="%.5f")
    
    return metrics_df

# Exemple d'utilisation:
# Créer un dictionnaire avec tous les portefeuilles
portfolios = {
    'MVP': monthly_returns['MVP'],
    'VW': monthly_returns['VW'],
    'MVP50': mvpc_series,
    'TE': te_series  # Remplacer par votre série de rendements TE
}

# Dictionnaire d'empreintes carbone
carbon_metrics = {
    'MVP': carbon_footprint_mvp,
    'VW': carbon_footprint_vw,
    'MVP50': constrained_cf_values,
    'TE': te_cf_values  # Remplacer par votre dictionnaire d'empreinte carbone TE
}

# Comparer les métriques, mais n'afficher le tracking error que pour 'VW' et 'TE'
metrics_table = compare_portfolio_metrics_selective_te(
    portfolios_dict=portfolios,
    benchmark_name='VW',  # Utiliser MVP comme benchmark
    rf_series=rf[rf_column],
    carbon_metrics_dict=carbon_metrics,
    latex_path=None,
    latex_column_names={
        'MVP': r'$P_{oos}^{(mv)}$',
        'VW': r'$P_{oos}^{(vw)}$',
        'MVP50': r'$P_{oos}^{(mv)}(0.5)$',
        'TE': r'$P_{oos}^{(vw)}(0.5)$'
    },
    show_te_only_for=['TE']  # Afficher le TE uniquement pour ces portefeuilles
)

print(metrics_table)


# In[224]:


# Créer un dictionnaire de tous les portefeuilles
all_portfolios = {
    'MVP': mvp_series, 
    'MVP05': mvpc_series, 
    'VW': vw_portfolio_returns, 
    'TE': te_series
}

# DataFrame pour stocker la matrice des tracking errors
tracking_error_matrix = pd.DataFrame(index=all_portfolios.keys(), columns=all_portfolios.keys())

# Calculer tous les tracking errors
for benchmark_name in all_portfolios:
    for portfolio_name in all_portfolios:
        if portfolio_name != benchmark_name:
            tracking_error = calculate_tracking_error(all_portfolios[portfolio_name], all_portfolios[benchmark_name])
            tracking_error_matrix.loc[portfolio_name, benchmark_name] = tracking_error
        else:
            tracking_error_matrix.loc[portfolio_name, benchmark_name] = 0.0

# Afficher la matrice des tracking errors
print("Matrice des tracking errors:")
print(tracking_error_matrix)


# In[225]:


def create_portfolio_comparison_dashboard(
    portfolio1_series, 
    portfolio2_series, 
    portfolio3_series=None,
    portfolio4_series=None,  # Ajout du paramètre pour le quatrième portefeuille
    rf_series=None, 
    portfolio1_name="Portfolio 1", 
    portfolio2_name="Portfolio 2", 
    portfolio3_name="Portfolio 3",
    portfolio4_name="Portfolio 4",  # Ajout du nom pour le quatrième portefeuille
    colors=None, 
    figsize=(16, 12), 
    save_path=None,
    title="Portfolio Comparison",
    # Paramètres de contrôle pour les shifts
    cumulative_return_horizontal_shift=5,  # Contrôle le shift horizontal des valeurs finales
    cumulative_return_vertical_spacing=5,  # Contrôle l'espacement vertical entre les annotations
    drawdown_horizontal_shift=30,          # Contrôle le shift horizontal des annotations de drawdown
    drawdown_arc_radius=0.3               # Contrôle la courbure des flèches de drawdown
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
    
    # Vérifier que les index des séries sont alignés
    all_series = [portfolio1_series, portfolio2_series]
    if portfolio3_series is not None:
        all_series.append(portfolio3_series)
    if portfolio4_series is not None:
        all_series.append(portfolio4_series)
    
    # Trouver l'index commun à toutes les séries
    common_index = all_series[0].index
    for series in all_series[1:]:
        common_index = common_index.intersection(series.index)
    
    if len(common_index) == 0:
        raise ValueError("Les séries de portefeuilles n'ont pas d'index commun")
    
    # Filtrer chaque série pour ne conserver que l'index commun
    portfolio1_series = portfolio1_series.loc[common_index]
    portfolio2_series = portfolio2_series.loc[common_index]
    if portfolio3_series is not None:
        portfolio3_series = portfolio3_series.loc[common_index]
    if portfolio4_series is not None:
        portfolio4_series = portfolio4_series.loc[common_index]
    
    print(f"⚠️ Les index des séries ont été alignés. {len(common_index)} points communs conservés.")
    
    # Définir les couleurs par défaut si non spécifiées
    if colors is None:
        colors = {
            portfolio1_name: '#1f77b4',  # Bleu
            portfolio2_name: '#ff7f0e',  # Orange
            portfolio3_name: '#2ca02c',   # Vert
            portfolio4_name: '#d62728'    # Rouge
        }
    
    # S'assurer que tous les index sont convertis au même format datetime
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
    
    # Calculate cumulative returns
    portfolio1_cumulative_returns = (1 + portfolio1_series).cumprod()
    portfolio2_cumulative_returns = (1 + portfolio2_series).cumprod()
    
    # DataFrame pour la comparaison des rendements cumulatifs
    cumulative_returns_comparison = {
        portfolio1_name: portfolio1_cumulative_returns,
        portfolio2_name: portfolio2_cumulative_returns
    }
    
    # Ajouter le troisième portefeuille s'il est fourni
    if portfolio3_series is not None:
        portfolio3_cumulative_returns = (1 + portfolio3_series).cumprod()
        cumulative_returns_comparison[portfolio3_name] = portfolio3_cumulative_returns
    
    # Ajouter le quatrième portefeuille s'il est fourni
    if portfolio4_series is not None:
        portfolio4_cumulative_returns = (1 + portfolio4_series).cumprod()
        cumulative_returns_comparison[portfolio4_name] = portfolio4_cumulative_returns
    
    cumulative_returns_df = pd.DataFrame(cumulative_returns_comparison)

    # Calculer les drawdowns pour chaque portefeuille
    # Pour le premier portefeuille
    portfolio1_drawdown = pd.Series(0.0, index=portfolio1_cumulative_returns.index, dtype=float)
    portfolio1_cummax = portfolio1_cumulative_returns.cummax()
    mask1 = portfolio1_cummax != 0
    portfolio1_drawdown.loc[mask1] = ((portfolio1_cummax.loc[mask1] - portfolio1_cumulative_returns.loc[mask1]) / 
                                    portfolio1_cummax.loc[mask1]).astype(float)

    # Pour le deuxième portefeuille
    portfolio2_drawdown = pd.Series(0.0, index=portfolio2_cumulative_returns.index, dtype=float)
    portfolio2_cummax = portfolio2_cumulative_returns.cummax()
    mask2 = portfolio2_cummax != 0
    portfolio2_drawdown.loc[mask2] = ((portfolio2_cummax.loc[mask2] - portfolio2_cumulative_returns.loc[mask2]) / 
                                    portfolio2_cummax.loc[mask2]).astype(float)
    
    # Pour le troisième portefeuille (si fourni)
    portfolio3_drawdown = None
    if portfolio3_series is not None:
        portfolio3_drawdown = pd.Series(0.0, index=portfolio3_cumulative_returns.index, dtype=float)
        portfolio3_cummax = portfolio3_cumulative_returns.cummax()
        mask3 = portfolio3_cummax != 0
        portfolio3_drawdown.loc[mask3] = ((portfolio3_cummax.loc[mask3] - portfolio3_cumulative_returns.loc[mask3]) / 
                                        portfolio3_cummax.loc[mask3]).astype(float)
    
    # Pour le quatrième portefeuille (si fourni)
    portfolio4_drawdown = None
    if portfolio4_series is not None:
        portfolio4_drawdown = pd.Series(0.0, index=portfolio4_cumulative_returns.index, dtype=float)
        portfolio4_cummax = portfolio4_cumulative_returns.cummax()
        mask4 = portfolio4_cummax != 0
        portfolio4_drawdown.loc[mask4] = ((portfolio4_cummax.loc[mask4] - portfolio4_cumulative_returns.loc[mask4]) / 
                                        portfolio4_cummax.loc[mask4]).astype(float)

    # Obtenir les valeurs maximales de drawdown
    portfolio1_max_dd = portfolio1_drawdown.max() 
    portfolio2_max_dd = portfolio2_drawdown.max()
    portfolio3_max_dd = None
    if portfolio3_drawdown is not None:
        portfolio3_max_dd = portfolio3_drawdown.max()
    portfolio4_max_dd = None
    if portfolio4_drawdown is not None:
        portfolio4_max_dd = portfolio4_drawdown.max()

    # Set up a modern style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Create a function for custom styling
    def apply_custom_style(ax):
        """Apply custom styling to a matplotlib axis"""
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.7, color='#cccccc')
        ax.xaxis.label.set_fontsize(12)
        ax.yaxis.label.set_fontsize(12)
        ax.tick_params(axis='both', which='major', labelsize=10, colors='#333333')
        return ax

    # Create the dashboard with two main visualizations
    fig = plt.figure(figsize=figsize, facecolor='white', dpi=50)
    
    # Définir une GridSpec avec 2 lignes, une plus grande pour les rendements cumulés (80%)
    # et une plus petite pour les drawdowns (20%)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[0.80, 0.20])
    
    # 1. Cumulative Returns (en haut, plus grand)
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(cumulative_returns_df.index, cumulative_returns_df[portfolio1_name], 
            color=colors[portfolio1_name], linewidth=2, label=portfolio1_name)
    ax1.plot(cumulative_returns_df.index, cumulative_returns_df[portfolio2_name], 
            color=colors[portfolio2_name], linewidth=2, label=portfolio2_name)
    
    # Ajouter le troisième portefeuille s'il existe
    if portfolio3_name in cumulative_returns_df.columns:
        ax1.plot(cumulative_returns_df.index, cumulative_returns_df[portfolio3_name], 
                color=colors[portfolio3_name], linewidth=2, label=portfolio3_name)
    
    # Ajouter le quatrième portefeuille s'il existe
    if portfolio4_name in cumulative_returns_df.columns:
        ax1.plot(cumulative_returns_df.index, cumulative_returns_df[portfolio4_name], 
                color=colors[portfolio4_name], linewidth=2, label=portfolio4_name)

    # Amélioration du placement des valeurs finales à droite avec shift vertical et horizontal
    # Créer un dictionnaire des valeurs finales
    final_values = {
        portfolio1_name: cumulative_returns_df[portfolio1_name].iloc[-1],
        portfolio2_name: cumulative_returns_df[portfolio2_name].iloc[-1],
    }
    
    if portfolio3_name in cumulative_returns_df.columns:
        final_values[portfolio3_name] = cumulative_returns_df[portfolio3_name].iloc[-1]
    
    if portfolio4_name in cumulative_returns_df.columns:
        final_values[portfolio4_name] = cumulative_returns_df[portfolio4_name].iloc[-1]
    
    # Trier les portefeuilles par valeur finale
    sorted_portfolios = sorted(final_values.items(), key=lambda x: x[1])
    
    # Placer les annotations avec les shifts contrôlés
    vertical_spacing = cumulative_return_vertical_spacing  # Contrôle l'espacement vertical
    
    for i, (name, value) in enumerate(sorted_portfolios):
        # Le shift horizontal est désormais constant pour toutes les annotations
        horizontal_shift = cumulative_return_horizontal_shift
        
        # Calculer le shift vertical pour éviter les chevauchements
        if i == 0:  # Premier portefeuille (valeur la plus basse)
            vertical_shift = 0
        else:
            # Distance entre la valeur actuelle et celle du portefeuille précédent
            prev_value = sorted_portfolios[i-1][1]
            diff = value - prev_value
            
            # Si la différence est petite, augmenter le shift vertical pour éviter les chevauchements
            if diff < 0.2:  # Seuil ajustable
                vertical_shift = -vertical_spacing * (len(sorted_portfolios) - i)
            else:
                vertical_shift = 0
        
        ax1.annotate(f'{value:.2f}x', 
                    xy=(cumulative_returns_df.index[-1], value),
                    xytext=(horizontal_shift, vertical_shift),  # Shift horizontal et vertical
                    textcoords='offset points',
                    ha='left', va='center', 
                    fontweight='bold', 
                    color=colors[name])

    # Ajouter le texte "Cumulative Return Growth of 1 USD" en bas à droite en gris discret
    ax1.annotate('Cumulative Return Growth of 1 USD', 
                xy=(0.95, 0.02),  # Position en bas à droite (coordonnées relatives)
                xycoords='axes fraction',  # Coordonnées relatives à l'axe
                ha='right', va='bottom',  # Alignement
                fontsize=10, color='#333333',  # Gris discret
                style='italic')  # Style italique pour plus de discrétion
                
    ax1.set_ylabel('Cumulative Return')
    
    # Hide x-axis labels for the top plot
    ax1.tick_params(axis='x', labelbottom=False)
    
    # Créer la légende pour les courbes
    ax1.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=10)
    apply_custom_style(ax1)

    # 2. Drawdowns (en bas, plus petit)
    ax2 = fig.add_subplot(gs[1])

    # Plot drawdowns with proper handling of values
    ax2.fill_between(portfolio1_drawdown.index, 0, -portfolio1_drawdown.values, 
                    alpha=0.3, color=colors[portfolio1_name], 
                    label=portfolio1_name, step=None)

    ax2.fill_between(portfolio2_drawdown.index, 0, -portfolio2_drawdown.values, 
                    alpha=0.3, color=colors[portfolio2_name], 
                    label=portfolio2_name, step=None)
    
    # Ajouter le drawdown du troisième portefeuille s'il existe
    if portfolio3_drawdown is not None:
        ax2.fill_between(portfolio3_drawdown.index, 0, -portfolio3_drawdown.values, 
                        alpha=0.3, color=colors[portfolio3_name], 
                        label=portfolio3_name, step=None)
    
    # Ajouter le drawdown du quatrième portefeuille s'il existe
    if portfolio4_drawdown is not None:
        ax2.fill_between(portfolio4_drawdown.index, 0, -portfolio4_drawdown.values, 
                        alpha=0.3, color=colors[portfolio4_name], 
                        label=portfolio4_name, step=None)

    ax2.plot(portfolio1_drawdown.index, -portfolio1_drawdown.values, alpha=0.7, color=colors[portfolio1_name], linewidth=1)
    ax2.plot(portfolio2_drawdown.index, -portfolio2_drawdown.values, alpha=0.7, color=colors[portfolio2_name], linewidth=1)
    
    # Ajouter les lignes de drawdown pour les portefeuilles supplémentaires
    if portfolio3_drawdown is not None:
        ax2.plot(portfolio3_drawdown.index, -portfolio3_drawdown.values, alpha=0.7, color=colors[portfolio3_name], linewidth=1)
    
    if portfolio4_drawdown is not None:
        ax2.plot(portfolio4_drawdown.index, -portfolio4_drawdown.values, alpha=0.7, color=colors[portfolio4_name], linewidth=1)

    # Find maximum drawdowns dates
    portfolio1_valley_date = portfolio1_drawdown.idxmax()
    portfolio2_valley_date = portfolio2_drawdown.idxmax()
    portfolio3_valley_date = None
    if portfolio3_drawdown is not None:
        portfolio3_valley_date = portfolio3_drawdown.idxmax()
    portfolio4_valley_date = None
    if portfolio4_drawdown is not None:
        portfolio4_valley_date = portfolio4_drawdown.idxmax()

    # Amélioration du placement des annotations de Max DD avec flèches externes et arcs
    # Collecter toutes les informations sur les drawdowns
    drawdowns = [
        (portfolio1_name, portfolio1_max_dd, portfolio1_valley_date, colors[portfolio1_name]),
        (portfolio2_name, portfolio2_max_dd, portfolio2_valley_date, colors[portfolio2_name])
    ]
    
    if portfolio3_drawdown is not None:
        drawdowns.append((portfolio3_name, portfolio3_max_dd, portfolio3_valley_date, colors[portfolio3_name]))
    
    if portfolio4_drawdown is not None:
        drawdowns.append((portfolio4_name, portfolio4_max_dd, portfolio4_valley_date, colors[portfolio4_name]))
    
    # Trier par date pour une meilleure organisation des annotations
    drawdowns.sort(key=lambda x: x[2])
    
    # Calculer la valeur de drawdown maximale pour dimensionner la zone des annotations
    max_dd_value = max([dd[1] for dd in drawdowns])
    
    # Positionner les annotations à l'extérieur avec des flèches courbes
    for i, (name, max_dd, valley_date, color) in enumerate(drawdowns):
        # Décaler les annotations horizontalement pour éviter les chevauchements
        # Alternance droite/gauche pour les annotations
        horizontal_shift = drawdown_horizontal_shift * (-1 if i % 2 == 0 else 1)
        
        # Direction de l'arc basée sur la position (gauche ou droite)
        arc_rad = drawdown_arc_radius * (-1 if i % 2 == 0 else 1)
        
        ax2.annotate(f'Max DD: -{max_dd:.2%}',
                    xy=(valley_date, -max_dd),  # Point à annoter (drawdown max)
                    xytext=(horizontal_shift, 0),  # Shift horizontal
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
    
    # Ajuster les limites pour avoir une marge en bas et en haut
    # pour accommoder les annotations extérieures
    all_max_dd = max([dd[1] for dd in drawdowns])
    ax2.set_ylim(-all_max_dd * 1.3, all_max_dd * 0.4)  # Plus d'espace en haut pour les annotations
    
    # Légende pour le graphique de drawdowns
    ax2.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=8)
    apply_custom_style(ax2)

    # Ajuster la mise en page avec peu d'espace entre les graphiques
    # plt.tight_layout()
    
    # Ajouter le titre principal tout en haut et en gras
    fig.suptitle(title, fontsize=18, fontweight='bold', x=0.5, y=0.98, ha='center')
    
    # Ajuster l'espace pour coller les graphiques et accommoder le titre
    plt.subplots_adjust(top=0.92, hspace=0.0)
    
    if save_path:
        plt.savefig(save_path, dpi=50, bbox_inches='tight')
        print(f"Dashboard saved to {save_path}")
    
    return fig, [ax1, ax2]

fig, axes = create_portfolio_comparison_dashboard(
    vw_series,           # Premier portefeuille (Value-Weighted)
    mvp_series,          # Deuxième portefeuille (Minimum-Variance)
    mvpc_series,         # Troisième portefeuille (MVP avec contrainte carbone 50%)
    te_series,           # Quatrième portefeuille (TE portfolio)
    portfolio1_name=r'$P^{(v w)}$',
    portfolio2_name=r'$P_{o o s}^{(m v)}$',
    portfolio3_name=r'$P_{o o s}^{(m v)}(0.5)$',
    portfolio4_name=r'$P_{o o s}^{(v w)}(0.5)$',
    save_path="Four_Portfolio_Comparison.png" if save_images else None,
    colors={
        r'$P^{(v w)}$': '#1f77b4',            # Bleu pour VW
        r'$P_{o o s}^{(m v)}$': '#ff7f0e',    # Orange pour MVP
        r'$P_{o o s}^{(m v)}(0.5)$': '#2ca02c', # Vert pour MVP05
        r'$P_{o o s}^{(v w)}(0.5)$': '#d62728'  # Rouge pour TE
    },
    figsize=(22, 8),
    title=r'Comparison between $P^{(v w)}$, $P_{o o s}^{(m v)}$, $P_{o o s}^{(m v)}(0.5)$ and $P_{o o s}^{(v w)}(0.5)$',
    # Paramètres de contrôle des shifts
    cumulative_return_horizontal_shift=10,
    cumulative_return_vertical_spacing=-5,
    drawdown_horizontal_shift=40,
    drawdown_arc_radius=0.3
)


# In[226]:


def plot_te_vw_carbon_comparison(te_cf_values, vw_cf_values, te_returns, vw_returns,
                               figsize=(14, 10), save_path=None, colors=None):
    """
    Crée une visualisation professionnelle comparant le portefeuille TE avec contrainte carbone (P_oos^(vw)(0.5))
    par rapport au benchmark VW (P^(vw)) montrant :
    1. L'empreinte carbone de chaque portefeuille et la contrainte de 50% (panneau supérieur)
    2. L'erreur de suivi (tracking error) du TE par rapport au VW (panneau inférieur)
    
    Parameters
    ----------
    te_cf_values : dict
        Valeurs d'empreinte carbone du portefeuille TE par année
    vw_cf_values : dict
        Valeurs d'empreinte carbone du portefeuille VW par année
    te_returns : pd.Series
        Rendements mensuels du portefeuille TE
    vw_returns : pd.Series
        Rendements mensuels du portefeuille VW (benchmark)
    figsize : tuple, optional
        Dimensions de la figure (largeur, hauteur)
    save_path : str, optional
        Chemin pour sauvegarder la figure
    colors : dict, optional
        Couleurs personnalisées pour les différents éléments
        
    Returns
    -------
    fig, axes : tuple
        Figure et axes pour d'éventuelles personnalisations supplémentaires
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import LinearSegmentedColormap
    from scipy import stats
    
    # Palette de couleurs personnalisée pour TE
    te_c = [
        (0.90, 0.90, 0.90),  # Gris clair (plus visible)
        (0.75, 0.76, 0.80),  # Gris bleuté (plus foncé)
        (0.60, 0.65, 0.75),  # Gris bleu moyen
        (0.45, 0.50, 0.60)   # Gris bleu foncé
    ]
    pastel_viridis = LinearSegmentedColormap.from_list('pastel_viridis', te_c)
    
    # Définir les couleurs par défaut si non fournies
    if colors is None:
        colors = {
            'te': '#d62728',      # Rouge pour le portefeuille TE
            'vw': '#ff7f0e',      # Orange pour le portefeuille VW
            'target': '#1f77b4',  # Bleu pour la contrainte cible
            'tracking_error': '#2ca02c'  # Vert pour l'erreur de suivi
        }
    
    # Style professionnel
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create figure with GridSpec for layout control with hspace=0 to make them connected
    fig = plt.figure(figsize=figsize, dpi=50, facecolor='white')
    gs = GridSpec(2, 1, height_ratios=[5, 1], hspace=0.05)
    
    # Titre principal unique pour toute la figure
    fig.suptitle(r"Carbon Constrained Tracking Error Portfolio $P_{oos}^{(vw)}(0.5)$ vs Benchmark $P^{(vw)}$", 
                fontsize=16, fontweight='bold', y=0.98)
    
    # ====================== PANNEAU SUPÉRIEUR: TRAJECTOIRE CARBONE ====================== #
    ax1 = fig.add_subplot(gs[0])
    
    # Années communes et tri chronologique
    common_years = sorted(set(te_cf_values) & set(vw_cf_values))
    numeric_years = [int(y) for y in common_years]
    x_positions = np.arange(len(numeric_years))
    
    # Extraire les valeurs
    te_values = [te_cf_values[y] for y in common_years]
    vw_values = [vw_cf_values[y] for y in common_years]
    carbon_constraints = [0.5 * vw_cf_values[y] for y in common_years]
    
    # Calculer les statistiques
    avg_reduction = np.mean([1 - te_values[i]/vw_values[i] for i in range(len(te_values))])
    min_reduction = min([1 - te_values[i]/vw_values[i] for i in range(len(te_values))])
    max_reduction = max([1 - te_values[i]/vw_values[i] for i in range(len(te_values))])
    
    # Créer le graphique à barres pour la comparaison d'empreinte carbone
    bar_width = 0.35
    bars1 = ax1.bar(x_positions - bar_width/2, vw_values, bar_width, color=colors['vw'], 
                  alpha=0.85, label=r'$P^{(vw)}$ Carbon Footprint', edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x_positions + bar_width/2, te_values, bar_width, color=colors['te'], 
                  alpha=0.85, label=r'$P_{oos}^{(vw)}(0.5)$ Carbon Footprint', edgecolor='black', linewidth=0.5)
    
    # Ajouter la ligne de contrainte cible
    ax1.plot(x_positions, carbon_constraints, '--', linewidth=1.8, color=colors['target'], 
           label='Carbon Constraint (50% of VW)')
    
    # Colorer la zone sous la contrainte pour mettre en évidence la zone cible
    ax1.fill_between(x_positions, [0]*len(numeric_years), carbon_constraints, 
                   color=colors['target'], alpha=0.1)
    
    # Ajouter les pourcentages de réduction au-dessus des barres TE
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        reduction_pct = (1 - bar2.get_height()/bar1.get_height()) * 100
        ax1.annotate(f'{reduction_pct:.1f}%', 
                   xy=(bar2.get_x() + bar2.get_width()/2, bar2.get_height()),
                   xytext=(0, 5), textcoords='offset points',
                   ha='center', va='bottom', fontsize=9, fontweight='bold',
                   color=colors['te'])
        
        # Ajouter les valeurs réelles sur les barres
        ax1.annotate(f'{bar1.get_height():.0f}',
                   xy=(bar1.get_x() + bar1.get_width()/2, bar1.get_height()),
                   xytext=(0, -15), textcoords='offset points', 
                   ha='center', va='top', fontsize=8,
                   color='white', fontweight='bold')
        
        ax1.annotate(f'{bar2.get_height():.0f}',
                   xy=(bar2.get_x() + bar2.get_width()/2, bar2.get_height()),
                   xytext=(0, -15), textcoords='offset points', 
                   ha='center', va='top', fontsize=8,
                   color='white', fontweight='bold')
    
    # Formater le panneau supérieur
    ax1.set_ylabel(r'Carbon Footprint (tCO$ _2$e/M\$)', fontsize=12)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([])  # Masquer les étiquettes x pour le panneau supérieur
    ax1.grid(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=10)
    ax1.set_axisbelow(True)
    
    # Ajouter une annotation sur la zone de contrainte
    ax1.annotate('Target Constraint Zone', 
                xy=(x_positions[len(x_positions)//2], carbon_constraints[len(carbon_constraints)//2]/2),
                xytext=(0, -20), textcoords='offset points',
                ha='center', va='top', fontsize=8, style='italic',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec='#cccccc'),
                arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2", color='#555555'))
    
    # ====================== PANNEAU INFÉRIEUR: ERREUR DE SUIVI ====================== #
    ax2 = fig.add_subplot(gs[1])
    
    # Calculer l'erreur de suivi (différences de rendement mensuelles)
    te_returns.index = pd.to_datetime(te_returns.index)
    vw_returns.index = pd.to_datetime(vw_returns.index)
    
    # Aligner les séries de rendement par index
    common_dates = te_returns.index.intersection(vw_returns.index)
    tracking_error = (te_returns.loc[common_dates] - vw_returns.loc[common_dates])
    
    # Calculer l'erreur de suivi annualisée
    annual_te = tracking_error.std() * np.sqrt(12) * 100  # Annualisée et en pourcentage
    
    # Grouper l'erreur de suivi par année pour un axe x cohérent avec le panneau carbone
    yearly_te = {}
    for year in numeric_years:
        year_mask = tracking_error.index.year == year
        if year_mask.any():
            yearly_te[str(year)] = tracking_error.loc[year_mask].values
    
    # Calculer l'erreur de suivi absolue moyenne par année
    avg_abs_te_by_year = [np.mean(np.abs(yearly_te[str(year)])) * 100 if str(year) in yearly_te else np.nan 
                         for year in numeric_years]
    
    # Utiliser la colormap personnalisée, normaliser les données entre 0-1 pour les couleurs
    abs_te_max = np.nanmax(avg_abs_te_by_year)
    if abs_te_max > 0:
        # Normaliser les valeurs entre 0 et 1
        color_idx = np.array(avg_abs_te_by_year) / abs_te_max
        # Remplacer les NaN par 0 pour éviter les erreurs
        color_idx = np.nan_to_num(color_idx)
        # Utiliser la colormap personnalisée pour chaque année
        bar_colors = [pastel_viridis(idx) for idx in color_idx]
    else:
        # Utiliser une couleur par défaut si toutes les valeurs sont NaN ou 0
        bar_colors = [pastel_viridis(0.5)] * len(avg_abs_te_by_year)
    
    # Tracer l'erreur de suivi absolue moyenne par année sous forme de barres
    bars3 = ax2.bar(x_positions, avg_abs_te_by_year, color=bar_colors, alpha=0.7,
                   width=0.6, edgecolor='black', linewidth=0.5)
    
    # Ajouter une ligne pour l'erreur de suivi annualisée sur toute la période
    ax2.axhline(annual_te, linestyle='--', color='#d62728', linewidth=1.5, 
               label=f'Annualized TE: {annual_te:.2f}%')
    
    # Ajouter les étiquettes de valeur sur les barres
    for i, bar in enumerate(bars3):
        if not np.isnan(bar.get_height()):
            ax2.annotate(f'{bar.get_height():.2f}%', 
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=9)
    
    ax2.text(0.02, 0.80, r'$\mathbf{P_{oos}^{(vw)}}$ - $\mathbf{P_{oos}^{(vw)}(0.5)}$ Tracking Error' , transform=ax2.transAxes, fontsize=8,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
            facecolor='white', alpha=0.9))
    
    # Formater le panneau d'erreur de suivi
    ax2.set_xlabel('Year', fontsize=12, labelpad=10)
    ax2.set_ylabel('Absolute TE (%)', fontsize=10, labelpad=10)
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(numeric_years, rotation=0)
    ax2.grid(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax2.set_axisbelow(True)
    
    # Ajuster la mise en page
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.12, top=0.92, hspace=0.25)
    
    # Sauvegarder si un chemin est fourni
    if save_path:
        plt.savefig(save_path, dpi=50, bbox_inches='tight')
        print(f"Figure sauvegardée dans {save_path}")
    
    return fig, (ax1, ax2)


# In[227]:


fig, axes = plot_te_vw_carbon_comparison(
    te_cf_values=te_cf_values,             # Empreinte carbone du portefeuille TE
    vw_cf_values=carbon_footprint_vw,             # Empreinte carbone du portefeuille VW
    te_returns=te_series,                  # Rendements mensuels du portefeuille TE
    vw_returns=vw_series,                  # Rendements mensuels du portefeuille VW
    figsize=(22, 8),
    save_path="te_vs_vw_carbon_analysis.png" if save_images else None,
    colors={
        'te': '#d62728',       # Rouge pour le portefeuille TE
        'vw': '#1f77b4',       # Orange pour le benchmark VW
        'target': '#d62728',   # Bleu pour la contrainte cible
        'tracking_error': '#657994'  # Vert pour l'erreur de suivi
    }
)


# In[228]:


# Tableau récapitulatif des résultats
te_metrics = calculate_portfolio_metrics(te_series, rf[rf_column])
# Créer un DataFrame pour comparer les trois portefeuilles
metrics_comparison = pd.DataFrame({
    'MVP': pd.Series(mvp_metrics),  # Métriques du portefeuille MVP
    'MVP05': pd.Series(mvpc_metrics),
    'VW': pd.Series(vw_metrics),  # Métriques du portefeuille VW
    'TE': pd.Series(te_metrics)  # Métriques du portefeuille TE
})

summary_df = pd.DataFrame({
    'Portefeuille': ['Value-Weighted', 'Minimum Variance', 'TE (50% Carbon Reduction)'],
    'Rendement Annualisé': [
        metrics_comparison.loc['Annualized Average Return', 'VW'] * 100,
        metrics_comparison.loc['Annualized Average Return', 'MVP'] * 100,
        metrics_comparison.loc['Annualized Average Return', 'TE'] * 100
    ],
    'Volatilité Annualisée': [
        metrics_comparison.loc['Annualized Volatility', 'VW'] * 100,
        metrics_comparison.loc['Annualized Volatility', 'MVP'] * 100,
        metrics_comparison.loc['Annualized Volatility', 'TE'] * 100
    ],
    'Maximum Drawdown': [
        metrics_comparison.loc['Maximum Monthly Return', 'VW'] * 100,
        metrics_comparison.loc['Maximum Monthly Return', 'MVP'] * 100,
        metrics_comparison.loc['Maximum Monthly Return', 'TE'] * 100
    ]
})

# Formater le tableau pour l'affichage
formatted_df = summary_df.copy()
formatted_df['Rendement Annualisé'] = formatted_df['Rendement Annualisé'].map('{:.2f}%'.format)
formatted_df['Volatilité Annualisée'] = formatted_df['Volatilité Annualisée'].map('{:.2f}%'.format)
formatted_df['Maximum Drawdown'] = formatted_df['Maximum Drawdown'].map('{:.2f}%'.format)

print("Récapitulatif des performances des portefeuilles:")
print(formatted_df.set_index('Portefeuille'))

# Calcul de la corrélation entre les rendements des différents portefeuilles
correlation_matrix = pd.DataFrame({
    'VW': vw_portfolio_returns,
    'MVP': portfolio_returns_all['Portfolio_Return'],
    'TE': te_portfolio_returns['Portfolio_Return']
}).corr()

print("\nMatrice de corrélation des rendements:")
print(correlation_matrix)


# In[229]:


def verify_te_optimization(te_weights, vw_weights, sigma, emission_intensity, carbon_constraint):
    """Vérifie que le portefeuille TE respecte les contraintes et calcule la tracking error"""
    # 1. Vérifier que sum(weights) = 1
    sum_weights = np.sum(te_weights)
    print(f"Somme des poids: {sum_weights:.6f} (objectif: 1.0)")
    
    # 2. Vérifier la contrainte carbone
    carbon_footprint = np.sum(te_weights * emission_intensity)
    print(f"Empreinte carbone: {carbon_footprint:.2f} (limite: {carbon_constraint:.2f})")
    print(f"Réduction carbone: {(1 - carbon_footprint/np.sum(vw_weights * emission_intensity)):.2%}")
    
    # 3. Calculer la tracking error
    tracking_diff = te_weights - vw_weights
    te_squared = tracking_diff @ sigma @ tracking_diff
    te_value = np.sqrt(te_squared)
    print(f"Tracking Error: {te_value:.4f}")
    
    # 4. Calculer les volatilités
    te_vol = np.sqrt(te_weights @ sigma @ te_weights)
    vw_vol = np.sqrt(vw_weights @ sigma @ vw_weights)
    print(f"Volatilité TE: {te_vol:.4f}")
    print(f"Volatilité VW: {vw_vol:.4f}")
    print(f"Ratio: {te_vol/vw_vol:.2f}")
    
    # 5. Vérifier les poids négatifs
    neg_weights = np.sum(te_weights < 0)
    print(f"Poids négatifs: {neg_weights} sur {len(te_weights)}")
    

def test_verify_te_optimization(year='2020'):
    """
    Fonction pour tester verify_te_optimization avec des données réelles
    pour une année spécifique
    """
    # Vérifier que l'année existe dans nos données
    if year not in te_weights or year not in weighted_mcap_by_year:
        print(f"L'année {year} n'est pas disponible dans les données")
        return
    
    # 1. Récupérer les poids TE pour l'année spécifiée
    te_weights_year = te_weights[year]
    
    # 2. Récupérer les poids VW pour la même année (dernière période)
    vw_weights_year = weighted_mcap_by_year[year].iloc[-1]
    
    # 3. Récupérer la matrice de covariance de la période d'estimation
    est_returns = windows_[year]['alpha_period']
    sigma = est_returns.cov().values
    
    # 4. Récupérer les données nécessaires pour l'intensité d'émission
    year_data = annual_filtered_data_[year]
    scope1 = year_data['scope1']
    scope2 = year_data['scope2']
    ycap = year_data['ycap']
    
    # Récupérer les valeurs (dernière période si plusieurs)
    if len(scope1.index) > 1:
        scope1_values = scope1.iloc[-1]
        scope2_values = scope2.iloc[-1]
        ycap_values = ycap.iloc[-1]
    else:
        scope1_values = scope1.iloc[0]
        scope2_values = scope2.iloc[0]
        ycap_values = ycap.iloc[0]
    
    # Calculer les émissions totales
    total_emissions = scope1_values + scope2_values
    
    # 5. Calculer l'intensité des émissions
    emission_intensity = total_emissions / ycap_values
    emission_intensity = emission_intensity.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 6. Trouver les entreprises communes à tous les datasets
    common_companies = list(set(te_weights_year.index) & 
                           set(vw_weights_year.index) & 
                           set(emission_intensity.index) &
                           set(est_returns.columns))
    
    # Filtrer les données pour ne garder que les entreprises communes
    te_weights_filtered = te_weights_year[common_companies].values
    vw_weights_filtered = vw_weights_year[common_companies].values
    emission_intensity_filtered = emission_intensity[common_companies].values
    sigma_filtered = est_returns[common_companies].cov().values
    
    # 7. Calculer la contrainte carbone (50% de l'empreinte carbone du VW)
    vw_carbon_footprint = calculate_vw_carbon_footprint(annual_filtered_data_, year)
    carbon_constraint = 0.5 * vw_carbon_footprint
    
    print(f"=== Test de verify_te_optimization pour l'année {year} ===")
    print(f"Nombre d'entreprises communes: {len(common_companies)}")
    
    # 8. Appeler la fonction de vérification
    verify_te_optimization(
        te_weights_filtered, 
        vw_weights_filtered, 
        sigma_filtered, 
        emission_intensity_filtered, 
        carbon_constraint
    )

# # Exécuter le test pour une année spécifique
# for i in range(14, 24):
#     test_verify_te_optimization(f'20{i}')  # Remplacer par une année disponible dans tes données


# ## 2.4
# Comment on the trade-off between the financial performance of the portfolio and the reduction in its carbon footprint. Elaborate on the difference between portfolios " $P_{o o s}^{(m v)}$ " and " $P_{o o s}^{(m v)}(0.5)$ " and between portfolios " $P_{o o s}^{(v w) " ~ a n d ~ " ~} P_{o o s}^{(v w)}(0.5)$ ".


# In[230]:


def create_carbon_by_year_dict(carbon_footprint, constrained_cf_values, te_cf_values, nz_cf_values=None, vw_cf_values=None):
    """
    Crée un dictionnaire des empreintes carbone par année pour tous les portefeuilles.
    
    Parameters
    ----------
    carbon_footprint : dict
        Empreinte carbone du portefeuille MVP par année
    constrained_cf_values : dict
        Empreinte carbone du portefeuille MVP(0.5) par année
    te_cf_values : dict
        Empreinte carbone du portefeuille TE(0.5) par année
    nz_cf_values : dict, optional
        Empreinte carbone du portefeuille Net Zero par année
    vw_cf_values : dict, optional
        Empreinte carbone du portefeuille Value-Weighted par année
    
    Returns
    -------
    dict
        Dictionnaire structuré des empreintes carbone par année et par portefeuille
    """
    # Collecter toutes les années disponibles
    all_years = set()
    for cf_dict in [carbon_footprint, constrained_cf_values, te_cf_values, nz_cf_values, vw_cf_values]:
        if cf_dict:
            all_years.update(cf_dict.keys())
    
    # Trier les années
    sorted_years = sorted(all_years)
    
    # Créer le dictionnaire résultat
    carbon_by_year = {}
    
    for year in sorted_years:
        carbon_by_year[year] = {}
        
        # Ajouter les valeurs disponibles pour chaque portefeuille
        if vw_cf_values and year in vw_cf_values:
            carbon_by_year[year]['Value-Weighted'] = vw_cf_values[year]
        
        if carbon_footprint and year in carbon_footprint:
            carbon_by_year[year]['Minimum Variance'] = carbon_footprint[year]
        
        if constrained_cf_values and year in constrained_cf_values:
            carbon_by_year[year]['MVP (50% Carbon Reduction)'] = constrained_cf_values[year]
        
        if te_cf_values and year in te_cf_values:
            carbon_by_year[year]['TE (50% Carbon Reduction)'] = te_cf_values[year]
        
        if nz_cf_values and year in nz_cf_values:
            carbon_by_year[year]['Net Zero'] = nz_cf_values[year]
    
    return carbon_by_year


# In[231]:


# Génération automatique du dictionnaire carbon_by_year
carbon_by_year = create_carbon_by_year_dict(
    carbon_footprint=carbon_footprint_mvp,          # MVP standard
    constrained_cf_values=constrained_cf_values, # MVP avec contrainte 50%
    te_cf_values=te_cf_values,                  # TE avec contrainte 50%
    vw_cf_values=carbon_footprint_vw                   # Value-Weighted
)



# ### 3 Allocation with a Net Zero Objective
# Finally, we want to construct a minimum variance portfolio, while cumulatively reducing its carbon emissions.
# 
# - 3.1 Finally, we implement a decarbonization strategy in which the carbon footprint of the portfolio is reduced by $\theta=10 \%$ per year every year from Dec. 2013 to Dec. 2023.
# We adopt the point of view of the otherwise passive investor. The optimization problem is the same as in point 2.3 except that the carbon emissions reduction constraint is now defined as
# 
# $$
# C F_Y^{(p)} \leq(1-\theta)^{Y-Y_0+1} \times C F_{Y_0}^{\left(P^{(v w)}\right)} \quad \text { for } Y=2013, \cdots, 2023
# $$
# 
# - with $Y_0=2013$.
# - We call this portfolio " $P_{o o s}^{(v w)}(N Z)$ ". Compute the characteristics of this portfolio over the sample. Again, plot the cumulative return series of both strategies and compare summary statistics.


# In[232]:


print('\n\n\n\n\n\n')
print('# ------------------------ 3.1  P(vw)(NZ) -------------------------- #')


# In[233]:


# ─────────────────────────────────────────────────────────────────────────────
# 1)  CONTRAINTE NET-ZERO : CF ≤ (1-θ)^(Δt+1) · CF_VW,2014
# ─────────────────────────────────────────────────────────────────────────────
def nz_bornes(cf_vw_dict: dict,                # ←  { 'YYYY': CF_VW }
              base_year : str = '2014',
              theta     : float = 0.10,
              last_year : str = '2024',
              verbose   : bool = True) -> dict[str, float]:
    """
    Renvoie un dict { 'YYYY': borne_CF } pour chaque année ≥ base_year.
    """
    if base_year not in cf_vw_dict:
        raise ValueError(f"CF_VW manquant pour l’année de référence {base_year}")

    cf_ref = cf_vw_dict[base_year]
    if verbose:
        print(f"CF_VW référence {base_year} = {cf_ref:.2f} tCO₂e/MUSD")

    bornes = {}
    for y in range(int(base_year), int(last_year) + 1):
        k      = y - int(base_year) + 1            # Δt+1
        factor = (1 - theta) ** k
        bornes[str(y)] = cf_ref * factor
        if verbose:
            print(f"{y}: borne = {bornes[str(y)]:.2f}  "
                  f"(réduction cumulée {(1 - factor):.1%})")
    return bornes



# ─────────────────────────────────────────────────────────────────────────────
# 2)  PORTFEUILLE NET-ZERO  (min TE vs VW  +  borne carbone)
# ─────────────────────────────────────────────────────────────────────────────
def compute_netzero_portfolio(filtered_windows      : dict,
                              annual_filtered_data  : dict,
                              vw_weights_by_year    : dict,
                              cf_vw_dict            : dict,
                              base_year : str = '2014',
                              theta     : float = 0.10,
                              verbose   : bool = True):
    """
    Retourne deux dictionnaires :
        nz_weights[year] : Series poids optimisés
        nz_cf[year]      : empreinte carbone obtenue (t/MUSD)
    """
    import numpy as np, pandas as pd
    from scipy.optimize import minimize

    # ─── 1. Borne annuelle ────────────────────────────────────────────────
    borne = nz_bornes(cf_vw_dict, base_year, theta, '2024', verbose)

    nz_weights, nz_cf = {}, {}
    if verbose:
        print("\n>>> Optimisation Net-Zero (Tracking-Error min + borne CO₂)\n")

    # ─── 2. Boucle années ────────────────────────────────────────────────
    for year in sorted(filtered_windows):          # e.g. '2014', …, '2023'
        if year not in borne:
            continue                               # rien avant 2014

        try:
            # ---------- REND. & POIDS VW ----------
            R_m   = filtered_windows[year]['alpha_period']
            vw_w  = vw_weights_by_year[year].iloc[-1]          # dernière obs.

            # ---------- DONNÉES ANNUELLES ----------
            ann   = annual_filtered_data[year]
            emis  = (ann['scope1'] + ann['scope2']).iloc[-1]   # t
            cap   = ann['ycap'].iloc[-1]                       # MUSD
            inten = (emis / cap).replace([np.inf, -np.inf], 0).fillna(0)

            # ---------- UNIVERS COMMUN ----------
            common = sorted(set(R_m.columns) & set(vw_w.index) & set(inten.index))
            if not common:
                if verbose: print(f"{year}: aucun titre commun – ignoré")
                continue

            R_m, vw_w, inten = R_m[common], vw_w[common], inten[common]
            n     = len(common)
            Σ     = R_m.cov().values                # covariance pair-wise
            # projection PSD légère pour éviter Σ semi-déf. non pos.
            # eigval, eigvec = np.linalg.eigh(Σ)
            # eigval[eigval < 1e-6] = 1e-6
            # Σ = (eigvec * eigval) @ eigvec.T

            vw   = vw_w.values
            ci   = inten.values
            cf_max = borne[year]

            # ---------- OPTIMISATION SLSQP ----------
            def obj(x):
                d = x - vw
                return 10000.0 * d @ Σ @ d           # TE² × 1000 (comme demandé)

            cons = [{'type': 'eq',  'fun': lambda x: x.sum() - 1},
                    {'type': 'ineq','fun': lambda x: cf_max - np.dot(x, ci)}]

            res = minimize(obj,
                           x0       = np.full(n, 1/n),
                           bounds   = [(0,1)] * n,
                           constraints = cons,
                           options  = {'ftol':1e-6, 'maxiter':1000, 'disp':False})

            if not res.success:
                if verbose: print(f"{year}: échec opti – {res.message}")
                continue

            w_opt = pd.Series(res.x, index=common)
            cf_opt= float(np.dot(w_opt, ci))

            nz_weights[year] = w_opt
            nz_cf[year]      = cf_opt

            if verbose:
                red_vs_vw = (1 - cf_opt/cf_vw_dict[year]) if cf_vw_dict[year] else np.nan
                print(f"✓ {year}: CF_opt={cf_opt:.2f}  borne={cf_max:.2f}  "
                      f"réduction vs VW {red_vs_vw:.1%}")

        except Exception as e:
            if verbose: print(f"ERREUR {year}: {e}")

    return nz_weights, nz_cf


# In[234]:


# 0)  CF_VW issu de tes DataFrame déjà calculés
# carbon_footprint_vw = dict(get_carbon_metrics(vw_metrics, 'CF'))

# # 1)  Optimisation Net-Zero
# nz_weights, nz_cf = compute_netzero_portfolio(
#     filtered_windows       = windows_,
#     annual_filtered_data   = annual_filtered_data_,
#     vw_weights_by_year     = weighted_mcap_by_year,
#     cf_vw_dict             = carbon_footprint_vw,   # ← nouvelle entrée
#     base_year              = '2014',                # car ton jeu commence en 2014
#     theta                  = 0.10,
#     verbose                = True
# )

# # 2)  Back-test identique à tes autres portefeuilles…


# In[235]:


# Fonction modifiée qui prend en compte use_pickle
def get_optimal_weights_nz(pickle_weights_window, 
                            windows_, 
                            annual_filtered_data_,
                            weighted_mcap_by_year,
                            carbon_footprint_vw,
                            base_year='2014',
                            thetha=0.10,
                            verbose=True,
                            force_recalculate=False,
                            use_pickle=True):
    """
    Récupère les poids optimaux, soit depuis un fichier sauvegardé, soit en les recalculant.
  
    Parameters:
    -----------
    pickle_weights_window : str
        Nom du dossier pour sauvegarder les poids
    windows_ : dict
        Dictionnaire des fenêtres pour l'optimisation
    annual_filtered_data_ : dict
        Données annuelles filtrées
    weighted_mcap_by_year : dict
        Poids de capitalisation boursière par année
    carbon_footprint_vw : float
        Empreinte carbone du portefeuille value-weighted
    base_year : str, optional
        Année de référence pour l'optimisation
    thetha : float, optional
        Paramètre thetha pour l'optimisation
    verbose : bool, optional
        Si True, affiche des informations pendant le calcul
    force_recalculate : bool, optional
        Si True, force le recalcul même si le fichier existe déjà
    use_pickle : bool, optional
        Si True, utilise le système de sauvegarde/chargement pickle, sinon calcule directement
      
    Returns:
    --------
    tuple
        Tuple contenant (nz_weights, nz_cf)
    """
    
    # Si use_pickle est False, on calcule directement sans utiliser les fichiers pickle
    if not use_pickle:
        print("🔄 Calcul direct des poids optimaux (sans pickle)...")
        nz_weights, nz_cf = compute_netzero_portfolio(
                                windows_,
                                annual_filtered_data_,
                                weighted_mcap_by_year,
                                carbon_footprint_vw,
                                base_year,
                                thetha,
                                verbose
                            )
        return nz_weights, nz_cf
    
    # Sinon, on utilise la logique existante avec pickle
    pickle_path = 'pickle_weights/' + f'{pickle_weights_window}'  + '/optimal_weights_NZ.pkl'
  
    # Si le fichier existe et qu'on ne force pas le recalcul
    if os.path.exists(pickle_path) and not force_recalculate:
        print("📂 Chargement des poids optimaux depuis le fichier...")
        with open(pickle_path, 'rb') as f:
            optimal_weights = pickle.load(f)
            return optimal_weights['nz_weights'], optimal_weights['nz_cf']
    else:
        print("🔄 Calcul des poids optimaux en cours...")
        # Calcul des poids optimaux
        nz_weights, nz_cf = compute_netzero_portfolio(
                                windows_,
                                annual_filtered_data_,
                                weighted_mcap_by_year,
                                carbon_footprint_vw,
                                base_year,
                                thetha,
                                verbose
                            )
      
        # Création d'une structure pour stocker les deux dictionnaires
        optimal_weights = {
            'nz_weights': nz_weights,
            'nz_cf': nz_cf
        }
        
        # Sauvegarde pour une utilisation future
        os.makedirs(os.path.dirname(pickle_path), exist_ok=True)  # Assurer que le dossier existe
        with open(pickle_path, 'wb') as f:
            pickle.dump(optimal_weights, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print("✅ Poids optimaux calculés et sauvegardés")
      
        return nz_weights, nz_cf

# Utilisation avec le paramètre use_pickle
nz_weights, nz_cf = get_optimal_weights_nz(pickle_weights_window, 
                                          windows_,
                                          annual_filtered_data_,
                                          weighted_mcap_by_year,
                                          carbon_footprint_vw,
                                          base_year='2014',
                                          thetha=0.10,
                                          verbose=True,
                                          force_recalculate=False,
                                          use_pickle=use_pickle)  # Utilisation de la variable use_pickle


# In[236]:


# Mise à jour des poids du portefeuille sur la période de test
nz_updated_weights = update_portfolio_weights(nz_weights, windows_)

# Calcul des rendements du portefeuille
nz_portfolio_returns = compute_portfolio_returns(nz_updated_weights, windows_, nz_weights)

# Afficher les premières lignes des rendements
print("\nAperçu des rendements du portefeuille Net Zero:")
print(nz_portfolio_returns.head())


# $$\textcolor{cyan}{\text{Metrics}}$$


# In[237]:


# # Calculer les métriques de performance
nz_series = nz_portfolio_returns['Portfolio_Return']
nz_metrics = calculate_portfolio_metrics(nz_series, rf[rf_column])

# # Créer un DataFrame pour comparer tous les portefeuilles
metrics_comparison_all = pd.DataFrame({
    'MVP': pd.Series(mvp_metrics),              # Métriques du portefeuille MVP
    'MVP05': pd.Series(mvpc_metrics),           # MVP avec contrainte carbone 50%
    'VW': pd.Series(vw_metrics),                # Value-Weighted
    'TE': pd.Series(te_metrics),                # TE avec contrainte carbone 50%
    'NZ': pd.Series(nz_metrics)                 # Net Zero (réduction progressive)
})

# Afficher le tableau comparatif
print(metrics_comparison_all)


# In[238]:


def compare_portfolio_metrics_selective_te(
    portfolios_dict, 
    benchmark_name=None, 
    rf_series=None,
    carbon_metrics_dict=None, 
    latex_path=None, 
    latex_column_names=None,
    show_te_only_for=None
):
    """
    Compare les métriques de performance de plusieurs portefeuilles, avec l'option
    d'afficher le tracking error seulement pour certains portefeuilles spécifiés.
    
    Parameters:
    -----------
    portfolios_dict : dict
        Dictionnaire avec les séries de rendements de chaque portefeuille
    benchmark_name : str, optional
        Nom du portefeuille de référence pour le calcul du tracking error
    rf_series : pd.Series, optional
        Série du taux sans risque pour les ratios de Sharpe
    carbon_metrics_dict : dict, optional
        Dictionnaire d'empreinte carbone par portefeuille
    latex_path : str, optional
        Chemin pour sauvegarder en format LaTeX
    latex_column_names : dict, optional
        Noms LaTeX pour les colonnes du tableau
    show_te_only_for : list, optional
        Liste des noms de portefeuilles pour lesquels afficher le tracking error
        
    Returns:
    --------
    pd.DataFrame
        Tableau des métriques comparatives
    """
    # Calculer les métriques standards pour chaque portefeuille
    portfolio_metrics = {}
    for name, returns in portfolios_dict.items():
        portfolio_metrics[name] = calculate_portfolio_metrics(returns, rf_series)
        
        # Ajouter l'empreinte carbone moyenne si disponible
        if carbon_metrics_dict and name in carbon_metrics_dict and carbon_metrics_dict[name]:
            portfolio_metrics[name]["Annual Footprint Average"] = np.mean(list(carbon_metrics_dict[name].values()))
        else:
            portfolio_metrics[name]["Annual Footprint Average"] = np.nan
    
    # Créer le DataFrame des métriques
    metrics_df = pd.DataFrame(portfolio_metrics)
    
    # Calculer le tracking error vs benchmark seulement pour les portefeuilles spécifiés
    if benchmark_name and benchmark_name in portfolios_dict:
        benchmark_returns = portfolios_dict[benchmark_name]
        for name, returns in portfolios_dict.items():
            # Vérifier si on doit calculer le TE pour ce portefeuille
            if name != benchmark_name and (show_te_only_for is None or name in show_te_only_for):
                metrics_df.loc[f"TE: {name} VS {benchmark_name} (ann.)", name] = calculate_tracking_error(returns, benchmark_returns)
    
    # Exporter au format LaTeX si demandé
    if latex_path:
        latex_df = metrics_df.copy()
        if latex_column_names:
            latex_df = latex_df.rename(columns=latex_column_names)
        latex_df.to_latex(latex_path, index=True, float_format="%.5f")
    
    return metrics_df

# Mise à jour des portfolios pour inclure NZ
portfolios = {
    'MVP': mvp_series,
    'VW': vw_series,
    'MVP50': mvpc_series,
    'TE': te_series,
    'NZ': nz_series  # Ajout du portefeuille Net Zero
}

# Mise à jour des métriques de carbone pour inclure NZ
carbon_metrics = {
    'MVP': carbon_footprint_mvp,
    'VW': carbon_footprint_vw,
    'MVP50': constrained_cf_values,
    'TE': te_cf_values,
    'NZ': nz_cf  # Ajout de l'empreinte carbone de Net Zero
}

# Appel de la fonction pour générer le tableau complet avec tous les portefeuilles
metrics_table_all = compare_portfolio_metrics_selective_te(
    portfolios_dict=portfolios,
    benchmark_name='VW',  # Utiliser VW comme benchmark
    rf_series=rf[rf_column],
    carbon_metrics_dict=carbon_metrics,
    latex_path=None,
    latex_column_names={
        'MVP': r'$P_{oos}^{(mv)}$',
        'VW': r'$P^{(vw)}$',
        'MVP50': r'$P_{oos}^{(mv)}(0.5)$',
        'TE': r'$P_{oos}^{(vw)}(0.5)$',
        'NZ': r'$P_{oos}^{(vw)}(NZ)$'  # Ajout du nom LaTeX pour Net Zero
    },
    show_te_only_for=['NZ']  # Afficher le TE uniquement pour TE et NZ
)
# print(metrics_table_all)
metrics_table_all


# In[239]:


def create_portfolio_comparison_dashboard(
    portfolio1_series, 
    portfolio2_series, 
    portfolio3_series=None,
    portfolio4_series=None,
    portfolio5_series=None,
    rf_series=None, 
    portfolio1_name="Portfolio 1", 
    portfolio2_name="Portfolio 2", 
    portfolio3_name="Portfolio 3",
    portfolio4_name="Portfolio 4",
    portfolio5_name="Portfolio 5",
    colors=None, 
    figsize=(16, 12), 
    save_path=None,
    title="Portfolio Comparison",
    # Paramètres de contrôle pour les shifts
    cumulative_return_horizontal_shift=5,
    # Espacements verticaux individuels pour chaque portefeuille
    cumulative_return_vertical_spacing_p1=0,  
    cumulative_return_vertical_spacing_p2=0,
    cumulative_return_vertical_spacing_p3=0,
    cumulative_return_vertical_spacing_p4=0,
    cumulative_return_vertical_spacing_p5=0,
    drawdown_horizontal_shift=30,
    drawdown_arc_radius=0.3
):
    """
    Crée un tableau de bord simplifié comparant jusqu'à cinq portefeuilles avec:
    1. Rendements cumulés (en haut, plus grand)
    2. Drawdowns (en bas, plus petit)
    
    Paramètres:
    -----------
    portfolio1_series, portfolio2_series, portfolio3_series, portfolio4_series, portfolio5_series: pd.Series
        Séries temporelles des rendements des portefeuilles
    rf_series: pd.Series, optional
        Série temporelle du taux sans risque
    portfolio1_name, portfolio2_name, portfolio3_name, portfolio4_name, portfolio5_name: str
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
    cumulative_return_vertical_spacing_p1, p2, p3, p4, p5: int
        Contrôle l'espacement vertical des annotations pour chaque portefeuille individuellement
    drawdown_horizontal_shift: int
        Contrôle le shift horizontal des annotations de drawdown
    drawdown_arc_radius: float
        Contrôle la courbure des flèches de drawdown (0=droit, >0=courbe)
    """
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch
    
    # Créer un dictionnaire des espacements verticaux par portfolio
    vertical_spacings = {
        portfolio1_name: cumulative_return_vertical_spacing_p1,
        portfolio2_name: cumulative_return_vertical_spacing_p2,
        portfolio3_name: cumulative_return_vertical_spacing_p3,
        portfolio4_name: cumulative_return_vertical_spacing_p4,
        portfolio5_name: cumulative_return_vertical_spacing_p5
    }
    
    # Vérifier que les index des séries sont alignés
    all_series = [portfolio1_series, portfolio2_series]
    if portfolio3_series is not None:
        all_series.append(portfolio3_series)
    if portfolio4_series is not None:
        all_series.append(portfolio4_series)
    if portfolio5_series is not None:
        all_series.append(portfolio5_series)
    
    # Trouver l'index commun à toutes les séries
    common_index = all_series[0].index
    for series in all_series[1:]:
        common_index = common_index.intersection(series.index)
    
    if len(common_index) == 0:
        raise ValueError("Les séries de portefeuilles n'ont pas d'index commun")
    
    # Filtrer chaque série pour ne conserver que l'index commun
    portfolio1_series = portfolio1_series.loc[common_index]
    portfolio2_series = portfolio2_series.loc[common_index]
    if portfolio3_series is not None:
        portfolio3_series = portfolio3_series.loc[common_index]
    if portfolio4_series is not None:
        portfolio4_series = portfolio4_series.loc[common_index]
    if portfolio5_series is not None:
        portfolio5_series = portfolio5_series.loc[common_index]
    
    print(f"⚠️ Les index des séries ont été alignés. {len(common_index)} points communs conservés.")
    
    # Définir les couleurs par défaut si non spécifiées
    if colors is None:
        colors = {
            portfolio1_name: '#1f77b4',  # Bleu
            portfolio2_name: '#ff7f0e',  # Orange
            portfolio3_name: '#2ca02c',  # Vert
            portfolio4_name: '#d62728',  # Rouge
            portfolio5_name: '#9467bd'   # Violet
        }
    
    # S'assurer que tous les index sont convertis au même format datetime
    all_series_processed = []
    for series in all_series:
        series.index = pd.to_datetime(series.index)
        series.index = series.index.normalize()
        all_series_processed.append(series)
    
    portfolio1_series = all_series_processed[0]
    portfolio2_series = all_series_processed[1]
    if portfolio3_series is not None and len(all_series_processed) > 2:
        portfolio3_series = all_series_processed[2]
    if portfolio4_series is not None and len(all_series_processed) > 3:
        portfolio4_series = all_series_processed[3]
    if portfolio5_series is not None and len(all_series_processed) > 4:
        portfolio5_series = all_series_processed[4]
    
    # Calculate cumulative returns
    portfolio1_cumulative_returns = (1 + portfolio1_series).cumprod()
    portfolio2_cumulative_returns = (1 + portfolio2_series).cumprod()
    
    # DataFrame pour la comparaison des rendements cumulatifs
    cumulative_returns_comparison = {
        portfolio1_name: portfolio1_cumulative_returns,
        portfolio2_name: portfolio2_cumulative_returns
    }
    
    # Ajouter les autres portefeuilles s'ils sont fournis
    if portfolio3_series is not None:
        portfolio3_cumulative_returns = (1 + portfolio3_series).cumprod()
        cumulative_returns_comparison[portfolio3_name] = portfolio3_cumulative_returns
    
    if portfolio4_series is not None:
        portfolio4_cumulative_returns = (1 + portfolio4_series).cumprod()
        cumulative_returns_comparison[portfolio4_name] = portfolio4_cumulative_returns
        
    if portfolio5_series is not None:
        portfolio5_cumulative_returns = (1 + portfolio5_series).cumprod()
        cumulative_returns_comparison[portfolio5_name] = portfolio5_cumulative_returns
    
    cumulative_returns_df = pd.DataFrame(cumulative_returns_comparison)

    # Calculer les drawdowns pour chaque portefeuille
    # Pour le premier portefeuille
    portfolio1_drawdown = pd.Series(0.0, index=portfolio1_cumulative_returns.index, dtype=float)
    portfolio1_cummax = portfolio1_cumulative_returns.cummax()
    mask1 = portfolio1_cummax != 0
    portfolio1_drawdown.loc[mask1] = ((portfolio1_cummax.loc[mask1] - portfolio1_cumulative_returns.loc[mask1]) / 
                                    portfolio1_cummax.loc[mask1]).astype(float)

    # Pour le deuxième portefeuille
    portfolio2_drawdown = pd.Series(0.0, index=portfolio2_cumulative_returns.index, dtype=float)
    portfolio2_cummax = portfolio2_cumulative_returns.cummax()
    mask2 = portfolio2_cummax != 0
    portfolio2_drawdown.loc[mask2] = ((portfolio2_cummax.loc[mask2] - portfolio2_cumulative_returns.loc[mask2]) / 
                                    portfolio2_cummax.loc[mask2]).astype(float)
    
    # Pour le troisième portefeuille (si fourni)
    portfolio3_drawdown = None
    if portfolio3_series is not None:
        portfolio3_drawdown = pd.Series(0.0, index=portfolio3_cumulative_returns.index, dtype=float)
        portfolio3_cummax = portfolio3_cumulative_returns.cummax()
        mask3 = portfolio3_cummax != 0
        portfolio3_drawdown.loc[mask3] = ((portfolio3_cummax.loc[mask3] - portfolio3_cumulative_returns.loc[mask3]) / 
                                        portfolio3_cummax.loc[mask3]).astype(float)
    
    # Pour le quatrième portefeuille (si fourni)
    portfolio4_drawdown = None
    if portfolio4_series is not None:
        portfolio4_drawdown = pd.Series(0.0, index=portfolio4_cumulative_returns.index, dtype=float)
        portfolio4_cummax = portfolio4_cumulative_returns.cummax()
        mask4 = portfolio4_cummax != 0
        portfolio4_drawdown.loc[mask4] = ((portfolio4_cummax.loc[mask4] - portfolio4_cumulative_returns.loc[mask4]) / 
                                        portfolio4_cummax.loc[mask4]).astype(float)
                                        
    # Pour le cinquième portefeuille (si fourni)
    portfolio5_drawdown = None
    if portfolio5_series is not None:
        portfolio5_drawdown = pd.Series(0.0, index=portfolio5_cumulative_returns.index, dtype=float)
        portfolio5_cummax = portfolio5_cumulative_returns.cummax()
        mask5 = portfolio5_cummax != 0
        portfolio5_drawdown.loc[mask5] = ((portfolio5_cummax.loc[mask5] - portfolio5_cumulative_returns.loc[mask5]) / 
                                        portfolio5_cummax.loc[mask5]).astype(float)

    # Obtenir les valeurs maximales de drawdown
    portfolio1_max_dd = portfolio1_drawdown.max() 
    portfolio2_max_dd = portfolio2_drawdown.max()
    portfolio3_max_dd = None
    if portfolio3_drawdown is not None:
        portfolio3_max_dd = portfolio3_drawdown.max()
    portfolio4_max_dd = None
    if portfolio4_drawdown is not None:
        portfolio4_max_dd = portfolio4_drawdown.max()
    portfolio5_max_dd = None
    if portfolio5_drawdown is not None:
        portfolio5_max_dd = portfolio5_drawdown.max()

    # Set up a modern style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Create a function for custom styling
    def apply_custom_style(ax):
        """Apply custom styling to a matplotlib axis"""
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.7, color='#cccccc')
        ax.xaxis.label.set_fontsize(12)
        ax.yaxis.label.set_fontsize(12)
        ax.tick_params(axis='both', which='major', labelsize=10, colors='#333333')
        return ax

    # Create the dashboard with two main visualizations
    fig = plt.figure(figsize=figsize, facecolor='white', dpi=50)
    
    # Définir une GridSpec avec 2 lignes, une plus grande pour les rendements cumulés (80%)
    # et une plus petite pour les drawdowns (20%)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[0.80, 0.20])
    
    # 1. Cumulative Returns (en haut, plus grand)
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(cumulative_returns_df.index, cumulative_returns_df[portfolio1_name], 
            color=colors[portfolio1_name], linewidth=2, label=portfolio1_name)
    ax1.plot(cumulative_returns_df.index, cumulative_returns_df[portfolio2_name], 
            color=colors[portfolio2_name], linewidth=2, label=portfolio2_name)
    
    # Ajouter le troisième portefeuille s'il existe
    if portfolio3_name in cumulative_returns_df.columns:
        ax1.plot(cumulative_returns_df.index, cumulative_returns_df[portfolio3_name], 
                color=colors[portfolio3_name], linewidth=2, label=portfolio3_name)
    
    # Ajouter le quatrième portefeuille s'il existe
    if portfolio4_name in cumulative_returns_df.columns:
        ax1.plot(cumulative_returns_df.index, cumulative_returns_df[portfolio4_name], 
                color=colors[portfolio4_name], linewidth=2, label=portfolio4_name)
                
    # Ajouter le cinquième portefeuille s'il existe
    if portfolio5_name in cumulative_returns_df.columns:
        ax1.plot(cumulative_returns_df.index, cumulative_returns_df[portfolio5_name], 
                color=colors[portfolio5_name], linewidth=2, label=portfolio5_name)

    # Amélioration du placement des valeurs finales à droite avec shift vertical et horizontal
    # Créer un dictionnaire des valeurs finales
    final_values = {
        portfolio1_name: cumulative_returns_df[portfolio1_name].iloc[-1],
        portfolio2_name: cumulative_returns_df[portfolio2_name].iloc[-1],
    }
    
    if portfolio3_name in cumulative_returns_df.columns:
        final_values[portfolio3_name] = cumulative_returns_df[portfolio3_name].iloc[-1]
    
    if portfolio4_name in cumulative_returns_df.columns:
        final_values[portfolio4_name] = cumulative_returns_df[portfolio4_name].iloc[-1]
        
    if portfolio5_name in cumulative_returns_df.columns:
        final_values[portfolio5_name] = cumulative_returns_df[portfolio5_name].iloc[-1]
    
    # Trier les portefeuilles par valeur finale
    sorted_portfolios = sorted(final_values.items(), key=lambda x: x[1])
    
    # Placer les annotations avec les shifts verticaux spécifiques à chaque portefeuille
    for name, value in sorted_portfolios:
        # Le shift horizontal est constant pour toutes les annotations
        horizontal_shift = cumulative_return_horizontal_shift
        
        # Utiliser le shift vertical spécifique au portefeuille
        vertical_shift = vertical_spacings[name]
        
        ax1.annotate(f'{value:.2f}x', 
                    xy=(cumulative_returns_df.index[-1], value),
                    xytext=(horizontal_shift, vertical_shift),  # Shift horizontal et vertical personnalisé
                    textcoords='offset points',
                    ha='left', va='center', 
                    fontweight='bold', 
                    color=colors[name])

    # Ajouter le texte "Cumulative Return Growth of 1 USD" en bas à droite en gris discret
    ax1.annotate('Cumulative Return Growth of 1 USD', 
                xy=(0.95, 0.02),  # Position en bas à droite (coordonnées relatives)
                xycoords='axes fraction',  # Coordonnées relatives à l'axe
                ha='right', va='bottom',  # Alignement
                fontsize=10, color='#333333',  # Gris discret
                style='italic')  # Style italique pour plus de discrétion
                
    ax1.set_ylabel('Cumulative Return')
    
    # Hide x-axis labels for the top plot
    ax1.tick_params(axis='x', labelbottom=False)
    
    # Créer la légende pour les courbes
    ax1.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=10)
    apply_custom_style(ax1)

    # 2. Drawdowns (en bas, plus petit)
    ax2 = fig.add_subplot(gs[1])

    # Plot drawdowns with proper handling of values
    ax2.fill_between(portfolio1_drawdown.index, 0, -portfolio1_drawdown.values, 
                    alpha=0.3, color=colors[portfolio1_name], 
                    label=portfolio1_name, step=None)

    ax2.fill_between(portfolio2_drawdown.index, 0, -portfolio2_drawdown.values, 
                    alpha=0.3, color=colors[portfolio2_name], 
                    label=portfolio2_name, step=None)
    
    # Ajouter le drawdown du troisième portefeuille s'il existe
    if portfolio3_drawdown is not None:
        ax2.fill_between(portfolio3_drawdown.index, 0, -portfolio3_drawdown.values, 
                        alpha=0.3, color=colors[portfolio3_name], 
                        label=portfolio3_name, step=None)
    
    # Ajouter le drawdown du quatrième portefeuille s'il existe
    if portfolio4_drawdown is not None:
        ax2.fill_between(portfolio4_drawdown.index, 0, -portfolio4_drawdown.values, 
                        alpha=0.3, color=colors[portfolio4_name], 
                        label=portfolio4_name, step=None)
                        
    # Ajouter le drawdown du cinquième portefeuille s'il existe
    if portfolio5_drawdown is not None:
        ax2.fill_between(portfolio5_drawdown.index, 0, -portfolio5_drawdown.values, 
                        alpha=0.3, color=colors[portfolio5_name], 
                        label=portfolio5_name, step=None)

    ax2.plot(portfolio1_drawdown.index, -portfolio1_drawdown.values, alpha=0.7, color=colors[portfolio1_name], linewidth=1)
    ax2.plot(portfolio2_drawdown.index, -portfolio2_drawdown.values, alpha=0.7, color=colors[portfolio2_name], linewidth=1)
    
    # Ajouter les lignes de drawdown pour les portefeuilles supplémentaires
    if portfolio3_drawdown is not None:
        ax2.plot(portfolio3_drawdown.index, -portfolio3_drawdown.values, alpha=0.7, color=colors[portfolio3_name], linewidth=1)
    
    if portfolio4_drawdown is not None:
        ax2.plot(portfolio4_drawdown.index, -portfolio4_drawdown.values, alpha=0.7, color=colors[portfolio4_name], linewidth=1)
        
    if portfolio5_drawdown is not None:
        ax2.plot(portfolio5_drawdown.index, -portfolio5_drawdown.values, alpha=0.7, color=colors[portfolio5_name], linewidth=1)

    # Find maximum drawdowns dates
    portfolio1_valley_date = portfolio1_drawdown.idxmax()
    portfolio2_valley_date = portfolio2_drawdown.idxmax()
    portfolio3_valley_date = None
    if portfolio3_drawdown is not None:
        portfolio3_valley_date = portfolio3_drawdown.idxmax()
    portfolio4_valley_date = None
    if portfolio4_drawdown is not None:
        portfolio4_valley_date = portfolio4_drawdown.idxmax()
    portfolio5_valley_date = None
    if portfolio5_drawdown is not None:
        portfolio5_valley_date = portfolio5_drawdown.idxmax()

    # Amélioration du placement des annotations de Max DD avec flèches externes et arcs
    # Collecter toutes les informations sur les drawdowns
    drawdowns = [
        (portfolio1_name, portfolio1_max_dd, portfolio1_valley_date, colors[portfolio1_name]),
        (portfolio2_name, portfolio2_max_dd, portfolio2_valley_date, colors[portfolio2_name])
    ]
    
    if portfolio3_drawdown is not None:
        drawdowns.append((portfolio3_name, portfolio3_max_dd, portfolio3_valley_date, colors[portfolio3_name]))
    
    if portfolio4_drawdown is not None:
        drawdowns.append((portfolio4_name, portfolio4_max_dd, portfolio4_valley_date, colors[portfolio4_name]))
        
    if portfolio5_drawdown is not None:
        drawdowns.append((portfolio5_name, portfolio5_max_dd, portfolio5_valley_date, colors[portfolio5_name]))
    
    # Trier par date pour une meilleure organisation des annotations
    drawdowns.sort(key=lambda x: x[2])
    
    # Calculer la valeur de drawdown maximale pour dimensionner la zone des annotations
    max_dd_value = max([dd[1] for dd in drawdowns])
    
    # Positionner les annotations à l'extérieur avec des flèches courbes
    for i, (name, max_dd, valley_date, color) in enumerate(drawdowns):
        # Décaler les annotations horizontalement pour éviter les chevauchements
        # Alternance droite/gauche pour les annotations
        horizontal_shift = drawdown_horizontal_shift * (-1 if i % 2 == 0 else 1)
        
        # Direction de l'arc basée sur la position (gauche ou droite)
        arc_rad = drawdown_arc_radius * (-1 if i % 2 == 0 else 1)
        
        ax2.annotate(f'Max DD: -{max_dd:.2%}',
                    xy=(valley_date, -max_dd),  # Point à annoter (drawdown max)
                    xytext=(horizontal_shift, 0),  # Shift horizontal
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
    
    # Ajuster les limites pour avoir une marge en bas et en haut
    # pour accommoder les annotations extérieures
    all_max_dd = max([dd[1] for dd in drawdowns])
    ax2.set_ylim(-all_max_dd * 1.3, all_max_dd * 0.4)  # Plus d'espace en haut pour les annotations
    
    # Légende pour le graphique de drawdowns
    ax2.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=8)
    apply_custom_style(ax2)

    # Ajuster la mise en page avec peu d'espace entre les graphiques
    # plt.tight_layout()
    
    # Ajouter le titre principal tout en haut et en gras
    fig.suptitle(title, fontsize=18, fontweight='bold', x=0.5, y=0.98, ha='center')
    
    # Ajuster l'espace pour coller les graphiques et accommoder le titre
    plt.subplots_adjust(top=0.92, hspace=0.0)
    
    if save_path:
        plt.savefig(save_path, dpi=50, bbox_inches='tight')
        print(f"Dashboard saved to {save_path}")
    
    return fig, [ax1, ax2]


# In[240]:


fig, axes = create_portfolio_comparison_dashboard(
    vw_series,
    mvp_series,
    mvpc_series,
    te_series,
    nz_series,
    portfolio1_name=r'$P^{(v w)}$',
    portfolio2_name=r'$P_{o o s}^{(m v)}$',
    portfolio3_name=r'$P_{o o s}^{(m v)}(0.5)$',
    portfolio4_name=r'$P_{o o s}^{(v w)}(0.5)$',
    portfolio5_name=r'$P_{o o s}^{(v w)}(NZ)$',
    save_path="Five_Portfolio_Comparison.png" if save_images else None,
    colors={
        r'$P^{(v w)}$': '#1f77b4',            # Bleu pour VW
        r'$P_{o o s}^{(m v)}$': '#ff7f0e',    # Orange pour MVP
        r'$P_{o o s}^{(m v)}(0.5)$': '#2ca02c', # Vert pour MVP05
        r'$P_{o o s}^{(v w)}(0.5)$': '#d62728',  # Rouge pour TE
        r'$P_{o o s}^{(v w)}(NZ)$': '#9467bd'   # Violet pour Net Zero
    },
    figsize=(22, 8),
    title=r'Comparison between Five Portfolio Strategies',
    # Contrôle individuel des espacements verticaux
    cumulative_return_vertical_spacing_p1=-4,    # VW: shift vers le haut
    cumulative_return_vertical_spacing_p2=-3,    # MVP: shift vers le bas
    cumulative_return_vertical_spacing_p3=6,   # MVP50: léger shift vers le bas
    cumulative_return_vertical_spacing_p4=7,   # TE: shift vers le haut
    cumulative_return_vertical_spacing_p5=-4,    # NZ: pas de shift
    # Autres paramètres
    cumulative_return_horizontal_shift=10,
    drawdown_horizontal_shift=40,
    drawdown_arc_radius=0.3
)


# In[241]:


vw_returns = vw_portfolio_returns if isinstance(vw_portfolio_returns, pd.Series) else vw_portfolio_returns['Portfolio_Return']
min_var_returns = portfolio_returns_all['Portfolio_Return']
mvpc_returns = mvpc['Portfolio_Return']
te_returns = te_portfolio_returns['Portfolio_Return']
nz_returns = nz_portfolio_returns['Portfolio_Return']


# In[242]:


def plot_nz_vw_carbon_comparison_corrected(nz_cf_values, vw_cf_values, nz_returns, vw_returns, 
                                         nz_constraints=None, figsize=(14, 10), save_path=None, 
                                         colors=None):
    """
    Crée une visualisation professionnelle corrigée comparant le portefeuille Net Zero
    par rapport au benchmark VW (P^(vw)), en assurant la cohérence des contraintes.
    
    Parameters
    ----------
    nz_cf_values : dict
        Valeurs d'empreinte carbone du portefeuille Net Zero par année
    vw_cf_values : dict
        Valeurs d'empreinte carbone du portefeuille VW par année
    nz_returns : pd.Series
        Rendements mensuels du portefeuille Net Zero
    vw_returns : pd.Series
        Rendements mensuels du portefeuille VW (benchmark)
    nz_constraints : dict, optional
        Contraintes Net Zero théoriques par année
    figsize : tuple, optional
        Dimensions de la figure (largeur, hauteur)
    save_path : str, optional
        Chemin pour sauvegarder la figure
    colors : dict, optional
        Couleurs personnalisées pour les différents éléments
        
    Returns
    -------
    fig, axes : tuple
        Figure et axes pour d'éventuelles personnalisations supplémentaires
    """
    from matplotlib.gridspec import GridSpec
    from scipy import stats
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import matplotlib.dates as mdates
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    
    # Palette de couleurs personnalisée pour TE
    te_c = [
        (0.90, 0.90, 0.90),  # Gris clair (plus visible)
        (0.75, 0.76, 0.80),  # Gris bleuté (plus foncé)
        (0.60, 0.65, 0.75),  # Gris bleu moyen
        (0.45, 0.50, 0.60)   # Gris bleu foncé
    ]
    pastel_viridis = LinearSegmentedColormap.from_list('pastel_viridis', te_c)
    
    # Définir les couleurs par défaut si non fournies
    if colors is None:
        colors = {
            'nz': '#9467bd',       # Violet pour le portefeuille Net Zero
            'vw': '#1f77b4',       # Bleu pour le portefeuille VW
            'target': '#ff5117',   # Orange pour la contrainte cible
            'constraint_zone': '#e6f3ff',  # Bleu très clair pour la zone de contrainte
            'tracking_error': '#657994',   # Bleu grisé pour l'erreur de suivi
            'compliant': '#9467bd',        # Violet pour contrainte respectée
            'non_compliant': '#d62728'     # Rouge pour contrainte non respectée
        }
    
    # Style professionnel
    plt.style.use('seaborn-v0_8-whitegrid')
    
    
    # Créer une figure avec GridSpec pour contrôle de mise en page
    fig = plt.figure(figsize=figsize, dpi=50, facecolor='white')
    gs = GridSpec(2, 1, height_ratios=[5, 1], hspace=0.05)  # Ajustement du ratio comme référence
    
    # Titre principal
    fig.suptitle(r"Net Zero Portfolio $P_{oos}^{(vw)}(NZ)$ vs Value-Weighted Benchmark $P^{(vw)}$ Carbon Footprint Trajectory", 
                fontsize=16, fontweight='bold', y=0.98)
    
    # ====================== PANNEAU SUPÉRIEUR: TRAJECTOIRE CARBONE ====================== #
    ax1 = fig.add_subplot(gs[0])
    
    # Années communes à tous les ensembles de données
    common_years = sorted(set(nz_cf_values) & set(vw_cf_values))
    numeric_years = [int(y) for y in common_years]
    x_positions = np.arange(len(numeric_years))
    
    # Extraire les valeurs d'empreintes carbones réelles
    nz_values = [nz_cf_values[y] for y in common_years]
    vw_values = [vw_cf_values[y] for y in common_years]
    
    # Traitement des contraintes Net Zero théoriques
    constraint_values = []
    
    # S'assurer que nous avons des contraintes pour chaque année
    if nz_constraints and len(nz_constraints) > 0:
        for year in common_years:
            if year in nz_constraints:
                constraint_values.append(nz_constraints[year])
            else:
                # Pour les années sans contraintes explicites, faire une extrapolation
                print(f"⚠️ Année {year} manquante dans les contraintes, utilisation d'extrapolation")
                
                # Trouver la contrainte la plus proche
                closest_year = min(nz_constraints.keys(), key=lambda y: abs(int(y) - int(year)))
                years_diff = int(year) - int(closest_year)
                base_value = nz_constraints[closest_year]
                
                # Extrapoler avec un taux de 10% par an
                constraint_values.append(base_value * ((1-0.10) ** years_diff))
    else:
        # Si pas de contraintes fournies, utiliser les valeurs VW de la première année avec réduction de 10%
        base_value = vw_values[0]
        constraint_values = [base_value * ((1-0.10) ** i) for i in range(len(common_years))]
    
    # Créer une liste qui indique si chaque année est conforme à la contrainte
    # Critère modifié: différence absolue <= 0.01 pour considérer respectée
    is_compliant = [abs(nz_values[i] - constraint_values[i]) <= 0.001 or nz_values[i] <= constraint_values[i] for i in range(len(nz_values))]
    
    # Afficher un tableau des valeurs pour vérification
    print("\n=== VALEURS UTILISÉES POUR LE GRAPHIQUE (CORRIGÉ) ===")
    print(f"{'Année':<6} {'VW':<10} {'NZ Actual':<10} {'NZ Constraint':<15} {'Compliant':<10} {'Reduction':<10}")
    print("-" * 65)
    for i, year in enumerate(common_years):
        reduction = (1 - nz_values[i]/vw_values[i]) * 100 if vw_values[i] > 0 else 0
        print(f"{year:<6} {vw_values[i]:<10.2f} {nz_values[i]:<10.2f} {constraint_values[i]:<15.2f} {'✓' if is_compliant[i] else '✗':<10} {reduction:<10.1f}%")
    
    # Créer le graphique à barres pour la VW et les valeurs réelles NZ
    bar_width = 0.35
    bars1 = ax1.bar(x_positions - bar_width/2, vw_values, bar_width, color=colors['vw'], 
                  alpha=0.85, label=r'$P^{(vw)}$ Carbon Footprint', edgecolor='black', linewidth=0.5)
    
    # Utiliser des couleurs différentes pour les barres NZ selon la conformité
    for i, x_pos in enumerate(x_positions + bar_width/2):
        bar_color = colors['compliant'] if is_compliant[i] else colors['non_compliant']
        bar = ax1.bar(x_pos, nz_values[i], bar_width, color=bar_color, 
                    alpha=0.85, edgecolor='black', linewidth=0.5)
        if i == 0:  # Ajouter seulement à la légende une fois
            bar[0].set_label(r'$P_{oos}^{(vw)}(NZ)$ Carbon Footprint')
    
    # Ajouter la ligne de contrainte Net Zero avec un style distinctif
    ax1.plot(x_positions, constraint_values, '--', linewidth=1.8, color=colors['target'], 
           label='Net Zero Constraint (10% reduction/year)', marker='o', markersize=4)
    
    # Colorer la zone sous la contrainte
    ax1.fill_between(x_positions, [0]*len(numeric_years), constraint_values, 
                   color=colors['constraint_zone'], alpha=0.3)
    
    # Ajouter les pourcentages de réduction et annotations avec espacement amélioré
    for i, (bar1, nz_value) in enumerate(zip(bars1, nz_values)):
        # Calculer le pourcentage de réduction de l'empreinte carbone réelle par rapport au benchmark
        reduction_pct = (1 - nz_value/vw_values[i]) * 100 if vw_values[i] > 0 else 0
        
        # Position verticale dynamique pour éviter les chevauchements
        vertical_offset = 5

        # Annotations pour les pourcentages de réduction
        ax1.annotate(f'{reduction_pct:.1f}%', 
                   xy=(x_positions[i] + bar_width/2, nz_value),
                   xytext=(3, vertical_offset), textcoords='offset points',
                   ha='center', va='bottom', fontsize=9, fontweight='bold',
                   color=colors['compliant'] if is_compliant[i] else colors['non_compliant'])
        
        # Ajouter les valeurs absolues sur les barres avec position décalée si nécessaire
        ax1.annotate(f'{vw_values[i]:.0f}',
                   xy=(x_positions[i] - bar_width/2, vw_values[i]),
                   xytext=(0, -15), textcoords='offset points', 
                   ha='center', va='top', fontsize=8,
                   color='white', fontweight='bold')
        
        ax1.annotate(f'{nz_value:.0f}',
                   xy=(x_positions[i] + bar_width/2, nz_value),
                   xytext=(0, -15), textcoords='offset points', 
                   ha='center', va='top', fontsize=8,
                   color='white', fontweight='bold')
    
    # Afficher explicitement les valeurs de contrainte sur le graphique avec espacement alterné
    for i, constraint in enumerate(constraint_values):
        # Position verticale alternée pour éviter les chevauchements
        vertical_pos = 25 if i % 2 == 0 else 40
        horizontal_pos = 10
        
        # Utiliser une connexion en arc pour éviter les conflits
        ax1.annotate(f'{constraint:.0f}',
                   xy=(x_positions[i], constraint),
                   xytext=(horizontal_pos, vertical_pos),  # Position élevée
                   textcoords='offset points',
                   ha='center', va='bottom', fontsize=8,
                   color=colors['target'],
                   bbox=dict(boxstyle="round,pad=0.2", fc='white', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=0.2", color=colors['target']))
    
    # Calculer les statistiques de conformité et de réduction
    compliant_years = sum(is_compliant)
    compliance_rate = (compliant_years / len(is_compliant)) * 100 if len(is_compliant) > 0 else 0
    
    # Calculer les statistiques de réduction
    reductions = [(1 - nz_values[i]/vw_values[i])*100 for i in range(len(nz_values))
                 if vw_values[i] > 0]
    avg_reduction = np.mean(reductions) if reductions else 0
    
    # Remplacer l'annotation de la première barre VW par Y_0 avec une flèche
    if len(bars1) > 0:
        # Supprimer l'annotation existante pour la première barre
        for txt in ax1.texts:
            if txt.get_position()[0] == x_positions[0] - bar_width/2 and txt.get_text() == f'{vw_values[0]:.0f}':
                txt.remove()
                break
        
        # Ajouter l'annotation Y_0 avec une flèche
        ax1.annotate(r'$Y_{0}$', 
                    xy=(x_positions[0] - bar_width/2, vw_values[0]),
                    xytext=(-20, 20),  # Position de l'étiquette à gauche et au-dessus de la barre
                    textcoords='offset points',
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='black'))
        
    # Formater le panneau supérieur
    ax1.set_ylabel(r'Carbon Footprint (tCO$ _2$e/M\$)', fontsize=12)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([])  # Masquer les étiquettes x pour le panneau supérieur
    ax1.grid(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=10)
    ax1.set_axisbelow(True)
    
    # Ajouter une annotation sur la zone de contrainte
    if len(constraint_values) > 3:  # S'assurer qu'il y a assez de points
        mid_idx = len(x_positions) // 2
        mid_y = constraint_values[mid_idx] // 2
        
        ax1.annotate('Net Zero Constraint Zone', 
                    xy=(x_positions[mid_idx], mid_y),
                    xytext=(0, -10), textcoords='offset points',
                    ha='center', va='top', fontsize=8, style='italic',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec='#cccccc'),
                    arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2", color='#555555'))
    
    # ====================== PANNEAU INFÉRIEUR: ERREUR DE SUIVI ====================== #
    ax2 = fig.add_subplot(gs[1])
    
    # Assurer que les index sont en datetime
    if not isinstance(nz_returns, pd.Series):
        if hasattr(nz_returns, 'columns') and 'Portfolio_Return' in nz_returns.columns:
            nz_returns = nz_returns['Portfolio_Return']
        elif hasattr(nz_returns, 'iloc'):
            nz_returns = nz_returns.iloc[:, 0]
        else:
            raise ValueError("Format de nz_returns non reconnu")
    
    if not isinstance(vw_returns, pd.Series):
        if hasattr(vw_returns, 'columns') and 'Portfolio_Return' in vw_returns.columns:
            vw_returns = vw_returns['Portfolio_Return']
        elif hasattr(vw_returns, 'iloc'):
            vw_returns = vw_returns.iloc[:, 0]
        else:
            raise ValueError("Format de vw_returns non reconnu")
    
    nz_returns.index = pd.to_datetime(nz_returns.index)
    vw_returns.index = pd.to_datetime(vw_returns.index)
    
    # Aligner les séries de rendement par index
    common_dates = nz_returns.index.intersection(vw_returns.index)
    tracking_error = (nz_returns.loc[common_dates] - vw_returns.loc[common_dates])
    
    # Calculer l'erreur de suivi annualisée
    annual_te = tracking_error.std() * np.sqrt(12) * 100  # Annualisée et en pourcentage
    
    # Grouper l'erreur de suivi par année pour un axe x cohérent avec le panneau carbone
    yearly_te = {}
    for year in numeric_years:
        year_mask = tracking_error.index.year == year
        if year_mask.any():
            yearly_te[str(year)] = tracking_error.loc[year_mask].values
        else:
            yearly_te[str(year)] = np.array([np.nan])
    
    # Calculer l'erreur de suivi absolue moyenne par année
    avg_abs_te_by_year = []
    for year in common_years:
        values = yearly_te[str(int(year))]
        if len(values) > 0:
            # Filtrer les valeurs non-NaN avant de calculer la moyenne
            # Si vous savez quelle valeur représente un NaN dans vos données
            valid_values = [v for v in values if v != 'NaN' and v is not None]
            if len(valid_values) > 0:
                avg_abs_te_by_year.append(np.mean(np.abs(valid_values)) * 100)
            else:
                avg_abs_te_by_year.append(np.nan)
        else:
            avg_abs_te_by_year.append(np.nan)
    
    # Utiliser la colormap personnalisée, normaliser les données entre 0-1 pour les couleurs
    abs_te_max = np.nanmax(avg_abs_te_by_year)
    if abs_te_max > 0:
        # Normaliser les valeurs entre 0 et 1
        color_idx = np.array(avg_abs_te_by_year) / abs_te_max
        # Remplacer les NaN par 0 pour éviter les erreurs
        color_idx = np.nan_to_num(color_idx)
        # Utiliser la colormap personnalisée pour chaque année
        bar_colors = [pastel_viridis(idx) for idx in color_idx]
    else:
        # Utiliser une couleur par défaut si toutes les valeurs sont NaN ou 0
        bar_colors = [pastel_viridis(0.5)] * len(avg_abs_te_by_year)
    
    # Tracer l'erreur de suivi absolue moyenne par année sous forme de barres
    bars3 = ax2.bar(x_positions, avg_abs_te_by_year, color=bar_colors, alpha=0.7,
                   width=0.6, edgecolor='black', linewidth=0.5)
    
    # Ajouter une ligne pour l'erreur de suivi annualisée sur toute la période
    ax2.axhline(annual_te, linestyle='--', color='#d62728', linewidth=1.5, 
               label=f'Annualized TE: {annual_te:.2f}%')
    
    # Ajouter les étiquettes de valeur sur les barres
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        if not np.isnan(height):
            # Espacer les annotations verticalement en fonction de l'index
            vertical_offset = 3 if i % 2 == 0 else 8
            
            ax2.annotate(f'{height:.2f}%', 
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, vertical_offset), textcoords='offset points',
                       ha='center', va='bottom', fontsize=9)
    
    ax2.text(0.02, 0.80, r'$\mathbf{P_{oos}^{(vw)}(NZ)}$ - $\mathbf{P^{(vw)}}$ Tracking Error', 
            transform=ax2.transAxes, fontsize=8,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
            facecolor='white', alpha=0.9))
    
    # Formater le panneau d'erreur de suivi
    ax2.set_xlabel('Year', fontsize=12, labelpad=10)
    ax2.set_ylabel('Absolute TE (%)', fontsize=10, labelpad=10)
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(common_years, rotation=0)
    ax2.grid(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax2.set_axisbelow(True)
    
    # Ajuster la mise en page
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.12, top=0.92, hspace=0.05)
    
    # Sauvegarder si un chemin est fourni
    if save_path:
        plt.savefig(save_path, dpi=50, bbox_inches='tight')
        print(f"Figure sauvegardée dans {save_path}")
    
    return fig, (ax1, ax2)


# In[243]:


nz_constraints = nz_bornes(carbon_footprint_vw, base_year='2014', theta=0.10, last_year='2024', verbose=True)




# Vérifier et afficher les contraintes pour s'assurer de leur exactitude
# print("Vérification des contraintes:")
# for year in sorted(nz_constraints.keys()):
#     print(f"{year}: {nz_constraints[year]:.2f}")

# Appel de la fonction avec debug=True pour voir les valeurs exactes utilisées
fig, axes = plot_nz_vw_carbon_comparison_corrected(
    nz_cf_values=nz_cf,
    vw_cf_values=carbon_footprint_vw,
    nz_returns=nz_portfolio_returns['Portfolio_Return'],
    vw_returns=vw_returns,
    nz_constraints=nz_constraints,
    figsize=(22,8),
    save_path='net_zero_portfolio_comparison_corrected.png' if save_images else None
)


# In[244]:


# Création du DataFrame pour le tableau
carbon_metrics_data_nz = []

# Récupération des années communes
common_years = sorted(set(nz_cf.keys()) & set(nz_constraints.keys()) & set(carbon_footprint_vw.keys()))

# Assemblage des données pour chaque année
for year in common_years:
    vw_footprint = carbon_footprint_vw[year]
    nz_constraint = nz_constraints[year]
    nz_footprint = nz_cf[year]
    
    # Calcul des réductions en pourcentage
    expected_reduction = ((nz_constraint - vw_footprint) / vw_footprint) * 100  # Réduction théorique par rapport au VW courant
    effective_reduction = ((nz_footprint - vw_footprint) / vw_footprint) * 100  # Réduction réellement obtenue
    
    carbon_metrics_data_nz.append({
        'Year': year,
        'VW Footprint': vw_footprint,
        'NZ Constraint': nz_constraint,
        'NZ Footprint': nz_footprint,
        'Expected Reduction (%)': expected_reduction,
        'Effective Reduction (%)': effective_reduction
    })

# Création du DataFrame et formatage
carbon_metrics_table = pd.DataFrame(carbon_metrics_data_nz)
carbon_metrics_display = carbon_metrics_table.set_index('Year')

# Arrondir les valeurs pour une meilleure lisibilité
carbon_metrics_display = carbon_metrics_display.round(2)

# Renommage des colonnes avec notation LaTeX pour le tableau final
carbon_metrics_latex = carbon_metrics_display.rename(columns={
    'VW Footprint': r'CF_{Y}^{(P^{(vw)})}',
    'NZ Constraint': r'CF_{Y}^{(NZ constraint)}',
    'NZ Footprint': r'CF_{Y}^{(P_{oos}^{(vw)}(NZ))}',
    'Expected Reduction (%)': r'\Delta_{expected} (\%)',
    'Effective Reduction (%)': r'\Delta_{effective} (\%)'
})

# Ajout d'une ligne de moyennes
carbon_metrics_latex.loc['Mean'] = carbon_metrics_latex.mean()

print('NZ Constraints Expected vs Realized')

print(carbon_metrics_latex)
# Afficher le tableau
# carbon_metrics_latex.to_latex('Article_Latex/NZ_Contraintes.tex', index=True, float_format="%.2f")


# In[245]:


print('\n\n\n\n\n\n')
print('# -----------------------WACI Comparison And other mesures.------------------- #')


# In[246]:


print('WACI Comparison MVPvsMVP50, VW vs TE vs TE')
# 1. Calcul du WACI et CF pour MVP50, TE et NZ
# Utilisation de la fonction carbon_metrics_all_years pour chaque portefeuille
mvp50_metrics_c = carbon_metrics_all_years(annual_filtered_data_, constrained_weights)
te_metrics_c = carbon_metrics_all_years(annual_filtered_data_, te_weights)
nz_metrics_c = carbon_metrics_all_years(annual_filtered_data_, nz_weights)

# Extraction des données WACI pour chaque portefeuille
waci_mvp50 = dict(get_carbon_metrics(mvp50_metrics_c, 'WACI'))
waci_te = dict(get_carbon_metrics(te_metrics_c, 'WACI'))
waci_nz = dict(get_carbon_metrics(nz_metrics_c, 'WACI'))

# 2. Création des DataFrames de comparaison
# Comparaison entre MVP et MVP50
compare_mvp_mvp50 = compare_carbon_metrics(mvp50_metrics_c, mvp_metrics_c, "MVP50", "MVP")

# Renommer les colonnes pour LaTeX - MVP vs MVP50
mvp_mvp50_latex = compare_mvp_mvp50[['WACI_MVP', 'WACI_MVP50', 'WACI Δ%', 'Firms_MVP']].round(2)
mvp_mvp50_latex = mvp_mvp50_latex.rename(columns={
    'WACI_MVP': 'WACI$^{(P_{oos}^{mv})}_{Y}$',
    'WACI_MVP50': 'WACI$^{(P_{oos}^{mv}(0.5))}_{Y}$',
    'WACI Δ%': '\\Delta (\\%)',
    'Firms_MVP': 'N$_{\\text{firms}}$'
})

# Ajouter une ligne avec les moyennes
mvp_mvp50_latex.loc['Average'] = mvp_mvp50_latex.mean()

print("Comparaison WACI entre MVP et MVP50 (format LaTeX):")
print(mvp_mvp50_latex)

# Exporter vers LaTeX si nécessaire
# mvp_mvp50_latex.to_latex('Article_Latex/mvp_vs_mvp50_waci.tex', index=True, float_format="%.2f")

# Pour la comparaison entre VW, TE et NZ, nous devons faire deux comparaisons séparées puis les fusionner
compare_vw_te = compare_carbon_metrics(te_metrics_c, vw_metrics_c, "TE", "VW")
compare_vw_nz = compare_carbon_metrics(nz_metrics_c, vw_metrics_c, "NZ", "VW")

# Fusionner les deux comparaisons
compare_te_nz_vw = compare_vw_te.join(compare_vw_nz['WACI_NZ'])
compare_te_nz_vw['WACI TE Δ%'] = 100 * (compare_te_nz_vw['WACI_TE']/compare_te_nz_vw['WACI_VW'] - 1)
compare_te_nz_vw['WACI NZ Δ%'] = 100 * (compare_te_nz_vw['WACI_NZ']/compare_te_nz_vw['WACI_VW'] - 1)

# Sélectionner et renommer les colonnes pour LaTeX - VW vs TE vs NZ
vw_te_nz_latex = compare_te_nz_vw[['WACI_VW', 'WACI_TE', 'WACI TE Δ%', 'WACI_NZ', 'WACI NZ Δ%', 'Firms_TE']].round(2)
vw_te_nz_latex = vw_te_nz_latex.rename(columns={
    'WACI_VW': 'WACI$^{(P^{(vw)})}_{Y}$',
    'WACI_TE': 'WACI$^{(P_{oos}^{(vw)}(0.5))}_{Y}$',
    'WACI TE Δ%': '\\Delta_{TE} (\\%)',
    'WACI_NZ': 'WACI$^{(P_{oos}^{(vw)}(NZ))}_{Y}$',
    'WACI NZ Δ%': '\\Delta_{NZ} (\\%)',
    'Firms_TE': 'N$_{\\text{firms}}$'
})

# Ajouter une ligne avec les moyennes
vw_te_nz_latex.loc['Average'] = vw_te_nz_latex.mean()

print("\nComparaison WACI entre VW, TE et NZ (format LaTeX):")
# print(vw_te_nz_latex)

# Exporter vers LaTeX si nécessaire
# vw_te_nz_latex.to_latex('Article_Latex/vw_vs_te_nz_waci.tex', index=True, float_format="%.2f")
vw_te_nz_latex


# In[247]:


# 1. Calcul du WACI pour MVP50, TE, NZ
# Pour MVP50 (portefeuille contraint à 50% de l'empreinte carbone du MVP)
mvpc_metrics_c = carbon_metrics_all_years(annual_filtered_data_, constrained_weights)
waci_mvpc = dict(get_carbon_metrics(mvpc_metrics_c, 'WACI'))

# Pour TE (portefeuille tracking error avec contrainte carbone)
te_metrics_c = carbon_metrics_all_years(annual_filtered_data_, te_weights)
waci_te = dict(get_carbon_metrics(te_metrics_c, 'WACI'))

# Pour NZ (portefeuille net-zero)
nz_metrics_c = carbon_metrics_all_years(annual_filtered_data_, nz_weights)
waci_nz = dict(get_carbon_metrics(nz_metrics_c, 'WACI'))


# In[248]:


# Script 1: Comparaison WACI entre MVP et MVP50
colors_companies = [
    (0.98, 0.98, 0.98),  # Blanc légèrement gris
    (0.85, 0.86, 0.88),  # Gris très légèrement bleuté
    (0.75, 0.78, 0.85),  # Gris bleu clair
    (0.65, 0.7, 0.8)     # Gris bleu argenté
]
pastel_viridis = LinearSegmentedColormap.from_list('pastel_viridis', colors_companies)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['text.usetex'] = False  # Use LaTeX for text rendering

# Créer la figure avec une résolution plus élevée
fig = plt.figure(figsize=(22, 8), dpi=50)
# Réduire l'espace entre les graphes avec hspace=0.05
gs = GridSpec(2, 1, height_ratios=[5, 1], hspace=0.05)

# Graphique principal (WACI)
ax = fig.add_subplot(gs[0])

# Préparation des données
years = list(waci_mvp.keys())
waci_values_mvp = list(waci_mvp.values())
waci_values_mvp50 = [waci_mvp50[year] for year in years]
x = np.arange(len(years))

# Calculer les moyennes pour les deux séries
avg_waci_mvp = np.mean(waci_values_mvp)
avg_waci_mvp50 = np.mean(waci_values_mvp50)

# Largeur des barres pour le groupe
bar_width = 0.35

# Tracer les barres pour MVP
bars_mvp = ax.bar(x - bar_width/2, waci_values_mvp, width=bar_width, 
                 color='#ff7f0e', edgecolor='black', linewidth=0.5,
                 alpha=0.85, zorder=10, label=r'$P_{oos}^{(mv)}$ WACI')

# Tracer les barres pour MVP50
bars_mvp50 = ax.bar(x + bar_width/2, waci_values_mvp50, width=bar_width,
                color='#2ca02c', edgecolor='black', linewidth=0.5,
                alpha=0.85, zorder=10, label=r'$P_{oos}^{(mv)}(0.5)$ WACI')

# Ajouter les lignes de moyenne
ax.axhline(avg_waci_mvp, color='#ff7f0e', linestyle='-', linewidth=1.2, 
           label=f'MVP Average: {avg_waci_mvp:.2f} tCO$_2$-eq/MUSD', zorder=15, alpha=0.7)
ax.axhline(avg_waci_mvp50, color='#2ca02c', linestyle='-', linewidth=1.2, 
           label=f'MVP50 Average: {avg_waci_mvp50:.2f} tCO$_2$-eq/MUSD', zorder=15, alpha=0.7)

# Ajouter les annotations de valeur sur chaque barre
for i, bar in enumerate(bars_mvp):
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 11),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#ff7f0e', zorder=30)

for i, bar in enumerate(bars_mvp50):
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 11),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#2ca02c', zorder=30)

# Identifier les valeurs extrêmes pour MVP
min_idx_mvp = np.argmin(waci_values_mvp)
max_idx_mvp = np.argmax(waci_values_mvp)

# Ajouter une annotation pour la valeur minimale du MVP
ax.annotate(f'MVP Min: {waci_values_mvp[min_idx_mvp]:.2f}',
            xy=(min_idx_mvp - bar_width/2, waci_values_mvp[min_idx_mvp] + 5),
            xytext=(-40, 150),
            textcoords="offset points",
            ha='center', va='bottom',
            arrowprops=dict(
                arrowstyle='->',
                connectionstyle='arc3,rad=-.2',
                color='#ff7f0e',
                shrinkB=18,
                relpos=(0.5, 0.0)
            ),
            fontweight='bold', color='#ff7f0e', zorder=30,
            fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Ajouter une annotation pour la valeur maximale du MVP
ax.annotate(f'MVP Max: {waci_values_mvp[max_idx_mvp]:.2f}',
            xy=(max_idx_mvp - bar_width/2, waci_values_mvp[max_idx_mvp] + 8),
            xytext=(40, 80),
            textcoords="offset points",
            ha='center', va='bottom',
            arrowprops=dict(
                arrowstyle='->',
                connectionstyle='arc3,rad=.2',
                color='#ff7f0e',
                shrinkB=18,
                relpos=(0.5, 0.0)
            ),
            fontweight='bold', color='#ff7f0e', zorder=30,
            fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Identifier les valeurs extrêmes pour MVP50
min_idx_mvp50 = np.argmin(waci_values_mvp50)
max_idx_mvp50 = np.argmax(waci_values_mvp50)

# Ajouter une annotation pour la valeur minimale du MVP50
ax.annotate(f'MVP50 Min: {waci_values_mvp50[min_idx_mvp50]:.2f}',
            xy=(min_idx_mvp50 + bar_width/2, waci_values_mvp50[min_idx_mvp50] + 5),
            xytext=(40, 140),
            textcoords="offset points",
            ha='center', va='bottom',
            arrowprops=dict(
                arrowstyle='->',
                connectionstyle='arc3,rad=.2',
                color='#2ca02c',
                shrinkB=18,
                relpos=(0.5, 0.0)
            ),
            fontweight='bold', color='#2ca02c', zorder=30,
            fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Ajouter une annotation pour la valeur maximale du MVP50
ax.annotate(f'MVP50 Max: {waci_values_mvp50[max_idx_mvp50]:.2f}',
            xy=(max_idx_mvp50 + bar_width/2, waci_values_mvp50[max_idx_mvp50] + 8),
            xytext=(-30, 180),
            textcoords="offset points",
            ha='center', va='bottom',
            arrowprops=dict(
                arrowstyle='->',
                connectionstyle='arc3,rad=-.2',
                color='#2ca02c',
                shrinkB=18,
                relpos=(0.5, 0.0)
            ),
            fontweight='bold', color='#2ca02c', zorder=30,
            fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Personnaliser les axes du graphique supérieur
ax.set_xticks(x)
ax.set_xticklabels([])  # Masquer les étiquettes x sur le graphique du haut
ax.set_title(r'Weighted Average Carbon Intensity Comparison (2014-2024) - $\mathbf{WACI_Y^{(P_{oos}^{(mv)})}}$ vs $\mathbf{WACI_Y^{(P_{oos}^{(mv)}(0.5))}}$', pad=20)
ax.set_ylabel('WACI (tCO$_2$-eq/MUSD)', labelpad=10)

# Formater l'axe Y pour afficher les nombres avec des séparateurs de milliers
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))

# Améliorer l'apparence générale du graphique supérieur
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(False)
ax.set_axisbelow(True)

# Ajouter de l'espace au-dessus du graphique pour les annotations
ax.set_ylim(0, max(max(waci_values_mvp), max(waci_values_mvp50)) * 1.3)

# Graphique secondaire (Nombre d'entreprises)
ax2 = fig.add_subplot(gs[1])

# Ajouter une ligne de démarcation en haut du graphique inférieur
ax2.axhline(0, color='black', linewidth=0.5, zorder=5)

# Obtenir le nombre d'entreprises pour chaque année
company_counts = [len(annual_filtered_data_[year]['valid_companies']) for year in years]

# Créer un dégradé de couleur pour les barres d'entreprises basé sur le nombre
company_norm = plt.Normalize(min(company_counts) if company_counts else 0, 
                            max(company_counts) if company_counts else 1)
company_colors = pastel_viridis(company_norm(company_counts))

# Tracer les barres d'entreprises
bars2 = ax2.bar(x, company_counts, color=company_colors, alpha=0.8, width=0.7, 
               edgecolor='black', linewidth=0.5, zorder=10)

# Ajouter les annotations de valeur
for i, bar in enumerate(bars2):
    height = bar.get_height()
    ax2.annotate(f'{height}',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=9, fontweight='bold')

# Ajouter une annotation centrée pour remplacer le titre
fig.text(0.19, 0.24, 'Companies Included in Calculation', 
         ha='center', va='center', fontsize=8, fontweight='bold')

ax2.set_xlabel('Year', labelpad=10)
ax2.set_ylabel('Count', labelpad=10)
ax2.set_xticks(x)

# Utiliser les années originales pour les étiquettes
ax2.set_xticklabels(years, rotation=0)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(False)
ax2.set_axisbelow(True)

# Légende pour le graphique principal
ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=10)

# Ajustement de la mise en page
plt.subplots_adjust(top=0.95, bottom=0.1, right=0.95)

# plt.tight_layout()
# plt.savefig('WACI_MVP_vs_MVP50.png', dpi=50)
plt.show()


# In[249]:


# Couleurs pastel inspirées de viridis pour le graphique secondaire
colors_companies = [
    (0.98, 0.98, 0.98),  # Blanc légèrement gris
    (0.85, 0.86, 0.88),  # Gris très légèrement bleuté
    (0.75, 0.78, 0.85),  # Gris bleu clair
    (0.65, 0.7, 0.8)     # Gris bleu argenté
]
pastel_viridis = LinearSegmentedColormap.from_list('pastel_viridis', colors_companies)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['text.usetex'] = False  # Use LaTeX for text rendering

# Créer la figure avec une résolution plus élevée
fig = plt.figure(figsize=(22, 8), dpi=50)
# Réduire l'espace entre les graphes avec hspace=0.05
gs = GridSpec(2, 1, height_ratios=[5, 1], hspace=0.05)

# Graphique principal (WACI)
ax = fig.add_subplot(gs[0])

# Préparation des données
years = list(waci_vw.keys())
waci_values_vw = list(waci_vw.values())
waci_values_te = [waci_te[year] for year in years]
waci_values_nz = [waci_nz[year] for year in years]
x = np.arange(len(years))

# Calculer les moyennes pour les trois séries
avg_waci_vw = np.mean(waci_values_vw)
avg_waci_te = np.mean(waci_values_te)
avg_waci_nz = np.mean(waci_values_nz)

# Largeur des barres pour le groupe
bar_width = 0.25

# Tracer les barres pour VW
bars_vw = ax.bar(x - bar_width, waci_values_vw, width=bar_width, 
                color='#1f77b4', edgecolor='black', linewidth=0.5,
                alpha=0.85, zorder=10, label=r'$P^{(vw)}$ WACI')

# Tracer les barres pour TE
bars_te = ax.bar(x, waci_values_te, width=bar_width,
                color='#d62728', edgecolor='black', linewidth=0.5,
                alpha=0.85, zorder=10, label=r'$P_{oos}^{(vw)}(0.5)$ WACI')

# Tracer les barres pour NZ
bars_nz = ax.bar(x + bar_width, waci_values_nz, width=bar_width,
                color='#9467bd', edgecolor='black', linewidth=0.5,
                alpha=0.85, zorder=10, label=r'$P_{oos}^{(vw)}(NZ)$ WACI')

# Ajouter les lignes de moyenne
ax.axhline(avg_waci_vw, color='#1f77b4', linestyle='-', linewidth=1.2, 
           label=f'VW Average: {avg_waci_vw:.2f} tCO$_2$-eq/MUSD', zorder=15, alpha=0.7)
ax.axhline(avg_waci_te, color='#d62728', linestyle='-', linewidth=1.2, 
           label=f'TE Average: {avg_waci_te:.2f} tCO$_2$-eq/MUSD', zorder=15, alpha=0.7)
ax.axhline(avg_waci_nz, color='#9467bd', linestyle='-', linewidth=1.2, 
           label=f'NZ Average: {avg_waci_nz:.2f} tCO$_2$-eq/MUSD', zorder=15, alpha=0.7)

# Ajouter les annotations de valeur sur chaque barre
for i, bar in enumerate(bars_vw):
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=8, fontweight='bold', color='#1f77b4', zorder=30)

for i, bar in enumerate(bars_te):
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=7, fontweight='bold', color='#d62728', zorder=30)

for i, bar in enumerate(bars_nz):
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=8, fontweight='bold', color='#9467bd', zorder=30)

# Identifier les valeurs extrêmes pour chaque portefeuille
# Pour VW
min_idx_vw = np.argmin(waci_values_vw)
max_idx_vw = np.argmax(waci_values_vw)

# Pour TE
min_idx_te = np.argmin(waci_values_te)
max_idx_te = np.argmax(waci_values_te)

# Pour NZ
min_idx_nz = np.argmin(waci_values_nz)
max_idx_nz = np.argmax(waci_values_nz)

# Ajouter une annotation pour la valeur minimale de VW
ax.annotate(f'VW Min: {waci_values_vw[min_idx_vw]:.2f}',
            xy=(min_idx_vw - bar_width, waci_values_vw[min_idx_vw] + 5),
            xytext=(60, 105),
            textcoords="offset points",
            ha='center', va='bottom',
            arrowprops=dict(
                arrowstyle='->',
                connectionstyle='arc3,rad=.2',
                color='#1f77b4',
                shrinkB=12,
                relpos=(0.5, 0.0)
            ),
            fontweight='bold', color='#1f77b4', zorder=30,
            fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Ajouter une annotation pour la valeur maximale de VW
ax.annotate(f'VW Max: {waci_values_vw[max_idx_vw]:.2f}',
            xy=(max_idx_vw - bar_width, waci_values_vw[max_idx_vw] + 8),
            xytext=(-60, 80),
            textcoords="offset points",
            ha='center', va='bottom',
            arrowprops=dict(
                arrowstyle='->',
                connectionstyle='arc3,rad=-.2',
                color='#1f77b4',
                shrinkB=15,
                relpos=(0.5, 0.0)
            ),
            fontweight='bold', color='#1f77b4', zorder=30,
            fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Ajouter des annotations pour les valeurs min et max de TE (tracking error)
ax.annotate(f'TE Min: {waci_values_te[min_idx_te]:.2f}',
            xy=(min_idx_te, waci_values_te[min_idx_te] + 5),
            xytext=(0, 190),  # Position au-dessus de la barre
            textcoords="offset points",
            ha='center', va='bottom',
            arrowprops=dict(
                arrowstyle='->',
                connectionstyle='arc3,rad=0',
                color='#d62728',
                shrinkB=12,
                relpos=(0.5, 0.0)
            ),
            fontweight='bold', color='#d62728', zorder=30,
            fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

ax.annotate(f'TE Max: {waci_values_te[max_idx_te]:.2f}',
            xy=(max_idx_te, waci_values_te[max_idx_te] + 8),
            xytext=(0, 130),  # Position au-dessus de la barre
            textcoords="offset points",
            ha='center', va='bottom',
            arrowprops=dict(
                arrowstyle='->',
                connectionstyle='arc3,rad=0',
                color='#d62728',
                shrinkB=15,
                relpos=(0.5, 0.0)
            ),
            fontweight='bold', color='#d62728', zorder=30,
            fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Ajout des annotations pour NZ
ax.annotate(f'NZ Min: {waci_values_nz[min_idx_nz]:.2f}',
            xy=(min_idx_nz + bar_width, waci_values_nz[min_idx_nz] + 5),
            xytext=(-60, 205),
            textcoords="offset points",
            ha='center', va='bottom',
            arrowprops=dict(
                arrowstyle='->',
                connectionstyle='arc3,rad=-.2',
                color='#9467bd',
                shrinkB=12,
                relpos=(0.5, 0.0)
            ),
            fontweight='bold', color='#9467bd', zorder=30,
            fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

ax.annotate(f'NZ Max: {waci_values_nz[max_idx_nz]:.2f}',
            xy=(max_idx_nz + bar_width, waci_values_nz[max_idx_nz] + 8),
            xytext=(-60, 60),
            textcoords="offset points",
            ha='center', va='bottom',
            arrowprops=dict(
                arrowstyle='->',
                connectionstyle='arc3,rad=-.2',
                color='#9467bd',
                shrinkB=12,
                relpos=(0.5, 0.0)
            ),
            fontweight='bold', color='#9467bd', zorder=30,
            fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Personnaliser les axes du graphique supérieur
ax.set_xticks(x)
ax.set_xticklabels([])  # Masquer les étiquettes x sur le graphique du haut
ax.set_title(r'Weighted Average Carbon Intensity Comparison (2014-2024) - $\mathbf{WACI_Y^{(P^{(vw)})}}$, $\mathbf{WACI_Y^{(P_{oos}^{(vw)}(0.5))}}$ and $\mathbf{WACI_Y^{(P_{oos}^{(vw)}(NZ))}}$', pad=20)
ax.set_ylabel('WACI (tCO$_2$-eq/MUSD)', labelpad=10)

# Formater l'axe Y pour afficher les nombres avec des séparateurs de milliers
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))

# Améliorer l'apparence générale du graphique supérieur
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(False)
ax.set_axisbelow(True)

# Ajouter de l'espace au-dessus du graphique pour les annotations
ax.set_ylim(0, max(max(waci_values_vw), max(waci_values_te), max(waci_values_nz)) * 1.3)

# Graphique secondaire (Nombre d'entreprises)
ax2 = fig.add_subplot(gs[1])

# Ajouter une ligne de démarcation en haut du graphique inférieur
ax2.axhline(0, color='black', linewidth=0.5, zorder=5)

# Obtenir le nombre d'entreprises pour chaque année
company_counts = company_counts

# Créer un dégradé de couleur pour les barres d'entreprises basé sur le nombre
company_norm = plt.Normalize(min(company_counts) if company_counts else 0, 
                            max(company_counts) if company_counts else 1)
company_colors = pastel_viridis(company_norm(company_counts))

# Tracer les barres d'entreprises
bars2 = ax2.bar(x, company_counts, color=company_colors, alpha=0.8, width=0.7, 
               edgecolor='black', linewidth=0.5, zorder=10)

# Ajouter les annotations de valeur
for i, bar in enumerate(bars2):
    height = bar.get_height()
    ax2.annotate(f'{height}',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=9, fontweight='bold')

# Ajouter une annotation centrée pour remplacer le titre
fig.text(0.19, 0.24, 'Companies Included in Calculation', 
         ha='center', va='center', fontsize=8, fontweight='bold')

ax2.set_xlabel('Year', labelpad=10)
ax2.set_ylabel('Count', labelpad=10)
ax2.set_xticks(x)

# Utiliser les années originales pour les étiquettes
ax2.set_xticklabels(years, rotation=0)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(False)
ax2.set_axisbelow(True)

# Légende pour le graphique principal
ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=10)

# Ajustement de la mise en page
plt.subplots_adjust(top=0.95, bottom=0.1, right=0.95)

# plt.tight_layout()

plt.show()


# In[250]:


import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_extended_metrics(returns_series, risk_free_series=None, benchmark_returns=None, annualization_factor=12):
    """
    Calcule des métriques étendues pour une série de rendements.
    
    Parameters
    ----------
    returns_series : pd.Series
        Série des rendements mensuels du portefeuille
    risk_free_series : pd.Series, optional
        Série du taux sans risque mensuel
    benchmark_returns : pd.Series, optional
        Série des rendements mensuels du benchmark (pour Treynor)
    annualization_factor : int, optional
        Facteur d'annualisation (12 pour des données mensuelles)
        
    Returns
    -------
    dict
        Dictionnaire des métriques calculées
    """
    # Convertir en série pandas et nettoyer les valeurs NaN
    returns_series = pd.Series(returns_series).dropna()
    
    if len(returns_series) == 0:
        return {
            "Monthly Average Return": np.nan,
            "Annualized Return": np.nan,
            "Monthly Volatility": np.nan,
            "Annualized Volatility": np.nan,
            "Monthly Sharpe Ratio": np.nan,
            "Annualized Sharpe Ratio": np.nan,
            "Treynor Ratio": np.nan,
            "Skewness": np.nan,
            "Kurtosis": np.nan,
            "Maximum Drawdown": np.nan
        }
    
    # Rendement moyen mensuel et annualisé
    monthly_return = returns_series.mean()
    annualized_return = (1 + monthly_return) ** annualization_factor - 1
    
    # Volatilité mensuelle et annualisée
    monthly_vol = returns_series.std()
    annualized_vol = monthly_vol * np.sqrt(annualization_factor)
    
    # Calcul du maximum drawdown
    cum_returns = (1 + returns_series).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max) - 1
    max_drawdown = drawdown.min()
    
    # Calcul des ratios de Sharpe
    monthly_sharpe = np.nan
    annualized_sharpe = np.nan
    excess_returns = None
    matched_rf = None
    
    if risk_free_series is not None:
        # Aligner le taux sans risque sur les dates des rendements
        matched_rf = risk_free_series.reindex(returns_series.index, method='ffill')
        if matched_rf.max() > 0.25:  # Si > 25%, probablement en pourcentage
            matched_rf = matched_rf / 100
        
        excess_returns = returns_series - matched_rf
        if excess_returns.std() > 0:
            monthly_sharpe = excess_returns.mean() / excess_returns.std()
            annualized_sharpe = monthly_sharpe * np.sqrt(annualization_factor)
    else:
        # Sans taux sans risque, on utilise simplement les rendements bruts
        if monthly_vol > 0:
            monthly_sharpe = monthly_return / monthly_vol
            annualized_sharpe = annualized_return / annualized_vol
    
    # Calcul du ratio de Treynor en utilisant pandas pour la covariance
    treynor_ratio = np.nan
    if benchmark_returns is not None and risk_free_series is not None and excess_returns is not None:
        try:
            # Aligner le benchmark sur les dates des rendements
            matched_benchmark = benchmark_returns.reindex(returns_series.index, method='ffill')
            excess_benchmark = matched_benchmark - matched_rf
            
            # Créer un DataFrame temporaire pour le calcul de la covariance
            temp_df = pd.DataFrame({
                'excess_returns': excess_returns,
                'excess_benchmark': excess_benchmark
            })
            
            # Utiliser pandas.DataFrame.cov() qui est plus robuste avec les types de données
            cov_matrix = temp_df.cov()
            
            # Calculer beta si possible
            if cov_matrix.loc['excess_benchmark', 'excess_benchmark'] > 0:
                beta = cov_matrix.loc['excess_returns', 'excess_benchmark'] / cov_matrix.loc['excess_benchmark', 'excess_benchmark']
                
                # Treynor = excess return / beta
                if not pd.isna(beta) and beta != 0:
                    annual_rf = matched_rf.mean() * annualization_factor
                    treynor_ratio = (annualized_return - annual_rf) / beta
        except Exception as e:
            print(f"Erreur dans le calcul du ratio Treynor: {e}")
            treynor_ratio = np.nan
    
    # Skewness et kurtosis
    skewness = stats.skew(returns_series)
    kurtosis = stats.kurtosis(returns_series)
    
    return {
        "Monthly Average Return": monthly_return,
        "Annualized Return": annualized_return,
        "Monthly Volatility": monthly_vol,
        "Annualized Volatility": annualized_vol,
        "Monthly Sharpe Ratio": monthly_sharpe,
        "Annualized Sharpe Ratio": annualized_sharpe,
        "Treynor Ratio": treynor_ratio,
        "Skewness": skewness,
        "Kurtosis": kurtosis,
        "Maximum Drawdown": max_drawdown
    }
def create_metrics_table(portfolios_dict, risk_free_series=None, benchmark_name=None):
    """
    Crée un tableau de métriques pour plusieurs portefeuilles.
    
    Parameters
    ----------
    portfolios_dict : dict
        Dictionnaire avec noms de portefeuilles comme clés et séries de rendements comme valeurs
    risk_free_series : pd.Series, optional
        Série du taux sans risque mensuel
    benchmark_name : str, optional
        Nom du portefeuille de benchmark (dans portfolios_dict) pour le calcul du ratio Treynor
        
    Returns
    -------
    pd.DataFrame
        Tableau de métriques pour tous les portefeuilles
    """
    metrics = {}
    benchmark_returns = portfolios_dict.get(benchmark_name) if benchmark_name else None
    
    for name, returns in portfolios_dict.items():
        metrics[name] = calculate_extended_metrics(
            returns, 
            risk_free_series=risk_free_series,
            benchmark_returns=benchmark_returns if name != benchmark_name else None
        )
    
    metrics_df = pd.DataFrame(metrics)
    
    # Renommer les lignes pour plus de clarté
    rename_dict = {
        "Monthly Average Return": "Rendement mensuel moyen",
        "Annualized Return": "Rendement annualisé",
        "Monthly Volatility": "Volatilité mensuelle",
        "Annualized Volatility": "Volatilité annualisée",
        "Monthly Sharpe Ratio": "Ratio de Sharpe mensuel",
        "Annualized Sharpe Ratio": "Ratio de Sharpe annualisé", 
        "Treynor Ratio": "Ratio de Treynor",
        "Skewness": "Asymétrie (Skewness)",
        "Kurtosis": "Aplatissement (Kurtosis)",
        "Maximum Drawdown": "Drawdown maximum"
    }
    
    metrics_df = metrics_df.rename(index=rename_dict)
    
    return metrics_df

def create_correlation_matrix(portfolios_dict):
    """
    Crée une matrice de corrélation entre plusieurs portefeuilles.
    
    Parameters
    ----------
    portfolios_dict : dict
        Dictionnaire avec noms de portefeuilles comme clés et séries de rendements comme valeurs
        
    Returns
    -------
    pd.DataFrame
        Matrice de corrélation
    """
    # Créer un DataFrame avec tous les rendements
    returns_df = pd.DataFrame({name: returns for name, returns in portfolios_dict.items()})
    
    # Calculer la matrice de corrélation
    corr_matrix = returns_df.corr()
    
    return corr_matrix

def format_for_latex(df, precision=3, percentage_cols=None):
    """
    Formate un DataFrame pour l'exportation LaTeX.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à formater
    precision : int, optional
        Précision décimale pour les valeurs numériques
    percentage_cols : list, optional
        Liste des colonnes à formater en pourcentage
        
    Returns
    -------
    pd.DataFrame
        DataFrame formaté pour LaTeX
    """
    formatted_df = df.copy()
    
    # Définir les colonnes en pourcentage si non spécifiées
    if percentage_cols is None:
        percentage_cols = [
            "Rendement mensuel moyen", "Rendement annualisé", 
            "Volatilité mensuelle", "Volatilité annualisée",
            "Maximum Drawdown"
        ]
    
    # Formater les nombres
    for col in formatted_df.columns:
        # Correction ici : utiliser percentage_cols au lieu de formatted_cols
        if col in percentage_cols:  
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.{precision}%}" if not pd.isna(x) else "-")
        else:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.{precision}f}" if not pd.isna(x) else "-")
    
    return formatted_df

def export_to_latex(df, filepath, caption=None, label=None, index=True, format_percentages=True):
    """
    Exporte un DataFrame en format LaTeX.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à exporter
    filepath : str
        Chemin du fichier de sortie
    caption : str, optional
        Titre du tableau LaTeX
    label : str, optional
        Label pour références croisées LaTeX
    index : bool, optional
        Si True, inclut l'index dans le tableau LaTeX
    format_percentages : bool, optional
        Si True, formate certaines métriques comme des pourcentages
        
    Returns
    -------
    None
    """
    # Copie du DataFrame pour éviter de modifier l'original
    df_to_export = df.copy()
    
    # Formater les pourcentages si demandé
    if format_percentages:
        percentage_cols = [
            "Rendement mensuel moyen", "Rendement annualisé", 
            "Volatilité mensuelle", "Volatilité annualisée",
            "Maximum Drawdown"
        ]
        
        # Convertir en pourcentages les colonnes appropriées
        for col in percentage_cols:
            if col in df_to_export.index:
                df_to_export.loc[col] = df_to_export.loc[col] * 100
            elif col in df_to_export.columns:
                df_to_export[col] = df_to_export[col] * 100
    
    # Options LaTeX
    latex_options = {
        "longtable": True,
        "index": index,
        "escape": False,
        "caption": caption if caption else None,
        "label": label if label else None,
        "position": "htbp"
    }
    
    # Exporter en LaTeX
    latex_code = df_to_export.to_latex(**latex_options)
    
    # Écrire dans un fichier
    with open(filepath, 'w') as f:
        f.write(latex_code)
    
    print(f"Tableau LaTeX exporté avec succès vers {filepath}")

# Fonction pour générer tous les tableaux demandés
def generate_all_comparison_tables(mvp_returns, vw_returns, mvpc_returns, te_returns, nz_returns, rf_series):
    """
    Génère tous les tableaux de comparaison demandés.
    
    Parameters
    ----------
    mvp_returns : pd.Series
        Rendements du portefeuille MVP
    vw_returns : pd.Series
        Rendements du portefeuille VW
    mvpc_returns : pd.Series
        Rendements du portefeuille MVP05 (MVP avec contrainte carbone)
    te_returns : pd.Series
        Rendements du portefeuille TE
    nz_returns : pd.Series
        Rendements du portefeuille Net Zero
    rf_series : pd.Series
        Série du taux sans risque
        
    Returns
    -------
    dict
        Dictionnaire contenant tous les tableaux générés
    """
    tables = {}
    
    # 1. Tableau MVP seul
    tables["mvp_only"] = create_metrics_table(
        {"MVP": mvp_returns},
        risk_free_series=rf_series
    )
    
    # 2. Comparaison MVP et VW avec matrice de corrélation
    mvp_vw_dict = {
        "MVP": mvp_returns,
        "VW": vw_returns
    }
    tables["mvp_vs_vw_metrics"] = create_metrics_table(
        mvp_vw_dict,
        risk_free_series=rf_series,
        benchmark_name="VW"
    )
    tables["mvp_vs_vw_corr"] = create_correlation_matrix(mvp_vw_dict)
    
    # 3. Comparaison MVP et MVP05 avec matrice de corrélation
    mvp_mvpc_dict = {
        "MVP": mvp_returns,
        "MVP05": mvpc_returns
    }
    tables["mvp_vs_mvpc_metrics"] = create_metrics_table(
        mvp_mvpc_dict,
        risk_free_series=rf_series,
        benchmark_name="MVP"
    )
    tables["mvp_vs_mvpc_corr"] = create_correlation_matrix(mvp_mvpc_dict)
    
    # 4. Comparaison TE et VW
    te_vw_dict = {
        "TE": te_returns,
        "VW": vw_returns
    }
    tables["te_vs_vw_metrics"] = create_metrics_table(
        te_vw_dict,
        risk_free_series=rf_series,
        benchmark_name="VW"
    )
    tables["te_vs_vw_corr"] = create_correlation_matrix(te_vw_dict)
    
    # 5. Comparaison VW, NZ et TE
    vw_nz_te_dict = {
        "VW": vw_returns,
        "NZ": nz_returns,
        "TE": te_returns
    }
    tables["vw_nz_te_metrics"] = create_metrics_table(
        vw_nz_te_dict,
        risk_free_series=rf_series,
        benchmark_name="VW"
    )
    tables["vw_nz_te_corr"] = create_correlation_matrix(vw_nz_te_dict)
    
    # 6. Tableau final avec tous les portefeuilles
    all_portfolios = {
        "MVP": mvp_returns,
        "MVP05": mvpc_returns,
        "VW": vw_returns,
        "TE": te_returns,
        "NZ": nz_returns
    }
    tables["all_portfolios_metrics"] = create_metrics_table(
        all_portfolios,
        risk_free_series=rf_series,
        benchmark_name="VW"
    )
    tables["all_portfolios_corr"] = create_correlation_matrix(all_portfolios)
    
    return tables

# Fonction pour exporter tous les tableaux en LaTeX
def export_all_tables_to_latex(tables, output_dir="latex_tables"):
    """
    Exporte tous les tableaux générés en format LaTeX.
    
    Parameters
    ----------
    tables : dict
        Dictionnaire contenant tous les tableaux générés
    output_dir : str, optional
        Répertoire de sortie pour les fichiers LaTeX
        
    Returns
    -------
    None
    """
    import os
    
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionnaire des titres et labels pour chaque tableau
    captions_and_labels = {
        "mvp_only": ("Métriques du portefeuille à variance minimale (MVP)", "tab:mvp_only"),
        "mvp_vs_vw_metrics": ("Comparaison des métriques entre MVP et VW", "tab:mvp_vs_vw_metrics"),
        "mvp_vs_vw_corr": ("Matrice de corrélation entre MVP et VW", "tab:mvp_vs_vw_corr"),
        "mvp_vs_mvpc_metrics": ("Comparaison des métriques entre MVP et MVP05", "tab:mvp_vs_mvpc_metrics"),
        "mvp_vs_mvpc_corr": ("Matrice de corrélation entre MVP et MVP05", "tab:mvp_vs_mvpc_corr"),
        "te_vs_vw_metrics": ("Comparaison des métriques entre TE et VW", "tab:te_vs_vw_metrics"),
        "te_vs_vw_corr": ("Matrice de corrélation entre TE et VW", "tab:te_vs_vw_corr"),
        "vw_nz_te_metrics": ("Comparaison des métriques entre VW, NZ et TE", "tab:vw_nz_te_metrics"),
        "vw_nz_te_corr": ("Matrice de corrélation entre VW, NZ et TE", "tab:vw_nz_te_corr"),
        "all_portfolios_metrics": ("Comparaison des métriques pour tous les portefeuilles", "tab:all_portfolios_metrics"),
        "all_portfolios_corr": ("Matrice de corrélation entre tous les portefeuilles", "tab:all_portfolios_corr")
    }
    
    # Exporter chaque tableau
    for name, table in tables.items():
        caption, label = captions_and_labels.get(name, (None, None))
        
        # Déterminer si c'est une matrice de corrélation
        is_corr = "corr" in name
        
        export_to_latex(
            table, 
            filepath=f"{output_dir}/{name}.tex",
            caption=caption,
            label=label,
            format_percentages=not is_corr  # Ne pas formater les corrélations en pourcentages
        )


# In[251]:


# Préparation des séries de rendements (à partir de votre code)
vw_series = pd.to_numeric(vw_portfolio_returns, errors='coerce').dropna()
mvp_series = pd.to_numeric(portfolio_returns_all['Portfolio_Return'], errors='coerce').dropna()

# Check if mvpc exists before trying to access it
if 'mvpc' in locals() and isinstance(mvpc, pd.DataFrame) and 'Portfolio_Return' in mvpc.columns:
    mvpc_series = pd.to_numeric(mvpc['Portfolio_Return'], errors='coerce').dropna()
else:
    print("mvpc DataFrame or 'Portfolio_Return' column not found. Creating an empty Series.")
    mvpc_series = pd.Series(dtype='float64')  # Create an empty series with float64 dtype

if 'te_portfolio_returns' in locals() and isinstance(te_portfolio_returns, pd.DataFrame) and 'Portfolio_Return' in te_portfolio_returns.columns:
    te_series = pd.to_numeric(te_portfolio_returns['Portfolio_Return'], errors='coerce').dropna()
else:
    print("te_portfolio_returns DataFrame or 'Portfolio_Return' column not found. Creating an empty Series.")
    te_series = pd.Series(dtype='float64')

if 'nz_portfolio_returns' in locals() and isinstance(nz_portfolio_returns, pd.DataFrame) and 'Portfolio_Return' in nz_portfolio_returns.columns:
    nz_series = pd.to_numeric(nz_portfolio_returns['Portfolio_Return'], errors='coerce').dropna()
else:
    print("nz_portfolio_returns DataFrame or 'Portfolio_Return' column not found. Creating an empty Series.")
    nz_series = pd.Series(dtype='float64')

# Taux sans risque
rf_series = pd.to_numeric(rf[rf.columns[0]], errors='coerce').dropna()



# Exemple: Créer un tableau comparatif personnalisé
custom_portfolios = {
    "MVP": mvp_series,
    "VW": vw_series,
    "MVP05": mvpc_series
}

custom_metrics = create_metrics_table(
    custom_portfolios,
    risk_free_series=rf_series,
    benchmark_name="VW"
)

# # Exporter ce tableau personnalisé
# export_to_latex(
#     custom_metrics,
#     filepath="tables_latex/custom_comparison.tex",
#     caption="Comparaison personnalisée des portefeuilles",
#     label="tab:custom_comparison"
# )
# print(custom_metrics)

# Générer tous les tableaux


# In[252]:


# Exemple d'utilisation :
# Supposons que vous avez déjà créé vos séries de rendements et le taux sans risque
tables = generate_all_comparison_tables(
    mvp_returns=mvp_series, 
    vw_returns=vw_series, 
    mvpc_returns=mvpc_series, 
    te_returns=te_series, 
    nz_returns=nz_series, 
    rf_series=rf_series
)


latex_names = {
    'MVP': r'$P_{oos}^{(mv)}$',
    'VW': r'$P^{(vw)}$',
    'MVP05': r'$P_{oos}^{(mv)}(0.5)$',
    'TE': r'$P_{oos}^{(vw)}(0.5)$',
    'NZ': r'$P_{oos}^{(vw)}(NZ)$'
}

# Application des renommages
tables['all_portfolios_corr'] = tables['all_portfolios_corr'].rename(
    index=latex_names, 
    columns=latex_names
)
print('All Portfoios correlation')
print(tables['all_portfolios_corr'] )
# tables['all_portfolios_corr'].to_latex('Article_Latex/CORRELATION_ALL.tex', index=True, float_format="%.3f")


# In[253]:


def create_carbon_metrics_comparison(carbon_footprint, constrained_cf_values, te_cf_values, nz_cf_values, vw_cf_values, 
                                    nz_constraints=None, format_for_display=True):
    """
    Crée un tableau comparatif des métriques carbone pour tous les portefeuilles.
    
    Parameters
    ----------
    carbon_footprint : dict
        Empreinte carbone du portefeuille MVP par année
    constrained_cf_values : dict
        Empreinte carbone du portefeuille MVP05 (avec contrainte 50%) par année
    te_cf_values : dict
        Empreinte carbone du portefeuille TE (avec contrainte 50%) par année
    nz_cf_values : dict
        Empreinte carbone du portefeuille Net Zero par année
    vw_cf_values : dict
        Empreinte carbone du portefeuille Value-Weighted par année
    nz_constraints : dict, optional
        Contraintes du portefeuille Net Zero par année
    format_for_display : bool, default=True
        Si True, formate les valeurs pour l'affichage (pourcentages, précision)
        
    Returns
    -------
    pd.DataFrame
        Tableau comparatif des métriques carbone
    """
    # Collecter toutes les années disponibles
    all_years = set()
    for cf_dict in [carbon_footprint, constrained_cf_values, te_cf_values, nz_cf_values, vw_cf_values]:
        if cf_dict:
            all_years.update(cf_dict.keys())
    
    # Trier les années
    years = sorted(all_years)
    
    # Créer un DataFrame pour stocker les métriques carbone
    metrics_data = []
    
    for year in years:
        # Récupérer les valeurs pour chaque portefeuille
        vw_value = vw_cf_values.get(year, None)
        mvp_value = carbon_footprint.get(year, None)
        mvp05_value = constrained_cf_values.get(year, None)
        te_value = te_cf_values.get(year, None)
        nz_value = nz_cf_values.get(year, None)
        
        # Contraintes (calculées ou extraites)
        mvp05_constraint = mvp_value * 0.5 if mvp_value is not None else None
        te_constraint = vw_value * 0.5 if vw_value is not None else None
        nz_constraint = nz_constraints.get(year, None) if nz_constraints else None
        
        # Calculer les réductions par rapport au VW (benchmark)
        if vw_value is not None:
            mvp_reduction = (1 - mvp_value / vw_value) if mvp_value is not None else None
            mvp05_reduction = (1 - mvp05_value / vw_value) if mvp05_value is not None else None
            te_reduction = (1 - te_value / vw_value) if te_value is not None else None
            nz_reduction = (1 - nz_value / vw_value) if nz_value is not None else None
        else:
            mvp_reduction = mvp05_reduction = te_reduction = nz_reduction = None
        
        # Vérifier si les contraintes ont été respectées
        mvp05_compliant =  0.00001 >= (mvp05_constraint-mvp05_value) if (mvp05_value is not None and mvp05_constraint is not None) else None
        te_compliant =  0.00001 >= (te_value-te_constraint) if (te_value is not None and te_constraint is not None) else None
        nz_compliant =  0.00001 >= (nz_value - nz_constraint) if (nz_value is not None and nz_constraint is not None) else None
        
        # Ajouter les données à la liste
        row = {
            'Year': year,
            'VW Footprint': vw_value,
            'MVP Footprint': mvp_value,
            'MVP05 Footprint': mvp05_value,
            'TE Footprint': te_value,
            'NZ Footprint': nz_value,
            'MVP05 Constraint': mvp05_constraint,
            'TE Constraint': te_constraint,
            'NZ Constraint': nz_constraint,
            'MVP Reduction vs VW': mvp_reduction,
            'MVP05 Reduction vs VW': mvp05_reduction,
            'TE Reduction vs VW': te_reduction,
            'NZ Reduction vs VW': nz_reduction,
            'MVP05 Compliant': mvp05_compliant,
            'TE Compliant': te_compliant,
            'NZ Compliant': nz_compliant
        }
        metrics_data.append(row)
    
    # Créer le DataFrame
    carbon_metrics_df = pd.DataFrame(metrics_data)
    
    # Formater les valeurs pour l'affichage si demandé
    if format_for_display:
        for col in carbon_metrics_df.columns:
            if 'Footprint' in col or 'Constraint' in col:
                carbon_metrics_df[col] = carbon_metrics_df[col].apply(
                    lambda x: f"{x:.2f}" if x is not None else "-")
            elif 'Reduction' in col:
                carbon_metrics_df[col] = carbon_metrics_df[col].apply(
                    lambda x: f"{x:.2%}" if x is not None else "-")
            elif 'Compliant' in col:
                carbon_metrics_df[col] = carbon_metrics_df[col].apply(
                    lambda x: "✓" if x == True else "✗" if x == False else "-")
    
    return carbon_metrics_df

# Créer le tableau des métriques carbone
carbon_metrics_table = create_carbon_metrics_comparison(
    carbon_footprint=carbon_footprint_mvp,          # MVP standard
    constrained_cf_values=constrained_cf_values, # MVP avec contrainte 50%
    te_cf_values=te_cf_values,                  # TE avec contrainte 50%
    nz_cf_values=nz_cf,                  # Net Zero
    vw_cf_values=carbon_footprint_vw,                  # Value-Weighted
    nz_constraints=nz_constraints                # Contraintes Net Zero
)


# Afficher le tableau
# print(carbon_metrics_table)

# Sélectionner seulement les colonnes pertinentes (empreintes et contraintes)
footprints_table = carbon_metrics_table[['Year', 
                                        'VW Footprint', 
                                        'MVP Footprint', 
                                        'MVP05 Constraint', 
                                        'MVP05 Footprint',
                                        'TE Constraint',
                                        'TE Footprint', 
                                        'NZ Footprint', 
                                        'NZ Constraint']]

# Renommer les colonnes en format LaTeX
latex_columns = {
    'Year': 'Year',
    'VW Footprint': r'CF$^{(P^{(vw)})}_{Y}$',
    'MVP Footprint': r'CF$^{(P_{oos}^{(mv)})}_{Y}$',
    'MVP05 Constraint': r'CF$^{(P_{oos}^{(mv)}(0.5))}_{Y, \text{Constraint}}$ ',
    'MVP05 Footprint': r'CF$^{(P_{oos}^{(mv)}(0.5))}_{Y}$',
    'TE Constraint': r'CF$^{(P_{oos}^{(vw)}(0.5))}_{Y, \text{Constraint}}$ ',
    'TE Footprint': r'CF$^{(P_{oos}^{(vw)}(0.5))}_{Y}$',
    'NZ Constraint': r'CF$^{(P_{oos}^{(vw)}(NZ))}_{Y, \text{Constraint}}$ ',
    'NZ Footprint': r'CF$^{(P_{oos}^{(vw)}(NZ))}_{Y}$'
}

# Appliquer le renommage
footprints_table_latex = footprints_table.rename(columns=latex_columns)

# Pour exporter en format LaTeX
# footprints_table_latex.to_latex('Article_Latex/carbon_footprints_constraints.tex', index=False, float_format='%.2f')

# Afficher le tableau
print(footprints_table_latex)

# carbon_metrics_table.to_latex('Article_Latex/carbon_metrics_all.tex', index=True)


# In[254]:


def create_consolidated_metrics_table(portfolios_dict, carbon_footprint_dict, waci_dict, rf_series=None,
                                     latex_path=None, portfolio_labels=None):
    """
    Crée une table synthétique consolidant µ, σ, SR, Max DD, CF et WACI des portefeuilles.
    
    Parameters
    ----------
    portfolios_dict : dict
        Dictionnaire avec les noms des portefeuilles (clés) et leurs séries de rendements (valeurs)
    carbon_footprint_dict : dict
        Dictionnaire avec les noms des portefeuilles (clés) et leurs valeurs CF par année (valeurs)
    waci_dict : dict
        Dictionnaire avec les noms des portefeuilles (clés) et leurs valeurs WACI par année (valeurs)
    rf_series : pd.Series, optional
        Série du taux sans risque pour calculer le ratio de Sharpe
    latex_path : str, optional
        Chemin pour sauvegarder la table en LaTeX
    portfolio_labels : dict, optional
        Dictionnaire pour renommer les portefeuilles dans la table finale
    
    Returns
    -------
    pd.DataFrame
        Table consolidée des métriques
    """
    # Variables pour stocker les résultats
    metrics_data = []
    
    # Pour chaque portefeuille
    for name, returns in portfolios_dict.items():
        # Calculer les métriques de performance
        metrics = calculate_portfolio_metrics(returns, rf_series)
        
        # Calculer les moyennes d'empreinte carbone
        cf_values = carbon_footprint_dict.get(name, {})
        waci_values = waci_dict.get(name, {})
        
        avg_cf = np.mean(list(cf_values.values())) if cf_values else np.nan
        avg_waci = np.mean(list(waci_values.values())) if waci_values else np.nan
        
        # Compiler les métriques clés
        row = {
            'Portfolio': name,
            'μ (%)': metrics['Annualized Average Return'] * 100,  # Rendement annualisé en %
            'σ (%)': metrics['Annualized Volatility'] * 100,      # Volatilité annualisée en %
            'Sharpe': metrics['Annualized Sharpe Ratio'],         # Ratio de Sharpe annualisé
            'Max DD (%)': metrics['Maximum Drawdown'] * 100,      # Drawdown maximum en %
            'CF': avg_cf,                                         # Empreinte carbone moyenne
            'WACI': avg_waci                                      # WACI moyen
        }
        metrics_data.append(row)
    
    # Créer le DataFrame
    metrics_table = pd.DataFrame(metrics_data)
    
    # Renommer les portefeuilles si nécessaire
    if portfolio_labels:
        metrics_table['Portfolio'] = metrics_table['Portfolio'].map(
            lambda x: portfolio_labels.get(x, x)
        )
        
    # Définir Portfolio comme index
    metrics_table = metrics_table.set_index('Portfolio')
    
    # Trier dans un ordre spécifique si souhaité
    if portfolio_labels:
        ordered_indices = [label for name, label in portfolio_labels.items() 
                          if label in metrics_table.index]
        metrics_table = metrics_table.reindex(ordered_indices)
    
    # Exporter en LaTeX
    if latex_path:
        metrics_table.to_latex(latex_path, float_format='%.2f', escape=False)
    
    return metrics_table


# In[255]:


# 1. Dictionnaires des séries de rendements
portfolios = {
    'VW': vw_series,
    'MVP': mvp_series,
    'MVP50': mvpc_series,
    'TE': te_series,
    'NZ': nz_series
}

# 2. Dictionnaires des empreintes carbone
carbon_footprints = {
    'VW': carbon_footprint_vw,
    'MVP': carbon_footprint_mvp,
    'MVP50': constrained_cf_values,
    'TE': te_cf_values,
    'NZ': nz_cf
}

# 3. Dictionnaires des WACI
waci_metrics = {
    'VW': waci_vw,
    'MVP': waci_mvp,
    'MVP50': waci_mvp50,
    'TE': waci_te,
    'NZ': waci_nz
}

# 4. Noms des portefeuilles pour l'affichage LaTeX
portfolio_labels = {
    'VW': r'$P^{(vw)}$',
    'MVP': r'$P_{oos}^{(mv)}$',
    'MVP50': r'$P_{oos}^{(mv)}(0.5)$',
    'TE': r'$P_{oos}^{(vw)}(0.5)$',
    'NZ': r'$P_{oos}^{(vw)}(NZ)$'
}

# 5. Générer la table
consolidated_table = create_consolidated_metrics_table(
    portfolios_dict=portfolios,
    carbon_footprint_dict=carbon_footprints,
    waci_dict=waci_metrics,
    rf_series=rf[rf_column],
    latex_path=None,
    portfolio_labels=portfolio_labels
)

# Afficher la table
print(consolidated_table)


# In[256]:


print('\n\n\n\n\n\n')
print('# ------------------------------- Graphs for the Slide presentation --------------------------------- #')


# In[257]:


def plot_carbon_metrics_comparison(carbon_metrics_df, figsize=(16, 10), save_path=None):
    """
    Crée une visualisation comparative des empreintes carbone et des contraintes.
    
    Parameters
    ----------
    carbon_metrics_df : pd.DataFrame
        DataFrame des métriques carbone (tel que créé par create_carbon_metrics_comparison)
    figsize : tuple, optional
        Taille de la figure (largeur, hauteur)
    save_path : str, optional
        Chemin pour sauvegarder la figure
        
    Returns
    -------
    fig, axes
        Figure et axes matplotlib pour personnalisation supplémentaire
    """
    plt.rcParams['figure.dpi'] = 50
    # Prétraitement: convertir les valeurs formatées en nombres pour le graphique
    df = carbon_metrics_df.copy()
    
    # Déterminer les colonnes numériques à traiter
    numeric_cols = [col for col in df.columns if 'Footprint' in col or 'Constraint' in col]
    
    # Convertir les colonnes de chaînes formatées en valeurs numériques
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col].str.replace('-', 'nan'), errors='coerce')
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Définir les couleurs pour chaque portefeuille et contrainte
    colors = {
        'VW Footprint': '#1f77b4',      # Bleu
        'MVP Footprint': '#ff7f0e',     # Orange
        'MVP05 Footprint': '#2ca02c',   # Vert
        'TE Footprint': '#d62728',      # Rouge
        'NZ Footprint': '#9467bd',      # Violet
        'MVP05 Constraint': '#2ca02c',  # Vert (même que MVP05)
        'TE Constraint': '#d62728',     # Rouge (même que TE)
        'NZ Constraint': '#9467bd'      # Violet (même que NZ)
    }
    
    # Barres groupées pour les empreintes carbone
    x = np.arange(len(df))
    width = 0.15  # largeur des barres
    
    # Tracer les barres pour les empreintes carbones réelles (zorder=5)
    portfolios = ['MVP', 'MVP05', 'VW', 'TE', 'NZ']
    offsets = [-2, -1, 0, 1, 2]  # Décalages pour les barres groupées
    
    for i, portfolio in enumerate(portfolios):
        footprint_col = f'{portfolio} Footprint'
        if footprint_col in df.columns:
            bars = ax.bar(x + width*offsets[i], df[footprint_col], width, 
                        label=f'{portfolio} Footprint', color=colors[footprint_col], 
                        alpha=0.8, zorder=5)
            
            # Ajouter les valeurs sur les barres
            for bar_idx, bar in enumerate(bars):
                height = bar.get_height()
                if not np.isnan(height):
                    ax.annotate(f'{height:.0f}',
                              xy=(bar.get_x() + bar.get_width()/2, height),
                              xytext=(0, 3),  # 3 points de décalage vertical
                              textcoords="offset points",
                              ha='center', va='bottom', fontsize=8, zorder=10)
    
    # Ajouter les lignes pour les contraintes avec zorder élevé (10) pour qu'elles apparaissent au-dessus des barres
    constraints = ['MVP05', 'TE', 'NZ']
    line_styles = ['--', '-.', ':']
    
    for i, portfolio in enumerate(constraints):
        constraint_col = f'{portfolio} Constraint'
        if constraint_col in df.columns:
            # Déterminer l'offset x correspondant au portfolio
            portfolio_idx = portfolios.index(portfolio) if portfolio in portfolios else -1
            offset = width*offsets[portfolio_idx] if portfolio_idx >= 0 else 0
            
            # Utiliser des lignes plus visibles pour les contraintes
            ax.plot(x + offset, df[constraint_col], line_styles[i % len(line_styles)], 
                   label=f'{portfolio} Constraint', color=colors[constraint_col], 
                   linewidth=3, marker='o', markersize=6, alpha=0.9, zorder=20)
    
    # Configurer les axes et les étiquettes
    ax.set_xlabel('Years', fontsize=12)
    ax.set_ylabel(r'Empreinte Carbone (tCO$_2$e/M\$)', fontsize=12)
    ax.set_title('Comparison Carbon Footprint and Constraints', fontsize=14, fontweight='bold')
    
    # Définir les étiquettes de l'axe x comme les années
    ax.set_xticks(x)
    year_labels = [str(year) for year in df.index] if df.index.dtype == 'object' else [str(year) for year in range(2014, 2014 + len(df))]
    ax.set_xticklabels(year_labels, rotation=0)
    
    # Ajouter une légende
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Ajouter un quadrillage pour faciliter la lecture
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.grid(axis='x', linestyle='--', alpha=0)
    
    # plt.tight_layout()
    
    # Enregistrer la figure si un chemin est fourni
    if save_path:
        plt.savefig(save_path, dpi=50, bbox_inches='tight')
        print(f"Figure enregistrée à {save_path}")
    
    return fig, ax

# Visualiser les métriques carbone
fig, ax = plot_carbon_metrics_comparison(carbon_metrics_table, 
                                       figsize=(22, 8),
                                       save_path="carbon_metrics_comparison.png" if save_images else None
                                       )


# In[258]:


def display_all_portfolios_waci(
    waci_dict_mvp, 
    waci_dict_vw, 
    waci_dict_mvp50=None, 
    waci_dict_te=None, 
    waci_dict_nz=None, 
    figure_size=(14, 8), 
    show_plot=True, 
    save_path=None, 
    show_table=True,
    round_decimals=2
):
    """
    Affiche et compare les WACI de tous les portefeuilles disponibles.
    
    Parameters:
    -----------
    waci_dict_mvp : dict
        Dictionnaire avec les WACI du portefeuille MVP par année
    waci_dict_vw : dict
        Dictionnaire avec les WACI du portefeuille Value-Weighted par année
    waci_dict_mvp50 : dict, optional
        Dictionnaire avec les WACI du portefeuille MVP avec 50% réduction carbone
    waci_dict_te : dict, optional
        Dictionnaire avec les WACI du portefeuille TE avec 50% réduction carbone
    waci_dict_nz : dict, optional
        Dictionnaire avec les WACI du portefeuille Net Zero
    figure_size : tuple, optional
        Taille de la figure (largeur, hauteur)
    show_plot : bool, optional
        Afficher le graphique
    save_path : str, optional
        Chemin pour sauvegarder le graphique
    show_table : bool, optional
        Afficher le tableau des valeurs
    round_decimals : int, optional
        Nombre de décimales pour l'arrondi dans le tableau
        
    Returns:
    --------
    pd.DataFrame
        DataFrame contenant les WACI de tous les portefeuilles
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Collecter toutes les années disponibles
    plt.rcParams['figure.dpi'] = 50
    all_years = set()
    for waci_dict in [waci_dict_mvp, waci_dict_vw, waci_dict_mvp50, waci_dict_te, waci_dict_nz]:
        if waci_dict:
            all_years.update(waci_dict.keys())
    
    # Trier les années
    years = sorted(all_years)
    
    # Créer le DataFrame pour stocker toutes les données
    waci_data = pd.DataFrame(index=years)
    
    # Ajouter les données de chaque portefeuille dans l'ordre souhaité
    # L'ordre sera: MVP, MVP50, VW, TE, NZ
    
    # Définir la liste des portefeuilles disponibles dans l'ordre souhaité
    portfolios = []
    
    # MVP toujours présent
    waci_data['MVP'] = pd.Series(waci_dict_mvp)
    portfolios.append('MVP')
    
    # MVP50 s'il est fourni
    if waci_dict_mvp50:
        waci_data['MVP50'] = pd.Series(waci_dict_mvp50)
        portfolios.append('MVP50')
    
    # VW toujours présent
    waci_data['VW'] = pd.Series(waci_dict_vw)
    portfolios.append('VW')
    
    # TE s'il est fourni
    if waci_dict_te:
        waci_data['TE'] = pd.Series(waci_dict_te)
        portfolios.append('TE')
    
    # NZ s'il est fourni
    if waci_dict_nz:
        waci_data['NZ'] = pd.Series(waci_dict_nz)
        portfolios.append('NZ')
    
    # Calculer les réductions par rapport aux portefeuilles de référence
    for p in ['MVP50', 'TE', 'NZ']:
        if p in waci_data.columns:
            ref = 'MVP' if p == 'MVP50' else 'VW'
            waci_data[f'{p} vs {ref} (%)'] = ((waci_data[p] / waci_data[ref]) - 1) * 100
    
    # Afficher le tableau si demandé
    if show_table:
        display_df = waci_data.copy()
        print(f"\n=== WACI par portefeuille (tCO2e/M$) ===")
        print(display_df.round(round_decimals))
        
        # Afficher les statistiques moyennes
        print(f"\n=== Moyennes des WACI et réductions ===")
        means = display_df.mean().round(round_decimals)
        print(means)
    
    # Créer le graphique si demandé
    if show_plot:
        # Créer la figure avec des axes explicites
        fig, ax = plt.subplots(figsize=figure_size)
        
        # Configuration du style
        sns.set_style('white')  # Utiliser un style sans grille
        
        # Définir les couleurs pour chaque portefeuille
        colors = {
            'MVP': '#ff7f0e',    # Orange
            'MVP50': '#2ca02c',  # Vert
            'VW': '#1f77b4',     # Bleu
            'TE': '#d62728',     # Rouge
            'NZ': '#9467bd'      # Violet
        }
        
        # Créer le graphique à barres groupées
        width = 0.15  # largeur des barres
        positions = np.arange(len(years))
        
        # Calculer les offsets pour les barres groupées
        offsets = np.linspace(-0.3, 0.3, len(portfolios))
        
        # Dictionnaire pour stocker les barres pour les annotations
        bars_by_portfolio = {}
        
        for i, portfolio in enumerate(portfolios):
            portfolio_positions = positions + offsets[i]
            
            # Tracer les barres
            bars = ax.bar(
                portfolio_positions,
                waci_data[portfolio],
                width=width,
                label=f"{portfolio}",
                color=colors.get(portfolio, None),
                alpha=0.8,
                edgecolor='black',
                linewidth=0.5,
                zorder=5
            )
            
            bars_by_portfolio[portfolio] = bars
            
            # Ajouter les valeurs WACI au-dessus de chaque barre
            for j, bar in enumerate(bars):
                height = bar.get_height()
                if not np.isnan(height):  # Vérifier si la valeur n'est pas NaN
                    ax.annotate(f'{height:.1f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0.5, 3),  # Décalage de 3 points au-dessus
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=8, fontweight='bold',
                                color="black",
                                zorder=10)
        
        # Ajouter un quadrillage pour faciliter la lecture
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.grid(axis='x', linestyle='--', alpha=0)
        
        # Ajouter les éléments du graphique
        ax.set_xlabel('Année', fontsize=12)
        ax.set_ylabel(r'WACI (tCO$_2$e/M\$)', fontsize=12)
        ax.set_title('WACI Comparison by Portfolio', fontsize=14, fontweight='bold')
        ax.set_xticks(positions)
        ax.set_xticklabels(years, rotation=0)
        
        # Ajouter des annotations moyennes avec des lignes
        for portfolio in portfolios:
            avg_value = waci_data[portfolio].mean()
            ax.axhline(y=avg_value, color=colors.get(portfolio, 'gray'), 
                      linestyle='--', alpha=0.5, 
                      label=f"Aver. {portfolio}: {avg_value:.2f}",
                      zorder=1)
        
        # Ajouter une légende
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        # plt.tight_layout()
        
        # Sauvegarder si un chemin est fourni
        if save_path:
            plt.savefig(save_path, dpi=50, bbox_inches='tight')
            print(f"Graphique sauvegardé dans: {save_path}")
        
        plt.show()
    
    return waci_data


# In[259]:


waci_mvp = dict(get_carbon_metrics(mvp_metrics_c, 'WACI'))
waci_vw = dict(get_carbon_metrics(vw_metrics_c, 'WACI'))

# Calculer les métriques WACI pour MVP50 (portefeuille avec contrainte carbone 50%)
mvp50_metrics_c = carbon_metrics_all_years(annual_filtered_data_, constrained_weights)
waci_values_mvp50 = dict(get_carbon_metrics(mvp50_metrics_c, 'WACI'))

# Calculer les métriques WACI pour TE (portefeuille tracking error)
te_metrics_c = carbon_metrics_all_years(annual_filtered_data_, te_weights)
waci_values_te = dict(get_carbon_metrics(te_metrics_c, 'WACI'))

# Calculer les métriques WACI pour NZ (portefeuille net zero)
nz_metrics_c = carbon_metrics_all_years(annual_filtered_data_, nz_weights)
waci_values_nz = dict(get_carbon_metrics(nz_metrics_c, 'WACI'))

# Utiliser la fonction display_all_portfolios_waci avec toutes les variables
waci_comparison = display_all_portfolios_waci(
    waci_dict_mvp=waci_mvp,      # WACI du MVP
    waci_dict_vw=waci_vw,        # WACI du Value-Weighted
    waci_dict_mvp50=waci_values_mvp50,   # WACI du MVP50
    waci_dict_te=waci_values_te,         # WACI du TE
    waci_dict_nz=waci_values_nz,         # WACI du Net Zero
    figure_size=(22, 8),                 # Taille adaptée 
    save_path="waci_all_portfolios.png"  if save_images else None # Sauvegarde du graphique
)


# In[260]:


waci_comparison.mean()


# In[261]:


def create_constraint_compliance_summary(carbon_metrics_df):
    """
    Crée un résumé du respect des contraintes carbone pour chaque portefeuille.
    
    Parameters
    ----------
    carbon_metrics_df : pd.DataFrame
        DataFrame des métriques carbone (tel que créé par create_carbon_metrics_comparison)
        
    Returns
    -------
    pd.DataFrame
        Tableau récapitulatif du respect des contraintes
    """
    # Convertir les colonnes de conformité en valeurs booléennes
    df = carbon_metrics_df.copy()
    
    # Colonnes de conformité
    compliance_cols = ['MVP05 Compliant', 'TE Compliant', 'NZ Compliant']
    
    # Convertir les symboles en valeurs booléennes
    for col in compliance_cols:
        if col in df.columns:
            df[col] = df[col].map(lambda x: True if x == "✓" else False if x == "✗" else None)
    
    # Créer un résumé par portefeuille
    summary_data = []
    
    portfolios = ['MVP05', 'TE', 'NZ']
    for portfolio in portfolios:
        compliance_col = f'{portfolio} Compliant'
        if compliance_col in df.columns:
            compliant_count = df[compliance_col].sum()
            total_count = df[compliance_col].count()  # Exclut les None
            compliance_rate = compliant_count / total_count if total_count > 0 else None
            
            # Années non conformes
            non_compliant_years = df.loc[df[compliance_col] == False, 'Year'].tolist()
            
            summary_data.append({
                'Portfolio': portfolio,
                'Years Compliant': compliant_count,
                'Total Years': total_count,
                'Compliance Rate': compliance_rate,
                'Non-Compliant Years': ', '.join(non_compliant_years) if non_compliant_years else "All compliant"
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Formater la colonne du taux de conformité
    if 'Compliance Rate' in summary_df.columns:
        summary_df['Compliance Rate'] = summary_df['Compliance Rate'].apply(
            lambda x: f"{x:.2%}" if x is not None else "-")
    
    return summary_df

# Créer le résumé de conformité aux contraintes
constraint_summary = create_constraint_compliance_summary(carbon_metrics_table)
print(constraint_summary)


# In[262]:


def create_comprehensive_metrics_table(
    portfolios_returns_dict, 
    carbon_footprints_dict,
    waci_dict,
    rf_series,
    benchmark_mapping=None
):
    """
    Crée un tableau complet de métriques pour tous les portefeuilles fournis.
    
    Parameters:
    -----------
    portfolios_returns_dict : dict
        Dictionnaire des séries de rendements pour chaque portefeuille
        Format: {'MVP': mvp_series, 'VW': vw_series, ...}
    
    carbon_footprints_dict : dict
        Dictionnaire des empreintes carbone pour chaque portefeuille
        Format: {'MVP': carbon_footprint_mvp, 'VW': carbon_footprint_vw, ...}
    
    waci_dict : dict
        Dictionnaire des WACI pour chaque portefeuille
        Format: {'MVP': waci_mvp, 'VW': waci_vw, ...}
    
    rf_series : pd.Series
        Série du taux sans risque
    
    benchmark_mapping : dict, optional
        Dictionnaire indiquant le benchmark pour chaque portefeuille
        Format: {'MVP50': 'MVP', 'TE': 'VW', 'NZ': 'VW'}
        Par défaut: {'MVP': None, 'MVP50': 'MVP', 'VW': None, 'TE': 'VW', 'NZ': 'VW'}
    
    Returns:
    --------
    pd.DataFrame
        Tableau des métriques pour tous les portefeuilles
    """
    import pandas as pd
    import numpy as np
    
    # Définir le mapping par défaut des benchmarks si non fourni
    if benchmark_mapping is None:
        benchmark_mapping = {
            'MVP': None,  # Pas de benchmark pour MVP
            'MVP50': 'MVP',  # MVP est le benchmark de MVP50
            'VW': None,  # Pas de benchmark pour VW
            'TE': 'VW',  # VW est le benchmark de TE
            'NZ': 'VW'   # VW est le benchmark de NZ
        }
    
    # Dictionnaire pour stocker les métriques de chaque portefeuille
    all_metrics = {}
    
    # Pour chaque portefeuille, calculer toutes les métriques
    for port_name, returns in portfolios_returns_dict.items():
        # 1. Récupérer les métriques financières de base
        metrics = calculate_portfolio_metrics(returns, rf_series)
        
        # 2. Ajouter les métriques carbone
        if port_name in carbon_footprints_dict:
            # Calculer l'empreinte carbone moyenne
            carbon_values = list(carbon_footprints_dict[port_name].values())
            if carbon_values:
                cf_mean = np.mean(carbon_values)
                
                # Calculer la réduction par rapport au benchmark si applicable
                benchmark_name = benchmark_mapping.get(port_name)
                if benchmark_name and benchmark_name in carbon_footprints_dict:
                    benchmark_cf_values = list(carbon_footprints_dict[benchmark_name].values())
                    if benchmark_cf_values:
                        benchmark_cf_mean = np.mean(benchmark_cf_values)
                        cf_reduction = (1 - cf_mean / benchmark_cf_mean) * 100  # En pourcentage
                    else:
                        cf_reduction = None
                else:
                    cf_reduction = None
            else:
                cf_mean = None
                cf_reduction = None
        else:
            cf_mean = None
            cf_reduction = None
        
        # 3. Ajouter la WACI moyenne
        if port_name in waci_dict:
            waci_values = list(waci_dict[port_name].values())
            waci_mean = np.mean(waci_values) if waci_values else None
        else:
            waci_mean = None
        
        # 4. Calculer le tracking error si un benchmark est défini
        benchmark_name = benchmark_mapping.get(port_name)
        if benchmark_name and benchmark_name in portfolios_returns_dict:
            benchmark_returns = portfolios_returns_dict[benchmark_name]
            te_annual = calculate_tracking_error(returns, benchmark_returns)
        else:
            te_annual = None
        
        # 5. Stocker les métriques dans le dictionnaire
        all_metrics[port_name] = {
            "Rendement Annualisé (%)": metrics["Annualized Average Return"] * 100,
            "Volatilité Annualisée (%)": metrics["Annualized Volatility"] * 100,
            "Ratio de Sharpe Annualisé": metrics["Annualized Sharpe Ratio"],
            "Maximum Drawdown (%)": metrics["Maximum Drawdown"] * 100,
            "Réduction Empreinte Carbone (%)": cf_reduction,
            "WACI Moyenne (tCO2e/M$)": waci_mean,
            "TE Annualisé (%)": te_annual * 100 if te_annual is not None else None
        }
    
    # Convertir le dictionnaire en DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    
    # Transposer pour avoir les métriques en lignes et les portefeuilles en colonnes
    metrics_df = metrics_df.T
    
    return metrics_df

def create_selected_metrics_table(comprehensive_table, selected_portfolios, custom_column_names=None):
    """
    Crée un tableau de métriques pour une sélection de portefeuilles.
    
    Parameters:
    -----------
    comprehensive_table : pd.DataFrame
        Tableau complet des métriques créé par create_comprehensive_metrics_table
    
    selected_portfolios : list
        Liste des noms de portefeuilles à inclure
    
    custom_column_names : dict, optional
        Dictionnaire pour renommer les colonnes du tableau résultant
    
    Returns:
    --------
    pd.DataFrame
        Tableau des métriques pour les portefeuilles sélectionnés
    """
    # Filtrer le tableau complet pour ne garder que les portefeuilles sélectionnés
    filtered_table = comprehensive_table.loc[selected_portfolios]
    
    # Renommer les colonnes si demandé
    if custom_column_names:
        filtered_table = filtered_table.rename(columns=custom_column_names)
    
    return filtered_table


# In[263]:


# Dictionnaire des rendements des portefeuilles
portfolios_returns = {
    'MVP': mvp_series,
    'MVP50': mvpc_series, 
    'VW': vw_series,
    'TE': te_series,
    'NZ': nz_series
}

# Dictionnaire des empreintes carbone
carbon_footprints = {
    'MVP': carbon_footprint_mvp,
    'MVP50': constrained_cf_values,
    'VW': carbon_footprint_vw,
    'TE': te_cf_values,
    'NZ': nz_cf
}

# Dictionnaire des WACI
waci_values = {
    'MVP': waci_mvp,
    'MVP50': waci_mvp50,
    'VW': waci_vw,
    'TE': waci_te,
    'NZ': waci_nz
}

# Créer le tableau complet
complete_metrics = create_comprehensive_metrics_table(
    portfolios_returns, 
    carbon_footprints,
    waci_values,
    rf[rf_column]
)

# Afficher le tableau complet
print("Tableau complet des métriques:")
print(complete_metrics.round(2))

# Créer un tableau pour une sélection de portefeuilles
selected = ['MVP', 'VW', 'MVP50', 'TE', 'NZ']
custom_names = {
    'Rendement Annualisé (%)': 'Rendement (%)',
    'Volatilité Annualisée (%)': 'Volatilité (%)',
    'Maximum Drawdown (%)': 'Max Drawdown (%)'
}

selected_metrics = create_selected_metrics_table(
    complete_metrics, 
    selected,
    custom_names
)

print("\nTableau des métriques pour les portefeuilles sélectionnés:")
print(selected_metrics.round(2))


if save_all_metrics_in_current_directory_csv:
    selected_metrics.T.to_csv(pickle_weights_window + '.csv')



# In[264]:


# selected_metrics.T.loc['TE Annualisé (%)'].to_latex('Article_Latex/Tracking_error_annualy_slides.tex', index=True)


# In[265]:


# Importer les bibliothèques nécessaires
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap

# Corriger la fonction plot_tracking_errors_comparison
def plot_tracking_errors_comparison(
    mvp_returns, mvpc_returns, 
    vw_returns, te_returns, nz_returns,
    figsize=(22, 8), save_path=None):
    """
    Crée une visualisation comparant les tracking errors de trois portefeuilles avec leurs benchmarks respectifs.
    """
    from matplotlib.gridspec import GridSpec
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import matplotlib.dates as mdates
    from matplotlib.colors import LinearSegmentedColormap
    
    # Palettes de couleurs personnalisées pour chaque graphique
    # MVP - Tons oranges/verts
    mvp_c = [
        (0.90, 0.90, 0.90),
        (0.85, 0.90, 0.80),
        (0.75, 0.85, 0.70),
        (0.65, 0.80, 0.60)
    ]
    mvp_palette = LinearSegmentedColormap.from_list('mvp_palette', mvp_c)
    
    # TE - Tons rouges
    te_c = [
        (0.90, 0.90, 0.90),
        (0.95, 0.85, 0.85),
        (0.90, 0.75, 0.75),
        (0.85, 0.65, 0.65)
    ]
    te_palette = LinearSegmentedColormap.from_list('te_palette', te_c)
    
    # NZ - Tons violets
    nz_c = [
        (0.90, 0.90, 0.90),
        (0.90, 0.85, 0.90),
        (0.85, 0.75, 0.85),
        (0.80, 0.65, 0.80)
    ]
    nz_palette = LinearSegmentedColormap.from_list('nz_palette', nz_c)
    
    # Définir les couleurs pour chaque portefeuille
    colors = {
        'mvp': '#ff7f0e',    # Orange pour MVP
        'mvpc': '#2ca02c',   # Vert pour MVP(0.5)
        'vw': '#1f77b4',     # Bleu pour VW
        'te': '#d62728',     # Rouge pour TE
        'nz': '#9467bd',     # Violet pour NZ
    }
    
    # Convertir toutes les séries en Series si ce sont des DataFrames
    series_dict = {
        'mvp': mvp_returns if isinstance(mvp_returns, pd.Series) else mvp_returns['Portfolio_Return'],
        'mvpc': mvpc_returns if isinstance(mvpc_returns, pd.Series) else mvpc_returns['Portfolio_Return'],
        'vw': vw_returns if isinstance(vw_returns, pd.Series) else vw_returns['Portfolio_Return'],
        'te': te_returns if isinstance(te_returns, pd.Series) else te_returns['Portfolio_Return'],
        'nz': nz_returns if isinstance(nz_returns, pd.Series) else nz_returns['Portfolio_Return']
    }
    
    # S'assurer que tous les index sont en datetime
    for name, series in series_dict.items():
        series_dict[name] = pd.Series(series, index=pd.to_datetime(series.index))
    
    # Style professionnel
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Créer la figure avec GridSpec pour 3 sous-graphiques
    fig = plt.figure(figsize=figsize, dpi=50, facecolor='white')
    gs = GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.2)
    
    # ====================== Fonction pour créer un graphique de TE ====================== #
    def create_te_plot(ax, returns_series, benchmark_series, palette, color, title):
        # Aligner les séries de rendement par index
        common_dates = returns_series.index.intersection(benchmark_series.index)
        tracking_error = (returns_series.loc[common_dates] - benchmark_series.loc[common_dates])
        
        # Calculer l'erreur de suivi annualisée
        annual_te = tracking_error.std() * np.sqrt(12) * 100  # Annualisée et en pourcentage
        
        # Grouper l'erreur de suivi par année
        years = sorted(set(date.year for date in common_dates))
        x_positions = np.arange(len(years))
        yearly_te = {}
        
        for year in years:
            year_mask = [date.year == year for date in common_dates]
            yearly_te[str(year)] = tracking_error.iloc[year_mask].values
        
        # Calculer l'erreur de suivi absolue moyenne par année
        avg_abs_te_by_year = []
        for year_str in [str(y) for y in years]:
            values = yearly_te[year_str]
            if len(values) > 0:
                avg_abs_te_by_year.append(np.mean(np.abs(values)) * 100)  # En pourcentage
            else:
                avg_abs_te_by_year.append(np.nan)
        
        # Utiliser la palette personnalisée pour les couleurs des barres
        abs_te_max = np.nanmax(avg_abs_te_by_year)
        if abs_te_max > 0:
            # Normaliser les valeurs entre 0 et 1
            color_idx = np.array(avg_abs_te_by_year) / abs_te_max  # CORRECTION ICI
            # Remplacer les NaN par 0 pour éviter les erreurs
            color_idx = np.nan_to_num(color_idx)
            # Utiliser la palette personnalisée pour chaque année
            bar_colors = [palette(idx) for idx in color_idx]
        else:
            # Utiliser une couleur par défaut
            bar_colors = [palette(0.5)] * len(avg_abs_te_by_year)
        
        # Tracer les barres d'erreur de suivi moyenne absolue
        bars = ax.bar(x_positions, avg_abs_te_by_year, color=bar_colors, 
                     alpha=0.7, width=0.7, edgecolor='black', linewidth=0.5)
        
        # Ajouter la ligne de tracking error annualisé
        ax.axhline(annual_te, linestyle='--', color=color, linewidth=1.5, 
                  label=f'Annualized TE: {annual_te:.2f}%')
        
        # Ajouter les annotations sur les barres
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:.2f}%', 
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontsize=9)
        
        # Ajouter le titre avec annotation explicative
        ax.text(0.02, 0.80, title, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
                facecolor='white', alpha=0.9))
        
        # Formatage
        ax.set_ylabel('Absolute TE (%)', fontsize=10, labelpad=10)
        ax.set_xticks(x_positions)
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
        ax.set_axisbelow(True)
        
        return years, x_positions
    
    # ====================== PREMIER GRAPHIQUE: MVP vs MVP50 ====================== #
    ax1 = fig.add_subplot(gs[0])
    years1, x1 = create_te_plot(
        ax1, 
        series_dict['mvpc'], 
        series_dict['mvp'], 
        mvp_palette, 
        colors['mvpc'], 
        r'$\mathbf{Benchmark~P_{oos}^{(mv)}}$ - $\mathbf{P_{oos}^{(mv)}(0.5)}$ Tracking Error'
    )
    # Cacher les étiquettes x du premier graphique
    ax1.set_xticklabels([])
    
    # ====================== DEUXIÈME GRAPHIQUE: VW vs TE ====================== #
    ax2 = fig.add_subplot(gs[1])
    years2, x2 = create_te_plot(
        ax2, 
        series_dict['te'], 
        series_dict['vw'], 
        te_palette, 
        colors['te'], 
        r'$\mathbf{Benchmark~P^{(vw)}}$ - $\mathbf{P_{oos}^{(vw)}(0.5)}$ Tracking Error'
    )
    # Cacher les étiquettes x du deuxième graphique
    ax2.set_xticklabels([])
    
    # ====================== TROISIÈME GRAPHIQUE: VW vs NZ ====================== #
    ax3 = fig.add_subplot(gs[2])
    years3, x3 = create_te_plot(
        ax3, 
        series_dict['nz'], 
        series_dict['vw'], 
        nz_palette, 
        colors['nz'], 
        r'$\mathbf{Benchmark~P^{(vw)}}$ - $\mathbf{P_{oos}^{(vw)}(NZ)}$ Tracking Error'
    )
    # N'afficher les étiquettes x que sur le dernier graphique
    ax3.set_xticklabels([str(y) for y in years3])
    ax3.set_xlabel('Year', fontsize=12, labelpad=10)
    
    # Titre principal
    fig.suptitle('Tracking Error Comparison Across Portfolio Strategies', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Ajuster la mise en page
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.92, hspace=0.1)
    
    # Sauvegarder si un chemin est fourni
    if save_path:
        plt.savefig(save_path, dpi=50, bbox_inches='tight')
        print(f"Figure sauvegardée dans {save_path}")
    
    return fig, (ax1, ax2, ax3)

# Utilisation de la fonction avec les variables disponibles
# Vous pouvez maintenant appeler la fonction avec vos séries de rendements

fig, axes = plot_tracking_errors_comparison(
    mvp_series,           # Rendements du portefeuille Minimum Variance
    mvpc_series,          # Rendements du portefeuille MVP avec contrainte carbone
    vw_series,            # Rendements du portefeuille Value-Weighted
    te_series,            # Rendements du portefeuille TE avec contrainte carbone
    nz_series,            # Rendements du portefeuille Net Zero
    figsize=(22, 8),
    save_path="tracking_errors_comparison.png" if save_images else None # Facultatif: chemin pour sauvegarder l'image
)

plt.show()  # Afficher le graphique


# In[266]:


# print('Some weight analysis (more focused and distinctives by applying RIDGE on ∑ matrix)')


# In[267]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
plt.rcParams['figure.dpi'] = 50

def plot_portfolio_weights_pca_improved(portfolio_weights_dict, figsize=(14, 12), save_path=None):
    """
    Visualise les poids de portefeuilles avec PCA de manière plus claire et explicite.
    """
    # Définition des paramètres visuels avec une palette plus contrastée
    portfolio_colors = {
        'MVP': '#FF7F00',     # Orange
        'MVP05': '#33A02C',  # Vert
        'TE': '#E31A1C',     # Rouge
        'VW': '#3366CC',    # Bleu
        'NZ': '#6A3D9A'      # Violet
    }
    
    # Années et marqueurs correspondants (utiliser uniquement des marqueurs remplissables)
    years = sorted({year for port_dict in portfolio_weights_dict.values() 
                   for year in port_dict.keys()})
    markers = ['o', 's', '^', 'P', 'D', '*', 'p', 'h', 'v', '>', '<']
    year_markers = {year: markers[i % len(markers)] for i, year in enumerate(years)}
    
    # Collecte de toutes les données pour la PCA
    all_weights = []
    labels = []
    colors = []
    marker_styles = []
    portfolio_types = []
    year_values = []
    
    for portfolio_type, yearly_weights in portfolio_weights_dict.items():
        for year, weights in yearly_weights.items():
            weights = pd.Series(weights).fillna(0)
            all_weights.append(weights)
            labels.append(f"{portfolio_type} {year}")
            colors.append(portfolio_colors.get(portfolio_type, '#AAAAAA'))
            marker_styles.append(year_markers.get(year, 'o'))
            portfolio_types.append(portfolio_type)
            year_values.append(year)
    
    # Calcul PCA
    common_indices = set(all_weights[0].index)
    for weights in all_weights[1:]:
        common_indices = common_indices.intersection(set(weights.index))
    
    filtered_weights = [weights[list(common_indices)] for weights in all_weights]
    X = np.vstack([weights.values for weights in filtered_weights])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Stocker les coordonnées par type de portefeuille
    coordinates_by_portfolio = {}
    
    # Graphique avec style moderne
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=figsize, dpi=150, facecolor='white')
    
    # Tracer les points avec les couleurs et marqueurs correspondants
    for i, (label, color, marker, portfolio_type, year) in enumerate(zip(labels, colors, marker_styles, portfolio_types, year_values)):
        ax.scatter(X_pca[i, 0], X_pca[i, 1], 
                  c=color, s=100, marker=marker, alpha=0.8, 
                  edgecolor='black', linewidth=0.5)
        
        # Stocker les coordonnées pour tracer les trajectoires
        if portfolio_type not in coordinates_by_portfolio:
            coordinates_by_portfolio[portfolio_type] = {'years': [], 'coords': []}
        
        coordinates_by_portfolio[portfolio_type]['years'].append(year)
        coordinates_by_portfolio[portfolio_type]['coords'].append((X_pca[i, 0], X_pca[i, 1]))
    
    # Tracer les trajectoires temporelles pour chaque portefeuille
    for portfolio_type, data in coordinates_by_portfolio.items():
        # Trier par année pour s'assurer de l'ordre chronologique
        sorted_indices = sorted(range(len(data['years'])), key=lambda k: data['years'][k])
        
        trajectory_x = [data['coords'][i][0] for i in sorted_indices]
        trajectory_y = [data['coords'][i][1] for i in sorted_indices]
        
        # Utiliser une ligne pointillée pour les trajectoires
        ax.plot(trajectory_x, trajectory_y, color=portfolio_colors[portfolio_type], 
               linestyle='--', alpha=0.5, linewidth=1, zorder=1)
    
    # Tracer les ellipses englobantes pour chaque type de portefeuille
    for portfolio_type, data in coordinates_by_portfolio.items():
        if len(data['coords']) > 1:  
            coords = np.array(data['coords'])
            mean_x, mean_y = np.mean(coords[:, 0]), np.mean(coords[:, 1])
            
            # Calcul de l'ellipse
            cov = np.cov(coords[:, 0], coords[:, 1])
            eigvals, eigvecs = np.linalg.eig(cov)
            
            sort_indices = np.argsort(eigvals)[::-1]
            eigvals = eigvals[sort_indices]
            eigvecs = eigvecs[:, sort_indices]
            
            n_std = 1.5  # 1.5 écart-type ~ 85% des points
            width, height = 2 * n_std * np.sqrt(eigvals)
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            
            ellipse = Ellipse(xy=(mean_x, mean_y),
                             width=width, height=height,
                             angle=angle,
                             edgecolor=portfolio_colors[portfolio_type],
                             facecolor=portfolio_colors[portfolio_type],
                             alpha=0.1)
            ax.add_patch(ellipse)
            
            # Label au centre de chaque groupe
            ax.text(mean_x, mean_y, portfolio_type, 
                   ha='center', va='center', 
                   fontsize=12, fontweight='bold', 
                   color=portfolio_colors[portfolio_type])
    
    # Annotation des explications de variance
    explained_var_ratio = pca.explained_variance_ratio_
    ax.set_xlabel(f'Composante Principale 1 ({explained_var_ratio[0]*100:.1f}%)', fontsize=14)
    ax.set_ylabel(f'Composante Principale 2 ({explained_var_ratio[1]*100:.1f}%)', fontsize=14)
    
    # Légendes séparées pour portefeuilles et années
    portfolio_legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                               markerfacecolor=color, markersize=10, label=port)
                              for port, color in portfolio_colors.items() 
                              if port in portfolio_types]
    
    year_legend_elements = [plt.Line2D([0], [0], marker=marker, color='black', 
                          linestyle='', markersize=8, label=year)
                         for year, marker in year_markers.items() 
                         if year in year_values]
    
    # Légende stratégies
    legend1 = ax.legend(handles=portfolio_legend_elements, loc='upper left', 
                      title="Stratégies de Portefeuille", frameon=True, fontsize=10, 
                      bbox_to_anchor=(1.01, 1))
    ax.add_artist(legend1)
    
    # Légende années
    legend2 = ax.legend(handles=year_legend_elements, loc='upper left', 
                      title="Années", frameon=True, fontsize=9, 
                      bbox_to_anchor=(1.01, 0.7))
    
    # Ajouter les infos statistiques
    stats_text = (
        f"Analyse en Composantes Principales\n"
        f"Variance expliquée totale: {sum(explained_var_ratio)*100:.1f}%\n"
        f"CP1: {explained_var_ratio[0]*100:.1f}% - CP2: {explained_var_ratio[1]*100:.1f}%\n"
        f"Nombre d'actifs analysés: {len(common_indices)}\n"
        f"Période: {min(year_values)} à {max(year_values)}\n"
        f"Nombre de portefeuilles: {len(portfolio_types)}"
    )
    
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, 
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='#CCCCCC', boxstyle='round,pad=0.5'),
            va='bottom', fontsize=10)
    
    # Titre avec plus d'informations
    ax.set_title('Analyse des Stratégies de Portefeuille par ACP\nÉvolution et Positionnement dans l\'Espace des Composantes', 
                fontsize=16, fontweight='bold')
    
    # plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=50)
        print(f"Figure sauvegardée: {save_path}")
    
    return fig, ax


# In[268]:


# Préparer les données pour la visualisation
portfolio_weights_dict = {
    'MVP': optimal_weights,
    'TE': te_weights,
    'MVP05': constrained_weights,
    'NZ': nz_weights,
    'VW':vw_weights
}

# Créer la visualisation améliorée
fig, ax = plot_portfolio_weights_pca_improved(
    portfolio_weights_dict=portfolio_weights_dict,
    figsize=(22, 8),
    save_path='portfolio_weights_comparison_pca_improved.png' if save_images else None
)

