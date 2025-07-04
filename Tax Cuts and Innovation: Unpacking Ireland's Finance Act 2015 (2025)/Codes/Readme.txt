/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */
/*******************************************************************************
 * ┌─────────────────────────────────────────────────────────────────────────┐ *
 * │ **English Version:**                                                    │ *
 * └─────────────────────────────────────────────────────────────────────────┘ *
 ******************************************************************************/
/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */

Hello,  

Below is a guide designed to help you review all the executed code related to the article:

**Article:**  
1. All `.py` files automatically detect their working directory, so you do not need to adjust any paths in the code.  
2. The `.R` files require manual path adjustments to run properly.  
3. In the `initial_cleaning` folder, you will find `.py` files dedicated to cleaning the raw data and saving the processed results.  
4. The file `0_DataTransformation` is responsible for transforming and saving the data. The `.csv` saving commands are currently commented out.  
5. The remaining files, from `0_` to `6_`, correspond to various stages of the analysis and produce identical outputs to those presented in the article.  
6. The SCM package contains some error in plotting the right treatement year, so the code must be adapted to plot the right year. Comments were made in the concerned files. 
**Important:** The files `5_` and `6_` as well as the Monte Carlo simulation in `7_` rely on the `pysynthdid` package located in the `package` folder. 
This package can be found on GitHub: [https://github.com/MasaAsami/pysynthdid](https://github.com/MasaAsami/pysynthdid)  

**Monte Carlo:**  
1. The `7_` file contains two classes that will run upon execution. Depending on the number of simulations, the runtime may be long (approximately 1 hour and 15 minutes for 1000 iterations on an M2 Silicon processor). For convenience, we have set the code to run only 5 simulations to avoid overly lengthy executions.  
2. To review the Monte Carlo results, use the `_8` file, which accesses datasets stored in the `simulation_ds` folder.  
3. The simulation’s `.csv` files include specific seeds, enabling you to reproduce the results as documented in the classes defined in `7_`.

Thank you for your attention, and we wish you a smooth code execution experience.

/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */
/*******************************************************************************
 * ┌─────────────────────────────────────────────────────────────────────────┐ *
 * │ **VERSION FRANÇAISE :**                                                 │ *
 * └─────────────────────────────────────────────────────────────────────────┘ *
 ******************************************************************************/
/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */

Bonjour,

Voici un guide destiné à vous aider à consulter l’ensemble du code exécuté dans le cadre de l’article :

**Article :**  
1. Tous les fichiers `.py` détectent automatiquement leur répertoire de travail, vous n’avez donc pas besoin de modifier les chemins d’accès.  
2. Les fichiers `.R` nécessitent une adaptation manuelle des chemins pour fonctionner correctement.  
3. Dans le dossier `initial_cleaning`, vous trouverez des fichiers `.py` utilisés pour nettoyer les données brutes et enregistrer les résultats traités.  
4. Le fichier `0_DataTransformation` transforme et enregistre les données. Les commandes d’enregistrement en `.csv` y sont actuellement commentées.  
5. Les autres fichiers, de `0_` à `6_`, correspondent aux différentes étapes de l’analyse et produisent les mêmes résultats que ceux présentés dans l’article.
6. Le package SCM contient une erreur dans le tracé de l'année de traitement correcte, donc le code doit être adapté pour tracer l'année correcte. Des commentaires ont été faits dans les fichiers concernés.
**Important :** Les fichiers `5_`, `6_` ainsi que la simulation de Monte Carlo dans `7_` font appel au package `pysynthdid` présent dans le dossier `package`. 
Ce package est disponible sur GitHub : [https://github.com/MasaAsami/pysynthdid](https://github.com/MasaAsami/pysynthdid)

**Monte Carlo :**  
1. Le fichier `7_` contient deux classes qui seront exécutées lors du lancement. En fonction du nombre de simulations, l’exécution peut être longue (environ 1h15 pour 1000 itérations sur un processeur M2 Silicon). Pour limiter cela, le code est paramétré pour ne lancer que 5 simulations.  
2. Pour consulter les résultats de la Monte Carlo, utilisez le fichier `_8` qui accède aux jeux de données dans le dossier `simulation_ds`.  
3. Les fichiers `.csv` générés par la simulation contiennent des seeds permettant de reproduire les résultats conformément aux classes définies dans `7_`.

Merci de votre attention, et bonne exécution du code.

/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */
/*******************************************************************************
 * ┌─────────────────────────────────────────────────────────────────────────┐ *
 * │ **Versione Italiana:**                                                  │ *
 * └─────────────────────────────────────────────────────────────────────────┘ *
 ******************************************************************************/
/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */

Salve,

Di seguito trovate una guida per consultare tutto il codice eseguito nell’ambito dell’articolo:

**Articolo:**  
1. Tutti i file `.py` rilevano automaticamente la directory di lavoro, quindi non è necessario modificare i percorsi nel codice.  
2. I file `.R` richiedono un adattamento manuale del percorso per funzionare correttamente.  
3. Nella cartella `initial_cleaning` troverete i file `.py` utilizzati per pulire i dati grezzi e salvare i risultati elaborati.  
4. Il file `0_DataTransformation` trasforma e salva i dati; attualmente i comandi di salvataggio in `.csv` sono commentati.  
5. Gli altri file, da `0_` a `6_`, si riferiscono alle diverse fasi dell’analisi e producono gli stessi output presentati nell’articolo.
6. Il pacchetto SCM contiene un errore nel tracciamento dell'anno di trattamento corretto per il grafico, quindi il codice deve essere adattato per tracciare l'anno corretto. Sono stati fatti commenti nei file interessati.
**Importante:** I file `5_` e `6_` e la simulazione Monte Carlo in `7_` utilizzano il pacchetto `pysynthdid` presente nella cartella `package`. 
Il pacchetto è disponibile su GitHub: [https://github.com/MasaAsami/pysynthdid](https://github.com/MasaAsami/pysynthdid)  

**Monte Carlo:**  
1. Il file `7_` contiene due classi che verranno eseguite all’avvio. A seconda del numero di simulazioni, il tempo di elaborazione può essere lungo (circa 1 ora e 15 minuti per 1000 iterazioni su un processore M2 Silicon). Per praticità, il codice è stato impostato per eseguire solo 5 simulazioni.  
2. Per consultare i risultati della Monte Carlo, utilizzate il file `_8` che accede ai dataset nella cartella `simulation_ds`.  
3. I file `.csv` della simulazione contengono i semi (seed) per riprodurre i risultati in conformità alle classi definite in `7_`.

Grazie per l’attenzione e buon proseguimento con l’esecuzione del codice.
