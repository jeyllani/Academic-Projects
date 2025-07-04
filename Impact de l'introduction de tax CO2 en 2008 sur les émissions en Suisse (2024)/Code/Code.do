********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
*-------------------------------------------------------------------------------
*                          Organisation du code :                              *
*-------------------------------------------------------------------------------
* # 1. Nettoyage des données + Graphiques + création  variable CO2/Population  *
*-------------------------------------------------------------------------------
* # 2. Statistiques descriptives + DID + Graphs +  Export output vers LaTeX    *
*-------------------------------------------------------------------------------
* # 3. Groupe de contrôle synthétique + Graphiques + Export output vers LaTeX  *
*-------------------------------------------------------------------------------
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
* ctrl + F   /Users/jey/Desktop/REME/Exercice/_REME_                           *
********************************************************************************
********************************************************************************
********************************************************************************
// Nettoyage
// Dataset : Passenger transport  : Inland passenger transport
// -> https://stats.oecd.org/index.aspx?r=149679

clear all
set more off, permanently

global dpath "/Users/jey/Desktop/REME/Exercice/_REME_/datasets"
global mypath "/Users/jey/Desktop/REME/Exercice/_REME_/datasets/cleaned"

import delimited "${dpath}/passagers.csv", clear
rename country id
rename v2 country

keep if country == "Switzerland" | country == "Austria" |  country == "Spain" | ///
 country == "France" | country == "Germany" | country == "Iceland" | ///
 country == "Luxembourg" | country == "Portugal" | country == "Ireland" | ///
 country == "Netherlands"

keep if v4 == "Rail passenger transport" // Les données totales sont non-disponibles pour AUT
drop country variable v4 v6 unitcode unit powercodecode powercode ///
 referenceperiodcode referenceperiod flagcodes flags

rename id country


reshape wide value, i(year) j(country) string
drop valueISL


label variable valueAUT "Passenger/KM (mio.) Austria"
label variable valueCHE "Passenger/KM (mio.) Switzerland"
label variable valueFRA "Passenger/KM (mio.) France"
label variable valueDEU "Passenger/KM (mio.) Germany"
label variable valueIRL "Passenger/KM (mio.) Ireland"
label variable valueLUX "Passenger/KM (mio.) Luxembourg"
label variable valueNLD "Passenger/KM (mio.) Netherlands"
label variable valuePRT "Passenger/KM (mio.) Portugal"
label variable valueESP "Passenger/KM (mio.) Spain"

rename valueAUT t_AUT
rename valueCHE t_CHE
rename valueFRA t_FRA
rename valueDEU t_DEU
rename valueIRL t_IRL
rename valueLUX t_LUX
rename valueNLD t_NLE
rename valuePRT t_PRT
rename valueESP t_ESP


drop if year < 1994
drop if year > 2018

gen post2008 = year >= 2008

egen AUT = mean(t_AUT), by(year)
egen CHE = mean(t_CHE), by(year)
egen FRA = mean(t_FRA), by(year)
egen DEU = mean(t_DEU), by(year)
egen IRL = mean(t_IRL), by(year)
egen LUX = mean(t_LUX), by(year)
egen NLE = mean(t_NLE), by(year)
egen PRT = mean(t_PRT), by(year)
egen ESP = mean(t_ESP), by(year)


twoway (scatter  CHE year if post2008 == 0, msymbol(.) msize(tiny) mcolor(black) legend(label(1 "CHE "))) ///
       (scatter  CHE year if post2008 == 1, msymbol(.) msize(tiny) mcolor(black) legend(label(2 "CHE "))) ///
       (lfit     CHE year if post2008 == 0, lcolor(midblue)     lpattern(solid) lwidth(thin) legend(label(3 "CHE "))) ///
       (lfit     CHE year if post2008 == 1, lcolor(midblue)     lpattern(dash) lwidth(thin) legend(label(4 "CHE "))) ///
       (lfit     AUT year if post2008 == 0, lcolor(sand)    lpattern(solid) lwidth(thin) legend(label(5 "AUT "))) ///
       (lfit     AUT year if post2008 == 1, lcolor(sand)    lpattern(dash) lwidth(thin) legend(label(6 "AUT "))) ///
       (lfit     FRA year if post2008 == 0, lcolor(blue)    lpattern(solid) lwidth(thin) legend(label(7 "FRA "))) ///
       (lfit     FRA year if post2008 == 1, lcolor(blue)    lpattern(dash) lwidth(thin) legend(label(8 "FRA"))) ///
       (lfit     DEU year if post2008 == 0, lcolor(brown)  lpattern(solid) lwidth(thin) legend(label(9 "DEU "))) ///
       (lfit     DEU year if post2008 == 1, lcolor(brown)  lpattern(dash) lwidth(thin) legend(label(10 "DEU "))) ///
       (lfit     IRL year if post2008 == 0, lcolor(purple)  lpattern(solid) lwidth(thin) legend(label(11 "IRL"))) ///
       (lfit     IRL year if post2008 == 1, lcolor(purple)  lpattern(dash) lwidth(thin) legend(label(12 "IRL"))) ///
       (lfit     LUX year if post2008 == 0, lcolor(red)   lpattern(solid) lwidth(thin) legend(label(13 "LUX"))) ///
       (lfit     LUX year if post2008 == 1, lcolor(red)   lpattern(dash) lwidth(thin) legend(label(14 "LUX"))) ///
       (lfit     NLE year if post2008 == 0, lcolor(orange)   lpattern(solid) lwidth(thin) legend(label(15 "NLE"))) ///
       (lfit     NLE year if post2008 == 1, lcolor(orange)   lpattern(dash) lwidth(thin) legend(label(16 "NLE"))) ///
       (lfit     PRT year if post2008 == 0, lcolor(pink)    lpattern(solid) lwidth(thin) legend(label(17 "PRT"))) ///
       (lfit     PRT year if post2008 == 1, lcolor(pink)    lpattern(dash) lwidth(thin) legend(label(18 "PRT"))) ///
       (lfit     ESP year if post2008 == 0, lcolor(grey)    lpattern(solid) lwidth(thin) legend(label(19 "ESP"))) ///
       (lfit     ESP year if post2008 == 1, lcolor(grey)    lpattern(dash) lwidth(thin) legend(label(20 "ESP"))), ///
       legend(order(1 3 5 7 9 11 13 15 17 19)) ///
       xline(2008, lpattern(dash) lcolor(black)) ///
       xticks(1994(2)2008 2010(2)2018) ///
       ytitle("Passagers/KM (mio)") xtitle("Années") ///
       title("Fréquentation Transports Rail (1994-2018)") 


drop t_FRA t_DEU t_ESP // Drop FRA, DEU, ESP
// car différence significative avec la Suisse 

keep t_* year


save "/Users/jey/Desktop/REME/Exercice/DID/cleaned/transport_rail.dta", replace
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
// Nettoyage
// Dataset : Historique Population
// -> https://stats.oecd.org/index.aspx

clear all
set more off, permanently
global dpath "/Users/jey/Desktop/REME/Exercice/_REME_/datasets"
global mypath "/Users/jey/Desktop/REME/Exercice/_REME_/datasets/cleaned"

import delimited "${dpath}/HISTPOP_22042024225056132.csv", clear


keep if country == "Switzerland" | country == "Austria" |  country == "Spain" | ///
 country == "France" | country == "Germany" | country == "Iceland" | ///
  country == "Luxembourg" | country == "Portugal" | country == "Ireland" | ///
  country == "Netherlands"

keep if age == "TOTAL" 
keep if v4 == "Total"  
rename time year
keep year value location
rename location country 


reshape wide value, i(year) j(country) string

drop valueISL

// Renommer les variables

label variable valueAUT "Population Austria"
label variable valueCHE "Population Switzerland"
label variable valueFRA "Population France"
label variable valueDEU "Population Germany"
label variable valueIRL "Population Ireland"
label variable valueLUX "Population Luxembourg"
label variable valueNLD "Population Netherlands"
label variable valuePRT "Population Portugal"
label variable valueESP "Population Spain"

rename valueAUT p_AUT
rename valueCHE p_CHE
rename valueFRA p_FRA
rename valueDEU p_DEU
rename valueIRL p_IRL
rename valueLUX p_LUX
rename valueNLD p_NLE
rename valuePRT p_PRT
rename valueESP p_ESP


drop if year < 1994
drop if year > 2021


gen post2008 = year >= 2008

egen AUT = mean(p_AUT/10^6), by(year)
egen CHE = mean(p_CHE/10^6), by(year)
egen FRA = mean(p_FRA/10^6), by(year)
egen DEU = mean(p_DEU/10^6), by(year)
egen IRL = mean(p_IRL/10^6), by(year)
egen LUX = mean(p_LUX/10^6), by(year)
egen NLE = mean(p_NLE/10^6), by(year)
egen PRT = mean(p_PRT/10^6), by(year)
egen ESP = mean(p_ESP/10^6), by(year)


twoway (lfit CHE year if post2008 == 0, lcolor(midblue)    lpattern(solid) lwidth(thin) legend(label(1 "CHE "))) ///
       (lfit CHE year if post2008 == 1, lcolor(midblue)    lpattern(dash) lwidth(thin) legend(label(2 "CHE "))) ///
       (lfit AUT year if post2008 == 0, lcolor(sand)       lpattern(solid) lwidth(thin) legend(label(3 "AUT "))) ///
       (lfit AUT year if post2008 == 1, lcolor(sand)       lpattern(dash) lwidth(thin) legend(label(4 "AUT"))) ///
       (lfit IRL year if post2008 == 0, lcolor(purple)     lpattern(solid) lwidth(thin) legend(label(5 "IRL"))) ///
       (lfit IRL year if post2008 == 1, lcolor(purple)     lpattern(dash) lwidth(thin) legend(label(6 "IRL"))) ///
       (lfit LUX year if post2008 == 0, lcolor(red)        lpattern(solid) lwidth(thin) legend(label(7 "LUX"))) ///
       (lfit LUX year if post2008 == 1, lcolor(red)        lpattern(dash) lwidth(thin) legend(label(8 "LUX"))) ///
       (lfit NLE year if post2008 == 0, lcolor(orange)     pattern(solid) lwidth(thin) legend(label(9 "NLE"))) ///
       (lfit NLE year if post2008 == 1, lcolor(orange)     pattern(dash) lwidth(thin) legend(label(10 "NLE"))) ///
       (lfit PRT year if post2008 == 0, lcolor(pink)       pattern(solid) lwidth(thin) legend(label(11 "PRT"))) ///
       (lfit PRT year if post2008 == 1, lcolor(pink)       pattern(dash) lwidth(thin) legend(label(12 "PRT"))) ///
       (lfit ESP year if post2008 == 0, lcolor(grey)       pattern(solid) lwidth(thin) legend(label(13 "ESP"))) ///
       (lfit ESP year if post2008 == 1, lcolor(grey)       pattern(dash) lwidth(thin) legend(label(14 "ESP"))), ///
       legend(order(1 3 5 7 9 11 13)) ///
       xline(2008, lpattern(dash) lcolor(black)) ///
       ytitle("Millions") xtitle("Années") ///
       title("Population Totale (1994-2018)")


drop p_ESP p_FRA p_DEU p_IRL // Différence significative avec la Suisse - Irlande car taxeDate = 2010

keep p_* year


save "/Users/jey/Desktop/REME/Exercice/DID/cleaned/population.dta", replace
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
// Nettoyage
// Dataset : RealGDP Per capita
// Productivity / Productivity and ULC – Annual, Total Economy
// Growth in GDP per capita, productivity and ULC
// -> https://stats.oecd.org/Index.aspx?DataSetCode=PDB_GR

clear all
set more off, permanently


global dpath "/Users/jey/Desktop/REME/Exercice/_REME_/datasets"
global mypath "/Users/jey/Desktop/REME/Exercice/_REME_/datasets/cleaned"

import delimited "${dpath}/AFDDANN_02052024031927178.csv", clear
rename country id 
rename v2 country 

keep if country == "Switzerland" | country == "Austria" | country == "Luxembourg" | ///
country == "Portugal" | country == "Ireland" | country == "Netherlands"

keep if tablename == "Table 4: Annual real GDP growth per capita, 1990-2028" 

keep value year id 
rename id country 

reshape wide value, i(year) j(country) string

label variable valueAUT "Real GDP (%) Austria"
label variable valueCHE "Real GDP (%) Switzerland"
label variable valueIRL "Real GDP (%) Ireland"
label variable valueLUX "Real GDP (%) Luxembourg"
label variable valueNLD "Real GDP (%) Netherlands"
label variable valuePRT "Real GDP (%) Portugal"

rename valueAUT g_AUT
rename valueCHE g_CHE
rename valueIRL g_IRL
rename valueLUX g_LUX
rename valueNLD g_NLE
rename valuePRT g_PRT

drop if year < 1994
drop if year > 2018


gen post2008 = year >= 2008

egen AUT = mean(g_AUT), by(year)
egen CHE = mean(g_CHE), by(year)
egen IRL = mean(g_IRL), by(year)
egen LUX = mean(g_LUX), by(year)
egen NLE = mean(g_NLE), by(year)
egen PRT = mean(g_PRT), by(year)


twoway (lfit CHE year   if post2008 == 0, lcolor(midblue) lpattern(solid) lwidth(thin) legend(label(1 "CHE "))) ///
       (lfit CHE year   if post2008 == 1, lcolor(midblue) lpattern(dash) lwidth(thin) legend(label(2 "CHE "))) ///
       (lfit AUT year   if post2008 == 0, lcolor(green) lpattern(solid) lwidth(thin) legend(label(3 "AUT "))) ///
       (lfit AUT year   if post2008 == 1, lcolor(green) lpattern(dash) lwidth(thin) legend(label(4 "AUT "))) ///
       (lfit IRL year   if post2008 == 0, lcolor(purple) lpattern(solid) lwidth(thin) legend(label(5 "IRL"))) ///
       (lfit IRL year   if post2008 == 1, lcolor(purple) lpattern(dash) lwidth(thin) legend(label(6 "IRL"))) ///
       (lfit LUX year   if post2008 == 0, lcolor(red) lpattern(solid) lwidth(thin) legend(label(7 "LUX"))) ///
       (lfit LUX year   if post2008 == 1, lcolor(red) lpattern(dash) lwidth(thin) legend(label(8 "LUX"))) ///
       (lfit NLE year   if post2008 == 0, lcolor(orange) lpattern(solid) lwidth(thin) legend(label(9 "NLE"))) ///
       (lfit NLE year   if post2008 == 1, lcolor(orange) lpattern(dash) lwidth(thin) legend(label(10 "NLE"))) ///
       (lfit PRT year   if post2008 == 0, lcolor(grey) lpattern(solid) lwidth(thin) legend(label(11 "PRT"))) ///
       (lfit PRT year   if post2008 == 1, lcolor(grey) lpattern(dash) lwidth(thin) legend(label(12 "PRT"))), ///
       legend(order(1 3 5 7 9 11)) ///
       xline(2008, lpattern(dash) lcolor(black)) ///
       ytitle("PIB Réel par habitant (en %)") xtitle("Années") ///
       title("Croissance du PIB Réel par habitant (1994-2018)") 

keep year g*

save "/Users/jey/Desktop/REME/Exercice/DID/cleaned/real_gdp.dta", replace
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
// Nettoyage
// Dataset : Energy Per capita
// src https://ourworldindata.org/energy

clear all
set more off, permanently

* dpath: c'est l'endroit ou les donnees brutes sont sauvegardees 
global dpath "/Users/jey/Desktop/REME/Exercice/_REME_/datasets"
global mypath "/Users/jey/Desktop/REME/Exercice/_REME_/datasets/cleaned"

import delimited "${dpath}/per-capita-energy-use.csv", clear

rename entity country 
keep if country == "Switzerland" | country == "Austria" | country == "Luxembourg" | ///
country == "Portugal" | country == "Netherlands"

drop if year < 1994
drop if year > 2014

drop country 
rename code country 

rename primaryenergyconsumptionpercapit value 

reshape wide value, i(year) j(country) string

label variable valueAUT "mW/h per person  AUT"
label variable valueCHE "mW/h per person  CHE"
label variable valueLUX "mW/h per person  LUX"
label variable valueNLD "mW/h per person  NLE"
label variable valuePRT "mW/h per person  PRT"

gen k_AUT = valueAUT/10^3
gen k_CHE = valueCHE/10^3
gen k_LUX = valueLUX/10^3
gen k_NLE = valueNLD/10^3
gen k_PRT = valuePRT/10^3

drop v*


gen post2008 = year >= 2008

egen AUT = mean(k_AUT/10^3), by(year)
egen CHE = mean(k_CHE/10^3), by(year)
egen LUX = mean(k_LUX/10^3), by(year)
egen NLE = mean(k_NLE/10^3), by(year)
egen PRT = mean(k_PRT/10^3), by(year)


twoway (lfit CHE year  if post2008 == 0, lcolor(midblue) lpattern(solid) lwidth(thin) legend(label(1 "CHE "))) ///
       (lfit CHE year  if post2008 == 1, lcolor(midblue) lpattern(dash) lwidth(thin) legend(label(2 "CHE "))) ///
       (lfit AUT year  if post2008 == 0, lcolor(green) lpattern(solid) lwidth(thin) legend(label(3 "AUT "))) ///
       (lfit AUT year  if post2008 == 1, lcolor(green) lpattern(dash) lwidth(thin) legend(label(4 "AUT "))) ///
       (lfit LUX year  if post2008 == 0, lcolor(red) lpattern(solid) lwidth(thin) legend(label(5 "LUX"))) ///
       (lfit LUX year  if post2008 == 1, lcolor(red) lpattern(dash) lwidth(thin) legend(label(6 "LUX"))) ///
       (lfit NLE year  if post2008 == 0, lcolor(orange) lpattern(solid) lwidth(thin) legend(label(7 "NLE"))) ///
       (lfit NLE year  if post2008 == 1, lcolor(orange) lpattern(dash) lwidth(thin) legend(label(8 "NLE"))) ///
       (lfit PRT year  if post2008 == 0, lcolor(grey) lpattern(solid) lwidth(thin) legend(label(9 "PRT"))) ///
       (lfit PRT year  if post2008 == 1, lcolor(grey) lpattern(dash) lwidth(thin) legend(label(10 "PRT"))), ///
       legend(order(1 3 5 7 9)) ///
       xline(2008, lpattern(dash) lcolor(black)) ///
       ytitle("Mw/h per person (en milliers)") xtitle("Années") ///
       title("Consommation Mw/h par habitant (1994-2014)") 


keep year k*

save "/Users/jey/Desktop/REME/Exercice/controls/cleaned/energy.dta", replace
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
// Nettoyage
// Dataset : OCDE Transport Equipement / Indicators Performance / Transport 
// -> https://stats.oecd.org/Index.aspx?DataSetCode=ITF_INLAND_INFR

clear all
global dpath "/Users/jey/Desktop/REME/Exercice/_REME_/datasets"
global mypath "/Users/jey/Desktop/REME/Exercice/_REME_/datasets/cleaned"
import delimited "${dpath}/vehicles_ITF_INDICATORS_03052024050655040.csv", clear


rename country id 
rename v2 country 

keep if country == "Switzerland" | country == "Austria" | country == "Luxembourg" | ///
country == "Netherlands" // country == "Portugal" | portugal données manquantes 


keep if v4 == "Buses per one thousand inhabitants"  | ///
v4 == "Passenger cars per one thousand inhabitants" | ///
v4 == "Motorised two-wheelers per one thousand inhabitants"

keep id year value 

rename id country 

drop if year < 1994
drop if year > 2018

collapse (sum) value, by(country year)

reshape wide value, i(year) j(country) string



label variable valueAUT "Vehicules/1000 habitants - Moto|voiture|bus Austria"
label variable valueCHE "Vehicules/1000 habitants - Moto|voiture|bus Switzerland"
label variable valueLUX "Vehicules/1000 habitants - Moto|voiture|bus Luxembourg"
label variable valueNLD "Vehicules/1000 habitants - Moto|voiture|bus Netherlands"

rename valueAUT v_AUT
rename valueCHE v_CHE
rename valueLUX v_LUX
rename valueNLD v_NLE


gen post2008 = year >= 2008

egen AUT = mean(v_AUT), by(year)
egen CHE = mean(v_CHE), by(year)
egen LUX = mean(v_LUX), by(year)
egen NLE = mean(v_NLE), by(year)



twoway (lfit CHE year if post2008 == 0, lcolor(midblue) lpattern(solid) lwidth(thin) legend(label(1 "CHE "))) ///
       (lfit CHE year if post2008 == 1, lcolor(midblue) lpattern(dash) lwidth(thin) legend(label(2 "CHE "))) ///
       (lfit AUT year if post2008 == 0, lcolor(green) lpattern(solid) lwidth(thin) legend(label(3 "AUT "))) ///
       (lfit AUT year if post2008 == 1, lcolor(green) lpattern(dash) lwidth(thin) legend(label(4 "AUT "))) ///
       (lfit NLE year if post2008 == 0, lcolor(orange) lpattern(solid) lwidth(thin) legend(label(5 "NLE"))) ///
       (lfit NLE year if post2008 == 1, lcolor(orange) lpattern(dash) lwidth(thin) legend(label(6 "NLE"))), ///
       legend(order(1 3 5 7 9)) ///
       xline(2008, lpattern(dash) lcolor(black)) ///
       ytitle("Vehicules par habitants (en milliers)") xtitle("Année") ///
       title("Vehicules par habitants (Moto,voiture,bus) (1994-2018)") 

save "/Users/jey/Desktop/REME/Exercice/DID/cleaned/vehicles.dta", replace

********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
// Nettoyage + Création de la variable "CO2/Population"
// Dataset : Historique Population
// -> https://stats.oecd.org/index.aspx
// Total emissions including LULUCF (CO2 equivalent, in million tonnes) 
// -> https://stats.oecd.org/index.aspx // Gaz à effet de serre
// Environnement / Air and Climate / Greenhouse gas emissions by source
// LULUCF = Land Use Change and Forestry (LULUCF) categories

clear all
set more off, permanently

global dpath "/Users/jey/Desktop/REME/Exercice/DID/datasets"
global mypath "/Users/jey/Desktop/REME/Exercice/DID/cleaned/"

import delimited "${dpath}/HISTPOP_22042024225056132.csv", clear

keep if country == "Switzerland" | country == "Austria" |  country == "Spain" | ///
country == "France" | country == "Germany" | country == "Iceland" | ///
country == "Luxembourg" | country == "Portugal" | country == "Ireland" | ///
country == "Netherlands"

keep if age == "TOTAL"  
keep if v4 == "Total"  
rename time year
keep year value country


reshape wide value, i(year) j(country) string

rename valueAustria AUT_POP
rename valueSwitzerland CHE_POP
rename valueSpain ESP_POP
rename valueFrance FRA_POP
rename valueGermany DEU_POP
rename valueIceland ISL_POP
rename valueLuxembourg LUX_POP
rename valuePortugal PRT_POP
rename valueIreland IRL_POP
rename valueNetherlands NLE_POP

drop if year < 1992
drop if year > 2018

save "${mypath}/All_country_Popoulation_1992_2018_clean.dta", replace
********************************************************************************
********************************************************************************

import delimited "${dpath}/AIR_GHG_18042024064906414.csv", clear

* Filtrer les données pour ne garder que celles concernant la Suisse et Aut
keep if country == "Switzerland" | country == "Austria" |  country == "Spain" | ///
 country == "France" | country == "Germany" | country == "Iceland" | country == "Luxembourg" | ///
 country == "Portugal" | country == "Ireland" | country == "Netherlands"

keep if variable == "Total emissions including LULUCF" // Total emissions including LULUCF
keep year value country


reshape wide value, i(year) j(country) string


rename valueAustria AUT_CO2
rename valueSwitzerland CHE_CO2
rename valueSpain ESP_CO2
rename valueFrance FRA_CO2
rename valueGermany DEU_CO2
rename valueIceland ISL_CO2
rename valueLuxembourg LUX_CO2
rename valuePortugal PRT_CO2
rename valueIreland IRL_CO2
rename valueNetherlands NLE_CO2

drop if year < 1992
drop if year > 2018

save "${mypath}/All_countries_C02_Total_emission_1992_2018_clean.dta", replace  


********************************************************************************
********************************************************************************
// Jonction des deux bases de données afin d'obtenir CO2/Population

use  "${mypath}/All_countries_C02_Total_emission_1992_2018_clean.dta", clear

merge 1:1 year using "${mypath}/All_country_Popoulation_1992_2018_clean.dta", keep(match) nogen


local countries CHE AUT ESP FRA DEU ISL LUX PRT IRL NLE

// Loop over countries
foreach country of local countries {
    gen `country'_CO2_POP = (`country'_CO2 / `country'_POP) * 1000 // -> Tonne par habitant `country'
}

keep year CHE_CO2_POP AUT_CO2_POP ESP_CO2_POP FRA_CO2_POP DEU_CO2_POP ISL_CO2_POP ///
LUX_CO2_POP PRT_CO2_POP IRL_CO2_POP NLE_CO2_POP

label variable year "Année"
rename CHE_CO2_POP y_CHE
label variable y_CHE "CHE CO2/hab. (t)  Taxe Date introd. 2008"
rename AUT_CO2_POP y_AUT
label variable y_AUT "AUT CO2/hab. (t)  Taxe Date introd. 2022"
rename ESP_CO2_POP y_ESP
label variable y_ESP "ESP CO2/hab. (t)  Taxe Date introd. 2014"
rename FRA_CO2_POP y_FRA
label variable y_FRA "FRA CO2/hab. (t)  Taxe Date introd. 2014"
rename DEU_CO2_POP y_DEU
label variable y_DEU "DEU CO2/hab. (t)  Taxe Date introd. 2021"
rename ISL_CO2_POP y_ISL
label variable y_ISL "ISL CO2/hab. (t)  Taxe Date introd. 2010"
rename LUX_CO2_POP y_LUX
label variable y_LUX "LUX CO2/hab. (t)  Taxe Date introd. 2021"
rename PRT_CO2_POP y_PRT
label variable y_PRT "PRT CO2/hab. (t)  Taxe Date introd. 2015"
rename IRL_CO2_POP y_IRL
label variable y_IRL "IRL CO2/hab. (t)  Taxe Date introd. 2010"
rename NLE_CO2_POP y_NLE
label variable y_NLE "NLE CO2/hab. (t)  Taxe Date introd. 2021"


save "${mypath}/co2_par_habitant.dta", replace

********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************

// Jonction des données 

clear all

global mypath "/Users/jey/Desktop/REME/Exercice/DID/cleaned/"

use "${mypath}/co2_par_habitant.dta",clear
merge 1:1 year using "${mypath}/vehicles.dta", keep(match) nogen
merge 1:1 year using "${mypath}/energy.dta", keep(match) nogen
merge 1:1 year using "${mypath}/real_gdp.dta", keep(match) nogen
merge 1:1 year using "${mypath}/population_cleaned.dta", keep(match) nogen
merge 1:1 year using "${mypath}/transport_rail.dta", keep(match) nogen

save "${mypath}/dataset.dta", replace

use "${mypath}/dataset.dta",clear


********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
// Statistiques descriptives 

clear
clear all
set more off
********************************************************************************
********************************************************************************
use "/Users/jey/Desktop/REME/Exercice/DID/cleaned/dataset.dta", clear 
// y_ = CO2/habitant
// p_ = Population  
// g_ = ∆ PIB per capita
// t_ = Transport Passager
// v_ = Véhicules (privés, commerciaux aggrégés)
********************************************************************************
********************************************************************************
keep y_AUT y_CHE g_AUT g_CHE p_AUT p_CHE t_AUT t_CHE v_AUT v_CHE year

keep y_NLE g_NLE p_NLE t_NLE v_NLE year
keep y_LUX g_LUX p_LUX t_LUX v_LUX year
reshape long y_ p_ g_ t_ v_, i(year) j(country) string 

rename y_ co2_par_habitant
rename p_ population
rename g_ pib_par_habitant
rename t_ transport_passager
rename v_ vehicules

********************************************************************************
// Output Statistiques : count mean sd min max skewness kurtosis
estpost summarize co2_par_habitant population pib_par_habitant transport_passager vehicules, detail 
esttab using statistiques_descriptives.tex, cells("count mean sd min max skewness kurtosis") replace
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
// Difference-in-Difference entre la Suisse et l'Autriche 
// y_ = CO2/habitant
// p_ = Population  
// g_ = ∆ PIB per capita
// t_ = Transport Passager
// v_ = Véhicules (privés, commerciaux aggrégés)
********************************************************************************
clear
clear all
set more off
********************************************************************************
********************************************************************************
use "/Users/jey/Desktop/REME/Exercice/DID/cleaned/dataset.dta", clear 

keep y_AUT y_CHE g_AUT g_CHE p_AUT p_CHE t_AUT t_CHE v_AUT v_CHE year

reshape long y_ p_ g_ t_ v_, i(year) j(country) string 

gen posTreat = year >= 2008
gen treated = 0
replace treated = 1 if country == "CHE"
gen interaction = posTreat*treated

encode country, gen(country_num)


tsset country_num year

// Effet^*** négatif significatif suite à la mise en place de la taxe CO2 en Suisse
// coef: -1.263466  (p-value: 0.000)
didregress (y_) (interaction) (p_) (t_) (g_) (v_) , group(treated) t(year) 

// Violation de l'hypothèse des tendances parallèles 
// H0: Linear trends are parallel : Prob > F =   0.0000

// Export the regression results to a LaTeX file
outreg2 using "/Users/jey/Desktop/REME/Exercice/G17_DID/did.tex", replace

estat ptrends

// estimates -> ptrends -> latex
local N = r(N)
local F = r(F)
local df_r = r(df_r)
local p = r(p)
local df_m = r(df_m)

// Fichier .tex pour écrire les résultats
file open myfile using "/Users/jey/Desktop/REME/Exercice/G17_DID/ptrends.tex", write replace
file write myfile "{\\begin{tabular}{|c|c|c|c|c|} \hline}"
file write myfile "N & F-statistic & df(m) & df(r) & p-value \\\\ \\hline \n"
file write myfile "`N' & `F' & `df_m' & `df_r' & `p' \\\\ \\hline \n"
file write myfile "\\end{tabular}\n"
file close myfile
// Nécessite un reformattage pour être utilisé dans un document LaTeX
// Car Stata n'interprète pas RegeX lors de l'exportation 

esttab using "/Users/jey/Desktop/REME/Exercice/G17_DID/ptrends.tex", replace

// Graphique des tendances
estat trendplots, omeans xline(2008) ///
    line1opts(lcolor(red) lpattern(solid)) ///
    line2opts(lcolor(blue) lpattern(solid)) ///
    title("Moyennes Observées") ///
    xtitle("Années") ///
    ytitle("CO2 Par habitant (en tonnes)") ///
    legend(order(1 "Contrôle (AUT)" 2 "Traitement (CHE)")) ///
    noxline ///

********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
********************************************************************************
// # Création d'un groupe control synthétique avec le package synth 
// # ssc install synth 
// # si ssc install synth ne fonctionne pas: MAC : Aller dans applications 
// # puis stata/stataSE.app lire les informations, ouvrir avec Rosetta
********************************************************************************
********************************************************************************
// # Pré-analyse *visuelle! des variables explicatives afin qu'elles matchent en 
// ~tendance et en niveau. 
********************************************************************************
********************************************************************************
// 1. Objectif : 
// Elles doivent expliquer la variabilité de la variable dépendante (CO2/habit), 
// ne pas avoir été impacté par la taxe C02 et matcher en tendance et en niveau.
// (Construction d'un intervalle de confiance autour des valeurs Suisse possible 
// comme condition de sélection)
********************************************************************************
// 2. Pays pour la création du contrefactuel (pays synthétique) :
// Allemagne, France, Espagnes : éliminés à cause de la taille de la population.
// Irlande, Islande: éliminés car l'introduction de la taxe C02 survient en 2010 
//                   dans un premier temps. 
// Variable Consommation mw/h par personne éliminée car seul AUT match et le 
// test en amont montre qu'il y a violation de l'hyp. des tendances parallèles. 
//
// # Les candidats ISL et IRL dont la taxe est introduite en 2010 sont éliminés
// # car la taxe est introduite en 2008 en Suisse -> maximisation étendue post-
// # traitement.
********************************************************************************
// 3. Candidats retenus : Hollande, Luxembourg, Autriche 
********************************************************************************

clear
clear all
use "/Users/jey/Desktop/REME/Exercice/G17_DID/dataset.dta", clear 

keep y_CHE y_AUT y_NLE y_LUX g_CHE g_AUT g_NLE g_LUX p_CHE p_AUT p_NLE p_LUX t_CHE ///
 t_AUT t_NLE t_LUX v_CHE v_AUT v_NLE v_LUX year

reshape long y_ t_ p_ g_ v_, i(year) j(country) string

encode country, gen(country_num) // CHE = 2 -> label list

tsset country_num year

label list country_num // Confirmer sur console le code pour la Suisse = 2 

rename y_ co2_par_habitant
rename year annees 

synth  co2_par_habitant t_ p_ g_ v_, trunit(2) trperiod(2008) fig // Alternative -> synth_runner

// Extraction des poids + RMSPE sous forme matricielle et export en .tex

ereturn list

matrix W = e(W_weights) 
matrix RMSPE = e(RMSPE)

esttab mat(W) using "/Users/jey/Desktop/REME/Exercice/G17_DID/synth_weights.tex", replace
esttab mat(RMSPE) using "/Users/jey/Desktop/REME/Exercice/G17_DID/synth_rmspe.tex", replace 


// Le graph montre que les variables explicatives communes ne sont pas 
// suffisantes pour recréer un contrefactuel synthétique de qualité 
// pré-traitement pour la Suisse.

// Root Mean Squared Prediction Error (RMSPE) (élevé)
// Le contrôle synthétique semble ne pas être une bonne approximation 
// de l'unité traitée car < 1% des données NLE/LUX sont utilisées
// Au vu de la contribution de AUT > 99%, le résultat de la DiD fait référence
// => Violation de l'hpothèse des tendances parallèles 
// Détails dans 17_papier.pdf 