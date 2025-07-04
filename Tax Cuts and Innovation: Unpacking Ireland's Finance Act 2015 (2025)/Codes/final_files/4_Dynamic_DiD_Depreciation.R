rm(list = ls())


library(fixest)
library(dplyr)
library(ggplot2)
library(tidyr)

# Please change ~ with your path 
dpath <- "~/cleanData/did/"
latexpath <- "~/Latex/"
data_file <- file.path("~/final_files/final_clean/depreciation_dataset.csv")


df <- read.csv(data_file, stringsAsFactors = FALSE)


df <- df %>% filter(country %in% c("IRL", "FIN"))



df <- df %>%
  mutate(
    treated = ifelse(country == "IRL", 1, 0),
    post = ifelse(year >= 2015, 1, 0),
    did = treated * post
  )


df <- df %>%
  mutate(
    event_time = year - 2015
  )


colnames(df)


numeric_vars <- c("depreciation", "did", "salary", "gdp", "pop", "cho", "NX", "IDE", "ppa")


df[numeric_vars] <- lapply(df[numeric_vars], function(x) as.numeric(as.character(x)))


df <- df %>% drop_na(all_of(numeric_vars))


event_study_formula <- depreciation ~ i(event_time, ref = -1) +  salary + cho  + ppa +  IDE | country


event_study_model <- feols(event_study_formula, data = df)


summary(event_study_model)

iplot(
  event_study_model,
  xlab = "Periods around the treatment",
  ylab = "Estimated effect - ∆ Depreciation PI Assets",
  main = "Anticipated and Lagged Effects"
)


abline(v = 0, col = rgb(1, 0, 0, alpha = 0.2), lwd = 2) # Rouge transparent


text(x = 0, y = par("usr")[4], labels = "Year 0", col = rgb(1, 0, 0, alpha = 0.5), pos = 3) # Texte légèrement plus opaque




latex_table <- etable(event_study_model, tex = TRUE)

output_path <- file.path("~/final_files/final_LaTeX/depreciation_event_study_table.tex")
writeLines(latex_table, output_path)
