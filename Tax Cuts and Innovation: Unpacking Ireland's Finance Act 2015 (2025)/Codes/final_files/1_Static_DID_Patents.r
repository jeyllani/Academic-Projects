rm(list = ls())



library(fixest)
library(dplyr)
library(ggplot2)
library(tidyr)


dpath <- "~/cleanData/did/"
latexpath <- "~/Latex/"

data_file <- file.path("~/final_files/final_clean/patents_dataset.csv")

df <- read.csv(data_file, stringsAsFactors = FALSE)



df <- df %>% filter(country %in% c("IRL", "CHE"))


df <- df %>%
  mutate(
    treated = ifelse(country == "IRL", 1, 0),
    post = ifelse(year >= 2015, 1, 0),
    did = treated * post
  )


# colnames(df)

numeric_vars <- c("brevets", "did", "gdp", "pop", "NX", "public_spending")


df[numeric_vars] <- lapply(df[numeric_vars], function(x) as.numeric(as.character(x)))


df <- df %>% drop_na(all_of(numeric_vars))



formula <- brevets ~ did + gdp + pop + NX + public_spending | country + year


model <- feols(formula, data = df)


summary(model)



latex_table <- etable(model, tex = TRUE)

output_path <- file.path("~/final_files/final_LaTeX/patents_table.tex")
writeLines(latex_table, output_path)