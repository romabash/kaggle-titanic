# Benchmark submission: All Females Survived

# Load packages
library(readr)
library(dplyr)

# Read in the Test data
titanic <- read_csv("Data/titanic_test.csv")
titanic

benchmark_df <- titanic %>%
  select(PassengerId, Sex) %>%
  mutate(Survived = ifelse(Sex == "female", 1, 0)) %>%
  select(PassengerId, Survived) 

benchmark_df

write_csv(benchmark_df, "Data/output/Benchmark-Gender.csv")
