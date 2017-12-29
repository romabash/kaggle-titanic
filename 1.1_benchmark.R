# Benchmark submission: setting all to 0

# Load packages
library(readr)
library(dplyr)

# Read in the Test data
titanic <- read_csv("Data/titanic_test.csv")
titanic
nrow(titanic) #418

benchmark_df <- titanic %>%
  select(PassengerId) %>%
  mutate(Survived = 0)

benchmark_df

write_csv(benchmark_df, "Data/benchmark.csv")
