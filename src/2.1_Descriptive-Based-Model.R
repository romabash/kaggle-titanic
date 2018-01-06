## Load packages

library(readr)
library(dplyr)
library(tibble)
library(ggplot2)
library(gridExtra)

library(caret)

############################## Load Titanic Data #################################

titanic_train <- read_csv("../Data/titanic_train.csv")
dim(titanic_train)
titanic_test <- read_csv("../Data/titanic_test.csv")
dim(titanic_test)

## Combine Train data with Test data to clean up together

# Add Survived Variable to the Test data to match the Train data in order to combine (Make it a second Column)
titanic_test <- titanic_test %>%
  mutate(Survived = NA) %>%
  select(PassengerId, Survived, everything())

titanic_test

#Combine Train and Test with rbind()
titanic_combine <- rbind(titanic_train, titanic_test)
dim(titanic_combine)
head(titanic_combine, 2)
tail(titanic_combine, 2)


############################## Clean the Missing Data ###########################


## Look at Missing Values (NA) and Replace them

titanic_combine %>% 
  select_if(function(x) any(is.na(x))) %>% 
  summarise_all(funs(sum(is.na(.)))) 

# - Ignore Missing Data in Test data Survived 
# - Ignore Cabin for now (too much missing data)
# - Focus on Age, Embarked and Fare Varibales

# Replace NA in Embarked with "S" 
titanic_combine <- titanic_combine %>%
  mutate(Embarked = replace(Embarked, is.na(Embarked), "S"))

# Replace NA in Age with Mean 
age_mean <- mean(titanic_combine$Age, na.rm = TRUE) # 29.88
titanic_combine <- titanic_combine %>%
  mutate(Age = replace(Age, is.na(Age), age_mean))

# Replace NA in Fair with Mean 
fare_mean <- mean(titanic_combine$Fare, na.rm = TRUE) # 33.29
titanic_combine <- titanic_combine %>%
  mutate(Fare = replace(Fare, is.na(Fare), fare_mean))

## Add another Varibale for Family Size
# - Add SibSp and Parch + self

titanic_combine <- titanic_combine %>%
  mutate(FamilySize = 1 + SibSp + Parch) 

## Convert "Pclass", "Survived", "Sex", and "Embarked" Variables into Factors
titanic_combine <- titanic_combine %>%
  mutate(Pclass = factor(Pclass), Survived = factor(Survived), Sex = factor(Sex), Embarked = factor(Embarked)) 

## Look at final version of the Dataset
str(titanic_combine)


############################## Data Partition ################################


## Separate the Train data and Test data from Combined Dataset
# - Separate based on NA in Survived
# - Set the Test data aside for now


titanic_test_final <- titanic_combine %>%
  filter(is.na(Survived))
dim(titanic_test_final)

titanic <- titanic_combine %>%
  filter(!is.na(Survived))
dim(titanic)

## Partinion the Train dataset into Train and Validation data
# - Partition using the "caret" package
# - Create Training set indecies with 80% of data

set.seed(333)

inTrain <- createDataPartition(y = titanic$Survived, p = 0.80, list = FALSE)
head(inTrain)

# Sub-set titanic data to Train and to Test
titanic_train <- titanic[inTrain, ]
titanic_test <- titanic[-inTrain, ]

# 714 rows for Training Data and 177 rows for Test Data
dim(titanic_train) # 714 by 13
dim(titanic_test) # 177 by 13

head(titanic_train)

## Looking At Survival by Gender, Pclass, Age and FamilySize

titanic_train %>%
  ggplot() +
  geom_point(aes(x = Age, y = FamilySize, color = Survived), alpha = 0.7) +
  facet_grid(Sex ~ Pclass) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5, size=18, colour = "#054354")) +
  ggtitle("Titanic Survival Rate") +
  scale_x_continuous(name= "Passenger Age", breaks = 10*c(0:8)) +
  scale_y_discrete(name = "Family Size") +
  scale_color_discrete(name = "Outcome", labels = c("Did NOT Survived", "Survived"))  


###################### Predictive Model based on Gender ########################


## Model to Predict Survival based on being a Female
# - Create a function that takes a dtaframe as an argument, and returns an updated dataframe
# - Mutate a new Variable Column "Prediction" to predict Survival
# - if Sex is "female", assign 1, else 0
# - 78.57% Accuracy on Training Set
# - 79.1% Accuracy on Validation Set
# - 78.68% Accuracy on the Whole Titanic Set

predict_model_female <- function(df) {
  df <- df %>%
    select(PassengerId, Pclass, Sex, Age, FamilySize, Survived) %>%
    mutate(Prediction = ifelse(Sex == "female",1,0))
  return(df)
}

# Assign the result of a function to result 
result <- predict_model_female(titanic_train)

# Create a table of Errors
table(result$Prediction, result$Survived)

# Create a Confusion Matrix manually
rbind("Model_Female:" = c(Accuracy = mean(result$Prediction == result$Survived), 
                          "Total Correct" = paste0(sum(result$Prediction == result$Survived), " of ", nrow(result))))

# Use confusionMatrix() function from "caret" package (Need to install e1071 package)
confusionMatrix(data = result$Prediction, reference = result$Survived) 

## Apply to the Validation set

result_test <- predict_model_female(titanic_test)
confusionMatrix(data = result_test$Prediction, reference = result_test$Survived)  

## Apply to the whole Titanic set

result_titanic <- predict_model_female(titanic)
confusionMatrix(data = result_titanic$Prediction, reference = result_titanic$Survived) 

## Apply to the Holdout Test Set
# - At the end, Select only the PassengerId and Prediction Column
# - Rename Prediction to Survived
# - Save as csv file to submit

result_titanic_test_final <- predict_model_female(titanic_test_final)

result_titanic_test_final <- result_titanic_test_final %>%
  select(PassengerId, Prediction) %>%
  rename(Survived = Prediction)

## Write to csv to Submit on Kaggle
write_csv(result_titanic_test_final, "../Data/output/Female-Model.csv")

