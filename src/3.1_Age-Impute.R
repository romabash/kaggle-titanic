## Imputing Missing Age with Caret package

# Load packages
library(readr)
library(dplyr)
library(stringr)
library(ggplot2)

library(caret)

############################## 
# Load Titanic Data 
##############################

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

# Combine Train and Test with rbind()
titanic_combine <- rbind(titanic_train, titanic_test)
dim(titanic_combine)
head(titanic_combine, 2)
tail(titanic_combine, 2)


############################## 
# Feature Engineering 
##############################

## Add another Varibale for Family Size
# - Add SibSp and Parch + self

titanic_combine <- titanic_combine %>%
  mutate(FamilySize = 1 + SibSp + Parch) 


## Add a new Varibale to the Dataset called "Title"
# - Add a title to each observation based on the name

titanic_combine <- titanic_combine %>%
  mutate(Title = ifelse(str_detect(Name, "(Mr(\\.+|\\s+))"), "Mr",
                 ifelse(str_detect(Name, "Master\\.?"), "Master",
                 ifelse(str_detect(Name, "Mrs\\.?"), "Mrs",
                 ifelse(str_detect(Name, "((Miss|Ms)(\\.+|\\s+))"), "Miss", "Other"))))
  )

# Total of 29 observations without a Title
titanic_combine %>%
  group_by(Title) %>%
  summarise(Count = n())

# Look at Observations with "Other"
titanic_combine %>%
  filter(Title == "Other") %>%
  select(Name, Age, Sex) %>%
  print(n = nrow(.))

## Replace "Other" with a Title Based on the Mean Age of Title
# - If Title is "Other" and Sex is "male" and Age is greater than the Mean of Master, replace with "Mr"

master_age <- mean(titanic_combine[titanic_combine$Title == "Master", ]$Age, na.rm = TRUE)
mr_age <- mean(titanic_combine[titanic_combine$Title == "Mr", ]$Age, na.rm = TRUE)
miss_age <- mean(titanic_combine[titanic_combine$Title == "Miss", ]$Age, na.rm = TRUE)
mrs_age <- mean(titanic_combine[titanic_combine$Title == "Mrs", ]$Age, na.rm = TRUE)

titanic_combine <- titanic_combine %>%
  mutate(Title = ifelse(Title == "Other" & (Sex == "male" & Age > master_age), "Mr",
                 ifelse(Title == "Other" & (Sex == "male" & Age < master_age), "Master",
                 ifelse(Title == "Other" & (Sex == "female" & Age > miss_age), "Mrs",
                 ifelse(Title == "Other" & (Sex == "female" & Age < miss_age), "Miss", Title))))
  ) 

# Look at Title
titanic_combine %>%
  group_by(Title) %>%
  summarise(Count = n())

## Replace 1 NA with a Title based on Name description
# - Since it's a Dr. and a Male, replace with Mr

titanic_combine %>%
  filter((is.na(Title))) %>%
  select(Name, Age, Sex) %>%
  print(n = nrow(.))

titanic_combine <- titanic_combine %>%
  mutate(Title = ifelse(is.na(Title), "Mr", Title))


############################## 
# Feature Selection 
##############################

## Select only the Variables needed (keeping PassengerId for Test set later)

titanic_combine <- titanic_combine %>%
  select(PassengerId, Survived, Pclass, Sex, Age, Fare, Embarked, Title, FamilySize)

## Convert "Pclass", "Survived", "Sex", "Embarked", "Title", and "FamilySize" Variables to Factors
titanic_combine <- titanic_combine %>%
  mutate(Pclass = factor(Pclass), 
         Survived = factor(Survived), 
         Sex = factor(Sex), 
         Embarked = factor(Embarked),
         Title = factor(Title)) 

str(titanic_combine)


############################## 
# Replace Missing Data
##############################

## Look at any Missing Data

titanic_combine %>% 
  select_if(function(x) any(is.na(x))) %>% 
  summarise_all(funs(sum(is.na(.)))) 

# Replace NA in Embarked with "S" 
titanic_combine <- titanic_combine %>%
  mutate(Embarked = replace(Embarked, is.na(Embarked), "S"))

# Replace NA in Fair with Mean 
fare_mean <- mean(titanic_combine$Fare, na.rm = TRUE) # 33.29
titanic_combine <- titanic_combine %>%
  mutate(Fare = replace(Fare, is.na(Fare), fare_mean))


############################## 
# Impute Missing Age
##############################


## Separate the Train data and Test data from Combined Dataset
# - Separate based on NA in Survived
# - Set the Test data aside for now

titanic_test_final <- titanic_combine %>%
  filter(is.na(Survived))
dim(titanic_test_final)

titanic <- titanic_combine %>%
  filter(!is.na(Survived))
dim(titanic)

## Impute Missing Age with caret 
# - First, transform all feature to dummy variables (exclude PassenderId and Survived)

dummy_vars <- dummyVars(~ ., data = titanic[, c(-1, -2)])

# - Train on Training set and predict on Training Set
train_dummy <- predict(dummy_vars, titanic[, c(-1, -2)])
View(train_dummy) # Transformed into Dummy still with Missing Data

# Now impute on Training Set
pre_process <- preProcess(train_dummy, method = "bagImpute")
imputed_data <- predict(pre_process, train_dummy)
View(imputed_data) # Returns a Matrix

titanic$Age <- imputed_data[, 6]
View(titanic)

## Now apply the Model to the Test Set

test_dummy <- predict(dummy_vars, titanic_test_final[, c(-1, -2)])
View(test_dummy) # Transformed into Dummy still with Missing Data

# Now impute on Test Set
pre_process <- preProcess(test_dummy, method = "bagImpute")
imputed_data <- predict(pre_process, test_dummy)
View(imputed_data) # Returns a Matrix

titanic_test_final$Age <- imputed_data[, 6]
View(titanic_test_final)


############################## 
# Data Partition
##############################


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
dim(titanic_train)
dim(titanic_test)

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


############################## 
# Build a Model
##############################


## Model to Predict Survival based on Gender, Pclass, Age and FamilySize
# - Females in Pclass 1 and 2 Survived (97% and 92%)
# - Females on Pclass 3 Survived if Family Size below 5
# - Males in Pclass 1 and 2 Survived if: Age is below 15
# - Males in Pclass 3 Survived if: Age is below 15 and Family Size is 4 and Below
# - 78.01% Accuracy on Training Set
# - 85.31% Accuracy on Validation Set (Might be too high)
# - 79.946% Accuracy on the Whole Titanic Set (Better than based only on Gender)

predict_model <- function(df) {
  df <- df %>%
    select(PassengerId, Pclass, Sex, Age, FamilySize, Survived) %>%
    mutate(Prediction = ifelse(Sex == "female" & (Pclass == 1 | Pclass == 2), 1,
                        ifelse(Sex == "female" & (Pclass == 3 & FamilySize < 5), 1,
                        ifelse(Sex == "male" & (Pclass == 1 | Pclass == 2 & (Age < 15)), 1,
                        ifelse(Sex == "male" & (Pclass == 3 & (Age < 15 & FamilySize < 5)), 1, 0))))
    )
  return(df)
}


# Assign the result of a function to result 
result <- predict_model(titanic_train)
confusionMatrix(data = result$Prediction, reference = result$Survived) 

## Apply to the Validation set

result_test <- predict_model(titanic_test)
confusionMatrix(data = result_test$Prediction, reference = result_test$Survived)  

## Apply to the whole Titanic set

result_titanic <- predict_model(titanic)
confusionMatrix(data = result_titanic$Prediction, reference = result_titanic$Survived) 

## Apply to the Holdout Test Set
# - At the end, Select only the PassengerId and Prediction Column
# - Rename Prediction to Survived
# - Save as csv file to submit

result_final <- predict_model(titanic_test_final)

result_final <- result_final %>%
  select(PassengerId, Prediction) %>%
  rename(Survived = Prediction)

## Write to csv to Submit on Kaggle

write_csv(result_final, "../Data/output/3.1_Imputed-Age.csv")

