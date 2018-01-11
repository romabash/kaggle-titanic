
## Descriptive based Model.  
## Replacing Missing Age with Mean Age based on the Title of the Name  

# Load packages
library(readr)
library(dplyr)
library(tibble)
library(ggplot2)
library(gridExtra)

library(stringr)
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


## Add another Varibale for Family Size
# - Add SibSp and Parch + self

titanic_combine <- titanic_combine %>%
  mutate(FamilySize = 1 + SibSp + Parch) 

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

# Replace NA in Fair with Mean 
fare_mean <- mean(titanic_combine$Fare, na.rm = TRUE) # 33.29
titanic_combine <- titanic_combine %>%
  mutate(Fare = replace(Fare, is.na(Fare), fare_mean))


## Convert "Pclass", "Survived", "Sex", and "Embarked" Variables into Factors
titanic_combine <- titanic_combine %>%
  mutate(Pclass = factor(Pclass), Survived = factor(Survived), Sex = factor(Sex), Embarked = factor(Embarked)) 


############################## Cleaning Missing Age ################################

## Replace NA in Age 
# - Use the mean of Age based on the Title in the Name Variable 
# - Take a mean of all Miss, Mrs, Mr, and Master

# Average Age on Titanic
age_mean <- mean(titanic_combine$Age, na.rm = TRUE) # 29.88

# Use stringr package to filter only rows that contain "Mr." in the name
titanic_mr <- titanic_combine %>%
  filter(str_detect(Name, "Mr\\."))

# Find the mean Age of all Mr.
mr_mean_age <- mean(titanic_mr$Age, na.rm = TRUE) #32.25 (Higher than overall mean)

# Use stringr package to filter only rows that contain "Master." in the name
titanic_master <- titanic_combine %>%
  filter(str_detect(Name, "Master\\."))

# Find the mean Age of all Mr.
master_mean_age <- mean(titanic_master$Age, na.rm = TRUE) #5.48 (Much lower than overall mean)

# Use stringr package to filter only rows that contain "Mrs." in the name
titanic_mrs <- titanic_combine %>%
  filter(str_detect(Name, "Mrs\\."))

# Find the mean Age of all Mr.
mrs_mean_age <- mean(titanic_mrs$Age, na.rm = TRUE) #36.99 (Higher than overall mean)

# Use stringr package to filter only rows that contain "Miss." in the name
titanic_miss <- titanic_combine %>%
  filter(str_detect(Name, "Miss\\."))

# Find the mean Age of all Mr.
miss_mean_age <- mean(titanic_miss$Age, na.rm = TRUE) #21.77 (Lower than overall mean)

# - All Title observations do not add up to total number in "titanic_combined"
# - 34 observations do not match
# - The ones that do not match will be assigned the Total Mean of Age
nrow(titanic_combine) - (nrow(titanic_mr) + nrow(titanic_master) + nrow(titanic_mrs) + nrow(titanic_miss))

## Add a new Varibale to the Dataset called "Title"
# - Add a title to each observation based on the name

titanic_combine <- titanic_combine %>%
  mutate(Title = ifelse(  str_detect(Name, "Mr\\."), "Mr",
                 ifelse(  str_detect(Name, "Master\\."), "Master",
                 ifelse(  str_detect(Name, "Mrs\\."), "Mrs",
                 ifelse(  str_detect(Name, "Miss\\."), "Miss", "Other"  ))))
  )

titanic_combine %>%
  filter(str_detect(Title, "Other"))

## Replace Missing Age with Mean Age of the Title

titanic_combine <- titanic_combine %>%
  mutate(Age = ifelse(  is.na(Age) & Title == "Mr", round(mr_mean_age),
               ifelse(  is.na(Age) & Title == "Master", round(master_mean_age),
               ifelse(  is.na(Age) & Title == "Mrs", round(mrs_mean_age),
               ifelse(  is.na(Age) & Title == "Miss", round(miss_mean_age), 
               ifelse(  is.na(Age) & Title == "Other", round(age_mean), Age)))))
  )


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

################  Predictive Model based on Gender, Pclass, Age and FamilySize ##############


## Model to Predict Survival based on Gender, Pclass, Age and FamilySize
# - Females in Pclass 1 and 2 Survived (97% and 92%)
# - Females on Pclass 3 Survived if: Below 35 Age and Fammily Size below 5
# - Males in Pclass 1 and 2 Survived if: Age is below 15
# - Males in Pclass 3 Survived if: Age is below 15 and Family Size is 4 and Below
# - 78.57% Accuracy on Training Set
# - 85.31% Accuracy on Validation Set (Might be too high)
# - 79.91% Accuracy on the Whole Titanic Set (Better than based only on Gender)

predict_model <- function(df) {
  df <- df %>%
    select(PassengerId, Pclass, Sex, Age, FamilySize, Survived) %>%
    mutate(Prediction = ifelse(Sex == "female" & (Pclass == 1 | Pclass == 2), 1,
                        ifelse(Sex == "female" & (Pclass == 3 & Age < 40 & FamilySize < 5), 1,
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

result_titanic_test_final <- predict_model(titanic_test_final)

result_titanic_test_final <- result_titanic_test_final %>%
  select(PassengerId, Prediction) %>%
  rename(Survived = Prediction)

## Write to csv to Submit on Kaggle

write_csv(result_titanic_test_final, "../Data/output/2.2_Descriptive-based-model-age-tuned.csv")
