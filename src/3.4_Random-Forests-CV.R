## Imputing Missing Age with Caret package using knnImpute
## Build Random Forests Model

# Load packages
library(readr)
library(dplyr)
library(stringr)
library(ggplot2)

library(caret)

## Libraries for Parallel Programming
library(parallel)
library (doParallel)


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
         Title = factor(Title),
         FamilySize = factor(FamilySize)) 

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
# - First, Create a Dummy Model to transform all feature to dummy variables (exclude PassenderId and Survived)
dummy_model <- dummyVars(~ ., data = titanic[, c(-1, -2)])

# - Transform Training Set into a Dummy set based on the Dummy Model
train_dummy <- predict(dummy_model, titanic[, c(-1, -2)])
View(train_dummy) # Transformed into Dummy still with Missing Data

# Create a Model to Impute based on the Transformed Train Dummy Set
impute_model <- preProcess(train_dummy, method = "bagImpute")

# Now impute on Training Set using the Impute Model
imputed_train <- predict(impute_model, train_dummy)
View(imputed_train) # Returns a Matrix

# Asign the Imputed Age to the Training Set
titanic$Age <- imputed_train[, 6]
View(titanic)

## Now apply the Impute Model to the Test Set
# - Transformed into Dummy using the Dummy Model created with the Training Set
test_dummy <- predict(dummy_model, titanic_test_final[, c(-1, -2)])
View(test_dummy) 

# Now impute on Test Dummy Set using the Impute Model created with Training Set
imputed_test <- predict(impute_model, test_dummy)
View(imputed_test) # Returns a Matrix

# Asign the Imputed Age to the Test Set
titanic_test_final$Age <- imputed_test[, 6]
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
# Build a Random Forests Model
##############################

# - Using Random Forests
# - 85.31% Accuracy on Validation Set 
# - Smallest OOB error rate with Survived, Pclass, Sex, Title, FamilySize, Fare

## Select the Training Variables
# - Looking at Variable with no Variability
# - All FALSE: No Zero Covariates

nearZeroVar(titanic_train, saveMetrics = TRUE)

# - Using all Variables except for Passenger ID and Embarked
training <- titanic_train %>%
  select(Survived, Pclass, Sex, Age, Title, FamilySize, Fare)

## Pararell Programming
# - Leave 1 core out
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)

# Conigure trainControl
fitControl <- trainControl(method = "repeatedcv", number = 3, repeats = 10, allowParallel = TRUE)

# Build a Model
set.seed(021818)
modelFit <- train(Survived ~ ., method = "rf", data = training, trControl = fitControl)

# Shut down the cluster
stopCluster(cluster)
registerDoSEQ()

modelFit
modelFit$finalModel

## Apply to the Validation set
pred <- predict(modelFit, newdata = titanic_test)

# Look at Confusion Matrix
confusionMatrix(data = pred, reference = titanic_test$Survived) 


############################## 
# Predict on the Holdout Test
##############################


## Apply to the Holdout Test Set
# - At the end, Select only the PassengerId and Prediction Column
# - Rename Prediction to Survived
# - Save as csv file to submit

pred <- predict(modelFit, newdata = titanic_test_final)
pred

result_final <- titanic_test_final %>%
  select(PassengerId) %>%
  mutate(Survived = pred)

## Write to csv to Submit on Kaggle

write_csv(result_final, "../Data/output/3.4.2_Random-Forests-CV.csv")



############################## 
# Build a Boosting Model
##############################

# - Using gbm
# - 85.88% Accuracy on Validation Set 
# - Using Survived, Pclass, Sex, Age, Title, FamilySize, Fare

training <- titanic_train %>%
  select(Survived, Pclass, Sex, Age, Title, FamilySize, Fare)

## Pararell Programming
# - Leave 1 core out
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)

# Conigure trainControl
fitControl <- trainControl(method = "repeatedcv", number = 5, repeats = 10, allowParallel = TRUE)

# Build a Model
set.seed(021818)
modelFit <- train(Survived ~ ., method = "gbm", data = training, trControl = fitControl, verbose = FALSE)

# Shut down the cluster
stopCluster(cluster)
registerDoSEQ()

modelFit
modelFit$finalModel

## Apply to the Validation set
pred <- predict(modelFit, newdata = titanic_test)

# Look at Confusion Matrix
confusionMatrix(data = pred, reference = titanic_test$Survived) 


############################## 
# Predict on the Holdout Test
##############################


## Apply to the Holdout Test Set
# - At the end, Select only the PassengerId and Prediction Column
# - Rename Prediction to Survived
# - Save as csv file to submit

pred <- predict(modelFit, newdata = titanic_test_final)
pred

result_final <- titanic_test_final %>%
  select(PassengerId) %>%
  mutate(Survived = pred)

## Write to csv to Submit on Kaggle

write_csv(result_final, "../Data/output/3.4.1_gbm-CV.csv")



