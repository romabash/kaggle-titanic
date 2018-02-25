
## Feature and Model Selection

# Load packages
library(readr)
library(dplyr)
library(stringr)
library(ggplot2)

library(caret)
library(rpart.plot)

## Libraries for Parallel Programming
library(parallel)
library (doParallel)

####################################
# Load Titanic Data 
####################################

titanic_train <- read_csv("../Data/titanic_train.csv")
dim(titanic_train)
titanic_test <- read_csv("../Data/titanic_test.csv")
dim(titanic_test)

## Combine Train data with Test data to clean up together
# - Add Survived Variable to the Test data to match the Train data in order to combine (Make it a second Column)
titanic_test <- titanic_test %>%
  mutate(Survived = NA) %>%
  select(PassengerId, Survived, everything())

titanic_test

## Combine Train and Test with rbind()
titanic_combine <- rbind(titanic_train, titanic_test)
dim(titanic_combine)
head(titanic_combine, 2)
tail(titanic_combine, 2)


#################################### 
# Feature Engineering 1: Family Size
####################################


## Add another Varibale for Family Size
# - Add SibSp and Parch + self
titanic_combine <- titanic_combine %>%
  mutate(FamilySize = 1 + SibSp + Parch) 


#################################### 
# Feature Engineering 2: Ticket 
####################################


## Looking at the Fare distribution
# - Median and Mean are not close, skewed
# - Also some Outliers (Max is $512 compared to Mean of $33) 
summary(titanic_combine$Fare)

## Take a look at Ticket Varibale to see if people share the same ticket
# - Few shared tickets with higher Fare in higher Pclass
# - Average Fare Price per person might be a better representation
titanic_combine %>%
  group_by(Ticket, Fare, Pclass) %>%
  summarise(Ticket_Count = n()) %>%
  arrange(desc(Count))

## Create a new Variable "Ticket_Count" to Represent the number of people on the same Ticket
titanic_combine <- titanic_combine %>% 
  group_by(Ticket) %>%
  mutate(Ticket_Count = n()) %>%
  ungroup()

## Closer look at Ticket_Count
# - Ticket_Count does not always equal to the FamilySize
titanic_combine %>%
  group_by(Ticket) %>%
  arrange(desc(Ticket_Count)) %>%
  print(n = 20)


#################################### 
# Feature Engineering 3: Average Fare Price
####################################


## Create a new Varibale for Average Fare Price based on the number of people on the same Ticket
titanic_combine <- titanic_combine %>% 
  mutate(Fare_Ave = Fare/Ticket_Count)

# - Look at the summary of Average Fare
summary(titanic_combine$Fare)
summary(titanic_combine$Fare_Ave)


####################################
# Replace Missing Data 1: Fare
####################################

## Look at any Missing Data
# - Ignore NA in Survived (Test set)
# - Cabin has too many Missing Data (1014), ignore for now
# - Focus on Fare, Embarked and Age
titanic_combine %>% 
  select_if(function(x) any(is.na(x))) %>% 
  summarise_all(funs(sum(is.na(.))))

## Look at the Observation with missing Fare
# - Single Male from Pclass 3, Embarked from S
titanic_combine %>%
  filter((is.na(Fare))) %>%
  print(n = nrow(.))

## Look at Median Average Fare price in Pclass3, Embarked from "S"
# - Median price is $7.7958 
titanic_combine %>%
  filter(Pclass == 3 & Embarked == "S") %>%
  summarise(Count = n(),
            Average_Fare = median(Fare_Ave, na.rm = TRUE))

# Replace NA in Fare with the Median Fare_Ave price in Pclass3, Embarked from "S"
# - Also Replace Fare_Ave based on new Fare and Ticket_Count
fare_median <- median(titanic_combine[titanic_combine$Pclass == '3' & titanic_combine$Embarked == 'S', ]$Fare_Ave, na.rm = TRUE)

titanic_combine <- titanic_combine %>%
  mutate(Fare = replace(Fare, is.na(Fare), fare_median)) %>%
  mutate(Fare_Ave = replace(Fare_Ave, is.na(Fare_Ave), fare_median))


####################################
# Feature Engineering 4: Title
####################################


## The name Varibale includes a Title of a person that can be extracted
titanic_combine %>% select(Name)

## Add a new Varibale called "Title"
# - Using regex to extract Title from the Name Variable
# - If a Title doesn't match the regex for Mr, Master, Mrs and Miss, assign it to "Other" for now
titanic_combine <- titanic_combine %>%
  mutate(Title = ifelse(str_detect(Name, "(Mr(\\.+|\\s+))"), "Mr",
                        ifelse(str_detect(Name, "Master\\.?"), "Master",
                               ifelse(str_detect(Name, "Mrs\\.?"), "Mrs",
                                      ifelse(str_detect(Name, "((Miss|Ms)(\\.+|\\s+))"), "Miss", "Other"))))
  )

## Closer look at the Title Varibale
# - Total of 29 observations without a Title
titanic_combine %>%
  group_by(Title) %>%
  summarise(Count = n())

## Look at Observations with "Other"
titanic_combine %>%
  filter(Title == "Other") %>%
  select(Name, Age, Sex) %>%
  print(n = nrow(.))

## Replace "Other" with a Title Based on the Mean Age of Title
# - If Title is "Other" and Sex is "male" and Age is greater than the Median of "Master", replace with "Mr"
master_age <- median(titanic_combine[titanic_combine$Title == "Master", ]$Age, na.rm = TRUE)
mr_age <- median(titanic_combine[titanic_combine$Title == "Mr", ]$Age, na.rm = TRUE)
miss_age <- median(titanic_combine[titanic_combine$Title == "Miss", ]$Age, na.rm = TRUE)
mrs_age <- median(titanic_combine[titanic_combine$Title == "Mrs", ]$Age, na.rm = TRUE)

titanic_combine <- titanic_combine %>%
  mutate(Title = ifelse(Title == "Other" & (Sex == "male" & Age > master_age), "Mr",
                        ifelse(Title == "Other" & (Sex == "male" & Age < master_age), "Master",
                               ifelse(Title == "Other" & (Sex == "female" & Age > miss_age), "Mrs",
                                      ifelse(Title == "Other" & (Sex == "female" & Age < miss_age), "Miss", Title))))
  ) 

## Look at Title again to see if anything was missed
# - 1 Observation was missed because of missing Age
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

## Look at Title again to see if Title corresponds to Sex
titanic_combine %>%
  group_by(Sex, Title) %>%
  summarise(Count = n())

table(titanic_combine$Sex)


####################################
# Replace Missing Data 2: Embarked
####################################


## Look at the Observations with missing Embarked
# - Both are Females from Pclass 1, with the same Cabin, Ticket and Fare price
# - Both are also single with FamilySize of 1
titanic_combine %>%
  filter((is.na(Embarked))) %>%
  print(n = nrow(.))

## Look at Average Fare price for Singles in Pclass1 based on Embarked locations
# - Most of people is Pclass 1 Embarked from "C" or "S", with only 3 from "Q"
# - Based on Median Average Fare price, "C" is the closest to the $40 
titanic_combine %>%
  filter(Pclass == 1) %>%
  group_by(Embarked) %>%
  summarise(Count = n(),
            Average_Fare = median(Fare_Ave))

## Replace NA in Embarked with "C" 
titanic_combine <- titanic_combine %>%
  mutate(Embarked = replace(Embarked, is.na(Embarked), "C"))


####################################
# Replace Missing Data 3: Impute Missing Age
####################################

## Look at Age distribution to compare later
age_dist_before <- titanic_combine %>%
  ggplot() +
  geom_histogram(aes(x = Age), binwidth = 5, color = "#355a63", fill = "#96e4f7") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5, size=18)) +
  ggtitle("Titanic Age Distribution") +
  scale_x_continuous(name= "Passenger Age", breaks = 5*c(0:18)) +
  scale_y_continuous(name = "Passenger Count")

age_dist_before

## Impute Missing Age with caret 
# - Break up the Combined dataset into Training and Test to use Training as a Model
# - Separate to not use Testing Dataset to influence the Imputing 
# - Select only Variables needed to impute as Factors
training <- titanic_combine %>%
  filter(!is.na(Survived)) %>%
  select(Pclass, Sex, Age, FamilySize, Embarked, Title) %>%
  mutate(Pclass = factor(Pclass), 
         Sex = factor(Sex), 
         Title = factor(Title),
         FamilySize = factor(FamilySize),
         Embarked = factor(Embarked)) 
dim(training)

testing <- titanic_combine %>%
  filter(is.na(Survived)) %>%
  select(Pclass, Sex, Age, FamilySize, Embarked, Title) %>%
  mutate(Pclass = factor(Pclass), 
         Sex = factor(Sex), 
         Title = factor(Title),
         FamilySize = factor(FamilySize),
         Embarked = factor(Embarked)) 
dim(testing)


## Create a Dummy Model to transform all Selected Features to dummy variables 
dummy_model <- dummyVars(~ ., data = training)
dummy_model

## Transform into a Dummy set based on the Dummy Model
training_dummy <- predict(dummy_model, training)
head(training_dummy) # Transformed into Dummy still with Missing Data

## Create a Model to Impute based on the Transformed Training Dummy Set
impute_model <- preProcess(training_dummy, method = "bagImpute")

## Now impute on Training Set using the Impute Model
imputed_train <- predict(impute_model, training_dummy)
head(imputed_train) # Returns a Matrix

## Assign the Imputed Age to the Training Set
training$Age <- imputed_train[, 6]
head(training)

## Now apply the Impute Model to the Test Set
# - Transformed into Dummy using the Dummy Model created with the Training Set
testing_dummy <- predict(dummy_model, testing)
head(testing_dummy) 

## Now impute on Test Dummy Set using the Impute Model created with Training Set
imputed_test <- predict(impute_model, testing_dummy)
head(imputed_test) # Returns a Matrix

## Asign the Imputed Age to the Test Set
testing$Age <- imputed_test[, 6]
head(testing)

## Combine the Training and the Testing together and Assign the Age to the titanic_combined dataframe
combined_age <- rbind(training, testing)

titanic_combine$Age <- combined_age$Age
head(titanic_combine)

## Look at Age distribution after Imputing Age
age_dist_after <- titanic_combine %>%
  ggplot() +
  geom_histogram(aes(x = Age), binwidth = 5, color = "#355a63", fill = "#96e4f7") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5, size=18)) +
  ggtitle("Titanic Age Distribution") +
  scale_x_continuous(name= "Passenger Age", breaks = 5*c(0:18)) +
  scale_y_continuous(name = "Passenger Count")

age_dist_after

####################################
# Feature Engineering 3: Child or Adult
####################################


## Add another Variable to distingush a Child from an Adult
# - Master is a good representation of a male child
# - Miss is not as good as average age of Miss is 22 compared to 4 of Master
# - Assign a Child if less than 15 years old

titanic_combine <- titanic_combine %>%
  mutate(Gender = ifelse(Age <= 15 & Sex == "female", "Girl",
                         ifelse(Age > 15 & Sex == "female", "Woman",
                                ifelse(Age <= 15 & Sex == "male", "Boy", "Man"))))


####################################
# Data Partition
####################################


## Convert Variables into Factors
titanic_combine <- titanic_combine %>%
  mutate(Survived = factor(Survived), 
         Pclass = factor(Pclass),
         Sex = factor(Sex), 
         Embarked = factor(Embarked),
         Title = factor(Title),
         Gender = factor(Gender)) 



## Split the Combined Dataset into the Train and Final Test 
titanic <- titanic_combine %>%
  filter(!is.na(Survived))  
dim(training)

titanic_test_final <- titanic_combine %>%
  filter(is.na(Survived))
dim(testing)


## Use the Training Set to Visualize the Survival based on Age, Gender, FamilySize and Pclass
titanic %>%
  ggplot() +
  geom_point(aes(x = Age, y = FamilySize, color = Survived), alpha = 0.7) +
  facet_grid(Gender ~ Pclass) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5, size=18, colour = "#054354")) +
  ggtitle("Titanic Survival Rate") +
  scale_x_continuous(name= "Passenger Age", breaks = 10*c(0:8)) +
  scale_y_discrete(name = "Family Size") +
  scale_color_discrete(name = "Outcome", labels = c("Did NOT Survived", "Survived")) 

## Visualize the Survival based on Average Fare, Title, Ticket_Count and Pclass
titanic %>%
  ggplot() +
  geom_point(aes(x = Fare_Ave, y = Ticket_Count, color = Survived), alpha = 0.7) +
  facet_grid(Title ~ Pclass, scales = "free") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5, size=18)) +
  ggtitle("Titanic Survival Rate") +
  scale_x_continuous(name= "Fare") +
  scale_y_discrete(name = "Ticket Party Size") +
  scale_color_discrete(name = "Outcome", labels = c("Did NOT Survived", "Survived"))


## Partinion the Train dataset into Train and Validation data
# - Partition using the "caret" package
# - Create Training set indecies with 80% of data

set.seed(123)

inTrain <- createDataPartition(y = titanic$Survived, p = 0.80, list = FALSE)
head(inTrain)

## Sub-set titanic data to Train and to Test
titanic_train <- titanic[inTrain, ]
titanic_test <- titanic[-inTrain, ]

## 714 rows for Training Data and 177 rows for Test Data
dim(titanic_train)
dim(titanic_test)

head(titanic_train)


####################################
# Build a Model Function
####################################


## Create a function for Pararell Programming 
# - Leave 1 core out
rpartModel <- function(x, y, method, train_control, seed){
  
  cluster <- makeCluster(detectCores() - 1)
  registerDoParallel(cluster)
  
  ## Build a Model
  set.seed(seed)
  modelFit <- train(x = x, 
                    y = y, 
                    method = method, 
                    tuneLength = 30,
                    trControl = train_control)
  
  ## Shut down the cluster
  stopCluster(cluster)
  registerDoSEQ()
  
  ## Return Model
  return (modelFit)
}


############################## 
# Using Decision Tree to choose best Features
# Model 1
##############################


## Select the Training Variables
# - Looking at Variable with no Variability
# - All FALSE: No Zero Covariates
nearZeroVar(titanic_train, saveMetrics = TRUE)

## rpart Model 1
# - Selecting a Label and a Training set instead of a formula
# - Set Training set as a datarame insead of a tibble
label <- as.factor(titanic_train$Survived)
  
training <- titanic_train %>%
  select(Pclass, Sex, Age, Title, Gender, FamilySize, Ticket_Count, Fare_Ave, Embarked) %>%
  as.data.frame()

## Conigure trainControl
fitControl <- trainControl(method = "repeatedcv", 
                           number = 3, 
                           repeats = 10, 
                           allowParallel = TRUE)

model1 <- rpartModel(x = training, y = label, 
                     method = "rpart", 
                     train_control = fitControl, 
                     seed = 32323)

## Final Model
model1
model1$finalModel

rpart.plot(model1$finalModel, type = 0, extra = 4)
prp(model1$finalModel, type = 0, extra = 1, under = TRUE)

## Apply to the Validation set
# - Look at Confusion Matrix: 83.62
pred <- predict(model1, newdata = titanic_test)
confusionMatrix(data = pred, reference = titanic_test$Survived) 


############################## 
# Model 2
##############################


## rpart Model 2
# - Selecting a Label and a Training set instead of a formula
# - Set Training set as a datarame insead of a tibble
label <- as.factor(titanic_train$Survived)

training <- titanic_train %>%
  select(Pclass, Age, Title, Gender, FamilySize, Ticket_Count, Fare_Ave) %>%
  as.data.frame()

## Conigure trainControl
fitControl <- trainControl(method = "repeatedcv", 
                           number = 3, 
                           repeats = 10, 
                           allowParallel = TRUE)

model2 <- rpartModel(x = training, y = label, 
                     method = "rpart", 
                     train_control = fitControl, 
                     seed = 32323)

## Final Model
model2
model2$finalModel

rpart.plot(model2$finalModel, type = 0, extra = 4)
prp(model2$finalModel, type = 0, extra = 1, under = TRUE)

## Apply to the Validation set
# - Look at Confusion Matrix: 83.62
pred <- predict(model2, newdata = titanic_test)
confusionMatrix(data = pred, reference = titanic_test$Survived)


############################## 
# Model 3
##############################


## rpart Model 3
# - Selecting a Label and a Training set instead of a formula
# - Set Training set as a datarame insead of a tibble
label <- as.factor(titanic_train$Survived)

training <- titanic_train %>%
  select(Pclass, Age, Title, FamilySize, Ticket_Count, Fare_Ave) %>%
  as.data.frame()

## Conigure trainControl
fitControl <- trainControl(method = "repeatedcv", 
                           number = 3, 
                           repeats = 10, 
                           allowParallel = TRUE)

model3 <- rpartModel(x = training, y = label, 
                     method = "rpart", 
                     train_control = fitControl, 
                     seed = 32323)

## Final Model
model3
model3$finalModel

rpart.plot(model3$finalModel, type = 0, extra = 4)
prp(model3$finalModel, type = 0, extra = 1, under = TRUE)

## Apply to the Validation set
# - Look at Confusion Matrix: 83.62
pred <- predict(model3, newdata = titanic_test)
confusionMatrix(data = pred, reference = titanic_test$Survived) 


############################## 
# Model 4
##############################


## rpart Model 4
# - Selecting a Label and a Training set instead of a formula
# - Set Training set as a datarame insead of a tibble
label <- as.factor(titanic_train$Survived)

training <- titanic_train %>%
  select(Pclass, Gender, Ticket_Count, Fare_Ave) %>%
  as.data.frame()

## Conigure trainControl
fitControl <- trainControl(method = "repeatedcv", 
                           number = 3, 
                           repeats = 10, 
                           allowParallel = TRUE)

model4 <- rpartModel(x = training, y = label, 
                     method = "rpart", 
                     train_control = fitControl, 
                     seed = 32323)

## Final Model
model4
model4$finalModel

rpart.plot(model4$finalModel, type = 0, extra = 4)
prp(model4$finalModel, type = 0, extra = 1, under = TRUE)

## Apply to the Validation set
# - Look at Confusion Matrix: 85.88
# - Better Accuracy with these Features
pred <- predict(model4, newdata = titanic_test)
confusionMatrix(data = pred, reference = titanic_test$Survived)


############################## 
# Predict on the Holdout Test
##############################


## Apply to the Holdout Test Set
# - At the end, Select only the PassengerId and Prediction Column
# - Rename Prediction to Survived
# - Save as csv file to submit

pred <- predict(model4, newdata = titanic_test_final)
pred

result_final <- titanic_test_final %>%
  select(PassengerId) %>%
  mutate(Survived = pred)

## Write to csv to Submit on Kaggle
# - .77511 on Kaggle
write_csv(result_final, "../Data/output/3.6.1_rpart-k3-n10.csv")

# Kaggle Score 0.79904: Higher than some Random Forests submitions


####################################
# Build a Random Forests Model Function
####################################


## Create a function for Pararell Programming 
# - Leave 1 core out
rfModel <- function(x, y, method, train_control, seed){
  
  cluster <- makeCluster(detectCores() - 1)
  registerDoParallel(cluster)
  
  ## Build a Model
  set.seed(seed)
  modelFit <- train(x = x, 
                    y = y, 
                    method = method, 
                    tuneLength = 3,
                    trControl = train_control)
  
  ## Shut down the cluster
  stopCluster(cluster)
  registerDoSEQ()
  
  ## Return Model
  return (modelFit)
}



############################## 
# RF Model 1
##############################


## RF Model 1
# - Selecting a Label and a Training set instead of a formula
# - Set Training set as a datarame insead of a tibble
label <- as.factor(titanic_train$Survived)

training <- titanic_train %>%
  select(Pclass, Gender, Ticket_Count, Fare_Ave) %>%
  as.data.frame()

## Conigure trainControl
fitControl <- trainControl(method = "repeatedcv", 
                           number = 3, 
                           repeats = 10, 
                           allowParallel = TRUE)

modelRF <- rfModel(x = training, y = label, 
                      method = "rf", 
                      train_control = fitControl, 
                      seed = 32323)

## Final Model: 15.97% OOB
modelRF
modelRF$finalModel

## Apply to the Validation set
# - Look at Confusion Matrix: 83.63
pred <- predict(modelRF, newdata = titanic_test)
confusionMatrix(data = pred, reference = titanic_test$Survived)


############################## 
# Predict on the Holdout Test
##############################


## Apply to the Holdout Test Set
# - At the end, Select only the PassengerId and Prediction Column
# - Rename Prediction to Survived
# - Save as csv file to submit

pred <- predict(modelRF, newdata = titanic_test_final)
pred

result_final <- titanic_test_final %>%
  select(PassengerId) %>%
  mutate(Survived = pred)

## Write to csv to Submit on Kaggle
# - .77511 on Kaggle
write_csv(result_final, "../Data/output/3.6.2_rf-k3-n10.csv")

# Kaggle Score 0.74641: Much Lower than rpart


############################## 
# RF Model 2: Title and Age instead of Gender
##############################


## RF Model 2
# - Selecting a Label and a Training set instead of a formula
# - Set Training set as a datarame insead of a tibble
label <- as.factor(titanic_train$Survived)

training <- titanic_train %>%
  select(Pclass, Age, Title, Ticket_Count, Fare_Ave) %>%
  as.data.frame()

## Conigure trainControl
fitControl <- trainControl(method = "repeatedcv", 
                           number = 3, 
                           repeats = 10, 
                           allowParallel = TRUE)

modelRF2 <- rfModel(x = training, y = label, 
                   method = "rf", 
                   train_control = fitControl, 
                   seed = 32323)

## Final Model: 14.99% OOB
modelRF2
modelRF2$finalModel

## Apply to the Validation set
# - Look at Confusion Matrix: 83.62
pred <- predict(modelRF2, newdata = titanic_test)
confusionMatrix(data = pred, reference = titanic_test$Survived)


############################## 
# Predict on the Holdout Test
##############################


## Apply to the Holdout Test Set
# - At the end, Select only the PassengerId and Prediction Column
# - Rename Prediction to Survived
# - Save as csv file to submit

pred <- predict(modelRF, newdata = titanic_test_final)
pred

result_final <- titanic_test_final %>%
  select(PassengerId) %>%
  mutate(Survived = pred)

## Write to csv to Submit on Kaggle
# - .77511 on Kaggle
write_csv(result_final, "../Data/output/3.6.3_rf-k3-n10.csv")

# Kaggle Score 0.74641: Same as beore Much Lower than rpart



############################## 
# RF Model 3: RF without CV
##############################


## RF Model 3
# - Selecting a Label and a Training set instead of a formula
# - Set Training set as a datarame insead of a tibble
label <- as.factor(titanic_train$Survived)

training <- titanic_train %>%
  select(Pclass, Age, Title, Ticket_Count, Fare_Ave) %>%
  as.data.frame()

modelRF3 <- train(x = training, y = label, method = "rf")

## Final Model: 15.83% OOB
modelRF3
modelRF3$finalModel

## Apply to the Validation set
# - Look at Confusion Matrix: 83.62
pred <- predict(modelRF3, newdata = titanic_test)
confusionMatrix(data = pred, reference = titanic_test$Survived)


############################## 
# Predict on the Holdout Test
##############################


## Apply to the Holdout Test Set
# - At the end, Select only the PassengerId and Prediction Column
# - Rename Prediction to Survived
# - Save as csv file to submit

pred <- predict(modelRF, newdata = titanic_test_final)
pred

result_final <- titanic_test_final %>%
  select(PassengerId) %>%
  mutate(Survived = pred)

## Write to csv to Submit on Kaggle
# - .77511 on Kaggle
write_csv(result_final, "../Data/output/3.6.4_rf-noCV.csv")

# Kaggle Score 0.74641: Same as beore Much Lower than rpart


####################################
# Build a Boosting Model gbm Function
####################################


## Create a function for Pararell Programming 
# - Leave 1 core out
gbmModel <- function(x, y, method, train_control, seed){
  
  cluster <- makeCluster(detectCores() - 1)
  registerDoParallel(cluster)
  
  ## Build a Model
  set.seed(seed)
  modelFit <- train(x = x, 
                    y = y, 
                    method = method, 
                    tuneLength = 3,
                    trControl = train_control)
  
  ## Shut down the cluster
  stopCluster(cluster)
  registerDoSEQ()
  
  ## Return Model
  return (modelFit)
}



############################## 
# gbm Model 1
##############################


## gbm Model 1
# - Selecting a Label and a Training set instead of a formula
# - Set Training set as a datarame insead of a tibble
label <- as.factor(titanic_train$Survived)

training <- titanic_train %>%
  select(Pclass, Gender, Ticket_Count, Fare_Ave) %>%
  as.data.frame()

## Conigure trainControl
fitControl <- trainControl(method = "repeatedcv", 
                           number = 3, 
                           repeats = 10, 
                           allowParallel = TRUE)

modelgbm1 <- gbmModel(x = training, y = label, 
                   method = "rf", 
                   train_control = fitControl, 
                   seed = 32323)

## Final Model: 15.97% OOB
modelgbm1
modelgbm1$finalModel

## Apply to the Validation set
# - Look at Confusion Matrix: 83.62
pred <- predict(modelgbm1, newdata = titanic_test)
confusionMatrix(data = pred, reference = titanic_test$Survived)


############################## 
# Predict on the Holdout Test
##############################


## Apply to the Holdout Test Set
# - At the end, Select only the PassengerId and Prediction Column
# - Rename Prediction to Survived
# - Save as csv file to submit

pred <- predict(modelgbm1, newdata = titanic_test_final)
pred

result_final <- titanic_test_final %>%
  select(PassengerId) %>%
  mutate(Survived = pred)

## Write to csv to Submit on Kaggle
# - .77511 on Kaggle
write_csv(result_final, "../Data/output/3.6.5_gbm-k3-n10.csv")

# Kaggle Score 0.74641: Same as RF, Much Lower than rpart


############################## 
# gbm Model 2
##############################


## gbm Model 2
# - Selecting a Label and a Training set instead of a formula
# - Set Training set as a datarame insead of a tibble
label <- as.factor(titanic_train$Survived)

training <- titanic_train %>%
  select(Pclass, Sex, Age, Title, Ticket_Count, Fare_Ave) %>%
  as.data.frame()

## Conigure trainControl
fitControl <- trainControl(method = "repeatedcv", 
                           number = 3, 
                           repeats = 10, 
                           allowParallel = TRUE)

modelgbm2 <- gbmModel(x = training, y = label, 
                      method = "rf", 
                      train_control = fitControl, 
                      seed = 32323)

## Final Model: 15.83% OOB
modelgbm2
modelgbm2$finalModel

## Apply to the Validation set
# - Look at Confusion Matrix: 83.05
pred <- predict(modelgbm2, newdata = titanic_test)
confusionMatrix(data = pred, reference = titanic_test$Survived)


############################## 
# Predict on the Holdout Test
##############################


## Apply to the Holdout Test Set
# - At the end, Select only the PassengerId and Prediction Column
# - Rename Prediction to Survived
# - Save as csv file to submit

pred <- predict(modelgbm2, newdata = titanic_test_final)
pred

result_final <- titanic_test_final %>%
  select(PassengerId) %>%
  mutate(Survived = pred)

## Write to csv to Submit on Kaggle
# - .77511 on Kaggle
write_csv(result_final, "../Data/output/3.6.6_gbm-k3-n10.csv")

# Kaggle Score 0.77990: Better than before, Much Lower than rpart


