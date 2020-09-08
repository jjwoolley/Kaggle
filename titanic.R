# https://www.kaggle.com/c/titanic/overview

# train dataset (n = 891)
# test dataset (n = 418)

# goal: predict which individuals survived the titanic

# variables 
# pclass (ticket class)
# sibsp (# of siblings/spouses aboard the titanic)
# ticket (ticket #)
# parch (# of parents/children aboard the titanic)
# embarked (port of Embarkation)
    # C = Cherbourg, Q = Queenstown, S = Southampton

# Loading Data------------------------------------------------------------------


library("tidyverse")
library("haven")
library("GGally")
library("MissMech")
library("mice")
library("Hmisc")
library("randomForest")

# load data and change certain variable types
titanic <- read_csv ("titanic.train.csv") %>%
  mutate(Sex = factor(Sex),
         Ticket = factor(Ticket),
         Embarked = factor(Embarked),
         PassengerId = as.character(PassengerId),
         Survived = Survived,
         Pclass = factor(Pclass)) %>%
  separate(Name, into = c("Last_name", "First_name"), sep = ",")

titanic.test <- read_csv("titanic.test.csv") %>%
  mutate(Sex = factor(Sex),
  Ticket = factor(Ticket),
  Embarked = factor(Embarked),
  PassengerId = as.character(PassengerId),
  Pclass = factor(Pclass)) %>%
  separate(Name, into = c("Last_name", "First_name"), sep = ",")

original <- titanic

# PLAN--------------------------------------------------------------------------

# cross folds (v = 5)
# run a few different models 
# (logistic, lasso logistic, ridge logistic, K-NN,
# Linear Discriminant Analysis, LDA, QDA, PCA)
# Determine which has the lowest classification error and submit the top few

# Issues
# Missing data: How to deal with missing data for age
#               imputation: look at family members ages or their ticket class,
#                           Look at title (Mrs vs Miss, Lady, etc)
# Variable Selection: Which to use and which not to use
    # create new variables: family that survived (yes, no, not sure)
    #                       age as a categorical var (children vs adults, 10 year bins, etc)
# Model Selection: which model to use

#group individuals as family/traveling partners based off of last name, ticket number,
#   cabin, etc




# Descriptive Stats-------------------------------------------------------------

# get quick look at data format
head(titanic)
str(titanic)

# check for NA's in each column
summary(titanic)
map(titanic, ~sum(is.na(.)))
# Age (177), Cabin (687), Embarked (2)

# same last names might have better chance of surviving

# survival rate based on gender and ticket class
age.by_sex_pclass <- titanic %>%
  group_by(Sex, Pclass) %>%
  summarise(Survival = mean(Survived),
            m_age = mean(Age, na.rm = T))

titanic %>%
  group_by(Embarked) %>%
  summarise(mean = mean(Survived))

# men vs women
titanic %>%
  count(Sex)

# look at distribution of age
titanic %>%
  summarise(mean_age = mean(Age, na.rm = T),
            min = quantile(Age, probs = 0, na.rm = T),
            "0.25" = quantile(Age, probs = 0.25, na.rm = T),
            median = quantile(Age, probs = 0.5, na.rm = T),
            "0.75" = quantile(Age, probs = 0.75, na.rm = T),
            max = max(Age, na.rm = T))

titanic %>%
  summarise_if(is.numeric, mean, na.rm =T)

titanic %>%
  summarise_if(is.numeric, quantile, na.rm =T)

# Visualizations----------------------------------------------------------------
ggpairs(titanic, columns = c(2:3, 6:9, 11, 13))


# Missing Data------------------------------------------------------------------
# convert titanic df to numeric/binary data only before testing

x <- select(titanic, Survived, Pclass, Sex, Age, SibSp, Parch)
t <- model.matrix(Survived ~., model.frame(~ ., data = x, na.action = na.pass))[,-1]
y <- cbind(titanic$Survived, t)
TestMCARNormality(data = t)
# we can confirm age is not missing completely at random

# Titanic full------------------------------------------------------------------
age.by_sex_pclass
titanic.full <- bind_rows(titanic,titanic.test) %>%
  mutate(group = ifelse(is.na(Survived), "test", "train")) %>%
  mutate(Age = ifelse(Sex == "female" & Pclass == "1" , 34.6, Age)) %>%
  mutate(Age = ifelse(Sex == "female" & Pclass == "2" , 28.7, Age)) %>%
  mutate(Age = ifelse(Sex == "female" & Pclass == "3" , 21.8, Age)) %>%
  mutate(Age = ifelse(Sex == "male" & Pclass == "1" , 41.3, Age)) %>%
  mutate(Age = ifelse(Sex == "male" & Pclass == "2" , 30.7, Age)) %>%
  mutate(Age = ifelse(Sex == "male" & Pclass == "3" , 26.5, Age)) %>%
  mutate(Survived = factor(Survived))

# give age based on mean of sex & pclass
titanic.full.test <- filter(titanic.full, group == "test")
titanic.full.train <- filter(titanic.full, group == "train")

# Model1: Logistic -------------------------------------------------------------
fit1.lm <- lm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
              data = titanic)
summary(fit1.lm)

titanic.test$Age <- impute(titanic.test$Age, mean)

fit2.log <- glm(Survived ~ Pclass + Sex + Age + SibSp,
                family = "binomial",
                data = titanic)
summary(fit2.log)


# Impute Age--------------------------------------------------------------------
# mean and logit

# 0.75837 26th percentile (not very good, but not awful for being so basic)
titanic.test$Age <- impute(titanic.test$Age, mean)

fit2.log <- glm(Survived ~ Pclass + Sex + Age + SibSp,
                family = "binomial",
                data = titanic)
summary(fit2.log)
pred.logit <- predict(fit2.log, titanic.test, type = "response")
guess.logit <- ifelse(pred.logit > 0.5, 1, 0)
submit.logit <- data.frame(PassengerId = titanic.test$PassengerId, Survived = guess.logit)
#write.csv(submit.logit, file = "submit_logit.csv", row.names = FALSE)

# grouped mean & logit 0.75837 26th percentile
fit3.log <- glm(Survived ~ Pclass + Sex + Age + SibSp,
                family = "binomial",
                data = titanic.full.train)
summary(fit3.log)
pred.logit.3 <- predict(fit3.log, titanic.full.test, type = "response")
guess.logit.3 <- ifelse(pred.logit.3 > 0.5, 1, 0)
submit.logit.3 <- data.frame(PassengerId = titanic.full.test$PassengerId, Survived = guess.logit.3)
#write.csv(submit.logit, file = "submit_logit_3.csv", row.names = FALSE)



# random forest 0.77272 54th percentile
fit4.rf <- randomForest(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare, 
             data = titanic.full.train, 
             mtry = 6,
             ntree = 5000)
pred.rf.4 <- predict(fit4.rf, titanic.full.test, type = "response")
submit.rf.4 <- tibble(PassengerId = titanic.full.test$PassengerId, Survived = pred.rf.4)
#write.csv(submit.rf.4, file = "submit_rf_4.csv", row.names = FALSE)
