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

# load data and change certain variable types
titanic <- read_csv ("train.csv") %>%
  mutate(Sex = factor(Sex),
         Ticket = factor(Ticket),
         Embarked = factor(Embarked),
         PassengerId = as.character(PassengerId),
         Survived = Survived,
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
# Variable Selection: Which to use and which not to use







# Descriptive Stats-------------------------------------------------------------

# get quick look at data format
head(titanic)
str(titanic)

# check for NA's in each column
map(titanic, ~sum(is.na(.)))
# Age (177), Cabin (687), Embarked (2)

# same last names might have better chance of surviving

# survival rate based on gender and ticket class
titanic %>%
  group_by(Sex, Pclass) %>%
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




# Models------------------------------------------------------------------------
fit1.lm <- lm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
              data = titanic)
summary(fit1.lm)

fit2.log <- glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
                family = "binomial",
                data = titanic)
summary(fit2.log)
