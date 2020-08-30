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


# get quick look at data format
head(titanic)
str(titanic)
# check for NA's in each column
map(titanic, ~sum(is.na(.)))
# Age (177), Cabin (687), Embarked (2)

# same last names might have better chance of surviving

# graph survivability based on gender and ticket class
titanic %>%
  group_by(Sex, Pclass) %>%
  summarise(mean = mean(Survived))

titanic %>%
  summarise(Age, Sex)

titanic %>%
  summarise_if(is.numeric, mean, na.rm =T)

titanic %>%
  summarise_if(is.numeric, quantile, na.rm =T)
