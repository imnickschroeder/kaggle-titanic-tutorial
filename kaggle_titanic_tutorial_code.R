train <- read.csv('~/R_Projects/titanic_tutorial/kaggle-titanic-tutorial/titanic_train.csv')
test <- read.csv('~/R_Projects/titanic_tutorial/kaggle-titanic-tutorial/titanic_test.csv')

# summarize data
summary(train$Sex)

prop.table(table(train$Sex,train$Survived),1)

# Gender variable
test$Survived <- 0
test$Survived[test$Sex == 'female'] <- 1

# Age variable
summary(train$Age)

train$Child <- 0
train$Child[train$Age<18] <- 1

aggregate(Survived ~ Child + Sex, data=train, FUN=sum)
aggregate(Survived ~ Child + Sex, data=train, FUN=length)
# Proportions of each unique group
aggregate(Survived ~ Child + Sex, data=train, FUN=function(x) {sum(x)/length(x)})

# Bin the fare variable
train$Fare2 <- '30+'
train$Fare2[train$Fare < 30 & train$Fare >= 20] <- '20-30'
train$Fare2[train$Fare < 20 & train$Fare >= 10] <- '10-20'
train$Fare2[train$Fare < 10] <- '10'
aggregate(Survived ~ Fare2 + Pclass + Sex, data=train, FUN=function(x) {sum(x)/length(x)})

# Make a new prediction
test$Survived <- 0
test$Survived[test$Sex == 'female'] <- 1
test$Survived[test$Sex == 'female' & test$Pclass == 3 & test$Fare >= 20] <- 0

# Tutorial 3: Decision trees
#install.packages('rpart')
library(rpart)
fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
             data=train,
             method="class")
plot(fit)
text(fit)
# Let's create better graphics
install.packages('rattle')
install.packages("RGtk2", depen=T, type="source")
#install.packages('rpart.plot')
#install.packages('RColorBrewer')
library(RGtk2)
library(rattle)
library(rpart.plot)
library(RColorBrewer)

install.packages("https://togaware.com/access/rattle_5.0.14.tar.gz", repos=NULL, type="source")

fancyRpartPlot(fit)

# Now use this fit to make a prediction
Prediction <- predict(fit,test,type="class")
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit,file = "decisiontree_titanic.csv",row.names=FALSE)

# Let's 'open up' the algorithm more. Change cp parameter to zero and the minsplit to 2
fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
             data=train,
             method="class", 
             control=rpart.control(minsplit=2, cp=0))
plot(fit)
text(fit)
# This model does not do better than even the simple model! Welcome to overfitting!

# Part 4: Feature engineering
# Feature engineering is the process of using domain knowledge of the data to create features tha make machine learning algorigthms work.
# Feature engineering is fundamental to the application of machine learning, and is both difficult and expensive.
train$Name[1]
# Notice how there is a 'Mr.' in the name field. There are others such as Miss, Mrs, etc.
# This may be a nice predictor!!
train <- train[,-c(13,14)]
test$Survived <- NA
combi <- rbind(train, test)

combi$Name <- as.character(combi$Name)
combi$Name[1]

# Lets break out the string by , and . symbols
strsplit(combi$Name[1],split='[,.]')
strsplit(combi$Name[1],split='[,.]')[[1]]

strsplit(combi$Name[1],split='[,.]')[[1]][2]

# The following code sends each row in the Name vector to the function that we will create, which pulls out the title of the person
combi$Title <- sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
# Now, just drop the spaces
combi$Title <- sub(' ', '', combi$Title)

# Let's combine the Mme and Mlle together
combi$Title[combi$Title %in% c('Mme', 'Mlle')] <- 'Mlle'
# Let's combine more groups
combi$Title[combi$Title %in% c('Capt','Don','Major','Sir')] <- 'Sir'
combi$Title[combi$Title %in% c('Dona','Lady','the Countess','Jonkheer')] <- 'Lady'

# Let's create a new variable called family size. It may be tough for larger families to find everyone before leaving the boat?
combi$FamilySize <- combi$SibSp + combi$Parch + 1

# Maybe certain families had an easier time getting off of the ship, let's create a variable called FamilyID
# First, extract last name
combi$Surname <- sapply(combi$Name, FUN = function(x) {strsplit(x, split='[,.]')[[1]][1]})
combi$FamilyID <- paste(as.character(combi$FamilySize), combi$Surname, sep="")
# Three JOhnsons may have the same Family ID. Let's knock out any family with a family size of less than two and call it a small family
combi$FamilyID[combi$FamilySize <= 2] <- 'Small'
# Let's see how we did
table(combi$FamilyID)
# Hmm, there seems to still be some families with less than 3 people! Let's take a closer look by storing this column and freq into a df
famIDs <- data.frame(table(combi$FamilyID))

# Lets subset the groups that have less than 3 family members
famIDs <- famIDs[famIDs$Freq <= 2,]
# We then need to overwrite any family IDs in our dataset for groups that were not correctly identified and finally convert it to a factor
combi$FamilyID[combi$FamilyID %in% famIDs$Var1] <- 'Small'
combi$FamilyID <- factor(combi$FamilyID)

# Great! Let's break the combi df apart once again and start making predictions using our newly engineered features
train <- combi[1:891,]
test <- combi[892:1309,]

# Let's do anther decision tree!
fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID,
             data=train, 
             method="class")
plot(fit)
text(fit)
# Our rank went up! It is important to create new variables using domain knowledge. This task will take time, and effort, but may be worth it!

# Part 5: Random forests
# Ensemble - a unit or group of complementary parts that contribute to a single effect, especially:
# A coordinated outfit or costume
# A coordinated set of furniture
# A group of musicians, singers, dancers, or actors who perform together

# Understanding what a random forest does
# Bagging example
sample(1:10,replace=TRUE)

# Random forest restriction #1: We can not have NA values
summary(combi$Age) # 20% of rows are missing from the Age variable

# Let's build a predictive model for age to fill in the missing NA values
Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize,
                data=combi[!is.na(combi$Age),],
                method="anova")
combi$Age[is.na(combi$Age)] <- predict(Agefit,combi[is.na(combi$Age),])
summary(combi)

# Embarked has a blank for two passengers
summary(combi$Embarked)
which(combi$Embarked == '')
combi$Embarked[c(62,830)] = "S"
combi$Embarked <- factor(combi$Embarked)

# The last variable with an NA is fare
summary(combi$Fare)
which(is.na(combi$Fare)) # Find na row inded
combi$Fare[1044] <- median(combi$Fare,na.rm=TRUE)

# Restriction #2: The random forest algorithm in R can only diest factors with up to 32 levels
# FamilyID has almost double that. We can take two approaches: One - convert the variable to it's underlying integer values with the unclass() function
# Two - manually reduce the number of levels to keep it under threshold. Let's take the second approach.
combi$FamilyID2 <- combi$FamilyID
combi$FamilyID2 <- as.character(combi$FamilyID2)
combi$FamilyID2[combi$FamilySize <= 3] <- 'Small'
combi$FamilyID2 <- factor(combi$FamilyID2)
levels(combi$FamilyID2)

# We are down to 22 levels, now we can use a random forest model
library(randomForest)
set.seed(415)

combi$Title <- as.factor(combi$Title)

train <- combi[1:891,]
test <- combi[892:1309,]

# Random forest model
fit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare +
                      Embarked + Title + FamilySize + FamilyID2,
                    data=train, 
                    importance=TRUE, 
                    ntree=2000)
varImpPlot(fit)

# Let's see how it did
Prediction <- predict(fit, test)
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit,file="randomforest_titanic.csv",row.names = FALSE)

# This model was not an improvement. This may be because fancier models do not perform well on small datasets, sometimes.
# We will continue to try, and use the a forest of conditional inference trees.
#install.packages('party')
library(party)

set.seed(415)
# Note: Conditional inference trees are able to handle factors with more levels than random forests can. We will use the original family id variable
fit <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare +
                 Embarked + Title + FamilySize + FamilyID,
               data = train,
               controls=cforest_unbiased(ntree=2000, mtry=3))

Prediction <- predict(fit, test, OOB=TRUE, type = "response")
# This model was the best one yet!

## Now, do some more feature engineering to try and obtain a better prediction
# Create child variable
combi$Child[combi$Age < 18] <- 'Child'
combi$Child[combi$Age >= 18] <- 'Adult'
table(combi$Child,combi$Survived)

# Create a mother variable
combi$Mother <- 'Not mother'
combi$Mother[combi$Sex == 'female' & combi$Parch > 0 & combi$Age > 18 & combi$Title != 'Miss'] <- 'Mother'
table(combi$Mother,combi$Survived)

# Change these to factors
combi$Child <- factor(combi$Child)
combi$Mother <- factor(combi$Mother)

# Let's predict once again
# Random forest
train <- combi[1:891,]
test <- combi[892:1309,]

fit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare +
                      Embarked + Title + FamilySize + FamilyID2 + Mother + Child,
                    data=train, 
                    importance=TRUE, 
                    ntree=2000)
varImpPlot(fit)

# Conditional inference trees
fit <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare +
                 Embarked + Title + FamilySize + FamilyID + Child + Mother,
               data = train,
               controls=cforest_unbiased(ntree=2000, mtry=3))
