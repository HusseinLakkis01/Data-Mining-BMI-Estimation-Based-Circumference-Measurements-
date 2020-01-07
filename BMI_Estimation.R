#install.packages("FactoMineR")
#install.packages("factoextra")
#install.packages("corrplot")
#install.packages("olsrr")
#install.packages("rpart")
#install.packages("rpart.plot")
#install.packages("caret")
#install.packages("rattle")
#install.packages("DAAG")
library("FactoMineR")
library("factoextra")
library("olsrr")
library("DAAG")
library("ipred")
library("rpart")
library("rpart.plot")
library("caret")
library("RColorBrewer")
library("rattle")
library("gbm")
library("corrplot")
getwd()
# setwd("/Users/husseinalilakkis/Desktop/")
my_data <- read.table("bmi.txt", header = TRUE, sep = "")
##**********************************************************************************************************
# make a dataframe out of the table
as.data.frame.matrix(my_data)
# remove rows with some empty fieds
my_data<-my_data[complete.cases(my_data),]
# print dimesions of data frame
dim(my_data)
#summarize the data
summary(my_data)
# provide the boxplot of the data frame
boxplot(my_data)
# attach BMI predictor according to the formula
my_data$BMI = (my_data$Weight*0.45)/(0.025*0.025*my_data$Height*my_data$Height)
# plot the density of some variables with ggplot
p <- ggplot(my_data, aes(x = Weight)) + geom_density(color = "darkblue", fill =" lightblue")
p + geom_vline(xintercept = mean(my_data$BMI), color = "black")
plot(p)
# identify and remove leverage points based on Cook's distance
linear_Model = lm(BMI ~ . , data = my_data)
high_Leverage = cooks.distance(linear_Model) > (4 / nrow(my_data))
my_data = my_data[!high_Leverage,]
#summarize the new data frame
summary(my_data)
# provide the boxplot of the new data frame
boxplot(my_data)
#*************************************************************************************************************
# PCA (unsupervised)
#create PCA object
pca = PCA(my_data, scale.unit = TRUE, graph = FALSE)
print(pca)
# get the eigenvalues
eig.val <- get_eigenvalue(pca)
eig.val
# plot te screeplot of dimensions
fviz_screeplot(pca, addlabels = TRUE, ylim = c(0, 50))
# Extract the results for variables
var <- get_pca_var(pca)
# Contribution of variables in the dimensions
var$coord
# Contributions of variables to PC1
fviz_contrib(pca, choice = "var", axes = 1, top = 10)
# Control variable colors using their contributions, plot default loading graph
fviz_pca_var(pca, col.var = "contrib",
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"), 
             repel = TRUE # Avoid text overlapping
)
# plot correlation of each variable with dimension
corrplot(var$contrib, is.corr=FALSE) 

##************************************************************************************************************
# split the data into train and test data
train = sample(1:nrow(my_data), nrow(my_data) * 3/4)
my_data.train = my_data[train, ]
my_data.test = my_data[-train,]
#************************************************************************************************************
#do regression trees with rpart, cp between 0 and 10
treee = rpart(BMI ~ . -Height -Weight, data = my_data.train, method = "anova", control = list(cp = 0, xval = 10))
# plot the treee
rpart.plot(treee)
fancyRpartPlot(treee)
# plot the size of the tree and error as a function of the cp
printcp(treee)
plotcp(treee)
# function that gets the minimum cp
get_cp <- function(x) {
  min    <- which.min(x$cptable[, "xerror"])
  cp <- x$cptable[min, "CP"] 
}
print(get_cp(treee))
# prune tree according to the CP that minimizes error
ptree <- prune(treee, treee$cptable[which.min(treee$cptable[,"xerror"]),"CP"])
# plot new tree
fancyRpartPlot(ptree, uniform=TRUE, main="Pruned Regression Tree")
# print training RMSE
print(get_min_error(ptree))
plotcp(ptree)
rsq.rpart(ptree)
# do the predictions on the test data
pred.tree = rpart.predict(ptree, my_data.test)
#plot predictions vs observed
plot(pred.tree, my_data.test$BMI)
# compute the test RMSE
print(sqrt(sum((pred.tree - my_data.test$BMI)^2)/length(my_data.test)))
##************************************************************************************************************
## use caret to do random forests
set.seed(123)
# use cross validation with k = 10 as training control
ctrl <- trainControl(method = "cv",  number = 10) 
#train the random forest
rf <- train(
  BMI ~ . -Height -Weight,
  data = my_data.train,
  method = "rf",
  trControl = ctrl,
  importance = TRUE
)
# print the final model established by caret
print(rf$finalModel)
print(rf)
# plot the importance of each predictor in the splits
plot(varImp(rf))
# use the test dataset to test the bagged tree
pred <- predict(rf, my_data.test)
# get the test error
RMSE(pred, my_data.test$BMI)
#*************************************************************************************************************
### we did not do this, dont include it
# stochastic gradient boosting
set.seed(99)
gbm.gbm <- gbm(BMI ~ . -Height -Weight
               , data=my_data.train
               , distribution="gaussian"
               , n.trees=1500
               , interaction.depth=3
               , n.minobsinnode=10
               , shrinkage=0.1
               , bag.fraction=0.75
               , cv.folds=10
               , verbose=FALSE
)
best.iter <- gbm.perf(gbm.gbm, method="cv")
# do predictions on our test set
train.predict <- predict.gbm(object=gbm.gbm, newdata=my_data.test, 150)
# compute and print test RMSE
rmse.gbm<-RMSE(my_data.test$BMI, train.predict)
print(rmse.gbm)
#*************************************************************************************************************

# classification random forest based on BMI high or low, ie normal or over weight
my_data$class = ifelse(my_data$BMI> 25, yes = "overweight", no = "normal")
my_data <- my_data[, -16]

## use caret to do random forests
set.seed(123)
# use cross validation with k = 10 as training control
ctrl <- trainControl(method = "cv",  number = 10) 
#train the bagged tre
rf <- train(class ~ . -Height -Weight,         
            data = my_data,
            method = "rf",
            trControl = ctrl,
            metric="Kappa",
            importance = TRUE
)
print(rf$finalModel)
print(rf)
# plot the importance of each predictor in the splits
plot(varImp(rf))
# use the test dataset to test the bagged tree
pred <- predict(rf, my_data.test)
#*************************************************************************************************************
# TO RUN RUN THIS PLEASE RUN THE CODE FROM THE BRGINING, TILL LINE 46
# base linear model
linear_Model = lm(BMI  ~. -Height - Weight, data = my_data.train)
summary(linear_Model)
# do the stepwise algorithm and check which variables are important
linear_Model2 = ols_step_both_p(linear_Model)
#plot mallows cp, adjusted R square, aic..
plot(linear_Model2)
# recreate the model with ones chosen by the stepwise algorithm based on p-val, we drop age 
linear_Model = lm(BMI ~ Chest_circumference  + Abdomen_circumference
                  + Knee_circumference
                  +Thigh_circumference +Biceps_circumference+ Forearm_circumference, data = my_data.train )
summary(linear_Model)
# showing linear model to screen for problems
par(mfrow=c(2,2))
plot(linear_Model)

# pairwise plot of training data
par(mfrow=c(1,1))
pairs(my_data.train)
# pairwise plot of BMI against the predictors
pairs(BMI ~ Chest_circumference + Hip_circumference + Abdomen_circumference
      + Knee_circumference
      +Thigh_circumference +Biceps_circumference, my_data.train)
# don our predictions on the test set
linear_Model.pred = predict(linear_Model, data = my_data.test)
# compute RMSE
RMSE(obs = my_data.test$BMI, linear_Model.pred)
# check for VIF 
ols_vif_tol(linear_Model)
##************************************************************************************************************
#Train PLS model with 10 fold cross validation
set.seed(123)
PLS_model <- train(BMI ~ Chest_circumference  + Abdomen_circumference
                   + Knee_circumference
                   +Thigh_circumference +Biceps_circumference+ Forearm_circumference
                   , my_data.train, method = "pls",
  scale = TRUE,
  trControl = trainControl("cv", number = 10),
  tuneLength = 10
)
# Plot model RMSE vs different values of components
plot(PLS_model)
# Print the best tuning parameter ncomp that
# minimize the cross-validation error, RMSE
PLS_model$bestTune

# Summarize the final model
summary(PLS_model$finalModel)
# Make predictions
predictions <- predict(PLS_model, my_data.test)
# Model performance metrics
data.frame(
  RMSE = caret::RMSE(predictions, my_data.test$BMI),
  Rsquare = caret::R2(predictions, my_data.test$BMI)
)
##************************************************************************************************************
# Train PCR model witghb 10- fold cross validation
set.seed(123)
PCR_model <- train(
  BMI ~ Chest_circumference  + Abdomen_circumference
  + Knee_circumference
  +Thigh_circumference +Biceps_circumference+ Forearm_circumference, my_data.train, method = "pcr",
  scale = TRUE,
  trControl = trainControl("cv", number = 10),
  tuneLength = 10
)
# Plot model RMSE vs different values of components
plot(PCR_model)
# Print the best tuning parameter ncomp that
# minimize the cross-validation error, RMSE
PCR_model$bestTune

# Summarize the final model
summary(PCR_model$finalModel)
# Make predictions
predictions <- predict(PCR_model, my_data.test)
# Model performance metrics
data.frame(
  RMSE = caret::RMSE(predictions, my_data.test$BMI),
  Rsquare = caret::R2(predictions, my_data.test$BMI)
)
##************************************************************************************************************
























