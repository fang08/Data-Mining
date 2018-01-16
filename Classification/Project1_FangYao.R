### This is project 1 for CIS 6930: Classification ###

### Install all packages needed
install.packages("caret")
install.packages("e1071")
install.packages("RWeka")
install.packages("ROCR")

library(caret)
library(e1071)
library(RWeka)
library(ROCR)

# read the original data
coarseData <- read.csv("classification_data.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)

# check the properties of our data
# cor(coarseData[1:4])
# 4 variables are highly correlated with each other
 
# rescale/normalize data for comparing different classifiers
# although decision tree algorithms ususally don't require rescaling
# both KNN and SVM need data normalization
# define a function to rescale all the data values range [0,1]
rescaleData <- function(x) {
  rvalue <- (x - min(x))/(max(x) - min(x))
  return (rvalue)
}
# now normalize/rescale the coarse data
normalData <- as.data.frame(lapply(coarseData[1:4], rescaleData))

# need to append the continent column
normalData <- cbind.data.frame(normalData, coarseData$Continent)
names(normalData) <- c("Rank", "Overall", "Male", "Female","Continent")

# set seed numbers and vectors to store the results
seednum <- c(25,76,39,44,81)
rs_knn <- vector('numeric')
rs_ripper <- vector('numeric')
rs_c45 <- vector('numeric')
rs_svm <- vector('numeric')
std_labels <- vector('numeric')


########################## Below is the part for tuning each model ##############################

# # define useful variables for the use of tuning
# set.seed(81) #25 76 39 44 81
# # Create index to split based on continent
# part <- createDataPartition(y = normalData$Continent, p = 0.8, list = FALSE)
# # Subset training set
# trainingSet <- normalData[part,]
# # Subset test set
# testSet <- normalData[-part,]
# 
# 
# # this part is used for tuning knn model
# for (j in 1:5) {
#   set.seed(seednum[j])
#   # create a stratified folds (=5)
#   cvIndex <- createFolds(factor(normalData$Continent), 5, returnTrain = T)
#   # define training control with cross validation
#   train_control <- trainControl(index = cvIndex, method="cv", number=5)
#   # train a knn model
#   model_knn <- train(Continent~.,data = normalData,trControl=train_control,method="knn")
#   print(model_knn)
# }
# 
# # this part is used for pruning RIPPER model
# # check parameters
# WOW("JRip")
# # default settings
# JRip(Continent~.,data = trainingSet)
# # after pruning
# JRip(Continent~.,data = trainingSet,control = Weka_control(F = 4))
# 
# # this part is used for pruning C4.5 model
# # Query J4.8 options:
# WOW("J48")
# # Learn J4.8 tree on iris data with default settings:
# J48(Continent~., data = trainingSet)
# # Learn J4.8 tree with reduced error pruning (-R) and 
# # minimum number of instances set to 5 (-M 5):
# J48(Continent~., data = trainingSet, control = Weka_control(R = TRUE, M = 5))
# 
# # this part is used for tuning svm model
# svm_tune <- tune(svm, train.x=trainingSet[,-5], train.y=trainingSet[,5], 
#                  kernel="radial", ranges=list(cost=10^(-2:2), gamma=2^(-2:2)))
# print(svm_tune)
# # tune.svm()


########################## Below is the part for define model functions ##############################

# define K-nearest neighbor function
myknn <- function(trainingSet, testSet){
  # Train a model
  model_knn <- train(trainingSet[, 1:4], trainingSet[, 5], method='knn',tuneLength = 7)
  # Predict the test set
  pred_knn <- predict.train(object=model_knn,testSet[,1:4], type="raw")
  # Confusion matrix 
  rs <- confusionMatrix(pred_knn,testSet[,5])
  # a list of overall accurcay, kappa; classification for ploting precision-recall later and f1 of each continent
  rslst <- c(as.numeric(rs$overall[1]),as.numeric(rs$overall[2]),as.numeric(rs$byClass[1:6,7]),pred_knn)
  return (rslst)
}


# define RIPPER decision tree function
myripper <- function(trainingSet, testSet){
  # train a model after tuning
  modelRipp <-JRip(Continent~.,data = trainingSet,control = Weka_control(F = 4))
  # making prediction using trained model
  pred_Ripp <- predict(modelRipp,testSet)
  # Confusion matrix
  rs <- confusionMatrix(pred_Ripp,testSet[,5])
  # a list of overall accurcay, kappa; classification for ploting precision-recall later and f1 of each continent
  rslst <- c(as.numeric(rs$overall[1]),as.numeric(rs$overall[2]),as.numeric(rs$byClass[1:6,7]),pred_Ripp)
  return (rslst)
}

# define C4.5 decision tree function
myc45 <- function(trainingSet, testSet){
  # train a model after tuning
  modelC45 <- J48(Continent~., data = trainingSet, control = Weka_control(R = TRUE, M = 5))
  # make predictions
  pred_C45 <- predict(modelC45, testSet)
  # Confusion matrix 
  rs <- confusionMatrix(pred_C45,testSet[,5])
  # a list of overall accurcay, kappa; classification for ploting precision-recall later and f1 of each continent
  rslst <- c(as.numeric(rs$overall[1]),as.numeric(rs$overall[2]),as.numeric(rs$byClass[1:6,7]),pred_C45)
  return (rslst)
}

# define Support Vector Machine function
mysvm <- function(trainingSet, testSet){
  # train a model after tuning
  svm_model <- svm(Continent~., data = trainingSet, method = "C-classification", 
                   kernel="radial", cost=100, gamma=0.25)
  # make predictions
  pred_svm <- predict(svm_model,testSet)
  # Confusion matrix 
  rs <- confusionMatrix(pred_svm,testSet[,5])
  # a list of overall accurcay, kappa; classification for ploting precision-recall later and f1 of each continent
  rslst <- c(as.numeric(rs$overall[1]),as.numeric(rs$overall[2]),as.numeric(rs$byClass[1:6,7]),as.numeric(pred_svm))
  return(rslst)
}

########################## Major part of model predictions ##############################

# main loop to run four models of 5 different training groups
for (i in 1:5) {
  # different seed number each run
  set.seed(seednum[i])
  # Create index to split based on continent
  part <- createDataPartition(y = normalData$Continent, p = 0.8, list = FALSE)
  # Subset training set
  trainingSet <- normalData[part,]
  # Subset test set
  testSet <- normalData[-part,]
  
  # add more groups of data including all the prediction results of one run
  rs_knn <- c(rs_knn,myknn(trainingSet,testSet))
  rs_ripper <- c(rs_ripper,myripper(trainingSet,testSet))
  rs_c45 <- c(rs_c45,myc45(trainingSet,testSet))
  rs_svm <- c(rs_svm,mysvm(trainingSet,testSet))
  std_labels <- c(std_labels,testSet$Continent)
}

# rs_knn
# rs_ripper
# rs_c45
# rs_svm

########################## Analysis of model prediction results ##############################

## plot precision-recall for every continent of each model

# extract prediction values from the lists of results
extractPredictionVL <- function(z){
  y <- c(z[9:51],z[60:102],z[111:153],z[162:204],z[213:255])
  return(y)
}

knn_prenum <- extractPredictionVL(rs_knn)
ripper_prenum <- extractPredictionVL(rs_ripper)
c45_prenum <- extractPredictionVL(rs_c45)
svm_prenum <- extractPredictionVL(rs_svm)

plotPR <- function(input,n) {
  # 6 figures arranged in 3 rows and 2 columns
  par(mfrow=c(3,2))
  nam <- c("Africa","Asia","Europe","North America","Oceania","South America")
  for (k in 1:6) {
    #Africa: 1, Asia: 2, Europe: 3, North America: 4, Oceania: 5, South America: 6
    correct_lb <- ifelse(std_labels == k, 1, ifelse(std_labels != k, 0, ""))
    # use ROCR to plot precision-recall
    pred <- prediction(input, correct_lb)
    perf <- performance(pred,"tpr","fpr")
    plot(perf,xlab = "Precision", ylab = "Recall", main = nam[k])
  }
  n <- paste(n, ".png",sep = "")
  dev.copy(png,n)
  dev.off()
}

plotPR(knn_prenum,"KNN model")
plotPR(ripper_prenum,"RIPPER model")
plotPR(c45_prenum,"C4.5 model")
plotPR(svm_prenum,"SVM model")


## comapre overall accuracies, kappa, F1 of four models
# extract the accuracies, kappa, F1 part from the result lists
extractAccKa <- function(a){
  b <- c(a[1:8],a[52:59],a[103:110],a[154:161],a[205:212])
  b[is.na(b)] <- 0
  b[is.nan(b)] <- 0
  acc <- matrix(b,ncol=8,byrow=TRUE)
  avg <- c(mean(acc[,1]),mean(acc[,2]),sd(acc[,1]),mean(acc[,3]),
           mean(acc[,4]),mean(acc[,5]),mean(acc[,6]),mean(acc[,7]),mean(acc[,8]))
  return(avg)
}

resultsVL <- vector('numeric')
resultsVL <- c(extractAccKa(rs_knn),extractAccKa(rs_ripper),extractAccKa(rs_c45),extractAccKa(rs_svm))
resultsVL <- matrix(data = resultsVL, ncol=9,byrow=TRUE)
colnames(resultsVL) <- c("Accuracy","Kappa", "Accuracy SD", "F1-Africa","F1-Asia","F1-Europe","F1-North America",
                         "F1-Oceania","F1-South America")
rownames(resultsVL) <- c("KNN","RIPPER","C4.5","SVM")
resultsVL


########################## This is the end of the project ##############################
