## K-Nearest Neighbors classifier 2

# import dataset
df <- read.csv("SampleData.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)
df <- df[,3:26]
# check dimension
dim(df)
head(df)
str(df)
# pre-processing:
# find columns with categorical data
for(i in which(sapply(1:24,function(x) class(df[,x]))=="character")) df[,i] <- as.factor(as.numeric(as.factor(df[,i])))
cat <- which(sapply(1:24,function(x) class(df[,x]))=="factor")

#check class by column
sapply(1:24,function(x) class(df[,x]))

# save classifications
luGen <- df$luGen

# remove categorical variable
df_nocat <- df[-cat]

# remove rows with missing data
missing <- list()
for(i in 1:ncol(df_nocat)) missing[[i]] <- which(is.na(df_nocat[,i]))
missing_row <- unique(unlist(missing))
df_nocat <- df_nocat[-missing_row,]

# rescale function
rescaleData <- function(x) {
  rvalue <- (x - min(x,na.rm=T))/(max(x,na.rm=T) - min(x,na.rm=T))
  return (rvalue)
}
# normalized all the data
df_nocat_normal <- as.data.frame(lapply(df_nocat[1:ncol(df_nocat)], rescaleData))

# add independent variable back
df_nocat_normal["luGen"] <- luGen[-missing_row]
colnames(df_nocat_normal)


# calculate mode function (to find the most likely class)
Mode <- function(x) {
  ux <- unique(x)
  return(ux[which.max(tabulate(match(x, ux)))])
}


## fast knn using kd-tree
set.seed(123)
knn_sample<- df_nocat_normal[sample(nrow(df_nocat_normal)),]
train_all <- knn_sample[1:as.integer(0.8*90746),]
test_all <- knn_sample[as.integer(0.8*90746 + 1):90746,]
train_data <- train_all[,1:21]
train_class <- train_all[,22]
test_data <- test_all[,1:21]
test_class <- test_all[,22]

install.packages("RANN")
library(RANN)

knn_predict <- function(train_data, train_class, test_data, k_value){

  # Find nearest neighbors using kd-tree
  knn_search <- RANN::nn2(data = train_data, query = test_data, k = 10, treetype = 'kd', 
                          searchtype = 'standard')
  # create label matrix for each test record
  label_mat <- matrix(train_class[knn_search$nn.idx], ncol = k_value)
  # find the most popolar class in each row
  pred <- apply(label_mat,1, Mode)
  return(pred)
}

# calculate accurcacy function
accuracy <- function(test_data, true_class){
  return(mean(test_data == true_class))
}


# calling knn_predict()
K <- 10
ptm <- proc.time()
predictions <- knn_predict(train_data, train_class, test_data, K)
proc.time() - ptm
predictions

# calling accuracy()
acc <- accuracy(predictions,as.character(test_class))
acc


## compare with other method
install.packages("caret")
install.packages("e1071")

library(caret)
library(e1071)

ptm <- proc.time()
# Train a model
model_knn <- train(train_data, train_class, method='knn',tuneLength = 10)
# Predict the test set
pred_knn <- predict.train(object=model_knn,test_data, type="raw")
proc.time() - ptm
