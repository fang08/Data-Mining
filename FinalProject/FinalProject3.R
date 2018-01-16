## K-Nearest Neighbors classifier - full data
install.packages("data.table")
library(data.table)

# read txt data
df_large <- fread("ProjectData.txt", header = TRUE, sep = ",", stringsAsFactors = FALSE)
df_large <- as.data.frame(df_large)
head(df_large)
dim(df_large)  #  9328622  19

# pre-processing:
# find columns with categorical data
cat <- c(1,12,14,16,17)
# remove categorical variable
df_nocat <- df_large[-cat]
# save classifications
luGen <- df_large$luGen
unique(luGen)

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
train_all <- knn_sample[1:as.integer(0.8*9129694),]
test_all <- knn_sample[as.integer(0.8*9129694 + 1):9129694,]
train_data <- train_all[,1:14]
train_class <- train_all[,15]
test_data <- test_all[,1:14]
test_class <- test_all[,15]

install.packages("RANN")
library(RANN)

knn_predict <- function(train_data, train_class, test_data, k_value){
  
  # Find nearest neighbors using kd-tree
  knn_search <- RANN::nn2(data = train_data, query = test_data, k = k_value, treetype = 'kd', 
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
K <- 8
ptm <- proc.time()
predictions <- knn_predict(train_data, train_class, test_data, K)
proc.time() - ptm
predictions

# runtime
# user     system    elapsed 
# 163.908  50.040    559.071

# save the result
saveRDS(predictions,"predictions.rds")
prds <- readRDS("predictions.rds")

# calling accuracy()
acc <- accuracy(predictions,as.character(test_class))
acc  # 0.9048906
