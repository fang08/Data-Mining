### This is CIS 6930 project 2 clustering ###

### Dataset 2:

dataset2 <- read.csv("dataset2.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)
# need to standardize dataset
dataset2.scaled <- scale(dataset2)
# summary(dataset2.scaled)

# use Elbow method to determine optimal Clusters
# this may take more than half an hour, please run line #22 instead
set.seed(1234)
k.max <- 25
wss <- sapply(5:k.max, function(k){kmeans(dataset2.scaled,k,nstart = 20,iter.max = 1000)$tot.withinss})
wss

# fviz_nbclust(dataset2.scaled, kmeans, method = "wss")
# Error: cannot allocate vector of size 3771.5 Gb

# wss was saved for re-use, please run readRDS line
saveRDS(wss,"wss.rds")
wss <- readRDS("wss.rds")

plot(5:k.max, wss, type = "b", pch = 19, frame = FALSE,
     xlab = "numer of clusters k", ylab = "total within sum of squared errors")
abline(v = 15, lty =2)

# k-means clustering
# this may take 3 to 5 mins, please run line #35 instead
fit2 <- kmeans(dataset2.scaled, centers = 15, nstart = 10)
fit2

# clustering results was saved for re-use, please run readRDS line
saveRDS(fit2, "fit2.rds")
fit2 <- readRDS("fit2.rds")
fit2

### the end ###
