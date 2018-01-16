### This is CIS 6930 project 2 clustering ###

### Dataset 1:

# Install all packages needed
install.packages("rgl")
#install.packages("dendextend")
install.packages("dbscan")

library(rgl)
#library(dendextend)
library(dbscan)


# read the original data
dataset1 <- read.csv("dataset1.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)
dataset1_test <- dataset1[,-4]
# dataset standardization
dataset1_scale <- scale(dataset1_test)
# store the correct class labels
dataset1_labels <- dataset1[,4]

# since the data is in 3d, first take a look at the plot to get clues of clustering
# spinning 3D scatter plots
plot3d(dataset1$x,dataset1$y,dataset1$z, col=dataset1$cluster, size=3)
# view from the top (x-y plain), the data points look like a letter "e"
# summary(dataset1)


################ 1. Hierarchical clustering ##################
# Dissimilarity matrix
d <- dist(dataset1_scale, method = "euclidean") 
# other methods include "euclidean", "maximum", "manhattan", "canberra", "binary" or "minkowski"

# Hierarchical clustering
hc1 <- hclust(d, method = "complete") # Complete Link MAX
hc2 <- hclust(d, method = "single") # single link MIN
hc3 <- hclust(d, method = "average") # Mean link group average
hc4 <- hclust(d, method = "centroid") # Centroid linkage centroid points
hc5 <- hclust(d, method = "ward.D") # Ward's method

# Plot the obtained dendrogram
plot(hc1, cex = 0.6, hang = -1)
plot(hc2, cex = 0.6, hang = -1)
plot(hc3, cex = 0.6, hang = -1)
plot(hc4, cex = 0.6, hang = -1)
plot(hc5, cex = 0.6, hang = -1)

#function to calculate accuracy
accs <- function(re, tr){
  b <- ifelse(re == tr, 1,NA)
  c <- length(which(b==1))
  return (c/1000)
}

# store the accuracy values
hcac <- vector(mode="numeric", length=5)

# Cut tree into 8 groups/clusters
sub_grp <- cutree(hc1, k = 8)
hcac[1] <- accs(sub_grp, dataset1_labels)
sub_grp <- cutree(hc2, k = 8)
hcac[2] <- accs(sub_grp, dataset1_labels)
sub_grp <- cutree(hc3, k = 8)
hcac[3] <- accs(sub_grp, dataset1_labels)
sub_grp <- cutree(hc4, k = 8)
hcac[4] <- accs(sub_grp, dataset1_labels)
sub_grp <- cutree(hc5, k = 8)
hcac[5] <- accs(sub_grp, dataset1_labels)

# accuracies for method 1-5 are: 0.141, 0.131, 0.112, 0.124, 0.135
hcac

# plot 3d clustering results (of the last method for example)
plot3d(dataset1$x,dataset1$y,dataset1$z, col=sub_grp, size=3)

# need to import package "dendextend" but process below may take 3-5 mins
# # Create two dendrograms
# dend1 <- as.dendrogram (hc1)
# dend2 <- as.dendrogram (hc5)
# # make comparison
# tanglegram(dend1, dend2)



################ 2. K-means clustering ##################
seednum <- c(10,87,35,104,91,600,23,49,234,832)
ac <- 0
for (i in 1:10) {
  # different seed number each run
  set.seed(seednum[i])

  # K-Means Cluster Analysis
  fit <- kmeans(dataset1_scale, centers = 8, nstart = 50) 
  # centers: either the number of clusters, or a set of initial (distinct) cluster centres
  # nstart: if centers is a number, how many random sets should be chosen
  
  # calculate the sum of accuracies
  ac <- ac + accs(fit$cluster, dataset1_labels)
}
# average accuracy of 10 runs: 0.1206
ac/10

# plot 3d clustering results (of the last run for example)
plot3d(dataset1$x,dataset1$y,dataset1$z, col=fit$cluster, size=3)


################ 3. Density-based clustering ##################
### DBSCAN
# find the optimal eps value
kNNdistplot(dataset1_scale, k =  5)
abline(h = 0.3, lty = 2)

# clustering using parameters defined above
db <- dbscan(dataset1_scale, eps = 0.31, minPts = 5)
db

# accuracy: 0.117
accs(db$cluster, dataset1_labels)

# plot 3d clustering results
plot3d(dataset1$x,dataset1$y,dataset1$z, col=db$cluster + 1, size=3)



################ 4. Graph-based clustering ##################
### SNN
s <- sNNclust(dataset1_test, k = 8, eps = 0.5, minPts =8) # not scaled
# plot(dataset1_test, col = s$cluster + 1L, cex = .5)

# cluster results
table(s$cluster)

# # Jarvis Patrick clustering
# cl <- jpclust(dataset1_test, k = 8, kt = 4)
# cl
# # K: Neighborhood size for nearest neighbor sparsification.
# # Kt: threshold on the number of shared nearest neighbors (including the points themselves) to form clusters. Kt<k

# accuracy: 0.128
accs(s$cluster, dataset1_labels)

# plot 3d clustering results
plot3d(dataset1$x,dataset1$y,dataset1$z, col=s$cluster + 1, size=3)


################# This is the end of dataset 1 ##################