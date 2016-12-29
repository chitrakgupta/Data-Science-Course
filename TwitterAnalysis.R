# Data Science course final project
# Analyze data on buzz in social media
# Two datasets provided, Tom's Hardware and Twitter
# Data provided at the following link:
# https://archive.ics.uci.edu/ml/datasets/Buzz+in+social+media+
#
# This code looks at the Twitter data and performs the classification task.
# The task is to predict whether or not a given instance will create a buzz.

dat <- read.csv("Twitter-Absolute-Sigma-500.data", header=FALSE, sep=",")
# 78th column is the outcome. 0 represents "not-buzz" while 1 represents "buzz"
preds <- dat[,78]

# This plot shows the percentages of buzz and not-buzz in the given data set.
pdf("ClassLabels.pdf", 5, 5)
plot(as.factor(preds))
dev.off()

# Every primary feature is observed for 7 days, once per day. So we have 77 variables for 11 feaures. I am summing over the 7-day period. Not a perfect solution, but a good starting point to get an idea of the "overall" behavior.
ncd <- dat[,1:7]
ai <- dat[,8:14]
alNa <- dat[,15:21]
bl <- dat[,22:28]
nac <- dat[,29:35]
alNac <- dat[,36:42]
cs <- dat[,43:49]
at <- dat[,50:56]
na <- dat[,57:63]
adl <- dat[,64:70]
nad <- dat[,71:77]

processTimeSeries2 = function(x) {
 row=unlist(x)
 return(sum(x))
}
pNcd2 <- apply(ncd, 1, processTimeSeries2);
pAi2 <- apply(ai, 1, processTimeSeries2);
pAlNa2 <- apply(alNa, 1, processTimeSeries2);
pBl2 <- apply(bl, 1, processTimeSeries2);
pNac2 <- apply(nac, 1, processTimeSeries2);
pAlNac2 <- apply(alNac, 1, processTimeSeries2);
pCs2 <- apply(cs, 1, processTimeSeries2);
pAt2 <- apply(at, 1, processTimeSeries2);
pNa2 <- apply(na, 1, processTimeSeries2);
pAdl2 <- apply(adl, 1, processTimeSeries2);
pNad2 <- apply(nad, 1, processTimeSeries2);

newDat2 <- cbind(pNcd2, pAi2, pAlNa2, pBl2, pNac2, pAlNac2, pCs2, pAt2, pNa2, pAdl2, pNad2, preds)

# newDat2 has 12 columns. First 11 are the 11 primary features (summed over 7 days). And 12th column is the outcome.

# dataBeforeNorm is the data frame of newDat2. This will be used for classification tree and random forest.

dataBeforeNorm <- data.frame(newDat2)

# For k-nearest neigbor, every variable has to be in the same unit.
# Here, it is realized by normalizing all the variables in 0:1 range.
newDat2[,1] <- newDat2[,1]/max(newDat2[,1])
newDat2[,2] <- newDat2[,2]/max(newDat2[,2])
newDat2[,3] <- newDat2[,3]/max(newDat2[,3])
newDat2[,4] <- newDat2[,4]/max(newDat2[,4])
newDat2[,5] <- newDat2[,5]/max(newDat2[,5])
newDat2[,6] <- newDat2[,6]/max(newDat2[,6])
newDat2[,7] <- newDat2[,7]/max(newDat2[,7])
newDat2[,8] <- newDat2[,8]/max(newDat2[,8])
newDat2[,9] <- newDat2[,9]/max(newDat2[,9])
newDat2[,10] <- newDat2[,10]/max(newDat2[,10])
newDat2[,11] <- newDat2[,11]/max(newDat2[,11])
newDat2[,12] <- newDat2[,12]/max(newDat2[,12])

# dataToFit is the data frame of normalized newDat2. This will be used for k nearest neighbor.
dataToFit <- data.frame(newDat2)


calcError = function(estPred, actualPred) {
  estPredNum <- as.numeric(estPred) - 1
  errors <- estPredNum - actualPred
  typeIerr <- sum(errors == 1)/ length(actualPred)
  typeIIerr <- sum(errors == -1) / length(actualPred)
  bothErrors <- c(typeIerr, typeIIerr)

  bothErrors
}

library(caret)
library(class)

# Now do k-fold CV to find optimal k

# First, see how much time one round takes. Note that this will depend on the choice of v.
v = 2
k = 3
t1 <- proc.time()
set.seed(998)
inTraining <- createDataPartition(dataToFit[,12], p = (1-(1/v)), list=FALSE)
training <- dataToFit[inTraining,1:11]
testing <- dataToFit[-inTraining,1:11]
trainPreds <- dataToFit[inTraining, 12]
actualPred = dataToFit[-inTraining,12]
estPred = knn(training, testing, trainPreds, k=k, l=0, prob=FALSE, use.all=TRUE)
netTime <- proc.time()-t1
print(netTime)
# ~ 2 mins (115.643 s elapsed, 115.612 s user) with v = 2
#          (127.994 s elapsed, 127.900 s user) with v = 3
#          (105.310 s elapsed, 105.232 s user) with v = 4
#          (100.842 s elapsed, 100.704 s user) with v = 5
#          (76.147  s elapsed, 76.056  s user) with v = 10

K= 80
v = 10
err1 = rep(0, K/5)
err2 = rep(0, K/5)
t1 <- proc.time()
set.seed(998)
inTraining <- createDataPartition(dataToFit[,12], p = (1-(1/v)), list=FALSE)
training <- dataToFit[inTraining,1:11]
testing <- dataToFit[-inTraining,1:11]
trainPreds <- dataToFit[inTraining, 12]
actualPred = dataToFit[-inTraining,12]

count = 0
for (k in seq(1,K, 5)) {
  print(c("Working with k: ", k))
  count = count + 1
  estPred = knn(training, testing, trainPreds, k=k, l=0, prob=FALSE, use.all=TRUE)
  bothErrors = calcError(estPred, actualPred)
  err1[count] = bothErrors[1]
  err2[count] = bothErrors[2]
  }
}
netTime <- proc.time()-t1
print(netTime)
# Time taken for v = 10 and K = 80 with steps of 5 is 1233.724 s (user) and 1234.524 s (elapsed)

#errDat <- cbind(seq(1,K,5), err1[1:16], err2[1:16])
#colnames(errDat) <- c("K", "Type I error", "Type II error")
#write.table(errDat[seq(1,80,5),], file = "errDat.dat", sep = " ")


### Decide on k = 26
## Type I and type II errors were, respectively, 0.01528074 0.02089552

errDat <- cbind(1:K, err1, err2)
colnames(errDat) <- c("K", "Type I error", "Type II error")
write.table(errDat, file = "errDat.dat", sep = " ")

# Decision tree
library(rpart)
rpartFit <- rpart(dataBeforeNorm[,12] ~ ., data = dataBeforeNorm[,1:11], method = 'class')
summmary(rpartFit)
#Node number 2: 113220 observations
#  predicted class=0  expected loss=0.02361774  P(node) =0.8046508
#    class counts: 110546  2674
#   probabilities: 0.976 0.024 

#Node number 3: 27487 observations
#  predicted class=1  expected loss=0.08680467  P(node) =0.1953492
#    class counts:  2386 25101
#   probabilities: 0.087 0.913 

v = 10
set.seed(998)
inTraining <- createDataPartition(dataBeforeNorm[,12], p = (1-(1/v)), list=FALSE)
training <- dataBeforeNorm[inTraining,1:11]
testing <- dataBeforeNorm[-inTraining,1:11]
trainPreds <- dataBeforeNorm[inTraining, 12]
actualPred = dataBeforeNorm[-inTraining,12]
rpartFit <- rpart(trainPreds ~ ., data = training, method = 'class')
preds <- predict(rpartFit, newdata = testing, type = "class")
# Type I and type II errors, respectively, are 0.01734186 0.01982942

t1 <- proc.time()
complexityVals = c(seq(0.00001, 0.0001, length=25),
                   seq(0.0001, 0.001, length=25))

fits <- lapply(complexityVals, function(x) {
	rpartObj <- rpart(trainPreds ~ ., data = training, method = "class", control = rpart.control(cp = x) )

	preds <- predict(rpartObj, newdata = testing, type = "class")
})
netTime <- proc.time()-t1
print(netTime)

err1 <- list(length(complexityVals))
err2 <- list(length(complexityVals))
for (i in 1:length(fits)) {
  bothErrors <- calcError(fits[[i]], actualPred)
  err1[[i]] <- bothErrors[1]
  err2[[i]] <- bothErrors[2]
}
errDat <- cbind(complexityVals, err1, err2)
colnames(errDat) <- c("V", "Type I error", "Type II error")
write.table(errDat, file = "errDat-complexity.dat", sep = " ")


# Random forest
library(randomForest)

set.seed(998)
v = 10
inTraining <- createDataPartition(dataBeforeNorm[,12], p = (1-(1/v)), list=FALSE)
training <- dataBeforeNorm[inTraining,1:11]
testing <- dataBeforeNorm[-inTraining,1:11]
trainPreds <- dataBeforeNorm[inTraining, 12]
actualPred = dataBeforeNorm[-inTraining,12]

trainLabs = factor(trainPreds, labels = c("0", "1"))
forPlot.rf <- randomForest(trainLabs ~ ., training, keep.forest=FALSE, ntree=100)
plot(forPlot.rf)


# Manually finding ntree
RFmanual <- list(1:200)
for (ntree in 1:200) {
  forPlot.rf <- randomForest(trainLabs ~ ., training, keep.forest=TRUE, ntree=ntree)
  prediction <- predict(forPlot.rf, newdata = testing, type = "class")
  bothErrors <- calcError(prediction, actualPred)
  RFmanual[[ntree]] = c(ntree, bothErrors[1], bothErrors[2])
}
lapply(RFmanual, write, "RF-manual.dat", append=TRUE, ncolumns=length(RFmanual))

Fit.rf <- randomForest(trainLabs ~ ., training, keep.forest=TRUE, ntree=100)
prediction <- predict(Fit.rf, newdata = testing, type = "class")
calcError(prediction, actualPred)
pdf("FinalRFpred.pdf", 5, 5)
plot(prediction)
dev.off()



#See the effect of V
##################################################################################
#### This section is not well-developed
##################################################################################

t1<-proc.time()
V = 20
err1 = rep(0, V-1)
err2 = rep(0, V-1)
t1 <- proc.time()
for (v in 2:V) {
  print(c("Working on: ", v))
  set.seed(998)
  inTraining <- createDataPartition(dataBeforeNorm[,12], p = (1-(1/v)), list=FALSE)
  training <- dataBeforeNorm[inTraining,1:11]
  testing <- dataBeforeNorm[-inTraining,1:11]
  trainPreds <- dataBeforeNorm[inTraining, 12]
  actualPred = dataBeforeNorm[-inTraining,12]
  trainLabs = factor(trainPreds, labels = c("0", "1"))
  Fit.rf <- randomForest(trainLabs ~ ., training, keep.forest=TRUE, ntree=80)
  prediction <- predict(Fit.rf, newdata = testing, type = "class")
  bothErrors = calcError(prediction, actualPred)
  err1[v-1] = bothErrors[1]
  err2[v-1] = bothErrors[2]
}
netTime <- proc.time()-t1
print(netTime)
varyingV <- cbind(2:V, err1, err2)
colnames(varyingV) <- c("v", "Type I error", "Type II error")
write.table(varyingV, file = "varyingV.dat", sep = " ")
##################################################################################
#### The end
##################################################################################
