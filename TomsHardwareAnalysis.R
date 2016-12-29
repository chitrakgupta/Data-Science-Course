# Data Science course final project
# Analyze data on buzz in social media
# Two datasets provided, Tom's Hardware and Twitter
# Data provided at the following link:
# https://archive.ics.uci.edu/ml/datasets/Buzz+in+social+media+
#
# This code looks at the Toms Hardware data and performs the classification task.
# The task is to predict whether or not a given instance will create a buzz.

dat <- read.csv("TomsHardware-Absolute-Sigma-500.data", header=FALSE, sep=",")
# 97th column is the outcome. 0 represents "not-buzz" while 1 represents "buzz"
preds <- dat[,97]

# This plot shows the percentages of buzz and not-buzz in given data set.
pdf("ClassLabels.pdf", 5, 5)
plot(as.factor(preds))
dev.off()

# Every primary feature is observed for 8 weeks, once per week. So we have 96 variables for 12 feaures. I am summing over the 8-week period. Not a perfect solution, but a good starting point to get an idea of the "overall" behavior.
ncd <- dat[,1:8]
bl <- dat[,9:16]
nad <- dat[,17:24]
ai <- dat[,25:32]
nac <- dat[,33:40]
nd <- dat[,41:48]
cs <- dat[,49:56]
at <- dat[,57:64]
na <- dat[,65:72]
adl <- dat[,73:80]
alNa <- dat[,81:88]
alNac <- dat[,89:96]


processTimeSeries2 = function(x) {
 row=unlist(x)
 return(sum(x))
}
pNcd2 <- apply(ncd, 1, processTimeSeries2);
pBl2 <- apply(bl, 1, processTimeSeries2);
pNad2 <- apply(nad, 1, processTimeSeries2);
pAi2 <- apply(ai, 1, processTimeSeries2);
pNac2 <- apply(nac, 1, processTimeSeries2);
pNd2 <- apply(nd, 1, processTimeSeries2);
pCs2 <- apply(cs, 1, processTimeSeries2);
pAt2 <- apply(at, 1, processTimeSeries2);
pNa2 <- apply(na, 1, processTimeSeries2);
pAdl2 <- apply(adl, 1, processTimeSeries2);
pAlNa2 <- apply(alNa, 1, processTimeSeries2);
pAlNac2 <- apply(alNac, 1, processTimeSeries2);

newDat2 <- cbind(pNcd2, pBl2, pNad2, pAi2, pNac2, pNd2, pCs2, pAt2, pNa2, pAdl2, pAlNa2, pAlNac2, preds)

# newDat2 has 13 columns. First 12 are the 12 primary features (summed over 8 weeks). And 13th column is the outcome.

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
k = 1
t1 <- proc.time()
set.seed(998)
inTraining <- createDataPartition(dataToFit[,13], p = (1-(1/v)), list=FALSE)
training <- dataToFit[inTraining,1:12]
testing <- dataToFit[-inTraining,1:12]
trainPreds <- dataToFit[inTraining, 13]
actualPred = dataToFit[-inTraining,13]
estPred = knn(training, testing, trainPreds, k=k, l=0, prob=FALSE, use.all=TRUE)
netTime <- proc.time()-t1
print(netTime)
# 0.436 s (user) and 0.444 s (elapsed) for v = 2.

K= 20
V = 10
err1 = rep(0, K)
err2 = rep(0, K)
t1 <- proc.time()
for (v in 2:V) {
  set.seed(998)
  inTraining <- createDataPartition(dataToFit[,13], p = (1-(1/v)), list=FALSE)
  training <- dataToFit[inTraining,1:12]
  testing <- dataToFit[-inTraining,1:12]
  trainPreds <- dataToFit[inTraining, 13]
  actualPred = dataToFit[-inTraining,13]

  count = 0
  for (k in 1:K) {
    count = count + 1
    estPred = knn(training, testing, trainPreds, k=k, l=0, prob=FALSE, use.all=TRUE)
    bothErrors = calcError(estPred, actualPred)
    err1[count] = err1[count] + bothErrors[1]
    err2[count] = err2[count] + bothErrors[2]
  }
}
netTime <- proc.time()-t1
print(netTime)
# 34.080 (user) and 34.116 (elapsed) for V = 10, K = 10

errDat <- cbind(1:K, err1, err2)
colnames(errDat) <- c("K", "Type I error", "Type II error")
write.table(errDat, file = "errDat.dat", sep = " ")


# Vary only k
v = 10
K = 20
err1 = rep(0, K)
err2 = rep(0, K)
t1 <- proc.time()
for (v in 2:V) {
  set.seed(998)
  inTraining <- createDataPartition(dataToFit[,13], p = (1-(1/v)), list=FALSE)
  training <- dataToFit[inTraining,1:12]
  testing <- dataToFit[-inTraining,1:12]
  trainPreds <- dataToFit[inTraining, 13]
  actualPred = dataToFit[-inTraining,13]

  count = 0
  for (k in 1:K) {
    count = count + 1
    estPred = knn(training, testing, trainPreds, k=k, l=0, prob=FALSE, use.all=TRUE)
    bothErrors = calcError(estPred, actualPred)
    err1[k] = bothErrors[1]
    err2[k] = bothErrors[2]
  }
}
netTime <- proc.time()-t1
print(netTime)
# 34.320 s (user) 34.373 s (elapsed)

errDat <- cbind(1:K, err1, err2)
colnames(errDat) <- c("K", "Type I error", "Type II error")
write.table(errDat, file = "errDat-onlyK.dat", sep = " ")
# Looks like kNN might not work for this data

# But still do a bigger scan, just for the heck of it
v = 10
K = 80
err1 = rep(0, K)
err2 = rep(0, K)
t1 <- proc.time()
for (v in 2:V) {
  set.seed(998)
  inTraining <- createDataPartition(dataToFit[,13], p = (1-(1/v)), list=FALSE)
  training <- dataToFit[inTraining,1:12]
  testing <- dataToFit[-inTraining,1:12]
  trainPreds <- dataToFit[inTraining, 13]
  actualPred = dataToFit[-inTraining,13]

  count = 0
  for (k in 1:K) {
    count = count + 1
    estPred = knn(training, testing, trainPreds, k=k, l=0, prob=FALSE, use.all=TRUE)
    bothErrors = calcError(estPred, actualPred)
    err1[k] = bothErrors[1]
    err2[k] = bothErrors[2]
  }
}
netTime <- proc.time()-t1
print(netTime)

errDat <- cbind(1:K, err1, err2)
colnames(errDat) <- c("K", "Type I error", "Type II error")
write.table(errDat, file = "errDat-onlyK-80.dat", sep = " ")


# Decision tree
library(rpart)
rpartFit <- rpart(dataBeforeNorm[,13] ~ ., data = dataBeforeNorm[,1:12], method = 'class')
summary(rpartFit)
#Node number 2: 3071 observations
#  predicted class=0  expected loss=0.04558776  P(node) =0.3884883
#    class counts:  2931   140
#   probabilities: 0.954 0.046 

#Node number 3: 4834 observations
#  predicted class=1  expected loss=0.02358295  P(node) =0.6115117
#    class counts:   114  4720
#   probabilities: 0.024 0.976 

v = 10
set.seed(998)
inTraining <- createDataPartition(dataBeforeNorm[,13], p = (1-(1/v)), list=FALSE)
training <- dataBeforeNorm[inTraining,1:12]
testing <- dataBeforeNorm[-inTraining,1:12]
trainPreds <- dataBeforeNorm[inTraining, 13]
actualPred = dataBeforeNorm[-inTraining,13]
rpartFit <- rpart(trainPreds ~ ., data = training, method = 'class')
preds <- predict(rpartFit, newdata = testing, type = "class")
# Type I and type II errors, respectively, are 0.01898734 0.02025316

t1 <- proc.time()
complexityVals = c(seq(0.00001, 0.0001, length=5),
                   seq(0.0001, 0.001, length=35), 
		   seq(0.001, 0.002, length=9))

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

#library(rpart.plot)
#prp(rpartFit, extra = 1)


# Scan over V

#V = 20
#err1 = rep(0, V-1)
#err2 = rep(0, V-1)
#for (v in 2:V) {
#  set.seed(998)
#  inTraining <- createDataPartition(dataBeforeNorm[,13], p = (1-(1/v)), list=FALSE)
#  training <- dataBeforeNorm[inTraining,1:12]
#  testing <- dataBeforeNorm[-inTraining,1:12]
#  trainPreds <- dataBeforeNorm[inTraining, 13]
#  actualPred = dataBeforeNorm[-inTraining,13]  
#  rpartFit <- rpart(trainPreds ~ ., data = training, method = 'class')
#  preds <- predict(rpartFit, newdata = testing, type = "class")
#  bothErrors <- calcError(preds, actualPred)
#  err1[v-1] <- bothErrors[1]
#  err2[v-1] <- bothErrors[2]
#}
#errDat <- cbind(2:V, err1, err2)
#colnames(errDat) <- c("V", "Type I error", "Type II error")
#write.table(errDat, file = "errDat-forV-decisionTree.dat", sep = " ")




# Random forest
library(randomForest)
rf.cv <- rfcv(dataBeforeNorm[,1:12], dataBeforeNorm[,13], fold = 10)
# Time taken: 135.476 (user), 135.493 (elapsed)
with(rf.cv, plot(n.var, error.cv, xlab="Number of variables", ylab="Error", pch=16, type="o"))

set.seed(998)
inTraining <- createDataPartition(dataBeforeNorm[,13], p = 0.7, list=FALSE)
training <- dataBeforeNorm[inTraining,1:12]
testing <- dataBeforeNorm[-inTraining,1:12]
trainPreds <- dataBeforeNorm[inTraining, 13]
actualPred = dataBeforeNorm[-inTraining,13]

##################################################################################
#### This section is not well-developed because the analysis wasn't going anywhere
##################################################################################
trainLabs = factor(trainPreds, labels = c("0", "1"))
forPlot.rf <- randomForest(trainLabs ~ ., training, keep.forest=FALSE, ntree=100)
plot(forPlot.rf)

# Find importance of features in terms of mean decrease in gini coefficient
imp <- importance(forPlot.rf)
ordering <- order(imp)

rfDat <- cbind(rownames(imp)[ordering], imp[ordering])
# I know it is 6 from the rfcv plot
Need <- c(rownames(imp)[ordering[7:12]], "preds")
subset <- dataBeforeNorm[,Need]
#names(subset) <- c("ALNAC-c", "ALNA-c", "ADL-c", "NAD-c", "AT-c", "ND-c", "preds")
subsettraining <- subset[inTraining,1:6]
subsettesting <- subset[-inTraining,1:6]
subsettrainPreds <- subset[inTraining, 7]
subsetactualPred = subset[-inTraining,7]
subsettrainLabs = factor(subsettrainPreds, labels = c("0", "1"))
subset.rf <- randomForest(subsettrainLabs ~ ., subsettraining, keep.forest=TRUE, ntree=100)
subsetprediction <- predict(subset.rf, newdata = subsettesting, type = "class")
rpartFit <- rpart(subsettrainLabs ~ ., subsettraining, method="class")
prp(rpartFit, extra=1)
##################################################################################
##### End of not-well-developed section
##################################################################################

# Manually finding ntree
RFmanual <- list(1:200)
for (ntree in 1:200) {
  subset.rf <- randomForest(subsettrainLabs ~ ., subsettraining, keep.forest=TRUE, ntree=ntree)
  subsetprediction <- predict(subset.rf, newdata = testing, type = "class")
  bothErrors <- calcError(subsetprediction, subsetactualPred)
  RFmanual[[ntree]] = c(ntree, bothErrors[1], bothErrors[2])
}
lapply(RFmanual, write, "RF-manual.dat", append=TRUE, ncolumns=length(RFmanual))

Fit.rf <- randomForest(trainLabs ~ ., training, keep.forest=TRUE, ntree=80)
prediction <- predict(Fit.rf, newdata = testing, type = "class")
calcError(prediction, actualPred)
pdf("FinalRFpred.pdf", 5, 5)
plot(prediction)
dev.off()

# type I: 0.01602699
# type II: 0.01940110


#See the effect of V
##################################################################################
#### This section is also not well-developed
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
  training <- dataBeforeNorm[inTraining,1:12]
  testing <- dataBeforeNorm[-inTraining,1:12]
  trainPreds <- dataBeforeNorm[inTraining, 13]
  actualPred = dataBeforeNorm[-inTraining,13]
  trainLabs = factor(trainPreds, labels = c("0", "1"))
  Fit.rf <- randomForest(trainLabs ~ ., training, keep.forest=TRUE, ntree=100)
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
