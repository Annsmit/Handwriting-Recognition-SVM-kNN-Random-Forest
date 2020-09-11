## LIBRARIES
library(ggplot2)
#install.packages("rpart")
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(class)
library(lattice)
library(e1071)
library(mlr)
library(naivebayes)
library(corrplot)

##Introduction
Digitally identifying handwritten images is one of the many uses of machine learning. The Kaggle Digit Recognizer competition contains pre-split datasets of a training and test set of handwritten images that have been digitized by pixilation. Handwritten images from zero to nine and their corresponding pixels which range from 0 to 255 to formulate the composition of each image and is therefore identifiable by data mining techniques. The classification task is to use machine learning techniques for pattern recognition to learn and classify the nine images. In this analysis, visualizations of the handwritten images as well as the five machine learning classification algorithms are explored.

#read in files
test_digit <- read.csv("data/Kaggle-digit-test-sample1000.csv", 
                       header = TRUE, sep = ",", stringsAsFactors = FALSE)
train_digit <- read.csv("data/Kaggle-digit-train-sample-small-1400.csv",
                        header = TRUE, sep = ",", stringsAsFactors = FALSE)

##Data pre-processing and preparation
After reading in the data using a csv, a quick check of the dimensions and structure was performed to verify the number of rows and columns as well as the data type within the dataset. These lines of codes verified the test set included 1000 rows and 785 columns and the training set included 1400 rows and 785 columns. A plot of bars confirms the distribution of the label class among the training set and a chunk of code to visualize the digits.

After validation of the dataset, the next chunk of code includes steps used to convert the data into formats needed for analysis. This includes converting the label column to a factor which is required in R to categorize the data. The data is then descretized for use on Naive Bayes modelling. 

##Data Visualization

#Check dimensions
test <- dim(test_digit)
train <- dim(train_digit)
(Dim <- data.frame(Test=test, Train=train))
#View levels
(table(train_digit$label))
plot(train_digit$label, col=rainbow(10,.4))



#Visualize the dataset
set.seed(123)
data <- sample(row.names(train_digit),100)
par(mfrow=c(10,10),mar=c(0.1,0.1,0.1,0.1))

for (k in data)
{
  row <- NULL
  for (n in 2:785)
    row[n-1] <- train_digit[k,n]
  
  matrix1 <- matrix(row,28,28,byrow=FALSE)
  matrix2 <- matrix(rep(0,784),28,28)
  
  for (i in 1:28)
    for (j in 1:28)
      matrix2[i,28-j+1] <- matrix1[i,j]
  if (k==26611)
    image(matrix2, axes=FALSE, col=heat.colors(12))
  else
    image(matrix2, axes=FALSE, col=topo.colors(12))
}
rm(i,n,j,k,row,matrix1,matrix2)



## Change class label into a factor
library(dplyr)
test_digit$label<- as.factor(test_digit$label)
train_digit$label<- as.factor(train_digit$label)
# Drop variables
cleantest <- na.omit(test_digit)
cleantrain <- na.omit(train_digit)
cleantest <- mutate_if(cleantest, is.integer, as.numeric)
cleantrain <- mutate_if(cleantrain, is.integer, as.numeric)

## Step 1: Create a new variable
train_digit$pixel97rank <- NULL
## Step 1: Create a new variable
test_digit$pixel97rank <- NULL
#descretize training set
train_digit$pixel97rank <- 
  cut(train_digit$pixel97, breaks=c(-Inf, 50, 100, 150, 200, Inf))
labels=c("1","2","3","4","5")
test_digit$pixel97rank <- 
  cut(test_digit$pixel97, breaks=c(-Inf, 50, 100, 150, 200, Inf))
labels=c("1","2","3","4","5")
#remove column pixel 97
train_digitrank <- train_digit[,-c(785)]
test_digitrank <- test_digit[,-c(785)]

##Analysis

#NAIVE BAYES using laplace
Bayes_digit <- naive_bayes(label ~., data=train_digitrank, laplace = 1)
## See the prediction probabilities...
pred_digit1<-predict(Bayes_digit, test_digitrank, type="class")
Pred_TABLE1 <- table(Predicted=pred_digit1, test_digit$label)
#model accuracy
NB_Accuracy <- sum(diag(Pred_TABLE1)/sum(Pred_TABLE1))


#DECISION TREE
Treefit <- rpart(cleantrain$label~., data = cleantrain[-1], method="class")
predicted= predict(Treefit,cleantest, type="class")
table_dt <- table(cleantest$label, predicted)
DT_Accuracy <- sum(diag(table_dt))/sum(table_dt)


#SVM - polynomial
SVM_Fed_fit_P <- svm(label ~., data=cleantrain, kernel="polynomial", cost=.1)
pred_Fed_train <- predict(SVM_Fed_fit_P, cleantest, type="class")
table_mat1 <- table(cleantest$label, pred_Fed_train)
SVM_Accuracy <- sum(diag(table_mat1)) / sum(table_mat1)


#KNN
# get a guess for k
k_guess <- round(sqrt(nrow(cleantrain)))
library(class)
cleantest1 <- mutate_if(cleantest, is.integer, as.numeric)
cleantrain1 <- mutate_if(cleantrain, is.integer, as.numeric)
fit_train <- class::knn(train = cleantrain[-1], test=cleantrain[-1], 
                        cl=cleantrain$label, k = k_guess, prob=F)
# Check the classification accuracy
table_matk <- (table(fit_train, cleantrain1$label))
accuracy_k <- sum(diag(table_matk)) / sum(table_matk)
## Test it on the test set now
fit_train2 <- class::knn(train=cleantrain[-1], test=cleantest[-1], 
                         cl=cleantrain$label, k = 10, prob=F)
table_mat_k2 <- (table(fit_train2, cleantest$label))
KNN_Accuracy <- sum(diag(table_mat_k2)) / sum(table_mat_k2)


#RANDOM FOREST
library(randomForest)
rforest <- randomForest(cleantrain$label~., data = cleantrain, ntree=37)
rforpred <- predict(rforest, cleantest)
(table(cleantest$label, rforpred))
table_matr <- (table(rforpred, cleantest$label))
RF_Accuracy <- sum(diag(table_matr)) / sum(table_matr)


##Conclusion
The five models used in this analysis include Naive Bayes, Decision Tree, Support Vector Model, K Nearest Neighbor, and Random Forest. The model that produced the best accuracy included Decision Tree. The Decision Tree model appeared to produce the best results with most accuracy. The table below outlines each models accuracy. This is not surprising as in general Decision Tree models tend to have very good accuracy. According to this analysis, Decision Tree should be used to classify the images. 


Accuracy <- data.frame(NaiveBayes = round((NB_Accuracy),3), 
                       DecisionTree = round((DT_Accuracy),3), 
                       SVM = round((SVM_Accuracy),3), 
                       KNN = round((KNN_Accuracy),3), 
                       RandomForest = round((RF_Accuracy),3))
#install.packages("pander")
library(pander)
pandoc.table(Accuracy)