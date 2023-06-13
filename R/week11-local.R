#Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(haven)
library(caret)
library(parallel)
library(doParallel)

set.seed(24)


#Data Import and Cleaning

#Same as for project 10 plus added line removing other two work hours variables. There are more 
#hours variables like HRS1, HRS2, USUALHRS, and LEASTHRS. Project instructions said to remove 2
#other variables only. HRS2 is already removed after filtering 75% missingness so I am electing 
#to remove USUALHRS and HRS1 because these two correlate higher (>.75) with MOSTHRS.

gss_tbl <- read_sav("../data/GSS2016.sav") %>%  
  rename(workhours = MOSTHRS) %>% 
  drop_na(workhours) %>% 
  mutate(workhours = as.numeric(workhours)) %>% 
  select(which(colMeans(is.na(.)) < 0.75))  %>% 
  select(-USUALHRS,-HRS1)


#Visualization

#No changes from project 10

gss_tbl %>% 
  ggplot(aes(x=workhours)) + geom_histogram()


#Analysis

#Many changes from project 10. First, I took out some code from the function so as not to run 
#the same lines repeatedly and increase code efficiency. These include the universal settings 
#for the machine learning models like setting the folds, creating the train/testing data sets,
#and setting the trainControl() custom settings. Now my function (renamed) only takes in the training 
#data, testing data, and the name of the ml model as parameters.
#I also added Sys.time() lines to capture the time it takes for caret::train() to run for each 
#model. The results tibble also includes a new column to store this time variable using the 
#difftime() function to set fixed seconds unit.
#I fixed error in calculating holdout Rsquared (project 10 only calculated r not r^2)


no_folds <- 10
cv_index <- createDataPartition(gss_tbl$workhours, p = 0.75, list = FALSE)
train_dat <- gss_tbl[cv_index,] 
test_dat <- gss_tbl[-cv_index,] 

fold_indices <- createFolds(train_dat$workhours, k = no_folds)

myControl <- trainControl(
  method = "cv", 
  number = no_folds, 
  verboseIter = TRUE,
  indexOut = fold_indices 
)


getMLResults <- function(train_data=train_dat, test_data=test_dat, ml_model =  c("lm","glmnet","ranger","xgbTree")) { 
  
  set.seed(24)
  start <- Sys.time()
  
  model <- train(
    workhours~.,
    data = train_data, 
    metric = "Rsquared",
    method = ml_model,
    preProcess = c("center","scale","nzv","medianImpute"), 
    na.action = na.pass,
    trControl = myControl
  )
  
  end <- Sys.time()
  
  predicted <- predict(model, test_data, na.action = na.pass)
  
  results <- tibble(
    model_name = ml_model,
    cv_rsq = max( model[["results"]][["Rsquared"]]),
    ho_rsq = cor(predicted, test_data$workhours)^2,
    no_seconds = difftime(end,start,units="secs")
  )
  
  return(results)
  
  
}


#In project 10, I used a for loop to iterate over the different models and run my function on
#each. After comparing the runtime of different approaches, I chose to use mapply() for this 
#purpose instead of the for loop. Mapply() was faster even when I fixed my fatal error in 
#project 10 of growing an empty list with no pre-defined length. 
#mapply() returns a list/matrix object if SIMPLIFY=TRUE. To return a list that I can rbind
#into a dataframe, I set SIMPLIFY=FALSE and call rbind

ml_methods <- c("lm","glmnet","ranger","xgbTree")  

#Normal
ml_results_norm <- mapply(getMLResults, SIMPLIFY = FALSE, ml_model=ml_methods)

ml_results_norm_df <- do.call("rbind", ml_results_norm)



#Parallelized
#To run the parallelized version of the ml models, I repeat the same code as above and
#made no changes to getMLResults(). I just called the parallelizing functions from
#parallel/doParallel. For this local run, I set the number of clusers to one less than
#the number of cores on the local machine (in my case, detectCores() returns 8). I leave
#one out so that other local processes can continue like running my browser.
#I also add a brief pause using Sys.sleep() because I noticed that the order in which I ran
#the normal vs parallelized code chunks changed the runtime results and adding a buffer 
#helped reduce the imbalance.
Sys.sleep(5)

local_cluster <- makeCluster(detectCores()-1)
registerDoParallel(local_cluster)

ml_results_prll <- mapply(getMLResults, SIMPLIFY = FALSE, ml_model=ml_methods)
ml_results_prll_df <- do.call("rbind", ml_results_prll)

stopCluster(local_cluster)
registerDoSEQ()


#Publication

#No changes to table1_tbl code except for deselecting no_seconds variable

#Added table2_tbl code to pull number of seconds for each of the 8 model runs
#from the final model result data frames.

table1_tbl <- ml_results_norm_df  %>% 
  mutate(algo = c("OLS Regression","Elastic Net","Random Forest", 
                  "eXtreme Gradient Boosting"),
         .before = cv_rsq)  %>% 
  select(-c(model_name, no_seconds)) %>% 
  mutate(across(ends_with("_rsq"),
                \(x) gsub("0\\.",".",
                          format(round(x, digits=2), nsmall = 2)) ) )

# # A tibble: 4 × 3
# algo                      cv_rsq ho_rsq
# <chr>                     <chr>  <chr> 
# 1 OLS Regression            .14    .01   
# 2 Elastic Net               .81    .31   
# 3 Random Forest             .92    .39   
# 4 eXtreme Gradient Boosting .95    .33 

table2_tbl <- tibble(
  algo = c("OLS Regression","Elastic Net","Random Forest", 
           "eXtreme Gradient Boosting"),
  original = round(ml_results_norm_df$no_seconds,2),
  parallelized = round(ml_results_prll_df$no_seconds,2)
)

# A tibble: 4 × 3 - usual order with sys.sleep
# algo                      original    parallelized
# <chr>                     <drtn>      <drtn>      
# 1 OLS Regression              5.18 secs  10.42 secs 
# 2 Elastic Net                11.40 secs   6.04 secs 
# 3 Random Forest              73.97 secs  79.25 secs 
# 4 eXtreme Gradient Boosting 255.27 secs 124.73 secs 


##Answers to Questions

#1. The elastic net and extreme gradient boost models both improved ~40% in decreased runtime. Random forest
#increased in time by ~5secs. The OLS regression model increased in runtime by about double. Parallelization is
#a trade-off for overhead processing and while 2 of the complex models benefited from parallelization, the simple 
#OLS regression did not and neither did random forest. For the OLS model, the actual model code is already quick so
#when I add the add the burden of creating multiple clusters, the computation involved in managing the different 
#threads ends up increasing runtime. For elastic net and xgb, however, the sequential runtime for each model run
#is more computationally expensive and the added overhead of parallelizing does not counteract the increase in 
#efficiency gained by running the models over more clusters. I did not notice any substantial change in runtime 
#for random forest during my tests (sometimes slower, sometimes faster) but this result may differ with larger
#datasets such that parallelization improves the speed if we were working with larger N/k.

#2. The fastest parallelization model was ~6 seconds for elastic net while the slowest parallelized model was the
#extreme gradient boost model at 125 seconds, giving a difference of about 2 minutes. This may be due to the 
#different contingencies between elastic net and extreme gradient boosting models. For elastic net, parallelization
#can effectively run independently across cores. For the latter, however, the model includes processes which require
#communication between the results on different cores. This additional burden may produce a threshold on how fast
#the parallelized process can be in comparison to elastic net.

#3. I would recommend the Random Forest model. Computationally, this model was faster than the parallelized extreme 
#gradient boosting when run without parallelization and had the next highest cross-validated and the highest holdout
#Rsquared among all models. While Elastic net was still much faster, it may not be advisable to base judgement entirely
#on only runtime as I also noticed that my own local machine was running random forest slower than other machines
#and a more powerful machine could decrease that runtime while still benefiting from the increased accuracy
#of random forest.