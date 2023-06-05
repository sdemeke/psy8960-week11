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
#hours variables like HRS1, HRS2, USUALHRS, and LEASHRS. Project instructions said to remove two
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
#and setting the trainControl() custom settings. Now my ml_function only takes in the training 
#data, testing data, and the name of the ml model as parameters.
#I also added Sys.time() lines to capture the time it takes for caret::train() to run for each 
#model. The results tibble also includes a new column to store this time variable using the difftime()
#function to set fixed seconds unit.


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


ml_function <- function(train_data=train_dat, test_data=test_dat, ml_model =  c("lm","glmnet","ranger","xgbTree")) { 
  

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
    ho_rsq = cor(predicted, test_data$workhours),
    no_seconds = difftime(end,start,units="secs")
  )
  
  return(results)
  
}


#In project 10, I used a for loop to iterate over the different models and run ml_function() on
#each. After comparing the runtime of different approaches, I chose to use mapply() for this 
#purpose instead of the for loop. This was true even when I fixed my fatal error in project 10 
#of growing an empty list with no pre-defined length. 
#mapply() also returns a list/matrix object if SIMPLIFY=TRUE. To return a list that I can rbind
#into a dataframe, I set SIMPLIFY=FALSE and call rbind to collapse the list to a dataframe.

ml_methods <- c("lm","glmnet","ranger","xgbTree")  

ml_results_norm <- mapply(ml_function, SIMPLIFY = FALSE, ml_model=ml_methods)
ml_results_norm_df <- do.call("rbind", ml_results_norm)



#To run the parallelized version of the ml models, I repeat the same code as above and
#made no changes to ml_function(). I just called the parallelizing functions from
#parallel/doParallel. For this local run, I set the number of clusers to one less than
#the number of cores on the local machine (in my case, detectCores() returns 8). I leave
#one out so that other local processes can continue like running my browser to view DataCamp
Sys.sleep(2)
local_cluster <- makeCluster(detectCores()-1)
registerDoParallel(local_cluster)
ml_results_prll <- mapply(ml_function, SIMPLIFY = FALSE, ml_model=ml_methods)
stopCluster(local_cluster)
registerDoSEQ()

ml_results_prll_df <- do.call("rbind", ml_results_prll)



#Publication

#No changes to table1_tbl code except for deselecting no_seconds variable

#Added table2_tbl code to pull number of seconds for each of the 8 model runs
#from the final model result dataframes.

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
# 1 OLS Regression            .13    .06   
# 2 Elastic Net               .85    .57   
# 3 Random Forest             .92    .62   
# 4 eXtreme Gradient Boosting .97    .59 

#prll results
# # A tibble: 4 × 4
# model_name cv_rsq ho_rsq no_seconds     
# * <chr>       <dbl>  <dbl> <drtn>         
# 1 lm          0.125 0.0633  10.452153 secs
# 2 glmnet      0.860 0.573    5.262019 secs
# 3 ranger      0.919 0.653   74.395088 secs
# 4 xgbTree     0.941 0.580  114.382495 secs

table2_tbl <- tibble(
  algo = c("OLS Regression","Elastic Net","Random Forest", 
           "eXtreme Gradient Boosting"),
  original = round(ml_results_norm_df$no_seconds,2),
  parallelized = round(ml_results_prll_df$no_seconds,2)
)

# # A tibble: 4 × 3
# algo                      original        parallelized   
# <chr>                     <drtn>          <drtn>         
# 1 OLS Regression              4.403648 secs  10.452153 secs
# 2 Elastic Net                 9.504120 secs   5.262019 secs ~45% faster
# 3 Random Forest             104.215855 secs  74.395088 secs ~30% faster
# 4 eXtreme Gradient Boosting 213.264331 secs 114.382495 secs ~47% faster


##Answers to Questions
#1. The elastic net and extreme gradient boost models both improved about 45% in decreased runtime. Random forest
#improved by about 30%. The OLS regression model increased in runtime by more than double. Parallelization can be
#a tradeoff for overhead processing and while the more complex models benefited from parallelization, the simple 
#OLS regression did not. For the OLS model, the actual model code is already quite quick so when I add the added
#burden of creating multiple clusters, the computation involved in managing the different threads ends up increasing
#the model runtime. For the more complex models, however, the sequential runtime for each model run is much more 
#computationally expensive and the added overhead of parallelizing does not counteract the increase in efficiency
#gained by running the expensive models over more clusters.

#2. The fastest parallelization model was ~5 seconds for elastic net while the slowest parallelized model was the
#extreme gradient boost model at 114 seconds, giving a difference of about 109 seconds.
#WHY

#3. I would recommend the Random Forest model. The rsquared results show that this model had a high cross-validated
#as well as holdout Rsquared compared to the other models, only equivalent to the extreme gradient boost model which 
#was significantly slower both in the original and the parallelized computation. So the random forest is as accurate 
#as the most complex model tested but still efficient in computation time which presents a useful balance for the 
#application of machine learning models.

