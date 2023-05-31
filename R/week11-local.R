#Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(haven)
library(caret)
library(parallel)
library(doParallel)


#Data Import and Cleaning

#Same as for project 10, added line removing other two work hours variables

gss_tbl <- read_sav("../data/GSS2016.sav") %>%  
  rename(workhours = MOSTHRS) %>% 
  drop_na(workhours) %>% 
  mutate(workhours = as.numeric(workhours)) %>% 
  select(which(colMeans(is.na(.)) < 0.75))  %>% 
  select(-USUALHRS,-LEASTHRS)

#Remove the other two “work hours” variables from your model (do not predict work hours from work hours).
#HRS1 - worked last week
#HRS2 - uusually work week
#USUALHRS - usually work
#MOSTHRS - most hrs/week worked in past month
#LEASTHRS - fewest hrs/week in past month
#does he mean remove USUALHRS and LEASTHRS?

#Visualization

gss_tbl %>% 
  ggplot(aes(x=workhours)) + geom_histogram()


#Analysis

#Edited to add Sys.time() to time, added new col to results
#added parallelize arg
#removed other nonessential parameters

ml_function <- function( ml_model =  c("lm","glmnet","ranger","xgbTree"), parallelize=FALSE) { 
  
  dat <- gss_tbl
  no_folds <- 10
  
  ml_model <- match.arg(ml_model)
 # parallelize <- FALSE
  
#if parallel-->
if(parallelize == TRUE) {
local_cluster <- makeCluster(detectCores()-1)
registerDoParallel(local_cluster)
}
  start <- Sys.time() #moved start time to after parallel registration
  #does this change results?
  
  set.seed(24)
  cv_index <- createDataPartition(dat$workhours, p = 0.75, list = FALSE)
  train_dat <- dat[cv_index,] 
  test_dat <- dat[-cv_index,] 
  
  fold_indices <- createFolds(train_dat$workhours, k = no_folds)
  
  
  model <- caret::train(
    workhours~.,
    data = train_dat, 
    metric = "Rsquared",
    method = ml_model,
    preProcess = c("center","scale","nzv","medianImpute"), 
    na.action = na.pass,
    trControl = trainControl(
      method = "cv", 
      number = no_folds, 
      verboseIter = TRUE,
      indexOut = fold_indices 
    )
  )
  
  predicted <- predict(model, test_dat, na.action = na.pass)
  
end <- Sys.time()

if(parallelize==TRUE) {
  stopCluster(local_cluster)
  registerDoSEQ()
  }

  # results <- tibble(
  #   model_name = ml_model,
  #   cv_rsq = max( model[["results"]][["Rsquared"]]),
  #   ho_rsq = cor(predicted, test_dat$workhours),
  #   no_seconds_og = difftime(end,start,units="secs")
  # )
  

  results <- list(
    "model_name" = ml_model,
    "cv_rsq" = max( model[["results"]][["Rsquared"]]),
    "ho_rsq" = cor(predicted, test_dat$workhours),
    "no_seconds_og" = difftime(end,start,units="secs")
  )
  
  return(results)
  
}

#Same as project 10 but pre-allocated length of ml_results_list
#also changed from for loop to mapply

#convert to list for easier use in sapply
ml_methods <- c("lm","glmnet","ranger","xgbTree")  #add xgbTree back

# ##mapply returns list of 16 like 4x4 matrix
ml_results_norm <- mapply(ml_function, ml_model=ml_methods, parallelize=FALSE)



#only keep mapply if faster than for loop -- true
#mapply times: 5.99367308616638 13.2867469787598 104.833606004715 410.396929979324



#run parallelized

ml_results_prll <- mapply(ml_function, ml_model=ml_methods, parallelize=FALSE)
#if i give each function run to one node
#that node works like normal rstudio for each model training run and is slow
#want to use multiple clusters for model training

#first parallelized results. compare to normal ml_function/mapply
# 18.0432209968567 -- much slower
# 17.6483490467072 -- a few seconds slower
# 108.075783967972 -- faster by about 30 seconds
# 244.542004108429 -- faster by less than 200 sec 

##SIMPLE PARALLEL RESULTS
#with glmnet and parallelize  TRUE vs FALSE
#when TRUE
# $model_name
# [1] "glmnet"
# 
# $cv_rsq
# [1] 0.8400087
# 
# $ho_rsq
# [1] 0.5734471
# 
# $no_seconds_og
# Time difference of 18.0526 secs
#when FALSE
# $model_name
# [1] "glmnet"
# 
# $cv_rsq
# [1] 0.8400087
# 
# $ho_rsq
# [1] 0.5734471
# 
# $no_seconds_og
# Time difference of 15.3306 secs




#Publication

#not needed with mapply
#ml_results_df <- do.call("rbind", ml_results_list)

table1_tbl <- tibble(
  algo = unlist(ml_results_list[1,]),
  cv_rsq = unlist(ml_results_list[2,]),
  ho_rsq = unlist(ml_results_list[3,]),
) %>% 
  mutate(across(ends_with("_rsq"),
                \(x) gsub("0\\.",".",
                          format(round(x, digits=2), nsmall = 2)) ) ) %>% 
  mutate(algo = c("OLS Regression","Elastic Net","Random Forest" 
               ))

# # A tibble: 3 × 3
# algo           cv_rsq ho_rsq
# <chr>          <chr>  <chr> 
# 1 OLS Regression .13    .06   
# 2 Elastic Net    .84    .57   
# 3 Random Forest  .91    .65 

# table1_tbl <- ml_results_df %>% 
#   mutate(algo = c("OLS Regression","Elastic Net","Random Forest", 
#                   "eXtreme Gradient Boosting"),

 

#table2_tbl should have 4 rows for each algo
#col1 - original - number of seconds normal
#col2 - parallelized - number of seconds parallel op

table2_tbl <- tibble(
  algo = unlist(ml_results_list[1,]),
  elapsed = unlist(ml_results_list[4,]),
  
)
# A tibble: 3 × 2
# algo   elapsed
# <chr>    <dbl>
#   1 lm      4.68
# 2 glmnet   9.99
# 3 ranger  112. 



