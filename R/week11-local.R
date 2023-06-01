#Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(haven)
library(caret)
library(parallel)
library(doParallel)

set.seed(24)


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

#make some of the code only run once

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


ml_function <- function(train_data=train_dat, test_data=test_dat, ml_model =  c("lm","glmnet","ranger","xgbTree"), parallelize=FALSE) { 
  
  start <- Sys.time() 
  
  if(parallelize == TRUE) {
    local_cluster <- makeCluster(detectCores()-1)
    registerDoParallel(local_cluster)
  }
  model <- train(
    workhours~.,
    data = train_data, 
    metric = "Rsquared",
    method = ml_model,
    preProcess = c("center","scale","nzv","medianImpute"), 
    na.action = na.pass,
    trControl = myControl
  )
  
  if(parallelize==TRUE) {
    stopCluster(local_cluster)
    registerDoSEQ()
  }
  
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

#Same as project 10 but pre-allocated length of ml_results_list
#also changed from for loop to mapply

#convert to list for easier use in sapply
ml_methods <- c("lm","glmnet","ranger","xgbTree")  #add xgbTree back
# ml_results_list <- vector(mode="list", length = 4)
# 
# #run normal
# for(i in 1:length(ml_methods)) {
#   ml_results_list[[i]] <- ml_function(ml_model = ml_methods[i])
# }
# 
# 
# ##mapply returns list of 16 like 4x4 matrix
ml_results_norm <- mapply(ml_function, SIMPLIFY = FALSE, ml_model=ml_methods, parallelize=FALSE)


##what does mapply return if i change function to return vector and not tibble?
#same thing

#only keep mapply if faster than for loop
#mapply times: 5.99367308616638 13.2867469787598 104.833606004715 410.396929979324


#run parallelized


ml_results_prll <- mapply(ml_function, SIMPLIFY = FALSE,ml_model=ml_methods, parallelize=TRUE)

#Publication
# > do.call("rbind", ml_results_norm)
# # A tibble: 4 × 4
# model_name cv_rsq ho_rsq no_seconds     
# * <chr>       <dbl>  <dbl> <drtn>         
# 1 lm          0.129 0.0633   4.880909 secs
# 2 glmnet      0.853 0.573   11.550487 secs
# 3 ranger      0.919 0.623  127.294593 secs
# 4 xgbTree     0.967 0.588  327.508267 secs
# > do.call("rbind", ml_results_prll)
# # A tibble: 4 × 4
# model_name cv_rsq ho_rsq no_seconds    
# * <chr>       <dbl>  <dbl> <drtn>        
# 1 lm          0.125 0.0633  18.51846 secs
# 2 glmnet      0.860 0.573   16.49879 secs
# 3 ranger      0.919 0.653  112.90714 secs
# 4 xgbTree     0.941 0.580  182.53742 secs
# > 
#not needed with mapply
#ml_results_df <- do.call("rbind", ml_results_list)



table1_tbl <- do.call("rbind", ml_results_norm) %>% 
  mutate(algo = c("OLS Regression","Elastic Net","Random Forest", 
                  "eXtreme Gradient Boosting"),
         .before = cv_rsq)  %>% 
  select(-c(model_name)) %>% 
  mutate(across(ends_with("_rsq"),
                \(x) gsub("0\\.",".",
                          format(round(x, digits=2), nsmall = 2)) ) )


# # A tibble: 3 × 3
# algo           cv_rsq ho_rsq
# <chr>          <chr>  <chr> 
# 1 OLS Regression .13    .06   
# 2 Elastic Net    .84    .57   
# 3 Random Forest  .91    .65 




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

