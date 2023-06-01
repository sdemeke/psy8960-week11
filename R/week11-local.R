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

#Same as project 10 but pre-allocated length of ml_results_list
#also changed from for loop to mapply

ml_methods <- c("lm","glmnet","ranger","xgbTree")  

ml_results_norm <- mapply(ml_function, SIMPLIFY = FALSE, ml_model=ml_methods)
ml_results_norm_df <- do.call("rbind", ml_results_norm)



#run parallelized

#is killing and restarting multiple cores for each run messing with efficiency? slightly
#run it here just once
#clusterMap very slow
local_cluster <- makeCluster(detectCores()-1)
registerDoParallel(local_cluster)
ml_results_prll <- mapply(ml_function, SIMPLIFY = FALSE, ml_model=ml_methods)
stopCluster(local_cluster)
registerDoSEQ()

ml_results_prll_df <- do.call("rbind", ml_results_prll)



#Publication


table1_tbl <- ml_results_norm_df  %>% 
  mutate(algo = c("OLS Regression","Elastic Net","Random Forest", 
                  "eXtreme Gradient Boosting"),
         .before = cv_rsq)  %>% 
  select(-c(model_name, no_seconds)) %>% 
  mutate(across(ends_with("_rsq"),
                \(x) gsub("0\\.",".",
                          format(round(x, digits=2), nsmall = 2)) ) )
# # A tibble: 4 × 4
# algo                      cv_rsq ho_rsq no_seconds     
# <chr>                     <chr>  <chr>  <drtn>         
# 1 OLS Regression            .22    .06      4.247859 secs
# 2 Elastic Net               .86    .57      9.763302 secs
# 3 Random Forest             .93    .65     74.108250 secs
# 4 eXtreme Gradient Boosting .97    .64    213.907211 secs




table2_tbl <- tibble(
  algo = c("OLS Regression","Elastic Net","Random Forest", 
           "eXtreme Gradient Boosting"),
  original = ml_results_norm_df$no_seconds,
  parallelized = ml_results_prll_df$no_seconds
)

# A tibble: 4 × 3
# algo                      original        parallelized   (using kill and restart option)
# <chr>                     <drtn>          <drtn>         
#   1 OLS Regression              4.507472 secs   4.870411 secs
# 2 Elastic Net                 9.497377 secs   6.788110 secs
# 3 Random Forest              70.580850 secs  85.530010 secs
# 4 eXtreme Gradient Boosting 202.817918 secs 138.839216 secs


# A tibble: 4 × 3
# algo                      original        parallelized   (using one run of make clusters)
# <chr>                     <drtn>          <drtn>         
# 1 OLS Regression              4.507472 secs   4.763399 secs
# 2 Elastic Net                 9.497377 secs   3.559537 secs
# 3 Random Forest              70.580850 secs  59.001915 secs
# 4 eXtreme Gradient Boosting 202.817918 secs 172.822029 secs
