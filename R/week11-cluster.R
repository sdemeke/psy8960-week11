#Script Settings and Resources
library(tidyverse)
library(haven)
library(caret)
library(parallel)
library(doParallel)

set.seed(24)


#Note- I use git directly from MSI/PuTTY to push and pull so there was no need to copy this file
#to a new folder called week11-cluster

#Data Import and Cleaning

gss_tbl <- read_sav("data/GSS2016.sav") %>%  
  rename(workhours = MOSTHRS) %>% 
  drop_na(workhours) %>% 
  mutate(workhours = as.numeric(workhours)) %>% 
  select(which(colMeans(is.na(.)) < 0.75))  %>% 
  select(-USUALHRS,-HRS1)


#Analysis

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
    ho_rsq = cor(predicted, test_data$workhours),
    no_seconds = difftime(end,start,units="secs")
  )
  
  return(results)
  
}



ml_methods <- c("lm","glmnet","ranger","xgbTree")  

#Normal
ml_results_norm <- mapply(getMLResults, SIMPLIFY = FALSE, ml_model=ml_methods)

ml_results_norm_df <- do.call("rbind", ml_results_norm)


#Paralleized
#after perusing the MSI job submission sites, i saw that amdsmall partition allows 128 cores 
#per node and advises 1900MB per core
#i switched to testing 128-1=127 cores 
local_cluster <- makeCluster(63)
registerDoParallel(local_cluster)

ml_results_prll <- mapply(getMLResults, SIMPLIFY = FALSE, ml_model=ml_methods)
ml_results_prll_df <- do.call("rbind", ml_results_prll)

stopCluster(local_cluster)
registerDoSEQ()



#Publication

table1_tbl <- ml_results_norm_df  %>% 
  mutate(algo = c("OLS Regression","Elastic Net","Random Forest", 
                  "eXtreme Gradient Boosting"),
         .before = cv_rsq)  %>% 
  select(-c(model_name, no_seconds)) %>% 
  mutate(across(ends_with("_rsq"),
                \(x) gsub("0\\.",".",
                          format(round(x, digits=2), nsmall = 2)) ) )

write_csv(table1_tbl, "out/table3.csv")


table2_tbl <- tibble(
  algo = c("OLS Regression","Elastic Net","Random Forest", 
           "eXtreme Gradient Boosting"),
  supercomputer = round(ml_results_norm_df$no_seconds,2),
  supercomputer_63  = round(ml_results_prll_df$no_seconds,2)
)

write_csv(table2_tbl, "out/table4.csv")
