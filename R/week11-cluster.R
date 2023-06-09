#Script Settings and Resources
library(tidyverse)
library(haven)
library(caret)
library(parallel)
library(doParallel)

set.seed(24)


#Note- I use git directly from MSI/PuTTY to push and pull so there was no need to copy this file
#to a new folder called week11-cluster
#I also saw that MSI made R/4.3.0 was available but had already installed all packages into 4/4.2.2

#Data Import and Cleaning

#Changed call to file path due to change in directory structure from within MSI. In local.R, wd is set by location of
#Rstudio file. In MSI, wd is the home directory of the git repo.

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
    ho_rsq = cor(predicted, test_data$workhours)^2,
    no_seconds = difftime(end,start,units="secs")
  )
  
  return(results)
  
}



ml_methods <- c("lm","glmnet","ranger","xgbTree")  

#Normal - no changes made
ml_results_norm <- mapply(getMLResults, SIMPLIFY = FALSE, ml_model=ml_methods)

ml_results_norm_df <- do.call("rbind", ml_results_norm)


#Paralleized
#after perusing the MSI job submission sites, i saw that amdsmall partition allows 128 cores 
#per node and advises 1900MB per core. I was having trouble with job submission failures at 
#128 cores so I opted for 64 cores and applied same logic of using 64-1=63 cores for models
#to run on
local_cluster <- makeCluster(63)
registerDoParallel(local_cluster)

ml_results_prll <- mapply(getMLResults, SIMPLIFY = FALSE, ml_model=ml_methods)
ml_results_prll_df <- do.call("rbind", ml_results_prll)

stopCluster(local_cluster)
registerDoSEQ()



#Publication
#Only changes here for saving csv output and changing column names in table2_tbl

table1_tbl <- ml_results_norm_df  %>% 
  mutate(algo = c("OLS Regression","Elastic Net","Random Forest", 
                  "eXtreme Gradient Boosting"),
         .before = cv_rsq)  %>% 
  select(-c(model_name, no_seconds)) %>% 
  mutate(across(ends_with("_rsq"),
                \(x) gsub("0\\.",".",
                          format(round(x, digits=2), nsmall = 2)) ) )

write_csv(table1_tbl, "out/table3.csv")

#table3.csv
# algo,cv_rsq,ho_rsq
# OLS Regression,.14,.01
# Elastic Net,.81,.31
# Random Forest,.92,.39
# eXtreme Gradient Boosting,.95,.33

table2_tbl <- tibble(
  algo = c("OLS Regression","Elastic Net","Random Forest", 
           "eXtreme Gradient Boosting"),
  supercomputer = round(ml_results_norm_df$no_seconds,2),
  supercomputer_63  = round(ml_results_prll_df$no_seconds,2)
)

write_csv(table2_tbl, "out/table4.csv")

#table4.csv
# algo,supercomputer,supercomputer_63
# OLS Regression,6.05,6.97
# Elastic Net,13.65,2.8
# Random Forest,42.5,7.3
# eXtreme Gradient Boosting,267.07,8.14



##Answers to Questions
#1. The extreme gradient boost model benefited most from moving to the supercomputer, at least
#for the parallelized runs. On my local machine, parallelization reduced runtime from ~255 to ~125
#seconds. On the supercomputer, using 63 cores, this was further reduced to just 8 seconds. The
#number of cores used between my local machine and the supercomputer increased 9 fold (from 7
#to 63) which helps explain this dramatic increase in speed of the parallelized process. The
#changes for random forest were similar but slightly less dramatic as the starting point for 
#speed was not as slow as for xgb. These two models benefited most from moving to supercomputer.

#2. What is the relationship between time and the number of cores used?
#On my local machine, I used maximum 7 cores. On the supercomputer, I set models to train on 63 cores.
#As the number of cores used increased, the time to train the four models decreased and this decrease
#was most noticeable for random forest and extreme gradient boost models which fell below 10seconds
#when using 9x the number of cores. 

#3. Based on my pick of random forest per its relatively high cross-validated and holdout Rsquared values,
#I would recommend use of supercomputer in a production model. On my local machine, the difference between
#normal and parallelized computational speed was not noticeable for random forest but this changed drastically
#when using the supercomputer. Table 1 and 3 recommend random forest or the extreme gradient boost models 
#as the most accurate.Table 2 shows that the latter model is much more computationally expensive even though
#parallelization reduces the training time. On the supercomputer, however, parallelizing over 63 cores
#makes random forest and extreme gradient boost models as speedy as OLS regression.



