#Script Settings and Resources
library(tidyverse)
library(haven)
library(caret)
library(parallel)
library(doParallel)

set.seed(24)


#Data Import and Cleaning


gss_tbl <- read_sav("data/GSS2016.sav") %>%  
  rename(workhours = MOSTHRS) %>% 
  drop_na(workhours) %>% 
  mutate(workhours = as.numeric(workhours)) %>% 
  select(which(colMeans(is.na(.)) < 0.75))  %>% 
  select(-USUALHRS,-LEASTHRS)


#Analysis


no_folds <- 3
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


ml_methods <- c("lm","glmnet","ranger")  

#Normal
ml_results_norm <- mapply(ml_function, SIMPLIFY = FALSE, ml_model=ml_methods)
ml_results_norm_df <- do.call("rbind", ml_results_norm)

#Paralleized
local_cluster <- makeCluster(detectCores()-1)
registerDoParallel(local_cluster)

ml_results_prll <- mapply(ml_function, SIMPLIFY = FALSE, ml_model=ml_methods)

stopCluster(local_cluster)
registerDoSEQ()

ml_results_prll_df <- do.call("rbind", ml_results_prll)



#Publication

table1_tbl <- ml_results_norm_df  %>% 
  mutate(algo = c("OLS Regression","Elastic Net","Random Forest" 
                  ),
         .before = cv_rsq)  %>% 
  select(-c(model_name, no_seconds)) %>% 
  mutate(across(ends_with("_rsq"),
                \(x) gsub("0\\.",".",
                          format(round(x, digits=2), nsmall = 2)) ) )

write_csv(table1_tbl, "out/table3test.csv")


table2_tbl <- tibble(
  algo = c("OLS Regression","Elastic Net","Random Forest"),
  supercomputer = ml_results_norm_df$no_seconds,
  supercomputer_7 = ml_results_prll_df$no_seconds
)

write_csv(table1_tbl, "out/table4test.csv")
