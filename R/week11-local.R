#Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(haven)
library(caret)


#Data Import and Cleaning

gss_tbl <- read_sav("../data/GSS2016.sav") %>%  
  rename(workhours = MOSTHRS) %>% 
  drop_na(workhours) %>% 
  mutate(workhours = as.numeric(workhours)) %>% 
  select(which(colMeans(is.na(.)) < 0.75))  


#Visualization

gss_tbl %>% 
  ggplot(aes(x=workhours)) + geom_histogram()


#Analysis

ml_function <- function(dat = gss_tbl, ml_model = "lm", no_folds = 10) { 
  
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


  results <- tibble(
    model_name = ml_model,
    cv_rsq = max( model[["results"]][["Rsquared"]]),
    ho_rsq = cor(predicted, test_dat$workhours)
  )
  
  return(results)
  
}



ml_methods <- c("lm","glmnet","ranger","xgbTree") 
ml_results_list <- vector(mode="list", length = 4)

for(i in 1:length(ml_methods)) {
  ml_results_list[[i]] <- ml_function(ml_model = ml_methods[i])
}


#Publication

table1_tbl <- do.call("rbind", ml_results_list) %>% 
  mutate(algo = c("OLS Regression","Elastic Net","Random Forest", 
                  "eXtreme Gradient Boosting"),
         .before = cv_rsq)  %>% 
  select(-c(model_name)) %>% 
  mutate(across(ends_with("_rsq"),
                \(x) gsub("0\\.",".",
                          format(round(x, digits=2), nsmall = 2)) ) )


# A tibble: 4 × 3
#  algo                      cv_rsq ho_rsq
# 1 OLS Regression            .12    .18   
# 2 Elastic Net               .83    .66   
# 3 Random Forest             .93    .66   
# 4 eXtreme Gradient Boosting .93    .68  



