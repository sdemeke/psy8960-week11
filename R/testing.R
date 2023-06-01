
set.seed(24)

start <- Sys.time() 
tic()
model <- train(
  workhours~.,
  data = train_dat, 
  metric = "Rsquared",
  method = "xgbTree",
  preProcess = c("center","scale","nzv","medianImpute"), 
  na.action = na.pass,
  trControl = myControl
)

end <- Sys.time()
toc()

predicted <- predict(model, test_dat, na.action = na.pass)

tibble(
  model_name = "xgbTree",
  cv_rsq = max( model[["results"]][["Rsquared"]]),
  ho_rsq = cor(predicted, test_dat$workhours),
  no_seconds = difftime(end,start,units="secs")
)


# # A tibble: 1 × 4
# model_name cv_rsq ho_rsq no_seconds   
# <chr>       <dbl>  <dbl> <drtn>       
# 1 lm          0.139 0.0633 4.556223 secs

# # A tibble: 1 × 4
# model_name cv_rsq ho_rsq no_seconds   
# <chr>       <dbl>  <dbl> <drtn>       
# 1 glmnet      0.829  0.573 9.550783 secs


# # A tibble: 1 × 4
# model_name cv_rsq ho_rsq no_seconds   
# <chr>       <dbl>  <dbl> <drtn>       
# 1 xgbTree      0.926  0.622 71.06108 secs


# # A tibble: 1 × 4
# model_name cv_rsq ho_rsq no_seconds   
# <chr>       <dbl>  <dbl> <drtn>       
#   1 xgbTree     0.951  0.593 262.9864 secs

#WITH MAPPLY: -- ESSENTIALLY THE SAME RUN TIMES
# # A tibble: 4 × 4
# algo                      cv_rsq ho_rsq no_seconds     
# <chr>                     <chr>  <chr>  <drtn>         
# 1 OLS Regression            .22    .06      4.247859 secs
# 2 Elastic Net               .86    .57      9.763302 secs
# 3 Random Forest             .93    .65     74.108250 secs
# 4 eXtreme Gradient Boosting .97    .64    213.907211 secs



set.seed(24)


  local_cluster <- makeCluster(detectCores()-1)
  registerDoParallel(local_cluster)

  start <- Sys.time() 
  
  model <- train(
    workhours~.,
    data = train_dat, 
    metric = "Rsquared",
    method = "xgbTree",
    preProcess = c("center","scale","nzv","medianImpute"), 
    na.action = na.pass,
    trControl = myControl
  )

  end <- Sys.time()
  
  stopCluster(local_cluster)
  registerDoSEQ()



predicted <- predict(model, test_dat, na.action = na.pass)

tibble(
  model_name = "xgbTree",
  cv_rsq = max( model[["results"]][["Rsquared"]]),
  ho_rsq = cor(predicted, test_dat$workhours),
  no_seconds = difftime(end,start,units="secs")
)


#lm - 8.7 seconds when include cluster ops in sys.time
#if only time model()

# # A tibble: 1 × 4
# model_name cv_rsq ho_rsq no_seconds  
# <chr>       <dbl>  <dbl> <drtn>      
# 1 lm          0.139 0.0633 4.95272 secs


# A tibble: 1 × 4
# model_name cv_rsq ho_rsq no_seconds   
# <chr>       <dbl>  <dbl> <drtn>       
# 1 glmnet      0.829  0.573 7.069333 secs

# # A tibble: 1 × 4
# model_name cv_rsq ho_rsq no_seconds   
# <chr>       <dbl>  <dbl> <drtn>       
# 1 ranger      0.926  0.622 62.31529 secs -- this ranges from 62 to ~80seconds


# A tibble: 1 × 4
# model_name cv_rsq ho_rsq no_seconds   
# <chr>       <dbl>  <dbl> <drtn>       
# 1 xgbTree     0.951  0.593 126.7301 secs