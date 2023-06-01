
set.seed(24)

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

predicted <- predict(model, test_dat, na.action = na.pass)

tibble(
  model_name = "xgbTree",
  cv_rsq = max( model[["results"]][["Rsquared"]]),
  ho_rsq = cor(predicted, test_dat$workhours),
  no_seconds = difftime(end,start,units="secs")
)

# A tibble: 1 × 4
# model_name cv_rsq ho_rsq no_seconds   
# <chr>       <dbl>  <dbl> <drtn>       
#   1 ranger      0.926  0.622 151.3299 secs

#with only medianImpute
# # A tibble: 1 × 4
# model_name cv_rsq ho_rsq no_seconds   
# <chr>       <dbl>  <dbl> <drtn>       
#   1 ranger      0.927  0.648 144.1519 secs

#with all preprocess
# A tibble: 1 × 4
# model_name cv_rsq ho_rsq no_seconds   
# <chr>       <dbl>  <dbl> <drtn>       
# 1 xgbTree     0.951  0.593 298.1691 secs






set.seed(24)

start <- Sys.time() 


  local_cluster <- makeCluster(detectCores()-1)
  registerDoParallel(local_cluster)

  model <- train(
    workhours~.,
    data = train_dat, 
    metric = "Rsquared",
    method = "xgbTree",
    preProcess = c("center","scale","nzv","medianImpute"), 
    na.action = na.pass,
    trControl = myControl
  )

  stopCluster(local_cluster)
  registerDoSEQ()


end <- Sys.time()

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
# 1 ranger      0.926  0.622 125.4791 secs

# # A tibble: 1 × 4
# model_name cv_rsq ho_rsq no_seconds   
# <chr>       <dbl>  <dbl> <drtn>       
# 1 xgbTree     0.951  0.593 182.7609 secs