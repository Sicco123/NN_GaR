#install.packages("quantreg")
library(quantreg)

horizon_list <- c(1,2,3,4,5,6,12)
quantile_levels <- c(0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
sample_sizes <- c(150, 250, 500)
M <- 10000
dir_to_store_simulation <- "simulated_data"
dir_to_store_reg_quantiles <- "linear_quantile_regression"
day <- '20221014'
test_size = 200

prepare_forecast_predictors <- function(horizon_list, h, X){
    max_horizon <- max(horizon_list)
    X_star <-  if (max_horizon - h != 0) tail(X,-(max_horizon-h)) else X 
    X_star <- head(X_star, -h)
    rownames(X_star) <- NULL
    
    return (X_star)
}

prepare_forecast_target<-function(horizon_list, y){
    max_horizon <- max(horizon_list)
    y_star <- tail(y, -max_horizon)
    rownames(y_star) <- NULL
    y_star <- unlist(y_star)
    return(y_star)
}
estimation <- function(X_train, X, sample_size, y_star, m, h){
    X_train_star <- prepare_forecast_predictors(horizon_list, h, X_train)
    X_star <- prepare_forecast_predictors(horizon_list, h, X)
    

    train_dataframe <- data.frame(cbind(y_star, X_train_star))
    colnames(train_dataframe) <- c('y_star','x1','x2','x3','x4')

    rqfit <- rqfit <- rq(y_star ~ ., data = train_dataframe, tau = quantile_levels )
    params <- rqfit$coef
    params <- c(quantile_levels, params)

    colnames(X_star) <- c('x1','x2','x3','x4')
    reg_quantiles <- predict(rqfit, newdata = X_star)#lm(y_star ~ X_star$V2 + X_star$V3 + X_star$V4 + X_star$V5))

    dir.create(file.path('simulation_results','r_results',sample_size,m), showWarnings = FALSE, recursive = TRUE)
    write.csv(reg_quantiles, file.path('simulation_results','r_results',sample_size,m,paste0(h,'_step_ahead.csv')),row.names = FALSE, col.names = FALSE)
    write.csv(params, file.path('simulation_results','r_results',sample_size,m,paste0(h,'_reg_params.csv')),  row.names = FALSE, col.names = FALSE) 

}
    
for (m in c(0:M)){
    for (sample_size in sample_sizes){

        data <- read.csv(file.path('simulated_data',sample_size,'221014',paste0(m,'.csv')), header = FALSE)

        y <- data[,c(1:1)]

        X <- data[,c(2:5)]
        y_train <- head(y, -test_size)
        X_train <- head(X, -test_size)

        y_star <- prepare_forecast_target(horizon_list, y_train)

  

        for (h in horizon_list){
            estimation(X_train, X, sample_size, y_star, m, h)
        }

        print(paste(m, sample_size))
    }
}