calcular_RMSE <- function(modelo, dados_de_teste, gabarito){
  previsões <- predict(modelo, dados_de_teste)
  erros <- previsões - gabarito
  RMSE <- sqrt(mean(erros^2, na.rm=TRUE))
  RMSE
}