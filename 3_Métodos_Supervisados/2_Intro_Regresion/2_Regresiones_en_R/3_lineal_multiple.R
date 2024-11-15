# Regresion Lineal Multiple Simple

#apertura de archivo csv
dataset = read.csv("c:\\Users\\jrgui\\Documents\\GitHub\\Machine_Learning\\Regresion\\datasets\\50_Startups.csv")

# Codificacion de Datos Categoricos

dataset$State = factor(dataset$State,
                         levels = c("New York", "California", "Florida"),
                         labels = c(1,2,3))


#Division de Datos - entrenamiento y validacion

#Cargo la libreria a utilizar
library(caTools)

#permite escoger la semilla para hacer la division
set.seed(123)

#genera una mascara con los 80% de True y 20 de False
split = sample.split(dataset$Profit, SplitRatio = 0.8)

#genera los datasets de entrenamiento y validacion a partir de split
dataset_train = subset(dataset, split == TRUE)
dataset_test = subset(dataset, split == FALSE)


#Ajustar el modelo de regresion lineal multiple con el conjunto de entrenamiento

regresion = lm(formula = Profit ~ . ,
               data = dataset_train)

#muestra el resumen estadistico del modelo
summary(regresion)



#Predecion los resultados con el modelo de test

y_pred = predict(regresion, dataset_test)

y_pred



# Construir modelo optimo con eliminacion hacia atras

#p valor debe ser menor a 0.05

#Ajustar el modelo de regresion lineal multiple con el conjunto de entrenamiento
#Se corre sobre todo el modelo y se colocan el nombre de las columnas para aplicar
#la eliminacion hacia atras
regresion = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = dataset)

summary(regresion)

#Se elimina State
regresion = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
               data = dataset)

summary(regresion)

#Se elimina Administration
regresion = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               data = dataset)

summary(regresion)

#Se elimina Marketing.Spend
regresion = lm(formula = Profit ~ R.D.Spend,
               data = dataset)

summary(regresion)



