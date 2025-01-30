#Regresion con Bosques Aleatorio

#apertura de archivo csv
dataset = read.csv("c:\\Users\\jrgui\\Documents\\GitHub\\Machine_Learning\\Regresion\\datasets\\Position_Salaries.csv")

#solo se necesitaran 2 columnas 
dataset = dataset[,2:3]



#Ajuste de regresion con Bosque Aleatorio

library(randomForest)

set.seed(1234)
regresion = randomForest(x=dataset[1], 
                         y=dataset$Salary,
                         ntree=10)



#predecir resultados con el conjunto de test
y_pred = predict(regresion, newdata = data.frame(Level = 6.5))

y_pred



#Visualizacion de regresion bosque aleatorio
library(ggplot2)


ggplot()+
  geom_point(aes(x=dataset$Level,y=dataset$Salary),
             colour="red")+
  geom_line(aes(x = dataset$Level,y = predict(regresion, newdata=dataset)),
            colour="blue")+
  ggtitle("Prediccion bosque aleatorio del sueldo de un empleado en funcion de los años de experiencia")+
  xlab("Nivel del Empleado")+
  ylab("Sueldo en $")