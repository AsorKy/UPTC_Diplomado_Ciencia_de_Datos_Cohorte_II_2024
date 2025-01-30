#Regresion con Arboles de Decision

#apertura de archivo csv
dataset = read.csv("c:\\Users\\jrgui\\Documents\\GitHub\\Machine_Learning\\Regresion\\datasets\\Position_Salaries.csv")

#solo se necesitaran 2 columnas 
dataset = dataset[,2:3]



#Ajuste de regresion con Arbol de decision

library(rpart)

regresion = rpart(formula = Salary ~ . ,
                data = dataset,
                control = rpart.control(minsplit = 1))

#control se encarga de crear varias ramas, mejorar el algotimo




#predecir resultados con el conjunto de test
y_pred = predict(regresion, newdata = data.frame(Level = 6.5))

y_pred



#Visualizacion de regresion arbol de decision
library(ggplot2)


ggplot()+
  geom_point(aes(x=dataset$Level,y=dataset$Salary),
             colour="red")+
  geom_line(aes(x = dataset$Level,y = predict(regresion, newdata=dataset)),
            colour="blue")+
  ggtitle("Prediccion arbol de decision del sueldo de un empleado en funcion de los años de experiencia")+
  xlab("Nivel del Empleado")+
  ylab("Sueldo en $")

#Hace una aproximacion con fallas

