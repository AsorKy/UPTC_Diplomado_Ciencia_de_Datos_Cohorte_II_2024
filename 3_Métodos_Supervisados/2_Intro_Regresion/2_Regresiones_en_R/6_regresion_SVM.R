#Regresion con Maquinas de Soporte Vectorial

#apertura de archivo csv
dataset = read.csv("c:\\Users\\jrgui\\Documents\\GitHub\\Machine_Learning\\Regresion\\datasets\\Position_Salaries.csv")

#solo se necesitaran 2 columnas 
dataset = dataset[,2:3]



#libreria para usar SVM
library(e1071)

regresion = svm(formula = Salary ~ . ,
                data = dataset,
                type = "eps-regression",
                kernel = "radial")




#Visualizacion de regresion SVM
library(ggplot2)

ggplot()+
  geom_point(aes(x=dataset$Level,y=dataset$Salary),
             colour="red")+
  geom_line(aes(x = dataset$Level,y = predict(regresion, newdata=dataset)),
            colour="blue")+
  ggtitle("Prediccion SVR del sueldo de un empleado en funcion de los años de experiencia")+
  xlab("Nivel del Empleado")+
  ylab("Sueldo en $")

#Hace una aproximacion decente, solo falla para el sueldo del ultimo punto

#Las SVM suelen menospreciar lo valores atipicos, que es como el caso del ultimo punto
