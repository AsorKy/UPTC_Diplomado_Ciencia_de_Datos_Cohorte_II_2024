#Regresion Lineal Polinomica

#apertura de archivo csv
dataset = read.csv("c:\\Users\\jrgui\\Documents\\GitHub\\Machine_Learning\\Regresion\\datasets\\Position_Salaries.csv")

#solo se necesitaran 2 columnas 
dataset = dataset[,2:3]

#Ajustar modelo de regresion lineal

regresion_lin = lm(formula = Salary ~ . ,
               data = dataset)

summary(regresion_lin)



#Ajustar modelo de regresion polinomica

#Grado2
dataset$Level2 = dataset$Level^2 

regresion_poli = lm(formula = Salary ~ . ,
                   data = dataset)

summary(regresion_poli)

#Grado3
dataset$Level3 = dataset$Level^3

regresion_poli = lm(formula = Salary ~ . ,
                    data = dataset)

summary(regresion_poli)


#Visualizacion de regresion lineal
library(ggplot2)

ggplot()+
  geom_point(aes(x=dataset$Level,y=dataset$Salary),
             colour="red")+
  geom_line(aes(x = dataset$Level,y = predict(regresion_lin, newdata=dataset)),
            colour="blue")+
  ggtitle("Prediccion lineal del sueldo de un empleado en funcion de los años de experiencia")+
  xlab("Nivel del Empleado")+
  ylab("Sueldo en $")
# la paroximacion lienal no es la mejor



#Visualizacion de regresion polinomica

ggplot()+
  geom_point(aes(x=dataset$Level,y=dataset$Salary),
             colour="red")+
  geom_line(aes(x = dataset$Level,y = predict(regresion_poli, newdata=dataset)),
            colour="blue")+
  ggtitle("Prediccion Polinomica del sueldo de un empleado en funcion de los años de experiencia")+
  xlab("Nivel del Empleado")+
  ylab("Sueldo en $")



# Predicción de nuevos resultados con Regresión Lineal
y_pred = predict(regresion_lin, newdata = data.frame(Level = 6.5))
y_pred

# Predicción de nuevos resultados con Regresión Polinómica
y_pred_poli = predict(regresion_poli, newdata = data.frame(Level = 6.5,
                                                     Level2 = 6.5^2,
                                                     Level3 = 6.5^3,
                                                     Level4 = 6.5^4))
y_pred_poli