# Regresion Lineal Simple

#apertura de archivo csv
dataset = read.csv("c:\\Users\\jrgui\\Documents\\GitHub\\Machine_Learning\\Regresion\\datasets\\Salary_Data.csv")



#Division de Datos - entrenamiento y validacion

#Cargo la libreria a utilizar
library(caTools)

#permite escoger la semilla para hacer la division
set.seed(123)

#genera una mascara con los 80% de True y 20 de False
split = sample.split(dataset$Salary, SplitRatio = 2/3)

#genera los datasets de entrenamiento y validacion a partir de split
dataset_train = subset(dataset, split == TRUE)
dataset_test = subset(dataset, split == FALSE)




#Ajustar modelo de regresion lienal con los datos de entrenamiento

#Crea la regresion lineal
regresion = lm(formula = Salary ~ YearsExperience,
               data = dataset_train)

#resumen de la regresion
summary(regresion)



#predecir resultados con el conjunto de test
y_pred = predict(regresion, newdata=dataset_test)

y_pred


#Visualizacion de los resultados en el conjunto de entrenamiento

library(ggplot2)

ggplot()+
  geom_point(aes(x=dataset_train$YearsExperience,y=dataset_train$Salary),
             colour="red")+
  geom_line(aes(x = dataset_train$YearsExperience,y = predict(regresion, newdata=dataset_train)),
            colour="blue")+
  ggtitle("Sueldo Vs Años de Entremaniento (Conjunto de Entrenamieto)")+
  xlab("Sueldo en $")+
  ylab("Años de Experiencia")
  


#Visualizacion de los resultados en el conjunto de test



ggplot()+
  geom_point(aes(x=dataset_test$YearsExperience,y=dataset_test$Salary),
             colour="red")+
  geom_line(aes(x = dataset_train$YearsExperience,y = predict(regresion, newdata=dataset_train)),
            colour="blue")+
  ggtitle("Sueldo Vs Años de Entremaniento (Conjunto de test)")+
  xlab("Sueldo en $")+
  ylab("Años de Experiencia")
