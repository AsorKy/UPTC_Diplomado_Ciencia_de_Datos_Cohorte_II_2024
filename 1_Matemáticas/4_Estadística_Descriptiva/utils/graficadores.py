
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#-------------------------------------------------------------------------------------------------------------------
#--------------------- Graficador de histograma de muestras, modelo y parámetro poblacional ------------------------
#-------------------------------------------------------------------------------------------------------------------

def histograma_frecuencias(data, parametro, bins, title, x_label='data (X)', y_label='Frecuencia', kde=True, data_label=None, color='blue'):
    
  plt.figure(figsize=(8, 4))
  sns.histplot(data, bins=bins, kde=kde, color=color, label=data_label)
  plt.axvline(parametro, color=color, linestyle='dashed', linewidth=2)
  plt.title(title)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.legend()
  plt.grid(linestyle='--', alpha=0.7)
  plt.show()


#-------------------------------------------------------------------------------------------------------------------
#---------------------------- Graficador de histogramas y medidas de tendencia central -----------------------------
#-------------------------------------------------------------------------------------------------------------------

def histograma_frecuencias_centrales(data,variable,bins,title,x_label='data (X)', y_label='Frecuencia', kde=True):
  plt.figure(figsize=(8,4))
  sns.histplot(data = data, bins=bins, x=variable, kde=True, color='blue', label='data')
  plt.axvline( x = data[variable].mean(), color = 'red', linestyle = 'dashed', label='mean')
  plt.axvline( x = data[variable].median(), color = 'green', linestyle = 'dashed', label='median')
  plt.axvline( x = data[variable].mode()[0],color = 'black', linestyle = 'dashed', label='mode')
  plt.title(title)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.grid(linestyle='--', alpha=0.7)
  plt.legend()
  plt.show()


#-------------------------------------------------------------------------------------------------------------------
#------------------------------------ Graficador de histogramas y dispersiones -------------------------------------
#-------------------------------------------------------------------------------------------------------------------


def histograma_dispersion(data,variable,bins,title, title_box,x_label='data (X)', y_label='Frecuencia', kde=True):
    fig, ax = plt.subplots(1,2,figsize=(14,5))
    sns.histplot(data = data, x=variable, bins=bins, kde=True, color='blue', ax = ax[0])
    sns.boxplot( data = data, x=variable, ax = ax[1])
    
    mean = data[variable].mean()
    q1 = data[variable].quantile(0.25)
    q2 = data[variable].quantile(0.5)
    q3 = data[variable].quantile(0.75)
    low_lim = q1 - 1.5*(q3-q1)
    up_lim  = q3 + 1.5*(q3-q1)
    
    sigma = data[variable].std()
    q3_sigma = mean + sigma
    q1_sigma = mean - sigma

    ax[0].axvline(x = low_lim, color='black', linestyle='dashed', label='low limit(IQR)')
    ax[0].axvline(x = up_lim, color='black', linestyle='dashed', label='upper limit(IQR)')
    ax[0].axvline(x = data[variable].min(), color='red', linestyle='dashed', label='x_min')

    ax[0].axvline(x = q1, color='green', linestyle='dashed', label='Q1')
    ax[0].axvline(x = q2, color='blue', linestyle='dashed', label='Q2')
    ax[0].axvline(x = q3, color='green', linestyle='dashed', label='Q3')
    
    ax[1].axvline(x = low_lim, color='black', linestyle='dashed', label='low limit(IQR)')
    ax[1].axvline(x = up_lim, color='black', linestyle='dashed', label='upper limit(IQR)')
    ax[1].axvline(x = data[variable].min(), color='red', linestyle='dashed', label='x_min')

    ax[1].axvline(x = q1_sigma, color='green', linestyle='dashed', label='media - sigma')
    ax[1].axvline(x = q1, color='purple', linestyle='dashed', label='Q1')
    ax[1].axvline(x = q2, color='blue', linestyle='dashed', label='Q2')
    ax[1].axvline(x = q3_sigma, color='green', linestyle='dashed', label='media + sigma')
    ax[1].axvline(x = q3, color='purple', linestyle='dashed', label='Q3')

    ax[0].set_title(title)
    ax[0].set_xlabel(x_label)
    ax[0].set_ylabel(y_label)
    ax[0].legend()
    ax[1].legend()
    ax[1].set_title(title_box)
    ax[1].set_xlabel(x_label)
    ax[1].set_ylabel(y_label)
    ax[0].grid(linestyle='--', alpha=0.7)
    ax[1].grid(linestyle='--', alpha=0.7)
    plt.show()
    
    
#-------------------------------------------------------------------------------------------------------------------
#----------------------------------------------- Conjunto Anscombe -------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------


def Anscombe():
    # Cuarteto de Anscombe
    x1 = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5])
    y1 = np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68])

    x2 = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5])
    y2 = np.array([9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74])

    x3 = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5])
    y3 = np.array([7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73])

    x4 = np.array([8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 19])
    y4 = np.array([6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 5.56, 7.91, 6.89, 12.50])

    # Calcular estadísticas descriptivas
    media_x = np.mean(x1)
    varianza_x = np.var(x1, ddof=1)
    estadisticas = {
        'Media de x': media_x,
        'Varianza de x': varianza_x,
        'Media de y1': np.mean(y1),
        'Varianza de y1': np.var(y1, ddof=1),
        'Media de y2': np.mean(y2),
        'Varianza de y2': np.var(y2, ddof=1),
        'Media de y3': np.mean(y3),
        'Varianza de y3': np.var(y3, ddof=1),
        'Media de y4': np.mean(y4),
        'Varianza de y4': np.var(y4, ddof=1),
    }
    

    # Crear un DataFrame para almacenar los resultados
    df_stats = pd.DataFrame(estadisticas, index=['Valores'])
    print(df_stats)

    # Graficar los datos
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    # Conjunto de datos 1
    axs[0, 0].scatter(x1, y1)
    axs[0, 0].plot(np.unique(x1), np.poly1d(np.polyfit(x1, y1, 1))(np.unique(x1)), color='red')
    axs[0, 0].set_title('Conjunto de datos 1')
    axs[0, 0].set_xlabel('x')
    axs[0, 0].set_ylabel('y1')
    axs[0, 0].grid(linestyle='--', alpha=0.7)

    # Conjunto de datos 2
    axs[0, 1].scatter(x2, y2)
    axs[0, 1].plot(np.unique(x2), np.poly1d(np.polyfit(x2, y2, 1))(np.unique(x2)), color='red')
    axs[0, 1].set_title('Conjunto de datos 2')
    axs[0, 1].set_xlabel('x')
    axs[0, 1].set_ylabel('y2')
    axs[0, 1].grid(linestyle='--', alpha=0.7)

    # Conjunto de datos 3
    axs[1, 0].scatter(x3, y3)
    axs[1, 0].plot(np.unique(x3), np.poly1d(np.polyfit(x3, y3, 1))(np.unique(x3)), color='red')
    axs[1, 0].set_title('Conjunto de datos 3')
    axs[1, 0].set_xlabel('x')
    axs[1, 0].set_ylabel('y3')
    axs[1, 0].grid(linestyle='--', alpha=0.7)

    # Conjunto de datos 4
    axs[1, 1].scatter(x4, y4)
    axs[1, 1].plot(np.unique(x4), np.poly1d(np.polyfit(x4, y4, 1))(np.unique(x4)), color='red')
    axs[1, 1].set_title('Conjunto de datos 4')
    axs[1, 1].set_xlabel('x')
    axs[1, 1].set_ylabel('y4')
    axs[1, 1].grid(linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()