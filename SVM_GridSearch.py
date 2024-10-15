
#  13/10/2024
#  OASM
#  Este programa realiza la búsqueda de hiperparámetros usando nivelación de cargas e implementación de procesos en hilos

import itertools
import multiprocess
import time
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


n_cores = 7

# Definir la cuadrícula de parámetros
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 0.001, 0.01]
}

# Generar combinaciones de parámetros para SVM
keys_svm, values_svm = zip(*param_grid_svm.items())
combinations_svm = [dict(zip(keys_svm, v)) for v in itertools.product(*values_svm)]

def Nivelacion_de_Cargas(n_cores, lista_inicial):
    lista_final = []
    longitud_li = len(lista_inicial)
    carga = longitud_li // n_cores 
    salidas = longitud_li % n_cores
    contador = 0

    for i in range(n_cores):
        if i < salidas:
            carga2 = contador + carga + 1
        else:
            carga2 = contador + carga
        lista_final.append(lista_inicial[contador:carga2])
        contador = carga2
    return lista_final


def evaluate_set(hyperparameter_set, lock):
    # Leer el dataset con etiquetas categóricas
    df = pd.read_csv('Data_for_UCI_named.csv') 


    df['stabf'] = df['stabf'].map({'unstable': 0, 'stable': 1})

    # características (X) y etiquetas (y)
    X = df.iloc[:, :-1].values  
    y = df.iloc[:, -1].values   
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20)

    for s in hyperparameter_set:
        clf = SVC()
        clf.set_params(C=s['C'], kernel=s['kernel'], gamma=s['gamma'])
        clf.fit(X_train, y_train)
        
        # Predecir con el conjunto de prueba
        y_pred = clf.predict(X_test)
        
        # Bloqueo para evitar problemas en la salida de varios procesos
        lock.acquire()
        print(f'Parametros:{s}, Accuracy en el proceso:', accuracy_score(y_test, y_pred))
        lock.release()


if __name__ == '__main__':
    threads = []
    N_THREADS = n_cores  
    splits = Nivelacion_de_Cargas(N_THREADS, combinations_svm)  
    lock = multiprocess.Lock()

    for i in range(N_THREADS):
        # Crear procesos paralelos para evaluar hiperparámetros
        threads.append(multiprocess.Process(target=evaluate_set, args=(splits[i], lock)))

    start_time = time.perf_counter()

    # Ejecutar los procesos
    for thread in threads:
        thread.start()

    # Esperar a que terminen todos los procesos
    for thread in threads:
        thread.join()

    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time - start_time} seconds")
