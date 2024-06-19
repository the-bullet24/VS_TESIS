from flask import Flask, render_template


# Importar librerias
import seaborn as sns
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib
matplotlib.use('Agg')



app = Flask(__name__)


@app.route('/')
def mostrar_grafico():

    # Cargar datos
    df = pd.read_csv("DATA_IUP_2025_arbol_decision.csv")
    #df.head(12)
    #df[5:]

    # Diccionario para mapear los meses a valores numéricos
    meses_a_numeros = {
    'Enero': 1,
    'Febrero': 2,
    'Marzo': 3,
    'Abril': 4,
    'Mayo': 5,
    'Junio': 6,
    'Julio': 7,
    'Agosto': 8,
    'Setiembre': 9,
    'Octubre': 10,
    'Noviembre': 11,
    'Diciembre': 12
    }

    # Crear una nueva columna con los valores numéricos de los meses
    df['MESES_NUM'] = df['MESES'].map(meses_a_numeros)

    X = df[['ANIO','MESES_NUM','TOTAL_AVENA']]
    y = df['TOTAL_VENTAS_EN_SOLES']

    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Crear el modelo de árbol de regresión
    regressor = DecisionTreeRegressor(max_depth=6)

    # Entrenar el modelo
    regressor.fit(X_train, y_train)

    # Hacer predicciones en el conjunto de prueba
    y_pred = regressor.predict(X_test)

    # Evaluar el modelo
    # R² (coeficiente de determinación)
    #El R² proporciona una medida del porcentaje de variación
    #en la variable dependiente que es explicada por el modelo. Un valor de R² cercano a 1 indica un buen ajuste.
    r2 = r2_score(y_test, y_pred)
    print(f"R-squared: {r2}")

    # Función para hacer predicciones con nuevos datos
    def predict_total_avena(new_data, model):
        # Convertir los nuevos datos en un DataFrame
        new_df = pd.DataFrame(new_data)
        # Hacer la predicción
        prediction = model.predict(new_df)
        return prediction[0]
   
    # Nuevos datos de ejemplo para hacer predicciones
    new_data_1 = {'ANIO': [2024], 'MESES_NUM': [1], 'TOTAL_AVENA': [20284]}
    new_data_2 = {'ANIO': [2024], 'MESES_NUM': [2], 'TOTAL_AVENA': [16378]}
    new_data_3 = {'ANIO': [2024], 'MESES_NUM': [3], 'TOTAL_AVENA': [20069]}
    new_data_4 = {'ANIO': [2024], 'MESES_NUM': [4], 'TOTAL_AVENA': [20744]}
    new_data_5 = {'ANIO': [2024], 'MESES_NUM': [5], 'TOTAL_AVENA': [21515]}
    new_data_6 = {'ANIO': [2024], 'MESES_NUM': [6], 'TOTAL_AVENA': [20540]}
    new_data_7 = {'ANIO': [2024], 'MESES_NUM': [7], 'TOTAL_AVENA': [18906]}
    new_data_8 = {'ANIO': [2024], 'MESES_NUM': [8], 'TOTAL_AVENA': [21231]}
    new_data_9 = {'ANIO': [2024], 'MESES_NUM': [9], 'TOTAL_AVENA': [20263]}
    new_data_10 = {'ANIO': [2024], 'MESES_NUM': [10], 'TOTAL_AVENA': [16968]}
    new_data_11 = {'ANIO': [2024], 'MESES_NUM': [11], 'TOTAL_AVENA': [17731]}
    new_data_12 = {'ANIO': [2024], 'MESES_NUM': [12], 'TOTAL_AVENA': [23560]} 


    # Lista para almacenar los nuevos datos
    new_data_list = [new_data_1, new_data_2, new_data_3, new_data_4, new_data_5,new_data_6, new_data_7, new_data_8, new_data_9, new_data_10,new_data_11, new_data_12]

    # Lista para almacenar las predicciones
    predictions = []


    # Hacer predicciones con los nuevos datos y almacenarlas en la lista
    for data in new_data_list:
     prediction = predict_total_avena(data, regressor)
     predictions.append(prediction)

    # Crear un DataFrame con las predicciones
    predictions_df = pd.DataFrame(predictions, columns=['Predicción VENTA_TOTAL_EN_SOLES_X_CANAL'])

    # Mostrar el DataFrame
    print(predictions_df)


    # Crear un DataFrame para la visualización
    visual_df = pd.DataFrame({
        'Mes': range(1, 13),
        'Predicción VENTA_TOTAL_EN_SOLES_X_CANAL': predictions
    })

    # Convertir las predicciones a porcentajes (asumiendo que son proporciones)
    visual_df['Predicción VENTA_TOTAL_EN_SOLES_X_CANAL'] = visual_df['Predicción VENTA_TOTAL_EN_SOLES_X_CANAL'] / visual_df['Predicción VENTA_TOTAL_EN_SOLES_X_CANAL'].sum()



    # Gráfico de barras
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Mes', y='Predicción VENTA_TOTAL_EN_SOLES_X_CANAL', data=visual_df)
    plt.title('Predicción de VENTA_TOTAL_EN_SOLES_X_CANAL por Mes')
    plt.xlabel('Mes')
    plt.ylabel('Porcentaje')
    plt.xticks(range(12), [f'Mes {i+1}' for i in range(12)])
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))  # Formatear el eje Y como porcentaje
    #plt.grid(True)

    # Anotar los valores en el gráfico
    for index, row in visual_df.iterrows():
        ax.text(row.Mes - 1, 
                row['Predicción VENTA_TOTAL_EN_SOLES_X_CANAL'] + 0.005, 
                f"{row['Predicción VENTA_TOTAL_EN_SOLES_X_CANAL']:.2%}", 
                color='black', 
                ha="center")

    #plt.show()

 # Guardar el gráfico como un archivo PNG
    plt.savefig('static/grafico.png')  # Guardamos el gráfico en la carpeta static

    # Renderizar la plantilla HTML con el gráfico incrustado
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)

 