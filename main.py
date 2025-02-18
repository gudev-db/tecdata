import folium
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from streamlit_folium import st_folium
import json

st.set_page_config(layout="wide")

# Carregar os dados
@st.cache
def load_data():
    # Simular a leitura de dados CSVs
    data1 = pd.DataFrame({
        'Unidades da Federação': ['Brasil', 'Rondônia', 'Acre', 'Amazonas', 'Roraima', 'Pará', 'Amapá', 'Tocantins', 'Maranhão', 'Piauí', 
                                  'Ceará', 'Rio Grande do Norte', 'Paraíba', 'Pernambuco', 'Alagoas', 'Sergipe', 'Bahia', 'Minas Gerais', 'Espírito Santo', 
                                  'Rio de Janeiro', 'São Paulo', 'Paraná', 'Santa Catarina', 'Rio Grande do Sul', 'Mato Grosso do Sul', 'Mato Grosso', 
                                  'Goiás', 'Distrito Federal'],
        'IPQV': [0.157, 0.193, 0.237, 0.215, 0.169, 0.242, 0.222, 0.186, 0.259, 0.211, 0.187, 0.203, 0.206, 0.205, 0.216, 0.185, 0.198, 0.137, 
                 0.138, 0.148, 0.112, 0.112, 0.099, 0.126, 0.152, 0.166, 0.165, 0.138],
        'B': [7.277, 6.994, 6.895, 6.793, 7.046, 6.687, 6.965, 6.794, 6.534, 6.856, 6.833, 7.006, 6.875, 6.963, 6.641, 7.130, 7.012, 7.296, 
              7.337, 7.437, 7.669, 7.440, 7.524, 7.573, 7.373, 7.324, 7.351, 8.028],
        'K': [1.130, 1.349, 1.636, 1.460, 1.194, 1.620, 1.548, 1.267, 1.693, 1.450, 1.279, 1.423, 1.420, 1.424, 1.437, 1.320, 1.390, 0.997, 
              1.014, 1.101, 0.857, 0.833, 0.743, 0.956, 1.120, 1.216, 1.212, 1.104],
        'B*IPQV': [6.147, 5.645, 5.259, 5.334, 5.852, 5.067, 5.416, 5.527, 4.841, 5.406, 5.554, 5.583, 5.455, 5.539, 5.204, 5.810, 5.622, 
                   6.299, 6.324, 6.336, 6.811, 6.607, 6.781, 6.617, 6.253, 6.108, 6.139, 6.923]
    })
    
    data2 = pd.DataFrame({
        'Unidades da Federação': ['Brasil', 'Rondônia', 'Acre', 'Amazonas', 'Roraima', 'Pará', 'Amapá', 'Tocantins', 'Maranhão', 'Piauí', 
                                  'Ceará', 'Rio Grande do Norte', 'Paraíba', 'Pernambuco', 'Alagoas', 'Sergipe', 'Bahia', 'Minas Gerais', 'Espírito Santo', 
                                  'Rio de Janeiro', 'São Paulo', 'Paraná', 'Santa Catarina', 'Rio Grande do Sul', 'Mato Grosso do Sul', 'Mato Grosso', 
                                  'Goiás', 'Distrito Federal'],
        'Proporção de pessoas das famílias residentes (%)': [100.0, 0.8, 0.4, 1.9, 0.2, 4.1, 0.4, 0.7, 3.4, 1.6, 4.4, 1.7, 1.9, 4.5, 1.6, 1.1, 
                                                              7.1, 10.1, 1.9, 8.3, 21.9, 5.5, 3.4, 5.5, 1.3, 1.6, 3.3, 1.4],
        'Proporção de pessoas com algum grau de vulnerabilidade (%)': [63.8, 84.8, 89.0, 83.6, 72.3, 89.1, 88.5, 80.7, 93.3, 85.2, 78.9, 81.9, 
                                                                     81.2, 81.4, 85.3, 76.4, 79.9, 58.1, 58.8, 59.9, 45.7, 45.7, 40.0, 
                                                                     53.6, 69.0, 74.7, 69.2, 57.8],
        'Moradia': [7.7, 10.2, 15.1, 12.8, 8.4, 15.7, 13.5, 9.7, 17.4, 12.4, 10.1, 11.7, 12.1, 11.9, 13.1, 10.0, 11.3, 5.6, 5.9, 6.8, 3.9, 
                    3.8, 2.6, 4.7, 6.7, 7.9, 8.1, 5.7],
        'Acesso aos serviços de utilidade pública': [15.0, 13.5, 15.2, 16.5, 17.8, 15.0, 16.1, 15.3, 15.1, 14.2, 15.3, 14.6, 14.6, 15.1, 15.1, 
                                                    14.7, 13.5, 13.9, 15.7, 16.8, 16.5, 14.6, 13.6, 15.3, 14.9, 14.2, 14.1, 14.4]
    })
    
    # Merge dos dados com base na coluna 'Unidades da Federação'
    merged_data = pd.merge(data1, data2, on='Unidades da Federação', how='inner')
    
    return merged_data

# Exibir os dados
def show_data(merged_data):
    st.write("Dados Combinados (merge entre data1 e data2)")
    st.dataframe(merged_data)

# Algoritmo de previsão
def run_prediction(merged_data):
    target_column = "IPQV"  # Alvo a ser previsto
    
    if target_column not in merged_data.columns:
        st.error(f"A coluna {target_column} não foi encontrada no DataFrame.")
        return

    # Selecionar as variáveis independentes (removendo o alvo)
    features = merged_data.drop(columns=[target_column, 'Unidades da Federação'])

    # Separar os dados em treino e teste
    X = features
    y = merged_data[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Modelos para comparação
    models = {
        "Regressão Linear": LinearRegression(),
        "Árvore de Decisão": DecisionTreeRegressor(),
        "Floresta Aleatória": RandomForestRegressor()
    }

    predictions = {}
    
    for model_name, model in models.items():
        # Treinar o modelo
        model.fit(X_train, y_train)
        
        # Fazer previsões
        y_pred = model.predict(X_test)
        
        # Armazenar as previsões
        predictions[model_name] = (y_test, y_pred)
    
    return predictions

# Adicionar mapa interativo e exibir comparações
def plot_map(merged_data, predictions):
    # Carregar o arquivo GeoJSON
    with open('br_states.json', 'r', encoding='utf-8') as f:
        geojson_data = json.load(f)
    
    # Criar o mapa Folium
    m = folium.Map(location=[-14.2350, -51.9253], zoom_start=4)
    
    # Adicionar camada GeoJSON
    folium.GeoJson(geojson_data).add_to(m)
    
    # Exibir o mapa no Streamlit
    st_folium(m, width=725)
    
    # Adicionar interatividade para mostrar as comparações
    for model_name, (y_test, y_pred) in predictions.items():
        for state, real, pred in zip(merged_data['Unidades da Federação'], y_test, y_pred):
            st.write(f"Estado: {state}")
            st.write(f"Modelo: {model_name}")
            st.write(f"Real: {real}")
            st.write(f"Previsto: {pred}")
            st.write("------")

# Exibir o aplicativo
def main():
    merged_data = load_data()
    predictions = run_prediction(merged_data)
    
    # Mostrar os dados
    show_data(merged_data)
    
    # Mapa interativo
    plot_map(merged_data, predictions)

if __name__ == "__main__":
    main()
