import folium
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import StandardScaler
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

# Exibir os dados combinados
def show_data(merged_data):
    st.write("Dados Combinados (merge entre data1 e data2)")
    st.dataframe(merged_data)

# Função para remover as 3 colunas com menor correlação com o IPQV
def remove_weak_correlations(merged_data):
    corr_matrix = merged_data.corr()
    ipqv_corr = corr_matrix["IPQV"].abs()
    weak_columns = ipqv_corr.nsmallest(4).index[1:].tolist()  # Remove "IPQV" da lista
    reduced_data = merged_data.drop(columns=weak_columns)
    return reduced_data, weak_columns

# Pré-processamento e previsão
def run_prediction(merged_data, reduced_data):
    target_column = "IPQV"  # Alvo a ser previsto
    
    if target_column not in merged_data.columns:
        st.error(f"A coluna {target_column} não foi encontrada no DataFrame.")
        return

    # Selecionar as variáveis independentes (removendo o alvo)
    features_merged = merged_data.drop(columns=[target_column, 'Unidades da Federação'])
    features_reduced = reduced_data.drop(columns=[target_column, 'Unidades da Federação'])

    # Separar os dados em treino e teste
    X_merged = features_merged
    y_merged = merged_data[target_column]
    
    X_reduced = features_reduced
    y_reduced = reduced_data[target_column]
    
    X_train_merged, X_test_merged, y_train_merged, y_test_merged = train_test_split(X_merged, y_merged, test_size=0.2, random_state=42)
    X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(X_reduced, y_reduced, test_size=0.2, random_state=42)
    
    # StandardScaler
    scaler = StandardScaler()
    X_train_merged = scaler.fit_transform(X_train_merged)
    X_test_merged = scaler.transform(X_test_merged)
    
    X_train_reduced = scaler.fit_transform(X_train_reduced)
    X_test_reduced = scaler.transform(X_test_reduced)
    
    # Modelos para comparação
    models = {
        "Regressão Linear": LinearRegression(),
        "Árvore de Decisão": DecisionTreeRegressor(),
        "Floresta Aleatória": RandomForestRegressor()
    }

    mse_results = {"Merged Data": [], "Reduced Data": []}
    
    for model_name, model in models.items():
        st.write(f"Modelo: {model_name}")
        
        # Treinar os modelos para os dois DataFrames
        model.fit(X_train_merged, y_train_merged)
        y_pred_merged = model.predict(X_test_merged)
        mse_merged = mean_squared_error(y_test_merged, y_pred_merged)
        mse_results["Merged Data"].append(mse_merged)
        
        model.fit(X_train_reduced, y_train_reduced)
        y_pred_reduced = model.predict(X_test_reduced)
        mse_reduced = mean_squared_error(y_test_reduced, y_pred_reduced)
        mse_results["Reduced Data"].append(mse_reduced)
        
        st.write(f"Erro Médio Quadrático (MSE) para {model_name} com dados completos: {mse_merged:.4f}")
        st.write(f"Erro Médio Quadrático (MSE) para {model_name} com dados reduzidos: {mse_reduced:.4f}")
    
    # Comparar os MSEs
    mse_df = pd.DataFrame(mse_results, index=[model_name for model_name in models.keys()])
    st.write("Comparação dos MSEs entre os modelos")
    st.dataframe(mse_df)

# Exibir o aplicativo
def main():
    merged_data = load_data()
    
    # Mostrar os dados
    show_data(merged_data)
    
    # Exibir o mapa
    st.write("Mapa Interativo com as Unidades da Federação")
    mapa = folium.Map(location=[-15.7801, -47.9292], zoom_start=4)
    
    # Adicionando marcadores de exemplo (a localização pode ser alterada conforme necessário)
    for idx, row in merged_data.iterrows():
        folium.Marker([row["IPQV"], row["B"]], popup=row["Unidades da Federação"]).add_to(mapa)
    
    # Exibir o mapa no Streamlit
    st_folium(mapa, width=700, height=500)

    # Remover colunas com as correlações mais fracas
    reduced_data, weak_columns = remove_weak_correlations(merged_data)
    st.write(f"Colunas removidas por correlação fraca com IPQV: {weak_columns}")
    
    # Previsão com os dois DataFrames
    run_prediction(merged_data, reduced_data)

if __name__ == "__main__":
    main()
