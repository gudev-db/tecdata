import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

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
    
    return data1, data2

# Exibir os dataframes no Streamlit
def show_dataframes(data1, data2):
    st.write("Data1 - Dados Socioeconômicos")
    st.dataframe(data1)
    st.write("Data2 - Índices de Qualidade de Vida")
    st.dataframe(data2)

# Estatísticas e Gráficos
def plot_statistics(data1):
    st.write("Estatísticas Descritivas do Conjunto de Dados:")
    st.dataframe(data1.describe())
    
    # Gráfico de Barra
    st.write("Gráfico de Barra das variáveis socioeconômicas")
    data1.drop(columns=['Unidades da Federação']).mean().plot(kind='bar', color='skyblue', title="Média das Variáveis Socioeconômicas")
    st.pyplot()

# Análise de Correlação
def plot_correlation(data1):
    st.write("Correlação entre as variáveis:")
    corr = data1.drop(columns=['Unidades da Federação']).corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot()

# Algoritmo de previsão
def run_prediction(data1):
    target_column = "IPQV"  # Alvo a ser previsto
    
    if target_column not in data1.columns:
        st.error(f"A coluna {target_column} não foi encontrada em 'data1'.")
        return

    # Selecionar as variáveis independentes (remover a coluna alvo)
    X = data1.drop(columns=[target_column, 'Unidades da Federação'])
    y = data1[target_column]
    
    # Normalizar os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Modelos de Regressão
    models = {
        "Regressão Linear": LinearRegression(),
        "Árvore de Decisão": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor()
    }
    
    mse_scores = {}
    
    # Treinar e avaliar os modelos
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_scores[name] = mse

    # Gráfico de comparação de MSE
    st.write("Comparação de MSE entre os Modelos")
    fig, ax = plt.subplots()
    ax.bar(mse_scores.keys(), mse_scores.values(), color='skyblue')
    ax.set_title('MSE dos Modelos')
    ax.set_ylabel('MSE')
    st.pyplot()
    
    return mse_scores

# Função para filtrar dados por estado e exibir no mapa
def filter_and_show_map(data1):
    st.sidebar.header("Filtrar por Estado")
    estados = data1['Unidades da Federação'].unique()
    selected_state = st.sidebar.selectbox("Selecione um estado", estados)
    
    filtered_data = data1[data1['Unidades da Federação'] == selected_state]
    
    if not filtered_data.empty:
        st.write(f"Dados para o estado de {selected_state}:")
        st.dataframe(filtered_data)
        
        # Coordenadas aproximadas dos estados brasileiros (latitude, longitude)
        coordenadas_estados = {
            'Brasil': [-14.2350, -51.9253],
            'Rondônia': [-11.5057, -63.5806],
            'Acre': [-9.0238, -70.8120],
            'Amazonas': [-3.4653, -62.2159],
            'Roraima': [2.7376, -62.0751],
            'Pará': [-5.4984, -52.9564],
            'Amapá': [1.4136, -51.5811],
            'Tocantins': [-10.1753, -48.2982],
            'Maranhão': [-4.9609, -45.2744],
            'Piauí': [-7.7183, -42.7289],
            'Ceará': [-5.4984, -39.3206],
            'Rio Grande do Norte': [-5.7945, -36.5947],
            'Paraíba': [-7.2396, -36.7827],
            'Pernambuco': [-8.8137, -36.9541],
            'Alagoas': [-9.5713, -36.7819],
            'Sergipe': [-10.5741, -37.3857],
            'Bahia': [-12.5797, -41.7007],
            'Minas Gerais': [-18.5122, -44.5550],
            'Espírito Santo': [-19.1834, -40.3089],
            'Rio de Janeiro': [-22.9068, -43.1729],
            'São Paulo': [-23.5505, -46.6333],
            'Paraná': [-24.7953, -51.7645],
            'Santa Catarina': [-27.2423, -50.2189],
            'Rio Grande do Sul': [-30.0346, -51.2177],
            'Mato Grosso do Sul': [-20.7722, -54.7852],
            'Mato Grosso': [-12.6819, -56.9211],
            'Goiás': [-15.8270, -49.8362],
            'Distrito Federal': [-15.7942, -47.8825]
        }
        
        # Criar o mapa
        mapa = folium.Map(location=coordenadas_estados[selected_state], zoom_start=6)
        
        # Adicionar marcador para o estado selecionado
        folium.Marker(
            location=coordenadas_estados[selected_state],
            popup=f"{selected_state}",
            icon=folium.Icon(color='blue')
        ).add_to(mapa)
        
        # Exibir o mapa no Streamlit
        st.write(f"Mapa do estado de {selected_state}:")
        folium_static(mapa)

# Interface Streamlit
def main():
    st.title("Análise de Dados Socioeconômicos e Qualidade de Vida")
    
    # Carregar os dados
    data1, data2 = load_data()
    
    # Mostrar os dataframes
    show_dataframes(data1, data2)
    
    # Estatísticas e gráficos
    plot_statistics(data1)
    
    # Análise de correlação
    plot_correlation(data1)
    
    # Rodar previsão
    mse_scores = run_prediction(data1)
    st.write("MSE dos Modelos de Previsão:", mse_scores)
    
    # Filtrar por estado e exibir no mapa
    filter_and_show_map(data1)

# Executar a aplicação Streamlit
if __name__ == "__main__":
    main()