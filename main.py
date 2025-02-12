import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
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

# Coordenadas aproximadas dos estados do Brasil
coordenadas_estados = {
    'Brasil': [-14.235, -51.9253],
    'Rondônia': [-10.83, -63.34],
    'Acre': [-9.02, -70.81],
    'Amazonas': [-3.07, -61.66],
    'Roraima': [2.05, -61.40],
    'Pará': [-3.79, -52.48],
    'Amapá': [1.41, -51.77],
    'Tocantins': [-10.25, -48.25],
    'Maranhão': [-5.42, -45.44],
    'Piauí': [-7.07, -42.80],
    'Ceará': [-5.20, -39.53],
    'Rio Grande do Norte': [-5.81, -36.59],
    'Paraíba': [-7.12, -36.72],
    'Pernambuco': [-8.38, -37.86],
    'Alagoas': [-9.57, -36.78],
    'Sergipe': [-10.57, -37.45],
    'Bahia': [-12.97, -41.39],
    'Minas Gerais': [-18.10, -44.38],
    'Espírito Santo': [-19.19, -40.34],
    'Rio de Janeiro': [-22.91, -43.20],
    'São Paulo': [-23.55, -46.64],
    'Paraná': [-25.25, -52.02],
    'Santa Catarina': [-27.33, -50.45],
    'Rio Grande do Sul': [-30.03, -51.22],
    'Mato Grosso do Sul': [-20.51, -54.54],
    'Mato Grosso': [-12.64, -55.42],
    'Goiás': [-15.82, -49.22],
    'Distrito Federal': [-15.78, -47.93]
}

@st.cache
def load_data():
    """Carregar os dados fictícios."""
    data1 = pd.DataFrame({
        'Unidades da Federação': list(coordenadas_estados.keys()),
        'IPQV': np.random.uniform(0.1, 0.3, len(coordenadas_estados)),  # Simulação de IPQV
    })
    return data1

def plot_map(data1):
    """Gerar um mapa interativo com Folium."""
    st.write("### Mapa Interativo - Índice de Qualidade de Vida por Estado")

    # Criar o mapa centralizado no Brasil
    mapa = folium.Map(location=[-14.235, -51.9253], zoom_start=4)

    # Criar um cluster de marcadores
    marker_cluster = MarkerCluster().add_to(mapa)

    # Adicionar marcadores para cada estado
    for _, row in data1.iterrows():
        estado = row['Unidades da Federação']
        ipqv = row['IPQV']
        
        if estado in coordenadas_estados:
            lat, lon = coordenadas_estados[estado]
            popup_text = f"<b>{estado}</b><br>IPQV: {ipqv:.3f}"
            folium.Marker(
                location=[lat, lon],
                popup=popup_text,
                icon=folium.Icon(color="blue", icon="info-sign")
            ).add_to(marker_cluster)

    # Exibir o mapa no Streamlit
    folium_static(mapa)

def main():
    st.title("Análise de Dados Socioeconômicos e Qualidade de Vida")
    
    # Carregar os dados
    data1 = load_data()
    
    # Mostrar o DataFrame
    st.write("## Dados Carregados")
    st.dataframe(data1)
    
    # Exibir o mapa interativo
    plot_map(data1)
    
    # Gráficos
    st.write("## Distribuição do IPQV")
    fig, ax = plt.subplots()
    sns.histplot(data1['IPQV'], bins=10, kde=True, ax=ax)
    st.pyplot(fig)

    # Regressão linear
    X = np.array(data1.index).reshape(-1, 1)
    y = data1['IPQV']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    st.write("## Regressão Linear - Predição do IPQV")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel("Valores Reais")
    ax.set_ylabel("Predições")
    ax.set_title("Regressão Linear: Predição vs Real")
    st.pyplot(fig)

    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Erro Quadrático Médio (MSE): {mse:.5f}")

if __name__ == "__main__":
    main()
