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
#
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

# Exibir o DataFrame merge
def show_data(merged_data):
    st.write("Dados Combinados (merge entre data1 e data2)")
    st.dataframe(merged_data)

def plot_statistics(merged_data):
    st.write("Estatísticas Descritivas do Conjunto de Dados Combinados:")
    st.dataframe(merged_data.describe())
    
    # Verificar se um estado foi selecionado
    if 'selected_state' in st.session_state and st.session_state.selected_state:
        selected_state = st.session_state.selected_state
        
        # Filtrar os dados para o estado selecionado
        filtered_data = merged_data[merged_data['Unidades da Federação'] == selected_state]
        
        if not filtered_data.empty:
            # Exibir a linha inteira de dados para o estado selecionado
            st.write(f"Dados para o estado {selected_state}:")
            st.dataframe(filtered_data)
        else:
            st.write(f"Nenhum dado encontrado para o estado {selected_state}.")
    else:
        st.write("Selecione um estado no mapa para ver os dados completos.")
    
    # Gráfico de Barra de IPQV por Estado
    st.write("IPQV por Estado")
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Unidades da Federação', y='IPQV', data=merged_data, palette="viridis")
    plt.xticks(rotation=90)
    plt.title('IPQV por Estado')
    st.pyplot(plt)


def plot_map(merged_data):
    # Carregar o arquivo GeoJSON
    with open('br_states.json', 'r', encoding='utf-8') as f:
        geojson_data = json.load(f)
    
    # Criar o mapa Folium
    m = folium.Map(location=[-14.2350, -51.9253], zoom_start=4)
    
    # Função para capturar o clique no estado
    def on_click(event):
        # Extrair o nome do estado a partir da feature do GeoJSON
        state_name = event['properties']['name']  # Nome do estado clicado no GeoJSON
        st.session_state.selected_state = state_name  # Armazenar no estado da sessão
        st.write(f"Estado selecionado: {state_name}")
        
        # Filtrar os dados para o estado selecionado
        filtered_data = merged_data[merged_data['Unidades da Federação'] == state_name]
        
        # Exibir o valor real de IPQV para o estado selecionado
        if not filtered_data.empty:
            real_ipqv = filtered_data['IPQV'].values[0]
            st.session_state.real_ipqv = real_ipqv  # Armazenar o IPQV no estado da sessão
        else:
            st.session_state.real_ipqv = None
    
    # Adicionar camada GeoJSON com interatividade
    folium.GeoJson(
        geojson_data, 
        name="Brasil", 
        tooltip="Clique para selecionar o estado",
        highlight_function=lambda x: {'weight': 3, 'color': 'blue'},
        popup=folium.Popup("Clique no estado", max_width=300)
    ).add_to(m)
    
    # Exibir o mapa no Streamlit
    st_folium(m, width=725)
    
    # Mostrar o valor de IPQV do estado selecionado abaixo do mapa
    if 'selected_state' in st.session_state and st.session_state.selected_state:
        selected_state = st.session_state.selected_state
        st.write(f"IPQV no estado {selected_state}: {st.session_state.real_ipqv:.3f}" if st.session_state.real_ipqv else "Dados não encontrados para o estado.")


def plot_correlation_matrix(merged_data):
    st.write("Matriz de Correlação entre as Variáveis Numéricas")

    # Selecionar apenas colunas numéricas
    numeric_cols = merged_data.select_dtypes(include=['float64', 'int64'])

    # Calcular a matriz de correlação
    correlation_matrix = numeric_cols.corr()

    # Exibir a matriz de correlação como uma tabela
    st.write(correlation_matrix)

    # Plotar a matriz de correlação com Seaborn
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Matriz de Correlação")
    st.pyplot(plt)



def run_prediction(merged_data):
    target_column = "IPQV"
    if target_column not in merged_data.columns:
        st.error(f"A coluna {target_column} não foi encontrada no DataFrame.")
        return

    features = merged_data.drop(columns=[target_column, 'Unidades da Federação'])
    if 'Acesso aos serviços de utilidade pública' in features.columns:
        features = features.drop(columns=['Acesso aos serviços de utilidade pública'])
    
    X = features
    y = merged_data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Regressão Linear": LinearRegression(),
        "Árvore de Decisão": DecisionTreeRegressor(),
        "Floresta Aleatória": RandomForestRegressor()
    }

    for model_name, model in models.items():
        st.write(f"Modelo: {model_name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        st.write(f"Erro Médio Quadrático (MSE): {mse:.4f}")
        st.write(f"Raiz do Erro Médio Quadrático (RMSE): {rmse:.4f}")
        
        st.scatter_chart(pd.DataFrame({"Valores Reais": y_test, "Valores Previstos": y_pred}))




# Exibir o aplicativo
def main():
    merged_data = load_data()
    
    # Mostrar os dados
    show_data(merged_data)
    
    # Estatísticas e Gráficos
    plot_statistics(merged_data)
    
    # Mapa
    plot_map(merged_data)

    # Matriz de Correlação
    plot_correlation_matrix(merged_data)
    
    # Previsão
    run_prediction(merged_data)

if __name__ == "__main__":
    main()
