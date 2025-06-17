import os
import joblib
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
import dash
from dash import dcc, html, Input, Output
import plotly.express as px

df = pd.read_csv("alunos_completos.csv")

MODELO_PATH = "modelo_randomforest.pkl" 
SCALER_PATH = "scaler.pkl"
COLUMNS_PATH = "colunas_treinadas.pkl"

modelo = None
scaler = None
trained_columns = None
numerical_cols = ['frequencia', 'nota'] 

if os.path.exists(MODELO_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(COLUMNS_PATH):
    try:
        modelo = joblib.load(MODELO_PATH)
        scaler = joblib.load(SCALER_PATH)
        trained_columns = joblib.load(COLUMNS_PATH)
        print("✅ Modelo, scaler e colunas carregados com sucesso.")
    except Exception as e:
        print(f"❌ Erro ao carregar os arquivos do modelo: {e}")
        print("⚠️  Re-treinando o modelo devido ao erro de carregamento.")
        modelo = None
else:
    print("⚠️  Modelos, scaler ou colunas não encontrados… treinando um novo modelo.")
    X_train_full = df.drop(["evasao", "MATRICULA", "NOME", "DATA_NASCIMENTO"], axis=1)
    y_train_full = df["evasao"]

    X_train_full = pd.get_dummies(X_train_full, drop_first=True)
    
    trained_columns = X_train_full.columns.tolist()
    joblib.dump(trained_columns, COLUMNS_PATH)

    X_temp_train, X_temp_test, y_temp_train, y_temp_test = train_test_split(
        X_train_full, y_train_full, test_size=0.25, stratify=y_train_full, random_state=42
    )

    scaler = StandardScaler()
    X_temp_train[numerical_cols] = scaler.fit_transform(X_temp_train[numerical_cols])
    X_temp_test[numerical_cols] = scaler.transform(X_temp_test[numerical_cols])
    joblib.dump(scaler, SCALER_PATH) 

    from sklearn.ensemble import RandomForestClassifier 
    modelo = RandomForestClassifier(n_estimators=300, max_depth=None, class_weight="balanced", random_state=42)
    modelo.fit(X_temp_train, y_temp_train)
    joblib.dump(modelo, MODELO_PATH) 
    print("✅ Modelo, scaler e colunas treinados e salvos com sucesso.")


dark_style = {
    "backgroundColor": "#1a1a2e",  
    "color": "#e0e0e0",           
    "fontFamily": "Roboto, sans-serif", 
    "padding": "25px",
    "minHeight": "100vh"          
}
card_style = {
    "backgroundColor": "#1f4068", 
    "padding": "25px",
    "borderRadius": "12px",       
    "marginBottom": "25px",
    "boxShadow": "0 4px 8px 0 rgba(0, 0, 0, 0.3)", 
    "border": "1px solid #162447" 
}
dropdown_style = {
    "color": "#333",              
    "backgroundColor": "#f0f0f0", 
    "borderRadius": "8px"
}
table_header_style = {
    "padding": "10px",
    "textAlign": "left",
    "backgroundColor": "#162447", 
    "color": "white",
    "borderBottom": "2px solid #e0e0e0"
}
table_cell_style = {
    "padding": "8px",
    "textAlign": "left",
    "borderBottom": "1px solid #2e5180" 
}


app = dash.Dash(__name__)
app.title = "Dashboard de Evasão Escolar"

app.layout = html.Div(style=dark_style, children=[

    html.H1("📊 Análise de Evasão Escolar", style={"textAlign": "center", "color": "#00ffc4", "marginBottom": "30px"}),

    html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "25px"}, children=[ 

        html.Div(children=[
            html.H3("Filtros", style={"marginBottom": "15px", "color": "#e0e0e0"}),
            html.Label("🎓 Escola:", style={"color": "#b0b0b0"}),
            dcc.Dropdown(
                options=[{"label": esc, "value": esc} for esc in df["ESCOLA"].dropna().unique()],
                id="filtro_escola",
                multi=True,
                placeholder="Selecione a escola",
                style=dropdown_style
            ),
            html.Br(),
            html.Label("📘 Série:", style={"color": "#b0b0b0"}),
            dcc.Dropdown(
                options=[{"label": s, "value": s} for s in df["SERIE"].dropna().unique()],
                id="filtro_serie",
                multi=True,
                placeholder="Selecione a série",
                style=dropdown_style
            ),
            html.Br(),
            html.Label("🕐 Turno:", style={"color": "#b0b0b0"}),
            dcc.Dropdown(
                options=[{"label": t, "value": t} for t in df["TURNO"].dropna().astype(str).unique()],
                id="filtro_turno",
                multi=True,
                placeholder="Selecione o turno",
                style=dropdown_style
            ),
        ], style={**card_style, "flex": "1", "minWidth": "300px"}),

        html.Div(children=[
            dcc.Graph(id="grafico_evasao"),
            dcc.Graph(id="grafico_frequencia_vs_nota"),
            html.H3("Alunos com Maior Risco de Evasão", style={"marginTop": "20px", "color": "#e0e0e0"}),
            html.Div(id="tabela_top_evasao")
        ], style={**card_style, "flex": "2", "minWidth": "600px"}),
    ]),

    html.Div(children=[
        html.H2("🔮 Previsão Individual de Evasão", style={"color": "#00ffc4"}),
        html.Label("Selecione o aluno:", style={"color": "#b0b0b0"}),
        dcc.Dropdown(
            options=[{"label": nome, "value": nome} for nome in df["NOME"].unique()],
            id="dropdown_aluno",
            placeholder="Escolha um aluno",
            style=dropdown_style
        ),
        html.Br(),
        html.Div(id="saida_previsao_aluno")
    ], style={**card_style, "marginTop": "25px"})
])


@app.callback(
    [Output("grafico_evasao", "figure"),
     Output("grafico_frequencia_vs_nota", "figure"),
     Output("tabela_top_evasao", "children")],
    [Input("filtro_escola", "value"),
     Input("filtro_serie", "value"),
     Input("filtro_turno", "value")]
)
def atualizar_dashboard(f_esc, f_serie, f_turno):
    df_filtrado = df.copy()
    if f_esc:
        df_filtrado = df_filtrado[df_filtrado["ESCOLA"].isin(f_esc)]
    if f_serie:
        df_filtrado = df_filtrado[df_filtrado["SERIE"].isin(f_serie)]
    if f_turno:
        df_filtrado = df_filtrado[df_filtrado["TURNO"].astype(str).isin(f_turno)]

    fig_evasao = px.histogram(
        df_filtrado, x="evasao", color="evasao",
        labels={"evasao": "Evasão (0 = Não, 1 = Sim)"},
        title="Distribuição de Evasão",
        template="plotly_dark",
        color_discrete_map={0: '#00ffc4', 1: '#ff4c4c'} 
    )
    fig_evasao.update_layout(
        plot_bgcolor="#1f4068", 
        paper_bgcolor="#1f4068",
        font_color="#e0e0e0"     
    )

    fig_scatter = px.scatter(
        df_filtrado, x="frequencia", y="nota", color="evasao",
        hover_data=["MATRICULA", "NOME"],
        title="Nota vs Frequência (colorido por evasão)",
        template="plotly_dark",
        color_discrete_map={0: '#00ffc4', 1: '#ff4c4c'}
    )
    fig_scatter.update_layout(
        plot_bgcolor="#1f4068",
        paper_bgcolor="#1f4068",
        font_color="#e0e0e0"
    )

    top = (df_filtrado[df_filtrado["evasao"] == 1]
           .sort_values(by=["frequencia", "nota"])
           .head(10))

    tabela = html.Table(
        [html.Tr([html.Th(c, style=table_header_style) for c in ["MATRICULA", "NOME", "frequencia", "nota"]])] +
        [html.Tr([html.Td(top.iloc[i][c], style=table_cell_style) for c in ["MATRICULA", "NOME", "frequencia", "nota"]])
         for i in range(len(top))],
        style={"width": "100%", "color": "#e0e0e0", "borderRadius": "8px", "overflow": "hidden"}
    )

    return fig_evasao, fig_scatter, tabela


@app.callback(
    Output("saida_previsao_aluno", "children"),
    Input("dropdown_aluno", "value")
)
def prever_evasao_aluno(nome):
    if not nome:
        return html.Div("Selecione um aluno para ver a previsão.", style={"color": "#b0b0b0"})
    
    if modelo is None or scaler is None or trained_columns is None:
        return html.Div("Modelo de ML não carregado/treinado. Por favor, verifique o console ou reinicie o aplicativo.", 
                        style={"color": "#ff4c4c", "fontWeight": "bold"})

    aluno = df[df["NOME"] == nome]
    if aluno.empty:
        return html.Div("Aluno não encontrado.", style={"color": "#ff4c4c"})


    X_aluno = aluno.drop(columns=["evasao", "MATRICULA", "NOME", "DATA_NASCIMENTO"])
    
    X_aluno_processed = pd.get_dummies(X_aluno, drop_first=True)
    

    X_aluno_final = X_aluno_processed.reindex(columns=trained_columns, fill_value=0)

    X_aluno_final[numerical_cols] = scaler.transform(X_aluno_final[numerical_cols])

    # Fazer a previsão
    previsao = modelo.predict(X_aluno_final)[0]
    prob = modelo.predict_proba(X_aluno_final)[0][1] 

    cor = "#ff4c4c" if previsao == 1 else "#00ffc4"
    texto_previsao = "Sim" if previsao == 1 else "Não"
    texto_emoji = "🚨" if previsao == 1 else "✅"

    return html.Div([
        html.P(f"Aluno: {nome}", style={"fontSize": "1.1em"}),
        html.P(f"Frequência: {float(aluno['frequencia']):.1f}%"),
        html.P(f"Nota: {float(aluno['nota']):.1f}"),
        html.P(f"{texto_emoji} Risco de Evasão: {texto_previsao} (Probabilidade: {prob:.1%})", 
               style={"color": cor, "fontWeight": "bold", "fontSize": "1.2em"})
    ])

if __name__ == "__main__":
    app.run(debug=True)
