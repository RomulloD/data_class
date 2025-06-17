📈 Dashboard de Análise e Previsão de Evasão Escolar
Este projeto consiste em um dashboard interativo desenvolvido para analisar e prever o risco de evasão escolar. Utiliza técnicas de Machine Learning e visualização de dados para fornecer insights valiosos sobre o comportamento dos alunos, ajudando instituições de ensino a identificar e intervir proativamente com estudantes em risco.



🚀 Tecnologias Utilizadas
O projeto foi construído utilizando as seguintes tecnologias principais:

Python: A linguagem de programação principal para o desenvolvimento do modelo de Machine Learning e da aplicação web.

Pandas: Biblioteca fundamental para manipulação e análise de dados. Utilizada para carregar, limpar, transformar e preparar o conjunto de dados de alunos.

Scikit-learn: A biblioteca de Machine Learning mais popular do Python. Foi empregada para:

Pré-processamento de Dados: StandardScaler para padronizar as features numéricas e pd.get_dummies para realizar one-hot encoding em variáveis categóricas, garantindo que os dados estejam no formato adequado para o modelo.

Modelagem Preditiva: RandomForestClassifier (ou KNeighborsClassifier, dependendo da configuração final do modelo salvo) para treinar um modelo capaz de prever o risco de evasão (evasao = 1) com base em características dos alunos como frequência e nota, e outras variáveis do dataset.

Avaliação do Modelo: classification_report e confusion_matrix para avaliar o desempenho do modelo em termos de precisão, recall, F1-score e acurácia.

Joblib: Utilizado para serializar (salvar) e desserializar (carregar) o modelo de Machine Learning treinado, o scaler e a lista de colunas utilizadas no treinamento. Isso permite que o modelo seja treinado uma vez e reutilizado no dashboard sem a necessidade de re-treinamento a cada inicialização.

Dash by Plotly: Um framework Python para a construção de dashboards analíticos interativos. O Dash foi essencial para:

Construção da Interface do Usuário (UI): Componentes dcc (Dash Core Components) e html (Dash HTML Components) foram usados para criar a estrutura do layout do dashboard, incluindo dropdowns de filtro, gráficos e tabelas.

Interatividade: Callbacks foram implementados para permitir que os usuários filtrem os dados por escola, série e turno, e para selecionar alunos individualmente para obter previsões de evasão em tempo real.

Plotly Express: Uma biblioteca de gráficos de alto nível integrada ao Dash. Utilizada para criar visualizações de dados dinâmicas e esteticamente agradáveis, como histogramas e gráficos de dispersão, que ajudam a entender a distribuição da evasão e a relação entre frequência e nota.

⚙️ Funcionalidades
Carregamento Dinâmico do Modelo: O dashboard verifica a existência de um modelo treinado e o carrega; caso contrário, realiza um treinamento rápido para garantir a funcionalidade.

Filtros Interativos: Permite filtrar dados por escola, série e turno para uma análise segmentada.

Visualizações Gráficas: Apresenta gráficos de evasão e de relação entre nota e frequência, com destaque para alunos em risco.

Lista de Alunos em Risco: Exibe uma tabela com os 10 alunos com maior risco de evasão, baseada nos filtros aplicados.

Previsão Individual de Evasão: Possibilita selecionar um aluno específico e obter uma previsão de risco de evasão em tempo real, juntamente com a probabilidade associada.

🛠️ Como Executar o Projeto Localmente
Clone o Repositório:

git clone [URL_DO_SEU_REPOSITORIO]
cd [NOME_DA_PASTA_DO_PROJETO]

Crie um Ambiente Virtual (Recomendado):

python -m venv venv
# No Windows:
.\venv\Scripts\activate
# No macOS/Linux:
source venv/bin/activate

Instale as Dependências:

pip install pandas scikit-learn dash plotly

Certifique-se de ter os dados: O arquivo alunos_completos.csv deve estar na mesma pasta do script principal do dashboard.

Execute o Dashboard:

python dashboard_evasao.py

(Substitua dashboard_evasao.py pelo nome do seu arquivo Python principal do dashboard)

Acesse o Dashboard: Abra seu navegador e navegue para a URL exibida no terminal (geralmente http://127.0.0.1:8050/).

🤝 Contribuição
Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.
