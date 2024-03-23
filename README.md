Churn prediction for ERP
==============================

Esse projeto tem como objetivo conseguir construir um modelo que faça a predição de Churn para uma empresa de ERP. Realizar essa previsão é importante para empresas pois permite acompanhar comportamentos do cliente que indiquem um futuro desligamento e assim agir para evitar que isso aconteça realizando algumas ações como descontos em produtos ou mesmo lembrando de reassinar o contrato.


## Como esse projeto está organizado?

1. O conjunto de dados:
O conjunto de dados utilizado está na pasta: `data/raw/custumer_churn_data-custumer_churn_data.csv`. Ele contém 22 features relacionadas ao tempo de permanência do cliente,
frequência de utilização de features, receita mensal, receita total e informações da empresa (ano de fundação, tipo de empresa e etc.)

2. Sobre os requisitos para rodar esse projeto:
O projeto em si foi baseado na venv conda. Para gerar esse ambiente localmente basta rodar `make create_environment`. Caso não seja possível, utilize o arquivo `requirements.txt` e rode o comando `pip install -r requirements.txt`

3. Da organização dos arquivos:

Esse projeto se baseou na estrutura do cookiecutter como organização dos arquivos. 


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

Scripts e diretórios construídos/utilizados:

 - `make_dataset.py`: realiza o carregamento do nosso dataset base e também realiza limpeza nos dados (conversão de algumas colunas de string para float, remoção de colunas repetidas ou sem informação relevante e renomeia colunas para as deixar padronizadas), os dados aqui são salvos para a pasta `raw/processed`. Para rodar esse script basta executar `python make_dataset.py [dir/to/get/raw_data] [dir/to/send/clear/data]`.
 - `build_features.py`: nesse arquivo contém todas as funções necessárias para realizar o feature engineering do nosso dataset base. 
 - `predict_model.py`: nesse arquivo temos a função `make_predict` que realiza as predições dos modelos e retorna tanto valores em probabilidades quanto as classes previstas.
 - diretório `notebooks`: nele contém todos os notebooks construídos desse projeto em ordem de construção, o processo se segue: EDA > construção de features > criação dos modelos baseline > criação dos modelos otimizados > avaliação de resultados.
 - `helper.py`: contém funções para fazer plot da matrix de confusão e avaliação de métricas. Está dentro do dir de notebooks

## Resultado obtido

A principal métrica a ser avaliada é a f1-score pois ela garante que atingirmos um bom equilíbrio entre falsos positivos e falsos negativos. Isso porque evita grandes gastos para manter clientes (falsos positivos) e evita perda em receita (falsos negativos).

Em resumo, o modelo otimizado se baseou no RandomForestClassifier obtendo 0.66 de f1-score. Em relação aos modelos baselines, isso representou um ganho de 123% e 33% em relação ao modelo heurístico (baseado em regra de negócio) e decision tree respectivamente.

## Sobre experimentos 

Por fim, esse projeto inclui arquivos sobre futura aplicação desse projeto em produção dentro da pasta `./experiments`
