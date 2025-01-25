# Classificação de Fraudes com Árvores de Decisão e Random Forest

## Descrição do Projeto
Este projeto utiliza algoritmos de aprendizado supervisionado para detectar fraudes em transações financeiras. A base de dados utilizada contém informações anonimizadas de transações de cartões de crédito, com variáveis numéricas geradas a partir de um PCA.

---

## Estrutura da Base de Dados

### Colunas Principais
- **Time**: Tempo decorrido em segundos entre cada transação.
- **V1 a V28**: Variáveis geradas por análise de componentes principais (PCA).
- **Amount**: Valor da transação.
- **Class**: Rótulo da transação (`0` para normal e `1` para fraude).

### Estatísticas da Base
- Total de transações: `284,807`
- Fraudes: `492` (0.17%)
- Transações normais: `284,315` (99.83%)

---

## Ferramentas e Bibliotecas Utilizadas
- **Pandas**: Manipulação e análise de dados.
- **NumPy**: Operações matemáticas e manipulação de arrays.
- **Scikit-learn**: Implementação de modelos de aprendizado de máquina e métricas.
- **Matplotlib**: Visualização de árvores de decisão.

---

## Fluxo de Trabalho

### 1. Divisão da Base
Os dados foram divididos em conjuntos de treino e teste utilizando `StratifiedShuffleSplit`, garantindo a proporção balanceada das classes.

```python
from sklearn.model_selection import StratifiedShuffleSplit

def executar_validador(X, y):
    validador = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    for treino_id, teste_id in validador.split(X, y):
        X_train, X_test = X[treino_id], X[teste_id]
        y_train, y_test = y[treino_id], y[teste_id]
    return X_train, X_test, y_train, y_test
```

### 2. Classificadores Utilizados
Utilizei o DecisionTreeClassifier com diferentes configurações de hiperparâmetros para avaliar o desempenho do modelo.

```python
from sklearn.tree import DecisionTreeClassifier

def testar_arvores(X_train, X_test, y_train, y_test):
    hiperparametros = [{'max_depth': None}, {'max_depth': 10}, {'max_depth': 5}]
    resultados = []

    for params in hiperparametros:
        clf = DecisionTreeClassifier(**params, random_state=0)
        clf.fit(X_train, y_train)

        acc = clf.score(X_test, y_test)
        precisao = precision_score(y_test, clf.predict(X_test))
        recall = recall_score(y_test, clf.predict(X_test))
        matriz_conf = confusion_matrix(y_test, clf.predict(X_test))

        resultados.append({'params': params, 'acc': acc, 'precisao': precisao, 'recall': recall, 'matriz_conf': matriz_conf})
    return resultados
```

### Random Forest
O RandomForestClassifier foi emmpregado para melhorar a estabilidade e a precisão do modelo.

```python
from sklearn.ensemble import RandomForestClassifier

def testar_random_forest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    precisao = precision_score(y_test, clf.predict(X_test))
    recall = recall_score(y_test, clf.predict(X_test))
    matriz_conf = confusion_matrix(y_test, clf.predict(X_test))

    resultados = {'acc': acc, 'precisao': precisao, 'recall': recall, 'matriz_conf': matriz_conf}
    return resultados
```

### 4. Visualização da Árvore de Decisão
A função plot_tree foi usada para gerar e salvar visualizações das árvores de decisão treinadas.

```python
import matplotlib.pyplot as plt
from sklearn import tree

def salvar_arvore(classificador, nome):
    plt.figure(figsize=(200, 100))
    tree.plot_tree(classificador, filled=True, fontsize=14)
    plt.savefig(nome)
    plt.close()
```

### Resultados Obtidos

### Árvores de Decisão
Configuração	Acurácia	Precisão	Recall	Matriz de Confusão
max_depth=None	0,9991	0,7347	0,7500	[[28420, 13], [12, 36]]
max_depth=10	0,9995	0,9474	0,7347	[[28430, 2], [13, 36]]
max_depth=5	0,9994	0,9211	0,7143	[[28429, 3], [14, 35]]

### Random Forest
Nº de Estimadores	Acurácia	Precisão	Recall	Matriz de Confusão
100	0,9995	0,9487	0,7551	[[28430, 2], [12, 37]]

### Conclusão

- **Árvores de Decisão**:
  -- Simples e interpretáveis.
  -- Bom desempenho, mas sensíveis ao ajuste de hiperparâmetros.

- **Random Forest**:
  -- Maior estabilidade e precisão geral.
  -- Ideal para detectar fraudes em bases de dados desbalanceadas.

