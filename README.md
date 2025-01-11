# Documentação: Classificação de Fraudes com Árvores de Decisão e Random Forest

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
