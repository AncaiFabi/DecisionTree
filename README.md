# 🌳 Classificação de Eventos de Câncer de Mama com Árvore de Decisão

##  Descrição do Projeto

Este trabalho de Data Science foca na construção de um modelo de **Classificação** para prever a recorrência de eventos de câncer de mama. O projeto utiliza o algoritmo **Árvore de Decisão (\`DecisionTreeClassifier\`)** com seus parâmetros padrões para estabelecer uma linha de base de desempenho.

### Dataset

* **Arquivo:** \`breast-cancer.csv\` (Baseado no dataset UCI - Breast Cancer Wisconsin Original).
* **Classe (Target):** Coluna \`Class\`, que contém as classes \`recurrence-events\` e \`no-recurrence-events\`.

---

##  Metodologia e Processamento de Dados

O processo de modelagem seguiu uma abordagem estruturada de pré-processamento, treinamento e avaliação:

### 1. Pré-processamento e Limpeza

1.  **Tratamento de Missing Values:** Valores ausentes, marcados com \`?\` no CSV, foram identificados e as instâncias (linhas) incompletas foram removidas, resultando em um total de **277 instâncias limpas**.
2.  **One-Hot Encoding (Codificação Categórica):** Como o \`DecisionTreeClassifier\` exige dados numéricos e a maioria dos atributos é categórica (ex: \`age\`, \`tumor-size\`), a técnica **One-Hot Encoding** (\`pd.get_dummies\`) foi aplicada. Isso transformou os 9 atributos originais em **39 colunas binárias** (0 ou 1) que representam as categorias.
3.  **Divisão Hold-Out:** O conjunto de dados foi dividido em treino (70%) e teste (30%) para simular a performance do modelo em dados não vistos (\`random_state=42\` garante a reprodutibilidade).

### 2. Treinamento e Avaliação

1.  **Treinamento do Modelo:** O \`DecisionTreeClassifier\` foi instanciado e treinado utilizando os **parâmetros padrão (default)** da Scikit-learn, sem ajustes de hiperparâmetros.
2.  **Previsão e Avaliação:** O modelo treinado foi testado no conjunto de teste (\`X_test\`) e avaliado usando a **Acurácia** e a **Matriz de Confusão**.

---

##  Resultados do Modelo

Os resultados refletem o desempenho do modelo de Árvore de Decisão com a configuração padrão no conjunto de teste.

### Desempenho no Conjunto de Teste

| Métrica | Valor |
| :--- | :--- |
| **Acurácia do Modelo** | **75.00%** |
| **Total de Instâncias no Teste** | 84 (30% de 277) |

### Matriz de Confusão

A matriz detalha os acertos e erros do modelo na classificação.

| | **Previsto: Não Recorrência** | **Previsto: Recorrência** |
| :--- | :--- | :--- |
| **Real: Não Recorrência** (\`no-recurrence-events\`) | **50** (Verdadeiros Negativos) | **6** (Falsos Positivos) |
| **Real: Recorrência** (\`recurrence-events\`) | **15** (Falsos Negativos) | **13** (Verdadeiros Positivos) |

*Matriz no formato de array:* \`[[50 6], [15 13]]\`

**Análise:** O modelo classificou incorretamente 15 casos como "Não Recorrência" quando, na verdade, houve recorrência (Falsos Negativos). Em um contexto de saúde, a minimização de Falsos Negativos seria uma prioridade para futuras otimizações.

---

##  Como Executar

1.  Certifique-se de que o arquivo \`breast-cancer.csv\` está na mesma pasta do seu script Python.
2.  Instale as bibliotecas necessárias:
    \`\`\`bash
    pip install pandas scikit-learn matplotlib
    \`\`\`
3.  Execute o script:
    \`\`\`bash
    python [NOME_DO_SEU_ARQUIVO].py
    **Ex: python clusterizacao.py**
    \`\`\`
