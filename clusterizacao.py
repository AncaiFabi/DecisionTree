import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ================================================
# 2. CARREGAR O ARQUIVO breast-cancer.csv
# ================================================
dados = pd.read_csv('breast-cancer.csv', na_values='?')

# Remove as linhas que contêm valores faltantes
dados = dados.dropna()

print("Prévia dos dados (após limpeza):")
print(dados.head())
print("Total de instâncias após limpeza:", len(dados))

# ================================================
# 3. SEPARAR ATRIBUTOS (X) E CLASSE (y) (AJUSTADO)
# ================================================
dados_classes = dados['Class'] # Target/Classe (y)

dados_atributos = pd.get_dummies(dados.drop(columns=['Class']))

print("\nPrévia dos atributos (após One-Hot Encoding):")
print(dados_atributos.head())
print("Número de colunas:", dados_atributos.shape[1])

# ================================================
# 4. DIVIDIR EM TREINO E TESTE (HOLD OUT 70/30)
# ================================================
X_train, X_test, y_train, y_test = train_test_split(
    dados_atributos, 
    dados_classes,
    test_size=0.3,
    random_state=42
)

# ================================================
# 5. CRIAR E TREINAR O MODELO DE ÁRVORE DE DECISÃO
# ================================================
tree = DecisionTreeClassifier()
modelo = tree.fit(X_train, y_train)

print("\nClasses aprendidas pelo modelo:")
print(modelo.classes_)

# ================================================
# 6. FAZER PREVISÕES
# ================================================
y_pred = modelo.predict(X_test)

print("\nComparação entre classe real e prevista (amostra):")
for i in range(10):
    print(f"Real: {y_test.iloc[i]} - Previsto: {y_pred[i]}")

# ================================================
# 7. AVALIAR ACURÁCIA
# ================================================
acuracia = metrics.accuracy_score(y_test, y_pred)
print("\nAcurácia do modelo:", acuracia)

# ================================================
# 8. MATRIZ DE CONFUSÃO
# ================================================
cm = confusion_matrix(y_test, y_pred, labels=modelo.classes_)

print("\nMatriz de Confusão:")
print(cm)

# Exibir graficamente
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=modelo.classes_)
disp.plot(cmap="Blues")
plt.title("Matriz de Confusão - Breast Cancer")
plt.show()

# ================================================
# 9. CLASSIFICAR UMA NOVA INSTÂNCIA (EXEMPLO)
# ================================================
nova_instancia_exemplo = X_test.iloc[[0]] 

classe_prevista = modelo.predict(nova_instancia_exemplo)
probabilidades = modelo.predict_proba(nova_instancia_exemplo)

print("\nClasse prevista para a nova instância:", classe_prevista)
print("Distribuição de probabilidades:", probabilidades)