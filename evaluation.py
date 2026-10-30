import numpy as np 
from sklearn.metrics import classification_report, confusion_matrix

class RegressaoLogistica:
    def __init__(self,lr = 0.1, epocas = 1000):
        self.lr = lr
        self.epocas = epocas
        self.W = None
        self.b = None
        self.classes_ = None
    
    def codificar_one_hot(self,y,num_classes):
        m = len(y)
        matriz_one_hot = np.zeros((m, num_classes))
        matriz_one_hot[np.arange(m),y] = 1
        return matriz_one_hot
    
    def softmax(self,z):
        z_estavel = z - np.max(z ,axis = 1, keepdims = True)
        exp_z = np.exp(z_estavel)
        return exp_z / np.sum(exp_z, axis = 1, keepdims = True)
    
    def fit(self,x,y):
        if len(x) != len(y):
            raise ValueError("X e y têm de ter o mesmo número de amostras.")
        m,n_caracteristicas = x.shape
        self.classes_ = np.unique(y)
        y_indices = np.searchsorted(self.classes_, y)
        num_classes = len(self.classes_)
        
        self.W = np.zeros((n_caracteristicas, num_classes))
        self.b = np.zeros((1, num_classes))
        
        y_one_hot = self.codificar_one_hot(y_indices, num_classes)
        
        for epoca in range(self.epocas):
            Z = x @ self.W + self.b
            A = self.softmax(Z)
            
            dZ = A - y_one_hot
            dW = x.T @ dZ / m
            db = np.sum(dZ, axis = 0, keepdims = True) / m 
            
            self.W -= self.lr * dW
            self.b -= self.lr * db 
        return self 
            
    def predict(self,x):
        if self.W is None or self.classes_ is None:
            raise ValueError("O classificador tem de ser treinado primeiro.")

        Z = x @ self.W + self.b
        A = self.softmax(Z)
        indices_previstos = np.argmax(A, axis= 1)
        return self.classes_[indices_previstos]
    
def treinar_e_avaliar(x_treino,y_treino,x_val,y_val,nome_do_espaco,mostrar_relatorio=False,silencioso = False):
    classificador = RegressaoLogistica(lr = 0.1 , epocas = 500)
    classificador.fit(x_treino,y_treino)
    
    previsoes = classificador.predict(x_val)
    precisao = np.mean(previsoes == y_val)
    
    if not silencioso:
        print(
            f"{nome_do_espaco}: "
            f"accuracy = {precisao:.4f} "
            f"({precisao * 100:.2f}%)"
        )
        
    if mostrar_relatorio:
        print("\nClassification report:")
        print(classification_report(
            y_val, previsoes, zero_division=0
        ))

        print("Confusion matrix:")
        print(confusion_matrix(y_val, previsoes))
    return precisao, previsoes 


def comparar_espacos(dicionario_dados_treino,dicionario_dados_val,y_treino,y_val):
    resultados = {}

    for nome, x_tr in dicionario_dados_treino.items():
        if nome not in dicionario_dados_val:
            print(f"Aviso: '{nome}' não existe nos dados de validação.")
            continue

        x_val_atual = dicionario_dados_val[nome]

        precisao, previsoes = treinar_e_avaliar(
            x_tr,
            y_treino,
            x_val_atual,
            y_val,
            nome
        )

        resultados[nome] = {
            "accuracy": precisao,
            "previsoes": previsoes
        }

    return resultados
