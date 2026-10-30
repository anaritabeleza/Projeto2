import numpy as np  
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
import umap 

class PCA:
    def __init__(self, n_componentes):
        self.n_componentes = n_componentes
        self.componentes = None 
        self.media = None
        self.valores_proprios = None
        
    def fit(self, x):
        self.media = np.mean(x, axis=0)
        x_centrado = x - self.media
        
        covariancia = np.cov(x_centrado, rowvar=False)
        valores_proprios, vetores_proprios = np.linalg.eigh(covariancia)
        
        indices_ordenados = np.argsort(valores_proprios)[::-1]
        valores_proprios = valores_proprios[indices_ordenados]
        vetores_proprios = vetores_proprios[:, indices_ordenados]
        
        self.valores_proprios = valores_proprios
        self.componentes = vetores_proprios[:, :self.n_componentes]
    
    def variancia_explicada(self):
        if self.valores_proprios is None:
            raise ValueError("O PCA ainda não foi treinado.")

        return (
            np.sum(self.valores_proprios[:self.n_componentes])
            / np.sum(self.valores_proprios)
        )
    
    def erro_reconstrucao(self, x):
        x_comprimido = self.transform(x)
        x_reconstruido = self.inverse_transform(x_comprimido)

        return np.mean((x - x_reconstruido) ** 2)
    
    def transform(self, x):
        x_centrado = x - self.media
        return np.dot(x_centrado, self.componentes)
    
    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
    
    def inverse_transform(self, x_comprimido):
        return np.dot(x_comprimido, self.componentes.T) + self.media


class Autoencoder:
    def __init__(self, n_componentes, lr=0.05, random_state=42):
        self.n_componentes = n_componentes
        self.lr = lr
        self.n_entradas = 784
       
        rng = np.random.default_rng(random_state)
        escala_w1 = np.sqrt(2.0 / self.n_entradas)
        self.W1 = rng.standard_normal((self.n_entradas, self.n_componentes)) * escala_w1
        self.b1 = np.zeros((1, self.n_componentes))

        escala_w2 = np.sqrt(2.0 / self.n_componentes)
        self.W2 = rng.standard_normal((self.n_componentes, self.n_entradas)) * escala_w2
        self.b2 = np.zeros((1, self.n_entradas))

    def sigmoide(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    
    def derivada_sigmoide(self, a):
        return a * (1.0 - a)
    
    def treinar(self, x, epocas=50, batch_size=256):
        m = x.shape[0]
        
        for epoca in range(epocas):
            # Baralha os dados em cada época
            indices = np.random.permutation(m)
            x_shuffled = x[indices]
            
            # Treino por mini-batches
            for i in range(0, m, batch_size):
                batch_x = x_shuffled[i:i + batch_size]
                b_size = batch_x.shape[0]
                
                # Forward Pass
                Z1 = np.dot(batch_x, self.W1) + self.b1
                A1 = self.sigmoide(Z1)
                
                Z2 = np.dot(A1, self.W2) + self.b2
                A2 = self.sigmoide(Z2)
                
                # Backward Pass (Gradientes)
                dZ2 = (A2 - batch_x) * self.derivada_sigmoide(A2)
                dW2 = np.dot(A1.T, dZ2) / b_size
                db2 = np.sum(dZ2, axis=0, keepdims=True) / b_size
                
                dZ1 = np.dot(dZ2, self.W2.T) * self.derivada_sigmoide(A1)
                dW1 = np.dot(batch_x.T, dZ1) / b_size
                db1 = np.sum(dZ1, axis=0, keepdims=True) / b_size
                
                # Atualização de Pesos
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2
            
            if (epoca + 1) % 10 == 0:
                Z1_full = np.dot(x, self.W1) + self.b1
                A1_full = self.sigmoide(Z1_full)
                Z2_full = np.dot(A1_full, self.W2) + self.b2
                A2_full = self.sigmoide(Z2_full)
                erro = np.mean((A2_full - x) ** 2)
                print(f"Época {epoca+1}/{epocas} - Erro de Reconstrução: {erro:.4f}")
                
    def comprimir_e_reconstruir(self, x):
        Z1 = np.dot(x, self.W1) + self.b1
        codigos = self.sigmoide(Z1)
        
        Z2 = np.dot(codigos, self.W2) + self.b2
        reconstrucoes = self.sigmoide(Z2)
        
        return codigos, reconstrucoes 


def desenhar_tsne(X, y, amostra=1000):
    tsne = TSNE(n_components=2, random_state=42)
    X_2d = tsne.fit_transform(X[:amostra])
    
    plt.figure(figsize=(10, 8))
    grafico = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y[:amostra], cmap='tab10', alpha=0.7)
    plt.colorbar(grafico, ticks=range(10), label="Categorias")
    plt.title("Visualização t-SNE")
    plt.show()


def extrair_textura_arestas(X):
    if X.shape[1] == 784:
        X_img = X.reshape(-1, 28, 28)
    else:
        X_img = X
        
    caracteristicas_arestas = []
    for img in X_img:
        gy, gx = np.gradient(img)
        magnitude = np.sqrt(gx**2 + gy**2)
        caracteristicas_arestas.append(magnitude.flatten())
        
    return np.array(caracteristicas_arestas)


def desenhar_umap(X, y, amostra=1000):
    redutor_umap = umap.UMAP(n_components=2, random_state=42)
    X_2d = redutor_umap.fit_transform(X[:amostra])
    
    plt.figure(figsize=(10, 8))
    grafico = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y[:amostra], cmap='tab10', alpha=0.7)
    plt.colorbar(grafico, ticks=range(10), label="Categorias (0 a 9)")
    plt.title("Visualização UMAP - Agrupamento de Roupas")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.show()
