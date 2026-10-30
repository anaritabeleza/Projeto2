import numpy as np  
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
import umap 

class PCA:
    def __init__(self,n_componentes):
        self.n_componentes = n_componentes
        self.componentes =  None 
        self.media = None
        self.valores_proprios = None
        
    def fit(self,x):
        #calculei o pixel médio de cada coluna e centrei xi subtraindo a media
        self.media = np.mean(x,axis = 0)
        x_centrado = x - self.media
        
        #calcular a matriz de covariancia
        #rowvar garante que calculamos a relação entre os 784 pixeis e não entre imagens
        covariancia = np.cov(x_centrado, rowvar = False)
        
        #calcular vetores e valores próprios
        #eigh é a função ideal para matrizes simétrias como a da covariância e devolve do menos importante para o mais importante
        valores_proprios, vetores_proprios = np.linalg.eigh(covariancia)
        
        #ordenar por importância
        #uso o [::-1]para ficar do maior para o menor 
        indices_ordenados = np.argsort(valores_proprios) [::-1]
        valores_proprios = valores_proprios[indices_ordenados]
        vetores_proprios = vetores_proprios[:, indices_ordenados]
        
        self.valores_proprios = valores_proprios
        #guardar apenas os k componentes principais 
        self.componentes = vetores_proprios[:, :self.n_componentes]
    
    def variancia_explicada(self):
        if self.valores_proprios is None:
            raise ValueError("O PCA ainda não foi treinado.")

        return (
            np.sum(self.valores_proprios[:self.n_componentes])
            / np.sum(self.valores_proprios)
        )
    
    def erro_reconstrucao(self, x):
        x_comprimido= self.transform(x)
        x_reconstruido = self.inverse_transform(x_comprimido)

        return np.mean((x - x_reconstruido) ** 2)
    
    def transform(self,x):
        #comprimir novas imagens usando o que aprendeu
        x_centrado = x -self.media
        #multipliquei as imagens pelos componentes (produto interno de matrizes)
        #passei de 784 colunas para n_componentes colunas
        return np.dot(x_centrado, self.componentes)
    
    def fit_transform(self,x):
        #treina e comprime de uma só vez 
        self.fit(x)
        return self.transform(x)
    
    def inverse_transform(self,x_comprimido):
        # tenta reconstruir a imagem original a partir de versão comprimida
        #fazemos a multiplicação inversa(transposta) e voltamos a somar a média para tirar a imagem do ponto zero
        return np.dot(x_comprimido,self.componentes.T) + self.media
    
    
class Autoencoder:
    def __init__(self,n_componentes,lr = 0.8,random_state = 42):
        self.n_componentes = n_componentes
        self.lr = lr
        self.n_entradas = 784 # 28x28 pixeis de cada imagem 
       
        # Gerador aleatório com semente fixa
        rng = np.random.default_rng(random_state)
        escala_w1 = np.sqrt(2.0 / self.n_entradas)
        self.W1 = rng.standard_normal(
            (self.n_entradas, self.n_componentes)
        ) * escala_w1

        self.b1 = np.zeros((1, self.n_componentes))

        # Pesos do descodificador: n_componentes -> 784
        escala_w2 = np.sqrt(2.0 / self.n_componentes)
        self.W2 = rng.standard_normal(
            (self.n_componentes, self.n_entradas)
        ) * escala_w2

        self.b2 = np.zeros((1, self.n_entradas))

        
    def sigmoide(self,z):
        #função de ativação que esmaga os valores entre 0 e 1 
        #np.clip evita números muito grandes que podem quebrar o cálculo da expoencial
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def derivada_sigmoide(self,a):
        #a derivada da função sigmoide é necessária para a descida do gradiente 
        return a * (1-a)
    
    def treinar(self,x,epocas = 500):
        #treina a rede ajustando os pesos passoa a passo
        m = x.shape[0] # número de imagens de treino 
        
        for epoca in range(epocas):
            
            #multiplicamos as imagens pelos pesos W1
            #x tem dimensão (m,784) e W1 tem (784, n_componentes) logo o resultado Z1 fica com (m, n_componentes)
            Z1 = np.dot(x,self.W1) + self.b1 
            #passo pela função sigmoide
            #A1 é a matriz de imagens já comprimidas
            A1 = self.sigmoide(Z1)
            
            #multiplicamos as imagens comprimidas pelos pesos W2
            # A1 tem (m, n_componentes) e W2 tem (n_componentes,784) logo Z2 tem (m, 784)
            Z2 = np.dot(A1, self.W2) + self.b2
            #passo pela função sigmoide 
            #A2 é a imagem final reconstruída pela rede
            A2 = self.sigmoide(Z2)
            
            #mede a diferença entre a imagem gerada (A2) e a real (x)
            erro = np.mean((A2 - x) ** 2) # erro mse 
            
            #dZ2 é a direção do erro na ultima camada (previsão - real) * derivada de sigmoide
            dZ2 = (A2 - x) * self.derivada_sigmoide(A2)
            #gradiente para a matriz W2. Usei a transposta de A1 para que as dimensões batam certo 
            dW2 = np.dot(A1.T, dZ2) / m 
            #é o gradiente do viés (soma se o erro de todas as imagens)
            db2 = np.sum(dZ2, axis = 0, keepdims = True) / m 
            
            #dZ1 propaga o erro de Z2 para trás, multiplicando pela transposta de W2
            dZ1 = np.dot(dZ2,self.W2.T) * self.derivada_sigmoide(A1)
            #é o gradiente da matriz W1 (multipliquei a entrada original transposta pelo erro de Z1)
            dW1 = np.dot(x.T, dZ1) / m 
            
            db1 = np.sum(dZ1, axis = 0, keepdims = True) / m 
            
            #atualizar os pesos 
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            
            if (epoca + 1) % 10 == 0:
                print(f"Época {epoca+1}/{epocas} - Erro de Reconstrução: {erro:.4f}")
                
    def comprimir_e_reconstruir(self,x):
        #devolve o código latente e a imagem reconstruída para os dados de teste e validação
        Z1 = np.dot(x,self.W1) + self.b1
        codigos = self.sigmoide(Z1)
        
        Z2 = np.dot(codigos,self.W2) + self.b2
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
        #Calcular gradientes verticais e horizontais
        gy, gx = np.gradient(img)
        # Magnitude do gradiente como representação de contornos e arestas
        magnitude = np.sqrt(gx**2 + gy**2)
        caracteristicas_arestas.append(magnitude.flatten())
        
    return np.array(caracteristicas_arestas)

def desenhar_umap(X, y, amostra=1000):
    # 1. Inicializar o UMAP para 2 dimensões (eixo X e Y)
    redutor_umap = umap.UMAP(n_components=2, random_state=42)
    
    # 2. Comprimir os dados
    X_2d = redutor_umap.fit_transform(X[:amostra])
    
    # 3. Desenhar o gráfico de dispersão[cite: 2]
    plt.figure(figsize=(10, 8))
    grafico = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y[:amostra], cmap='tab10', alpha=0.7)
    plt.colorbar(grafico, ticks=range(10), label="Categorias (0 a 9)")
    plt.title("Visualização UMAP - Agrupamento de Roupas")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.show()
