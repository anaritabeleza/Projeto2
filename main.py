import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from representation import PCA, Autoencoder
from representation import desenhar_tsne, desenhar_umap, extrair_textura_arestas
from evaluation import treinar_e_avaliar
from retrieval import encontrar_roupas_parecidas, desenhar_resultados_pesquisas, precisao_em_k,avaliar_multiplos_espacos
from failure_cases import encontrar_e_mostrar_falhas

df_treino = pd.read_csv('data/fashion_mnist_train.csv')
df_teste = pd.read_csv('data/fashion_mnist_test.csv')

y_treino_total = df_treino['label'].values
x_treino_total = df_treino.drop('label', axis = 1).values

y_teste = df_teste['label'].values
x_teste = df_teste.drop('label', axis = 1).values

x_treino_total = x_treino_total/ 255.0
x_teste = x_teste / 255.0
#normalizei o x treino e teste, para que todos os valores fiquem numa escala de 0 a 1 

x_treino,x_val,y_treino,y_val = train_test_split(
    x_treino_total,
    y_treino_total,
    test_size = 0.2, #corte a 20%
    random_state = 42,#baralha os dados antes de os cortar e o número 42 garante que a forma como ela baralha é sempre igual
    stratify=y_treino_total
)
#faz o corte e 20% dos dados de treino vão ser o conjunto de validação e 80% os dados de treino

print(f"Dimensões de Treino: X={x_treino.shape}, y={y_treino.shape}")
print(f"Dimensões de Validação: X={x_val.shape}, y={y_val.shape}")
print(f"Dimensões de Teste: X={x_teste.shape}, y={y_teste.shape}")

dimensoes = [2, 8, 16, 32, 64, 128]
resultados_dim_acc = {}

for n in dimensoes:
    pca_experiencia = PCA(n_componentes=n)
    x_treino_reduzido = pca_experiencia.fit_transform(x_treino)
    x_val_reduzido = pca_experiencia.transform(x_val)
    
    erro = pca_experiencia.erro_reconstrucao(x_val)

    variancia = pca_experiencia.variancia_explicada()
    precisao_n, _ = treinar_e_avaliar(
        x_treino_reduzido, y_treino,
        x_val_reduzido, y_val,
        f"PCA ({n} comp.)",
        mostrar_relatorio=False,
        silencioso = True
    )

    resultados_dim_acc[n] = precisao_n 
    print(
        f"PCA com {n} componentes | "
        f"Variância explicada: {variancia:.2%} | "
        f"Erro Recon.: {erro:.6f} | "
        f"Acurácia Val: {precisao_n:.4f} ({precisao_n * 100:.2f}%)"
    )

pca = PCA(n_componentes = 32) #reduzir as imagens a 32 dimensões
x_treino_pca = pca.fit_transform(x_treino) # treinar e comprimir os dados de treino
x_val_pca = pca.transform(x_val) # comprimir validação e teste
x_teste_pca = pca.transform(x_teste)

x_treino_reconstruido_pca = pca.inverse_transform(x_treino_pca)#reconstruir para depois usar em gráficos e relatório
x_val_reconstrucao_pca = pca.inverse_transform(x_val_pca)
x_teste_reconstrucao_pca = pca.inverse_transform(x_teste_pca)

ae = Autoencoder(n_componentes = 32, lr = 0.05)
ae.treinar(x_treino, epocas = 500)
x_treino_ae,x_treino_reconstruido_ae = ae.comprimir_e_reconstruir(x_treino)
x_val_ae,x_val_reconstruido_ae = ae.comprimir_e_reconstruir(x_val)
x_teste_ae,x_teste_reconstruido_ae = ae.comprimir_e_reconstruir(x_teste)

x_treino_arestas = extrair_textura_arestas(x_treino)
x_val_arestas = extrair_textura_arestas(x_val)
x_teste_arestas = extrair_textura_arestas(x_teste)

nomes_classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

indices_por_classe = []
for classe in range(10):
    idx = np.where(y_teste == classe)[0][0]
    indices_por_classe.append(idx)

fig, eixos = plt.subplots(10, 3, figsize=(9, 20))

for i, idx in enumerate(indices_por_classe):
    classe_nome = nomes_classes[i]
    
    original = x_teste[idx].reshape(28, 28)
    recon_pca = x_teste_reconstrucao_pca[idx].reshape(28, 28)
    recon_ae = x_teste_reconstruido_ae[idx].reshape(28, 28)
    
    eixos[i, 0].imshow(original, cmap="gray", vmin=0, vmax=1)
    eixos[i, 0].set_ylabel(classe_nome, fontsize=10, rotation=0, labelpad=40, ha='right', va='center')
    if i == 0:
        eixos[i, 0].set_title("Original", fontsize=12, fontweight='bold')
        
    eixos[i, 1].imshow(recon_pca, cmap="gray", vmin=0, vmax=1)
    if i == 0:
        eixos[i, 1].set_title("Reconstrução PCA (32)", fontsize=12, fontweight='bold')
        
    eixos[i, 2].imshow(recon_ae, cmap="gray", vmin=0, vmax=1)
    if i == 0:
        eixos[i, 2].set_title("Reconstrução AE (32)", fontsize=12, fontweight='bold')
        
    for col in range(3):
        eixos[i, col].set_xticks([])
        eixos[i, col].set_yticks([])

plt.tight_layout()
plt.show()

desenhar_tsne(x_treino, y_treino, amostra=1000)
desenhar_umap(x_treino, y_treino, amostra=1000)

resultados = {}

resultados["Píxeis originais (784)"] = treinar_e_avaliar(
    x_treino, y_treino,
    x_val, y_val,
    "Píxeis originais (784)"
)[0]

resultados["PCA (32 componentes)"] = treinar_e_avaliar(
    x_treino_pca, y_treino,
    x_val_pca, y_val,
    "PCA (32 componentes)"
)[0]

resultados["Autoencoder (32 componentes)"] = treinar_e_avaliar(
    x_treino_ae, y_treino,
    x_val_ae, y_val,
    "Autoencoder (32 componentes)"
)[0]

print("\nResultados de validação:")

for nome, accuracy in resultados.items():
    print(f"{nome}: {accuracy:.4f} ({accuracy * 100:.2f}%)")

#escolher a primeira imagem do conjunto de teste para ser a nossa pesquisa
indice_pesquisa = 0 

# A imagem original (784 píxeis) que usamos para ver no ecrã
pesquisa_pixeis = x_teste[indice_pesquisa] 

# Procurar os 5 vizinhos mais próximos no espaço dos pixeis originais 
vizinhos_pixeis = encontrar_roupas_parecidas(
    pesquisa_pixeis,
    x_treino,
    k = 5
)

print("Resultados por píxeis originais:")

desenhar_resultados_pesquisas(
    pesquisa_pixeis,
    x_treino,
    vizinhos_pixeis
)

pesquisa_pca = x_teste_pca[indice_pesquisa]

vizinhos_pca = encontrar_roupas_parecidas(
    pesquisa_pca,
    x_treino_pca,
    k=5
)

print("Resultados por PCA:")

desenhar_resultados_pesquisas(
    pesquisa_pixeis,
    x_treino,
    vizinhos_pca
)

pesquisa_ae = x_teste_ae[indice_pesquisa]

vizinhos_ae = encontrar_roupas_parecidas(
    pesquisa_ae,
    x_treino_ae,
    k=5
)

print("Resultados por autoencoder:")

desenhar_resultados_pesquisas(
    pesquisa_pixeis,
    x_treino,
    vizinhos_ae
)

precision_pixeis = precisao_em_k(
    vizinhos_pixeis,
    y_teste[indice_pesquisa],
    y_treino
)

precision_pca = precisao_em_k(
    vizinhos_pca,
    y_teste[indice_pesquisa],
    y_treino
)

precision_ae = precisao_em_k(
    vizinhos_ae,
    y_teste[indice_pesquisa],
    y_treino
)

print(f"Precision@5 — píxeis: {precision_pixeis:.2%}")
print(f"Precision@5 — PCA: {precision_pca:.2%}")
print(f"Precision@5 — Autoencoder: {precision_ae:.2%}")

# Comparar a recuperação em vários espaços (incluindo Arestas)
consultas = {
    "Píxeis originais": x_teste,
    "PCA": x_teste_pca,
    "Autoencoder": x_teste_ae,
    "Textura/Arestas": x_teste_arestas  
}

bases_treino = {
    "Píxeis originais": x_treino,
    "PCA": x_treino_pca,
    "Autoencoder": x_treino_ae,
    "Textura/Arestas": x_treino_arestas  
}

resultados_retrieval = avaliar_multiplos_espacos(
    consultas,
    bases_treino,
    y_teste,
    y_treino,
    k=5,
    n_consultas=500
)

encontrar_e_mostrar_falhas(x_teste_pca, y_teste, x_treino_pca, y_treino, k=5, max_falhas=5)