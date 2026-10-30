import numpy as np
import matplotlib.pyplot as plt
from retrieval import encontrar_roupas_parecidas

def encontrar_e_mostrar_falhas(base_consultas, y_consultas, base_treino, y_treino, k=5, max_falhas=5):
    base_consultas = np.asarray(base_consultas)
    y_consultas = np.asarray(y_consultas)
    base_treino = np.asarray(base_treino)
    y_treino = np.asarray(y_treino)
    
    falhas_encontradas = 0
    fig, eixos = plt.subplots(max_falhas, k + 1, figsize=(3 * (k + 1), 3 * max_falhas))
    if max_falhas == 1:
        eixos = np.array([eixos])

    for i in range(len(base_consultas)):
        if falhas_encontradas >= max_falhas:
            break
            
        consulta = base_consultas[i]
        etiqueta_consulta = y_consultas[i]
        
        # Encontrar vizinhos
        vizinhos = encontrar_roupas_parecidas(consulta, base_treino, k=k)
        etiquetas_vizinhos = y_treino[vizinhos]
        
        # Consideramos caso de falha/interesse se o vizinho mais próximo tiver etiqueta diferente
        if etiquetas_vizinhos[0] != etiqueta_consulta:
            # Desenhar imagem de consulta
            eixos[falhas_encontradas, 0].imshow(consulta.reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            eixos[falhas_encontradas, 0].set_title(f"Consulta (Classe {etiqueta_consulta})")
            eixos[falhas_encontradas, 0].axis("off")
            
            # Desenhar vizinhos retribuídos
            for j, idx_v in enumerate(vizinhos):
                eixos[falhas_encontradas, j + 1].imshow(base_treino[idx_v].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
                eixos[falhas_encontradas, j + 1].set_title(f"Vizinho {j+1}\n(Classe {y_treino[idx_v]})")
                eixos[falhas_encontradas, j + 1].axis("off")
                
            falhas_encontradas += 1

    plt.tight_layout()
    plt.show()
    print(f"Foram mostrados {falhas_encontradas} casos de falha visuais.")