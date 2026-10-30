
import numpy as np
import matplotlib.pyplot as plt


def encontrar_roupas_parecidas(imagem_consulta, base_dados, k=5):
    imagem_consulta = np.asarray(imagem_consulta)
    base_dados = np.asarray(base_dados)

    if imagem_consulta.ndim != 1 or base_dados.ndim != 2:
        raise ValueError(
            "A consulta deve ser um vetor e a base de dados deve ser uma matriz."
        )

    if len(base_dados) == 0:
        raise ValueError("A base de dados está vazia.")

    if imagem_consulta.shape[0] != base_dados.shape[1]:
        raise ValueError(
            "A consulta e a base de dados têm dimensões incompatíveis."
        )

    if not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)) or k < 1:
        raise ValueError("k deve ser um número inteiro positivo.")

    if not np.all(np.isfinite(imagem_consulta)) or not np.all(np.isfinite(base_dados)):
        raise ValueError("A consulta e a base de dados não podem conter NaN ou infinito.")

    k = min(k, len(base_dados))

    # Distância Euclidiana entre a consulta e cada imagem
    distancias = np.linalg.norm(
        base_dados - imagem_consulta,
        axis=1
    )

    indices_vizinhos = np.argsort(distancias)[:k]

    return indices_vizinhos


def pesquisar_multiplos_espacos(imagem_idx, dicionario_bases, k=5):
    resultados_espacos = {}

    if not isinstance(imagem_idx, (int, np.integer)) or isinstance(imagem_idx, (bool, np.bool_)):
        raise ValueError("O índice da imagem deve ser um número inteiro.")

    for nome_espaco, base_dados in dicionario_bases.items():
        base_dados = np.asarray(base_dados)

        if base_dados.ndim != 2 or len(base_dados) == 0:
            raise ValueError(
                f"A base do espaço '{nome_espaco}' deve ser uma matriz não vazia."
            )

        if imagem_idx < 0 or imagem_idx >= len(base_dados):
            raise ValueError(
                f"O índice da consulta é inválido no espaço '{nome_espaco}'."
            )

        # A mesma imagem de consulta é retirada da base de pesquisa
        imagem_consulta = base_dados[imagem_idx]
        base_pesquisa = np.delete(base_dados, imagem_idx, axis=0)

        if len(base_pesquisa) == 0:
            resultados_espacos[nome_espaco] = np.array([], dtype=int)
            continue

        indices = encontrar_roupas_parecidas(
            imagem_consulta,
            base_pesquisa,
            k=k
        )

        # Converter os índices para as posições da base original
        indices_ajustados = np.where(
            indices < imagem_idx,
            indices,
            indices + 1
        )

        resultados_espacos[nome_espaco] = indices_ajustados

    return resultados_espacos


def desenhar_resultados_pesquisas(
    imagem_consulta_pixeis,
    base_dados_pixeis,
    indices_vizinhos
):
    imagem_consulta_pixeis = np.asarray(imagem_consulta_pixeis)
    base_dados_pixeis = np.asarray(base_dados_pixeis)
    indices_vizinhos = np.asarray(indices_vizinhos, dtype=int)

    if imagem_consulta_pixeis.size != 784:
        raise ValueError("A imagem de consulta deve conter 784 píxeis.")

    if base_dados_pixeis.ndim != 2 or base_dados_pixeis.shape[1] != 784:
        raise ValueError(
            "A base de visualização deve conter imagens com 784 píxeis."
        )

    if indices_vizinhos.ndim != 1 or len(indices_vizinhos) == 0:
        raise ValueError("É necessário fornecer pelo menos um vizinho.")

    if np.any(indices_vizinhos < 0) or np.any(
        indices_vizinhos >= len(base_dados_pixeis)
    ):
        raise ValueError("Existem índices de vizinhos inválidos.")

    fig, eixos = plt.subplots(
        1,
        len(indices_vizinhos) + 1,
        figsize=(3 * (len(indices_vizinhos) + 1), 4)
    )

    eixos[0].imshow(
        imagem_consulta_pixeis.reshape(28, 28),
        cmap="gray",
        vmin=0,
        vmax=1
    )
    eixos[0].set_title("Imagem pesquisada")
    eixos[0].axis("off")

    for posicao, indice in enumerate(indices_vizinhos, start=1):
        eixos[posicao].imshow(
            base_dados_pixeis[indice].reshape(28, 28),
            cmap="gray",
            vmin=0,
            vmax=1
        )
        eixos[posicao].set_title(f"Vizinho {posicao}")
        eixos[posicao].axis("off")

    plt.tight_layout()
    plt.show()


def precisao_em_k(indices_vizinhos, etiqueta_consulta, etiquetas_base):
    indices_vizinhos = np.asarray(indices_vizinhos)
    etiquetas_base = np.asarray(etiquetas_base)

    if indices_vizinhos.ndim != 1 or len(indices_vizinhos) == 0:
        raise ValueError("A lista de vizinhos está vazia ou tem formato inválido.")

    if not np.issubdtype(indices_vizinhos.dtype, np.integer):
        raise ValueError("Os índices dos vizinhos devem ser números inteiros.")

    if etiquetas_base.ndim != 1:
        raise ValueError("As etiquetas da base devem formar um vetor.")

    if np.any(indices_vizinhos < 0) or np.any(
        indices_vizinhos >= len(etiquetas_base)
    ):
        raise ValueError("Existem índices inválidos na lista de vizinhos.")

    # As etiquetas só são consultadas depois de definidos os vizinhos
    etiquetas_vizinhos = etiquetas_base[indices_vizinhos]

    correspondencias = etiquetas_vizinhos == etiqueta_consulta

    return np.mean(correspondencias)


def avaliar_multiplos_espacos(
    dicionario_consultas,
    dicionario_bases_treino,
    y_consultas,
    y_treino,
    k=5,
    n_consultas=500,
    random_state=42
):
    y_consultas = np.asarray(y_consultas)
    y_treino = np.asarray(y_treino)

    if not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)) or k < 1:
        raise ValueError("k deve ser um número inteiro positivo.")

    if not isinstance(n_consultas, (int, np.integer)) or isinstance(n_consultas, (bool, np.bool_)) or n_consultas < 1:
        raise ValueError("n_consultas deve ser um número inteiro positivo.")

    if len(y_consultas) == 0 or len(y_treino) == 0:
        raise ValueError("Os conjuntos de consulta e treino não podem estar vazios.")

    if set(dicionario_consultas) != set(dicionario_bases_treino):
        raise ValueError(
            "As consultas e as bases de treino devem ter os mesmos espaços."
        )

    # Usar as mesmas imagens de teste em todos os espaços
    n = min(n_consultas, len(y_consultas))
    rng = np.random.default_rng(random_state)
    indices_consultas = rng.choice(
        len(y_consultas), size=n, replace=False
    )

    resultados = {}

    for nome, consultas in dicionario_consultas.items():
        consultas = np.asarray(consultas)
        base_treino = np.asarray(dicionario_bases_treino[nome])

        if consultas.ndim != 2 or base_treino.ndim != 2:
            raise ValueError(
                f"Consultas e base de treino de '{nome}' devem ser matrizes."
            )

        if len(consultas) != len(y_consultas):
            raise ValueError(
                f"O número de consultas em '{nome}' não corresponde às etiquetas."
            )

        if len(base_treino) != len(y_treino):
            raise ValueError(
                f"O número de imagens de treino em '{nome}' não corresponde às etiquetas."
            )

        if consultas.shape[1] != base_treino.shape[1]:
            raise ValueError(
                f"As dimensões das consultas e da base não coincidem em '{nome}'."
            )

        precisões = []

        for indice in indices_consultas:
            # A pesquisa usa apenas as características, não as etiquetas
            vizinhos = encontrar_roupas_parecidas(
                consultas[indice],
                base_treino,
                k=k
            )

            # As etiquetas são usadas apenas depois da pesquisa
            precisao = precisao_em_k(
                vizinhos,
                y_consultas[indice],
                y_treino
            )

            precisões.append(precisao)

        resultados[nome] = float(np.mean(precisões))

        print(
            f"{nome} — Precision@{k} média "
            f"({n} consultas): {resultados[nome]:.2%}"
        )

    return resultados
