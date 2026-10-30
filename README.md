# Projeto 2 - Machine Learning
**Redução de Dimensionalidade, Reconstrução e Recuperação de Imagens**

Este projeto foca-se na aprendizagem de representações compactas para imagens de vestuário (utilizando o dataset **Fashion-MNIST**), explorando duas tarefas principais: **classificação** e **recuperação de imagens semelhantes**.

---

## 📂 Estrutura do Repositório

O projeto está modularizado da seguinte forma:
* `main.py`: Script principal que executa o fluxo de trabalho (carregamento, divisão dos dados, testes de PCA, Autoencoder, t-SNE/UMAP, classificação e recuperação).
* `representation.py`: Contém a implementação de raiz dos algoritmos de redução de dimensionalidade (**PCA** e **Autoencoder**), bem como as funções de visualização (t-SNE e UMAP).
* `evaluation.py`: Contém a implementação do classificador de **Regressão Logística (com Softmax)** e a função de avaliação.
* `retrieval.py`: Módulo responsável pela busca de vizinhos mais próximos ($k$-NN), visualização dos resultados e cálculo da métrica `Precision@5`.
* `failure_cases.py`: Módulo de suporte para identificação visual de casos de falha no modelo de recuperação.

## Dataset e Extensão de Cor/Textura
- **Dataset Principal:** O projeto utiliza o **Fashion-MNIST**, composto por 60.000 imagens de treino e 10.000 de teste em escala de cinzentos ($28 \times 28$ píxeis) divididas por 10 classes de vestuário.
- **Perspetiva de Textura e Arestas:** Como o Fashion-MNIST é estritamente em escala de cinzentos, a perspetiva de cor e variações cromáticas não se aplica diretamente aos dados brutos. Para colmatar a perspetiva de texturas e contornos exigida pelo projeto, implementou-se um extrator baseado no gradiente de imagem (`extrair_textura_arestas` em `representation.py`), realçando arestas e padrões estruturais das peças.
---

## 🚀 Como Executar o Projeto

Certifica-te de que tens as dependências necessárias instaladas (como `pandas`, `numpy`, `scikit-learn`, `matplotlib` e `umap-learn`). 

Para correr o projeto completo, executa o script principal a partir da raiz do repositório:
```bash
python main.py