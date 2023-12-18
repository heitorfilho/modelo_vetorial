# Modelo Vetorial de Recuperação de Informação

O Modelo Vetorial de Recuperação de Informação é um sistema que realiza a leitura de um arquivo contendo uma base de documentos. Em seguida, extrai o conteúdo de cada documento e executa o pré-processamento textual, que envolve etapas como tokenização, remoção de stopwords e extração de radicais.

## Ponderação TF-IDF

Após o pré-processamento, o sistema constrói um índice invertido, representando a relação entre os termos e os documentos da base. Cada entrada no índice invertido é no formato `termo:(documento, peso)`, onde o peso é calculado usando a ponderação TF-IDF (Term Frequency-Inverse Document Frequency).


A fórmula da ponderação TF-IDF é dada por:

TF-IDF = TF * IDF

Onde:

- TF (Term Frequency) é calculado como:
  - 1 + log(freq), se freq > 0
  - 0, caso contrário

- IDF (Inverse Document Frequency) é calculado como:
  - log(N/n)

Onde \(N\) é o número total de documentos e \(n\) é o número de documentos nos quais o termo aparece.


## Execução do Sistema

Ao receber uma consulta, o sistema realiza o mesmo pré-processamento na consulta, incluindo o operador lógico &(AND).

Em seguida, é calculada a similaridade entre a consulta e cada documento da base usando o Modelo Vetorial e a ponderação TF-IDF. A similaridade entre o documento dj e a consulta q é obtida através do cosseno do ângulo entre seus vetores. Onde é feito pelo produto interno dos dois vetores dividido pela multiplicação da raiz quadrada da soma dos termos ao quadrado de um vetor pelo outro.

O sistema gera um arquivo de pesos contendo os documentos e o peso de cada termo no documento no seguinte padrão:

doc1.txt: termo1, 0.1845 termo2, 0.3010

doc2.txt: termo1, 0.1625 termo2, 0.6021

Também é gerado um arquivo de respostas que contém a quantidade de documentos com similaridade maior que 0.001. O arquivo exibe a quantidade de documentos retornados e o nome dos documentos e suas similaridades no seguinte padrão:

3

doc2.txt 0.9983

doc3.txt 0.2031

doc1.txt 0.1061

## Como Executar

Para executar o sistema, digite no terminal:

```bash
python modelo_vetorial.py base.txt consulta.txt
