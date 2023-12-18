'''
Primeiro passo é a entrada de documentos

Segundo passo é a filtragem dos documentos: tokenização, remoção de stopwords e extração de radical

Terceiro passo é gerar o indice invertido: relação entre documentos e termos: termo : (Documento, qtd)

Quarto passo é fazer a ponderação de termos usando TF-IDF -> saber o peso do termo em cada documento

Quinto passo é calcular a similaridade entre o documento e a consulta (modelo vetorial)

Sexto passo é fazer o ranqueamento de documentos

Consultas apenas com AND

Apresentar documentos na ordem correta de ranqueamento

pesos.txt : arquivo que contém ponderação TF-IDF de cada documento
resposta.txt : arquivo com os nomes dos documentos que atendem a consulta do usuário e a similaridade

Apenas o pesos diferentes de 0 devem ser apresentados
doc1.txt: W, 0.1845 X, 0.3010
doc2.txt: W, 0.1625 Y, 0.6021

Considere que apenas os documentos com similaridade maior ou igual a 0.001 podem atender à consulta
'''



import nltk
import string
import sys
import math

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as nltk_stopwords

nltk.download("stopwords") # lista de palavras irrelevantes
nltk.download('punkt') # tokenizador
nltk.download("rslp") # extrator de radical



# função que le a base de documentos e retorna uma lista contendo o conteudo de cada documento
def ler_base(base_filename):
    
    documentos = []

    with open(base_filename, 'r', encoding='utf-8') as base_file:
        for line in base_file:
            # Suponha que cada linha do arquivo "base.txt" contenha o nome de um documento
            documento_filename = line.strip()
            with open(documento_filename, 'r', encoding='utf-8') as doc_file:
                documento = doc_file.read()
                documentos.append(documento)

    return documentos


# armazena em um dicionario contendo o [nome_documento : conteudo]
def ler_base_dict(base_filename):

    documentos = {}

    with open(base_filename, 'r', encoding='utf-8') as base_file:
        for line in base_file:
            documento_filename = line.strip()
            with open(documento_filename, 'r', encoding='utf-8') as doc_file:
                documento = doc_file.read()
                documentos[documento_filename] = documento

    return documentos


# função que vai ler as consultas e retornar o conteudo
def ler_query(consulta_filename):
    with open(consulta_filename, 'r', encoding='utf-8') as consulta_file:
        consulta = consulta_file.read()
    return consulta


# função que faz o pre processamento do texto (tokenização, remoção de stopwords e pontuação, extração de radical) e retorna uma lista de tokens
def preprocessar_texto(texto):
    # Tokenização
    tokens = word_tokenize(texto.lower())

    # Remoção de stopwords e pontuação
    stop_words = set(nltk_stopwords.words('portuguese') + list(string.punctuation) + ['..', '...'])
    tokens_sem_stopwords = [token for token in tokens if token not in stop_words]

    # Extração de radical
    stemmer = nltk.stem.RSLPStemmer()
    tokens_stemizados = [stemmer.stem(token) for token in tokens_sem_stopwords]

    return tokens_stemizados


# função que gera o indice invertido de cada documento
def construir_indice_invertido(documentos):
    indice_invertido = {}
    
    for doc_id, documento in enumerate(documentos, start=1):
        tokens_stemizados = preprocessar_texto(documento)
        
        for token in tokens_stemizados:
            if token not in indice_invertido:
                indice_invertido[token] = [(doc_id, 1)]
            else:
                # Verificar se o documento já está na lista associada à palavra
                doc_na_lista = False
                for i, (doc, freq) in enumerate(indice_invertido[token]):
                    if doc == doc_id:
                        indice_invertido[token][i] = (doc, freq + 1)
                        doc_na_lista = True
                        break
                if not doc_na_lista:
                    indice_invertido[token].append((doc_id, 1))
    
    indice_invertido_ordenado = dict(sorted(indice_invertido.items()))

    return indice_invertido_ordenado


# função que gera a ocorrencia total de cada termo na base de documentos -> {termo1: qtdTotal, termo2: qtdTotal, ...}
def gerar_ocorrencia_termos(indice_invertido):

    ocorrencia_termos = {}

    # gera a ocorrencia de cada termo na base de documentos
    for termo, lista_documentos in indice_invertido.items():
        freq_total = 0
        for documento in lista_documentos:
            freq_total += documento[1]
        ocorrencia_termos[termo] = freq_total
            

    return ocorrencia_termos
    # retorna {'am': 1, 'cas': 8, 'comig': 2, 'engrac': 1, 'fac': 1, 'favor': 1, 'mor': 2, 'nad': 1, 'nao': 3, 'qu': 4, 'tamb': 1, 'tet': 1}


# função que gera a ocorrencia de cada termo em cada documento -> {doc1: [('termo1', qtd), ('termo2', qtd)], doc2: [('termo1', qtd), ('termo2', qtd)]...}
def gerar_ocorrencia_documento(indice_invertido):

    documentos_termos_frequencias = {}
    # Preencha o dicionário com documentos, termos e frequências
    for termo, lista_documentos in indice_invertido.items():
        for doc_id, freq in lista_documentos:
            if doc_id not in documentos_termos_frequencias:
                documentos_termos_frequencias[doc_id] = []
            documentos_termos_frequencias[doc_id].append((termo, freq))

    # Ordenar o dicionário com base nas chaves (números dos documentos)
    documentos_termos_frequencias_ordenado = dict(sorted(documentos_termos_frequencias.items()))

    return documentos_termos_frequencias_ordenado
    # retorna {1: [('cas', 1), ('engrac', 1), ('nad', 1), ('nao', 2), ('tet', 1)], 2: [('cas', 4), ('mor', 1), ('nao', 1), ('qu', 2), ('tamb', 1)], 3: [('am', 1), ('cas', 3), ('comig', 2), ('fac', 1), ('favor', 1), ('mor', 1), ('qu', 2)]}


# função que calcular o peso baseado no TF-IDF
def ponderar_termos(indice_invertido):

    # TF-IDF = TF * IDF
    # TF = 1 + log(freq) se freq > 0, 0 caso contrário
    # IDF = log(N/n) onde N é o número total de documentos e n é o número de documentos que o termo aparece

    # em quantos documentos o termo aparece -> calcular o IDF
    # possivel solução é usar o indice invertido com a relação de doc e termo , ex: {'doc1' : [(am, 0), (cas, 1)]} = doc1 contem 'am' 0 vezes para o calculo do TF
    # da forma que está o IDF é calculado mais fácil e o TF mais dificl, se fizer a alteração acima a dificuldade inverte

    # vai armazenar o peso de cada termo em cada documento -> {doc1: [('termo1', peso), ('termo2', peso)], doc2: [('termo1', peso), ('termo2', peso)]...}
    peso_termos = {}

    # Número total de documentos
    N = len(set(doc_id for termo, lista_documentos in indice_invertido.items() for doc_id, _ in lista_documentos))

    ocorrencia_termos = gerar_ocorrencia_termos(indice_invertido)
    ocorrencia_documento = gerar_ocorrencia_documento(indice_invertido)

    # armazena o idf de cada termo
    idf = {}
    
    # calcula o IDF
    for termo, lista_documentos in indice_invertido.items():
        # armazena a qtd de documentos que o termo aparece
        qtd_documentos = len(lista_documentos)
        # calcula o idf
        idf_atual = math.log(N/qtd_documentos, 10)
        '''
        Calculando log na base 10, se o termo aparecer em todos os documentos então o idf será 0
        Para contornar isso podemos adicionar 1 ao idf, assim o peso do TF será priorizado
        '''
        # se o idf for 0 ele será atribuido como 1
        #if idf_atual == 0:
        #    idf_atual = 1
        idf[termo] = idf_atual

    # calcula o TF-IDF
    for doc_id, lista_termos in ocorrencia_documento.items():
        for termo, freq in lista_termos:
            # calcula o tf
            tf = 1 + math.log(freq, 10) if freq > 0 else 0
            # calcula o tf-idf
            tf_idf = tf * idf[termo]
            # armazena o peso do termo no documento
            if doc_id not in peso_termos:
                peso_termos[doc_id] = []
            peso_termos[doc_id].append((termo, tf_idf))

    return peso_termos


# função que calcula a similaridade entre a consulta e cada documento
def modelo_vetorial(indice_invertido, consulta, base_filename):

    # Calcular os pesos dos termos em cada documento
    pesos_documentos = ponderar_termos(indice_invertido)

    # Tokenizar a consulta
    tokens_consulta = [token for token in preprocessar_texto(consulta) if token != '&']

    # Calcular os pesos dos termos na consulta
    pesos_consulta = {}
    for token in tokens_consulta:
        if token in indice_invertido:
            # Calcula o IDF
            idf = math.log(len(pesos_documentos) / len(indice_invertido[token]), 10)
            # Calcula o TF
            tf = 1 + math.log(tokens_consulta.count(token), 10) if tokens_consulta.count(token) > 0 else 0
            # Calcula o TF-IDF
            pesos_consulta[token] = tf * idf

    # Calcular a similaridade entre a consulta e cada documento
    similaridades = {}
    for doc_id, pesos_doc in pesos_documentos.items():
        produto_interno = sum(pesos_consulta.get(termo, 0) * peso for termo, peso in pesos_doc)
        raiz_quadrada_consulta = math.sqrt(sum(peso**2 for peso in pesos_consulta.values()))
        raiz_quadrada_doc = math.sqrt(sum(peso**2 for _, peso in pesos_doc))
        similaridades[doc_id] = produto_interno / (raiz_quadrada_consulta * raiz_quadrada_doc)

    # Ordenar os documentos por similaridade
    ranking = sorted(similaridades.items(), key=lambda x: x[1], reverse=True)

    # Gravar o arquivo de pesos
    gravar_pesos(pesos_documentos, base_filename)

    return ranking


# função que vai gerar o arquivo de resposta e gravar os nomes dos documentos que atendem a consulta junto com a similaridade
def gravar_resposta(ranking, base_filename):
    # Ler os nomes dos documentos do arquivo base.txt
    with open(base_filename, "r", encoding="utf-8") as base_file:
        nomes_documentos = [linha.strip() for linha in base_file]

    with open("resposta.txt", "w", encoding="utf-8") as resposta_file:
        # Filtrar documentos com similaridade maior ou igual a 0.001
        ranking_filtrado = [(doc_id, similaridade) for doc_id, similaridade in ranking if similaridade >= 0.001]
        
        # Escrever a quantidade de documentos
        resposta_file.write(f"{len(ranking_filtrado)}\n")

        for doc_id, similaridade in ranking_filtrado:
            # Ajustar o índice para corresponder ao doc_id
            index = doc_id - 1
            # Verificar se index está dentro do intervalo válido
            if index < len(nomes_documentos):
                # Usar o nome real do documento em vez do ID
                nome_documento = nomes_documentos[index]
                resposta_file.write(f"{nome_documento} {similaridade}\n")


# função que vai gerar o arquivo de pesos contendo os pesos de cada termo em cada documento
def gravar_pesos(pesos_documentos, base_filename):
    # Ler os nomes dos documentos do arquivo base.txt
    with open(base_filename, "r", encoding="utf-8") as base_file:
        nomes_documentos = [linha.strip() for linha in base_file]

    with open("pesos.txt", "w", encoding="utf-8") as pesos_file:
        for doc_id, pesos in pesos_documentos.items():
            # Ajustar o índice para corresponder ao doc_id
            index = doc_id - 1
            # Verificar se index está dentro do intervalo válido
            if index < len(nomes_documentos):
                # Usar o nome real do documento em vez do ID
                nome_documento = nomes_documentos[index]
                pesos_file.write(f"{nome_documento}: ")
                for termo, peso in pesos:
                    if peso != 0:
                        pesos_file.write(f"{termo}, {peso}  ")  # Adicionado dois espaços extras
                pesos_file.write("\n")


# função que grava o indice invertido em um arquivo
def gravar_indice_invertido(indice_invertido):
    with open("indice_invertido.txt", "w", encoding="utf-8") as indice_invertido_file:
        for termo, lista_documentos in indice_invertido.items():
            indice_invertido_file.write(f"{termo}: ")
            for doc_id, freq in lista_documentos:
                indice_invertido_file.write(f"{doc_id}, {freq}  ")  # Adicionado dois espaços extras
            indice_invertido_file.write("\n")


def main(base_filename, consulta_filename):
    # ler a base de documentos
    documentos = ler_base(base_filename)

    # gerar o indice invertido
    indice_invertido = construir_indice_invertido(documentos)

    # ler a consulta
    consulta = ler_query(consulta_filename)

    # gerar o modelo vetorial
    ranking = modelo_vetorial(indice_invertido, consulta, base_filename)

    # gerar o arquivo de resposta
    gravar_resposta(ranking, base_filename)
    
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python modelo_booleano.py base.txt consulta.txt")
    else:
        base_filename = sys.argv[1]
        consulta_filename = sys.argv[2]
        main(base_filename, consulta_filename)     