import pandas as pd
import pydea
from itertools import combinations
from datetime import datetime

begin = datetime.now()

#carregando dataset com fatores
df_fatores_candidatos = pd.read_excel('3-dataset_completo.xlsx', decimal=',')

#criando lista com nomes de fatores 
fatores_lista = list(df_fatores_candidatos.columns)

#gerando as combinacoes 7 a 7
lista_combinacoes = list(combinations(fatores_lista, 7))

print(f'Tamanho da lista de combinacoes antes da remocao: {len(lista_combinacoes)}')

#removendo combinacoes que possuem somente inputs ou somente outputs
nova_lista_combinacoes = []
for combinacao in lista_combinacoes:
    contador = sum(1 for item in combinacao if 'input' in item)
    if contador != len(combinacao) and contador != 0:
        nova_lista_combinacoes.append(combinacao)

print(f'Tamanho da lista de combinacoes apos a remocao: {len(nova_lista_combinacoes)}')


with open('combinacoes.txt', 'w') as e:
    e.write("Numero de combinacoes: ")
    e.write(str(len(nova_lista_combinacoes)))
    e.write("\n")

#dataframe que ira armazenar o melhor resultado
melhor_resultado = pd.DataFrame()

#lista que ira armazenar os fatores do melhor resultado
melhores_fatores = []

cont = 0
for i in nova_lista_combinacoes:
    cont = cont + 1
    with open('execucao.txt', 'w') as x:
        x.write("Processando a combinacao ")
        x.write(str(cont))
        x.write("\n")
    inputs = pd.DataFrame()
    outputs = pd.DataFrame()
    resultado_candidato = pd.DataFrame()

    for j in i:
        if 'input' in j: 
            inputs[j] = df_fatores_candidatos[j]
        else:
            outputs[j] = df_fatores_candidatos[j]

    modelo = 'VRS'
    problema = pydea.DEAProblem(inputs, outputs, returns=modelo)
    resultado = problema.solve()
    resultado_candidato['eficiencia'] = resultado['Efficiency']
    if len(melhores_fatores) == 0: #se for a primeira execucao, popula as variaveis
        melhor_resultado['eficiencia'] = resultado_candidato['eficiencia']
        melhores_fatores = list(inputs.columns) + list(outputs.columns)
    else: #se nao, compara o resultado atual com o melhor resultado e substitui se for o caso
        variancia_candidato = resultado_candidato['eficiencia'].var()
        variancia_melhor = melhor_resultado['eficiencia'].var()

        if variancia_candidato > variancia_melhor: #se o resultado atual for melhor, substitui
            melhor_resultado['eficiencia'] = resultado_candidato['eficiencia']
            melhores_fatores = list(inputs.columns) + list(outputs.columns)

end = datetime.now()

begin_time = begin.strftime("%H:%M:%S")
end_time = end.strftime("%H:%M:%S")
k = melhor_resultado['eficiencia'].var()

arquivo_resultado = '4-resultados_sem_restricao_' + modelo + '.txt'
with open(arquivo_resultado, 'w') as f:
    f.write("Inicio: ")
    f.write(begin_time)
    f.write("\n")
    f.write("Termino: ")
    f.write(end_time)
    f.write("\n")
    f.write("Variancia melhor resultado: ")
    f.write(str(k))
    f.write("\n")
    f.write("Eficiencias:")
    f.write("\n")
    f.write(str(melhor_resultado['eficiencia']))
    f.write("\n")
    f.write("Fatores:")
    f.write("\n")
    f.write(str(melhores_fatores))