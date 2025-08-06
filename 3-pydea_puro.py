import pandas as pd
import pydea
from datetime import datetime

begin = datetime.now()

#carregando dataset com fatores
inputs = pd.read_excel('1-dataset_inputs.xlsx')
outputs = pd.read_excel('2-dataset_outputs.xlsx')

modelo = 'CRS'
problema = pydea.DEAProblem(inputs, outputs, returns=modelo)
resultado = problema.solve()

tempo_total = begin - datetime.now()

arquivo = f'3-pydea_puro_resultado_'+modelo+'.txt'

# Salvando a Series em um arquivo de texto
with open(arquivo, 'w') as f:
    f.write(resultado['Efficiency'].to_string())
