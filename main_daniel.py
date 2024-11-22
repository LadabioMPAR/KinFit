#importando dados da planilha
def ler_ensaio(nome_planilha):
    U=[]
    dados=[]
    nome_ensaios=[]
    nome_U=[]
    for index, (sheet_name, df) in enumerate(teste.items()):
        if index%2==0:
            dados.append(df.drop(df.columns[0], axis=1))
            nome_ensaios.append(sheet_name)
        else:
            U.append(df)
            nome_U.append(df.drop(df.columns[0], axis=1))
    
    return dados, U, nome_ensaios



#importando ODE
def ode():
#importando configurações

#Realizando otimização

#salvando dados

#plota dados



import pandas as pd

teste = {name: sheet for name, sheet in pd.read_excel("Entradas_exemplo.xlsx",sheet_name=None).items() if name != 'Condição Inicial'}


