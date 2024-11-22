
import pandas as pd
import numpy as np
teste = {name: sheet for name, sheet in pd.read_excel("Entradas_exemplo.xlsx",sheet_name=None).items() if name != 'Condição Inicial'}
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
def ode(t, y, Par):


    # Parametros
    mmax = Par[0]
    Ks = Par[1]
    Sm = Par[2]
    Pm = Par[3]
    Yxs = Par[4]
    Yps = Par[5]

    # VAriáveis de estado
    V = y[0]
    Cx = y[1]
    Cs = y[2]
    Cp = y[3]

    
    mi = (mmax * Cs / (Ks + Cs)) * np.exp(-Cs / Sm) * (1 - Cp / Pm)

    # Batelada
    F = 0

    # ODE 
    dV = F
    dCx = mi * Cx
    dCs = -(1 / Yxs) * mi * Cx
    dCp = (Yps / Yxs) * mi * Cx
    
    return np.array([dV, dCx, dCs, dCp])
#importando configurações






#Realizando otimização


#salvando dados

#plota dados






