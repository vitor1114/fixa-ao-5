# fixa-ao-5


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



np.random.seed(42)


datas_transacoes = pd.date_range(
    start="2026-09-01",
    periods=1000,
    freq="h"
)

df_transacoes = pd.DataFrame({
    "id_cliente": np.random.choice(
        ["C100", "C101", "C102", "C103", "C104"],
        size=1000
    ),
    "data_transacao": datas_transacoes,
    "valor": np.random.choice(
        [np.nan, 150.0, 3000.0, 7500.0, 12000.0, 50.0],
        size=1000,
        p=[0.05, 0.40, 0.30, 0.15, 0.05, 0.05]
    ),
    "estado_cliente": np.random.choice(
        ["SP", "RJ", "MG", "RS"],
        size=1000
    ),
    "origem": np.random.choice(
        ["web", "mobile_app", "atm"],
        size=1000
    )
})


df_transacoes = pd.concat(
    [df_transacoes, df_transacoes.iloc[:15]],
    ignore_index=True
)

df_transacoes.to_csv(
    "transacoes.csv",
    index=False,
    encoding="latin1"
)

datas_cotacoes = pd.date_range(
    start="2026-09-01",
    end="2026-10-01",
    freq="D"
)

variacoes = np.random.normal(
    loc=0.001,
    scale=0.015,
    size=len(datas_cotacoes)
)

preco_inicial = 5.20

precos = preco_inicial * np.exp(
    np.cumsum(variacoes)
)

df_cotacoes = pd.DataFrame({
    "data": datas_cotacoes,
    "cotacao_usd": np.round(precos, 4),
    "volume_negociado": np.random.randint(
        10000,
        500000,
        size=len(datas_cotacoes)
    )
})


df_cotacoes.loc[5, "cotacao_usd"] = np.nan
df_cotacoes.loc[18, "cotacao_usd"] = np.nan

df_cotacoes.to_csv(
    "cotacoes.csv",
    index=False,
    encoding="utf-8"
)

print("Arquivos CSV gerados com sucesso!")



df = pd.read_csv(
    "transacoes.csv",
    encoding="latin1"
)


cotacoes = pd.read_csv(
    "cotacoes.csv",
    encoding="utf-8"
)

print("\nDados carregados com sucesso!")




df = df.drop_duplicates(
    keep="first"
)




df["data_transacao"] = pd.to_datetime(
    df["data_transacao"]
)


df["data_transacao"] = df[
    "data_transacao"
].dt.tz_localize(
    "America/Sao_Paulo"
)



mediana_por_estado = df.groupby(
    "estado_cliente"
)["valor"].transform("median")

df["valor"] = df["valor"].fillna(
    mediana_por_estado
)



df["plataforma"] = "Mobile"




df["dia_semana"] = df[
    "data_transacao"
].dt.day_name()

df["mes"] = df[
    "data_transacao"
].dt.month



risco_dict = {
    "C100": "Baixo",
    "C101": "Alto",
    "C102": "Médio",
    "C103": "Baixo",
    "C104": "Alto"
}


df["nivel_risco"] = df[
    "id_cliente"
].map(risco_dict)




filtro = (
    (df["mes"] == 9)
    &
    (
        (df["estado_cliente"] == "SP")
        |
        (df["estado_cliente"] == "RJ")
    )
    &
    (df["valor"] > 5000)
)

df_setembro = df.loc[
    filtro
].copy()



tabela_pivot = pd.pivot_table(
    df,
    index="mes",
    columns="nivel_risco",
    values="valor",
    aggfunc="sum",
    margins=True,
    margins_name="Total",
    fill_value=0
)



media_estado = df.groupby(
    "estado_cliente"
)["valor"].transform("mean")

desvio_estado = df.groupby(
    "estado_cliente"
)["valor"].transform("std")

df["z_score"] = (
    df["valor"] - media_estado
) / desvio_estado




anomalias = df.loc[
    df["z_score"] > 2.5
].copy()



total_diario = (
    df.set_index("data_transacao")
    ["valor"]
    .resample("D")
    .sum()
)



media_movel = total_diario.rolling(
    7
).mean()




fig, ax = plt.subplots(
    figsize=(12, 6)
)

ax.plot(
    total_diario.index,
    total_diario.values,
    label="Valor total de transações"
)

ax.plot(
    media_movel.index,
    media_movel.values,
    label="Média móvel de 7 dias"
)


ax.set_ylim(
    bottom=0
)

ax.set_title(
    "Desempenho Diário das Transações"
)

ax.set_xlabel(
    "Data"
)

ax.set_ylabel(
    "Valor das Transações (R$)"
)

ax.legend()

fig.tight_layout()

plt.show()




print("\n======================================")
print("        RELATÓRIO DA ANÁLISE")
print("======================================")

print(
    "\nQuantidade de transações após "
    "remoção de duplicadas:",
    len(df)
)

print(
    "\nTransações de setembro em SP/RJ "
    "acima de R$ 5.000:",
    len(df_setembro)
)

print(
    "\nQuantidade de possíveis anomalias "
    "(Z > 2.5):",
    len(anomalias)
)

print("\n======================================")
print("          TABELA DINÂMICA")
print("======================================")

print(tabela_pivot)

print("\n======================================")
print("       POSSÍVEIS ANOMALIAS")
print("======================================")

print(
    anomalias[
        [
            "id_cliente",
            "data_transacao",
            "valor",
            "estado_cliente",
            "nivel_risco",
            "z_score"
        ]
    ]
)


df.to_csv(
    "transacoes_tratadas.csv",
    index=False,
    encoding="utf-8-sig"
)

df_setembro.to_csv(
    "transacoes_setembro_sp_rj_acima_5000.csv",
    index=False,
    encoding="utf-8-sig"
)

tabela_pivot.to_csv(
    "tabela_pivot_risco.csv",
    encoding="utf-8-sig"
)

anomalias.to_csv(
    "anomalias_zscore.csv",
    index=False,
    encoding="utf-8-sig"
)

fig.savefig(
    "grafico_transacoes.png",
    dpi=150
)

print("\n======================================")
print("Arquivos de resultado criados:")
print("======================================")
print("transacoes_tratadas.csv")
print("transacoes_setembro_sp_rj_acima_5000.csv")
print("tabela_pivot_risco.csv")
print("anomalias_zscore.csv")
print("grafico_transacoes.png")
print("\nAnálise concluída!")
