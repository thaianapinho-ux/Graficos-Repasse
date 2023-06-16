import streamlit as st

import altair as alt

from functions import *

import pandas as pd
import polars as pl

from PIL import Image as PILImage
import io
import base64

def convert_image(str):
    image = PILImage.open(str)
    output = io.BytesIO()    
    image.save(output, format='PNG')
    encoded_string = "data:image/jpeg;base64,"+base64.b64encode(output.getvalue()).decode()
    return encoded_string

if check_for_new_file('data/repasse/graficos.xlsx', 'data/repasse/*.parquet'):
    read_excel_parquets('data/repasse/graficos.xlsx')
    depara_repasse = pl.read_parquet('data/repasse/depara_repasse.parquet').to_pandas()
    depara_repasse['Caminho'] = depara_repasse['Caminho'].apply(convert_image)    
    depara_repasse.to_parquet('data/repasse/depara_repasse.parquet')

def make_text(df, texto, fonte, offset, x, y, conditional_color = False, f_weight = 'normal', mono=False):
    text_chart = alt.Chart(df).mark_text(
        align="center",
        dy=offset,
        size=fonte, fontWeight=f_weight, lineBreak=r'\n',
        font = 'courier' if mono else ''
    ).encode(
        x=f'{x + "label"}:N',  # pixels from left
        y=alt.Y(y,axis=alt.Axis(labels=False)),  # pixels from top
        text=texto,
        color = alt.condition(alt.datum.int > 0, alt.value('red'), alt.value('green')) if conditional_color else alt.value('black')
        )
    return text_chart

def make_graph_repasse(df, h_chart, w_chart, h_pic, w_pic, x, y, canal):
    
    y_min = df[y].min()
    y_max = df[y].max()
    
    df['text1'] = df[x]
    
    df['text2'] = 'TTV Pré: ' + df['NORMAL TTC PRÉ'].astype(str) if canal in ['asr','bar'] else 'TTC Pré: ' + df['NORMAL TTC PRÉ'].astype(str)
    df['text3'] = 'TTV Pos: ' + df['NORMAL TTC PÓS'].astype(str) if canal in ['asr','bar'] else 'TTC Pos: ' + df['NORMAL TTC PÓS'].astype(str)
    df['int'] = df['NORMAL TTC PÓS'].astype(float) - df['NORMAL TTC PRÉ'].astype(float)
    df['text4'] = 'Delta: R$ ' + df['int'].round(2).astype(str)
    df['text5'] = 'R$ Promo | % Promo'.center(25)
    df['text6'] = 'Pré: R$ ' + df['PROMO TTC PRÉ'].astype(str) + ' | ' + df['PROMO % VOL PRÉ'].astype(str)
    df['text6'] = df['text6'].str.center(25)
    df['text7'] = 'Pós: R$ ' + df['PROMO TTC PÓS'].astype(str).str.rjust(3) + ' | ' + df['PROMO % VOL PÓS'].astype(str)

    
    df[x + 'label'] = df[y].astype(str).str.replace('.','').replace('10','z') + df[x]
    
    df[x + 'label'] = df[x + 'label'].str.replace('10','z')
    
    chart = alt.Chart(df).mark_image(
    height=h_pic,
    width=w_pic,
    baseline='bottom').encode(
        x=alt.X(f'{x + "label"}:N', axis=alt.Axis(labels=False, title="")),
        y=alt.Y(f'{y}:Q', axis=alt.Axis(labels=False, grid=False, title=""), scale=alt.Scale(domain=[y_min-0.5,y_max+0.5])),
        url='Caminho').properties(
            height = h_chart,
            width = w_chart
        )
    
    tick_offset = 12
    font_size = 15
    
    tick = chart.mark_tick(
        yOffset=tick_offset,
        color='black',
        thickness=2,
        size = w_chart/len(df[x].unique()) - 20  # controls width of tick.
    ).encode(
        x=f'{x + "label"}',
        y=alt.Y(y, axis=alt.Axis(labels=False))
    )
    
    tick2 = chart.mark_tick(
        yOffset=tick_offset + font_size * 4 + 12,
        color='black',
        thickness=2,
        size = w_chart/len(df[x].unique()) - 20  # controls width of tick.
    ).encode(
        x=f'{x + "label"}',
        y=alt.Y(y, axis=alt.Axis(labels=False))
    )
    
    #texts = [text for text in df.columns if text.startswith('text')]
    
    #text_graphs = [make_text(df, text, font_size, tick_offset + font_size * i+1 + 12, x, y) for i,text in enumerate(texts)]

    text1 = make_text(df, 'text1', font_size, tick_offset + font_size, x, y, f_weight='bold')
    text2 = make_text(df, 'text2', font_size, tick_offset + font_size * 2, x, y)
    text3 = make_text(df, 'text3', font_size, tick_offset + font_size * 3, x, y)
    text4 = make_text(df, 'text4', font_size, tick_offset + font_size * 4, x, y, conditional_color=True)   
    
    if canal in ['varejo', 'atacado']:
        text5 = make_text(df, 'text5', font_size, tick_offset + font_size * 5 + 12, x, y)
        text6 = make_text(df, 'text6', font_size, tick_offset + font_size * 6 + 12, x, y)
        text7 = make_text(df, 'text7', font_size, tick_offset + font_size * 7 + 12, x, y)
        return (chart + text1 + text2 + text3 + text4 + text5 + text6 + text7 + tick + tick2)
    
    return (chart + text1 + text2 + text3 + text4 + tick)

repasse = pl.read_parquet('data/repasse/repasse.parquet')
depara_repasse = pl.read_parquet('data/repasse/depara_repasse.parquet')

print(repasse.columns)

repasse = repasse.select(['UF', 'GEO', 'Emb.', 'SKU', 'KPI', 'TOP 30', 'ATAC. NAC.', 'ASR', 'BAR', 'TOP 30_duplicated_0', 'ATAC. NAC._duplicated_0'])

repasse.columns = ['uf', 'geo', 'embalagem', 'sku', 'kpi', 'varejo', 'atacado', 'asr', 'bar', 'promo varejo', 'desin atacado']

#repasse = repasse.filter(pl.col('kpi').is_in(['TTC PRÉ', 'TTC PÓS', '% TTC']))

#repasse = repasse.pivot(values=['varejo', 'atacado', 'asr', 'bar'], index=['uf','embalagem', "sku", 'varejo', 'atacado', 'asr', 'bar'], columns="kpi")

repasse = repasse.melt(['geo', 'uf', 'embalagem', 'sku', 'kpi'],['varejo','atacado','asr', 'bar', 'promo varejo', 'desin atacado'],'canal','price')

repasse = repasse.with_columns([
    pl.when(pl.col('canal').str.starts_with('promo')).then('PROMO').otherwise('NORMAL').alias('promo')
]).with_columns([
    pl.concat_str(['promo', 'kpi'], separator=' ').alias('kpi'),
    pl.col('canal').str.replace('promo ','')
]).drop('promo')

repasse = repasse.pivot(['price'],['geo', 'uf', 'embalagem', 'sku', 'canal'], ['kpi'])
repasse = repasse.fill_null(0)

repasse = repasse.join(depara_repasse, left_on='sku', right_on = 'SKU').to_pandas()

repasse_full = repasse.copy(deep=True)

st.set_page_config(
    page_title="Resumo Repasse", page_icon="📖", initial_sidebar_state="expanded", layout='wide'
)

reduce_header_height_style = """
    <style>
        div.block-container {padding-top:1rem;}
    </style>
"""
st.markdown(reduce_header_height_style, unsafe_allow_html=True)

st.title('Resume do Repasse NAB - 20/Mar/2023')

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    uf = st.selectbox('UF', repasse['uf'].unique(), 0, key='uf')
with col2:
    canal = st.selectbox('Canal', repasse['canal'].unique(), 2, key='canal')
with col3:
    Grupo = st.selectbox('Grupo', ['Single', 'Multi'], 0, key='grupo')
    #repasse = repasse[repasse['Grupo'] == st.session_state['grupo']]
with col4:
    embalagem = st.multiselect('Embalagem', repasse['embalagem'].unique(), key='embalagem')
with col5:
    sku = st.multiselect('SKU', repasse['sku'].unique(), key='sku')
    
repasse = repasse[repasse['uf'] == st.session_state['uf']]
#repasse = repasse[repasse['Grupo'] == st.session_state['grupo']]
repasse = repasse[repasse['canal'] == st.session_state['canal']]

for x in ['embalagem', 'sku']:
    if st.session_state[x]:
        repasse = repasse[repasse[x].isin(st.session_state[x])]
    
grupo_dict = {'Single': ['Single', 'Premium'],
               'Multi': ['Multi1', 'Multi2']}

graphs = [make_graph_repasse(repasse[repasse['Grupo'] == grupo], 250, 1600, 120, 75, 'nome_slide', 'NORMAL TTC PÓS', st.session_state['canal']) for grupo in grupo_dict[st.session_state['grupo']]]

graph = alt.vconcat(*graphs).properties(title=f"Resumo Repasse NAB - {st.session_state['uf']} - {st.session_state['canal'].capitalize()} - {st.session_state['grupo']}")

st.altair_chart(graph)

# from altair_saver import save

# save(graph, "chart.png", method='selenium') 


