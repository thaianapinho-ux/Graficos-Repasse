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
        y=y,  # pixels from top
        text=texto,
        color = alt.condition(alt.datum.int > 0, alt.value('red'), alt.value('green')) if conditional_color else alt.value('black')
        )
    return text_chart

def make_graph_repasse(df, h_chart, w_chart, h_pic, w_pic, x, y, canal):
    
    y_min = df[y].min()
    y_max = df[y].max()
    
    df['text1'] = df[x]

    df['text2'] = 'TTC: R$ ' + (df['TTC'].map('{:,.2f}'.format).astype(str))
    df['text3'] = 'TTV: R$ ' + (df['TTV CX']).round(2).map('{:,.2f}'.format).astype(str)
    df['text4'] = 'Mg: ' + df['Mg'].map('{:,.2%}'.format).astype(str)
    
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
    font_size = 20
    
    tick = chart.mark_tick(
        yOffset=tick_offset,
        color='#F78F3B',
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
    text2 = make_text(df, 'text2', font_size, tick_offset + font_size * 3, x, y, f_weight='bold')
    text3 = make_text(df, 'text3', font_size, tick_offset + font_size * 4, x, y, f_weight='bold')
    text4 = make_text(df, 'text4', font_size, tick_offset + font_size * 5, x, y, f_weight='bold')   
        
    return (chart + text1 + text2 + text3 + text4 + tick)

repasse = pl.read_parquet('data/repasse/arvore.parquet')
depara_repasse = pl.read_parquet('data/repasse/depara_repasse.parquet')

repasse = repasse.groupby(['UF', 'SKU Abrev']).agg([
    pl.col("^.*Rota$").first(),
    pl.col("^.*ASR$").first()
])

cols = [x for x in repasse.columns if x not in ['UF', 'SKU Abrev']]

repasse = repasse.melt(['UF', 'SKU Abrev'], cols ,'Canal','Preço').with_columns([
    pl.col('Canal').str.split(' ').arr[-1].alias('seg'),
    pl.col('Canal').str.split(' ').arr.join(' ').str.replace(' Rota','').str.replace(' ASR','').alias('kpi')
]).pivot('Preço',['UF', 'SKU Abrev', 'seg'],'kpi').with_columns([
    pl.col(["TTV","TTV CX"]).cast(pl.Float64).round(2),
    pl.col(["TTC","Mg"]).fill_null('-').str.replace('-','0').cast(pl.Float64)
    ])

repasse = repasse.join(depara_repasse, left_on='SKU Abrev', right_on = 'SKU').to_pandas()

#repasse_full = repasse.copy(deep=True)

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
    uf = st.selectbox('UF', sorted(repasse['UF'].unique()), 0, key='uf')
with col2:
    canal = st.selectbox('Canal', repasse['seg'].unique(), 0, key='canal')
with col3:
    Grupo = st.selectbox('Grupo', ['Single', 'Multi'], 0, key='grupo')
    #repasse = repasse[repasse['Grupo'] == st.session_state['grupo']]
    
repasse = repasse[repasse['UF'] == st.session_state['uf']]
#repasse = repasse[repasse['Grupo'] == st.session_state['grupo']]
repasse = repasse[repasse['seg'] == st.session_state['canal']]
    
grupo_dict = {'Single': ['Single', 'Premium'],
               'Multi': ['Multi1', 'Multi2']}

graphs = [make_graph_repasse(repasse[repasse['Grupo'] == grupo], 300, 1600, 120, 75, 'nome_slide', 'TTC', st.session_state['canal']) for grupo in grupo_dict[st.session_state['grupo']]]

graph = alt.vconcat(*graphs)#.properties(title=f"Resumo Repasse NAB - {st.session_state['uf']} - {st.session_state['canal'].capitalize()} - {st.session_state['grupo']}")

st.altair_chart(graph)

# from altair_saver import save

# save(graph, "chart.png", method='selenium') 


