import streamlit as st
from pathlib import Path
import pandas as pd
import os
import re
from googletrans import Translator
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from src.utilities import load_obj
import ast
translator = Translator()
st.set_page_config()
root = Path("data")
with st.sidebar:
    task = st.selectbox('请选择任务：',['建议穴位','查找穴位','搜索穴位','更新穴位信息','添加穴位'])

if task == '建议穴位':
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local(root.joinpath("body_parts").__str__(), embeddings)
    st.title('五音堂选穴仪')
    meridians = st.multiselect(
        '患病经络(可以多选)：',
        ['肺经', '大肠经', '胃经', '脾经', '心经', '小肠经', '膀胱经小趾外侧线', '膀胱经小趾内侧线', '肾经前线', '肾经后线', '心包经', '三焦经', '胆经前线', '胆经后线', '肝经', '督脉', '任脉'],
    )
    parts_text = st.text_input('请列出患病部位。如果有多个部位，请用，分开不同部位。')

    #parts = st.multiselect('病痛部位(可以多选)：',data.columns[25:].to_list())
    n = st.text_input('建议学位数量', '3')
    N = int(n)
    if st.button('选穴'):
        conn = st.connection('data_db', type='sql')
        engine = conn.engine

        #embeddings = load_obj('all-mpnet-base-v2.model')



        normal_map = pd.read_excel(root.joinpath('normal_map_long.xlsx'), engine='openpyxl')
        image_df = pd.DataFrame({'path': list(root.joinpath('accupoints').glob('*.png'))})
        image_df['name'] = image_df['path'].apply(lambda e: e.name)
        image_df['point'] = image_df['name'].apply(lambda e: re.search('(\w+)\d\.png', e).group(1))
        images = image_df.groupby('point')['path'].apply(lambda s: s.to_list()).reset_index()


        available_parts = normal_map['CPart'].to_list()
        if len(meridians) ==0:
            if len(parts_text)==0:
                st.header('没有选择经络和部位。至少需要输入一个病痛部位。')
            else:
                rparts = parts_text.split('，')
                parts = []
                for part in rparts:
                    if part in available_parts:
                        parts.append(normal_map.loc[normal_map['CPart'] == part, 'CLabel'].values[0])
                    else:
                        epart = translator.translate(part, src='zh-CN').text
                        epart = db.similarity_search(epart, k=1)[0].page_content
                        parts.append(normal_map.loc[normal_map['EPart'] == epart, 'CLabel'].iloc[0])
                data = pd.read_sql(f'SELECT {",".join(["name1", "穴名", "point"]+parts)} FROM points', con=conn.engine)
                data = data.merge(images.rename(columns={'point': 'name1'}), on='name1', how='left')
                df = data[['穴名']+parts].copy()
                df['总经络数'] = (df[parts]>0).sum(axis=1)
                df['总经络强度'] = df[parts].sum(axis=1)
                df.sort_values(['总经络数', '总经络强度'], ascending=False, inplace=True)

                selected_points = df.iloc[:N]['穴名'].to_list()
                st.header(selected_points)
                st.dataframe(df.iloc[:N])

                for point in selected_points:
                    pinfo = data.loc[data['穴名'] == point]
                    st.header(point)
                    if type(pinfo['path'].iloc[0]) ==list:
                        for img in pinfo['path'].iloc[0]:
                            st.image(img.__str__())
                    st.write(pinfo['point'].iloc[0].replace('\n', '<br>'), unsafe_allow_html=True)
        else:
            rparts = parts_text.split('，')
            parts = []
            for part in rparts:
                if part in available_parts:
                    parts.append(normal_map.loc[normal_map['CPart'] == part, 'CLabel'].values[0])
                else:
                    epart = translator.translate(part, src='zh-CN').text
                    epart = db.similarity_search(epart, k=1)[0].page_content
                    parts.append(normal_map.loc[normal_map['EPart'] == epart, 'CLabel'].iloc[0])
            data = pd.read_sql(f'SELECT {",".join(["name1", "穴名", "point"] + meridians + parts)} FROM points', con=conn.engine)
            data = data.merge(images.rename(columns={'point': 'name1'}), on='name1', how='left')
            df = data[['穴名']+meridians+parts].copy()
            df['总经络数'] = (df[meridians] > 0).sum(axis=1)
            df['总经络强度'] = df[meridians].sum(axis=1)
            df['总部位数'] = (df[parts] > 0).sum(axis=1)
            df['总部位强度'] = df[parts].sum(axis=1)
            df['总强度'] = (df['总经络强度']+df['总部位强度'])
            df.sort_values(['总经络数', '总强度'], ascending=False, inplace=True)

            selected_points = df.iloc[:N]['穴名'].to_list()
            st.header(selected_points)
            st.dataframe(df.iloc[:N])

            for point in selected_points:
                pinfo = data.loc[data['穴名']==point]
                st.header(point)
                if type(pinfo['path'].iloc[0]) ==list:
                    for img in pinfo['path'].iloc[0]:
                        st.image(img.__str__())
                st.write(pinfo['point'].iloc[0].replace('\n', '<br>'), unsafe_allow_html=True)
elif task == '查找穴位':
    conn = st.connection('data_db', type='sql')
    engine = conn.engine
    image_df = pd.DataFrame({'path': list(root.joinpath('accupoints').glob('*.png'))})
    image_df['name'] = image_df['path'].apply(lambda e: e.name)
    image_df['point'] = image_df['name'].apply(lambda e: re.search('(\w+)\d\.png', e).group(1))
    images = image_df.groupby('point')['path'].apply(lambda s: s.to_list()).reset_index()
    point_df = pd.read_sql('SELECT 经名, 穴名 FROM points', engine).fillna('奇穴').sort_values('经名')
    point_list = (point_df['经名'] + '-' +point_df['穴名']).to_list()
    select_point = st.selectbox('请选择穴位', point_list)
    query_point = select_point.split('-')[1]
    if st.button('查穴'):
        pinfo=pd.read_sql(f'SELECT * FROM points WHERE 穴名="{query_point}"', engine)
        pinfo = pinfo.merge(images.rename(columns={'point': 'name1'}), on='name1', how='left')
        if pinfo.shape[0]>0:
            st.header(f'{query_point}')
            if type(pinfo['path'].iloc[0]) == list:
                for img in pinfo['path'].iloc[0]:
                    st.image(img.__str__())
            st.write(re.sub('\n+', '<br>', pinfo['point'].iloc[0]), unsafe_allow_html=True)
        else:
            st.header('抱歉！现在数据库并无此穴。')
elif task == '搜索穴位':
    conn = st.connection('data_db', type='sql')
    engine = conn.engine
    image_df = pd.DataFrame({'path': list(root.joinpath('accupoints').glob('*.png'))})
    image_df['name'] = image_df['path'].apply(lambda e: e.name)
    image_df['point'] = image_df['name'].apply(lambda e: re.search('(\w+)\d\.png', e).group(1))
    images = image_df.groupby('point')['path'].apply(lambda s: s.to_list()).reset_index()
    query_point = st.text_input('请输入穴位名称：')
    if st.button('搜穴'):
        pinfo=pd.read_sql(f'SELECT * FROM points WHERE 穴名="{query_point}"', engine)
        pinfo = pinfo.merge(images.rename(columns={'point': 'name1'}), on='name1', how='left')
        if pinfo.shape[0]>0:
            st.header(f'{query_point}')
            if type(pinfo['path'].iloc[0]) == list:
                for img in pinfo['path'].iloc[0]:
                    st.image(img.__str__())
            st.write(re.sub('\n+', '<br>', pinfo['point'].iloc[0]), unsafe_allow_html=True)
        else:
            st.header('抱歉！现在数据库并无此穴。')


