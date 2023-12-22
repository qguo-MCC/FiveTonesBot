import streamlit as st
from pathlib import Path
import pandas as pd
import re
st.set_page_config(layout="wide")
root = Path("data")
#load data
image_df = pd.DataFrame({'path': list(root.joinpath('accupoints').glob('*.png'))})
image_df['name'] = image_df['path'].apply(lambda e: e.name)
image_df['point'] = image_df['name'].apply(lambda e: re.search('(\w+)\d\.png', e).group(1))
images = image_df.groupby('point')['path'].apply(lambda s: s.to_list()).reset_index()
pdf = pd.read_excel(root.joinpath('meridian17.xlsx'), engine='openpyxl')
pdf = pdf.merge(images.rename(columns={'point':'name'}), on='name', how='left')

with st.sidebar:
    point = st.selectbox('选择穴位',tuple(pdf['name'].to_list()))

st.title(point)
pdata = pdf.loc[pdf['name']==point]
for img in pdata['path'].iloc[0]:
    st.image(img.__str__())
st.write(pdata['point'].iloc[0].replace('\n', '<br>'), unsafe_allow_html=True)