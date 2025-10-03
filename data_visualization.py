import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time

st.set_page_config(
    page_title="Data Visualization",  
    page_icon="icon.png",            
    layout="wide"              
)

st.header("Data Visualization App")

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

file=st.file_uploader("Upload a CSV file",type=["csv"])

if file is not None:
    df=load_data(file)
    st.success("File uploaded successfully!")

    n_rows=st.slider("Number of rows to view",min_value=5000,max_value=len(df),value=5000,step=1000)

    columns=st.multiselect("Select columns to display",df.columns.tolist(),default=df.columns.tolist())

    st.write(df[:n_rows][columns])

    numerical_columns=df.select_dtypes(include=np.number).columns.tolist()
    all_columns=df.columns.tolist()

    tap1,tap2,tap3,tap4,tap5=st.tabs(["Scatter Plot","Histogram","Box Plot","Line Plot","Bar Plot"])


    with tap1:
      col1,col2,col3=st.columns(3)
      with st.spinner('Generating plot...'):
       time.sleep(2)
       with col1:
        x_axis=st.selectbox("Select X-axis",numerical_columns)
       with col2:
        y_axis=st.selectbox("Select Y-axis",numerical_columns)
       with col3:
        color=st.selectbox("Select column to be a color",numerical_columns)
       figer_scatter=px.scatter(df,x=x_axis,y=y_axis,color=color,title=f"Scatter plot of {y_axis} vs {x_axis} colored by {color}")
       st.success('Plot generated!')
       st.plotly_chart(figer_scatter)
    
    
    with tap2:
     hist_col1, hist_col2=st.columns(2)
     with st.spinner('Generating plot...'):
       time.sleep(2)
       with hist_col1:
        hist_feature=st.selectbox("Select feature for histogram",numerical_columns)
       with hist_col2:
        hist_color=st.selectbox("Select column to be a color for histogram",numerical_columns)
       fig_hist=px.histogram(df,x=hist_feature,color=hist_color,title=f"Histogram of {hist_feature} colored by {hist_color}")
       st.success('Plot generated!')
       st.plotly_chart(fig_hist)

    with tap3:
     box_col1, box_col2=st.columns(2)
     with st.spinner('Generating plot...'):
       time.sleep(2)
       with box_col1:
        box_y=st.selectbox("Select Y-axis for box plot",numerical_columns)
       with box_col2:
        box_color=st.selectbox("Select column to be a color for box plot",numerical_columns)
       fig_box=px.box(df,y=box_y,color=box_color,title=f"Box plot of {box_y} colored by {box_color}")
       st.success('Plot generated!')
       st.plotly_chart(fig_box)

    with tap4:
     line_col1, line_col2=st.columns(2)
     with st.spinner('Generating plot...'):
       time.sleep(2)
       with line_col1:
        line_x=st.selectbox("Select X-axis for line plot",numerical_columns)
       with line_col2:
        line_y=st.selectbox("Select Y-axis for line plot",numerical_columns)
       fig_line=px.line(df,x=line_x,y=line_y,title=f"Line plot of {line_y} vs {line_x}")
       st.success('Plot generated!')
       st.plotly_chart(fig_line)

    with tap5:
     bar_col1, bar_col2=st.columns(2)
     with st.spinner('Generating plot...'):
       time.sleep(2)
       with bar_col1:
        bar_x=st.selectbox("Select X-axis for bar plot",all_columns)
       with bar_col2:
        bar_y=st.selectbox("Select Y-axis for bar plot",all_columns)
       fig_bar=px.bar(df,x=bar_x,y=bar_y,title=f"Bar plot of {bar_y} vs {bar_x}")
       st.success('Plot generated!')
       st.plotly_chart(fig_bar)

       # --- Footer ---
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0E1117;
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        border-top: 1px solid #333;
    }
    .footer a {
        color: #4CAF50;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
        color: #408EE0;
    }
    </style>
    <div class="footer">
        Developed by <b>Mohamed Yasser</b> | 
        <a href="https://www.linkedin.com/in/mohamed--yasser" target="_blank">LinkedIn</a> | 
        <a href="mailto:mohamedshahen11223344@gmail.com">Email</a> | 
        <a href="https://github.com/Mohamed-Yasser-SE/" target="_blank">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)

   