import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter

import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import missingno



def create_correlation_chart(corr_df): ## function to create correlation chart using matplotlib
    fig = plt.figure(figsize=(15,15))
    plt.imshow(corr_df.values, cmap='Blues')
    plt.xticks(range(corr_df.shape[0]), corr_df.columns, rotation=90, fontsize=15)
    plt.yticks(range(corr_df.shape[0]), corr_df.columns, fontsize=15)
    plt.colorbar()
    
    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[0]):
            plt.text(i, j, "{:.2f}".format(corr_df.values[i,j]), color="red", ha='center', fontsize=14, fontweight="bold")
            
    return fig

def create_missing_values_bar(df): ## function to create missing values bar chart
    missing_fig = plt.figure(figsize=(10,5))
    ax = missing_fig.add_subplot(111)
    missingno.bar(df, figsize=(10,5), fontsize=12, ax=ax)
    
    return missing_fig

def find_cat_cont_columns(df): ## logic to separate continuous & categorical columns
    cont_columns, cat_columns = [],[]
    for col in df.columns:
        if len(df[col].unique()) <= 25 or df[col].dtype == np.object_: ## if less than 25 unique values or string
            cat_columns.append(col.strip())
        else:
            cont_columns.append(col.strip())
    return cont_columns, cat_columns

## web app / dashboard code
st.set_page_config(page_icon=":bar_chart:", page_title="EDA Automated using Python")

upload = st.file_uploader(label="Upload File Here:", type=["csv"])

if upload: ## file as Bytes
    df = pd.read_csv(upload)
    cont_columns, cat_columns = find_cat_cont_columns(df)
    
    tab1, tab2, tab3 = st.tabs(["Dataset Overview :clipboard:", "Individual column stats :bar_chart:", "Explore Relation Between Features :chart:"])
    
    with tab1: ## dataset overview tab
        st.subheader("1. Dataset")
        st.write(df)
        
        st.subheader("2. Dataset Overview")
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}</span>".format("Rows", df.shape[0]), unsafe_allow_html=True)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}</span>".format("Duplicates", df.shape[0] - df.drop_duplicates().shape[0]), unsafe_allow_html=True)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}</span>".format("Features", df.shape[1]), unsafe_allow_html=True)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}</span>".format("Categorical Columns", len(cont_columns)), unsafe_allow_html=True)
        st.write(cat_columns)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}</span>".format("Continuous Columns", len(cont_columns)), unsafe_allow_html=True)
        st.write(cont_columns)
        
        st.subheader("3. Correlation Chart")
        corr_df = df[cont_columns].corr()
        corr_fig = create_correlation_chart(corr_df)
        st.pyplot(corr_fig, use_container_width=True)
        
        st.subheader("4. Missing Values Disturbution")
        missing_fig = create_missing_values_bar(df)
        st.pyplot(missing_fig, use_container_width=True)

    with tab2: # Individual Column stats
        df_descr = df.describe()
        st.subheader("Analyze Individual Feature Distribution")

        st.markdown("### 1. Understand Continuous Feature")
        feature = st.selectbox(label="Select Continuous Feature", options=cont_columns, index=0)

        na_cnt = df[feature].isna().sum()
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}".format("Count", df_descr[feature]['count']), unsafe_allow_html=True)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {} / ({:.2f} %)".format("Missing Count", na_cnt, na_cnt/df.shape[0]), unsafe_allow_html=True)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {:.2f}".format("mean", df_descr[feature]['mean']), unsafe_allow_html=True)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {:.2f}".format("Standard Deviation", df_descr[feature]['std']), unsafe_allow_html=True)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}".format("Minimum", df_descr[feature]['min']), unsafe_allow_html=True)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}".format("Maximum", df_descr[feature]['max']), unsafe_allow_html=True)
        st.markdown("<span style='font-weight:bold;'>{}</span> :".format("Quantiles"), unsafe_allow_html=True)
        st.write(df_descr[[feature]].T[['25%', "50%", "75%"]])

        # Histogram
        hist_fig = px.histogram(df, x=feature, nbins=50)
        hist_fig.update_layout(height=500, width=600)
        st.plotly_chart(hist_fig, use_container_width=True)
        
    with tab3: ## Explore relation between features
        st.subheader("Explore Relation Between Features of Dataset")

        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox(label="X-Axis", options=cont_columns, index=0)
        with col2:
            y_axis = st.selectbox(label="Y-Axis", options=cont_columns, index=1)

        color_encode = st.selectbox(label="Color-Encode", options=[None,] + cat_columns)
        
        ## Scatter chart showing relationship between data features.
        scatter_fig = px.scatter(df, x=x_axis, y=y_axis, color=color_encode)
        scatter_fig.update_layout(
            title="{} vs {}".format(x_axis.capitalize(), y_axis.capitalize()),
            height=500, width=600,
            font=dict(size=20),
            xaxis=dict(title=x_axis),
            yaxis=dict(title=y_axis)
        )
        st.plotly_chart(scatter_fig, use_container_width=True)
        
    