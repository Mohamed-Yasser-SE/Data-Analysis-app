import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
st.set_page_config(
    page_title="Data Science App",  
    page_icon="icon.png",            
    layout="wide"              
)

st.header("Data Visualization App")

# Load Data

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

file=st.file_uploader("Upload a CSV file",type=["csv"])
if 'df' not in st.session_state:
    st.session_state.df=None

if file is not None:
    data=load_data(file)
    if st.session_state.df is None:
      st.session_state.df=data
    st.success("File uploaded successfully!")

    csv_bytes = st.session_state.df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download your data as Data_science_app.csv",
                                               data=csv_bytes,
                                               file_name="Data_science_app.csv",
                                               mime="text/csv",
                                               key="download_data")


# Feature Selection, Data Cleaning, Data Visualization Tabs
    fstab,dctab,dvtab,mtab=st.tabs(["Feature Selection","Data Cleaning","Data Visualization","Modeling"])

# Feature Selection Tab
    with fstab:
     number_of_rows=st.session_state.df.shape[0]
     number_of_columns=st.session_state.df.shape[1]
     st.write(f"Dataset contains {number_of_rows} rows and {number_of_columns} columns.")
     fs1,fs2=st.columns(2)
     with fs1:
      n_rows=st.number_input("Enter number of rows to display",min_value=1,max_value=number_of_rows,value=number_of_rows,step=1)
     with fs2:
      columns=st.multiselect("Select columns to display",st.session_state.df.columns.tolist(),default=st.session_state.df.columns.tolist())
     st.dataframe(st.session_state.df)
     if st.button("Apply Selection"):
      if len (st.session_state.df)!=n_rows:
       st.session_state.df=st.session_state.df[:(n_rows)][columns]
      else:
       st.session_state.df=st.session_state.df[columns]
      st.rerun()
      st.success("Feature selection applied!")
     

    with dctab:
     natab,duplicatstab,outliertab=st.tabs(["NA","Duplicsts","outliers"])
     with natab:
        na_counts=st.session_state.df.isna().sum()
        na_counts=na_counts[na_counts>0]
        if not na_counts.empty:
         st.write("Columns with NA values:")
         st.write(na_counts)
         na_col=st.selectbox("Select column to handle NA values",na_counts.index.tolist())
         na_option=st.selectbox("Select option to handle NA values",["Drop rows with NA","Fill NA with mean","Fill NA with median","Fill NA with mode"])
         if st.button("Handle NA"):
            # Drop NA option
            if na_option=="Drop rows with NA":
             st.session_state.df=st.session_state.df.dropna(subset=[na_col])
             st.rerun()
             st.dataframe(st.session_state.df)
             st.success(f"Rows with NA in {na_col} dropped")

            # Fill NA with mean option
            elif na_option=="Fill NA with mean":
             st.session_state.df[na_col]=st.session_state.df[na_col].fillna(st.session_state.df[na_col].mean())
             st.success(f"NA values filled with mean : {st.session_state.df[na_col].mean()}")
             st.rerun()
             st.dataframe(st.session_state.df)
             st.success(f"NA values in {na_col} handled using {na_option}={st.session_state.df[na_col].mean()}")

            # Fill NA with median option
            elif na_option=="Fill NA with median":
             st.session_state.df[na_col]=st.session_state.df[na_col].fillna(st.session_state.df[na_col].median())
             st.success(f"NA values filled with median : {st.session_state.df[na_col].median()}")
             st.rerun()
             st.dataframe(st.session_state.df)
             st.success(f"NA values in {na_col} handled using {na_option}={st.session_state.df[na_col].median()}")

            # Fill NA with mode option
            elif na_option=="Fill NA with mode":
             st.session_state.df[na_col]=st.session_state.df[na_col].fillna(st.session_state.df[na_col].mode()[0])
             st.success(f"NA values filled with mode : {st.session_state.df[na_col].mode()[0]}")
             st.dataframe(st.session_state.df)
             st.rerun()
             st.dataframe(st.session_state.df)
             st.success(f"NA values in {na_col} handled using {na_option}={st.session_state.df[na_col].mode()[0]}")

        else:
         st.write("No NA values found in the dataset.")

     with duplicatstab:
        dup_count=st.session_state.df.duplicated().sum()
        st.write(f"Number of duplicate rows: {dup_count}")
        if dup_count>0:
         if st.button("Remove Duplicates"):
            st.session_state.df=st.session_state.df.drop_duplicates()
            st.dataframe(st.session_state.df)
            st.success("Duplicate rows removed")
            st.rerun()

     with outliertab:
        numerical_columns=st.session_state.df.select_dtypes(include=np.number).columns.tolist()
        if numerical_columns:
         outlier_col=st.selectbox("Select column to check for outliers",numerical_columns)
         q1=st.session_state.df[outlier_col].quantile(0.25)
         q3=st.session_state.df[outlier_col].quantile(0.75)
         iqr=q3-q1
         lower_bound=q1-1.5*iqr
         upper_bound=q3+1.5*iqr
         outliers=st.session_state.df[(st.session_state.df[outlier_col]<lower_bound) | (st.session_state.df[outlier_col]>upper_bound)]
         st.write(f"Number of outliers in {outlier_col}: {len(outliers)}")
         fig_box=px.box(st.session_state.df,y=outlier_col,title=f"Box plot of {outlier_col} to visualize outliers")
         st.plotly_chart(fig_box)
         if not outliers.empty:
          st.write(outliers)
          if st.button("Remove Outliers"):
             st.session_state.df=st.session_state.df[(st.session_state.df[outlier_col]>=lower_bound) & (st.session_state.df[outlier_col]<=upper_bound)]
             st.dataframe(st.session_state.df)
             st.success(f"Outliers in {outlier_col} removed")
             st.rerun()
        else:
         st.write("No numerical columns available to check for outliers.") 
 
     

    with dvtab:
       numerical_columns=st.session_state.df.select_dtypes(include=np.number).columns.tolist()
       all_columns=st.session_state.df.columns.tolist()

       tap1,tap2,tap3,tap4,tap5=st.tabs(["Scatter Plot","Histogram","Box Plot","Line Plot","Bar Plot"])


       with tap1:
        col1,col2,col3=st.columns(3)
        with st.spinner('Generating plot...'):
          with col1:
           x_axis=st.selectbox("Select X-axis", numerical_columns)
          with col2:
           y_axis=st.selectbox("Select Y-axis",numerical_columns)
          with col3:
           color=st.selectbox("Select column to be a color",numerical_columns)
          figer_scatter=px.scatter(st.session_state.df,x=x_axis,y=y_axis,color=color,title=f"Scatter plot of {y_axis} vs {x_axis} colored by {color}")
          st.success('Plot generated!')
          st.plotly_chart(figer_scatter)
    
    
       with tap2:
        hist_col1, hist_col2=st.columns(2)
        with st.spinner('Generating plot...'):
         with hist_col1:
          hist_feature=st.selectbox("Select feature for histogram",numerical_columns)
         with hist_col2:
          hist_color=st.selectbox("Select column to be a color for histogram",numerical_columns)
         fig_hist=px.histogram(st.session_state.df,x=hist_feature,color=hist_color,title=f"Histogram of {hist_feature} colored by {hist_color}")
         st.success('Plot generated!')
         st.plotly_chart(fig_hist)

       with tap3:
        box_col1, box_col2=st.columns(2)
        with st.spinner('Generating plot...'):
         with box_col1:
          box_y=st.selectbox("Select Y-axis for box plot",numerical_columns)
         with box_col2:
          box_color=st.selectbox("Select column to be a color  for box plot",numerical_columns)
         fig_box=px.box(st.session_state.df,y=box_y,color=box_color,title=f"Box plot of {box_y} colored by {box_color}")
         st.success('Plot generated!')
         st.plotly_chart(fig_box)

       with tap4:
        line_col1, line_col2=st.columns(2)
        with st.spinner('Generating plot...'):
         with line_col1:
          line_x=st.selectbox("Select X-axis for line plot",numerical_columns)
         with line_col2:
          line_y=st.selectbox("Select Y-axis for line plot",numerical_columns)
         fig_line=px.line(st.session_state.df,x=line_x,y=line_y,title=f"Line plot of {line_y} vs {line_x}")
         st.success('Plot generated!')
         st.plotly_chart(fig_line)

       with tap5:
        bar_col1, bar_col2=st.columns(2)
        with st.spinner('Generating plot...'):
         with bar_col1:
          bar_x=st.selectbox("Select X-axis for bar plot",all_columns)
         with bar_col2:
          bar_y=st.selectbox("Select Y-axis for bar plot",all_columns)
         fig_bar=px.bar(st.session_state.df,x=bar_x,y=bar_y,title=f"Bar plot of {bar_y} vs {bar_x}")
         st.success('Plot generated!')
         st.plotly_chart(fig_bar)
    
    with mtab:
      
     st.header("Machine Learning Modeling")

     from sklearn.model_selection import train_test_split
     from sklearn.linear_model import LogisticRegression
     from sklearn.ensemble import RandomForestClassifier
     from sklearn.ensemble import RandomForestRegressor
     from sklearn.svm import SVC
     from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
     import seaborn as sns
     import matplotlib.pyplot as plt
     import joblib

     st.header("Correlation Map (for Feature Selection)")
     mcol1,mcol2=st.columns(2)
     with mcol1:
        numeric_df=st.session_state.df.select_dtypes(include=["int64", "float64"])
        if numeric_df.shape[1] < 2:
            st.warning("⚠️ Not enough numeric columns to show correlation.")
        else:
            corr_matrix = numeric_df.corr()

            fig, ax = plt.subplots(figsize=(5,3))
            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                linewidths=0.5,
                ax=ax
            )

            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)

     df_model =st.session_state.df.copy()
     ml1col1, ml2col2 = st.columns(2)
     with ml1col1:
        #Target Selection  
        target = st.selectbox("Select Target Column", df_model.columns)

    # Feature and Target Split
     X = df_model.drop(columns=[target])
     y = df_model[target]

     X = pd.get_dummies(X)

    #Data split
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

     with ml2col2:
        #select Model
        model_name = st.selectbox("Select Model",["Logistic Regression", "Random Forest","Random Forest Regression", "Support Vector Machine"])

        if model_name == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)

        elif model_name == "Random Forest":
                model = RandomForestClassifier()
        elif model_name == "Random Forest Regression":
                model = RandomForestRegressor()

        elif model_name == "Support Vector Machine":
                model = SVC()

     if st.button("Train Model"):
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)


            if model_name=="Random Forest Regression":
             from sklearn.metrics import mean_squared_error, r2_score
             mse = mean_squared_error(y_test, y_pred)
             r2 = r2_score(y_test, y_pred)
             st.write(f"Mean Squared Error: {mse:.2f}")
             st.write(f"R² Score: {r2:.2f}")
             m111col1, m122col2 = st.columns(2)
             with m111col1:
                fig, ax = plt.subplots(figsize=(6,4))
                ax.scatter(y_test, y_pred, alpha=0.7)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # خط المثالي
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title("Regression Predictions vs Actuals")
                st.pyplot(fig)
            else:
                accuracy = accuracy_score(y_test, y_pred) * 100  
                st.write(f"Accuracy: {accuracy:.2f}%")
                #  Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                m11col1, m12col2 = st.columns(2)
                with m11col1:
                    fig, ax = plt.subplots(figsize=(6,4))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    ax.set_title("Confusion Matrix")
                    st.pyplot(fig)
            
            


            joblib.dump(model, "trained_model.joblib")

            with open("trained_model.joblib", "rb") as file:
                st.download_button(
                    label="⬇️ Download Trained Model as trained_model.joblib",
                    data=file,
                    file_name="trained_model.joblib",
                    mime="application/octet-stream"
                )





        

    

    

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

   