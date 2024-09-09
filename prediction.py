import streamlit as st
import pandas as pd
import plotly.express as px
from xgboost import XGBRegressor
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the model
loaded_model = joblib.load('xgboost_model.pkl')

# Streamlit app title
st.title("Dynamic Visualization App with Datasets")

# Upload dataset
uploaded_file = st.file_uploader("Upload Dataset (CSV or Excel)", type=["csv", "xlsx"])

# Check if a file is uploaded
if uploaded_file is not None:
    # Read the uploaded file into a DataFrame
    selected_df = pd.read_excel(uploaded_file, engine='openpyxl')  # Use engine='openpyxl' for Excel files

    # Streamlit app title
    st.title("Dynamic Visualization App with Uploaded Dataset")

    # Select dataset source
    dataset_option = st.selectbox("Select Dataset Source:", ["Uploaded Dataset", "Hardcoded Dataset"])

    if dataset_option == "Uploaded Dataset":
        # Select visualization type
        visualization_type = st.selectbox("Select Visualization Type:", ["Bar Chart", "Pie Chart", "Line Chart"], key="uploaded_vis_type")

        # Check if a visualization type is selected
        if visualization_type:
            if visualization_type == "Bar Chart":
                # Select columns for X and Y axes
                x_column = st.selectbox("Select X-axis Data:", selected_df.columns, key="uploaded_x_column")
                y_column = st.selectbox("Select Y-axis Data:", selected_df.columns, key="uploaded_y_column")
                # Create bar chart
                fig = px.bar(selected_df, x=x_column, y=y_column, title="Bar Chart (Uploaded Dataset)")
                st.plotly_chart(fig)

            elif visualization_type == "Pie Chart":
                # Select column for pie chart
                pie_column = st.selectbox("Select Data for Pie Chart:", selected_df.columns, key="uploaded_pie_column")
                # Create pie chart
                fig = px.pie(selected_df, names=pie_column, title="Pie Chart (Uploaded Dataset)")
                st.plotly_chart(fig)

            elif visualization_type == "Line Chart":
                # Select columns for X and Y axes
                x_column = st.selectbox("Select X-axis Data:", selected_df.columns, key="uploaded_x_column")
                y_column = st.selectbox("Select Y-axis Data:", selected_df.columns, key="uploaded_y_column")
                # Create line chart
                fig = px.line(selected_df, x=x_column, y=y_column, title="Line Chart (Uploaded Dataset)")
                st.plotly_chart(fig)

    elif dataset_option == "Hardcoded Dataset":
        # Select visualization type
        visualization_type = st.selectbox("Select Visualization Type:", ["Bar Chart", "Pie Chart", "Line Chart"], key="hardcoded_vis_type")

        # Check if a visualization type is selected
        if visualization_type:
            if visualization_type == "Bar Chart":
                # Select columns for X and Y axes
                x_column = st.selectbox("Select X-axis Data:", selected_df.columns, key="uploaded_x_column")
                y_column = st.selectbox("Select Y-axis Data:", selected_df.columns, key="uploaded_y_column")
                # Create bar chart
                fig = px.bar(selected_df, x=x_column, y=y_column, title="Bar Chart (Uploaded Dataset)")
                st.plotly_chart(fig)

            elif visualization_type == "Pie Chart":
                # Select column for pie chart
                pie_column = st.selectbox("Select Data for Pie Chart:", selected_df.columns, key="uploaded_pie_column")
                # Create pie chart
                fig = px.pie(selected_df, names=pie_column, title="Pie Chart (Uploaded Dataset)")
                st.plotly_chart(fig)

            elif visualization_type == "Line Chart":
                # Select columns for X and Y axes
                x_column = st.selectbox("Select X-axis Data:", selected_df.columns, key="uploaded_x_column")
                y_column = st.selectbox("Select Y-axis Data:", selected_df.columns, key="uploaded_y_column")
                # Create line chart
                fig = px.line(selected_df, x=x_column, y=y_column, title="Line Chart (Uploaded Dataset)")
                st.plotly_chart(fig)
    selected_col1 = st.selectbox("Select a feature from Reporting Economy:", selected_df['Reporting Economy'].unique())
    selected_col2 = st.selectbox("Select a feature from Product Sector:", selected_df['Product/Sector'].unique())
    # Label Encoding
    label_encoder = LabelEncoder()
    selected_col1_encoded = label_encoder.fit_transform(selected_df['Reporting Economy'])[selected_df['Reporting Economy'] == selected_col1][0]
    selected_col2_encoded = label_encoder.fit_transform(selected_df['Product/Sector'])[selected_df['Product/Sector'] == selected_col2][0]
    if st.button("Predict"):
        try:
            # Make predictions using the loaded XGBoost model
            predictions = loaded_model.predict([[selected_col1_encoded, selected_col2_encoded, 2021]])

            # Display the predictions
            st.write("Predicted Value:", predictions[0])
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    st.info("Please make sure the selected features match the format expected by the model.")