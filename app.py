import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
import json 

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Mineral Deposit Analysis & Classifier",
    page_icon="🌍",
    layout="wide",
)

# --- 2. Caching Data and Model Loading ---
@st.cache_data
def load_data():
    df = pd.read_csv('master_mineral_data.csv')
    df.drop(columns=['Total'], inplace=True, errors='ignore')
    return df

@st.cache_resource
def load_prediction_assets():
    model = joblib.load('mineral_deposit_classifier_sklearn.pkl')
    encoder = joblib.load('label_encoder.pkl')
    features = joblib.load('model_features.pkl')
    with open('model_performance.json', 'r') as f:
        performance = json.load(f)
    return model, encoder, features, performance

master_df = load_data()
model_pipeline, label_encoder, model_features, model_performance = load_prediction_assets()

# --- 3. Sidebar ---
st.sidebar.title("Dashboard Controls")
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Deposit Comparison", "Predict Deposit Type"])
st.sidebar.header("Global Filters")
deposit_types = ['All'] + sorted(master_df['Deposit_Type'].unique().tolist())
selected_deposit = st.sidebar.selectbox("Select Deposit Type", deposit_types)
if selected_deposit == 'All':
    filtered_df = master_df
else:
    filtered_df = master_df[master_df['Deposit_Type'] == selected_deposit]

# --- 4. Main Page Content ---

if page == "Home":
    st.title("Geospatial Mineral Deposit Analysis & Classifier")
    st.markdown("An interactive dashboard for analyzing geochemical signatures and predicting mineral deposit types with machine learning.")
    st.header("Project Overview")
    st.write("""
    - **Data Analysis & Comparison:** Explore the geochemical signatures of known deposits.
    - **Predict Deposit Type:** Upload a CSV file with new sample data to get batch predictions from our trained model.
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.header("Dataset at a Glance")
        st.metric("Total Samples Analyzed", len(master_df))
        st.metric("Number of Deposit Types", master_df['Deposit_Type'].nunique())
        st.metric("Total Elements Measured", len(master_df.select_dtypes(include=np.number).columns))
        st.write("Sample Distribution by Type:")
        deposit_counts = master_df['Deposit_Type'].value_counts()
        st.bar_chart(deposit_counts)
    with col2:
        st.header("Predictive Model Performance")
        st.markdown("The classification model is a **Random Forest Classifier** trained and tuned using Scikit-learn. Here are its key performance metrics:")
        
        cv_accuracy = model_performance.get("cross_validation_accuracy", "N/A")
        train_accuracy = model_performance.get("full_training_set_accuracy", "N/A")
        
        st.metric(
            label="Cross-Validation Accuracy",
            value=f"{cv_accuracy:.4f}",
            help="The average accuracy of the model on unseen data folds during training. This is the most realistic measure of performance."
        )
        
        st.metric(
            label="Training Set Accuracy",
            value=f"{train_accuracy:.4f}",
            help="The accuracy of the model on the data it was trained on. A large gap between this and CV Accuracy can indicate overfitting."
        )

        st.subheader("Best Model Parameters")
        st.markdown("The following parameters were found to be optimal through a cross-validated grid search, balancing accuracy and preventing overfitting:")
        
        best_params = model_performance.get("best_parameters", {})
        for param, value in best_params.items():
            clean_param = param.replace('classifier__', '').replace('_', ' ').title()
            st.text(f"- {clean_param}: {value}")
elif page == "Data Analysis":
    st.title("🔬 Single Deposit Analysis")
    if selected_deposit == 'All':
        st.warning("Please select a single deposit type from the sidebar to perform analysis.")
    else:
        st.header(f"Analyzing: {selected_deposit} Deposits")
        element_list = sorted(filtered_df.select_dtypes(include='number').columns.tolist())
        analysis_option = ["Overall Composition", "Correlation Heatmap"] + element_list
        selected_option = st.selectbox("Select an Analysis to View", analysis_option)

        if selected_option == "Overall Composition":
            st.subheader("Overall Elemental Composition (Mean Values)")
            composition_data = filtered_df[element_list].mean().reset_index()
            composition_data.columns = ['Element', 'Mean_Concentration']
            composition_data = composition_data[composition_data['Mean_Concentration'] > 0]
            fig_pie = px.pie(composition_data, names='Element', values='Mean_Concentration', title=f"Average Elemental Composition of {selected_deposit} Deposits", hole=0.3)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

        elif selected_option == "Correlation Heatmap":
            st.subheader(f"Element Correlation Matrix for {selected_deposit} Deposits")
            numeric_df = filtered_df.select_dtypes(include=np.number)
            present_elements_df = numeric_df.dropna(axis='columns', how='all')
            if present_elements_df.empty or present_elements_df.shape[1] < 2:
                st.warning("Not enough element data present for this deposit type to generate a correlation matrix.")
            else:
                corr_matrix = present_elements_df.corr()
                fig_heatmap = px.imshow(corr_matrix, text_auto=False, aspect="auto", color_continuous_scale='RdBu_r', title="Interactive Heatmap of Element Correlations")
                st.plotly_chart(fig_heatmap, use_container_width=True)
                st.markdown("This heatmap shows the correlation between pairs of elements. Red indicates a positive correlation, while blue indicates a negative correlation.")

        else:
            st.subheader(f"Distribution of {selected_option}")
            col1, col2 = st.columns(2)
            plot_data = filtered_df.dropna(subset=[selected_option])
            with col1:
                fig_hist = px.histogram(plot_data, x=selected_option, title=f"Histogram of {selected_option}", marginal="box")
                st.plotly_chart(fig_hist)
            with col2:
                fig_box = px.box(plot_data, y=selected_option, title=f"Box Plot of {selected_option}")
                st.plotly_chart(fig_box)
            st.subheader("Statistical Summary")
            st.dataframe(plot_data[[selected_option, 'Deposit_Type']].describe())

elif page == "Deposit Comparison":
    st.title("Comparing Element Distributions Across Deposits")
    element_list = sorted(master_df.select_dtypes(include='number').columns.tolist())
    selected_element_comp = st.selectbox("Select an Element to Compare", element_list, key="comp_element")
    if selected_element_comp:
        st.subheader(f"Comparison of {selected_element_comp} Across All Deposits")
        comp_data = master_df.dropna(subset=[selected_element_comp])
        fig_comp_box = px.box(comp_data, x='Deposit_Type', y=selected_element_comp, title=f"Distribution of {selected_element_comp} by Deposit Type", color='Deposit_Type')
        st.plotly_chart(fig_comp_box, use_container_width=True)
        fig_comp_violin = px.violin(comp_data, x='Deposit_Type', y=selected_element_comp, title=f"Violin Plot of {selected_element_comp} by Deposit Type", color='Deposit_Type')
        st.plotly_chart(fig_comp_violin, use_container_width=True)

elif page == "Predict Deposit Type":
    st.title("Predict Deposit Type via CSV Upload")
    st.markdown("Upload a CSV file with new sample data. The model will predict the deposit type for each sample.")
    st.info(f"The model expects the following columns: **{', '.join(model_features)}**. Missing columns in your upload will be automatically treated as zero (0).")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.subheader("Uploaded Data Preview")
            st.dataframe(uploaded_df.head())

            if st.button("Run Predictions on Uploaded File"):
                with st.spinner("Analyzing samples"):
                    prediction_template = pd.DataFrame(columns=model_features)
                    input_df = pd.concat([prediction_template, uploaded_df], ignore_index=True)
                    input_df_for_model = input_df[model_features]
                    input_df_for_model.fillna(0, inplace=True)
                    predictions_numeric = model_pipeline.predict(input_df_for_model)
                    predictions_text = label_encoder.inverse_transform(predictions_numeric)
                    prediction_probas = model_pipeline.predict_proba(input_df_for_model)
                    
                    results_df = uploaded_df.copy()
                    results_df['Predicted_Deposit_Type'] = predictions_text
                    results_df['Confidence_Score'] = np.max(prediction_probas, axis=1)

                    st.success("Prediction complete!")
                    st.subheader("Prediction Results")
                    st.dataframe(results_df)

                    @st.cache_data
                    def convert_df_to_csv(df):
                        return df.to_csv(index=False).encode('utf-8')

                    csv_output = convert_df_to_csv(results_df)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv_output,
                        file_name='prediction_results.csv',
                        mime='text/csv',
                    )
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")