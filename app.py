import streamlit as st
import pandas as pd
import plotly.express as px
from pycaret.classification import load_model, predict_model

# --- Page Configuration ---
st.set_page_config(
    page_title="Geospatial Mineral Analysis Dashboard",
    page_icon="🌍",
    layout="wide",
)

# --- Caching Data and Model Loading ---
@st.cache_data
def load_data():
    df = pd.read_csv('master_mineral_data.csv')
    df.drop(columns=['Total'], inplace=True, errors='ignore')
    return df

@st.cache_resource
def load_classification_model():
    pipeline = load_model('mineral_deposit_classifier')
    return pipeline

master_df = load_data()
model_pipeline = load_classification_model()

# --- Sidebar ---
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

# --- Main Page Content ---

if page == "Home":
    st.title("🌍 Geospatial Mineral Deposits Analysis")
    st.markdown("Welcome to the interactive dashboard for analyzing and predicting mineral deposit geochemistry.")
    st.header("Project Overview")
    st.write("""
    This application allows for the exploration, comparison, and classification of trace element data.
    - **Data Analysis & Comparison:** Explore the geochemical signatures of known deposits.
    - **Predict Deposit Type:** Use our trained Machine Learning model to classify a new sample.
    """)
    st.header("Dataset at a Glance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Samples", len(master_df))
        st.metric("Number of Deposit Types", master_df['Deposit_Type'].nunique())
        st.metric("Total Elements Measured", len(master_df.columns) - 1)
    with col2:
        st.write("Sample Distribution:")
        deposit_counts = master_df['Deposit_Type'].value_counts()
        st.bar_chart(deposit_counts)

elif page == "Data Analysis":
    st.title("🔬 Single Deposit and Element Analysis")
    
    if selected_deposit == 'All':
        st.warning("Please select a single deposit type from the sidebar to analyze element distributions.")
    else:
        st.header(f"Analyzing: {selected_deposit} Deposits")

        element_list = sorted(filtered_df.select_dtypes(include='number').columns.tolist())
        analysis_option = ["Overall Composition"] + element_list
        selected_option = st.selectbox("Select an Analysis to View", analysis_option)

        if selected_option == "Overall Composition":
            st.subheader("Overall Elemental Composition (Mean Values)")

            # *** THIS IS THE FIX ***
            # Exclude the non-element 'Total' column before calculating the mean
            elements_for_pie = [el for el in element_list if el != 'Total']
            
            composition_data = filtered_df[elements_for_pie].mean().reset_index()
            composition_data.columns = ['Element', 'Mean_Concentration']
            
            composition_data = composition_data[composition_data['Mean_Concentration'] > 0]

            fig_pie = px.pie(
                composition_data,
                names='Element',
                values='Mean_Concentration',
                title=f"Average Elemental Composition of {selected_deposit} Deposits",
                hole=0.3
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
            
            st.markdown("This chart shows the percentage contribution of each element's average concentration to the total average concentration of all measured elements for this deposit type.")
        
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

            st.subheader("Data Summary")
            st.dataframe(plot_data[[selected_option, 'Deposit_Type']].describe())

elif page == "Deposit Comparison":
    st.title("📊 Comparing Multiple Deposits")
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
    st.title("🤖 Predict Deposit Type with Machine Learning")
    st.markdown("Enter the trace element values for a new sample below. The model will predict its deposit type.")

    final_estimator = model_pipeline.steps[-1][1]
    model_features = list(final_estimator.feature_names_in_)
    
    input_data = {}
    st.header("Enter Geochemical Data (ppm)")
    
    cols = st.columns(4)
    
    for i, feature in enumerate(model_features):
        default_value = master_df[feature].median()
        input_data[feature] = cols[i % 4].number_input(
            label=feature, 
            value=float(default_value),
            format="%.4f",
            key=f"input_{feature}"
        )

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        
        prediction = predict_model(model_pipeline, data=input_df)
        
        predicted_type = prediction['prediction_label'].iloc[0]
        confidence_score = prediction['prediction_score'].iloc[0]
        
        st.success(f"**Predicted Deposit Type:** {predicted_type}")
        st.info(f"**Confidence Score:** {confidence_score:.2%}")
        
        st.write("---")
        st.subheader("Full Prediction Output:")
        st.dataframe(prediction)