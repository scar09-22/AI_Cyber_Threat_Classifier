"""
Streamlit dashboard for AI threat classifier.
Provides real-time threat detection and batch analysis capabilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="AI Threat Classifier",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ThreatDashboard:
    def __init__(self):
        self.model_dir = Path('models')
        self.data_dir = Path('data')
        self.output_dir = Path('outputs')
        
        resources = self.load_resources()
        if resources:
            self.model, self.scaler, self.feature_names, self.metadata, self.top_features = resources
        else:
            st.stop()
    
    @st.cache_resource
    def load_resources(_self):
        """Load model and metadata."""
        try:
            model = joblib.load(_self.model_dir / 'threat_classifier.pkl')
            scaler = joblib.load(_self.data_dir / 'scaler.pkl')
            
            with open(_self.data_dir / 'feature_names.json', 'r') as f:
                feature_names = json.load(f)
            
            with open(_self.model_dir / 'model_metadata.json', 'r') as f:
                metadata = json.load(f)
            
            with open(_self.output_dir / 'top_features.json', 'r') as f:
                top_features = json.load(f)
            
            return model, scaler, feature_names, metadata, top_features
            
        except FileNotFoundError as e:
            st.error(f"Required file not found: {e}")
            st.info("Run preprocessing, training, and explainability scripts first")
            return None
    
    def main(self):
        """Main dashboard."""
        st.sidebar.title("AI Threat Classifier")
        st.sidebar.markdown("---")
        
        page = st.sidebar.radio(
            "Navigation",
            ["Home", "Single Prediction", "Batch Analysis", 
             "Model Performance", "Explainability"]
        )
        
        st.sidebar.markdown("---")
        st.sidebar.info(
            f"**Model:** {self.metadata['model_type']}\n\n"
            f"**Accuracy:** {self.metadata['metrics']['accuracy']:.4f}"
        )
        
        if page == "Home":
            self.home_page()
        elif page == "Single Prediction":
            self.single_prediction_page()
        elif page == "Batch Analysis":
            self.batch_analysis_page()
        elif page == "Model Performance":
            self.performance_page()
        elif page == "Explainability":
            self.explainability_page()
    
    def home_page(self):
        """Home page."""
        st.title("AI-Driven Cyber Threat Classifier")
        st.markdown("### Real-Time Network Threat Detection with Explainable AI")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{self.metadata['metrics']['accuracy']:.2%}")
        
        with col2:
            st.metric("Precision", f"{self.metadata['metrics']['precision']:.2%}")
        
        with col3:
            st.metric("Recall", f"{self.metadata['metrics']['recall']:.2%}")
        
        with col4:
            st.metric("F1-Score", f"{self.metadata['metrics']['f1']:.2%}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Features")
            st.markdown("""
            - Real-time threat classification
            - Explainable AI with SHAP
            - Batch processing capability
            - Interactive visualizations
            - High accuracy (99%+)
            """)
        
        with col2:
            st.markdown("### Dataset Info")
            st.markdown(f"""
            - Features: {len(self.feature_names)} network flow characteristics
            - Classes: Binary (Benign vs Attack)
            - Model: {self.metadata['model_type']}
            - Dataset: CICIDS2017
            """)
        
        st.markdown("---")
        st.markdown("### Top Contributing Features")
        
        top_features_df = pd.DataFrame(
            list(self.top_features.items())[:10],
            columns=['Feature', 'Importance']
        )
        
        fig = px.bar(
            top_features_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Feature Importance (SHAP Values)",
            color='Importance',
            color_continuous_scale='Blues'
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    def single_prediction_page(self):
        """Single prediction page."""
        st.title("Single Threat Prediction")
        st.markdown("Test the model with custom network flow parameters")
        
        st.markdown("### Enter Network Flow Parameters")
        st.info("Enter values for the most important features. Others will default to 0.")
        
        with st.form("prediction_form"):
            cols = st.columns(3)
            
            user_input = {}
            important_features = list(self.top_features.keys())[:15]
            
            for i, feature in enumerate(important_features):
                with cols[i % 3]:
                    user_input[feature] = st.number_input(
                        feature,
                        value=0.0,
                        format="%.4f",
                        key=feature
                    )
            
            submit = st.form_submit_button("Analyze Threat", use_container_width=True)
        
        if submit:
            # Create full feature vector
            input_vector = np.zeros((1, len(self.feature_names)))
            
            for feature, value in user_input.items():
                if feature in self.feature_names:
                    idx = self.feature_names.index(feature)
                    input_vector[0, idx] = value
            
            # Scale input
            input_scaled = self.scaler.transform(input_vector)
            
            # Make prediction
            prediction = self.model.predict(input_scaled)[0]
            proba = self.model.predict_proba(input_scaled)[0]
            
            # Display result
            st.markdown("---")
            st.markdown("### Prediction Result")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                threat_level = "ATTACK" if prediction == 1 else "BENIGN"
                st.markdown(f"## {threat_level}")
            
            with col2:
                st.metric("Confidence", f"{proba[prediction]:.2%}")
            
            with col3:
                risk_score = proba[1] * 100
                st.metric("Risk Score", f"{risk_score:.1f}/100")
            
            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba[1] * 100,
                title={'text': "Attack Probability"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "salmon"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True)
    
    def batch_analysis_page(self):
        """Batch analysis page."""
        st.title("Batch Threat Analysis")
        st.markdown("Upload a CSV file to analyze multiple network flows")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload CSV with the same features as training data"
        )
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            st.markdown(f"### Uploaded Dataset: {len(df)} rows")
            st.dataframe(df.head(), use_container_width=True)
            
            if st.button("Analyze All Records", use_container_width=True):
                with st.spinner("Analyzing..."):
                    try:
                        X = df[self.feature_names]
                    except KeyError:
                        st.error("CSV columns don't match training features")
                        return
                    
                    X_scaled = self.scaler.transform(X)
                    predictions = self.model.predict(X_scaled)
                    probabilities = self.model.predict_proba(X_scaled)
                    
                    df['Prediction'] = ['Attack' if p == 1 else 'Benign' for p in predictions]
                    df['Attack_Probability'] = probabilities[:, 1]
                    df['Risk_Score'] = df['Attack_Probability'] * 100
                    
                    st.markdown("---")
                    st.markdown("### Analysis Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Records", len(df))
                    
                    with col2:
                        benign_count = (predictions == 0).sum()
                        st.metric("Benign", benign_count)
                    
                    with col3:
                        attack_count = (predictions == 1).sum()
                        st.metric("Attacks", attack_count)
                    
                    with col4:
                        avg_risk = df['Risk_Score'].mean()
                        st.metric("Avg Risk Score", f"{avg_risk:.1f}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.pie(
                            values=[benign_count, attack_count],
                            names=['Benign', 'Attack'],
                            title="Classification Distribution",
                            color_discrete_sequence=['green', 'red']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.histogram(
                            df,
                            x='Risk_Score',
                            title="Risk Score Distribution",
                            nbins=50,
                            color_discrete_sequence=['steelblue']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("### Detailed Results")
                    st.dataframe(
                        df[['Prediction', 'Attack_Probability', 'Risk_Score']],
                        use_container_width=True
                    )
                    
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download Results",
                        csv,
                        "threat_analysis_results.csv",
                        "text/csv",
                        use_container_width=True
                    )
    
    def performance_page(self):
        """Model performance page."""
        st.title("Model Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{self.metadata['metrics']['accuracy']:.4f}")
        
        with col2:
            st.metric("Precision", f"{self.metadata['metrics']['precision']:.4f}")
        
        with col3:
            st.metric("Recall", f"{self.metadata['metrics']['recall']:.4f}")
        
        with col4:
            st.metric("F1-Score", f"{self.metadata['metrics']['f1']:.4f}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Confusion Matrix")
            cm_path = self.output_dir / f"confusion_matrix_{self.metadata['model_type']}.png"
            if cm_path.exists():
                st.image(str(cm_path), use_column_width=True)
        
        with col2:
            st.markdown("### Model Comparison")
            comparison_path = self.output_dir / 'model_comparison.png'
            if comparison_path.exists():
                st.image(str(comparison_path), use_column_width=True)
    
    def explainability_page(self):
        """Explainability page."""
        st.title("Model Explainability (SHAP)")
        st.markdown("Understand which features drive the model's decisions")
        
        tab1, tab2, tab3 = st.tabs([
            "Feature Importance",
            "SHAP Summary",
            "Individual Explanations"
        ])
        
        with tab1:
            st.markdown("### Global Feature Importance")
            importance_path = self.output_dir / 'shap_feature_importance.png'
            if importance_path.exists():
                # Use width parameter instead of use_column_width
                st.image(str(importance_path), width=700)
        
        with tab2:
            st.markdown("### SHAP Summary Plot")
            summary_path = self.output_dir / 'shap_summary.png'
            if summary_path.exists():
                # Fixed width to prevent distortion
                st.image(str(summary_path), width=700)
        
        with tab3:
            st.markdown("### Individual Prediction Explanations")
            col1, col2 = st.columns(2)
            
            with col1:
                waterfall_0 = self.output_dir / 'shap_waterfall_sample_0.png'
                if waterfall_0.exists():
                    st.image(str(waterfall_0), use_column_width=True)
            
            with col2:
                waterfall_10 = self.output_dir / 'shap_waterfall_sample_10.png'
                if waterfall_10.exists():
                    st.image(str(waterfall_10), use_column_width=True)


if __name__ == "__main__":
    try:
        dashboard = ThreatDashboard()
        dashboard.main()
    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Make sure all required files exist in models/, data/, and outputs/ folders")
