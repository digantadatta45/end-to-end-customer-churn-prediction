"""
Customer Churn Prediction - Combined Single File App
Everything in one file - training happens on first run
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report)
import warnings
import os

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {padding: 0rem 1rem;}
    h1 {color: #1f77b4; padding-bottom: 1rem;}
    h2 {color: #2c3e50; padding-top: 1rem;}
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_or_train_models():
    """Load models if they exist, otherwise train them"""
    
    # Check if models already exist
    if os.path.exists('models_cache.pkl'):
        with open('models_cache.pkl', 'rb') as f:
            cache = pickle.load(f)
        return (cache['models'], cache['results'], cache['scaler'], 
                cache['label_encoders'], cache['target_encoder'], 
                cache['feature_names'], cache['df_raw'], 
                cache['df_cleaned'], cache['df_preprocessed'])
    
    # If not, train the models
    with st.spinner('🔄 Training models for the first time... This will take 2-3 minutes'):
        
        # Load dataset
        url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
        df = pd.read_csv(url)
        df_raw = df.copy()
        
        # Data Cleaning
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
        
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
        
        df_cleaned = df.copy()
        
        # Preprocessing
        le_target = LabelEncoder()
        df['Churn'] = le_target.fit_transform(df['Churn'])
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        label_encoders = {}
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        
        df_preprocessed = df.copy()
        
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB()
        }
        
        results = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, (name, model) in enumerate(models.items()):
            status_text.text(f'Training {name}...')
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            results[name] = {
                'model': model,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            progress_bar.progress((idx + 1) / len(models))
        
        progress_bar.empty()
        status_text.empty()
        
        # Save everything
        cache = {
            'models': models,
            'results': results,
            'scaler': scaler,
            'label_encoders': label_encoders,
            'target_encoder': le_target,
            'feature_names': X.columns.tolist(),
            'df_raw': df_raw,
            'df_cleaned': df_cleaned,
            'df_preprocessed': df_preprocessed
        }
        
        with open('models_cache.pkl', 'wb') as f:
            pickle.dump(cache, f)
        
        st.success('✅ Models trained and cached successfully!')
        
        return (models, results, scaler, label_encoders, le_target, 
                X.columns.tolist(), df_raw, df_cleaned, df_preprocessed)

def main():
    """Main application"""
    
    # Load or train models
    (models, results, scaler, label_encoders, target_encoder, 
     feature_names, df_raw, df_cleaned, df_preprocessed) = load_or_train_models()
    
    # Sidebar
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["🏠 Home", "📊 Dataset Overview", "🧹 Data Cleaning", 
         "🔍 EDA", "🤖 Model Training", "📈 Model Evaluation", 
         "🎯 Prediction", "💡 Insights"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 About")
    st.sidebar.info(
        "This application predicts customer churn using machine learning. "
        "Models are trained once and cached for fast predictions."
    )
    
    # Route to pages
    if page == "🏠 Home":
        show_home()
    elif page == "📊 Dataset Overview":
        show_dataset_overview(df_raw)
    elif page == "🧹 Data Cleaning":
        show_data_cleaning(df_raw, df_cleaned)
    elif page == "🔍 EDA":
        show_eda(df_raw, df_preprocessed)
    elif page == "🤖 Model Training":
        show_model_training()
    elif page == "📈 Model Evaluation":
        show_model_evaluation(models, results)
    elif page == "🎯 Prediction":
        show_prediction(models, scaler, label_encoders, feature_names)
    elif page == "💡 Insights":
        show_insights(results)

def show_home():
    st.title("📊 Customer Churn Prediction System")
    st.markdown("### Professional ML Application for Business Intelligence")
    
    st.markdown("""
    <div style='background-color: #e8f4f8; padding: 2rem; border-radius: 10px; margin: 2rem 0;'>
        <h3 style='color: #1f77b4;'>🎯 What is Customer Churn?</h3>
        <p style='font-size: 1.1rem;'>
        Customer churn refers to when customers stop doing business with a company. 
        Predicting churn helps businesses retain customers by identifying at-risk individuals.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h4>📊 Dataset Analysis</h4>
            <p>Comprehensive data exploration and visualization</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h4>🤖 ML Models</h4>
            <p>7 different algorithms trained and compared</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <h4>🎯 Predictions</h4>
            <p>Real-time churn prediction for new customers</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.info("👈 Use the sidebar to navigate through different sections.")

def show_dataset_overview(df_raw):
    st.title("📊 Dataset Overview")
    
    st.markdown("### 📁 Telco Customer Churn Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{len(df_raw):,}")
    with col2:
        st.metric("Total Features", len(df_raw.columns))
    with col3:
        churn_count = df_raw['Churn'].value_counts().get('Yes', 0)
        st.metric("Churned Customers", f"{churn_count:,}")
    with col4:
        churn_rate = (churn_count / len(df_raw)) * 100
        st.metric("Churn Rate", f"{churn_rate:.2f}%")
    
    st.markdown("---")
    st.markdown("### 📋 Dataset Sample")
    st.dataframe(df_raw.head(20), width='stretch')
    
    if st.checkbox("Show full dataset"):
        st.dataframe(df_raw, width='stretch')
    
    st.markdown("---")
    st.markdown("### 📈 Statistical Summary")
    st.dataframe(df_raw.describe(), width='stretch')

def show_data_cleaning(df_raw, df_cleaned):
    st.title("🧹 Data Cleaning & Preprocessing")
    
    st.markdown("### 🔄 Cleaning Process")
    st.markdown("""
    1. **Removed Customer ID**: Non-predictive identifier
    2. **Handled TotalCharges**: Converted to numeric, filled missing with median
    3. **Categorical Imputation**: Missing values filled with mode
    4. **Numerical Imputation**: Missing values filled with median
    5. **Encoding**: Label encoding for categorical variables
    6. **Scaling**: Standard scaling for numerical features
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Raw Data")
        st.info(f"**Rows:** {df_raw.shape[0]:,}  \n**Columns:** {df_raw.shape[1]}")
        missing_raw = df_raw.isnull().sum().sum()
        st.error(f"Missing Values: {missing_raw}")
    
    with col2:
        st.markdown("### ✅ Cleaned Data")
        st.info(f"**Rows:** {df_cleaned.shape[0]:,}  \n**Columns:** {df_cleaned.shape[1]}")
        missing_cleaned = df_cleaned.isnull().sum().sum()
        st.success(f"Missing Values: {missing_cleaned}")
    
    st.markdown("---")
    st.markdown("### 📋 Cleaned Dataset Sample")
    st.dataframe(df_cleaned.head(20), width='stretch')

def show_eda(df_raw, df_preprocessed):
    st.title("🔍 Exploratory Data Analysis")
    
    st.markdown("### 📊 Churn Distribution")
    
    churn_counts = df_raw['Churn'].value_counts()
    
    fig = go.Figure(data=[
        go.Bar(
            x=churn_counts.index,
            y=churn_counts.values,
            text=churn_counts.values,
            textposition='auto',
            marker_color=['#2ecc71', '#e74c3c']
        )
    ])
    fig.update_layout(
        title="Customer Churn Distribution",
        xaxis_title="Churn Status",
        yaxis_title="Number of Customers",
        height=400
    )
    st.plotly_chart(fig, width='stretch')
    
    st.markdown("---")
    st.markdown("### 📈 Numerical Features Analysis")
    
    if 'tenure' in df_raw.columns and 'MonthlyCharges' in df_raw.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df_raw, x='tenure', color='Churn', 
                             title='Tenure Distribution',
                             color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'})
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            fig = px.histogram(df_raw, x='MonthlyCharges', color='Churn',
                             title='Monthly Charges Distribution',
                             color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'})
            st.plotly_chart(fig, width='stretch')
    
    st.markdown("---")
    st.markdown("### 📊 Categorical Features")
    
    if 'Contract' in df_raw.columns:
        fig = px.histogram(df_raw, x='Contract', color='Churn',
                         barmode='group',
                         title='Contract Type vs Churn',
                         color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'})
        st.plotly_chart(fig, width='stretch')

def show_model_training():
    st.title("🤖 Model Training")
    
    st.markdown("### 🎓 Machine Learning Models")
    
    models_info = {
        "Logistic Regression": "Linear model for binary classification",
        "Decision Tree": "Tree-based decision boundaries",
        "Random Forest": "Ensemble of decision trees",
        "Gradient Boosting": "Sequential error correction",
        "SVM": "Maximum margin classifier",
        "KNN": "Instance-based learning",
        "Naive Bayes": "Probabilistic classifier"
    }
    
    for model, desc in models_info.items():
        with st.expander(f"📌 {model}"):
            st.write(desc)
    
    st.markdown("---")
    st.success("✅ All models trained and cached!")

def show_model_evaluation(models, results):
    st.title("📈 Model Evaluation")
    
    st.markdown("### 🏆 Model Performance Comparison")
    
    comparison_data = []
    for name, result in results.items():
        comparison_data.append({
            'Model': name,
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1-Score': result['f1_score'],
            'ROC-AUC': result['roc_auc']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('ROC-AUC', ascending=False)
    
    st.dataframe(
        comparison_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']),
        width='stretch'
    )
    
    st.markdown("---")
    
    fig = go.Figure()
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    for metric in metrics:
        fig.add_trace(go.Bar(
            name=metric,
            x=comparison_df['Model'],
            y=comparison_df[metric],
            text=comparison_df[metric].round(3),
            textposition='auto',
        ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Models",
        yaxis_title="Score",
        barmode='group',
        height=500
    )
    
    st.plotly_chart(fig, width='stretch')
    
    st.markdown("---")
    
    selected_model = st.selectbox("Select model for details:", list(results.keys()))
    
    if selected_model:
        result = results[selected_model]
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Accuracy", f"{result['accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{result['precision']:.4f}")
        with col3:
            st.metric("Recall", f"{result['recall']:.4f}")
        with col4:
            st.metric("F1-Score", f"{result['f1_score']:.4f}")
        with col5:
            st.metric("ROC-AUC", f"{result['roc_auc']:.4f}")
        
        st.markdown("---")
        
        cm = np.array(result['confusion_matrix'])
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted No', 'Predicted Yes'],
            y=['Actual No', 'Actual Yes'],
            text=cm,
            texttemplate='%{text}',
            colorscale='Blues'
        ))
        fig.update_layout(title="Confusion Matrix", height=400)
        st.plotly_chart(fig, width='stretch')

def show_prediction(models, scaler, label_encoders, feature_names):
    st.title("🎯 Customer Churn Prediction")
    
    st.markdown("### 👤 Enter Customer Information")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
        
        with col2:
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        
        with col3:
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        
        col4, col5 = st.columns(2)
        
        with col4:
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment_method = st.selectbox("Payment Method", 
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        
        with col5:
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0)
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 500.0)
        
        model_choice = st.selectbox("Select Model", list(models.keys()))
        submit = st.form_submit_button("🔮 Predict Churn", width='stretch')
    
    if submit:
        input_data = {
            'gender': gender,
            'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        input_df = pd.DataFrame([input_data])
        
        for col in input_df.select_dtypes(include=['object']).columns:
            if col in label_encoders:
                le = label_encoders[col]
                try:
                    input_df[col] = le.transform(input_df[col])
                except:
                    input_df[col] = 0
        
        input_scaled = scaler.transform(input_df)
        
        model = models[model_choice]
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        st.markdown("---")
        st.markdown("### 📊 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            churn_status = "WILL CHURN" if prediction == 1 else "WILL NOT CHURN"
            color = '#e74c3c' if prediction == 1 else '#2ecc71'
            st.markdown(f"<h2 style='text-align: center; color: {color};'>{churn_status}</h2>", 
                       unsafe_allow_html=True)
        
        with col2:
            st.metric("Churn Probability", f"{prediction_proba[1]:.2%}")
        
        with col3:
            st.metric("Retention Probability", f"{prediction_proba[0]:.2%}")
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction_proba[1] * 100,
            title={'text': "Churn Risk Level"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "#2ecc71"},
                    {'range': [30, 70], 'color': "#f39c12"},
                    {'range': [70, 100], 'color': "#e74c3c"}
                ]
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, width='stretch')
        
        st.markdown("---")
        st.markdown("### 💡 Recommendations")
        
        if prediction == 1:
            st.warning("""
            **High Churn Risk!** Actions:
            - Offer retention incentives
            - Provide exclusive discounts
            - Schedule satisfaction call
            - Review service quality
            """)
        else:
            st.success("""
            **Low Churn Risk** - Customer likely to stay:
            - Maintain service quality
            - Explore upselling
            - Request feedback
            """)

def show_insights(results):
    st.title("💡 Business Insights")
    
    st.markdown("### 🎯 Key Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 📊 High-Risk Factors
        1. Month-to-month contracts
        2. Electronic check payments
        3. Fiber optic service
        4. New customers (0-12 months)
        5. No additional services
        """)
    
    with col2:
        st.markdown("""
        #### ✅ Retention Factors
        1. Long-term contracts
        2. Multiple services
        3. Automatic payments
        4. Longer tenure (24+ months)
        5. Tech support access
        """)
    
    st.markdown("---")
    
    best_model = max(results.items(), key=lambda x: x[1]['roc_auc'])
    
    st.success(f"""
    **Best Model:** {best_model[0]}
    - ROC-AUC: {best_model[1]['roc_auc']:.4f}
    - Accuracy: {best_model[1]['accuracy']:.4f}
    """)
    
    st.markdown("---")
    st.markdown("### 🚀 Recommendations")
    
    st.info("""
    **For High-Risk Customers:**
    - Immediate personalized outreach
    - Contract upgrade incentives
    - Service bundle offers
    - Premium support access
    
    **Expected Impact:**
    - 15-25% churn reduction
    - 20-30% CLV increase
    - Reduced acquisition costs
    """)

if __name__ == "__main__":
    main()