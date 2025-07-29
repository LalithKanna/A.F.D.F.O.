import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
page_title="Fiber Attenuation Prediction System",
page_icon="🔬",
layout="wide",
initial_sidebar_state="expanded"
)
@st.cache_resource
def load_all_models():
    model_files = {
        'fiber_loss': 'fiber_attenuation_model_component_fiberloss.pkl',
        'connector_loss': 'fiber_attenuation_model_component_connectorloss.pkl',
        'splice_loss': 'fiber_attenuation_model_component_spliceloss.pkl',
        'splitter_loss': 'fiber_attenuation_model_component_splitterloss.pkl',
        'bend_loss': 'fiber_attenuation_model_component_bendloss.pkl',
        'environmental_loss': 'fiber_attenuation_model_component_environmentalloss.pkl',
        'quantile_lower': 'fiber_attenuation_model_quantile_lower.pkl',
        'quantile_median': 'fiber_attenuation_model_quantile_median.pkl',
        'quantile_upper': 'fiber_attenuation_model_quantile_upper.pkl'
    }

    models = {}
    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            st.warning(f"⚠️ Model file not found: {path}")
    return models
# Custom CSS for better styling
st.markdown("""
<style>
   .main-header {
       font-size: 3rem;
       font-weight: bold;
       text-align: center;
       color: #667eea;
       margin-bottom: 2rem;
       text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
   }
   
   .nav-card {
       background: #f8f9fa;
       padding: 1.5rem;
       border-radius: 10px;
       box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
       margin: 1rem 0;
       border-left: 4px solid #667eea;
       color: #333;
   }
   
   .nav-card h2, .nav-card h3 {
       color: #333;
       margin-bottom: 1rem;
   }
   
   .nav-card p, .nav-card li {
       color: #555;
       line-height: 1.6;
   }
   
   .metric-card {
       background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
       padding: 1.5rem;
       border-radius: 10px;
       color: white;
       text-align: center;
       margin: 0.5rem;
       box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
   }
   
   .metric-card h3 {
       margin-bottom: 0.5rem;
       font-size: 1.2rem;
   }
   
   .metric-card p {
       margin: 0.2rem 0;
       font-weight: 500;
   }
   
   .feature-card {
       background: #f8f9ff;
       padding: 1rem;
       border-radius: 8px;
       border: 1px solid #e1e5e9;
       margin: 0.5rem 0;
   }
   
   /* Navigation button styling */
   .stButton > button {
       width: 100%;
       border-radius: 8px;
       border: 1px solid #ddd;
       margin-bottom: 0.5rem;
       transition: all 0.3s ease;
   }
   
   .stButton > button:hover {
       background-color: #667eea;
       color: white;
       border-color: #667eea;
   }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
st.session_state.page = 'Home'


# Add the missing load_training_metrics function (move it here so it's defined before use)
@st.cache_data
def load_training_metrics():
"""Load training metrics from YAML file or return None if not found"""
try:
# Try to load from common paths
possible_paths = [
'cleaned_training_log.yaml'
]

for path in possible_paths:
if os.path.exists(path):
with open(path, 'r') as file:
return yaml.safe_load(file)

# If no file found, return None
return None

except Exception as e:
st.error(f"Error loading training metrics: {str(e)}")
return None

# Mock data for model loading (replace with actual pickle loading)
@st.cache_data
def load_model_info():
"""Load model information and performance metrics"""
# This would be replaced with actual pickle loading
model_info = {
'lightgbm': {
'name': 'LightGBM Regression Model',
'mae': 0.342,
'rmse': 0.518,
'r2': 0.924,
'mape': 8.73,
'within_0.5db': 68.5,
'within_1db': 89.2,
'within_3db': 97.8,
'safety_accuracy': 92.4
},
'random_forest': {
'name': 'Random Forest Regression',
'mae': 0.398,
'rmse': 0.623,
'r2': 0.891,
'mape': 9.87,
'within_0.5db': 62.1,
'within_1db': 85.3,
'within_3db': 96.2,
'safety_accuracy': 88.7
},
'quantile_lower': {
'name': 'Quantile Regression (Lower)',
'description': 'Predicts 10th percentile for uncertainty estimation'
},
'quantile_median': {
'name': 'Quantile Regression (Median)',
'description': 'Predicts 50th percentile'
},
'quantile_upper': {
'name': 'Quantile Regression (Upper)', 
'description': 'Predicts 90th percentile for uncertainty estimation'
}
}

component_models = {
'fiber_loss': 'Fiber Loss Component',
'connector_loss': 'Connector Loss Component',
'splice_loss': 'Splice Loss Component',
'splitter_loss': 'Splitter Loss Component',
'bend_loss': 'Bend Loss Component',
'environmental_loss': 'Environmental Loss Component'
}

return model_info, component_models


# Feature definitions - FIXED: Ensuring all numerical values are of the same type
def get_feature_definitions():
return {
'Physical Features': {
'length_km': {'min': 0.1, 'max': 45.0, 'default': 5.0, 'step': 0.1},
'wavelength_nm': {'options': [850, 1300, 1310, 1550], 'default': 1550},
'fiber_type_encoded': {'options': ['SMF', 'MMF'], 'default': 'SMF'},
'fiber_subtype_encoded': {'options': ['G.652.D', 'G.657.A1', 'G.657.A2', 'G.657.B3', 'OM1', 'OM2', 'OM3', 'OM4', 'OM5'], 'default': 'G.652.D'}
},
'Connection Features': {
'num_splices': {'min': 0, 'max': 40, 'default': 2, 'step': 1},
'num_connectors': {'min': 2, 'max': 68, 'default': 2, 'step': 1},
'splitter_ratio': {'options': [1, 2, 4, 8, 16, 32], 'default': 1}
},
'Installation Features': {
'installation_type_encoded': {'options': ['Aerial', 'Underground', 'Indoor','Building'], 'default': 'Underground'},
'cable_type_encoded': {'options': ['Loose Tube', 'Tight Buffer', 'Ribbon','Drop_cable'], 'default': 'Loose Tube'},
'age_years': {'min': 0, 'max': 15, 'default': 5, 'step': 1}
},
'Environmental Features': {
'temperature_C': {'min': -30.0, 'max': 82.948849, 'default': 25.0, 'step': 1.0},
'humidity_percent': {'min': 15.550313, 'max': 94.937013, 'default': 50.0, 'step': 5.0},
'environmental_stress': {'min': 0.0, 'max': 1.0, 'default': 0.3, 'step': 0.1}
},
'Mechanical Features': {
'bend_radius_mm': {'min': 5.0, 'max': 94.874319, 'default': 15.0, 'step': 1.0},
'num_bends': {'min': 0, 'max': 347, 'default': 2, 'step': 1}
},
'Power Budget Features': {
'transmitter_power_dbm': {'min': -10.0, 'max': 10.0, 'default': 0.0, 'step': 0.1},
'receiver_sensitivity_dbm': {'min': -40.0, 'max': -10.0, 'default': -25.0, 'step': 0.1},
'safety_margin_db': {'min': 1.0, 'max': 10.0, 'default': 3.0, 'step': 0.1}
}
}

# Component loss prediction models (mock implementation)
def predict_component_losses(inputs):
    """Predict component losses using real ML models"""
    component_losses = {}
    input_df = pd.DataFrame([inputs])

    for comp in ['fiber_loss', 'connector_loss', 'splice_loss', 'splitter_loss', 'bend_loss', 'environmental_loss']:
        model = models.get(comp)
        if model:
            try:
                component_losses[comp] = model.predict(input_df)[0]
            except Exception as e:
                st.error(f"{comp} prediction failed: {e}")
                component_losses[comp] = 0.0
        else:
            component_losses[comp] = 0.0

    return component_losses


def calculate_power_budget_analysis(total_loss, inputs):
"""Calculate power budget and determine link safety"""

# Calculate available power budget
tx_power = inputs['transmitter_power_dbm']
rx_sensitivity = inputs['receiver_sensitivity_dbm'] 
safety_margin = inputs['safety_margin_db']

# Total available power budget
total_budget = tx_power - rx_sensitivity  # This gives us positive dB value

# Required power budget (total loss + safety margin)
required_budget = total_loss + safety_margin

# Power margin calculation
power_margin = total_budget - required_budget

# Link classification
if power_margin >= safety_margin:
link_status = "SAFE"
status_color = "green"
status_icon = "✅"
elif power_margin >= 0:
link_status = "MARGINAL"  
status_color = "orange"
status_icon = "⚠️"
else:
link_status = "UNSAFE"
status_color = "red" 
status_icon = "❌"

return {
'total_budget': total_budget,
'required_budget': required_budget,
'power_margin': power_margin,
'link_status': link_status,
'status_color': status_color,
'status_icon': status_icon,
'tx_power': tx_power,
'rx_sensitivity': rx_sensitivity,
'safety_margin': safety_margin
}

# Navigation
def render_navigation():
st.sidebar.title("🔬 Navigation")

pages = {
'Home': '🏠',
'AI Prediction': '🤖', 
'Normal Calculation': '📊',
'Model Evaluation': '📈',
'Documentation': '📚'
}

# Display current page
st.sidebar.info(f"📍 Current: {st.session_state.page}")

for page, icon in pages.items():
# Highlight current page
button_type = "primary" if st.session_state.page == page else "secondary"
if st.sidebar.button(f"{icon} {page}", key=f"nav_{page}", use_container_width=True, type=button_type):
st.session_state.page = page
st.rerun()

# Home Page
def render_home_page():
st.markdown('<h1 class="main-header">A.F.D.F.O – AI-Powered Fiber Deployment Feasibility & Optimization Tool</h1>', unsafe_allow_html=True)

# Hero Section
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
st.markdown("""
       <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #e0f7fa, #fce4ec);
                   border-radius: 15px; margin: 1rem 0; border: 2px solid #80cbc4;">
           <h2 style="color: #00695c; margin-bottom: 1rem;">🔧 Smarter Fiber Planning Starts Here</h2>
           <p style="color: #444; font-size: 1.1rem;">Predict, analyze, and improve your fiber optic deployments with AI</p>
           <div style="margin: 1rem 0; font-size: 2rem;">📡──🔍──📊──✅</div>
           <p style="color: #666; font-style: italic;">Feasibility • Optimization • Confidence</p>
       </div>
       """, unsafe_allow_html=True)

# Introduction
st.markdown("""
   <div class="nav-card">
       <h2>📌 About the Tool</h2>
       <p><strong>A.F.D.F.O</strong> is an intelligent assistant that helps you evaluate the feasibility of fiber optic deployments. 
       It uses AI to estimate signal loss (attenuation) based on your design, installation, and environmental conditions. 
       This tool is ideal for network engineers, planners, and technical decision-makers looking for reliable deployment insights.</p>
   </div>
   """, unsafe_allow_html=True)

# Core Features
st.markdown("## 🧰 What Can This Tool Do?")
col1, col2 = st.columns(2)

with col1:
st.markdown("""
       <div class="nav-card">
           <h3>📏 Predict Total Attenuation</h3>
           <ul>
               <li>Input key parameters like fiber type, length, splices, and connectors</li>
               <li>Include environmental conditions such as temperature and humidity</li>
               <li>Supports installation-specific data like cable type and deployment method</li>
           </ul>
       </div>
       """, unsafe_allow_html=True)

with col2:
st.markdown("""
       <div class="nav-card">
           <h3>🔍 Estimate Missing Values</h3>
           <ul>
               <li>If some input data is missing (e.g., bend radius), the system can intelligently estimate it</li>
               <li>Automatically calculates bend loss using estimated or provided data</li>
               <li>Reduces the burden of manual measurements</li>
           </ul>
       </div>
       """, unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
st.markdown("""
       <div class="nav-card">
           <h3>📊 Analyze Loss Components</h3>
           <ul>
               <li>Breaks down total loss into splice loss, connector loss, bend loss, and more</li>
               <li>Highlights which part contributes most to the overall attenuation</li>
               <li>Helps identify areas for design or installation improvement</li>
           </ul>
       </div>
       """, unsafe_allow_html=True)

with col2:
st.markdown("""
       <div class="nav-card">
           <h3>📈 Confidence in Predictions</h3>
           <ul>
               <li>Provides a range of likely attenuation values (uncertainty estimation)</li>
               <li>Displays how confident the system is about its prediction</li>
               <li>Makes planning more robust and informed</li>
           </ul>
       </div>
       """, unsafe_allow_html=True)

# Prototype Disclaimer
st.markdown("## 🧪 Prototype Notice")
col1, col2 = st.columns([3, 1])

with col1:
st.markdown("""
       <div class="nav-card">
           <h4 style="color: #00838f; margin-bottom: 0.5rem;">This is a Prototype</h4>
           <p>This version of A.F.D.F.O is under active development. It is designed to demonstrate the potential of AI in fiber network planning and feasibility checking. More advanced features and refinements will be added in future versions based on feedback and testing.</p>
       </div>
       """, unsafe_allow_html=True)

with col2:
st.markdown("""
       <div style="background: #fff3e0; padding: 1rem; border-radius: 8px; text-align: center;">
           <h3 style="color: #ef6c00; margin: 0;">🧪</h3>
           <p style="color: #e65100; margin: 0.5rem 0; font-weight: bold;">Prototype</p>
           <p style="color: #555; font-size: 0.9rem;">Early Version</p>
       </div>
       """, unsafe_allow_html=True)

# AI Prediction Page  
def render_ai_prediction_page():
    st.title("🤖 AI-Powered Attenuation Prediction")

    # Load models
    models = load_all_models()
    feature_defs = get_feature_definitions()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 🔧 Input Parameters")

        # Collect input features
        inputs = {}
        for group_name, features in feature_defs.items():
            with st.expander(f"{group_name}", expanded=True):
                for feature, config in features.items():
                    if 'options' in config:
                        inputs[feature] = st.selectbox(
                            feature.replace('_', ' ').title(),
                            config['options'],
                            index=config['options'].index(config.get('default', config['options'][0]))
                        )
                    else:
                        inputs[feature] = st.number_input(
                            feature.replace('_', ' ').title(),
                            min_value=config['min'],
                            max_value=config['max'],
                            value=config['default'],
                            step=config.get('step', 0.1)
                        )

        # Model selection
        st.markdown("### 🎯 Model Selection")
        selected_models = st.multiselect(
            "Choose models for prediction:",
            ['quantile_lower', 'quantile_median', 'quantile_upper'],
            default=['quantile_lower', 'quantile_median', 'quantile_upper']
        )

        predict_button = st.button("🚀 Predict Attenuation", type="primary", use_container_width=True)

    with col2:
        st.markdown("### 📊 Prediction Results")

        if predict_button:
            # Get component losses
            component_losses = predict_component_losses(inputs)
            total_component_loss = sum(component_losses.values())

            input_df = pd.DataFrame([inputs])
            results = {}

            # Predict using quantile models
            if 'quantile_lower' in selected_models and 'quantile_lower' in models:
                results['Lower Bound (10%)'] = models['quantile_lower'].predict(input_df)[0]
            if 'quantile_median' in selected_models and 'quantile_median' in models:
                results['Median (50%)'] = models['quantile_median'].predict(input_df)[0]
            if 'quantile_upper' in selected_models and 'quantile_upper' in models:
                results['Upper Bound (90%)'] = models['quantile_upper'].predict(input_df)[0]

            main_pred = results.get('Median (50%)', total_component_loss)

            # Power budget analysis
            power_analysis = calculate_power_budget_analysis(main_pred, inputs)

            # Display status
            st.markdown(f"""
            <div style="background: {power_analysis['status_color']}; color: white; padding: 1.5rem; 
                        border-radius: 10px; text-align: center; margin-bottom: 1rem; 
                        box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
                <h2>{power_analysis['status_icon']} LINK STATUS: {power_analysis['link_status']}</h2>
                <h3>Power Margin: {power_analysis['power_margin']:.2f} dB</h3>
            </div>
            """, unsafe_allow_html=True)

            # Main prediction and budget
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🔍 Total Loss Prediction</h3>
                    <h2>{main_pred:.3f} dB</h2>
                    <p>Predicted by Quantile Model</p>
                </div>
                """, unsafe_allow_html=True)
            with col_b:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>⚡ Power Budget</h3>
                    <h2>{power_analysis['total_budget']:.1f} dB</h2>
                    <p>Available Budget</p>
                </div>
                """, unsafe_allow_html=True)

            # Budget metrics
            st.markdown("### 🔋 Power Budget Analysis")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("TX Power", f"{power_analysis['tx_power']:.1f} dBm")
            c2.metric("RX Sensitivity", f"{power_analysis['rx_sensitivity']:.1f} dBm")
            c3.metric("Safety Margin", f"{power_analysis['safety_margin']:.1f} dB")
            c4.metric("Required Budget", f"{power_analysis['required_budget']:.2f} dB")

            # Budget visualization
            fig = px.bar(
                x=['Available Budget', 'Total Loss', 'Safety Margin', 'Remaining Margin'],
                y=[
                    power_analysis['total_budget'],
                    main_pred,
                    power_analysis['safety_margin'],
                    max(0, power_analysis['power_margin'])
                ],
                color=['Available Budget', 'Total Loss', 'Safety Margin', 'Remaining Margin'],
                color_discrete_sequence=['lightblue', 'orange', 'yellow', 'lightgreen' if power_analysis['power_margin'] > 0 else 'red'],
                title="Power Budget Breakdown"
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Uncertainty band
            if 'Lower Bound (10%)' in results and 'Upper Bound (90%)' in results:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[results['Lower Bound (10%)']]*2,
                    fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[results['Upper Bound (90%)']]*2,
                    fill='tonexty', mode='lines', name='Uncertainty Band (80%)',
                    fillcolor='rgba(102, 126, 234, 0.3)', line_color='rgba(0,0,0,0)'
                ))
                fig.add_trace(go.Scatter(
                    x=[0.5], y=[main_pred],
                    mode='markers', name='Primary Prediction',
                    marker=dict(size=15, color='red')
                ))
                fig.add_hline(
                    y=power_analysis['total_budget'] - power_analysis['safety_margin'],
                    line_dash="dash", line_color="green",
                    annotation_text="Max Allowable Loss"
                )
                fig.update_layout(
                    title="Prediction with Uncertainty vs Power Budget",
                    yaxis_title="Attenuation (dB)",
                    height=300,
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)

            # Prediction table
            df_results = pd.DataFrame(list(results.items()), columns=['Model', 'Prediction (dB)'])
            df_results['Prediction (dB)'] = df_results['Prediction (dB)'].round(3)
            df_results['Within Budget'] = df_results['Prediction (dB)'] <= (power_analysis['total_budget'] - power_analysis['safety_margin'])
            st.dataframe(df_results, use_container_width=True)

            # 🔍 Component Loss Breakdown
            st.markdown("### 🔍 Component Loss Breakdown (Model-Based)")
            fig = px.pie(
                values=list(component_losses.values()),
                names=[k.replace('_', ' ').title() for k in component_losses],
                title="Loss Attribution by Component (Predicted)"
            )
            st.plotly_chart(fig, use_container_width=True)

            df_components = pd.DataFrame([
                {
                    'Component': k.replace('_', ' ').title(),
                    'Loss (dB)': round(v, 3),
                    'Percentage': round((v / total_component_loss * 100), 1),
                    'Model Used': 'ML Model'
                } for k, v in component_losses.items()
            ])
            st.dataframe(df_components, use_container_width=True)

            # 🛡️ Risk Assessment
            st.markdown("### 🛡️ Risk Assessment")
            risk_factors = []
            if power_analysis['power_margin'] < 1:
                risk_factors.append("⚠️ Low power margin - consider system upgrades")
            if component_losses['bend_loss'] > 0.5:
                risk_factors.append("⚠️ High bend loss - check installation practices")
            if component_losses['environmental_loss'] > 0.3:
                risk_factors.append("⚠️ High environmental stress - monitor conditions")
            if inputs['age_years'] > 10:
                risk_factors.append("⚠️ Aging infrastructure - plan for maintenance")

            if risk_factors:
                for risk in risk_factors:
                    st.warning(risk)
            else:
                st.success("✅ No significant risk factors identified")

            # 💡 Recommendations
            st.markdown("### 💡 Optimization Recommendations")
            recommendations = []
            if component_losses['connector_loss'] > 1.0:
                recommendations.append("🔧 Consider reducing number of connectors or using higher-grade connectors")
            if component_losses['splice_loss'] > 0.5:
                recommendations.append("🔧 Review splicing quality - consider fusion splicing training")
            if component_losses['bend_loss'] > 0.2:
                recommendations.append("🔧 Increase bend radius to reduce losses")
            if power_analysis['power_margin'] < 0:
                recommendations.append("🔧 Consider higher power transmitter or more sensitive receiver")

            if recommendations:
                for rec in recommendations:
                    st.info(rec)
            else:
                st.success("✅ System appears well-optimized")


# Model Evaluation Page
def render_model_evaluation_page():
st.title("📈 Model Performance Evaluation")

# Load real training metrics
training_metrics = load_training_metrics()

if training_metrics:
st.success("✅ Real training metrics loaded successfully!")

# Display training information from YAML
st.markdown("## 📋 Training Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
st.markdown(f"""
           <div class="metric-card">
               <h3>📅 Training Date</h3>
               <p>{training_metrics.get('timestamp', 'Unknown')}</p>
           </div>
           """, unsafe_allow_html=True)

with col2:
st.markdown(f"""
           <div class="metric-card">
               <h3>🔢 Version</h3> 
               <p>v{training_metrics.get('model_version', '1.0.0')}</p>
           </div>
           """, unsafe_allow_html=True)

with col3:
st.markdown(f"""
           <div class="metric-card">
               <h3>🏆 Best Model</h3>
               <p>{training_metrics.get('best_model', 'Unknown').upper()}</p>
           </div>
           """, unsafe_allow_html=True)

with col4:
features_count = len(training_metrics.get('features_used', []))
st.markdown(f"""
           <div class="metric-card">
               <h3>📊 Features</h3>
               <p>{features_count} Features</p>
           </div>
           """, unsafe_allow_html=True)

# Performance metrics from YAML
if 'performance_metrics' in training_metrics:
st.markdown("## 🎯 Model Performance Metrics")

metrics = training_metrics['performance_metrics']

# Extract values from numpy scalars if needed
def extract_metric_value(metric_obj):
if hasattr(metric_obj, 'item'):
return metric_obj.item()
return metric_obj

# Create metrics display
col1, col2, col3 = st.columns(3)

with col1:
mae_val = extract_metric_value(metrics.get('mae', 0))
rmse_val = extract_metric_value(metrics.get('rmse', 0))
st.markdown(f"""
               <div class="nav-card">
                   <h4>📉 Error Metrics</h4>
                   <p><strong>MAE:</strong> {mae_val:.3f} dB</p>
                   <p><strong>RMSE:</strong> {rmse_val:.3f} dB</p>
               </div>
               """, unsafe_allow_html=True)

with col2:
r2_val = extract_metric_value(metrics.get('r2', 0))
mape_val = extract_metric_value(metrics.get('mape', 0))
st.markdown(f"""
               <div class="nav-card">
                   <h4>🎯 Accuracy Metrics</h4>
                   <p><strong>R² Score:</strong> {r2_val:.3f}</p>
                   <p><strong>MAPE:</strong> {mape_val:.2f}%</p>
               </div>
               """, unsafe_allow_html=True)

with col3:
safety_acc = extract_metric_value(metrics.get('safety_accuracy', 0))
st.markdown(f"""
               <div class="nav-card">
                   <h4>🛡️ Safety Metrics</h4>
                   <p><strong>Safety Accuracy:</strong> {safety_acc:.1f}%</p>
               </div>
               """, unsafe_allow_html=True)

# Tolerance analysis
st.markdown("## 🎯 Prediction Tolerance Analysis")

tolerance_data = {
'Tolerance Level': ['Within 0.5dB', 'Within 1dB', 'Within 3dB'],
'Accuracy (%)': [
extract_metric_value(metrics.get('within_0.5db', 0)),
extract_metric_value(metrics.get('within_1db', 0)),
extract_metric_value(metrics.get('within_3db', 0))
]
}

fig = px.bar(
tolerance_data,
x='Tolerance Level',
y='Accuracy (%)',
title="Model Accuracy at Different Tolerance Levels",
color='Accuracy (%)',
color_continuous_scale='viridis'
)
fig.update_layout(height=400)
st.plotly_chart(fig, use_container_width=True)

# Feature information
if 'features_used' in training_metrics:
st.markdown("## 🔍 Feature Analysis")

features = training_metrics['features_used']
feature_groups = training_metrics.get('feature_groups', {})

col1, col2 = st.columns(2)

with col1:
st.markdown("### 📋 Features Used")
# Group features for better display
for i in range(0, len(features), 8):
feature_batch = features[i:i+8]
feature_list = "• " + "\n• ".join(feature_batch)
st.text(feature_list)

with col2:
if feature_groups:
st.markdown("### 📊 Feature Groups")

# Create pie chart for feature groups
fig = px.pie(
values=list(feature_groups.values()),
names=list(feature_groups.keys()),
title="Features by Category"
)
st.plotly_chart(fig, use_container_width=True)

# Hyperparameters
if 'hyperparameters' in training_metrics:
st.markdown("## ⚙️ Model Configuration")

col1, col2 = st.columns(2)

with col1:
st.markdown("### 🔧 Hyperparameters")
hyperparams = training_metrics['hyperparameters']

# Format hyperparameters for display
param_text = []
for key, value in hyperparams.items():
if isinstance(value, float):
param_text.append(f"**{key}:** {value:.4f}")
else:
param_text.append(f"**{key}:** {value}")

st.markdown("\n".join(param_text))

with col2:
st.markdown("### 📝 Training Notes")
if 'notes' in training_metrics:
for note in training_metrics['notes']:
st.info(f"✅ {note}")

# Component models info
if 'component_models_trained' in training_metrics:
st.markdown("### 🧩 Component Models")
components = training_metrics['component_models_trained']
for comp in components:
st.text(f"• {comp}")

else:
# Fallback to mock data if real metrics not available
st.warning("⚠️ Real training metrics not found. Displaying sample data.")

model_info, _ = load_model_info()

# Model comparison metrics
st.markdown("## 🏆 Model Performance Comparison")

# Create metrics comparison
metrics_data = []
for model, info in model_info.items():
if 'mae' in info:  # Only for models with performance metrics
metrics_data.append({
'Model': info['name'],
'MAE (dB)': info['mae'],
'RMSE (dB)': info['rmse'], 
'R² Score': info['r2'],
'MAPE (%)': info['mape'],
'Within 0.5dB (%)': info['within_0.5db'],
'Within 1dB (%)': info['within_1db'],
'Within 3dB (%)': info['within_3db'],
'Safety Accuracy (%)': info['safety_accuracy']
})

df_metrics = pd.DataFrame(metrics_data)
st.dataframe(df_metrics, use_container_width=True)

# Visualization of metrics
col1, col2 = st.columns(2)

with col1:
# Accuracy metrics
fig = go.Figure()
fig.add_trace(go.Bar(
x=df_metrics['Model'],
y=df_metrics['R² Score'],
name='R² Score',
marker_color='lightblue'
))
fig.update_layout(title="Model Accuracy (R² Score)", yaxis_title="R² Score")
st.plotly_chart(fig, use_container_width=True)

with col2:
# Error metrics
fig = go.Figure()
fig.add_trace(go.Bar(
x=df_metrics['Model'],
y=df_metrics['MAE (dB)'],
name='MAE',
marker_color='lightcoral'
))
fig.update_layout(title="Model Error (MAE)", yaxis_title="MAE (dB)")
st.plotly_chart(fig, use_container_width=True)

# Feature importance (mock data)
st.markdown("## 🔍 Feature Importance Analysis")

features = ['Length (km)', 'Wavelength', 'Temperature', 'Num Splices', 'Fiber Type', 
'Age', 'Bend Radius', 'Humidity', 'Num Connectors', 'Installation Type']
importance = np.random.uniform(0.05, 0.25, len(features))
importance = importance / importance.sum()

fig = px.bar(
x=importance,
y=features,
orientation='h',
title="Feature Importance (SHAP Values)",
labels={'x': 'Importance Score', 'y': 'Features'}
)
fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)

# Placeholder pages
def render_documentation_page():
st.title("📚 Documentation")

st.markdown("""
   # 🔧 A.F.D.F.O Documentation

   ## 🛰️ What This System Does

   **A.F.D.F.O** (AI-powered Fiber Deployment Feasibility & Optimization Tool) is a smart planning assistant for fiber optic network deployment. It helps engineers and planners:

   - ✅ Predict the **total signal attenuation (in dB)** over a planned fiber link
   - ✅ Estimate **uncertainty** in predictions (how confident the model is)
   - ✅ Automatically fill in **missing input parameters** like bend radius
   - ✅ Break down total attenuation into **component-wise losses**:
       - Fiber
       - Connectors
       - Splices
       - Splitters
       - Bends
       - Environment
   - ✅ Identify **which component contributes most** to signal loss
   - ✅ Support early-stage decision-making in **design, budgeting, and optimization**

   ---

   ## ⚙️ How It Works (High-Level)

   1. **User Input**: Physical, mechanical, and environmental features of the fiber route are entered into the system.
   2. **Feature Engineering**: The system creates additional informative variables by combining or transforming user inputs.
   3. **Missing Data Handling**: If certain values (like bend radius) are not available, they are imputed using learned statistical relationships.
   4. **Prediction**:
       - The model predicts the **total attenuation**
       - It also provides a **prediction range** (e.g., 3.2–3.8 dB) using quantile regression
       - It **separates the loss** into contributing components (e.g., 60% from fiber, 20% from connectors)
   5. **Results Display**: The user receives a clear view of the total loss, where it comes from, and how confident the system is.

    > All predictions are based on previously learned patterns from historical and simulated deployment scenarios.
    > All predictions are based on previously learned patterns from  simulated deployment scenarios.

   ---

   ## 🧠 ML Models Used

   - **Best Performing Model**: `LightGBM` (Gradient Boosted Decision Trees)
   - **Model Type**: Regression
   - **Features Used**: 25+ input and engineered variables, including environmental, physical, mechanical, and interaction terms
   - **Uncertainty Estimation**: Enabled via **quantile regression**
   - **Component-wise Models**:
       - fiber_loss_dB
       - connector_loss_dB
       - splice_loss_dB
       - splitter_loss_dB
       - bend_loss_dB
       - environmental_loss_dB
   - **Explainability**: Integrated via SHAP (not shown in UI yet)
   - **Tuning Method**: Optuna for hyperparameter optimization

   ### 🔍 Feature Groups

   - Connections: 3 features
   - Engineered: 5
   - Environmental: 4
   - Interactions: 5
   - Mechanical: 3
   - Physical: 4

   ---

   ## 📁 Dataset Overview

   The model was trained on a diverse, high-quality dataset that includes:

   - Thousands of deployment samples
   - Cleaned and filtered using domain rules (no impossible or negative loss values)
   - Its completely based on simulated data covering:
       - SMF/MMF
       - Various subtypes (G.652.D, G.657.A1/A2/B3, OM3–OM5)
       - All major installation types: underground, aerial, building, indoor
       - Cable types: loose tube, tight buffered, drop, ribbon

   ### ⚠️ Filtered Data Flags
   - No negative/invalid losses
   - No extreme or inconsistent values
   - Valid physics-based safety margins and component consistency

   ### 🔢 Numerical Feature Ranges

   | Feature | Range |
   |--------|--------|
   | Wavelength (nm) | 850 – 1550 |
   | Length (km) | 0.1 – 45 |
   | Splices | 0 – 40 |
   | Connectors | 2 – 68 |
   | Splitter Ratio | 0 – 32 |
   | Temperature (°C) | -30.5 – 82.9 |
   | Humidity (%) | 15.5 – 94.9 |
   | Bend Radius (mm) | 5 – 94.8 |
   | Age (years) | 0.1 – 15 |
   | Total Attenuation | 0.22 – 80 dB |

   > Values outside these ranges may reduce prediction accuracy.

   ---

   ## 📈 Model Accuracy & Performance

   ### 🏆 Overall Metrics (Total Attenuation Prediction)

   | Metric              | Value |
   |---------------------|--------|
   | MAE (Mean Abs Error)| 0.29 dB |
   | RMSE                | 0.65 dB |
   | MAPE                | 1.84% |
   | R² Score            | 0.9959 |
   | Predictions within 0.5 dB | 85.96% |
   | Predictions within 1 dB   | 94.73% |
   | Predictions within 3 dB   | 99.28% |
   | Safety Rule Accuracy      | 99.24% |

   > These metrics were validated on held-out test data from within the training distribution.

   ### 🟢 Most Accurate For:

   - Standard single-mode deployments (SMF, G.652.D, G.657.A1/A2)
   - Typical lengths (1–30 km)
   - Common install types: underground, aerial
   - Reasonable environmental ranges (10–60 °C, humidity < 90%)
   - Loss budgets between 0.5 and 25 dB

   ### 🔴 Use With Caution:

   - Extremely long or short fibers (<0.5 km or >45 km)
   - Exotic fiber types (OM5, OM1)
   - Highly aged installations (>15 years)
   - Unusual or extreme weather conditions

   ---

   ## 🧑‍💻 Deployment Notes for Engineers

   - This model assumes all inputs are consistent and physics-aligned.
   - Inputs outside trained ranges **will still be accepted**, but predictions may have higher uncertainty.
   - For production deployment, monitoring prediction uncertainty is strongly advised.
   - Outputs are intended to **assist**, not replace, expert judgment in deployment planning.

   ---

   ## 🆕 Prototype Version: `1.0`

   **Changelog Highlights**
   - ➕ Added uncertainty prediction (quantile regression)
   - 🔍 Component-level loss breakdown
   - 📐 Better feature engineering (interactions and polynomial terms)
   - ⚙️ Optuna-based tuning for model robustness
   - 🔎 SHAP explainability integration

   ---

   ## 📬 Support

   For issues or feature requests, please contact Phone Number:7094501353,Gmail: rlalithkanna@gmail.com"
   """)


def render_normal_calculation_page():
st.title("📊 Normal Calculation")
st.info("🧮 Traditional calculation methods coming soon...")
st.markdown("""
   This page will include:
   - Manual attenuation calculations
   - Industry standard formulas
   - Component-wise calculations
   - Comparison with AI predictions
   - Educational tools
   """)

# Main app logic
def main():
render_navigation()

if st.session_state.page == 'Home':
render_home_page()
elif st.session_state.page == 'AI Prediction':
render_ai_prediction_page()
elif st.session_state.page == 'Model Evaluation':
render_model_evaluation_page()
elif st.session_state.page == 'Documentation':
render_documentation_page()
elif st.session_state.page == 'Normal Calculation':
render_normal_calculation_page()

if __name__ == "__main__":
main()
