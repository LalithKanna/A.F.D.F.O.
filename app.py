import joblib
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
import numpy as np

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Fiber Attenuation Prediction System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

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

@st.cache_data
def load_training_metrics():
    """Load training metrics from YAML file or return None if not found"""
    try:
        possible_paths = ['cleaned_training_log.yaml']
        for path in possible_paths:
            if os.path.exists(path):
                with open(path, 'r') as file:
                    return yaml.safe_load(file)
    except Exception as e:
        st.error(f"Error loading training metrics: {str(e)}")
        return None

@st.cache_data
def load_model_info():
    """Load model information and performance metrics"""
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

def engineer_features(inputs):
    """
    Create all the engineered features that the model expects
    This function replicates the feature engineering from training
    """
    # Create a copy to avoid modifying the original
    features = inputs.copy()
    
    # Convert categorical features to numeric if needed
    # Fiber type encoding
    if 'fiber_type_encoded' in features:
        if isinstance(features['fiber_type_encoded'], str):
            features['fiber_type_encoded'] = 1 if features['fiber_type_encoded'] == 'SMF' else 0
    
    # Installation type encoding (you may need to adjust based on your training encoding)
    installation_mapping = {'Aerial': 0, 'Underground': 1, 'Indoor': 2, 'Building': 3}
    if 'installation_type_encoded' in features and isinstance(features['installation_type_encoded'], str):
        features['installation_type_encoded'] = installation_mapping.get(features['installation_type_encoded'], 1)
    
    # Cable type encoding
    cable_mapping = {'Loose Tube': 0, 'Tight Buffer': 1, 'Ribbon': 2, 'Drop_cable': 3}
    if 'cable_type_encoded' in features and isinstance(features['cable_type_encoded'], str):
        features['cable_type_encoded'] = cable_mapping.get(features['cable_type_encoded'], 0)
    
    # Fiber subtype encoding (simplified - you may need to adjust)
    subtype_mapping = {
        'G.652.D': 0, 'G.657.A1': 1, 'G.657.A2': 2, 'G.657.B3': 3,
        'OM1': 4, 'OM2': 5, 'OM3': 6, 'OM4': 7, 'OM5': 8
    }
    if 'fiber_subtype_encoded' in features and isinstance(features['fiber_subtype_encoded'], str):
        features['fiber_subtype_encoded'] = subtype_mapping.get(features['fiber_subtype_encoded'], 0)
    
    # Now create engineered features
    length_km = features.get('length_km', 1.0)
    
    # 1. Attenuation per km (typical values based on wavelength and fiber type)
    wavelength = features.get('wavelength_nm', 1550)
    fiber_type = features.get('fiber_type_encoded', 1)  # 1 for SMF, 0 for MMF
    
    if fiber_type == 1:  # SMF
        if wavelength == 1310:
            features['attenuation_per_km'] = 0.35
        elif wavelength == 1550:
            features['attenuation_per_km'] = 0.25
        else:
            features['attenuation_per_km'] = 0.30
    else:  # MMF
        if wavelength == 850:
            features['attenuation_per_km'] = 3.0
        elif wavelength == 1300:
            features['attenuation_per_km'] = 1.0
        else:
            features['attenuation_per_km'] = 2.0
    
    # 2. Environmental impact
    temp = features.get('temperature_C', 25)
    humidity = features.get('humidity_percent', 50)
    env_stress = features.get('environmental_stress', 0.3)
    features['env_impact'] = (abs(temp - 25) / 100) + (humidity / 100) + env_stress
    
    # 3. Connections per km
    num_connectors = features.get('num_connectors', 2)
    features['connections_per_km'] = num_connectors / length_km if length_km > 0 else num_connectors
    
    # 4. Bend severity
    bend_radius = features.get('bend_radius_mm', 15)
    num_bends = features.get('num_bends', 2)
    features['bend_severity'] = num_bends / max(bend_radius, 1)  # Avoid division by zero
    
    # 5. Link complexity (combination of various complexity factors)
    num_splices = features.get('num_splices', 0)
    splitter_ratio = features.get('splitter_ratio', 1)
    features['link_complexity'] = (num_splices + num_connectors + num_bends + 
                                   np.log2(max(splitter_ratio, 1))) / max(length_km, 0.1)
    
    # 6. Splitter ratio x length interaction
    features['splitter_ratio_x_length'] = splitter_ratio * length_km
    
    # 7. Connectors per km
    features['connectors_per_km'] = num_connectors / max(length_km, 0.1)
    
    # 8. Bends per radius
    features['bends_per_radius'] = num_bends / max(bend_radius, 1)
    
    # 9. Environmental stress x length interaction
    features['env_stress_x_length'] = env_stress * length_km
    
    # 10. Length squared
    features['length_squared'] = length_km ** 2
    
    return features

def predict_component_losses(inputs, models):
    """
    Enhanced prediction function with proper feature engineering
    """
    # First, engineer the features
    engineered_inputs = engineer_features(inputs)
    
    # Create DataFrame with all expected features in correct order
    expected_features = [
        'length_km', 'wavelength_nm', 'num_splices', 'num_connectors', 
        'fiber_type_encoded', 'fiber_subtype_encoded', 'installation_type_encoded', 
        'cable_type_encoded', 'temperature_C', 'humidity_percent', 'age_years', 
        'environmental_stress', 'bend_radius_mm', 'num_bends', 'splitter_ratio',
        'attenuation_per_km', 'env_impact', 'connections_per_km', 'bend_severity',
        'link_complexity', 'splitter_ratio_x_length', 'connectors_per_km',
        'bends_per_radius', 'env_stress_x_length', 'length_squared'
    ]
    
    # Create feature vector in correct order
    feature_vector = []
    for feature in expected_features:
        if feature in engineered_inputs:
            feature_vector.append(engineered_inputs[feature])
        else:
            # Provide default values for missing features
            print(f"Warning: Missing feature {feature}, using default value")
            feature_vector.append(0.0)
    
    input_df = pd.DataFrame([feature_vector], columns=expected_features)
    
    component_losses = {}
    for comp in ['fiber_loss', 'connector_loss', 'splice_loss', 'splitter_loss', 'bend_loss', 'environmental_loss']:
        model = models.get(comp)
        if model:
            try:
                component_losses[comp] = model.predict(input_df)[0]
            except Exception as e:
                print(f"{comp} prediction failed: {e}")
                component_losses[comp] = 0.0
        else:
            component_losses[comp] = 0.0
    
    return component_losses

def predict_quantile_models(inputs, models, selected_models):
    """
    Predict using quantile regression models with proper feature engineering
    """
    # Engineer features
    engineered_inputs = engineer_features(inputs)
    
    # Create feature vector (same as above)
    expected_features = [
        'length_km', 'wavelength_nm', 'num_splices', 'num_connectors', 
        'fiber_type_encoded', 'fiber_subtype_encoded', 'installation_type_encoded', 
        'cable_type_encoded', 'temperature_C', 'humidity_percent', 'age_years', 
        'environmental_stress', 'bend_radius_mm', 'num_bends', 'splitter_ratio',
        'attenuation_per_km', 'env_impact', 'connections_per_km', 'bend_severity',
        'link_complexity', 'splitter_ratio_x_length', 'connectors_per_km',
        'bends_per_radius', 'env_stress_x_length', 'length_squared'
    ]
    
    feature_vector = []
    for feature in expected_features:
        if feature in engineered_inputs:
            feature_vector.append(engineered_inputs[feature])
        else:
            feature_vector.append(0.0)
    
    input_df = pd.DataFrame([feature_vector], columns=expected_features)
    
    results = {}
    # Predict using quantile models
    if 'quantile_lower' in selected_models and 'quantile_lower' in models:
        try:
            results['Lower Bound (10%)'] = models['quantile_lower'].predict(input_df)[0]
        except Exception as e:
            print(f"quantile_lower prediction failed: {e}")
            
    if 'quantile_median' in selected_models and 'quantile_median' in models:
        try:
            results['Median (50%)'] = models['quantile_median'].predict(input_df)[0]
        except Exception as e:
            print(f"quantile_median prediction failed: {e}")
            
    if 'quantile_upper' in selected_models and 'quantile_upper' in models:
        try:
            results['Upper Bound (90%)'] = models['quantile_upper'].predict(input_df)[0]
        except Exception as e:
            print(f"quantile_upper prediction failed: {e}")
    
    return results

def calculate_power_budget_analysis(total_loss, inputs):
    """Calculate power budget and determine link safety"""
    tx_power = inputs['transmitter_power_dbm']
    rx_sensitivity = inputs['receiver_sensitivity_dbm']
    safety_margin = inputs['safety_margin_db']
    total_budget = tx_power - rx_sensitivity
    required_budget = total_loss + safety_margin
    power_margin = total_budget - required_budget
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

def render_navigation():
    st.sidebar.title("🔬 Navigation")
    pages = {
        'Home': '🏠',
        'AI Prediction': '🤖', 
        'Normal Calculation': '📊',
        'Model Evaluation': '📈',
        'Documentation': '📚'
    }
    st.sidebar.info(f"📍 Current: {st.session_state.page}")
    for page, icon in pages.items():
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
            # Get component losses using the enhanced function
            component_losses = predict_component_losses(inputs, models)
            total_component_loss = sum(component_losses.values())

            # Get quantile predictions using the enhanced function
            results = predict_quantile_models(inputs, models, selected_models)

            # Use median prediction as main prediction, fallback to component sum
            main_pred = results.get('Median (50%)', total_component_loss)

            # Rest of your existing code remains the same...
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
                    'Percentage': round((v / total_component_loss * 100), 1) if total_component_loss > 0 else 0,
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

# Documentation Page
def render_documentation_page():
    st.title("📚 Documentation")

    st.markdown("""
    # 📚 A.F.D.F.O Complete Documentation

## 🛰️ Executive Summary

**A.F.D.F.O** (AI-powered Fiber Deployment Feasibility & Optimization Tool) is a decision support and planning system that leverages machine learning to optimize fiber optic network deployment. Designed specifically for the needs of small ISPs in India, especially those contributing to government-backed initiatives like BharatNet, this tool provides accurate, component-level attenuation prediction, risk analysis, and planning recommendations.

It replaces error-prone manual calculations with an intelligent, transparent, and automated planning assistant—helping ISPs reduce costs, time delays, and deployment failures.

---

## 🔧 What This System Does

**A.F.D.F.O** is a smart planning assistant for fiber optic network deployment. It helps engineers and planners:

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

## 🔍 Understanding Attenuation and Its Impact

### What Is Attenuation?

Attenuation refers to the loss of signal strength as light travels through an optical fiber. It is typically measured in decibels (dB) and increases due to factors such as:

- Fiber length
- Connectors and splices
- Bends in the cable
- Environmental conditions
- Aging of the fiber

### Why It Matters for ISPs

Attenuation directly impacts:
- **Network reliability** (signal fails if too much loss occurs)
- **Design choices** (e.g., splitter type, fiber grade, amplifiers)
- **Budget and equipment planning** (e.g., additional components to compensate for loss)

Even small miscalculations can lead to:
- **Overbudgeting** (overestimating loss → unnecessary spend)
- **Network failure** (underestimating loss → signal issues and rework)
- **Delays** in rollout and increased customer dissatisfaction

---

## 🇮🇳 Current Practices in Small Indian ISPs

### Manual Calculation Workflow

Most small ISPs in India continue to use:
- Excel-based or paper-based methods
- Static formula-based estimation
- Heuristics or technician judgement for safety margins

### Why They Prefer Manual Over Advanced Tools

The primary reason small Indian ISPs avoid complex attenuation tools is **cost and practicality**. They operate on tight budgets and often deploy on small scales, so they don't find it justifiable to invest in high-cost enterprise software.

Manual methods, though time-consuming, are perceived as sufficient for their limited-scale needs. Many are willing to compensate for inaccuracies with extra time or material, as the cost of errors usually remains within acceptable bounds of their local budgets.

However, a key unresolved pain point still exists: **Overestimating or underestimating attenuation** often results in either wasted components or network performance issues, which—while not catastrophic—lead to avoidable overspending, extra trips, and customer complaints.

---

## ❌ Key Pain Points in the Existing Approach

| Issue | Impact |
|-------|---------|
| ❌ Inaccurate Attenuation Estimates | Over- or under-provisioning of components |
| ❌ Late Discovery of Power Mismatch | Redesign and rework during or post-deployment |
| ❌ Lack of Component Diagnostics | Inability to identify weak points in the fiber link |
| ❌ No Uncertainty Estimation | Risk goes unquantified, affecting confidence |
| ❌ Heavy Technician Dependence | High variability in results and knowledge retention |

These lead to:
- **Resource Wastage** (extra cable, wrong splitters, incorrect routing)
- **Time Loss** (multiple site visits and recalculations)
- **Financial Drain** (procurement, re-installation, labor, downtime)

---

## ⚙️ How A.F.D.F.O Works (High-Level)

1. **User Input**: Physical, mechanical, and environmental features of the fiber route are entered into the system.
2. **Feature Engineering**: The system creates additional informative variables by combining or transforming user inputs.
3. **Missing Data Handling**: If certain values (like bend radius) are not available, they are imputed using learned statistical relationships.
4. **Prediction**:
   - The model predicts the **total attenuation**
   - It also provides a **prediction range** (e.g., 3.2–3.8 dB) using quantile regression
   - It **separates the loss** into contributing components (e.g., 60% from fiber, 20% from connectors)
5. **Results Display**: The user receives a clear view of the total loss, where it comes from, and how confident the system is.

> All predictions are based on previously learned patterns from simulated deployment scenarios.

### System Workflow Diagram

The following diagram illustrates the complete A.F.D.F.O workflow, from traditional and AI-enhanced planning approaches:
    """)
    
    # Display the image here
    st.image("Architecture diagram.png", caption="A.F.D.F.O System Workflow - Traditional vs AI-Enhanced Planning", use_column_width=True)
    
    st.markdown("""
---

## 🚀 The A.F.D.F.O Advantage

### What the Tool Offers

- Predicts total and per-component attenuation using ML
- Estimates uncertainty bounds with quantile regression
- Recommends adjustments for:
  - Splitter ratios
  - Fiber route optimization
  - Better component selection
- Visualizes risk, loss breakdown, and margin forecasts
- Works efficiently in low-connectivity environments

### Workflow Comparison

| Step | Traditional Planning | A.F.D.F.O Planning |
|------|---------------------|---------------------|
| Input Design | Manual, static | Real-time + learned patterns |
| Loss Calculation | Fixed formulas | ML-based with confidence intervals |
| Risk Analysis | Not available | Built-in, data-driven |
| Optimization | Trial and error | Guided recommendations |
| Visual Tools | None | Real-time dashboards |
| Decision Support | Experience-driven | Transparent, explainable AI |

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

- **Connections**: 3 features
- **Engineered**: 5 features
- **Environmental**: 4 features
- **Interactions**: 5 features
- **Mechanical**: 3 features
- **Physical**: 4 features

---

## 📁 Dataset Overview

The model was trained on a diverse, high-quality dataset that includes:

- Thousands of deployment samples
- Cleaned and filtered using domain rules (no impossible or negative loss values)
- Completely based on simulated data covering:
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
|---------|-------|
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

| Metric | Value |
|--------|-------|
| MAE (Mean Abs Error) | 0.29 dB |
| RMSE | 0.65 dB |
| MAPE | 1.84% |
| R² Score | 0.9959 |
| Predictions within 0.5 dB | 85.96% |
| Predictions within 1 dB | 94.73% |
| Predictions within 3 dB | 99.28% |
| Safety Rule Accuracy | 99.24% |

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

## 🇮🇳 Application Areas in India

### Indian Market Fit

A.F.D.F.O is tailored for India's growing digital infrastructure and small-provider ecosystem. It aligns with the needs of:

- BharatNet FTTH Phase III deployments
- Small & regional ISPs in Tier 2 and Tier 3 cities
- Rural digitization initiatives under Digital India
- Smart Village projects, government campuses, or private institutions

### 📊 Business Opportunity

India's FTTH market is growing rapidly:

- Projected market size: ₹80,000+ crore by 2030
- Over 1000 small ISPs operate in regional markets
- Many serve 10–50 km loops, perfect for our model's target range
- Low access to advanced analytics presents a large tech enablement gap

### 💰 Why This Tool Fits Perfectly

- Low-cost alternative to expensive international tools
- Designed specifically for Indian field realities
- Offers fast, accurate, and uncertainty-aware planning in one tool
- Reduces unnecessary spending due to overestimation
- Prevents rework costs caused by underestimation
- Helps small ISPs scale confidently and compete with larger players

---

## 🧑‍💻 Deployment Notes for Engineers

- This model assumes all inputs are consistent and physics-aligned
- Inputs outside trained ranges **will still be accepted**, but predictions may have higher uncertainty
- For production deployment, monitoring prediction uncertainty is strongly advised
- Outputs are intended to **assist**, not replace, expert judgment in deployment planning

---

## 🆕 Prototype Version: `1.1.0`

**Changelog Highlights**
- ➕ Added uncertainty prediction (quantile regression)
- 🔍 Component-level loss breakdown
- 📐 Better feature engineering (interactions and polynomial terms)
- ⚙️ Optuna-based tuning for model robustness
- 🔎 SHAP explainability integration

---

## 🎯 Conclusion

Attenuation is a critical factor in the success of fiber network deployment. But existing tools are inaccessible for smaller Indian ISPs due to their cost, complexity, and lack of localization.

A.F.D.F.O is the first tool purpose-built for this gap, bringing:

- AI precision for fiber loss prediction
- Real-time, component-level insights
- Risk estimation and guided optimization

All in a simple, affordable, and explainable interface.

It empowers ISPs to build better networks, avoid costly mistakes, and expand rural and urban fiber in line with India's digital goals.

---

## 📬 Support

For issues or feature requests, please contact:
- **Phone Number**: 7094501353
- **Gmail**: rlalithkanna@gmail.com
    """)


def render_normal_calculation_page():
    st.title("📊 Normal Calculation - Traditional Methods")
    
    st.markdown("""
    <div class="nav-card">
        <h3>🧮 Traditional Fiber Optic Attenuation Calculation</h3>
        <p>This page provides industry-standard manual calculation methods for fiber optic link budgets. 
        Compare these traditional approaches with AI predictions to validate and understand the differences.</p>
    </div>
    """, unsafe_allow_html=True)

    # Get feature definitions for consistent input handling
    feature_defs = get_feature_definitions()
    
    # Create tabs for different calculation methods
    tab1, tab2, tab3, tab4 = st.tabs(["🔧 Manual Calculator", "📋 Industry Standards", "📊 Comparison Tool", "📚 Educational Guide"])
    
    with tab1:
        st.markdown("### 🔧 Traditional Link Budget Calculator")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### 📝 Input Parameters")
            
            # Collect inputs using the same structure as AI prediction
            inputs = {}
            
            # Physical Parameters
            with st.expander("🔗 Physical Parameters", expanded=True):
                inputs['length_km'] = st.number_input(
                    "Cable Length (km)", 
                    min_value=0.1, max_value=100.0, value=5.0, step=0.1
                )
                inputs['wavelength_nm'] = st.selectbox(
                    "Wavelength (nm)", 
                    [850, 1300, 1310, 1550], 
                    index=3
                )
                inputs['fiber_type'] = st.selectbox(
                    "Fiber Type", 
                    ["SMF (Single Mode)", "MMF (Multi Mode)"], 
                    index=0
                )
                inputs['fiber_subtype'] = st.selectbox(
                    "Fiber Grade", 
                    ["G.652.D", "G.657.A1", "G.657.A2", "G.657.B3", "G.655", "OM1", "OM2", "OM3", "OM4", "OM5"],
                    index=0
                )
            
            # Connection Parameters
            with st.expander("🔌 Connection Parameters", expanded=True):
                inputs['num_splices'] = st.number_input(
                    "Number of Splices", 
                    min_value=0, max_value=50, value=2, step=1
                )
                inputs['num_connectors'] = st.number_input(
                    "Number of Connectors", 
                    min_value=2, max_value=20, value=2, step=1
                )
                inputs['splitter_ratio'] = st.selectbox(
                    "Splitter Ratio (1:N)", 
                    [1, 2, 4, 8, 16, 32, 64], 
                    index=0
                )
            
            # Environmental & Installation
            with st.expander("🌡️ Environmental Factors", expanded=False):
                inputs['temperature_c'] = st.number_input(
                    "Operating Temperature (°C)", 
                    min_value=-40.0, max_value=85.0, value=25.0, step=5.0
                )
                inputs['installation_type'] = st.selectbox(
                    "Installation Type", 
                    ["Underground", "Aerial", "Indoor", "Duct"], 
                    index=0
                )
                inputs['age_years'] = st.number_input(
                    "Cable Age (years)", 
                    min_value=0, max_value=30, value=0, step=1
                )
            
            # Advanced Parameters
            with st.expander("⚙️ Advanced Parameters", expanded=False):
                inputs['bend_radius_mm'] = st.number_input(
                    "Minimum Bend Radius (mm)", 
                    min_value=5.0, max_value=100.0, value=15.0, step=1.0
                )
                inputs['num_bends'] = st.number_input(
                    "Number of Bends", 
                    min_value=0, max_value=20, value=4, step=1
                )
                inputs['safety_margin_db'] = st.number_input(
                    "Safety Margin (dB)", 
                    min_value=1.0, max_value=10.0, value=3.0, step=0.5
                )
            
            # Power Budget Parameters
            with st.expander("⚡ Power Budget", expanded=True):
                inputs['tx_power_dbm'] = st.number_input(
                    "Transmitter Power (dBm)", 
                    min_value=-10.0, max_value=10.0, value=0.0, step=0.1
                )
                inputs['rx_sensitivity_dbm'] = st.number_input(
                    "Receiver Sensitivity (dBm)", 
                    min_value=-40.0, max_value=-10.0, value=-25.0, step=0.1
                )
            
            calculate_button = st.button("🧮 Calculate Traditional Loss", type="primary", use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Traditional Calculation Results")
            
            if calculate_button:
                # Perform traditional calculations
                traditional_results = calculate_traditional_loss(inputs)
                
                # Display main results
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>📉 Total Loss</h3>
                        <h2>{traditional_results['total_loss']:.3f} dB</h2>
                        <p>Traditional Method</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_b:
                    power_budget = inputs['tx_power_dbm'] - inputs['rx_sensitivity_dbm']
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>⚡ Power Budget</h3>
                        <h2>{power_budget:.1f} dB</h2>
                        <p>Available</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_c:
                    margin = power_budget - traditional_results['total_loss'] - inputs['safety_margin_db']
                    margin_color = "green" if margin > 0 else "red"
                    st.markdown(f"""
                    <div class="metric-card" style="background: {margin_color};">
                        <h3>🎯 System Margin</h3>
                        <h2>{margin:.2f} dB</h2>
                        <p>{'PASS' if margin > 0 else 'FAIL'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Component breakdown
                st.markdown("#### 🔍 Loss Component Breakdown")
                
                components_df = pd.DataFrame([
                    {"Component": "Fiber Loss", "Loss (dB)": traditional_results['fiber_loss'], "Formula": f"{traditional_results['attenuation_coeff']:.2f} × {inputs['length_km']}"},
                    {"Component": "Splice Loss", "Loss (dB)": traditional_results['splice_loss'], "Formula": f"{inputs['num_splices']} × 0.05"},
                    {"Component": "Connector Loss", "Loss (dB)": traditional_results['connector_loss'], "Formula": f"{inputs['num_connectors']} × 0.25"},
                    {"Component": "Splitter Loss", "Loss (dB)": traditional_results['splitter_loss'], "Formula": f"10 × log10({inputs['splitter_ratio']})"},
                    {"Component": "Bend Loss", "Loss (dB)": traditional_results['bend_loss'], "Formula": "Calculated from bend parameters"},
                    {"Component": "Environmental Loss", "Loss (dB)": traditional_results['environmental_loss'], "Formula": "Temperature + Age factors"},
                    {"Component": "Total Loss", "Loss (dB)": traditional_results['total_loss'], "Formula": "Sum of all components"}
                ])
                
                st.dataframe(components_df, use_container_width=True)
                
                # Visualization
                fig = px.pie(
                    values=[traditional_results['fiber_loss'], traditional_results['splice_loss'], 
                           traditional_results['connector_loss'], traditional_results['splitter_loss'],
                           traditional_results['bend_loss'], traditional_results['environmental_loss']],
                    names=['Fiber Loss', 'Splice Loss', 'Connector Loss', 'Splitter Loss', 'Bend Loss', 'Environmental Loss'],
                    title="Traditional Loss Breakdown"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Power budget chart
                budget_fig = go.Figure()
                budget_fig.add_trace(go.Bar(
                    x=['Available Budget', 'Total Loss', 'Safety Margin', 'Remaining Margin'],
                    y=[power_budget, traditional_results['total_loss'], inputs['safety_margin_db'], max(0, margin)],
                    marker_color=['lightblue', 'orange', 'yellow', 'lightgreen' if margin > 0 else 'red']
                ))
                budget_fig.update_layout(title="Power Budget Analysis", yaxis_title="Power (dB)")
                st.plotly_chart(budget_fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 📋 Industry Standard Values & Formulas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔗 Fiber Attenuation Coefficients")
            fiber_specs = pd.DataFrame([
                {"Fiber Type": "SMF G.652.D", "1310nm (dB/km)": "≤ 0.35", "1550nm (dB/km)": "≤ 0.25"},
                {"Fiber Type": "SMF G.657.A1", "1310nm (dB/km)": "≤ 0.35", "1550nm (dB/km)": "≤ 0.25"},
                {"Fiber Type": "SMF G.657.A2", "1310nm (dB/km)": "≤ 0.35", "1550nm (dB/km)": "≤ 0.25"},
                {"Fiber Type": "MMF OM1", "850nm (dB/km)": "≤ 3.5", "1300nm (dB/km)": "≤ 1.5"},
                {"Fiber Type": "MMF OM2", "850nm (dB/km)": "≤ 3.5", "1300nm (dB/km)": "≤ 1.5"},
                {"Fiber Type": "MMF OM3", "850nm (dB/km)": "≤ 3.5", "1300nm (dB/km)": "≤ 1.5"},
                {"Fiber Type": "MMF OM4", "850nm (dB/km)": "≤ 3.5", "1300nm (dB/km)": "≤ 1.5"},
                {"Fiber Type": "MMF OM5", "850nm (dB/km)": "≤ 3.0", "1300nm (dB/km)": "≤ 1.5"}
            ])
            st.dataframe(fiber_specs, use_container_width=True)
            
            st.markdown("#### 🔌 Connection Loss Standards")
            connection_specs = pd.DataFrame([
                {"Component": "Fusion Splice", "Typical Loss (dB)": "0.05", "Max Loss (dB)": "0.10"},
                {"Component": "Mechanical Splice", "Typical Loss (dB)": "0.10", "Max Loss (dB)": "0.20"},
                {"Component": "SC/FC Connector", "Typical Loss (dB)": "0.25", "Max Loss (dB)": "0.50"},
                {"Component": "LC Connector", "Typical Loss (dB)": "0.20", "Max Loss (dB)": "0.40"},
                {"Component": "ST Connector", "Typical Loss (dB)": "0.30", "Max Loss (dB)": "0.60"}
            ])
            st.dataframe(connection_specs, use_container_width=True)
        
        with col2:
            st.markdown("#### 📐 Splitter Loss Values")
            splitter_specs = pd.DataFrame([
                {"Split Ratio": "1:2", "Loss (dB)": "3.5", "Theoretical (dB)": "3.01"},
                {"Split Ratio": "1:4", "Loss (dB)": "7.0", "Theoretical (dB)": "6.02"},
                {"Split Ratio": "1:8", "Loss (dB)": "10.5", "Theoretical (dB)": "9.03"},
                {"Split Ratio": "1:16", "Loss (dB)": "14.0", "Theoretical (dB)": "12.04"},
                {"Split Ratio": "1:32", "Loss (dB)": "17.5", "Theoretical (dB)": "15.05"},
                {"Split Ratio": "1:64", "Loss (dB)": "21.0", "Theoretical (dB)": "18.06"}
            ])
            st.dataframe(splitter_specs, use_container_width=True)
            
            st.markdown("#### 🧮 Key Formulas")
            st.markdown("""
            **Total Link Loss:**
            ```
            Total Loss = Fiber Loss + Splice Loss + Connector Loss + 
                        Splitter Loss + Bend Loss + Environmental Loss
            ```
            
            **Fiber Loss:**
            ```
            Fiber Loss = Attenuation Coefficient × Length
            ```
            
            **Splice Loss:**
            ```
            Splice Loss = Number of Splices × Loss per Splice
            ```
            
            **Connector Loss:**
            ```
            Connector Loss = Number of Connectors × Loss per Connector
            ```
            
            **Splitter Loss:**
            ```
            Splitter Loss = 10 × log10(Split Ratio)
            ```
            
            **Power Budget:**
            ```
            Available Budget = TX Power - RX Sensitivity
            Required Budget = Total Loss + Safety Margin
            System Margin = Available Budget - Required Budget
            ```
            """)
    
    with tab3:
        st.markdown("### 📊 AI vs Traditional Comparison Tool")
        
        st.info("🔍 Compare traditional calculations with AI predictions using the same input parameters")
        
        # Load AI models for comparison
        models = load_all_models()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### 🎛️ Comparison Parameters")
            
            # Quick input form for comparison
            comp_inputs = {}
            comp_inputs['length_km'] = st.slider("Length (km)", 0.5, 20.0, 5.0, 0.5)
            comp_inputs['wavelength_nm'] = st.selectbox("Wavelength", [1310, 1550], key="comp_wl")
            comp_inputs['fiber_type'] = st.selectbox("Fiber Type", ["SMF", "MMF"], key="comp_fiber")
            comp_inputs['num_splices'] = st.slider("Splices", 0, 10, 2)
            comp_inputs['num_connectors'] = st.slider("Connectors", 2, 10, 2)
            comp_inputs['splitter_ratio'] = st.selectbox("Splitter", [1, 2, 4, 8, 16], key="comp_split")
            
            # Set defaults for other required parameters
            comp_inputs.update({
                'fiber_subtype': 'G.652.D',
                'temperature_c': 25.0,
                'installation_type': 'Underground',
                'age_years': 2,
                'bend_radius_mm': 15.0,
                'num_bends': 4,
                'safety_margin_db': 3.0,
                'tx_power_dbm': 0.0,
                'rx_sensitivity_dbm': -25.0
            })
            
            compare_button = st.button("🔍 Compare Methods", type="primary", use_container_width=True)
        
        with col2:
            if compare_button:
                st.markdown("#### 📈 Comparison Results")
                
                # Calculate traditional
                traditional = calculate_traditional_loss(comp_inputs)
                
                # Calculate AI prediction (simplified)
                ai_inputs_formatted = format_inputs_for_ai(comp_inputs)
                ai_components = predict_component_losses(ai_inputs_formatted, models)
                ai_total = sum(ai_components.values())
                
                # Comparison metrics
                difference = ai_total - traditional['total_loss']
                percent_diff = (difference / traditional['total_loss']) * 100
                
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("🧮 Traditional", f"{traditional['total_loss']:.3f} dB")
                with col_b:
                    st.metric("🤖 AI Prediction", f"{ai_total:.3f} dB")
                with col_c:
                    st.metric("📊 Difference", f"{difference:.3f} dB", f"{percent_diff:.1f}%")
                
                # Component comparison
                comparison_df = pd.DataFrame({
                    'Component': ['Fiber', 'Splice', 'Connector', 'Splitter', 'Bend', 'Environmental', 'TOTAL'],
                    'Traditional (dB)': [
                        traditional['fiber_loss'],
                        traditional['splice_loss'], 
                        traditional['connector_loss'],
                        traditional['splitter_loss'],
                        traditional['bend_loss'],
                        traditional['environmental_loss'],
                        traditional['total_loss']
                    ],
                    'AI Prediction (dB)': [
                        ai_components.get('fiber_loss', 0),
                        ai_components.get('splice_loss', 0),
                        ai_components.get('connector_loss', 0),
                        ai_components.get('splitter_loss', 0),
                        ai_components.get('bend_loss', 0),
                        ai_components.get('environmental_loss', 0),
                        ai_total
                    ]
                })
                
                comparison_df['Difference (dB)'] = comparison_df['AI Prediction (dB)'] - comparison_df['Traditional (dB)']
                comparison_df['Difference (%)'] = (comparison_df['Difference (dB)'] / comparison_df['Traditional (dB)']) * 100
                
                st.dataframe(comparison_df.round(3), use_container_width=True)
                
                # Visualization
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=comparison_df['Component'][:-1],  # Exclude total
                    y=comparison_df['Traditional (dB)'][:-1],
                    name='Traditional',
                    marker_color='lightblue'
                ))
                fig.add_trace(go.Bar(
                    x=comparison_df['Component'][:-1],
                    y=comparison_df['AI Prediction (dB)'][:-1],
                    name='AI Prediction',
                    marker_color='lightcoral'
                ))
                fig.update_layout(
                    title="Component Loss Comparison: Traditional vs AI",
                    yaxis_title="Loss (dB)",
                    barmode='group'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("### 📚 Educational Guide - Understanding Fiber Loss")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 🔍 What is Fiber Optic Attenuation?
            
            **Attenuation** is the reduction in optical power as light travels through a fiber optic cable. It's measured in decibels (dB) and represents the ratio of input power to output power.
            
            **Formula:** `Attenuation (dB) = 10 × log₁₀(P_input / P_output)`
            
            #### 📊 Types of Losses
            
            **1. Intrinsic Losses (inherent to fiber):**
            - Rayleigh Scattering
            - Absorption (OH⁻ ions, impurities)
            - Material imperfections
            
            **2. Extrinsic Losses (installation-related):**
            - Bending losses (macro and micro)
            - Connection losses (splices, connectors)
            - Environmental factors
            
            **3. System Losses:**
            - Splitter losses
            - Wavelength division multiplexer (WDM) losses
            - Patch panel losses
            """)
        
        with col2:
            st.markdown("""
            #### 🧮 Traditional Calculation Process
            
            **Step 1: Gather Parameters**
            - Cable length and type
            - Number and type of connections
            - Environmental conditions
            - System requirements
            
            **Step 2: Apply Standard Values**
            - Use ITU-T recommendations
            - Apply manufacturer specifications
            - Include safety margins
            
            **Step 3: Calculate Each Component**
            ```python
            fiber_loss = length × attenuation_coefficient
            splice_loss = num_splices × 0.05  # typical
            connector_loss = num_connectors × 0.25  # typical
            splitter_loss = 10 × log10(split_ratio)
            ```
            
            **Step 4: Sum All Losses**
            ```python
            total_loss = fiber_loss + splice_loss + 
                        connector_loss + splitter_loss + 
                        bend_loss + environmental_loss
            ```
            
            **Step 5: Check Power Budget**
            ```python
            available_budget = tx_power - rx_sensitivity
            required_budget = total_loss + safety_margin
            margin = available_budget - required_budget
            ```
            """)
        
        st.markdown("#### 🎯 Best Practices for Link Budget Calculations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **📐 Design Phase:**
            - Use conservative attenuation values
            - Include 15-20% safety margin
            - Plan for future connections
            - Consider environmental extremes
            """)
        
        with col2:
            st.markdown("""
            **🔧 Installation Phase:**
            - Verify actual splice losses
            - Measure bend radii
            - Document connection types
            - Test with OTDR if available
            """)
        
        with col3:
            st.markdown("""
            **✅ Verification Phase:**
            - Measure end-to-end loss
            - Compare with calculations
            - Document deviations
            - Update future calculations
            """)
        
        # Interactive loss calculator widget
        st.markdown("#### 🧮 Quick Loss Calculator")
        
        calc_col1, calc_col2, calc_col3 = st.columns(3)
        
        with calc_col1:
            quick_length = st.number_input("Cable Length (km)", value=1.0, min_value=0.1, key="quick_len")
            quick_atten = st.number_input("Attenuation Coeff (dB/km)", value=0.25, min_value=0.1, key="quick_att")
        
        with calc_col2:
            quick_splices = st.number_input("Splices", value=0, min_value=0, key="quick_splice")
            quick_connectors = st.number_input("Connectors", value=2, min_value=0, key="quick_conn")
        
        with calc_col3:
            quick_split = st.selectbox("Splitter Ratio", [1, 2, 4, 8, 16, 32], key="quick_split_ratio")
            
            # Calculate quick result
            quick_fiber = quick_length * quick_atten
            quick_splice_loss = quick_splices * 0.05
            quick_conn_loss = quick_connectors * 0.25
            quick_split_loss = 10 * np.log10(quick_split) if quick_split > 1 else 0
            quick_total = quick_fiber + quick_splice_loss + quick_conn_loss + quick_split_loss
            
            st.metric("Quick Total Loss", f"{quick_total:.2f} dB")


def calculate_traditional_loss(inputs):
    """
    Calculate attenuation using traditional industry standard methods
    """
    # Initialize results dictionary
    results = {}
    
    # 1. Fiber Loss Calculation
    # Standard attenuation coefficients based on fiber type and wavelength
    fiber_type = inputs.get('fiber_type', 'SMF (Single Mode)')
    wavelength = inputs.get('wavelength_nm', 1550)
    length = inputs.get('length_km', 1.0)
    
    # Determine attenuation coefficient
    if 'SMF' in fiber_type or fiber_type == 'SMF':
        if wavelength == 1310:
            attenuation_coeff = 0.35  # dB/km
        elif wavelength == 1550:
            attenuation_coeff = 0.25  # dB/km
        else:
            attenuation_coeff = 0.30  # default for other wavelengths
    else:  # MMF
        if wavelength == 850:
            attenuation_coeff = 3.0   # dB/km
        elif wavelength == 1300:
            attenuation_coeff = 1.0   # dB/km
        else:
            attenuation_coeff = 2.0   # default
    
    results['attenuation_coeff'] = attenuation_coeff
    results['fiber_loss'] = length * attenuation_coeff
    
    # 2. Splice Loss Calculation
    num_splices = inputs.get('num_splices', 0)
    splice_loss_per_splice = 0.05  # dB per fusion splice (typical)
    results['splice_loss'] = num_splices * splice_loss_per_splice
    
    # 3. Connector Loss Calculation
    num_connectors = inputs.get('num_connectors', 2)
    connector_loss_per_connector = 0.25  # dB per connector (typical)
    results['connector_loss'] = num_connectors * connector_loss_per_connector
    
    # 4. Splitter Loss Calculation
    splitter_ratio = inputs.get('splitter_ratio', 1)
    if splitter_ratio > 1:
        results['splitter_loss'] = 10 * np.log10(splitter_ratio)
    else:
        results['splitter_loss'] = 0.0
    
    # 5. Bend Loss Calculation
    bend_radius = inputs.get('bend_radius_mm', 15.0)
    num_bends = inputs.get('num_bends', 0)
    
    # Simplified bend loss calculation
    if 'SMF' in fiber_type:
        critical_radius = 10.0  # mm for SMF
    else:
        critical_radius = 30.0  # mm for MMF
    
    if bend_radius < critical_radius:
        bend_loss_per_bend = 0.1 * (critical_radius / bend_radius)
    else:
        bend_loss_per_bend = 0.01  # minimal loss for proper bends
    
    results['bend_loss'] = num_bends * bend_loss_per_bend
    
    # 6. Environmental Loss Calculation
    temperature = inputs.get('temperature_c', 25.0)
    age_years = inputs.get('age_years', 0)
    installation_type = inputs.get('installation_type', 'Underground')
    
    # Temperature factor
    temp_factor = abs(temperature - 25) * 0.001  # 0.001 dB per degree deviation from 25°C
    
    # Age factor
    age_factor = age_years * 0.01  # 0.01 dB per year of aging
    
    # Installation factor
    installation_factors = {
        'Underground': 0.05,
        'Aerial': 0.10,
        'Indoor': 0.02,
        'Duct': 0.03
    }
    installation_factor = installation_factors.get(installation_type, 0.05)
    
    results['environmental_loss'] = temp_factor + age_factor + installation_factor
    
    # 7. Total Loss Calculation
    results['total_loss'] = (results['fiber_loss'] + 
                           results['splice_loss'] + 
                           results['connector_loss'] + 
                           results['splitter_loss'] + 
                           results['bend_loss'] + 
                           results['environmental_loss'])
    
    return results


def format_inputs_for_ai(inputs):
    """
    Format traditional calculation inputs for AI model prediction
    """
    ai_inputs = {}
    
    # Map traditional inputs to AI model expected inputs
    ai_inputs['length_km'] = inputs.get('length_km', 1.0)
    ai_inputs['wavelength_nm'] = inputs.get('wavelength_nm', 1550)
    ai_inputs['num_splices'] = inputs.get('num_splices', 0)
    ai_inputs['num_connectors'] = inputs.get('num_connectors', 2)
    ai_inputs['splitter_ratio'] = inputs.get('splitter_ratio', 1)
    
    # Convert fiber type
    fiber_type = inputs.get('fiber_type', 'SMF')
    if 'SMF' in fiber_type or fiber_type == 'SMF':
        ai_inputs['fiber_type_encoded'] = 'SMF'
    else:
        ai_inputs['fiber_type_encoded'] = 'MMF'
    
    ai_inputs['fiber_subtype_encoded'] = inputs.get('fiber_subtype', 'G.652.D')
    
    # Convert installation type
    installation_mapping = {
        'Underground': 'Underground',
        'Aerial': 'Aerial', 
        'Indoor': 'Indoor',
        'Duct': 'Building'
    }
    ai_inputs['installation_type_encoded'] = installation_mapping.get(
        inputs.get('installation_type', 'Underground'), 'Underground'
    )
    
    ai_inputs['cable_type_encoded'] = 'Loose Tube'  # default
    ai_inputs['temperature_C'] = inputs.get('temperature_c', 25.0)
    ai_inputs['humidity_percent'] = 50.0  # default
    ai_inputs['age_years'] = inputs.get('age_years', 0)
    ai_inputs['environmental_stress'] = 0.3  # default
    ai_inputs['bend_radius_mm'] = inputs.get('bend_radius_mm', 15.0)
    ai_inputs['num_bends'] = inputs.get('num_bends', 4)
    ai_inputs['transmitter_power_dbm'] = inputs.get('tx_power_dbm', 0.0)
    ai_inputs['receiver_sensitivity_dbm'] = inputs.get('rx_sensitivity_dbm', -25.0)
    ai_inputs['safety_margin_db'] = inputs.get('safety_margin_db', 3.0)
    
    return ai_inputs

# Main app logic
def main():
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'Home'
    
    # Add back to home button at the top
    if st.button("🏠 Back to Home"):
        st.session_state.page = 'Home'
        st.rerun()
    
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
