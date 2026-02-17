"""
streamlit_app.py - Canadian Smartphone Market Forecast Dashboard
Production-grade with 8 model support (SARIMAX, XGBoost, LightGBM, CatBoost, GP, Prophet×2, Ensemble)
Enhanced with live AI inference, professional reporting, comprehensive model analytics, and business metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
import os

# ================================================================
# PAGE CONFIG
# ================================================================

st.set_page_config(
    page_title="Canadian Smartphone Forecast 2026",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add outputs directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'outputs'))

# ================================================================
# LOAD AI ENGINE (Cached for Performance)
# ================================================================

@st.cache_resource
def load_ai_engine():
    """
    Initialize forecasting engine (cached in memory)
    """
    try:
        # Search for files in multiple locations
        base_dir = os.path.dirname(os.path.abspath(__file__))
        search_paths = [
            base_dir,
            os.path.join(base_dir, 'outputs'),
            '.',
            'outputs'
        ]
        
        for base_path in search_paths:
            core_path = os.path.join(base_path, 'app_core_data.pkl')
            models_path = os.path.join(base_path, 'app_models.pkl')
            config_path = os.path.join(base_path, 'app_feature_config.pkl')
            
            if all(os.path.exists(p) for p in [core_path, models_path, config_path]):
                # Import utils from this path
                sys.path.insert(0, os.path.abspath(base_path))
                
                try:
                    from utils import load_app_state, FeatureEngineer, ForecastEngine
                    
                    # Load frozen state
                    core, models, config = load_app_state(
                        core_path, models_path, config_path
                    )
                    
                    # Initialize components
                    engineer = FeatureEngineer(
                        core['df_historical'], 
                        config['brands']
                    )
                    
                    # Pass scalers to engine (required for SARIMAX/GP/ML models)
                    engine = ForecastEngine(
                        models['production_models'],
                        models['production_params'],
                        core['model_selection'],
                        core['df_historical'],
                        engineer,
                        core_data=core,
                        production_scalers=models.get('production_scalers', {})
                    )
                    
                    return engine, config, core
                    
                except ImportError:
                    continue
        
        # In case of failure, show error
        st.error("❌ Could not find frozen state files. Run the notebook freeze cell first.")
        st.stop()
        
    except Exception as e:
        st.error(f"❌ Error loading AI engine: {str(e)}")
        st.error("Make sure you've run the freeze cell in your notebook.")
        st.stop()

# Load engine (happens once and cached)
with st.spinner("🚀 Loading AI Engine..."):
    engine, config, core_data = load_ai_engine()
    brands = config['brands']

# ================================================================
# MODEL INFORMATION CONSTANTS
# ================================================================

MODEL_INFO = {
    'SARIMAX': {
        'full_name': 'Seasonal AutoRegressive Integrated Moving Average with eXogenous variables',
        'description': 'A statistical time series model that combines autoregression, differencing, and moving averages with seasonal patterns and external regressors.',
        'how_it_works': [
            'Captures temporal dependencies through autoregressive (AR) terms',
            'Uses differencing to make data stationary',
            'Incorporates moving average (MA) terms for error correction',
            'Explicitly models seasonal patterns (12-month cycles)',
            'Includes external variables like CPI and product launches'
        ],
        'strengths': [
            '✅ Excellent for data with strong seasonal patterns',
            '✅ Provides statistically valid confidence intervals',
            '✅ Interpretable coefficients and diagnostics',
            '✅ Handles autocorrelation naturally',
            '✅ Well-established theory and statistical foundation'
        ],
        'weaknesses': [
            '❌ Assumes linear relationships',
            '❌ Requires manual parameter tuning (p, d, q)',
            '❌ Sensitive to outliers',
            '❌ Struggles with complex non-linear patterns',
            '❌ Requires exog variable scaling for numerical stability'
        ],
        'best_for': 'Stable, seasonal time series with clear linear patterns'
    },
    'XGBoost': {
        'full_name': 'eXtreme Gradient Boosting',
        'description': 'An ensemble machine learning algorithm that builds multiple decision trees sequentially, where each tree corrects errors from previous ones.',
        'how_it_works': [
            'Builds trees iteratively to minimize prediction error',
            'Each tree learns from residuals of previous trees',
            'Uses gradient descent optimization with regularization',
            'Combines weak learners into strong predictor',
            'Automatically handles feature interactions'
        ],
        'strengths': [
            '✅ Captures complex non-linear relationships',
            '✅ Handles feature interactions automatically',
            '✅ Robust to outliers and missing data',
            '✅ Excellent predictive accuracy on structured data',
            '✅ Built-in feature importance analysis'
        ],
        'weaknesses': [
            '❌ "Black box" - limited interpretability',
            '❌ No native confidence intervals',
            '❌ Requires careful hyperparameter tuning',
            '❌ Can overfit without proper regularization',
            '❌ Computationally intensive for large datasets'
        ],
        'best_for': 'Complex patterns with many interacting features and non-linear relationships'
    },
    'LightGBM': {
        'full_name': 'Light Gradient Boosting Machine',
        'description': 'A fast, distributed gradient boosting framework optimized for efficiency and speed, using histogram-based learning.',
        'how_it_works': [
            'Uses histogram-based algorithms for faster training',
            'Employs leaf-wise tree growth (vs level-wise)',
            'Optimizes memory usage with gradient-based sampling',
            'Supports categorical features natively',
            'Parallel and GPU computing capable'
        ],
        'strengths': [
            '✅ Faster training than XGBoost (3-15x speedup)',
            '✅ Lower memory consumption',
            '✅ Handles large datasets efficiently (millions of rows)',
            '✅ High accuracy with less overfitting',
            '✅ Native categorical variable support'
        ],
        'weaknesses': [
            '❌ Can be sensitive to small datasets (<1000 samples)',
            '❌ Leaf-wise growth may overfit without tuning',
            '❌ Less interpretable than statistical models',
            '❌ No statistical confidence intervals',
            '❌ More hyperparameters than simpler models'
        ],
        'best_for': 'Large datasets requiring fast training with high accuracy'
    },
    'CatBoost': {
        'full_name': 'Categorical Boosting',
        'description': 'A gradient boosting algorithm optimized for categorical features using ordered boosting and symmetric trees.',
        'how_it_works': [
            'Uses ordered boosting to reduce prediction shift',
            'Handles categorical features natively without encoding',
            'Builds symmetric decision trees for efficiency',
            'Applies Bayesian bootstrap for robustness',
            'Automatically detects and handles feature interactions'
        ],
        'strengths': [
            '✅ Best-in-class categorical feature handling',
            '✅ Reduces overfitting via ordered boosting',
            '✅ Requires less hyperparameter tuning than XGBoost',
            '✅ Fast inference with symmetric trees',
            '✅ Built-in protection against target leakage'
        ],
        'weaknesses': [
            '❌ Slower training than LightGBM (2-5x)',
            '❌ No native confidence intervals',
            '❌ Black-box model (low interpretability)',
            '❌ Memory intensive for very large datasets',
            '❌ Newer library (since 2017) with smaller community'
        ],
        'best_for': 'Datasets with many categorical variables or complex interactions'
    },
    'GP': {
        'full_name': 'Gaussian Process Regression',
        'description': 'A probabilistic non-parametric Bayesian model that provides predictions with uncertainty estimates using kernel functions.',
        'how_it_works': [
            'Defines probability distribution over possible functions',
            'Uses kernel functions to measure similarity between data points',
            'Computes mean prediction and uncertainty (variance)',
            'Bayesian framework naturally quantifies prediction confidence',
            'Requires double-scaling: features (X) and target (y)'
        ],
        'strengths': [
            '✅ Native uncertainty quantification (true confidence intervals)',
            '✅ Excellent for small-medium datasets (<5000 samples)',
            '✅ Flexible kernel selection for different patterns',
            '✅ No overfitting with proper kernel choice',
            '✅ Probabilistic predictions with well-calibrated uncertainty'
        ],
        'weaknesses': [
            '❌ Computationally expensive (O(n³) complexity)',
            '❌ Requires both X and y scaling for numerical stability',
            '❌ Struggles with large datasets (>5,000 samples)',
            '❌ Sensitive to kernel and hyperparameter choice',
            '❌ Long training and inference time'
        ],
        'best_for': 'Small datasets where uncertainty quantification is critical'
    },
    'Prophet-Multi': {
        'full_name': 'Prophet (Multivariate)',
        'description': "Facebook's forecasting tool designed for business time series, decomposing trends, seasonality, and holidays with added regressors.",
        'how_it_works': [
            'Decomposes time series into trend, seasonality, and holidays',
            'Uses piecewise linear or logistic growth curves',
            'Automatically detects changepoints in trends',
            'Incorporates multiple seasonal patterns (yearly, monthly)',
            'Adds external regressors (CPI, product launches)'
        ],
        'strengths': [
            '✅ Robust to missing data and outliers',
            '✅ Automatically handles holidays and events',
            '✅ Intuitive hyperparameters (non-expert friendly)',
            '✅ Works well with irregular/gappy time series',
            '✅ Provides uncertainty intervals via Monte Carlo simulation'
        ],
        'weaknesses': [
            '❌ Struggles with short time series (<2 years)',
            '❌ May overfit changepoints on noisy data',
            '❌ Assumes additive or multiplicative decomposition',
            '❌ Less accurate for complex non-linear interactions',
            '❌ Slower than ARIMA for simple patterns'
        ],
        'best_for': 'Business time series with holidays, events, and irregular patterns'
    },
    'Prophet-Uni': {
        'full_name': 'Prophet (Univariate)',
        'description': 'Prophet model using only historical values without external regressors, focusing purely on time-based patterns.',
        'how_it_works': [
            'Same decomposition as Prophet-Multi (trend + seasonality)',
            'Relies only on historical time series data',
            'Automatically identifies seasonal patterns',
            'Detects trend changes without external inputs',
            'Uses Bayesian framework for uncertainty estimation'
        ],
        'strengths': [
            '✅ Simple - no external data required',
            '✅ Automatic feature detection (minimal tuning)',
            '✅ Good for data with strong inherent patterns',
            '✅ Robust uncertainty quantification',
            '✅ Fast and easy to implement'
        ],
        'weaknesses': [
            '❌ Ignores external factors (CPI, launches, events)',
            '❌ Lower accuracy than multivariate models',
            '❌ Can miss regime changes driven by external shocks',
            '❌ Not suitable for sparse or irregular data',
            '❌ May underperform in volatile markets'
        ],
        'best_for': 'Quick baseline forecasts when external data unavailable'
    },
    'Ensemble': {
        'full_name': 'Caruana Ensemble (Weighted Combination of 7 Models)',
        'description': 'Meta-model that combines predictions from 7 base models using optimized weights via Caruana greedy forward selection.',
        'how_it_works': [
            'Trains 7 different base models independently on full data',
            'Uses Caruana greedy selection to find optimal weights',
            'Each model contributes based on holdout validation performance',
            'Weighted average in log-space, then inverse-transformed',
            'Automatically balances model diversity and accuracy'
        ],
        'strengths': [
            '✅ More robust than any single model (variance reduction)',
            '✅ Reduces overfitting through model diversification',
            '✅ Captures both linear (SARIMAX, Prophet) and non-linear patterns (ML)',
            '✅ Better handles uncertainty and regime changes',
            '✅ Often achieves best overall accuracy across brands'
        ],
        'weaknesses': [
            '❌ Most computationally expensive (trains 7 models)',
            '❌ Hardest to interpret (complex decision path)',
            '❌ Requires all base models to work correctly',
            '❌ Complex deployment and maintenance',
            '❌ Slower inference time (7x single model)'
        ],
        'best_for': 'Maximum accuracy and robustness when computational cost acceptable'
    }
}

# ================================================================
# SIDEBAR: SCENARIO CONTROLS
# ================================================================

st.sidebar.title("🎛️ Forecast Controls")
st.sidebar.markdown("Adjust parameters to regenerate forecast with live AI inference.")

# Section 1: View Mode
st.sidebar.subheader("📊 View Mode")
view_mode = st.sidebar.radio(
    "Select Mode",
    options=["Baseline", "Custom Scenario"],
    help="Baseline: View original forecast. Scenario: Adjust parameters and re-run AI inference."
)

# Section 2: Economic Parameters (only if custom scenario)
if view_mode == "Custom Scenario":
    st.sidebar.markdown("---")
    st.sidebar.subheader("💰 Economic Adjustments")
    
    cpi_adjustment = st.sidebar.slider(
        "Inflation Rate (CPI)",
        min_value=-50.0,
        max_value=50.0,
        value=0.0,
        step=1.0,
        format="%+.1f%%",
        help="Adjust annual inflation rate. 0% = baseline forecast."
    ) / 100
else:
    cpi_adjustment = 0.0

# Section 3: Launch Calendar (only if custom scenario)
if view_mode == "Custom Scenario":
    st.sidebar.markdown("---")
    st.sidebar.subheader("🚀 Product Launch Calendar")
    st.sidebar.caption("Select launch months for each brand")
    
    launch_overrides = {}
    months_short = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for brand in brands:
        defaults = config['launch_defaults'].get(brand, [])
        
        selected = st.sidebar.multiselect(
            f"**{brand}**",
            options=list(range(1, 13)),
            format_func=lambda x: months_short[x-1],
            default=defaults,
            key=f"launch_{brand}"
        )
        
        if selected != defaults:
            launch_overrides[brand] = selected
else:
    launch_overrides = None

# Section 4: System Info
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ System Info")
st.sidebar.caption(f"**Forecast Period:** {core_data['df_future_baseline']['month'].min().strftime('%b %Y')} - {core_data['df_future_baseline']['month'].max().strftime('%b %Y')}")
st.sidebar.caption(f"**Generated:** {core_data['metadata']['generated']}")
st.sidebar.caption(f"**Engine Status:** {'🟢 Custom' if view_mode == 'Custom Scenario' else '🔵 Baseline'}")

# Check if GP models are used and scalers are available
gp_brands = [b for b in brands if core_data['model_selection'][b]['model'] == 'GP']
if gp_brands:
    has_scalers = all(b in engine.production_scalers for b in gp_brands)
    st.sidebar.caption(f"**GP Models:** {'✅ Scalers loaded' if has_scalers else '⚠️ Missing scalers'}")

# Check if SARIMAX models are used and have scalers
sarimax_brands = [b for b in brands if core_data['model_selection'][b]['model'] == 'SARIMAX']
if sarimax_brands:
    sarimax_with_exog = [
        b for b in sarimax_brands 
        if engine.production_params.get(b, {}).get('exog_cols')
    ]
    if sarimax_with_exog:
        has_sarimax_scalers = all(
            b in engine.production_scalers and engine.production_scalers[b] is not None
            for b in sarimax_with_exog
        )
        st.sidebar.caption(f"**SARIMAX Scalers:** {'✅ Loaded' if has_sarimax_scalers else '⚠️ Missing'}")

# ================================================================
# RUN AI INFERENCE (Live or Baseline)
# ================================================================

if view_mode == "Custom Scenario":
    # Show progress
    with st.spinner("⚡ Running AI inference with custom parameters..."):
        # Step 1: Create feature scenario
        actual_cpi_override = cpi_adjustment if abs(cpi_adjustment) > 0.001 else None
        
        # Check if launch_overrides actually differ from defaults
        actual_launch_overrides = {}
        if launch_overrides:
            for brand, months in launch_overrides.items():
                defaults = set(config['launch_defaults'].get(brand, []))
                if set(months) != defaults:
                    actual_launch_overrides[brand] = months
        
        # Only create new scenario if something actually changed
        if actual_cpi_override is not None or actual_launch_overrides:
            df_future_scenario = engine.feature_engineer.create_future_features(
                core_data['df_future_baseline'],
                launch_overrides=actual_launch_overrides if actual_launch_overrides else None,
                cpi_override=actual_cpi_override
            )
            
            # Step 2: Run live inference
            forecasts = engine.forecast_all_brands(df_future_scenario, actual_launch_overrides if actual_launch_overrides else None)
            scenario_active = True
        else:
            # No actual changes - use baseline
            df_future_scenario = core_data['df_future_baseline']
            forecasts = core_data['forecasts_baseline']
            scenario_active = False
else:
    # Use baseline forecasts
    df_future_scenario = core_data['df_future_baseline']
    forecasts = core_data['forecasts_baseline']
    scenario_active = False

# ================================================================
# MAIN HEADER & KPI CARDS
# ================================================================

st.title("📱 Canadian Smartphone Market 2026 Forecast")
st.caption("**8-Model AI System:** SARIMAX • XGBoost • LightGBM • CatBoost • GP • Prophet×2 • Ensemble")

if scenario_active:
    st.info(f"**Custom Scenario Active:** Inflation {cpi_adjustment:+.1%} • {sum(1 for b in actual_launch_overrides.values() if b) if actual_launch_overrides else 0} launch dates modified")
else:
    st.markdown("Viewing baseline forecast based on historical trends and planned product launches.")

st.markdown("---")

# ================================================================
# KPI CALCULATION (DYNAMIC ROLLING PERIOD)
# ================================================================

# 1. Determine Forecast Period
forecast_start = df_future_scenario['month'].min()
forecast_end = df_future_scenario['month'].max()
forecast_len = len(df_future_scenario)

# Labels for display 
fc_label = f"Forecast ({forecast_start.strftime('%b %y')} - {forecast_end.strftime('%b %y')})"
hist_label = f"Prior Period (Last {forecast_len} Mos)"

# 2. Calculate Volumes
total_forecast = sum(f['forecast'].sum() for f in forecasts.values())
# Compare vs exact same number of months from history
total_hist = sum(core_data['df_historical'][f'{b}_units'].tail(forecast_len).sum() for b in brands)

# 3. Calculate Growth & Stats
yoy_growth = ((total_forecast - total_hist) / total_hist) * 100 if total_hist > 0 else 0
market_leader = max(brands, key=lambda b: forecasts[b]['forecast'].sum())
avg_monthly = total_forecast / forecast_len

# 4. Scenario Delta
if scenario_active:
    baseline_total = sum(core_data['forecasts_baseline'][b]['forecast'].sum() for b in brands)
    scenario_diff = total_forecast - baseline_total
    scenario_diff_pct = (scenario_diff / baseline_total) * 100
else:
    scenario_diff = 0
    scenario_diff_pct = 0

# Display KPI cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Forecast Vol",
        value=f"{total_forecast:,.0f}",
        delta=f"{scenario_diff:+,.0f}" if scenario_active else None,
        help=f"Total units for {fc_label}"
    )

with col2:
    st.metric(
        label="Period Growth",
        value=f"{yoy_growth:+.1f}%",
        delta=f"{total_forecast - total_hist:+,.0f} units",
        help=f"Growth vs {hist_label}"
    )

# Calculate leader's historical volume
leader_hist = core_data['df_historical'][f'{market_leader}_units'].tail(forecast_len).sum()
leader_growth = forecasts[market_leader]['forecast'].sum() - leader_hist
with col3:
    st.metric(
        label="Market Leader",
        value=market_leader,
        delta=f"{leader_growth:,.0f} units", 
        help="Brand with highest forecast volume"
    )

with col4:
    st.metric(
        label="Avg Monthly",
        value=f"{avg_monthly:,.0f}",
        delta=f"{scenario_diff_pct:+.1f}%" if scenario_active else None,
        help="Average units per month"
    )

st.markdown("---")

# ================================================================
# MAIN TABS 
# ================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Forecast View", 
    "🔍 Brand Breakdown", 
    "🤖 Model Performance",
    "💼 Business Insights",  
    "🔬 Methodology"  
])

# Color scheme
colors = {
    'Apple': '#007AFF',
    'Samsung': '#1428A0',
    'Google': '#4285F4',
    'Motorola': '#E8710A'
}

# ================================================================
# TAB 1: FORECAST VIEW
# ================================================================

with tab1:
    st.subheader("Market Forecast: Historical + 2026")
    
    # Prepare historical data (last 18 months for context)
    df_hist_chart = core_data['df_historical'].tail(18).copy()
    
    # Create combined chart
    fig = go.Figure()
    
    for brand in brands:
        # Historical line (semi-transparent)
        fig.add_trace(go.Scatter(
            x=df_hist_chart['month'],
            y=df_hist_chart[f'{brand}_units'],
            name=f'{brand} (Historical)',
            line=dict(color=colors[brand], width=1.5, dash='dot'),
            opacity=0.4,
            legendgroup=brand,
            showlegend=True,
            hovertemplate='<b>%{x|%b %Y}</b><br>%{y:,.0f} units<extra></extra>'
        ))
        
        # Bridge line
        last_hist_date = df_hist_chart['month'].iloc[-1]
        last_hist_value = df_hist_chart[f'{brand}_units'].iloc[-1]
        first_forecast_date = df_future_scenario['month'].iloc[0]
        first_forecast_value = forecasts[brand]['forecast'][0]
        
        fig.add_trace(go.Scatter(
            x=[last_hist_date, first_forecast_date],
            y=[last_hist_value, first_forecast_value],
            line=dict(color=colors[brand], width=2, dash='dot'),
            mode='lines',
            showlegend=False,
            legendgroup=brand,
            hoverinfo='skip'
        ))
        
        # Baseline ghost line (only in custom scenario)
        if scenario_active:
            fig.add_trace(go.Scatter(
                x=df_future_scenario['month'],
                y=core_data['forecasts_baseline'][brand]['forecast'],
                name=f"{brand} (Baseline)",
                line=dict(color=colors[brand], width=1, dash='dot'),
                opacity=0.3,
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Forecast line (bold)
        fig.add_trace(go.Scatter(
            x=df_future_scenario['month'],
            y=forecasts[brand]['forecast'],
            name=brand,
            line=dict(color=colors[brand], width=3),
            mode='lines+markers',
            marker=dict(size=6),
            legendgroup=brand,
            showlegend=False,
            hovertemplate='<b>%{x|%b %Y}</b><br>%{y:,.0f} units (forecast)<extra></extra>'
        ))
        
        # Confidence intervals
        if 'lower_bound' in forecasts[brand] and 'upper_bound' in forecasts[brand]:
            fig.add_trace(go.Scatter(
                x=df_future_scenario['month'].tolist() + df_future_scenario['month'].tolist()[::-1],
                y=forecasts[brand]['upper_bound'].tolist() + forecasts[brand]['lower_bound'].tolist()[::-1],
                fill='toself',
                fillcolor=colors[brand],
                opacity=0.15,
                line=dict(width=0),
                showlegend=False,
                legendgroup=brand,
                hoverinfo='skip'
            ))
    
    # Add forecast start line
    forecast_start = df_future_scenario['month'].min()
    forecast_start_numeric = forecast_start.timestamp() * 1000
    
    fig.add_vline(
        x=forecast_start_numeric,
        line_dash="dash",
        line_color="gray",
        line_width=2,
        annotation_text="← Forecast Start",
        annotation_position="top left"
    )
    
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Units",
        hovermode='x unified',
        height=600,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary table below chart
    st.markdown("---")
    st.subheader(f"📋 Forecast Summary: {fc_label}")
    
    # Build summary data
    summary_data = []
    
    for brand in brands:
        forecast_vals = forecasts[brand]['forecast']
        total_fc_brand = forecast_vals.sum()
        avg_monthly_brand = forecast_vals.mean()
        
        # Peak detection
        peak_idx = forecast_vals.argmax()
        peak_month = df_future_scenario.iloc[peak_idx]['month'].strftime('%B')
        peak_value = forecast_vals[peak_idx]
        
        # Historical comparison (Last N months)
        total_hist_brand = core_data['df_historical'][f'{brand}_units'].tail(forecast_len).sum()
        yoy_brand = ((total_fc_brand - total_hist_brand) / total_hist_brand) * 100 if total_hist_brand > 0 else 0
        
        # Market share
        market_share = (total_fc_brand / total_forecast) * 100
        
        summary_data.append({
            'Brand': brand,
            'Prior Period': f"{total_hist_brand:,.0f}",
            'Forecast': f"{total_fc_brand:,.0f}",
            'Growth': f"{yoy_brand:+.1f}%",
            'Share': f"{market_share:.1f}%",
            'Avg Mo.': f"{avg_monthly_brand:,.0f}",
            'Peak': f"{peak_month} ({peak_value:,.0f})"
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    st.dataframe(
        df_summary,
        use_container_width=True,
        hide_index=True
    )

# ================================================================
# TAB 2: BRAND BREAKDOWN
# ================================================================

with tab2:
    st.subheader("Individual Brand Forecasts")
    
    # Individual brand charts in 2x2 grid
    col_left, col_right = st.columns(2)
    
    for idx, brand in enumerate(brands):
        with col_left if idx % 2 == 0 else col_right:
            # Check if forecast used fallback
            is_fallback = forecasts[brand].get('fallback', False)
            has_error = 'error' in forecasts[brand]
            has_ci = forecasts[brand].get('confidence_intervals', False)
            
            # Status indicator
            if has_error:
                status = "⚠️ Fallback (Simulation)"
                status_color = "orange"
            elif is_fallback:
                status = "⚙️ Simulation Mode"
                status_color = "blue"
            else:
                model_type = core_data['model_selection'][brand]['model']
                if has_ci:
                    status = f"✅ {model_type} (with CI)"
                    status_color = "green"
                else:
                    status = f"✅ {model_type}"
                    status_color = "green"
            
            st.markdown(f"**{brand}** <span style='color:{status_color}; font-size:12px'>({status})</span>", unsafe_allow_html=True)
            
            fig_brand = go.Figure()
            
            # Forecast line
            fig_brand.add_trace(go.Scatter(
                x=df_future_scenario['month'],
                y=forecasts[brand]['forecast'],
                name='Forecast',
                line=dict(color=colors[brand], width=3),
                mode='lines+markers'
            ))
            
            # Confidence intervals
            if 'lower_bound' in forecasts[brand]:
                fig_brand.add_trace(go.Scatter(
                    x=df_future_scenario['month'].tolist() + df_future_scenario['month'].tolist()[::-1],
                    y=forecasts[brand]['upper_bound'].tolist() + forecasts[brand]['lower_bound'].tolist()[::-1],
                    fill='toself',
                    fillcolor=colors[brand],
                    opacity=0.2,
                    line=dict(width=0),
                    name='80% CI' if has_ci else 'Range',
                    hoverinfo='skip'
                ))
            
            fig_brand.update_layout(
                height=300,
                template='plotly_white',
                showlegend=False,
                margin=dict(l=0, r=0, t=10, b=0)
            )
            
            st.plotly_chart(fig_brand, use_container_width=True)

    # ============================================================
    # DETAILED FORECAST BREAKDOWN 
    # ============================================================
    
    st.markdown("---")
    st.subheader("📊 Detailed Forecast Breakdown")
    
    # Brand selector
    selected_brand = st.selectbox(
        "Select Brand for Detailed View",
        options=brands,
        key="detailed_forecast_brand"
    )
    
    # Build detailed table for selected brand
    if selected_brand in forecasts:
        f = forecasts[selected_brand]
        
        df_detail = df_future_scenario[['month']].copy()
        df_detail['Month'] = df_detail['month'].dt.strftime('%B %Y')
        df_detail['Forecast'] = f['forecast'].round(0).astype(int)
        
        # Add confidence/uncertainty intervals if available
        has_ci = f.get('confidence_intervals', False)
        interval_type = "80% Confidence Interval" if has_ci else "Estimated Range (±15%)"
        
        df_detail['Lower Bound'] = f.get('lower_bound', f['forecast'] * 0.85).round(0).astype(int)
        df_detail['Upper Bound'] = f.get('upper_bound', f['forecast'] * 1.15).round(0).astype(int)
        
        # Calculate interval width
        df_detail['Uncertainty Width'] = (df_detail['Upper Bound'] - df_detail['Lower Bound']).astype(int)
        df_detail['Uncertainty %'] = ((df_detail['Uncertainty Width'] / df_detail['Forecast']) * 100).round(1)
        
        # Drop datetime column
        df_detail = df_detail.drop('month', axis=1)
        
        # Display info about interval type
        model_type = core_data['model_selection'][selected_brand]['model']
        
        if has_ci:
            st.info(f"""
            **{selected_brand}** uses **{model_type}** which provides statistical confidence intervals.
            - **80% Confidence Interval**: There's an 80% probability actual sales will fall within this range.
            - Models with true CI: SARIMAX, Prophet, Gaussian Process
            """)
        else:
            st.info(f"""
            **{selected_brand}** uses **{model_type}** which provides estimated uncertainty ranges.
            - **Estimated Range**: ±15% around forecast (heuristic, not statistically derived).
            - For tighter bounds, consider models with native uncertainty quantification.
            """)
        
        # Display table
        st.dataframe(
            df_detail,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Month": st.column_config.TextColumn("Month", width="medium"),
                "Forecast": st.column_config.NumberColumn("Forecast", format="%d"),
                "Lower Bound": st.column_config.NumberColumn("Lower Bound", format="%d"),
                "Upper Bound": st.column_config.NumberColumn("Upper Bound", format="%d"),
                "Uncertainty Width": st.column_config.NumberColumn("Range Width", format="%d"),
                "Uncertainty %": st.column_config.NumberColumn("Range %", format="%.1f%%")
            }
        )
        
        # Add summary statistics
        st.caption(f"""
        **Summary Statistics for {selected_brand}:**
        - Average Monthly Forecast: {df_detail['Forecast'].mean():,.0f} units
        - Average Uncertainty Range: ±{df_detail['Uncertainty Width'].mean():,.0f} units ({df_detail['Uncertainty %'].mean():.1f}%)
        - Total Annual Forecast: {df_detail['Forecast'].sum():,.0f} units
        - Total Uncertainty Exposure: {df_detail['Lower Bound'].sum():,.0f} - {df_detail['Upper Bound'].sum():,.0f} units
        """)

    # Monthly breakdown
    with st.expander("📅 View Monthly Breakdown"):
        st.subheader("Detailed Monthly Forecast")
        
        # Create monthly table
        df_monthly = df_future_scenario[['month']].copy()
        df_monthly['Month'] = df_monthly['month'].dt.strftime('%B %Y')
        
        for brand in brands:
            df_monthly[f'{brand}'] = forecasts[brand]['forecast'].round(0).astype(int)
        
        # Add total column
        df_monthly['Total Market'] = df_monthly[brands].sum(axis=1)
        
        # Drop datetime column
        df_monthly = df_monthly.drop('month', axis=1)
        
        st.dataframe(
            df_monthly,
            use_container_width=True,
            hide_index=True
        )

# ================================================================
# TAB 3: MODEL PERFORMANCE
# ================================================================

with tab3:
    st.subheader("🤖 Model Performance & Selection")
    
    st.markdown("""
    This section provides detailed insights into the 8 AI models evaluated, 
    their performance metrics, and why each model was selected for each brand.
    """)
    
    # ============================================================
    # Section 1: Model Overview
    # ============================================================
    
    st.markdown("---")
    st.subheader("📚 Model Types Used")
    
    # Get unique models
    unique_models = list(set(core_data['model_selection'][b]['model'] for b in brands))
    
    st.markdown(f"""
    This forecasting system evaluated **8 different model types** and 
    selected the best performer for each brand. Currently using **{len(unique_models)} 
    different model types** across all brands.
    
    **8 Models Evaluated:** SARIMAX • XGBoost • LightGBM • CatBoost • Gaussian Process • Prophet-Multi • Prophet-Uni • Ensemble
    """)
    
    # Model introduction cards
    st.markdown("### 📚 Glossary of All Evaluated Models")

    # Iterate through ALL defined models in MODEL_INFO
    for model_name in sorted(MODEL_INFO.keys()):
        info = MODEL_INFO[model_name]
        
        # Add a visual tag if this model is currently active
        is_active = model_name in unique_models
        active_tag = " ✅ (Active)" if is_active else ""
        
        with st.expander(f"📖 {info['full_name']}{active_tag}", expanded=False):
            st.markdown(f"**Description:** {info['description']}")
            
            col_how, col_pros, col_cons = st.columns(3)
            
            with col_how:
                st.markdown("**How It Works:**")
                for item in info['how_it_works']:
                    st.markdown(f"- {item}")
            
            with col_pros:
                st.markdown("**Strengths:**")
                for item in info['strengths']:
                    st.markdown(item)
            
            with col_cons:
                st.markdown("**Weaknesses:**")
                for item in info['weaknesses']:
                    st.markdown(item)
            
            st.info(f"**Best Used For:** {info['best_for']}")
    
    # ============================================================
    # Section 2: Performance Metrics Table
    # ============================================================
    
    st.markdown("---")
    st.subheader("📊 Model Selection & Performance")
    
    # Build performance table
    perf_data = []
    
    for brand in brands:
        model_type = core_data['model_selection'][brand]['model']
        holdout_mape = core_data['model_selection'][brand].get('holdout_mape', 0)
        cv_mape = core_data['model_selection'][brand].get('cv_mape', 0)
        r2 = core_data['model_selection'][brand].get('r2', 0)
        
        perf_data.append({
            'Brand': brand,
            'Selected Model': model_type,
            'Holdout MAPE': f"{holdout_mape:.2f}%",
            'CV MAPE': f"{cv_mape:.2f}%",
            'R² Score': f"{r2:.3f}",
            'MAPE_numeric': holdout_mape  # For sorting/charting
        })
    
    df_perf = pd.DataFrame(perf_data)
    
    st.dataframe(
        df_perf[['Brand', 'Selected Model', 'Holdout MAPE', 'CV MAPE', 'R² Score']],
        use_container_width=True,
        hide_index=True
    )
    
    st.caption("""
    **Metrics Explained:**
    - **Holdout MAPE**: Mean Absolute Percentage Error on last 12 months (lower is better)
    - **CV MAPE**: Cross-validation MAPE across multiple folds (measures stability)
    - **R² Score**: Proportion of variance explained by model (higher is better, max 1.0)
    """)
    
    # ============================================================
    # Section 3: MAPE Comparison Chart
    # ============================================================
    
    st.markdown("---")
    st.subheader("📉 Accuracy Comparison by Brand")
    
    col_chart, col_insight = st.columns([2, 1])
    
    with col_chart:
        # Create bar chart
        fig_mape = go.Figure()
        
        fig_mape.add_trace(go.Bar(
            x=df_perf['Brand'],
            y=df_perf['MAPE_numeric'],
            marker_color=[colors.get(b, 'gray') for b in df_perf['Brand']],
            text=df_perf['Holdout MAPE'],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>MAPE: %{y:.2f}%<extra></extra>'
        ))
        
        fig_mape.update_layout(
            title="Holdout MAPE by Brand (Lower = Better)",
            xaxis_title="Brand",
            yaxis_title="MAPE (%)",
            height=400,
            template='plotly_white',
            showlegend=False,
            yaxis=dict(range=[0, 25])
        )
        
        # Reference line for target MAPE at industry-acceptable threshold (15%)
        fig_mape.add_hline(
            y=15,
            line_dash="dash",
            line_color="green",
            annotation_text="Target (<15%)",
            annotation_position="right"
        )
        
        st.plotly_chart(fig_mape, use_container_width=True)
    
    with col_insight:
        st.markdown("**Performance Insights:**")
        
        best_brand = df_perf.loc[df_perf['MAPE_numeric'].idxmin(), 'Brand']
        best_mape = df_perf['MAPE_numeric'].min()
        
        worst_brand = df_perf.loc[df_perf['MAPE_numeric'].idxmax(), 'Brand']
        worst_mape = df_perf['MAPE_numeric'].max()
        
        avg_mape = df_perf['MAPE_numeric'].mean()
        
        st.metric("Best Accuracy", best_brand, f"{best_mape:.1f}%")
        st.metric("Avg MAPE", f"{avg_mape:.1f}%")
        st.metric("Highest Variance", worst_brand, f"{worst_mape:.1f}%")
        
        # Dynamic quality assessment
        if avg_mape < 10:
            quality = "Excellent"
        elif avg_mape < 15:
            quality = "Very Good"
        else:
            quality = "Good"
        
        st.info(f"**Overall Quality:** {quality} (Avg MAPE: {avg_mape:.1f}%)")
    
    # ============================================================
    # Section 4: Individual Brand Model Selection Rationale
    # ============================================================
    
    st.markdown("---")
    st.subheader("🔍 Why Each Model Was Selected")
    
    for brand in brands:
        model_type = core_data['model_selection'][brand]['model']
        holdout_mape = core_data['model_selection'][brand].get('holdout_mape', 0)
        cv_mape = core_data['model_selection'][brand].get('cv_mape', 0)
        r2 = core_data['model_selection'][brand].get('r2', 0)
        
        with st.expander(f"**{brand}** - {model_type}", expanded=False):
            col_metric, col_reason = st.columns([1, 2])
            
            with col_metric:
                st.metric("Holdout MAPE", f"{holdout_mape:.2f}%")
                st.metric("CV MAPE", f"{cv_mape:.2f}%")
                st.metric("R² Score", f"{r2:.3f}")
            
            with col_reason:
                st.markdown("**Why This Model Won:**")
                
                # Generate dynamic rationale based on model type
                if model_type == 'SARIMAX':
                    st.markdown(f"""
                    - {brand} exhibits **strong seasonal patterns** that SARIMAX captures well
                    - Time series is **relatively stable** with clear trends
                    - External variables (CPI, launches) have **linear relationships** with demand
                    - Provides **statistically valid confidence intervals** for risk assessment
                    """)
                
                elif model_type in ['XGBoost', 'LightGBM']:
                    st.markdown(f"""
                    - {brand} data shows **complex non-linear patterns** that tree-based models handle better
                    - **Multiple feature interactions** (launch timing × seasonality × CPI)
                    - Outperformed linear models on validation data
                    - Captures **sudden market shifts** more effectively than statistical models
                    """)
                
                elif model_type == 'CatBoost':
                    st.markdown(f"""
                    - {brand} benefits from **ordered boosting** which reduces prediction shift
                    - **Native categorical feature handling** (month, quarter) improves accuracy
                    - Less overfitting than standard gradient boosting
                    - Achieved **{holdout_mape:.2f}% MAPE** on holdout set
                    """)
                
                elif model_type == 'GP':
                    st.markdown(f"""
                    - {brand} requires **uncertainty quantification** for risk management
                    - **Small-medium dataset** size suits GP's O(n³) complexity
                    - Provides **probabilistic predictions** with well-calibrated confidence intervals
                    - Kernel selection captures {brand}'s specific temporal patterns
                    """)
                
                elif model_type in ['Prophet-Multi', 'Prophet-Uni']:
                    st.markdown(f"""
                    - {brand} benefits from **automatic holiday/event detection**
                    - Prophet's **changepoint detection** captures market regime shifts
                    - Robust to **outliers and missing data** in historical record
                    - Provides **uncertainty quantification** through Bayesian framework
                    """)
                
                elif model_type == 'Ensemble':
                    st.markdown(f"""
                    - {brand} is **most challenging to forecast** - no single model dominates
                    - Ensemble **reduces overfitting** by combining 7 diverse model types
                    - Achieves **{holdout_mape:.2f}% MAPE** vs higher error for best single model
                    - Provides **maximum robustness** against model assumptions
                    """)
                
                st.markdown("**Model Selection Criteria:**")
                st.markdown("""
                Models were ranked using weighted scoring:
                - 60% Holdout MAPE (last 12 months)
                - 25% Cross-validation MAPE (stability)
                - 15% R² Score (variance explained)
                """)

# ================================================================
# TAB 4: BUSINESS INSIGHTS
# ================================================================

with tab4:
    st.subheader("💼 Business Impact & Risk Analysis")
    
    st.markdown("""
    This section translates forecast uncertainty into actionable business metrics:
    overstock/stockout risks, forecast bias, directional accuracy, and performance vs. simple baseline methods.
    """)
    
    # Check if enhanced metrics are available
    has_enhanced = 'enhanced_metrics' in core_data
    has_baseline_comp = 'baseline_comparison' in core_data
    has_business_impact = 'business_impact' in core_data
    
    if not (has_enhanced or has_baseline_comp or has_business_impact):
        st.warning("""
        ⚠️ **Enhanced metrics not available** 
            - This section relies on additional performance analyses that are not currently present in the data.
            """)
    else:
        # ============================================================
        # Section 1: Enhanced Metrics (Bias & Directional Accuracy)
        # ============================================================
        
        if has_enhanced:
            st.markdown("---")
            st.subheader("📊 Forecast Quality Metrics")
            
            enhanced_data = []
            for brand in brands:
                if brand in core_data['enhanced_metrics']:
                    metrics = core_data['enhanced_metrics'][brand]
                    
                    # Industry-standard bias assessment (12-month horizon, consumer electronics)
                    bias_val = metrics.get('bias', 0)
                    if abs(bias_val) < 8:
                        bias_status = "✅ Within Target"
                        bias_color = "green"
                    elif abs(bias_val) < 15:
                        bias_status = "✓ Acceptable"
                        bias_color = "blue"
                    else:
                        bias_status = "⚠️ Review Needed"
                        bias_color = "orange"
                    
                    # Industry-standard directional accuracy (trend prediction)
                    dir_acc = metrics.get('directional_accuracy', 0) * 100
                    if dir_acc >= 80:
                        dir_status = "✅ Strong"
                        dir_color = "green"
                    elif dir_acc >= 70:
                        dir_status = "✓ Good"
                        dir_color = "blue"
                    elif dir_acc >= 60:
                        dir_status = "🟡 Marginal"
                        dir_color = "blue"
                    else:
                        dir_status = "⚠️ Unreliable"
                        dir_color = "orange"
                    
                    enhanced_data.append({
                        'Brand': brand,
                        'MAPE': f"{metrics.get('mape', 0):.2f}%",
                        'Bias': f"{bias_val:+.2f}%",
                        'Bias Assessment': bias_status,
                        'Dir. Accuracy': f"{dir_acc:.1f}%",
                        'Dir. Assessment': dir_status
                    })
            
            if enhanced_data:
                df_enhanced = pd.DataFrame(enhanced_data)
                st.dataframe(
                    df_enhanced[['Brand', 'MAPE', 'Bias', 'Bias Assessment', 'Dir. Accuracy', 'Dir. Assessment']],
                    use_container_width=True,
                    hide_index=True
                )
                
                st.caption("""
                **Metrics Explained:**
                - **MAPE**: Overall forecast accuracy (lower is better)
                - **Bias**: Systematic over/underforecasting (±8% target for 12-month horizon)
                - **Directional Accuracy**: % of periods where growth/decline direction was correct (80%+ is strong)
        
                **Industry Context:** Consumer electronics with 12-month horizon typically shows ±5-15% bias. 
                Values within ±8% indicate well-calibrated models. Directional accuracy above 80% demonstrates 
                strong trend-capture capability, while above 70% is good. Above 60% is marginal but still provide value for inventory planning. Values below 60% suggest unreliable directional signals, which may require model refinement or additional features to capture market dynamics.
                """)
        
        # ============================================================
        # Section 2: Baseline Comparison (ML vs Simple Methods)
        # ============================================================
        
        if has_baseline_comp:
            st.markdown("---")
            st.subheader("🎯 ML vs Baseline Methods")
            
            st.markdown("""
            Comparing sophisticated ML models against simple baseline methods validates the value 
            of advanced forecasting vs. naive approaches (Same Period Last Year, Moving Averages).
            """)
            
            baseline_data = []
            for brand in brands:
                if brand in core_data['baseline_comparison']:
                    comp = core_data['baseline_comparison'][brand]
                    
                    baseline_data.append({
                        'Brand': brand,
                        'ML MAPE': f"{comp['ml_mape']:.2f}%",
                        'Best Baseline': f"{comp['baseline_mape']:.2f}% ({comp['baseline_method'].upper()})",
                        'ML Advantage': f"{comp['improvement_pct']:+.1f}%",
                        'Assessment': comp['verdict']
                    })
            
            if baseline_data:
                df_baseline = pd.DataFrame(baseline_data)
                st.dataframe(df_baseline, use_container_width=True, hide_index=True)
                
                # Overall summary if available
                if 'overall' in core_data['baseline_comparison']:
                    overall = core_data['baseline_comparison']['overall']
                    
                    col_b1, col_b2, col_b3 = st.columns(3)
                    
                    with col_b1:
                        st.metric(
                            "Portfolio ML MAPE",
                            f"{overall['ml_mape']:.2f}%"
                        )
                    
                    with col_b2:
                        st.metric(
                            "Best Baseline MAPE",
                            f"{overall['baseline_mape']:.2f}%"
                        )
                    
                    with col_b3:
                        improvement = overall['improvement_pct']
                        st.metric(
                            "ML Advantage",
                            f"{improvement:+.1f}%",
                            delta="Better" if improvement > 0 else "Baseline wins",
                            delta_color="normal" if improvement > 0 else "inverse"
                        )
                
                st.caption("""
                **Interpretation:**
                - **Positive advantage**: ML models outperform simple baseline methods
                - **Negative advantage**: Baseline methods are more accurate (suggests overfitting or insufficient data)
                - **Industry benchmark**: 10-30% improvement over SPLY is typical for ML in consumer electronics
                """)
        
        # ============================================================
        # Section 3: Business Impact (Safety Stock & Risk)
        # ============================================================
        
        if has_business_impact:
            st.markdown("---")
            st.subheader("⚠️ Inventory Risk & Safety Stock")
            
            st.markdown("""
            Forecast uncertainty translates to inventory decisions. This analysis quantifies potential 
            overstock (carrying cost risk) and stockout (lost sales risk) exposure.
            """)
            
            # Brand-level impact
            if 'brand_analysis' in core_data['business_impact']:
                brand_analysis = core_data['business_impact']['brand_analysis']
                
                impact_data = []
                for brand in brands:
                    if brand in brand_analysis:
                        impact = brand_analysis[brand]
                        
                        buffer_pct = (impact['safety_stock'] / impact['annual_forecast']) * 100

                        impact_data.append({
                            'Brand': brand,
                            'Forecast': f"{impact['annual_forecast']:,.0f}",
                            'Overstock Risk': f"{impact['overstock_risk']:,.0f}",
                            'Stockout Risk': f"{impact['stockout_risk']:,.0f}",
                            'Rec. Safety Stock': f"{impact['safety_stock']:,.0f}",
                            'Buffer %': f"{buffer_pct:.1f}%"
                        })
                
                if impact_data:
                    df_impact = pd.DataFrame(impact_data)
                    st.dataframe(df_impact, use_container_width=True, hide_index=True)
                    
                    st.caption("""
                    **Risk Metrics:**
                    - **Overstock Risk**: Units potentially unsold if demand hits lower bound (excess inventory cost)
                    - **Stockout Risk**: Additional units needed if demand hits upper bound (lost sales opportunity)
                    - **Recommended Safety Stock**: Optimal buffer balancing both risks
                    - **Buffer %**: Safety stock as % of forecast (typical range: 10-20% for consumer electronics)
                    
                    **Note:** These represent 80% confidence intervals. Actual results may fall outside bounds.
                    """)
                
                # Portfolio summary
                if 'portfolio' in brand_analysis:
                    portfolio = brand_analysis['portfolio']
                    
                    st.markdown("---")
                    st.subheader("📦 Portfolio Summary")
                    
                    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
                    
                    with col_p1:
                        st.metric(
                            "Total Forecast",
                            f"{portfolio['total_forecast']:,.0f}"
                        )
                    
                    with col_p2:
                        st.metric(
                            "Downside Exposure",
                            f"{portfolio['total_overstock_risk']:,.0f}",
                            help="Potential excess if demand is lower than forecast"
                        )
                    
                    with col_p3:
                        st.metric(
                            "Upside Exposure",
                            f"{portfolio['total_stockout_risk']:,.0f}",
                            help="Potential shortage if demand exceeds forecast"
                        )
                    
                    with col_p4:
                        buffer_pct = (portfolio['total_safety_stock']/portfolio['total_forecast'])*100
                        st.metric(
                            "Recommended Buffer",
                            f"{portfolio['total_safety_stock']:,.0f}",
                            delta=f"{buffer_pct:.1f}% of forecast"
                        )
            
            # Recommendations if available
            if 'recommendations' in core_data['business_impact']:
                st.markdown("---")
                st.subheader("💡 Operational Guidance")
                
                for brand in brands:
                    if brand in core_data['business_impact']['recommendations']:
                        with st.expander(f"**{brand}** Strategic Actions"):
                            recs = core_data['business_impact']['recommendations'][brand]
                            
                            for rec in recs:
                                # Risk-based indicators
                                if rec['risk_level'] == 'High':
                                    icon = "🔴"
                                elif rec['risk_level'] == 'Moderate':
                                    icon = "🟡"
                                else:
                                    icon = "🟢"
                                
                                st.markdown(f"{icon} **{rec['category']}**: {rec['recommendation']}")
                
                st.info("""
                **Using These Insights:**
                - Green indicators: Standard operations sufficient
                - Yellow indicators: Monitor closely, prepare contingency plans
                - Red indicators: Consider pre-positioning inventory or demand-shaping actions
                """)

# ================================================================
# TAB 5: METHODOLOGY
# ================================================================

with tab5:
    st.subheader("🔬 How This Forecast Was Built")
    
    st.markdown("""
    This forecasting system combines Canadian smartphone import data with market intelligence 
    to predict 2026 demand. Here's how it works, in plain English.
    """)
    
    # ============================================================
    # DATA COLLECTION
    # ============================================================
    
    st.markdown("---")
    st.markdown("### 📦 Step 1: Data Collection")
    
    col1, col2 = st.columns([1.5, 2])
    
    with col1:
        st.metric("Data Span", "Jan 2011 - Nov 2025")
        st.metric("Total Months", "180")
        st.metric("Total Units", "191.8M")
    
    with col2:
        st.markdown("""
        **Import Data** (Statistics Canada)
        - Smartphone HS codes: 8517.12 and 8517.13
        - Monthly import quantities for Canada
        - Processed from raw StatCan CSV files
        
        **Supply Chain Lag**
        - Imports don't instantly become sales
        - Statistical analysis found **1-month lag** is optimal
        - Applied: October imports → November sales
        
        **Google Trends Integration**
        - Search interest data for each brand
        - Granger causality testing to validate predictive power
        - Statistical weighting (0-100%) based on correlation strength
        - Applied where search patterns show causal relationship with sales
        
        **Market Share Split** (StatCounter)
        - Monthly brand shares: Apple, Samsung, Google, Motorola
        - Applied to import totals to get brand-level units
        
        **Calibration** (SellCell Annual Report)
        - Aligned with known annual sales figures (2011-2025)
        - December 2025 synthesized using historical patterns
        - Ensures forecast baseline matches market reality
        """)
    
    # ============================================================
    # FEATURE ENGINEERING
    # ============================================================
    
    st.markdown("---")
    st.markdown("### 🛠️ Step 2: Feature Engineering")
    
    st.markdown("""
    Features are the "inputs" models use to make predictions. We only use features that can be known or forecasted for 2026.
    """)
    
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        st.markdown("**Time Features**")
        st.markdown("""
        - Month (1-12)
        - Quarter (Q1-Q4)
        - Year
        - Time index (trend)
        """)
    
    with col_f2:
        st.markdown("**Calendar Events**")
        st.markdown("""
        - Holiday season (Nov-Dec)
        - Back to school (Aug)
        - Black Friday (Nov)
        - Product launches
        """)
    
    with col_f3:
        st.markdown("**Economic & History**")
        st.markdown("""
        - CPI (inflation) from OECD
        - Previous month sales
        - Year-ago sales
        - Moving averages
        """)
    
    st.info("""
    **Design principle:** Excluded variables requiring forecasts (competitor prices, exchange rates) 
    or with weak correlation (GDP, unemployment). Focused on statistically significant, forecastable factors.
    """)
    
    # ============================================================
    # BRAND-SPECIFIC START DATES
    # ============================================================
    
    with st.expander("📅 **Brand-Specific Training Periods**"):
        st.markdown("""
        Different brands have different relevant histories:
        
        | Brand | Training Starts | Reason |
        |-------|----------------|---------|
        | **Apple** | Jan 2011 | iPhone 4/4S era - consistent product strategy |
        | **Samsung** | Jan 2011 | Galaxy S2 era - full data available |
        | **Google** | Oct 2016 | Pixel launch (excludes Nexus era) |
        | **Motorola** | Jan 2014 | Lenovo acquisition - new market approach |
        
        Using the right history prevents models from learning outdated patterns.
        """)
    
    # ============================================================
    # MODEL TRAINING
    # ============================================================
    
    st.markdown("---")
    st.markdown("### 🤖 Step 3: Model Training & Selection")
    
    st.markdown("""
    We tested **8 different forecasting approaches** to find the best one for each brand:
    """)
    
    # Model categories
    col_m1, col_m2, col_m3 = st.columns(3)
    
    with col_m1:
        st.markdown("**Statistical**")
        st.markdown("- SARIMAX")
        st.caption("Traditional time series model")
    
    with col_m2:
        st.markdown("**Machine Learning**")
        st.markdown("""
        - XGBoost
        - LightGBM
        - CatBoost
        - Gaussian Process
        """)
        st.caption("Modern ML algorithms")
    
    with col_m3:
        st.markdown("**Specialized**")
        st.markdown("""
        - Prophet (w/ variables)
        - Prophet (time only)
        - Ensemble (combines all 7)
        """)
        st.caption("Purpose-built forecasting tools")
    
    st.markdown("---")
    
    # Training process
    st.markdown("**Training Process:**")
    
    col_t1, col_t2, col_t3 = st.columns(3)
    
    with col_t1:
        st.markdown("**1. Hyperparameter Tuning**")
        st.markdown("""
        - Used Optuna optimization
        - 50-100 trials per model
        - Finds best settings automatically
        """)
    
    with col_t2:
        st.markdown("**2. Cross-Validation**")
        st.markdown("""
        - 5-fold walk-forward CV
        - Respects time order
        - Tests stability
        """)
    
    with col_t3:
        st.markdown("**3. Final Test**")
        st.markdown("""
        - Last 12 months held out
        - Simulates real prediction
        - Prevents data leakage
        """)
    
    st.markdown("---")
    
    # Selection criteria
    st.markdown("**How Champions Were Selected:**")
    
    st.markdown("""
    Each model was scored using:
    - **60%** - Accuracy on test period (last 12 months)
    - **25%** - Stability across different time periods
    - **15%** - How well it explains the patterns
    
    The **highest-scoring model for each brand** was selected as the champion.
    """)
    
    # Current champions
    st.markdown("**Current Champions:**")
    
    champ_data = []
    for brand in brands:
        model_name = core_data['model_selection'][brand]['model']
        mape = core_data['model_selection'][brand].get('holdout_mape', 0)
        champ_data.append({
            'Brand': brand,
            'Champion Model': model_name,
            'Test Accuracy': f"{mape:.1f}% error"
        })
    
    df_champs = pd.DataFrame(champ_data)
    st.dataframe(df_champs, use_container_width=True, hide_index=True)
    
    # ============================================================
    # TECHNICAL DETAILS
    # ============================================================
    
    with st.expander("🔢 **Technical Details: Transformations & Scaling**"):
        st.markdown("""
        **Log Transformation**
        
        Smartphone sales grow exponentially, not linearly. Log transformation:
        - Stabilizes variance (makes patterns more consistent)
        - Handles exponential growth better
        - Standard practice for demand forecasting
        
        **Process:**
        1. Convert sales to log-space: `log(units + 1)`
        2. Train models on log values
        3. Convert predictions back: `exp(prediction) - 1`
        
        **StandardScaler (Feature Normalization)**
        
        Different features have different scales (e.g., CPI ~140, time_index ~0-200, sales ~thousands).
        - Scales features to mean=0, std=1
        - Critical for models like GP, SARIMAX, and gradient boosting
        - Applied after train/test split to prevent data leakage
        - Ensures numerical stability and faster convergence
        
        **Result:** More accurate forecasts across all model types.
        """)
    
    # ============================================================
    # FORECASTING FOR 2026
    # ============================================================
    
    st.markdown("---")
    st.markdown("### 🔮 Step 4: Generating 2026 Forecasts")
    
    st.markdown("""
    **For each month in 2026:**
    
    1. **Build features** using historical data and previous forecast months
    2. **Run champion model** to predict that month
    3. **Store prediction** and use it for next month's features
    4. **Repeat** for all 12 months
    
    **Example (March 2026):**
    - Uses actual data through Nov 2025
    - Uses forecasted Dec, Jan, & Feb 2026 values
    - Predicts March based on these inputs
    """)
    
    # Uncertainty
    st.markdown("**Uncertainty Intervals:**")
    
    st.markdown("""
    Different models provide uncertainty in different ways:
    
    - **SARIMAX, Prophet, GP:** Statistical confidence intervals (80% probability)
    - **XGBoost, LightGBM, CatBoost:** Estimated range (±15% around forecast)
    - **Ensemble:** Combined uncertainty from base models
    
    The uncertainty bands show the reasonable range where actual sales might land.
    """)
    
    # ============================================================
    # LIVE SCENARIOS
    # ============================================================
    
    st.markdown("---")
    st.markdown("### 🎛️ Custom Scenarios (Live Inference)")
    
    st.markdown("""
    When you adjust inflation or launch dates in the sidebar:
    
    1. **Features are recalculated** with your new parameters
    2. **Models re-run** with updated inputs (2-5 seconds)
    3. **New forecast generated** reflecting your scenario
    
    This is **true AI inference**, not simple multiplication. The models re-learn 
    patterns based on your assumptions.
    
    **If a model fails:** The system falls back to elasticity-based simulation 
    (adjusts baseline forecast using economic principles).
    """)
    
    # ============================================================
    # CONSIDERATIONS
    # ============================================================
    
    st.markdown("---")
    st.markdown("### ⚠️ Model Considerations")
    
    col_l1, col_l2 = st.columns(2)
    
    with col_l1:
        st.markdown("**Data Approach:**")
        st.markdown("""
        - Import proxy validated with known annual sales
        - 1-month lag consistently applied across brands
        - Import and Google trends data combined as proxy for unit sales
        - Market share data from StatCounter monthly reports
        """)
    
    with col_l2:
        st.markdown("**Forecast Scope:**")
        st.markdown("""
        - Optimized for 6-12 month horizon
        - Confidence intervals widen over time
        - Assumes continuation of historical patterns
        - External shocks (policy changes, supply disruptions) not modeled
        """)
    
    # ============================================================
    # DATA SOURCES
    # ============================================================
    
    st.markdown("---")
    st.markdown("### 📚 Data Sources")
    
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        st.markdown("""
        **Primary Data:**
        - Statistics Canada (Import Data)
        - StatCounter (Market Share)
        - SellCell (Annual Sales Calibration)
        """)
    
    with col_s2:
        st.markdown("""
        **Economic Data:**
        - Bank of Canada (Historical CPI)
        - OECD (CPI Forecasts)
        - Google Trends (Search Interest)
        """)

# ================================================================
# EXPORT FUNCTIONALITY
# ================================================================

st.markdown("---")
st.subheader("💾 Export Forecast")

col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    # CSV export
    @st.cache_data
    def create_csv_export():
        """Create CSV for download"""
        df_csv = df_future_scenario[['month']].copy()
        df_csv['Month'] = df_csv['month'].dt.strftime('%Y-%m')
        
        for brand in brands:
            df_csv[brand] = forecasts[brand]['forecast'].round(0).astype(int)
        
        df_csv['Total'] = df_csv[brands].sum(axis=1)
        
        return df_csv.drop('month', axis=1).to_csv(index=False).encode('utf-8')
    
    csv_data = create_csv_export()
    
    filename_suffix = f"_cpi{cpi_adjustment*100:+.0f}pct" if scenario_active else "_baseline"
    
    st.download_button(
        label="📄 Download CSV",
        data=csv_data,
        file_name=f"smartphone_forecast_2026{filename_suffix}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        help="Download simple forecast table"
    )

with col_exp2:
    # Excel export
    @st.cache_data
    def create_excel_export():
        """Create comprehensive Excel workbook"""
        from io import BytesIO
        
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Sheet 1: Executive Summary
            df_summary.to_excel(writer, sheet_name='Executive Summary', index=False)
            
            # Sheet 2: Monthly Forecast
            df_export = df_future_scenario[['month']].copy()
            df_export['Month'] = df_export['month'].dt.strftime('%Y-%m')
            
            for brand in brands:
                df_export[f'{brand}_Forecast'] = forecasts[brand]['forecast'].round(0).astype(int)
                if 'lower_bound' in forecasts[brand]:
                    df_export[f'{brand}_Lower_CI'] = forecasts[brand]['lower_bound'].round(0).astype(int)
                    df_export[f'{brand}_Upper_CI'] = forecasts[brand]['upper_bound'].round(0).astype(int)
            
            df_export = df_export.drop('month', axis=1)
            df_export.to_excel(writer, sheet_name='Monthly Forecast', index=False)
            
            # Sheet 3: Model Performance
            df_perf[['Brand', 'Selected Model', 'Holdout MAPE', 'CV MAPE', 'R² Score']].to_excel(
                writer, sheet_name='Model Performance', index=False
            )
            
            # Sheet 4: Enhanced Metrics (if available)
            if has_enhanced and 'enhanced_data' in locals():
                df_enhanced.to_excel(writer, sheet_name='Enhanced Metrics', index=False)
            
            # Sheet 5: Baseline Comparison (if available)
            if has_baseline_comp and 'baseline_data' in locals():
                df_baseline.to_excel(writer, sheet_name='Baseline Comparison', index=False)
            
            # Sheet 6: Business Impact (if available)
            if has_business_impact and 'impact_data' in locals():
                df_impact.to_excel(writer, sheet_name='Business Impact', index=False)
            
            # Sheet 7: Scenario Details
            if scenario_active:
                scenario_details = pd.DataFrame([
                    {'Parameter': 'CPI Adjustment', 'Value': f"{cpi_adjustment:.1%}"},
                    {'Parameter': 'Scenario Type', 'Value': 'Custom'},
                    {'Parameter': 'Generated Date', 'Value': datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
                    {'Parameter': 'vs Baseline', 'Value': f"{scenario_diff:+,.0f} units ({scenario_diff_pct:+.1f}%)"}
                ])
            else:
                scenario_details = pd.DataFrame([
                    {'Parameter': 'Scenario Type', 'Value': 'Baseline'},
                    {'Parameter': 'Generated Date', 'Value': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                ])
            
            scenario_details.to_excel(writer, sheet_name='Scenario Info', index=False)
        
        return output.getvalue()
    
    try:
        excel_data = create_excel_export()
        
        st.download_button(
            label="📊 Download Excel Report",
            data=excel_data,
            file_name=f"smartphone_forecast_2026{filename_suffix}_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download comprehensive report with all data and model performance"
        )
    except ImportError:
        st.warning("⚠️ Install 'openpyxl' to enable Excel export: `pip install openpyxl`")

# ================================================================
# FOOTER
# ================================================================

st.markdown("---")

col_foot1, col_foot2, col_foot3 = st.columns(3)

with col_foot1:
    st.caption("**Data Source**")
    st.caption("Statistics Canada HS Code Import Data")
    st.caption("1-month supply chain lag applied")

with col_foot2:
    st.caption("**Methodology**")
    st.caption("8-model evaluation system")
    st.caption("Walk-forward CV with holdout validation")

with col_foot3:
    st.caption("**Forecast Status**")
    st.caption(f"{'🟢 Live Inference' if scenario_active else '🔵 Baseline'}")
    st.caption(f"Total Market: {int(total_forecast):,} units")

# Help section
with st.expander("ℹ️ How to Use This Dashboard"):
    st.markdown("""
    ### Quick Start Guide
    
    **Baseline Mode:**
    - View original forecasts based on historical trends
    - Explore model performance and selection rationale
    - Export data for reporting
    
    **Custom Scenario Mode:**
    - Adjust inflation rate to simulate economic changes
    - Modify product launch dates by brand
    - Run live AI inference to see updated forecasts
    - Compare results vs baseline
    
    ### Tab Guide
    
    **📈 Forecast View:**
    - Combined historical and forecast chart
    - Summary table with key metrics
    
    **🔍 Brand Breakdown:**
    - Individual brand charts with confidence intervals
    - Detailed monthly breakdown table
    - Status indicators (AI inference vs simulation)
    
    **🤖 Model Performance:**
    - Learn about each of the 8 model types evaluated
    - Compare accuracy across brands
    - Understand model selection rationale
    
    **💼 Business Insights:**
    - Forecast bias and directional accuracy
    - ML vs baseline comparison
    - Overstock/stockout risk analysis
    - Safety stock recommendations
                
    **🔬 Methodology:**
    - Transparent walkthrough of the entire forecasting pipeline
    - Data lineage explanation (StatCan Imports → Supply Chain Lag → Calibration)
    - Details on feature engineering (CPI, Seasonality, Google Trends)
    - Technical deep-dive into the training, validation, and selection process
    
    ### Understanding Model Performance
    
    - **MAPE < 10%**: Excellent accuracy
    - **MAPE 10-15%**: Very good accuracy
    - **MAPE 15-20%**: Good accuracy
    - **MAPE > 20%**: Fair accuracy (challenging forecast)
    
    ### 8 Models Evaluated
    
    1. **SARIMAX** - Statistical time series with exogenous variables
    2. **XGBoost** - Extreme gradient boosting trees
    3. **LightGBM** - Fast gradient boosting machine
    4. **CatBoost** - Categorical boosting with ordered boosting
    5. **Gaussian Process** - Probabilistic Bayesian regression
    6. **Prophet-Multi** - FB Prophet with external regressors
    7. **Prophet-Uni** - FB Prophet univariate (time-only)
    8. **Ensemble** - Caruana weighted combination of all 7 models
    """)