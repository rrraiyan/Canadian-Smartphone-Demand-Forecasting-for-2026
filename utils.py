"""
utils.py - Production-Grade Forecasting Engine (8 Model Support)
complete log-scaling pipeline and scaler integration
True live inference + ensemble re-assembly + smart caching
Business metrics and baseline comparison utilities
"""

import pandas as pd
import numpy as np
import warnings
import os
import pickle
from functools import lru_cache
from sklearn.metrics import mean_absolute_percentage_error
warnings.filterwarnings('ignore')

# ================================================================
# FEATURE ENGINEERING
# ================================================================

class FeatureEngineer:
    """
    Feature engineering for forecasting with dynamic updates
    Supports all 8 model types with proper feature construction
    """
    
    def __init__(self, df_historical, brands=None):
        self.df_historical = df_historical.copy()
        
        # Auto-detect brands from column names
        if brands is None:
            self.brands = [
                c.replace('_units', '') 
                for c in df_historical.columns 
                if c.endswith('_units')
            ]
        else:
            self.brands = brands
        
        # ════════════════════════════════════════════════════════════
        # Launch calendar 
        # ════════════════════════════════════════════════════════════
        self.defaults = {
            'Apple': [9, 3],      # Sept (iPhone), March (SE/Spring)
            'Samsung': [2, 7],    # Feb (Galaxy S), July (Z Fold/Flip)
            'Google': [8, 2],     # Aug (Pixel flagship), Feb (Pixel 'a')
            'Motorola': [1, 7]    # Jan (G-series), July (Razr)
        }
        
        # Calibrate elasticities and impacts
        self.elasticities = self._calculate_elasticities()
        self.launch_lifts = self._calculate_launch_lifts()
    
    def create_future_features(self, df_future, launch_overrides=None, cpi_override=None):
        """
        Generate dynamic feature set based on user inputs
        """
        df = df_future.copy()
    
        # Apply CPI override if specified AND non-zero
        if cpi_override is not None and abs(cpi_override) > 0.001:
            df = self._recalculate_cpi(df, cpi_override)
            df.attrs['cpi_delta'] = cpi_override
        else:
            df.attrs['cpi_delta'] = 0.0
    
        # Apply launch overrides only if provided
        if launch_overrides:
            for brand, months in launch_overrides.items():
                col = f'launch_{brand.lower()}'
                if col in df.columns:  # Only modify if column exists
                    df[col] = 0
                    df.loc[df['month'].dt.month.isin(months), col] = 1
    
        # Update temporal features 
        df = self._add_time_features(df)
    
        return df
    
    def _recalculate_cpi(self, df, annual_rate):
        """
        Adjust existing OECD CPI forecast by a delta rate (Progressive Drift).
        
        Preserves the shape/seasonality of the OECD forecast but 
        tilts the trend up or down based on the user's input.
        """
        df = df.copy()
        
        last_date = self.df_historical['month'].max()
        
        # Calculate monthly drift factor
        # If user adds +2% inflation (0.02), we apply (1.02)^(1/12) compound monthly
        monthly_drift = (1 + annual_rate) ** (1/12)
        
        drift_factors = []
        for _, row in df.iterrows():
            # Calculate months ahead from the last historical data point
            months_ahead = (row['month'].year - last_date.year) * 12 + \
                           (row['month'].month - last_date.month)
            
            # Drift accumulates: Month 1 is small, Month 12 is the full annual_rate
            drift = monthly_drift ** months_ahead
            drift_factors.append(drift)
            
        # Apply drift to the EXISTING OECD curve (Multiplicative)
        df['cpi_index'] = df['cpi_index'] * drift_factors
        
        return df
    
    def _add_time_features(self, df):
        """Add/update temporal features for models"""
        df = df.copy()
        
        # Basic time features
        if 'month_num' not in df.columns:
            df['month_num'] = df['month'].dt.month
        if 'quarter' not in df.columns:
            df['quarter'] = df['month'].dt.quarter
        if 'year' not in df.columns:
            df['year'] = df['month'].dt.year
        
        # Seasonal indicators (matching training pipeline)
        df['is_holiday_season'] = df['month_num'].isin([11, 12]).astype(int)
        df['is_back_to_school'] = df['month_num'].isin([8]).astype(int)
        df['is_black_friday'] = (df['month_num'] == 11).astype(int)
        df['is_new_year_promo'] = (df['month_num'] == 1).astype(int)
        
        # Time index for trend
        if 'time_index' not in df.columns:
            last_idx = len(self.df_historical)
            df['time_index'] = range(last_idx + 1, last_idx + len(df) + 1)
        
        return df
    
    def build_ml_features_iterative(self, brand, df_future, forecast_values, feature_cols=None):
        """
        Build features for iterative ML forecasting
        
        Handles both capitalized (Apple_lag1) and lowercase (apple_lag1) conventions
        Critical for recursive forecasting where each step depends on previous predictions
        
        IMPORTANT: All values are in ORIGINAL SCALE (not log-space)
        Models will apply log-transform internally during prediction
        """
        month_idx = len(forecast_values)
        future_row = df_future.iloc[month_idx]
        
        def get_val(lag):
            """Get lagged value from forecast history or historical data"""
            if month_idx >= lag:
                # Use forecast value (already in real units)
                return forecast_values[month_idx - lag]
            else:
                # Use historical value (real units)
                hist_col = f'{brand}_units'
                return self.df_historical[hist_col].iloc[-(lag - month_idx)]
        
        # Calculate lags (all in real units)
        lag1 = get_val(1)
        lag12 = get_val(12)
        
        # Calculate moving averages (on real units)
        vals_ma3 = [get_val(i) for i in range(1, 4)]
        ma3 = np.mean(vals_ma3)
        
        vals_ma12 = [get_val(i) for i in range(1, 13)]
        ma12 = np.mean(vals_ma12)
        
        # Build feature dictionary with both naming conventions
        features = {
            # Capitalized (Apple_lag1)
            f'{brand}_lag1': lag1,
            f'{brand}_lag12': lag12,
            f'{brand}_ma3': ma3,
            f'{brand}_ma12': ma12,
            
            # Lowercase (apple_lag1) for compatibility
            f'{brand.lower()}_lag1': lag1,
            f'{brand.lower()}_lag12': lag12,
            f'{brand.lower()}_ma3': ma3,
            f'{brand.lower()}_ma12': ma12,
            
            # Exogenous features
            f'launch_{brand.lower()}': future_row.get(f'launch_{brand.lower()}', 0),
            'cpi_index': future_row['cpi_index'],
            'month_num': future_row['month_num'],
            'quarter': future_row['quarter'],
            'year': future_row.get('year', future_row['month'].year),
            'time_index': future_row['time_index'],
            'is_holiday_season': future_row['is_holiday_season'],
            'is_back_to_school': future_row['is_back_to_school'],
            'is_black_friday': future_row.get('is_black_friday', 0),
            'is_new_year_promo': future_row.get('is_new_year_promo', 0)
        }
        
        # Filter to only requested features if specified
        if feature_cols:
            features = {k: v for k, v in features.items() if k in feature_cols}
        
        return features
    
    def _calculate_elasticities(self):
        """
        Calculate price elasticity per brand
        
        Industry standard: -0.5 for consumer electronics
        (1% price increase → 0.5% demand decrease)
        """
        return {brand: -0.5 for brand in self.brands}
    
    def _calculate_launch_lifts(self):
        """
        Calculate launch impact multiplier
        
        Standard smartphone launch generates ~15% demand lift
        """
        return {brand: 1.15 for brand in self.brands}


# ================================================================
# FORECASTING ENGINE
# ================================================================

class ForecastEngine:
    """
    Production forecasting engine with live inference for 8 model types
    
    Supported models:
    - SARIMAX (statistical time series)
    - XGBoost (gradient boosting)
    - LightGBM (fast gradient boosting)
    - CatBoost (ordered boosting with categorical support)
    - Gaussian Process (probabilistic regression)
    - Prophet-Multi (FB Prophet with regressors)
    - Prophet-Uni (FB Prophet univariate)
    - Ensemble (weighted combination)
    
    LOG-SCALING PIPELINE:
    - All models trained on log-transformed targets
    - All models predict in log-space
    - All predictions inverse-transformed to real units
    - Scalers applied consistently (SARIMAX exog, ML features, GP X+y)
    
    Features:
    - True model re-inference (not simulation)
    - Full ensemble support with sub-model loading
    - Smart caching for performance
    - Graceful fallbacks if models fail
    """
    
    def __init__(self, production_models, production_params, model_selection,
                 df_historical, feature_engineer, core_data=None, production_scalers=None):
        self.production_models = production_models
        self.production_params = production_params
        self.model_selection = model_selection
        self.production_scalers = production_scalers or {}
        self.core_data = core_data
        self.df_historical = df_historical
        self.feature_engineer = feature_engineer
        self.brands = list(production_models.keys())
    
    def forecast_all_brands(self, df_future, launch_overrides=None):
        """
        Generate forecasts for all brands with automatic model routing
        """
        forecasts = {}
        
        for brand in self.brands:
            model = self.production_models.get(brand)
            params = self.production_params.get(brand, {})
            model_type = self.model_selection[brand]['model']
            
            try:
                # Route to appropriate inference method
                if model_type in ['XGBoost', 'LightGBM']:
                    f = self._forecast_ml(brand, model, params, df_future)
                
                elif model_type == 'CatBoost':
                    f = self._forecast_catboost(brand, model, params, df_future)
                
                elif model_type == 'GP':
                    f = self._forecast_gp(brand, model, params, df_future)
                
                elif model_type in ['Prophet-Multi', 'Prophet-Uni']:
                    f = self._forecast_prophet(brand, model, params, df_future, model_type)
                
                elif model_type == 'SARIMAX':
                    f = self._forecast_sarimax(brand, model, params, df_future)
                
                elif model_type == 'Ensemble':
                    f = self._forecast_ensemble_live(brand, model, params, df_future)
                
                else:
                    # Unknown model type - fallback
                    f = self._forecast_simulation(brand, df_future, launch_overrides)
                    f['warning'] = f'Unknown model type: {model_type}'
            
            except Exception as e:
                # Graceful fallback to simulation
                print(f"⚠️ {brand} inference failed: {e}. Using fallback.")
                f = self._forecast_simulation(brand, df_future, launch_overrides)
                f['error'] = str(e)
                f['fallback'] = True
            
            forecasts[brand] = f
        
        return forecasts
    
    # ================================================================
    # MODEL-SPECIFIC INFERENCE METHODS
    # ================================================================
    
    def _forecast_ml(self, brand, model, params, df_future):
        """
        XGBoost/LightGBM iterative forecast with full log-scaling pipeline
        
        Pipeline:
        1. Build features (in real units)
        2. Scale features using training scaler
        3. Model predicts in log-space
        4. Inverse log-transform to real units
        
        Uses recursive approach where each prediction feeds into next timestep
        """
        feature_cols = params.get('feature_cols', [])
        forecast_values = []
        
        for i in range(len(df_future)):
            # Build features (real units)
            features = self.feature_engineer.build_ml_features_iterative(
                brand, df_future, forecast_values, feature_cols
            )
            
            X_row = np.array([[features.get(col, 0) for col in feature_cols]])
            
            # ========================================================
            # Scale features (matching training pipeline)
            # ========================================================
            if brand in self.production_scalers:
                X_row = self.production_scalers[brand].transform(X_row)
            
            # Model predicts in log-space
            pred_log = model.predict(X_row)[0]
            
            # Inverse transform to real units
            pred_real = np.expm1(pred_log)
            pred_real = max(0, pred_real)
            
            forecast_values.append(pred_real)
        
        arr = np.array(forecast_values)
        
        return {
            'forecast': arr,
            'lower_bound': arr * 0.85,
            'upper_bound': arr * 1.15,
            'confidence_intervals': False
        }
    
    def _forecast_catboost(self, brand, model, params, df_future):
        """
        CatBoost iterative forecast with numeric-only scaling
        
        Pipeline:
        1. Build features (in real units)
        2. Scale ONLY numeric columns (not categorical)
        3. Model predicts in log-space
        4. Inverse log-transform to real units
        
        Handles categorical features via DataFrame input
        """
        feature_cols = params.get('feature_cols', [])
        cat_features = params.get('cat_features', [])
        numeric_cols = params.get('numeric_cols', [c for c in feature_cols if c not in cat_features])
        
        forecast_values = []
        
        for i in range(len(df_future)):
            # Build features (real units)
            features = self.feature_engineer.build_ml_features_iterative(
                brand, df_future, forecast_values, feature_cols
            )
            
            # CatBoost expects DataFrame for categorical handling
            X_row = pd.DataFrame([features])[feature_cols]
            
            # ========================================================
            # Scale ONLY numeric columns 
            # ========================================================
            if brand in self.production_scalers and numeric_cols:
                scaler = self.production_scalers[brand]
                X_row[numeric_cols] = scaler.transform(X_row[numeric_cols])
            
            # Ensure categorical columns are properly typed (int, not category)
            for cat_col in cat_features:
                if cat_col in X_row.columns:
                    X_row[cat_col] = X_row[cat_col].astype(int)
            
            # Model predicts in log-space
            pred_log = model.predict(X_row)[0]
            
            # Inverse transform to real units
            pred_real = np.expm1(pred_log)
            pred_real = max(0, pred_real)
            
            forecast_values.append(pred_real)
        
        arr = np.array(forecast_values)
        
        return {
            'forecast': arr,
            'lower_bound': arr * 0.85,
            'upper_bound': arr * 1.15,
            'confidence_intervals': False
        }
    
    def _forecast_gp(self, brand, model, params, df_future):
        """
        Gaussian Process iterative forecast with double-scaling pipeline
        
        Pipeline:
        1. Build features (in real units)
        2. Scale X using X_scaler
        3. Model predicts in scaled log-space
        4. Inverse scale prediction using y_scaler (scaled log → log)
        5. Inverse log-transform (log → real units)
        
        Returns true probabilistic confidence intervals
        """
        feature_cols = params.get('feature_cols', [])
        
        # Check for scalers
        if brand not in self.production_scalers:
            raise RuntimeError(
                f"GP model selected for {brand} but scalers not available. "
                f"Falling back to simulation."
            )
        
        scaler_X = self.production_scalers[brand]['X_scaler']
        scaler_y = self.production_scalers[brand]['y_scaler']
        
        forecast_values = []
        stds = []
        
        for i in range(len(df_future)):
            # Build features (real units)
            features = self.feature_engineer.build_ml_features_iterative(
                brand, df_future, forecast_values, feature_cols
            )
            
            X_raw = np.array([[features.get(col, 0) for col in feature_cols]])
            
            # Scale X
            X_scaled = scaler_X.transform(X_raw)
            
            # Predict with uncertainty (in scaled log-space)
            pred_scaled, std_scaled = model.predict(X_scaled, return_std=True)
            
            # Inverse scale to log-space
            pred_log = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()[0]
            std_log = std_scaled[0] * scaler_y.scale_[0]
            
            # Inverse log transform to real units
            pred_real = np.expm1(pred_log)
            pred_real = max(0, pred_real)
            
            # Transform std to real space 
            std_real = pred_real * np.abs(std_log)
            
            forecast_values.append(pred_real)
            stds.append(std_real)
        
        forecast_arr = np.array(forecast_values)
        stds_arr = np.array(stds)
        
        # 80% confidence intervals (1.28 × std)
        return {
            'forecast': forecast_arr,
            'lower_bound': np.maximum(0, forecast_arr - (1.28 * stds_arr)),
            'upper_bound': forecast_arr + (1.28 * stds_arr),
            'confidence_intervals': True,
            'uncertainty': stds_arr
        }
    
    def _forecast_prophet(self, brand, model, params, df_future, model_type):
        """
        Prophet forecast with inverse log-transform
        
        Pipeline:
        1. Prophet predicts in log-space (trained on log targets)
        2. Inverse log-transform all predictions to real units
        
        Prophet provides native confidence intervals
        """
        future = df_future[['month']].copy()
        future.columns = ['ds']
        
        # Add regressors if multivariate
        if 'Multi' in model_type:
            regressor_cols = params.get('regressor_cols', [])
            for reg in regressor_cols:
                if reg in df_future.columns:
                    future[reg] = df_future[reg].values
        
        fcst = model.predict(future)
        
        # Prophet predictions are in log-space - inverse transform
        forecast_real = np.expm1(fcst['yhat'].values)
        lower_real = np.expm1(fcst['yhat_lower'].values)
        upper_real = np.expm1(fcst['yhat_upper'].values)
        
        return {
            'forecast': np.maximum(0, forecast_real),
            'lower_bound': np.maximum(0, lower_real),
            'upper_bound': np.maximum(0, upper_real),
            'confidence_intervals': True
        }
    
    def _forecast_sarimax(self, brand, model, params, df_future):
        """
        SARIMAX forecast with scaled exog + inverse log-transform
        
        Pipeline:
        1. Scale exogenous variables using training scaler 
        2. SARIMAX predicts in log-space
        3. Inverse log-transform all predictions to real units
        
        Uses get_forecast() for proper interval estimation
        """
        exog_cols = params.get('exog_cols', [])
        
        # =============================================================
        # Scale exog variables 
        # =============================================================
        if exog_cols:
            if brand in self.production_scalers and self.production_scalers[brand] is not None:
                X_raw = df_future[exog_cols].values
                X = self.production_scalers[brand].transform(X_raw)
            else:
                # Fallback: unscaled (will reduce accuracy)
                X = df_future[exog_cols].values
                print(f"⚠️ SARIMAX {brand}: Using unscaled exog (scaler missing)")
        else:
            X = None
        
        # Get forecast with confidence intervals
        res = model.get_forecast(steps=len(df_future), exog=X)
        
        # SARIMAX predictions are in log-space - inverse transform
        forecast_log = res.predicted_mean.values
        conf_int_log = res.conf_int(alpha=0.2)  # 80% CI
        
        forecast_real = np.expm1(forecast_log)
        lower_real = np.expm1(conf_int_log.iloc[:, 0].values)
        upper_real = np.expm1(conf_int_log.iloc[:, 1].values)
        
        return {
            'forecast': np.maximum(0, forecast_real),
            'lower_bound': np.maximum(0, lower_real),
            'upper_bound': np.maximum(0, upper_real),
            'confidence_intervals': True
        }
    
    def _forecast_ensemble_live(self, brand, model, params, df_future):
        """
        True ensemble re-inference with all base models
        
        PIPELINE:
        1. Each base model predicts in REAL units (already inverse-transformed)
        2. Convert predictions BACK to log-space
        3. Weighted average in LOG-SPACE
        4. Inverse log-transform final average to real units
        
        This matches training where:
        - Base models predict in log-space
        - Ensemble averages in log-space
        - Final prediction inverse-transformed
        
        Supports: SARIMAX, XGBoost, LightGBM, CatBoost, GP, Prophet×2
        """
        # Get weights
        weights = params.get('weights', {})
        if not weights and hasattr(model, 'weights_'):
            weights = model.weights_
        if not weights:
            # Fallback: equal weights for available models
            weights = {
                'SARIMAX': 0.2,
                'XGBoost': 0.2,
                'LightGBM': 0.2,
                'Prophet-Multi': 0.2,
                'Prophet-Uni': 0.2
            }
        
        preds_log = np.zeros(len(df_future))  # Accumulate in LOG-SPACE
        total_weight = 0
        
        # Map model names to cache files
        file_map = {
            'SARIMAX': 'sarimax_results.pkl',
            'XGBoost': 'xgboost_results.pkl',
            'LightGBM': 'lightgbm_results.pkl',
            'CatBoost': 'catboost_results.pkl',
            'GP': 'gp_results.pkl',
            'Prophet-Multi': 'prophet_results.pkl',
            'Prophet-Uni': 'prophet_uni_results.pkl'
        }
        
        # Run each sub-model
        for model_name, weight in weights.items():
            if weight < 0.01:  # Skip negligible weights
                continue
            
            fname = file_map.get(model_name)
            if not fname:
                continue
            
            try:
                # Load sub-model (cached for performance)
                sub_model_data = self._load_cached_model_file(fname)
                
                if brand not in sub_model_data['trained_models']:
                    continue
                
                sub_model = sub_model_data['trained_models'][brand]
                sub_params = sub_model_data.get('best_params', {}).get(brand, {})
                
                # Get predictions in REAL space
                if 'Prophet' in model_name:
                    res = self._forecast_prophet(brand, sub_model, sub_params,
                                                 df_future, model_name)
                elif model_name == 'SARIMAX':
                    res = self._forecast_sarimax(brand, sub_model, sub_params,
                                                 df_future)
                elif model_name == 'CatBoost':
                    res = self._forecast_catboost(brand, sub_model, sub_params,
                                                  df_future)
                elif model_name == 'GP':
                    if brand in self.production_scalers:
                        res = self._forecast_gp(brand, sub_model, sub_params,
                                               df_future)
                    else:
                        continue  # Skip GP if no scalers
                else:  # XGBoost, LightGBM
                    res = self._forecast_ml(brand, sub_model, sub_params, df_future)
                
                # Convert BACK to log-space for averaging 
                pred_real = res['forecast']
                pred_log = np.log1p(pred_real)
                
                preds_log += pred_log * weight
                total_weight += weight
            
            except Exception as e:
                print(f"⚠️ Ensemble sub-model {model_name} failed: {e}")
                continue
        
        if total_weight == 0:
            raise ValueError(f"All ensemble sub-models failed for {brand}")
        
        # Normalize by actual weights used (in log-space)
        avg_pred_log = preds_log / total_weight
        
        # Inverse transform averaged log prediction to real units
        final_pred_real = np.expm1(avg_pred_log)
        final_pred_real = np.maximum(0, final_pred_real)
        
        return {
            'forecast': final_pred_real,
            'lower_bound': final_pred_real * 0.85,
            'upper_bound': final_pred_real * 1.15,
            'confidence_intervals': False,
            'ensemble_weight_used': total_weight
        }
    
    @lru_cache(maxsize=10)
    def _load_cached_model_file(self, filename):
        """
        Load and cache model files for performance
        
        Critical for ensemble - prevents repeated disk I/O
        Uses LRU cache to keep frequently accessed files in memory
        """
        # Try multiple locations
        search_paths = [
            filename,
            os.path.join('outputs', filename),
            os.path.join('..', filename),
            os.path.join('..', 'outputs', filename),
            os.path.join(os.path.dirname(__file__), filename),
            os.path.join(os.path.dirname(__file__), 'outputs', filename),
            os.path.join(os.path.dirname(__file__), '..', 'outputs', filename)
        ]
        
        for p in search_paths:
            if os.path.exists(p):
                with open(p, 'rb') as f:
                    return pickle.load(f)
        
        raise FileNotFoundError(
            f"Could not find {filename} in any of: {search_paths}"
        )
    
    def _forecast_simulation(self, brand, df_future, launch_overrides):
        """
        Simulation fallback when live inference fails
    
        Applies elasticity and launch timing adjustments to baseline forecast.
        Baseline forecasts are already in REAL units (not log-space).
    
        Ensures app never crashes - always returns valid forecast.
        """
        if not self.core_data or 'forecasts_baseline' not in self.core_data:
            # Last resort: return zeros
            return {
                'forecast': np.zeros(len(df_future)),
                'lower_bound': np.zeros(len(df_future)),
                'upper_bound': np.zeros(len(df_future)),
                'confidence_intervals': False,
                'fallback': True,
                'error': 'No baseline data available'
            }
    
        if brand not in self.core_data['forecasts_baseline']:
            return {
                'forecast': np.zeros(len(df_future)),
                'lower_bound': np.zeros(len(df_future)),
                'upper_bound': np.zeros(len(df_future)),
                'confidence_intervals': False,
                'fallback': True,
                'error': f'No baseline data for {brand}'
            }
    
        base = self.core_data['forecasts_baseline'][brand]
        forecast_val = base['forecast'].copy()
    
        # 1. Apply CPI Adjustment (Price Elasticity)
        cpi_delta = df_future.attrs.get('cpi_delta', 0.0)
        if abs(cpi_delta) > 0.001:  # Only apply if materially different from zero
            elasticity = self.feature_engineer.elasticities.get(brand, -0.5)
            # Price elasticity: Higher inflation → higher prices → lower demand
            forecast_val *= (1 + (cpi_delta * elasticity))
    
        # 2. Apply Launch Shift
        if launch_overrides and brand in launch_overrides:
            new_months = set(launch_overrides[brand])
            def_months = set(self.feature_engineer.defaults.get(brand, []))
        
            # Only apply if actually different
            if new_months != def_months:
                lift = self.feature_engineer.launch_lifts.get(brand, 1.15)
            
                # Volume Adjustment (more/fewer launches)
                if len(new_months) > len(def_months):
                    forecast_val *= lift
                elif len(new_months) < len(def_months):
                    forecast_val /= lift
            
                # Phase Shift (timing change for same number of launches)
                if len(new_months) == len(def_months) and len(new_months) > 0:
                    shift = int(sum(new_months)/len(new_months) - sum(def_months)/len(def_months))
                    if shift != 0:
                        forecast_val = np.roll(forecast_val, shift)
                        # Fill edge values
                        if shift > 0:
                            forecast_val[:shift] = forecast_val[shift]
                        else:
                            forecast_val[shift:] = forecast_val[shift-1]
    
        return {
            'forecast': forecast_val,
            'lower_bound': base.get('lower_bound', forecast_val * 0.85),
            'upper_bound': base.get('upper_bound', forecast_val * 1.15),
            'confidence_intervals': False,
            'fallback': True
        }


# ================================================================
# BUSINESS METRICS UTILITIES 
# ================================================================

def calculate_bias(y_true, y_pred):
    """
    Calculate Mean Percentage Error (bias)
    
    Positive = Overforecasting (predicting too high)
    Negative = Underforecasting (predicting too low)
    """
    mask = y_true > 0
    if mask.sum() == 0:
        return 0.0
    
    return np.mean((y_pred[mask] - y_true[mask]) / y_true[mask]) * 100


def calculate_directional_accuracy(y_true, y_pred):
    """
    Calculate percentage of periods where direction of change was correct
    
    E.g., if actual grew 10% and forecast grew 5%, direction is correct
          if actual grew 10% but forecast declined 5%, direction is wrong
    """
    if len(y_true) < 2:
        return 0.0
    
    actual_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))
    
    correct = np.sum(actual_direction == pred_direction)
    total = len(actual_direction)
    
    return (correct / total) if total > 0 else 0.0


def calculate_weighted_mape(y_true_dict, y_pred_dict, weights):
    weighted_sum = 0
    
    for brand, weight in weights.items():
        if brand not in y_true_dict or brand not in y_pred_dict:
            continue
        
        y_true = y_true_dict[brand]
        y_pred = y_pred_dict[brand]
        
        mask = y_true > 0
        if mask.sum() == 0:
            continue
        
        brand_mape = mean_absolute_percentage_error(
            y_true[mask], 
            y_pred[mask]
        ) * 100
        
        weighted_sum += brand_mape * weight
    
    return weighted_sum


def identify_worst_months(y_true, y_pred, dates, threshold=30):
    """
    Flag months where error exceeds threshold percentage
    """
    mask = y_true > 0
    errors = np.abs((y_pred - y_true) / y_true) * 100
    
    worst_mask = (errors > threshold) & mask
    
    return dates[worst_mask], errors[worst_mask]


def calculate_seasonal_error(y_true, y_pred, dates, quarter=4):
    """
    Calculate MAPE for a specific quarter
    """
    quarter_mask = pd.DatetimeIndex(dates).quarter == quarter
    
    if not quarter_mask.any():
        return None
    
    y_true_q = y_true[quarter_mask]
    y_pred_q = y_pred[quarter_mask]
    
    mask = y_true_q > 0
    if mask.sum() == 0:
        return None
    
    return mean_absolute_percentage_error(
        y_true_q[mask], 
        y_pred_q[mask]
    ) * 100


# ================================================================
# DATA LOADING
# ================================================================

def load_app_state(core_path, models_path, config_path):
    """
    Load frozen state files with dill
    """
    import dill
    
    try:
        with open(core_path, 'rb') as f:
            core = dill.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load core data from {core_path}: {e}")
    
    try:
        with open(models_path, 'rb') as f:
            models = dill.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load models from {models_path}: {e}")
    
    try:
        with open(config_path, 'rb') as f:
            config = dill.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load config from {config_path}: {e}")
    
    return core, models, config


# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def validate_forecast_output(forecasts, brands):
    """
    Validate forecast structure and data quality
    """
    issues = []
    
    for brand in brands:
        if brand not in forecasts:
            issues.append(f"Missing forecast for {brand}")
            continue
        
        f = forecasts[brand]
        
        # Check required keys
        required_keys = ['forecast', 'lower_bound', 'upper_bound', 'confidence_intervals']
        missing = [k for k in required_keys if k not in f]
        if missing:
            issues.append(f"{brand}: Missing keys {missing}")
        
        # Check forecast validity
        if 'forecast' in f:
            if len(f['forecast']) != 12:
                issues.append(f"{brand}: Forecast length is {len(f['forecast'])}, expected 12")
            
            if np.isnan(f['forecast']).any():
                issues.append(f"{brand}: Forecast contains NaN values")
            
            if (f['forecast'] < 0).any():
                issues.append(f"{brand}: Forecast contains negative values")
        
        # Check if fallback was used
        if f.get('fallback', False):
            issues.append(f"{brand}: Using fallback simulation (live inference failed)")
    
    return len(issues) == 0, issues


def summarize_forecasts(forecasts, brands):
    """
    Generate summary statistics for forecasts
    """
    summary = {
        'total_market': 0,
        'by_brand': {},
        'fallback_count': 0,
        'confidence_interval_count': 0
    }
    
    for brand in brands:
        if brand not in forecasts:
            continue
        
        f = forecasts[brand]
        total = f['forecast'].sum()
        avg = f['forecast'].mean()
        peak = f['forecast'].max()
        
        summary['by_brand'][brand] = {
            'total': float(total),
            'average_monthly': float(avg),
            'peak_month': float(peak),
            'has_ci': f.get('confidence_intervals', False),
            'is_fallback': f.get('fallback', False)
        }
        
        summary['total_market'] += total
        
        if f.get('fallback', False):
            summary['fallback_count'] += 1
        
        if f.get('confidence_intervals', False):
            summary['confidence_interval_count'] += 1
    
    return summary