#!/usr/bin/env python3
"""
Portfolio Management Dashboard
Time Series Forecasting for Portfolio Management Optimization

A comprehensive Streamlit dashboard for visualizing financial data analysis,
risk metrics, and portfolio optimization results.

Author: Financial Analytics Team
Date: August 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import custom modules
try:
    from data_loader import FinancialDataLoader
    from financial_metrics import FinancialMetricsCalculator
    from preprocessing import FinancialDataPreprocessor
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Portfolio Management Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .risk-high { color: #ff4b4b; }
    .risk-medium { color: #ffa500; }
    .risk-low { color: #00cc00; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_financial_data():
    """Load and cache financial data."""
    try:
        loader = FinancialDataLoader(start_date="2020-01-01", end_date="2024-12-31")
        raw_data = loader.load_all_assets()
        
        preprocessor = FinancialDataPreprocessor()
        processed_data = preprocessor.preprocess_asset_data(raw_data)
        
        return processed_data, loader.get_data_summary()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

@st.cache_data
def calculate_metrics(processed_data):
    """Calculate and cache financial metrics."""
    try:
        calculator = FinancialMetricsCalculator()
        metrics = calculator.calculate_all_metrics(processed_data)
        return metrics
    except Exception as e:
        st.error(f"Error calculating metrics: {e}")
        return None

def create_price_chart(data, assets):
    """Create interactive price chart."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Normalized Price Performance', 'Daily Returns'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, asset in enumerate(assets):
        if asset in data:
            # Normalize prices to 100 for comparison
            normalized_prices = (data[asset]['Close'] / data[asset]['Close'].iloc[0]) * 100
            
            fig.add_trace(
                go.Scatter(
                    x=normalized_prices.index,
                    y=normalized_prices.values,
                    name=f"{asset} Price",
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=f"<b>{asset}</b><br>Date: %{{x}}<br>Normalized Price: %{{y:.2f}}<extra></extra>"
                ),
                row=1, col=1
            )
            
            # Add returns
            if 'Daily_Return' in data[asset].columns:
                returns = data[asset]['Daily_Return'] * 100
                fig.add_trace(
                    go.Scatter(
                        x=returns.index,
                        y=returns.values,
                        name=f"{asset} Returns",
                        line=dict(color=colors[i % len(colors)], width=1),
                        opacity=0.7,
                        hovertemplate=f"<b>{asset}</b><br>Date: %{{x}}<br>Return: %{{y:.2f}}%<extra></extra>"
                    ),
                    row=2, col=1
                )
    
    fig.update_layout(
        title="Asset Performance Analysis",
        height=700,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Normalized Price (Base=100)", row=1, col=1)
    fig.update_yaxes(title_text="Daily Return (%)", row=2, col=1)
    
    return fig

def create_risk_metrics_chart(metrics):
    """Create risk metrics visualization."""
    risk_data = []
    
    for asset, asset_metrics in metrics.items():
        if 'Sharpe_Ratio' in asset_metrics and asset_metrics['Sharpe_Ratio']:
            sharpe = asset_metrics['Sharpe_Ratio'].get('Sharpe_Ratio', 0)
            annual_vol = asset_metrics['Sharpe_Ratio'].get('Annual_Volatility', 0) * 100
            annual_return = asset_metrics['Sharpe_Ratio'].get('Annual_Excess_Return', 0) * 100
            
            max_dd = 0
            if 'Maximum_Drawdown' in asset_metrics and asset_metrics['Maximum_Drawdown']:
                max_dd = abs(asset_metrics['Maximum_Drawdown'].get('Max_Drawdown_Pct', 0))
            
            risk_data.append({
                'Asset': asset,
                'Sharpe_Ratio': sharpe,
                'Annual_Volatility': annual_vol,
                'Annual_Return': annual_return,
                'Max_Drawdown': max_dd
            })
    
    if not risk_data:
        return None
    
    df_risk = pd.DataFrame(risk_data)
    
    # Create risk-return scatter plot
    fig = px.scatter(
        df_risk,
        x='Annual_Volatility',
        y='Annual_Return',
        size='Max_Drawdown',
        color='Sharpe_Ratio',
        hover_data=['Asset'],
        title="Risk-Return Profile",
        labels={
            'Annual_Volatility': 'Annual Volatility (%)',
            'Annual_Return': 'Annual Return (%)',
            'Sharpe_Ratio': 'Sharpe Ratio'
        },
        color_continuous_scale='RdYlGn'
    )
    
    # Add asset labels
    for _, row in df_risk.iterrows():
        fig.add_annotation(
            x=row['Annual_Volatility'],
            y=row['Annual_Return'],
            text=row['Asset'],
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="black"
        )
    
    fig.update_layout(height=500)
    return fig

def create_correlation_heatmap(data):
    """Create correlation heatmap."""
    returns_data = {}
    for asset, df in data.items():
        if 'Daily_Return' in df.columns:
            returns_data[asset] = df['Daily_Return']
    
    if len(returns_data) < 2:
        return None
    
    returns_df = pd.DataFrame(returns_data).dropna()
    correlation_matrix = returns_df.corr()
    
    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu',
        title="Asset Correlation Matrix"
    )
    
    fig.update_layout(height=400)
    return fig

def create_var_comparison(metrics):
    """Create VaR comparison chart."""
    var_data = []
    
    for asset, asset_metrics in metrics.items():
        if 'VaR' in asset_metrics:
            var_metrics = asset_metrics['VaR']
            for var_type, var_result in var_metrics.items():
                if var_result and 'VaR_Annual' in var_result:
                    var_data.append({
                        'Asset': asset,
                        'VaR_Type': var_type,
                        'VaR_Annual': abs(var_result['VaR_Annual'] * 100),
                        'Confidence_Level': var_result.get('Confidence_Level', 0.95) * 100
                    })
    
    if not var_data:
        return None
    
    df_var = pd.DataFrame(var_data)
    
    fig = px.bar(
        df_var,
        x='Asset',
        y='VaR_Annual',
        color='VaR_Type',
        title="Value at Risk Comparison (Annual %)",
        labels={'VaR_Annual': 'VaR (%)'},
        barmode='group'
    )
    
    fig.update_layout(height=400)
    return fig

def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Portfolio Management Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Time Series Forecasting for Portfolio Management Optimization**")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Dashboard Controls")
    st.sidebar.markdown("### Asset Selection")
    
    # Load data
    with st.spinner("Loading financial data..."):
        processed_data, data_summary = load_financial_data()
    
    if processed_data is None:
        st.error("Failed to load data. Please check your internet connection and try again.")
        return
    
    # Asset selection
    available_assets = list(processed_data.keys())
    selected_assets = st.sidebar.multiselect(
        "Select assets to analyze:",
        available_assets,
        default=available_assets
    )
    
    if not selected_assets:
        st.warning("Please select at least one asset to analyze.")
        return
    
    # Calculate metrics
    with st.spinner("Calculating financial metrics..."):
        metrics = calculate_metrics(processed_data)
    
    # Main dashboard layout
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Performance", "⚠️ Risk Analysis", "🔗 Correlations"])
    
    with tab1:
        st.header("Portfolio Overview")
        
        # Key metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Assets Analyzed",
                value=len(selected_assets)
            )
        
        with col2:
            if data_summary is not None and not data_summary.empty:
                total_records = data_summary['Total Records'].sum()
                st.metric(
                    label="Total Data Points",
                    value=f"{total_records:,}"
                )
        
        with col3:
            if processed_data:
                date_range = max(len(df) for df in processed_data.values())
                years = date_range / 252  # Approximate trading days per year
                st.metric(
                    label="Analysis Period",
                    value=f"{years:.1f} years"
                )
        
        with col4:
            st.metric(
                label="Last Updated",
                value=datetime.now().strftime("%Y-%m-%d")
            )
        
        # Data summary table
        if data_summary is not None:
            st.subheader("Asset Summary")
            filtered_summary = data_summary[data_summary['Symbol'].isin(selected_assets)]
            st.dataframe(filtered_summary, use_container_width=True)
    
    with tab2:
        st.header("Performance Analysis")
        
        # Price performance chart
        price_fig = create_price_chart(processed_data, selected_assets)
        st.plotly_chart(price_fig, use_container_width=True)
        
        # Performance metrics
        if metrics:
            st.subheader("Performance Metrics")
            
            perf_data = []
            for asset in selected_assets:
                if asset in metrics and 'Sharpe_Ratio' in metrics[asset]:
                    sharpe_data = metrics[asset]['Sharpe_Ratio']
                    basic_stats = metrics[asset].get('Basic_Statistics', {})
                    
                    perf_data.append({
                        'Asset': asset,
                        'Annual Return (%)': f"{sharpe_data.get('Annual_Excess_Return', 0) * 100:.2f}",
                        'Annual Volatility (%)': f"{sharpe_data.get('Annual_Volatility', 0) * 100:.2f}",
                        'Sharpe Ratio': f"{sharpe_data.get('Sharpe_Ratio', 0):.3f}",
                        'Skewness': f"{basic_stats.get('Skewness', 0):.3f}",
                        'Kurtosis': f"{basic_stats.get('Kurtosis', 0):.3f}"
                    })
            
            if perf_data:
                perf_df = pd.DataFrame(perf_data)
                st.dataframe(perf_df, use_container_width=True)
    
    with tab3:
        st.header("Risk Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk-return scatter plot
            if metrics:
                risk_fig = create_risk_metrics_chart(metrics)
                if risk_fig:
                    st.plotly_chart(risk_fig, use_container_width=True)
        
        with col2:
            # VaR comparison
            if metrics:
                var_fig = create_var_comparison(metrics)
                if var_fig:
                    st.plotly_chart(var_fig, use_container_width=True)
        
        # Risk metrics table
        if metrics:
            st.subheader("Risk Metrics Summary")
            
            risk_data = []
            for asset in selected_assets:
                if asset in metrics:
                    asset_metrics = metrics[asset]
                    
                    # Get VaR data
                    var_95 = 0
                    if 'VaR' in asset_metrics and 'Historical_95' in asset_metrics['VaR']:
                        var_95 = abs(asset_metrics['VaR']['Historical_95'].get('VaR_Annual', 0) * 100)
                    
                    # Get drawdown data
                    max_dd = 0
                    if 'Maximum_Drawdown' in asset_metrics and asset_metrics['Maximum_Drawdown']:
                        max_dd = abs(asset_metrics['Maximum_Drawdown'].get('Max_Drawdown_Pct', 0))
                    
                    # Get Sortino ratio
                    sortino = 0
                    if 'Sortino_Ratio' in asset_metrics and asset_metrics['Sortino_Ratio']:
                        sortino = asset_metrics['Sortino_Ratio'].get('Sortino_Ratio', 0)
                    
                    risk_data.append({
                        'Asset': asset,
                        'VaR 95% (Annual %)': f"{var_95:.2f}",
                        'Max Drawdown (%)': f"{max_dd:.2f}",
                        'Sortino Ratio': f"{sortino:.3f}"
                    })
            
            if risk_data:
                risk_df = pd.DataFrame(risk_data)
                st.dataframe(risk_df, use_container_width=True)
    
    with tab4:
        st.header("Correlation Analysis")
        
        # Correlation heatmap
        corr_fig = create_correlation_heatmap(processed_data)
        if corr_fig:
            st.plotly_chart(corr_fig, use_container_width=True)
        
        # Correlation insights
        st.subheader("Correlation Insights")
        
        returns_data = {}
        for asset in selected_assets:
            if asset in processed_data and 'Daily_Return' in processed_data[asset].columns:
                returns_data[asset] = processed_data[asset]['Daily_Return']
        
        if len(returns_data) >= 2:
            returns_df = pd.DataFrame(returns_data).dropna()
            correlation_matrix = returns_df.corr()
            
            # Find highest and lowest correlations
            correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    asset1 = correlation_matrix.columns[i]
                    asset2 = correlation_matrix.columns[j]
                    corr_value = correlation_matrix.iloc[i, j]
                    correlations.append((asset1, asset2, corr_value))
            
            correlations.sort(key=lambda x: abs(x[2]), reverse=True)
            
            if correlations:
                st.write("**Strongest Correlations:**")
                for asset1, asset2, corr in correlations[:3]:
                    correlation_strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.3 else "Weak"
                    correlation_direction = "Positive" if corr > 0 else "Negative"
                    st.write(f"• {asset1} vs {asset2}: {corr:.3f} ({correlation_strength} {correlation_direction})")
    
    # Footer
    st.markdown("---")
    st.markdown("**Dashboard Features:**")
    st.markdown("• Real-time financial data analysis • Risk metrics calculation • Interactive visualizations • Portfolio correlation analysis")

if __name__ == "__main__":
    main()
