import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from process_capex import process_capex_from_dataframe
import io
import base64
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Capex Data Pipeline - Processing & Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
    .file-info {
        background-color: #e3f2fd;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def create_download_link(df, filename, link_text):
    """Create a download link for a DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="text-decoration: none; color: #1f77b4; font-weight: bold;">{link_text}</a>'
    return href

def display_dataframe_summary(df, title):
    """Display a summary of a DataFrame"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        if 'AssetItemAmount' in df.columns:
            total_amount = df['AssetItemAmount'].sum()
            st.metric("Total Amount", f"₹{total_amount:,.2f}")
        else:
            st.metric("Data Type", "Mapping Data")
    with col4:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")

def load_office_locations():
    """Load office locations from fixed file"""
    try:
        if os.path.exists('office_location.csv'):
            return pd.read_csv('office_location.csv')
        else:
            st.error("Office location file (office_location.csv) not found. Please ensure the file exists in the project directory.")
            return None
    except Exception as e:
        st.error(f"Error loading office locations: {str(e)}")
        return None

def main():
    # Main header
    st.markdown('<h1 class="main-header">📊 Capex Data Pipeline - Processing & Analysis</h1>', unsafe_allow_html=True)
    
    # Sidebar for file uploads
    st.sidebar.markdown("## 📁 File Upload")
    
    # File uploader
    raw_data_file = st.sidebar.file_uploader(
        "Upload Raw Capex Data (CSV)",
        type=['csv'],
        help="Upload the raw Capex data file"
    )
    
    # Office locations info
    st.sidebar.info("ℹ️ Office locations are loaded from the fixed 'office_location.csv' file in the project directory.")
    
    # Process button
    process_button = st.sidebar.button("🚀 Process Data", type="primary")
    
    # Initialize session state
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'pivot_data' not in st.session_state:
        st.session_state.pivot_data = None
    if 'raw_data' not in st.session_state:
        st.session_state.raw_data = None
    if 'office_locations' not in st.session_state:
        st.session_state.office_locations = None
    if 'amc_items' not in st.session_state:
        st.session_state.amc_items = None
    if 'sorter_items' not in st.session_state:
        st.session_state.sorter_items = None
    if 'rental_items' not in st.session_state:
        st.session_state.rental_items = None
    if 'processing_log' not in st.session_state:
        st.session_state.processing_log = []
    
    # Process data when button is clicked
    if process_button:
        if raw_data_file is not None:
            try:
                # Load uploaded files
                raw_data = pd.read_csv(raw_data_file)
                st.session_state.raw_data = raw_data
                
                # Load office locations from fixed file
                office_locations = load_office_locations()
                if office_locations is not None:
                    st.session_state.office_locations = office_locations
                else:
                    st.error("Cannot proceed without office locations data.")
                    return
                
                # Process the data
                with st.spinner("Processing data... This may take a few moments."):
                    processed_data, pivot_data = process_capex_from_dataframe(
                        raw_data
                    )
                
                st.session_state.processed_data = processed_data
                st.session_state.pivot_data = pivot_data
                
                # Load additional files if they exist
                if os.path.exists('amc_items.csv'):
                    st.session_state.amc_items = pd.read_csv('amc_items.csv')
                if os.path.exists('sorter_items.csv'):
                    st.session_state.sorter_items = pd.read_csv('sorter_items.csv')
                if os.path.exists('rental_opex_items.csv'):
                    st.session_state.rental_items = pd.read_csv('rental_opex_items.csv')
                
                # Add to processing log
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.processing_log.append({
                    'timestamp': timestamp,
                    'initial_records': len(raw_data),
                    'final_records': len(processed_data),
                    'total_amount': processed_data['AssetItemAmount'].sum()
                })
                
                st.success("✅ Data processed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error processing data: {str(e)}")
                st.exception(e)
        else:
            st.error("❌ Please upload a raw data file first.")
    
    # Main content area - Show only generated sheets
    if st.session_state.processed_data is not None:
        st.markdown('<h2 class="section-header">📤 Generated Sheets & Analysis</h2>', unsafe_allow_html=True)
        
        # Main output tabs
        output_tab1, output_tab2, output_tab3, output_tab4, output_tab5 = st.tabs([
            "📊 Processed Data", 
            "📈 Pivot Table", 
            "🔧 Specialized Items",
            "📊 Analytics & Charts",
            "📋 Processing Log"
        ])
        
        with output_tab1:
            st.markdown("### Processed Capex Data")
            display_dataframe_summary(st.session_state.processed_data, "Processed Data")
            
            # Show all columns
            st.markdown("#### All Data (All Columns)")
            st.dataframe(st.session_state.processed_data, use_container_width=True)
            
            # Download link
            st.markdown(create_download_link(
                st.session_state.processed_data, 
                "processed_capex_data.csv", 
                "📥 Download Processed Data"
            ), unsafe_allow_html=True)
            
        with output_tab2:
            st.markdown("### Pivot Table")
            display_dataframe_summary(st.session_state.pivot_data, "Pivot Table")
            
            # Display pivot table with all columns
            st.markdown("#### Pivot Table (All Columns)")
            st.dataframe(st.session_state.pivot_data, use_container_width=True)
            
            # Download link
            st.markdown(create_download_link(
                st.session_state.pivot_data, 
                "capex_pivot_table.csv", 
                "📥 Download Pivot Table"
            ), unsafe_allow_html=True)
            
        with output_tab3:
            st.markdown("### Specialized Items")
            
            # AMC Items
            if st.session_state.amc_items is not None:
                st.markdown("#### AMC Items (All Columns)")
                display_dataframe_summary(st.session_state.amc_items, "AMC Items")
                st.dataframe(st.session_state.amc_items, use_container_width=True)
                st.markdown(create_download_link(
                    st.session_state.amc_items, 
                    "amc_items.csv", 
                    "📥 Download AMC Items"
                ), unsafe_allow_html=True)
            else:
                st.info("No AMC items found in the data.")
            
            # Sorter Items
            if st.session_state.sorter_items is not None:
                st.markdown("#### Sorter Items (All Columns)")
                display_dataframe_summary(st.session_state.sorter_items, "Sorter Items")
                st.dataframe(st.session_state.sorter_items, use_container_width=True)
                st.markdown(create_download_link(
                    st.session_state.sorter_items, 
                    "sorter_items.csv", 
                    "📥 Download Sorter Items"
                ), unsafe_allow_html=True)
            else:
                st.info("No Sorter items found in the data.")
            
            # Rental Items
            if st.session_state.rental_items is not None:
                st.markdown("#### Rental Opex Items (All Columns)")
                display_dataframe_summary(st.session_state.rental_items, "Rental Items")
                st.dataframe(st.session_state.rental_items, use_container_width=True)
                st.markdown(create_download_link(
                    st.session_state.rental_items, 
                    "rental_opex_items.csv", 
                    "📥 Download Rental Items"
                ), unsafe_allow_html=True)
            else:
                st.info("No Rental Opex items found in the data.")
            
        with output_tab4:
            st.markdown("### Analytics & Visualizations")
            
            # Zone distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Zone Distribution")
                zone_counts = st.session_state.processed_data['Zone'].value_counts()
                fig_zone = px.pie(
                    values=zone_counts.values, 
                    names=zone_counts.index,
                    title="Records by Zone"
                )
                st.plotly_chart(fig_zone, use_container_width=True)
            
            with col2:
                st.markdown("#### Asset Category Distribution")
                asset_counts = st.session_state.processed_data['AssetCategoryName'].value_counts()
                fig_asset = px.bar(
                    x=asset_counts.index, 
                    y=asset_counts.values,
                    title="Records by Asset Category"
                )
                st.plotly_chart(fig_asset, use_container_width=True)
            
            # Amount distribution
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown("#### Zone-wise Capex Amount")
                zone_amounts = st.session_state.processed_data.groupby('Zone')['AssetItemAmount'].sum().sort_values(ascending=False)
                fig_amount = px.bar(
                    x=zone_amounts.index, 
                    y=zone_amounts.values,
                    title="Capex Amount by Zone"
                )
                fig_amount.update_layout(yaxis_title="Amount (₹)")
                st.plotly_chart(fig_amount, use_container_width=True)
            
            with col4:
                st.markdown("#### Request Function Distribution")
                function_counts = st.session_state.processed_data['RequestFunction'].value_counts()
                fig_function = px.pie(
                    values=function_counts.values, 
                    names=function_counts.index,
                    title="Records by Request Function"
                )
                st.plotly_chart(fig_function, use_container_width=True)
            
            # Detailed statistics
            st.markdown("#### Detailed Statistics")
            
            # Zone-wise summary
            st.markdown("##### Zone-wise Summary")
            zone_summary = st.session_state.processed_data.groupby('Zone').agg({
                'AssetItemAmount': ['count', 'sum', 'mean']
            }).round(2)
            zone_summary.columns = ['Count', 'Total_Amount', 'Average_Amount']
            st.dataframe(zone_summary, use_container_width=True)
            
            # Asset category summary
            st.markdown("##### Asset Category Summary")
            asset_summary = st.session_state.processed_data.groupby('AssetCategoryName').agg({
                'AssetItemAmount': ['count', 'sum', 'mean']
            }).round(2)
            asset_summary.columns = ['Count', 'Total_Amount', 'Average_Amount']
            st.dataframe(asset_summary, use_container_width=True)
            
            # Request function summary
            st.markdown("##### Request Function Summary")
            function_summary = st.session_state.processed_data.groupby('RequestFunction').agg({
                'AssetItemAmount': ['count', 'sum', 'mean']
            }).round(2)
            function_summary.columns = ['Count', 'Total_Amount', 'Average_Amount']
            st.dataframe(function_summary, use_container_width=True)
            
        with output_tab5:
            st.markdown("### Processing Log")
            
            if st.session_state.processing_log:
                log_df = pd.DataFrame(st.session_state.processing_log)
                st.dataframe(log_df, use_container_width=True)
                
                # Download log
                st.markdown(create_download_link(
                    log_df, 
                    "processing_log.csv", 
                    "📥 Download Processing Log"
                ), unsafe_allow_html=True)
            else:
                st.info("No processing log available yet.")
    
    else:
        # Welcome message
        st.markdown("""
        <div class="metric-card">
            <h3>Welcome to Capex Data Pipeline! 🚀</h3>
            <p>This application processes raw Capex data according to business rules and generates clean, analyzed data with multiple output sheets.</p>
            <p><strong>To get started:</strong></p>
            <ol>
                <li>Upload your raw Capex data file (CSV format)</li>
                <li>Click the "Process Data" button</li>
                <li>View and download all generated sheets and analysis</li>
            </ol>
            <p><strong>Note:</strong> Office locations are automatically loaded from the fixed 'office_location.csv' file.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Features section
        st.markdown('<h2 class="section-header">✨ Generated Outputs</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **📊 Main Outputs**
            - Processed Capex Data
            - Pivot Table Analysis
            - Zone/Region Mapping
            - Asset Category Breakdown
            """)
        
        with col2:
            st.markdown("""
            **🔧 Specialized Items**
            - AMC Items (Annual Maintenance)
            - Sorter Items (Sorting Equipment)
            - Rental Opex Items
            - Plant & Machinery Sub-categories
            """)
        
        with col3:
            st.markdown("""
            **📈 Analytics & Reports**
            - Interactive Charts
            - Summary Statistics
            - Processing Log
            - Download Options
            """)
        
        # Instructions
        st.markdown('<h2 class="section-header">📋 Instructions</h2>', unsafe_allow_html=True)
        st.markdown("""
        1. **Upload Raw Data**: Use the sidebar to upload your raw Capex data CSV file
        2. **Process Data**: Click the "Process Data" button to run the pipeline
        3. **View Results**: Navigate through the tabs to see all generated sheets
        4. **Download**: Use the download links to save any sheet you need
        
        **Note:** Office locations are automatically loaded from the fixed 'office_location.csv' file in the project directory.
        """)

if __name__ == "__main__":
    main()
