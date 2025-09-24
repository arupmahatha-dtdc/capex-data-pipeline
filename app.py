import streamlit as st
import pandas as pd
import numpy as np
from process_capex import process_capex_from_dataframe, validate_processed_data, validate_all_sheets_composite_keys
import io
import base64
import os
from datetime import datetime

# Robust CSV loading with encoding fallbacks
def read_csv_with_fallback(source):
    """Read CSV from a Streamlit UploadedFile or file path with encoding fallbacks.

    Tries UTF-8, then UTF-8 with BOM, then Windows-1252, then Latin-1.
    """
    encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin1"]

    # Uploaded file from Streamlit
    try:
        import streamlit.runtime.uploaded_file_manager  # type: ignore
        is_streamlit_uploaded = hasattr(source, "getvalue") and callable(source.getvalue)
    except Exception:
        is_streamlit_uploaded = hasattr(source, "getvalue") and callable(source.getvalue)

    if is_streamlit_uploaded:
        data_bytes = source.getvalue()
        last_error = None
        for enc in encodings_to_try:
            try:
                return pd.read_csv(io.BytesIO(data_bytes), encoding=enc)
            except Exception as e:
                last_error = e
                continue
        # Last resort: replace invalid characters
        try:
            decoded = data_bytes.decode("utf-8", errors="replace")
            return pd.read_csv(io.StringIO(decoded))
        except Exception:
            raise last_error if last_error else UnicodeDecodeError("utf-8", b"", 0, 1, "decode failed")

    # Local file path
    if isinstance(source, str):
        last_error = None
        for enc in encodings_to_try:
            try:
                return pd.read_csv(source, encoding=enc)
            except Exception as e:
                last_error = e
                continue
        # Last resort with replacement
        try:
            with open(source, "r", encoding="utf-8", errors="replace") as f:
                return pd.read_csv(f)
        except Exception:
            if last_error:
                raise last_error
            raise

    # Unsupported source type
    raise ValueError("Unsupported CSV source type for read_csv_with_fallback")

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


def load_office_locations():
    """Load office locations from fixed file"""
    try:
        if os.path.exists('office_location.csv'):
            return read_csv_with_fallback('office_location.csv')
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
    
    # Validation section
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🔍 Data Validation")
    
    # Reference data uploader
    reference_data_file = st.sidebar.file_uploader(
        "Upload Reference Data (CSV)",
        type=['csv'],
        help="Upload reference data (e.g., final_data.csv) to validate against processed output",
        key="reference_uploader"
    )
    
    # Validate button
    validate_button = st.sidebar.button("🔍 Validate Data", type="secondary")
    
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
    if 'validation_results' not in st.session_state:
        st.session_state.validation_results = None
    if 'reference_data' not in st.session_state:
        st.session_state.reference_data = None
    
    # Process data when button is clicked
    if process_button:
        if raw_data_file is not None:
            try:
                # Load uploaded raw data with encoding fallbacks
                raw_data = read_csv_with_fallback(raw_data_file)
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
                    processed_data, pivot_data, amc_items, sorter_items, rental_items = process_capex_from_dataframe(
                        raw_data
                    )
                
                st.session_state.processed_data = processed_data
                st.session_state.pivot_data = pivot_data
                st.session_state.amc_items = amc_items if len(amc_items) > 0 else None
                st.session_state.sorter_items = sorter_items if len(sorter_items) > 0 else None
                st.session_state.rental_items = rental_items if len(rental_items) > 0 else None
                
                
                st.success("✅ Data processed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error processing data: {str(e)}")
                st.exception(e)
        else:
            st.error("❌ Please upload a raw data file first.")
    
    # Validation logic
    if validate_button:
        if st.session_state.processed_data is not None and reference_data_file is not None:
            try:
                # Load reference data with encoding fallbacks
                reference_data = read_csv_with_fallback(reference_data_file)
                st.session_state.reference_data = reference_data
                
                # Run comprehensive validation
                with st.spinner("Running comprehensive validation across all sheets... This may take a few moments."):
                    validation_results = validate_all_sheets_composite_keys(
                        st.session_state.raw_data,  # Input data
                        st.session_state.processed_data,  # Processed data
                        reference_data  # Reference data
                    )
                
                st.session_state.validation_results = validation_results
                st.success("✅ Comprehensive validation completed!")
                
            except Exception as e:
                st.error(f"❌ Error during validation: {str(e)}")
                st.exception(e)
        elif st.session_state.processed_data is None:
            st.error("❌ Please process data first before validation.")
        elif reference_data_file is None:
            st.error("❌ Please upload reference data file for validation.")
    
    # Main content area - Show only generated sheets
    if st.session_state.processed_data is not None:
        st.markdown('<h2 class="section-header">📤 Generated Sheets & Analysis</h2>', unsafe_allow_html=True)
        
        # Main output tabs
        output_tab1, output_tab2, output_tab3, output_tab4, output_tab5 = st.tabs([
            "📊 Processed Data", 
            "📈 Pivot Table", 
            "🔧 Specialized Items",
            "📊 Analytics & Summary",
            "🔍 Data Validation"
        ])
        
        with output_tab1:
            st.markdown("### Processed Capex Data")
            
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
                st.dataframe(st.session_state.rental_items, use_container_width=True)
                st.markdown(create_download_link(
                    st.session_state.rental_items, 
                    "rental_opex_items.csv", 
                    "📥 Download Rental Items"
                ), unsafe_allow_html=True)
            else:
                st.info("No Rental Opex items found in the data.")
            
        with output_tab4:
            st.markdown("### Analytics & Summary")
            
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
            category_col = 'AssetCategoryName_2' if 'AssetCategoryName_2' in st.session_state.processed_data.columns else 'AssetCategoryName'
            asset_summary = st.session_state.processed_data.groupby(category_col).agg({
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
            st.markdown("### Comprehensive Data Validation Results")
            
            if st.session_state.validation_results is not None:
                validation_results = st.session_state.validation_results
                
                # Overall validation summary
                st.markdown("#### ML Metrics Summary")
                summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                
                with summary_col1:
                    status_color = "🟢" if validation_results['summary']['overall_status'] == 'PASS' else "🔴"
                    st.metric("Overall Status", f"{status_color} {validation_results['summary']['overall_status']}")
                
                with summary_col2:
                    st.metric("Precision", f"{validation_results['summary']['precision']:.4f}")
                
                with summary_col3:
                    st.metric("Recall", f"{validation_results['summary']['recall']:.4f}")
                
                with summary_col4:
                    st.metric("F1-Score", f"{validation_results['summary']['f1_score']:.4f}")
                
                # Removed per request: Additional Metrics and Individual Sheet Validation
                
                # ML Validation Results
                st.markdown("#### ML Validation Results")
                ml_val = validation_results.get('ml_validation', {})
                if ml_val and ml_val.get('ml_metrics'):
                    ml_metrics = ml_val['ml_metrics']
                    
                    # True Positives, False Positives, False Negatives
                    st.markdown("##### Confusion Matrix Components")
                    cm_col1, cm_col2, cm_col3 = st.columns(3)
                    with cm_col1:
                        st.metric("True Positives", ml_metrics.get('true_positives', 0))
                    with cm_col2:
                        st.metric("False Positives", ml_metrics.get('false_positives', 0))
                    with cm_col3:
                        st.metric("False Negatives", ml_metrics.get('false_negatives', 0))
                    
                    # Field-level accuracy if available
                    if 'amount_accuracy' in ml_metrics:
                        st.markdown("##### Field-Level Accuracy")
                        field_col1, field_col2, field_col3 = st.columns(3)
                        with field_col1:
                            st.metric("Amount Accuracy", f"{ml_metrics.get('amount_accuracy', 0):.2f}%")
                        with field_col2:
                            st.metric("Zone Accuracy", f"{ml_metrics.get('zone_accuracy', 0):.2f}%")
                        with field_col3:
                            st.metric("Category Accuracy", f"{ml_metrics.get('category_accuracy', 0):.2f}%")
                else:
                    st.info("ML validation results not available")
                
                # Mismatches table
                st.markdown("#### Mismatches Found")
                if validation_results['all_mismatches']:
                    # Convert mismatches to DataFrame for better display
                    all_mismatches_df = pd.DataFrame(validation_results['all_mismatches'])
                    
                    # Display detailed mismatches in two separate tables
                    if not all_mismatches_df.empty:
                        # Split into two tables based on mismatch type
                        false_positives = all_mismatches_df[
                            all_mismatches_df['type'] == 'False Positive'
                        ]
                        false_negatives = all_mismatches_df[
                            all_mismatches_df['type'] == 'False Negative'
                        ]
                        
                        # Table 1: False Positives (Only in Processed)
                        st.markdown("##### False Positives (Incorrectly Included)")
                        if not false_positives.empty:
                            # Prefer to show all components of the composite key
                            cols_fp = [c for c in ['RequestNo', 'AssetItemName', 'VendorName', 'CompositeKey'] if c in false_positives.columns]
                            fp_display = false_positives[cols_fp].copy()
                            st.dataframe(fp_display, use_container_width=True)
                            
                            # Download false positive mismatches
                            st.markdown(create_download_link(
                                fp_display, 
                                "false_positives_mismatches.csv", 
                                "📥 Download False Positives"
                            ), unsafe_allow_html=True)
                        else:
                            st.success("✅ No False Positives found!")
                        
                        # Table 2: False Negatives (Only in Reference)
                        st.markdown("##### False Negatives (Incorrectly Excluded)")
                        if not false_negatives.empty:
                            # Include exclusion reason if available
                            cols = [c for c in ['RequestNo', 'AssetItemName', 'VendorName', 'CompositeKey'] if c in false_negatives.columns]
                            # Try to expand structured reason into columns if present
                            if 'exclusion_reason' in false_negatives.columns:
                                # Normalize dict-like reasons to columns
                                def unpack_reason(x):
                                    if isinstance(x, dict):
                                        return x
                                    return {'label': x}
                                reason_df = false_negatives['exclusion_reason'].apply(unpack_reason).apply(pd.Series)
                                # Merge for display
                                fn_display = pd.concat([false_negatives[cols].reset_index(drop=True), reason_df.reset_index(drop=True)], axis=1)
                                # Friendly headers
                                fn_display = fn_display.rename(columns={
                                    'label': 'Excluded By (rules.txt)',
                                    'column': 'Trigger Column',
                                    'value': 'Trigger Value'
                                })
                            else:
                                fn_display = false_negatives[cols].copy()
                            st.dataframe(fn_display, use_container_width=True)
                            
                            # Download false negative mismatches
                            st.markdown(create_download_link(
                                fn_display, 
                                "false_negatives_mismatches.csv", 
                                "📥 Download False Negatives"
                            ), unsafe_allow_html=True)
                        else:
                            st.success("✅ No False Negatives found!")
                    else:
                        st.success("✅ No mismatches found between Processed and Reference data!")
                    
                    # Show count of mismatches
                    total_mismatches = len(all_mismatches_df)
                    false_positive_count = len(false_positives)
                    false_negative_count = len(false_negatives)
                    st.info(f"Showing {false_positive_count} False Positives and {false_negative_count} False Negatives out of {total_mismatches} total mismatches.")
                    
                else:
                    st.success("🎉 No mismatches found! All validations passed completely.")
                
                # Removed per request: Duplicate Keys Details section
                
            else:
                st.info("No validation results available. Please upload reference data and run validation.")
                
                if st.session_state.reference_data is not None:
                    st.markdown("#### Reference Data Summary")
            
    
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
                <li><strong>Optional:</strong> Upload reference data (e.g., final_data.csv) to validate output accuracy</li>
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
            - Summary Statistics
            - Data Validation
            - Processing Log
            - Download Options
            """)
        
        # Instructions
        st.markdown('<h2 class="section-header">📋 Instructions</h2>', unsafe_allow_html=True)
        st.markdown("""
        1. **Upload Raw Data**: Use the sidebar to upload your raw Capex data CSV file
        2. **Process Data**: Click the "Process Data" button to run the pipeline
        3. **View Results**: Navigate through the tabs to see all generated sheets
        4. **Validate Data** (Optional): Upload reference data (e.g., final_data.csv) and click "Validate Data" to check accuracy
        5. **Download**: Use the download links to save any sheet you need
        
        **Note:** Office locations are automatically loaded from the fixed 'office_location.csv' file in the project directory.
        """)

if __name__ == "__main__":
    main()