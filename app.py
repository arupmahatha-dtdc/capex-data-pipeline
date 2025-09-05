import streamlit as st
import pandas as pd
from process_capex import process_capex

st.title("Capex Data Processor")

uploaded_file = st.file_uploader("Upload Raw Capex CSV", type=["csv"])
branch_file = st.file_uploader("Upload Branch Mapping CSV (optional)", type=["csv"])

if uploaded_file:
    # Save uploaded files temporarily
    raw_path = "uploaded_raw.csv"
    with open(raw_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    branch_path = None
    if branch_file:
        branch_path = "uploaded_branch.csv"
        with open(branch_path, "wb") as f:
            f.write(branch_file.getbuffer())

    # Show input data
    st.subheader("Raw Input CSV")
    raw_df = pd.read_csv(raw_path)
    st.dataframe(raw_df)

    # Process
    processed_df = process_capex(raw_path, branch_path)

    # Show output data
    st.subheader("Processed Output CSV")
    st.dataframe(processed_df)

    # Download button
    csv = processed_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Processed CSV",
        data=csv,
        file_name="processed_capex.csv",
        mime="text/csv"
    )