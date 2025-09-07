import pandas as pd
import numpy as np
from datetime import datetime
import re

def load_office_locations():
    """Load office location mapping file"""
    print("Loading office location mapping...")
    office_locations = pd.read_csv('office_location.csv')
    print(f"Office locations loaded: {len(office_locations)} rows")
    return office_locations

def add_zone_region_mapping(df, office_locations):
    """Add Zone and Region columns by mapping branch codes"""
    print("Adding Zone and Region mapping...")
    
    # Create a mapping from branch code to zone/region
    # The office_locations has 'name' column with branch codes like 'A01', 'B35', etc.
    branch_to_zone = {}
    branch_to_region = {}
    
    for _, row in office_locations.iterrows():
        branch_code = row['office']  # The branch code is in the 'office' column
        zone = row['zone']
        region = row['region']
        branch_to_zone[branch_code] = zone
        branch_to_region[branch_code] = region
    
    # Map zone and region to the dataframe using BranchCode
    df['Zone'] = df['BranchCode'].map(branch_to_zone)
    df['Region'] = df['BranchCode'].map(branch_to_region)
    
    # Check for unmapped branch codes
    unmapped_mask = df['Zone'].isna()
    if unmapped_mask.any():
        print(f"Found {unmapped_mask.sum()} unmapped branch codes")
        unmapped_codes = df[unmapped_mask]['BranchCode'].unique()
        print(f"Unmapped codes: {unmapped_codes[:10]}...")
    
    # Fill remaining missing values with 'Unknown'
    df['Zone'] = df['Zone'].fillna('Unknown')
    df['Region'] = df['Region'].fillna('Unknown')
    
    print(f"Zone/Region mapping completed. Unique zones: {df['Zone'].nunique()}")
    print(f"Zone distribution: {df['Zone'].value_counts().to_dict()}")
    return df

def remove_rejected_capex(df):
    """Remove rows with 'Rejected' Capex status"""
    print("Removing rejected Capex requests...")
    initial_count = len(df)
    df = df[df['CurrentStatus'] != 'Rejected']
    removed_count = initial_count - len(df)
    print(f"Removed {removed_count} rejected requests. Remaining: {len(df)} rows")
    return df

def filter_asset_categories(df):
    """Keep only COMPUTER, PLANT & MACHINERY, LEASEHOLD in AssetCategoryName"""
    print("Filtering asset categories...")
    initial_count = len(df)
    
    # Define allowed asset categories
    allowed_categories = ['COMPUTER', 'PLANT & MACHINERY', 'LEASEHOLD']
    
    # Filter the dataframe
    df = df[df['AssetCategoryName'].isin(allowed_categories)]
    
    removed_count = initial_count - len(df)
    print(f"Removed {removed_count} rows with non-allowed asset categories. Remaining: {len(df)} rows")
    return df

def remove_unwanted_request_functions(df):
    """Remove CS, FA, Sales, Channel, Vigilance from RequestFunction after checking user remarks"""
    print("Removing unwanted request functions...")
    initial_count = len(df)
    
    # Define functions to remove
    functions_to_remove = ['CS', 'FA', 'Sales', 'Channel', 'Vigilance']
    
    # Remove rows with these functions
    df = df[~df['RequestFunction'].isin(functions_to_remove)]
    
    removed_count = initial_count - len(df)
    print(f"Removed {removed_count} rows with unwanted request functions. Remaining: {len(df)} rows")
    return df

def remove_dash_vendors(df):
    """Remove rows with '-' in IsSelectedVendor column"""
    print("Removing rows with '-' in IsSelectedVendor...")
    initial_count = len(df)
    df = df[df['IsSelectedVendor'] != '-']
    removed_count = initial_count - len(df)
    print(f"Removed {removed_count} rows with '-' in IsSelectedVendor. Remaining: {len(df)} rows")
    return df

def filter_it_requests(df):
    """Filter IT requests and remove non-relevant ones based on user remarks"""
    print("Filtering IT requests...")
    initial_count = len(df)
    
    # Get IT requests
    it_requests = df[df['RequestFunction'] == 'IT'].copy()
    non_it_requests = df[df['RequestFunction'] != 'IT'].copy()
    
    # Define keywords that indicate non-relevant IT requests
    non_relevant_keywords = [
        'test', 'demo', 'sample', 'trial', 'pilot', 'experimental',
        'personal', 'individual', 'non-business', 'non-operational'
    ]
    
    # Filter out non-relevant IT requests
    relevant_it_requests = it_requests.copy()
    for keyword in non_relevant_keywords:
        mask = relevant_it_requests['UserRemarks'].str.contains(keyword, case=False, na=False)
        relevant_it_requests = relevant_it_requests[~mask]
    
    # Combine relevant IT requests with non-IT requests
    df = pd.concat([relevant_it_requests, non_it_requests], ignore_index=True)
    
    removed_count = initial_count - len(df)
    print(f"Removed {removed_count} non-relevant IT requests. Remaining: {len(df)} rows")
    return df

def remove_approval_progress_requests(df):
    """Remove 'approval in progress' and 'sent for approval' status rows after checking user remarks"""
    print("Removing approval in progress and sent for approval requests...")
    initial_count = len(df)
    
    # Define statuses to remove
    statuses_to_remove = ['Approval in Progress', 'Sent for Approval']
    
    # Remove rows with these statuses
    df = df[~df['CurrentStatus'].isin(statuses_to_remove)]
    
    removed_count = initial_count - len(df)
    print(f"Removed {removed_count} rows with approval statuses. Remaining: {len(df)} rows")
    return df

def remove_dark_store_requests(df):
    """Remove rows with 'dark store' mentioned in UserRemarks"""
    print("Removing dark store requests...")
    initial_count = len(df)
    
    # Remove rows containing 'dark store' in user remarks
    mask = df['UserRemarks'].str.contains('dark store', case=False, na=False)
    df = df[~mask]
    
    removed_count = initial_count - len(df)
    print(f"Removed {removed_count} dark store requests. Remaining: {len(df)} rows")
    return df

def filter_admin_requests(df):
    """Filter Admin requests and remove non-relevant ones based on user remarks"""
    print("Filtering Admin requests...")
    initial_count = len(df)
    
    # Get Admin requests
    admin_requests = df[df['RequestFunction'] == 'Admin'].copy()
    non_admin_requests = df[df['RequestFunction'] != 'Admin'].copy()
    
    # Define keywords that indicate non-relevant Admin requests
    non_relevant_keywords = [
        'personal', 'individual', 'non-business', 'non-operational',
        'test', 'demo', 'sample'
    ]
    
    # Filter out non-relevant Admin requests
    relevant_admin_requests = admin_requests.copy()
    for keyword in non_relevant_keywords:
        mask = relevant_admin_requests['UserRemarks'].str.contains(keyword, case=False, na=False)
        relevant_admin_requests = relevant_admin_requests[~mask]
    
    # Combine relevant Admin requests with non-Admin requests
    df = pd.concat([relevant_admin_requests, non_admin_requests], ignore_index=True)
    
    removed_count = initial_count - len(df)
    print(f"Removed {removed_count} non-relevant Admin requests. Remaining: {len(df)} rows")
    return df

def filter_ops_requests(df):
    """Filter Ops requests and remove non-relevant ones based on user remarks"""
    print("Filtering Ops requests...")
    initial_count = len(df)
    
    # Get Ops requests
    ops_requests = df[df['RequestFunction'] == 'Ops'].copy()
    non_ops_requests = df[df['RequestFunction'] != 'Ops'].copy()
    
    # Define keywords that indicate non-relevant Ops requests
    non_relevant_keywords = [
        'personal', 'individual', 'non-business', 'non-operational',
        'test', 'demo', 'sample'
    ]
    
    # Filter out non-relevant Ops requests
    relevant_ops_requests = ops_requests.copy()
    for keyword in non_relevant_keywords:
        mask = relevant_ops_requests['UserRemarks'].str.contains(keyword, case=False, na=False)
        relevant_ops_requests = relevant_ops_requests[~mask]
    
    # Combine relevant Ops requests with non-Ops requests
    df = pd.concat([relevant_ops_requests, non_ops_requests], ignore_index=True)
    
    removed_count = initial_count - len(df)
    print(f"Removed {removed_count} non-relevant Ops requests. Remaining: {len(df)} rows")
    return df

def filter_ops_through_it_requests(df):
    """Filter 'Ops through IT' requests and remove non-relevant ones based on user remarks"""
    print("Filtering Ops through IT requests...")
    initial_count = len(df)
    
    # Get Ops through IT requests
    ops_it_requests = df[df['RequestFunction'] == 'Ops through IT'].copy()
    non_ops_it_requests = df[df['RequestFunction'] != 'Ops through IT'].copy()
    
    # Define keywords that indicate non-relevant Ops through IT requests
    non_relevant_keywords = [
        'personal', 'individual', 'non-business', 'non-operational',
        'test', 'demo', 'sample'
    ]
    
    # Filter out non-relevant Ops through IT requests
    relevant_ops_it_requests = ops_it_requests.copy()
    for keyword in non_relevant_keywords:
        mask = relevant_ops_it_requests['UserRemarks'].str.contains(keyword, case=False, na=False)
        relevant_ops_it_requests = relevant_ops_it_requests[~mask]
    
    # Combine relevant Ops through IT requests with others
    df = pd.concat([relevant_ops_it_requests, non_ops_it_requests], ignore_index=True)
    
    removed_count = initial_count - len(df)
    print(f"Removed {removed_count} non-relevant Ops through IT requests. Remaining: {len(df)} rows")
    return df

def separate_plant_machinery_items(df):
    """Separate X-Ray, DWS, Sorter, TBC from Plant & Machinery and calculate separately"""
    print("Separating specific items from Plant & Machinery...")
    
    # Define items to separate
    items_to_separate = ['X-Ray', 'DWS', 'Sorter', 'TBC']
    
    # Create a new column to track separated items
    df['AssetCategoryName_2'] = df['AssetCategoryName'].copy()
    
    # Check if any of these items are mentioned in AssetItemName
    for item in items_to_separate:
        mask = df['AssetItemName'].str.contains(item, case=False, na=False)
        df.loc[mask, 'AssetCategoryName_2'] = f'PLANT & MACHINERY - {item}'
    
    print("Plant & Machinery items separated successfully")
    return df

def handle_office_equipments(df):
    """In Office Equipments: change non-CCTV/FireEx/Projector/Chairs/AC/Fans/Stools to Plant & Machinery"""
    print("Handling Office Equipments categorization...")
    
    # Define items that should remain as Office Equipments
    office_equipment_items = ['CCTV', 'FireEx', 'Projector', 'Chairs', 'AC', 'Fans', 'Stools']
    
    # Find Office Equipments that should be changed to Plant & Machinery
    office_equipments_mask = df['AssetCategoryName'] == 'OFFICE EQUIPMENTS'
    
    # Check if the item is NOT one of the allowed office equipment items
    should_change_to_plant_machinery = office_equipments_mask.copy()
    
    for item in office_equipment_items:
        item_mask = df['AssetItemName'].str.contains(item, case=False, na=False)
        should_change_to_plant_machinery = should_change_to_plant_machinery & ~item_mask
    
    # Change the category
    df.loc[should_change_to_plant_machinery, 'AssetCategoryName'] = 'PLANT & MACHINERY'
    df.loc[should_change_to_plant_machinery, 'AssetCategoryName_2'] = 'PLANT & MACHINERY'
    
    # Remove dark store, counter, DS related items
    dark_store_keywords = ['dark store', 'counter', 'DS']
    for keyword in dark_store_keywords:
        mask = df['UserRemarks'].str.contains(keyword, case=False, na=False)
        df = df[~mask]
    
    print("Office Equipments categorization completed")
    return df

def add_mum_region_comments(df):
    """Add separate comments for MUM region as most Capex raised centrally for Pan-India"""
    print("Adding MUM region comments...")
    
    # Add comment for MUM region
    mum_mask = df['Region'] == 'MUMBAI'
    df.loc[mum_mask, 'UserRemarks'] = df.loc[mum_mask, 'UserRemarks'].astype(str) + ' [MUM Region - Centrally raised for Pan-India]'
    
    print("MUM region comments added")
    return df

def remove_non_ops_equipment(df):
    """Check AssetItemName & ItemCategory for removing CCTV or other equipment not part of Ops Capex budget"""
    print("Removing non-Ops equipment...")
    initial_count = len(df)
    
    # Define non-Ops equipment keywords
    non_ops_keywords = [
        'CCTV', 'Security', 'Surveillance', 'Personal', 'Individual',
        'Non-operational', 'Administrative only'
    ]
    
    # Remove rows with non-Ops equipment
    for keyword in non_ops_keywords:
        mask = (df['AssetItemName'].str.contains(keyword, case=False, na=False) |
                df['ItemCategory'].str.contains(keyword, case=False, na=False))
        df = df[~mask]
    
    removed_count = initial_count - len(df)
    print(f"Removed {removed_count} non-Ops equipment items. Remaining: {len(df)} rows")
    return df

def handle_amc_sorter_movement(df):
    """Check for AMC and Sorter movement and separate Rental Opex from overall utilization calculation"""
    print("Handling AMC and Sorter movement...")
    
    # Create separate dataframes for AMC and Sorter items
    amc_keywords = ['AMC', 'Annual Maintenance Contract', 'Maintenance Contract']
    sorter_keywords = ['Sorter', 'Sorting Machine', 'Sorting Equipment']
    
    # Identify AMC items
    amc_mask = df['AssetItemName'].str.contains('|'.join(amc_keywords), case=False, na=False)
    amc_items = df[amc_mask].copy()
    amc_items['Category_Type'] = 'AMC'
    
    # Identify Sorter items
    sorter_mask = df['AssetItemName'].str.contains('|'.join(sorter_keywords), case=False, na=False)
    sorter_items = df[sorter_mask].copy()
    sorter_items['Category_Type'] = 'Sorter'
    
    # Identify Rental Opex items
    rental_keywords = ['Rental', 'Lease', 'Hire', 'Rent']
    rental_mask = df['AssetItemName'].str.contains('|'.join(rental_keywords), case=False, na=False)
    rental_items = df[rental_mask].copy()
    rental_items['Category_Type'] = 'Rental_Opex'
    
    # Add category type to main dataframe
    df['Category_Type'] = 'Regular'
    df.loc[amc_mask, 'Category_Type'] = 'AMC'
    df.loc[sorter_mask, 'Category_Type'] = 'Sorter'
    df.loc[rental_mask, 'Category_Type'] = 'Rental_Opex'
    
    # Save separate files for analysis
    if len(amc_items) > 0:
        amc_items.to_csv('amc_items.csv', index=False)
        print(f"AMC items saved: {len(amc_items)} rows")
    
    if len(sorter_items) > 0:
        sorter_items.to_csv('sorter_items.csv', index=False)
        print(f"Sorter items saved: {len(sorter_items)} rows")
    
    if len(rental_items) > 0:
        rental_items.to_csv('rental_opex_items.csv', index=False)
        print(f"Rental Opex items saved: {len(rental_items)} rows")
    
    print("AMC and Sorter movement handling completed")
    return df

def create_pivot_table(df):
    """Create pivot table with Zone, Region, AssetCategoryName, AssetItemAmount, RequestDate"""
    print("Creating pivot table...")
    
    # Convert RequestDate to datetime for better analysis
    df['RequestDate'] = pd.to_datetime(df['RequestDate'], format='%d-%m-%Y', errors='coerce')
    
    # Create pivot table
    pivot_table = df.pivot_table(
        index=['Zone', 'Region', 'AssetCategoryName'],
        values=['AssetItemAmount'],
        aggfunc={'AssetItemAmount': 'sum'},
        fill_value=0
    ).reset_index()
    
    print(f"Pivot table created with {len(pivot_table)} rows")
    return pivot_table

def generate_summary_report(df):
    """Generate a summary report of the processed data"""
    print("\n" + "="*50)
    print("CAPEX DATA PROCESSING SUMMARY REPORT")
    print("="*50)
    
    # Basic statistics
    print(f"Total processed records: {len(df)}")
    print(f"Total Capex amount: ₹{df['AssetItemAmount'].sum():,.2f}")
    
    # Zone-wise summary
    print("\nZone-wise Summary:")
    zone_summary = df.groupby('Zone').agg({
        'AssetItemAmount': ['count', 'sum']
    }).round(2)
    zone_summary.columns = ['Count', 'Total_Amount']
    print(zone_summary)
    
    # Asset category summary
    print("\nAsset Category Summary:")
    asset_summary = df.groupby('AssetCategoryName').agg({
        'AssetItemAmount': ['count', 'sum']
    }).round(2)
    asset_summary.columns = ['Count', 'Total_Amount']
    print(asset_summary)
    
    # Request function summary
    print("\nRequest Function Summary:")
    function_summary = df.groupby('RequestFunction').agg({
        'AssetItemAmount': ['count', 'sum']
    }).round(2)
    function_summary.columns = ['Count', 'Total_Amount']
    print(function_summary)
    
    # Status summary
    print("\nStatus Summary:")
    status_summary = df.groupby('CurrentStatus').agg({
        'AssetItemAmount': ['count', 'sum']
    }).round(2)
    status_summary.columns = ['Count', 'Total_Amount']
    print(status_summary)
    
    print("="*50)

def process_capex_data(raw_data=None, office_locations=None):
    """Main function to process Capex data according to rules
    
    Args:
        raw_data (pd.DataFrame, optional): Raw Capex data. If None, will try to load from 'raw_data.csv'
        office_locations (pd.DataFrame, optional): Office location mapping. If None, will load from fixed 'office_location.csv'
    
    Returns:
        tuple: (processed_data, pivot_table)
    """
    print("Starting Capex data processing...")
    print("="*50)
    
    # Load or use provided data
    if raw_data is None:
        print("Loading raw data from file...")
        raw_data = pd.read_csv('raw_data.csv')
        print(f"Raw data loaded: {len(raw_data)} rows")
    else:
        print(f"Using provided raw data: {len(raw_data)} rows")
    
    # Always load office locations from fixed file
    if office_locations is None:
        office_locations = load_office_locations()
    else:
        print(f"Using provided office locations: {len(office_locations)} rows")
    
    # Start with raw data
    df = raw_data.copy()
    initial_count = len(df)
    print(f"Initial data: {initial_count} rows")
    
    # Apply all cleaning rules
    print("\nApplying data cleaning rules...")
    df = add_zone_region_mapping(df, office_locations)
    df = remove_rejected_capex(df)
    df = filter_asset_categories(df)
    df = remove_unwanted_request_functions(df)
    df = remove_dash_vendors(df)
    df = filter_it_requests(df)
    df = remove_approval_progress_requests(df)
    df = remove_dark_store_requests(df)
    df = filter_admin_requests(df)
    df = filter_ops_requests(df)
    df = filter_ops_through_it_requests(df)
    df = separate_plant_machinery_items(df)
    df = handle_office_equipments(df)
    df = add_mum_region_comments(df)
    df = remove_non_ops_equipment(df)
    df = handle_amc_sorter_movement(df)
    
    # Create pivot table
    pivot_table = create_pivot_table(df)
    
    # Save processed data
    df.to_csv('processed_capex_data.csv', index=False)
    pivot_table.to_csv('capex_pivot_table.csv', index=False)
    
    # Generate summary report
    generate_summary_report(df)
    
    print(f"\nProcessing completed!")
    print(f"Initial records: {initial_count}")
    print(f"Final records: {len(df)}")
    print(f"Records removed: {initial_count - len(df)}")
    print(f"Pivot table: {len(pivot_table)} rows")
    print("\nFiles saved:")
    print("- processed_capex_data.csv")
    print("- capex_pivot_table.csv")
    print("- amc_items.csv (if any)")
    print("- sorter_items.csv (if any)")
    print("- rental_opex_items.csv (if any)")
    
    return df, pivot_table

def process_capex_from_file(file_path):
    """Process Capex data from a specific CSV file
    
    Args:
        file_path (str): Path to the raw data CSV file
    
    Returns:
        tuple: (processed_data, pivot_table)
    """
    print(f"Loading data from {file_path}...")
    raw_data = pd.read_csv(file_path)
    
    # Office locations are always loaded from fixed file
    return process_capex_data(raw_data, None)

def process_capex_from_dataframe(df):
    """Process Capex data from a pandas DataFrame
    
    Args:
        df (pd.DataFrame): Raw Capex data
    
    Returns:
        tuple: (processed_data, pivot_table)
    """
    # Office locations are always loaded from fixed file
    return process_capex_data(df, None)

if __name__ == "__main__":
    # Example usage - you can modify this section as needed
    processed_data, pivot_data = process_capex_data()
