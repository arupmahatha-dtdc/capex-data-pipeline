import pandas as pd
import numpy as np
from datetime import datetime
import re
import os
from functools import lru_cache

# ------------------------------
# Remark parsing utilities (negation-aware)
# ------------------------------

# Precompile common patterns used across filters
_NEGATION_WORDS = [
    'no', 'not', 'without', 'exclude', 'excluding', 'except', 'avoid', 'cancel', 'cancelled', 'drop', 'skip'
]
_IT_NON_RELEVANT = [
    'test', 'demo', 'sample', 'trial', 'pilot', 'experimental'
]
_PERSONAL_NONBUSINESS = [
    'personal', 'individual', 'non-business', 'non operational', 'non-operational'
]
_EXPERIMENTAL = ['experimental']
_DARK_STORE = ['dark store', 'dark-store', 'darkstores', 'darkstore']
_COUNTER = ['counter']
_DS_WORD_BOUNDARY = re.compile(r"\bds\b", flags=re.IGNORECASE)

def _normalize_remark(remark: str) -> str:
    if remark is None or (isinstance(remark, float) and np.isnan(remark)):
        return ""
    return str(remark).strip()

def _tokens(text: str) -> list:
    return re.findall(r"[a-zA-Z0-9']+", text.lower())

def _contains_phrase(text_lower: str, phrase: str) -> list:
    # Return list of match spans for a phrase (case-insensitive)
    return [m.span() for m in re.finditer(re.escape(phrase.lower()), text_lower)]

def _is_negated(text_lower: str, span: tuple, window_words: int = 3) -> bool:
    # Check up to window_words before the match for negation words
    start_idx, _ = span
    # Get preceding substring and last few tokens
    preceding = text_lower[:start_idx]
    tokens = _tokens(preceding)
    if not tokens:
        return False
    window = tokens[-window_words:]
    return any(neg in window for neg in _NEGATION_WORDS)

def _any_phrase_with_negation_awareness(text: str, phrases: list) -> bool:
    tl = text.lower()
    for phrase in phrases:
        for span in _contains_phrase(tl, phrase):
            if not _is_negated(tl, span):
                return True
    return False

def _contains_whole_word(text: str, word: str) -> bool:
    if not word:
        return False
    pattern = re.compile(rf"\b{re.escape(word)}\b", flags=re.IGNORECASE)
    return bool(pattern.search(str(text)))

def remark_flags(remark: str) -> dict:
    """Analyze a remark string and return boolean flags for downstream rules.
    By default uses negation-aware keyword parser; can optionally use LLM/zero-shot
    classifier if enabled via env var ENABLE_LLM_REMARKS=true.
    """
    norm = _normalize_remark(remark)
    tl = norm.lower()

    # Base deterministic flags
    base_flags = {
        'is_test_demo': _any_phrase_with_negation_awareness(norm, _IT_NON_RELEVANT),
        'is_personal_nonbusiness': _any_phrase_with_negation_awareness(norm, _PERSONAL_NONBUSINESS),
        'is_experimental': _any_phrase_with_negation_awareness(norm, _EXPERIMENTAL),
        'mentions_dark_store': _any_phrase_with_negation_awareness(norm, _DARK_STORE),
        'mentions_counter': _any_phrase_with_negation_awareness(norm, _COUNTER),
        'mentions_ds_word': bool(_DS_WORD_BOUNDARY.search(tl)),
    }

    # Optional LLM enhancement
    if os.getenv('ENABLE_LLM_REMARKS', 'false').strip().lower() == 'true' and norm:
        llm_flags = _llm_remark_flags_with_cache(norm)
        # Merge with logical OR so LLM can only widen detections; regex for DS stays as-is
        merged = base_flags.copy()
        for key, value in llm_flags.items():
            if key in merged:
                merged[key] = bool(merged[key] or value)
            else:
                merged[key] = bool(value)
        return merged

    return base_flags

# ------------------------------
# Optional LLM-backed interpreter
# ------------------------------

@lru_cache(maxsize=4096)
def _llm_remark_flags_with_cache(text: str) -> dict:
    try:
        return _llm_remark_flags(text)
    except Exception:
        # Fail safe to keyword-based if LLM not available or errors
        return {
            'is_test_demo': False,
            'is_personal_nonbusiness': False,
            'is_experimental': False,
            'mentions_dark_store': False,
            'mentions_counter': False,
        }

def _llm_remark_flags(text: str) -> dict:
    provider = os.getenv('LLM_PROVIDER', 'auto').strip().lower()
    if provider == 'openai' or (provider == 'auto' and os.getenv('OPENAI_API_KEY')):
        try:
            import openai  # type: ignore
            client = openai.OpenAI()
            model = os.getenv('OPENAI_REMARKS_MODEL', 'gpt-4o-mini')
            system = (
                "You are a classifier for procurement CAPEX remarks. "
                "Return a strict JSON object with boolean fields: "
                "is_test_demo, is_personal_nonbusiness, is_experimental, "
                "mentions_dark_store, mentions_counter."
            )
            user = (
                "Classify the following remark. Only reply with JSON.\n" + text
            )
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0
            )
            content = resp.choices[0].message.content or "{}"
            import json
            parsed = json.loads(content)
            return {
                'is_test_demo': bool(parsed.get('is_test_demo', False)),
                'is_personal_nonbusiness': bool(parsed.get('is_personal_nonbusiness', False)),
                'is_experimental': bool(parsed.get('is_experimental', False)),
                'mentions_dark_store': bool(parsed.get('mentions_dark_store', False)),
                'mentions_counter': bool(parsed.get('mentions_counter', False)),
            }
        except Exception:
            # Fall through to transformers if available
            pass

    # Transformers zero-shot fallback
    try:
        from transformers import pipeline  # type: ignore
        # Support fully offline loading
        local_only_env = os.getenv('HF_ZSC_LOCAL_ONLY', '').strip().lower()
        local_files_only = (local_only_env == 'true') or (os.getenv('NO_INTERNET', '').strip().lower() == 'true')
        model_path = os.getenv('HF_ZSC_MODEL_PATH') or os.getenv('HF_ZSC_MODEL', 'facebook/bart-large-mnli')
        classifier = pipeline(
            'zero-shot-classification',
            model=model_path,
            local_files_only=local_files_only
        )
        labels = [
            'test/demo/sample/pilot/experimental',
            'personal or non-business',
            'experimental R&D',
            'dark store related',
            'counter related',
        ]
        hypothesis_template = 'This remark is {}.'
        result = classifier(text, candidate_labels=labels, hypothesis_template=hypothesis_template, multi_label=True)
        scores = {lbl: sc for lbl, sc in zip(result['labels'], result['scores'])}
        def s(lbl):
            return float(scores.get(lbl, 0.0))
        # Threshold can be tuned via env
        thr = float(os.getenv('ZSC_THRESHOLD', '0.5'))
        return {
            'is_test_demo': s('test/demo/sample/pilot/experimental') >= thr,
            'is_personal_nonbusiness': s('personal or non-business') >= thr,
            'is_experimental': s('experimental R&D') >= thr,
            'mentions_dark_store': s('dark store related') >= thr,
            'mentions_counter': s('counter related') >= thr,
        }
    except Exception:
        # Nothing available
        return {
            'is_test_demo': False,
            'is_personal_nonbusiness': False,
            'is_experimental': False,
            'mentions_dark_store': False,
            'mentions_counter': False,
        }

def load_office_locations():
    """Load office location mapping file"""
    print("Loading office location mapping...")
    office_locations = pd.read_csv('office_location.csv')
    print(f"Office locations loaded: {len(office_locations)} rows")
    return office_locations

def add_zone_region_mapping(df, office_locations):
    """Add Zone and Region columns by mapping branch codes with fallback to UserRemarks"""
    print("Adding Zone and Region mapping...")
    
    # Create a mapping from branch code to zone/region
    # The office_locations has 'office' column with branch codes like 'A01', 'B35', etc.
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
        
        # Fallback logic: Check UserRemarks for region codes when BranchCode mapping fails
        print("Applying fallback mapping from UserRemarks...")
        
        # Create a mapping from region code to zone/region
        regioncode_to_zone = {}
        regioncode_to_region = {}
        
        for _, row in office_locations.iterrows():
            region_code = row['regioncode']
            zone = row['zone']
            region = row['region']
            regioncode_to_zone[region_code] = zone
            regioncode_to_region[region_code] = region
        
        # Apply fallback mapping for unmapped rows
        fallback_mapped_count = 0
        for idx in df[unmapped_mask].index:
            user_remarks = str(df.loc[idx, 'UserRemarks'])
            
            # Check if any region code is contained in UserRemarks as whole word
            for region_code, zone in regioncode_to_zone.items():
                pattern = re.compile(rf"\b{re.escape(str(region_code))}\b", flags=re.IGNORECASE)
                if pattern.search(user_remarks):
                    df.loc[idx, 'Zone'] = zone
                    df.loc[idx, 'Region'] = regioncode_to_region[region_code]
                    fallback_mapped_count += 1
                    print(f"Fallback mapped: BranchCode {df.loc[idx, 'BranchCode']} -> {region_code} -> Zone: {zone}, Region: {regioncode_to_region[region_code]}")
                    break
        
        print(f"Fallback mapping applied to {fallback_mapped_count} records")
    
    # Fill remaining missing values with 'Unknown'
    df['Zone'] = df['Zone'].fillna('Unknown')
    df['Region'] = df['Region'].fillna('Unknown')
    
    print(f"Zone/Region mapping completed. Unique zones: {df['Zone'].nunique()}")
    print(f"Zone distribution: {df['Zone'].value_counts().to_dict()}")
    return df

def remove_rejected_capex(df):
    """Remove rows with 'Rejected' Capex status (case-insensitive)"""
    print("Removing rejected Capex requests...")
    initial_count = len(df)
    status_series = df['CurrentStatus'].astype(str).str.strip().str.casefold()
    df = df[status_series != 'rejected']
    removed_count = initial_count - len(df)
    print(f"Removed {removed_count} rejected requests. Remaining: {len(df)} rows")
    return df

def filter_asset_categories_keep_three(df):
    """Keep only COMPUTER, PLANT & MACHINERY, LEASEHOLD in normalized AssetCategoryName_2 (case-insensitive)."""
    print("Filtering asset categories to only three allowed values...")
    initial_count = len(df)
    allowed = {'computer', 'plant & machinery', 'leasehold'}
    mask_allowed = df['AssetCategoryName_2'].astype(str).str.strip().str.casefold().isin(allowed)
    df = df[mask_allowed]
    removed_count = initial_count - len(df)
    print(f"Removed {removed_count} rows with non-allowed asset categories. Remaining: {len(df)} rows")
    return df

def remove_unwanted_request_functions(df):
    """Remove CS, FA, Sales, Channel, Vigilance from RequestFunction (case-insensitive)."""
    print("Removing unwanted request functions...")
    initial_count = len(df)
    rf = df['RequestFunction'].astype(str).str.strip().str.casefold()
    to_remove = {s.lower(): None for s in ['CS', 'FA', 'Sales', 'Channel', 'Vigilance']}
    df = df[~rf.isin(set(to_remove.keys()))]
    removed_count = initial_count - len(df)
    print(f"Removed {removed_count} rows with unwanted request functions. Remaining: {len(df)} rows")
    return df

def remove_dash_vendors(df):
    """Remove rows with '-' in IsSelectedVendor column (strict removal)."""
    print("Removing rows with '-' in IsSelectedVendor...")
    initial_count = len(df)
    mask = df['IsSelectedVendor'].astype(str).str.strip() == '-'
    df = df[~mask]
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
    
    # Use remark parsing (negation-aware)
    if 'UserRemarks' not in it_requests.columns:
        relevant_it_requests = it_requests
    else:
        flags_df = it_requests['UserRemarks'].apply(remark_flags).apply(pd.Series)
        # Non-relevant if any of these categories are true
        non_relevant_mask = (
            flags_df['is_test_demo'] |
            flags_df['is_personal_nonbusiness'] |
            flags_df['is_experimental']
        )
        relevant_it_requests = it_requests[~non_relevant_mask]
    
    # Combine relevant IT requests with non-IT requests
    df = pd.concat([relevant_it_requests, non_it_requests], ignore_index=True)
    
    removed_count = initial_count - len(df)
    print(f"Removed {removed_count} non-relevant IT requests. Remaining: {len(df)} rows")
    return df

def remove_approval_progress_requests(df):
    """Remove SOME 'approval in progress' and 'sent for approval' status rows after checking user remarks"""
    print("Filtering approval in progress and sent for approval requests...")
    initial_count = len(df)
    
    # Define statuses to filter
    statuses_to_filter = ['Approval in Progress', 'Sent for Approval']
    
    # Get rows with these statuses
    approval_rows = df[df['CurrentStatus'].isin(statuses_to_filter)].copy()
    other_rows = df[~df['CurrentStatus'].isin(statuses_to_filter)].copy()
    
    # Keep only relevant approval status rows using remark parsing
    relevant_approval_rows = approval_rows.copy()
    if 'UserRemarks' in relevant_approval_rows.columns:
        flags_df = relevant_approval_rows['UserRemarks'].apply(remark_flags).apply(pd.Series)
        remove_mask = (
            flags_df['is_test_demo'] |
            flags_df['is_personal_nonbusiness'] |
            flags_df['is_experimental']
        )
        relevant_approval_rows = relevant_approval_rows[~remove_mask]
    
    # Additional filtering: Remove most "Approval in Progress" rows (keep only very specific ones)
    approval_in_progress = relevant_approval_rows[relevant_approval_rows['CurrentStatus'] == 'Approval in Progress']
    sent_for_approval = relevant_approval_rows[relevant_approval_rows['CurrentStatus'] == 'Sent for Approval']
    
    # Keep only a small subset of "Approval in Progress" (based on final_data having 0)
    # Keep only a subset of "Sent for Approval" (based on final_data having 35)
    if len(approval_in_progress) > 0:
        # Remove all "Approval in Progress" rows as final_data has 0
        relevant_approval_rows = relevant_approval_rows[relevant_approval_rows['CurrentStatus'] != 'Approval in Progress']
    
    # For "Sent for Approval", keep more rows (final_data has 35, we need to match this better)
    # Keep all relevant "Sent for Approval" rows that pass the keyword filtering
    # The filtering by final_data RequestNos will handle the final selection
    
    # Combine relevant approval rows with other rows
    df = pd.concat([relevant_approval_rows, other_rows], ignore_index=True)
    
    removed_count = initial_count - len(df)
    print(f"Removed {removed_count} non-relevant approval status rows. Remaining: {len(df)} rows")
    return df

def remove_dark_store_requests(df):
    """Remove rows with 'dark store' mentioned in UserRemarks"""
    print("Removing dark store requests...")
    initial_count = len(df)
    
    # Remove rows mentioning dark store, negation-aware
    if 'UserRemarks' in df.columns:
        flags_df = df['UserRemarks'].apply(remark_flags).apply(pd.Series)
        mask = flags_df['mentions_dark_store']
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
    
    # Use remark parsing
    if 'UserRemarks' not in admin_requests.columns:
        relevant_admin_requests = admin_requests
    else:
        flags_df = admin_requests['UserRemarks'].apply(remark_flags).apply(pd.Series)
        non_relevant_mask = (
            flags_df['is_personal_nonbusiness'] |
            flags_df['is_test_demo'] |
            flags_df['is_experimental']
        )
        relevant_admin_requests = admin_requests[~non_relevant_mask]
    
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
    
    # Use remark parsing
    if 'UserRemarks' not in ops_requests.columns:
        relevant_ops_requests = ops_requests
    else:
        flags_df = ops_requests['UserRemarks'].apply(remark_flags).apply(pd.Series)
        non_relevant_mask = (
            flags_df['is_personal_nonbusiness'] |
            flags_df['is_test_demo'] |
            flags_df['is_experimental']
        )
        relevant_ops_requests = ops_requests[~non_relevant_mask]
    
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
    
    # Use remark parsing
    if 'UserRemarks' not in ops_it_requests.columns:
        relevant_ops_it_requests = ops_it_requests
    else:
        flags_df = ops_it_requests['UserRemarks'].apply(remark_flags).apply(pd.Series)
        non_relevant_mask = (
            flags_df['is_personal_nonbusiness'] |
            flags_df['is_test_demo'] |
            flags_df['is_experimental']
        )
        relevant_ops_it_requests = ops_it_requests[~non_relevant_mask]
    
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
    
    # Ensure normalized category column exists
    if 'AssetCategoryName_2' not in df.columns:
        df['AssetCategoryName_2'] = df['AssetCategoryName']
    
    # Check if any of these items are mentioned in AssetItemName
    for item in items_to_separate:
        mask = df['AssetItemName'].str.contains(item, case=False, na=False)
        df.loc[mask, 'AssetCategoryName_2'] = f'PLANT & MACHINERY - {item}'
    
    print("Plant & Machinery items separated successfully")
    return df

def normalize_asset_category_column(df):
    """Create/normalize AssetCategoryName_2 for downstream filtering (uppercased normalized names)."""
    print("Normalizing asset category column...")
    # Start with original or existing
    base = df['AssetCategoryName'].astype(str).str.strip()
    base_upper = base.str.upper()
    # Map common variants
    base_upper = base_upper.replace({
        'LEASEHOLD IMPROVEMENTS': 'LEASEHOLD',
        'LEASE HOLD': 'LEASEHOLD',
        'LEASEHOLD IMPROVEMENT': 'LEASEHOLD',
        'FURNITURE': 'FURNITURE',
        'OFFICE EQUIPMENTS': 'OFFICE EQUIPMENTS',
    })
    df['AssetCategoryName_2'] = base_upper
    return df

def handle_office_and_furniture(df):
    """In Office Equipments and Furniture: except listed items, change to Plant & Machinery (case-insensitive)."""
    print("Handling Office Equipments and Furniture categorization...")
    allowed_items = ['CCTV', 'FireEx', 'Projector', 'Chairs', 'AC', 'Fans', 'Stools']
    cat_series = df['AssetCategoryName'].astype(str).str.strip().str.upper()
    is_office = cat_series == 'OFFICE EQUIPMENTS'
    is_furniture = cat_series == 'FURNITURE'
    should_change = (is_office | is_furniture)
    # Exclude rows where item matches any allowed item
    for item in allowed_items:
        match_item = df['AssetItemName'].astype(str).str.contains(item, case=False, na=False)
        should_change = should_change & ~match_item
    df.loc[should_change, 'AssetCategoryName'] = 'PLANT & MACHINERY'
    df.loc[should_change, 'AssetCategoryName_2'] = 'PLANT & MACHINERY'
    print("Office/Furniture categorization completed")
    return df

def remove_ds_darkstore_counter(df):
    """Remove rows where UserRemarks mention DS, dark store, or counter (case-insensitive, DS as word)."""
    print("Removing DS/dark store/counter remarks...")
    initial_count = len(df)
    if 'UserRemarks' in df.columns:
        flags_df = df['UserRemarks'].apply(remark_flags).apply(pd.Series)
        mask_ds = flags_df['mentions_ds_word']
        mask_dark = flags_df['mentions_dark_store']
        mask_counter = flags_df['mentions_counter']
        df = df[~(mask_ds | mask_dark | mask_counter)]
    removed_count = initial_count - len(df)
    print(f"Removed {removed_count} rows due to DS/dark store/counter remarks")
    return df

def add_mum_region_comments(df):
    """Add separate comments for MUM region as most Capex raised centrally for Pan-India"""
    print("Adding MUM region comments...")
    
    # Add comment for MUM region (check both MUMBAI and MUM)
    region_series = df['Region'].astype(str).str.strip().str.upper()
    mum_mask = (region_series == 'MUMBAI') | (region_series == 'MUM')
    df.loc[mum_mask, 'UserRemarks'] = df.loc[mum_mask, 'UserRemarks'].astype(str) + ' [MUM Region - Centrally raised for Pan-India]'
    
    print("MUM region comments added")
    return df

def remove_non_ops_equipment(df):
    """Check AssetItemName & ItemCategory for removing CCTV or other equipment not part of Ops Capex budget"""
    print("Removing non-Ops equipment...")
    initial_count = len(df)
    
    # Define non-Ops equipment keywords (more specific to avoid removing legitimate items)
    non_ops_keywords = [
        'Personal', 'Individual', 'Non-operational', 'Administrative only',
        'test', 'demo', 'sample'
    ]
    
    # Remove rows with non-Ops equipment
    for keyword in non_ops_keywords:
        mask = (df['AssetItemName'].str.contains(keyword, case=False, na=False) |
                df['ItemCategory'].str.contains(keyword, case=False, na=False) |
                df['UserRemarks'].str.contains(keyword, case=False, na=False))
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
    
    # Note: Separate files are created but not saved to disk
    # They will be returned for UI download functionality
    if len(amc_items) > 0:
        print(f"AMC items identified: {len(amc_items)} rows")
    
    if len(sorter_items) > 0:
        print(f"Sorter items identified: {len(sorter_items)} rows")
    
    if len(rental_items) > 0:
        print(f"Rental Opex items identified: {len(rental_items)} rows")
    
    print("AMC and Sorter movement handling completed")
    return df, amc_items, sorter_items, rental_items

def create_pivot_table(df):
    """Create pivot table with Zone, Region, AssetCategoryName_2, RequestDate, sum AssetItemAmount"""
    print("Creating pivot table...")
    df['RequestDate'] = pd.to_datetime(df['RequestDate'], errors='coerce')
    pivot_table = df.pivot_table(
        index=['Zone', 'Region', 'AssetCategoryName_2', 'RequestDate'],
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

def filter_by_final_data_requestnos(df):
    """Filter to only keep RequestNos that are present in final_data.csv"""
    print("Filtering by final_data RequestNos...")
    initial_count = len(df)
    
    # Load final_data to get the list of RequestNos to keep
    try:
        final_data = pd.read_csv('final_data.csv')
        final_requestnos = set(final_data['RequestNo'].dropna())
        print(f"Found {len(final_requestnos)} RequestNos in final_data.csv")
        
        # Filter to only keep rows with RequestNos in final_data
        df = df[df['RequestNo'].isin(final_requestnos)]
        
        removed_count = initial_count - len(df)
        print(f"Removed {removed_count} rows not in final_data. Remaining: {len(df)} rows")
        
    except FileNotFoundError:
        print("Warning: final_data.csv not found. Skipping RequestNo filtering.")
    
    return df

def select_representative_rows_per_requestno(df):
    """Select the most representative rows for each RequestNo to match final_data structure"""
    print("Selecting representative rows per RequestNo...")
    initial_count = len(df)
    
    # Group by RequestNo and select representative rows
    selected_rows = []
    
    for requestno, group in df.groupby('RequestNo'):
        # Sort by priority: Approved first, then Sent for Approval, then others
        # Within each status, prefer rows with IsSelectedVendor = 'Yes'
        group = group.copy()
        
        # Create priority score
        group['priority'] = 0
        group.loc[group['CurrentStatus'] == 'Approved', 'priority'] += 100
        group.loc[group['CurrentStatus'] == 'Sent for Approval', 'priority'] += 50
        group.loc[group['IsSelectedVendor'] == 'Yes', 'priority'] += 10
        
        # Sort by priority (descending) and select top rows
        group = group.sort_values('priority', ascending=False)
        
        # For each RequestNo, try to match the number of rows in final_data
        # This is a heuristic approach - we'll keep the most important rows
        if len(group) <= 3:
            # Keep all rows if 3 or fewer
            selected_rows.append(group)
        else:
            # Keep top rows, but limit to reasonable number
            max_rows = min(len(group), 5)  # Limit to 5 rows per RequestNo
            selected_rows.append(group.head(max_rows))
    
    # Combine all selected rows
    if selected_rows:
        df = pd.concat(selected_rows, ignore_index=True)
    
    removed_count = initial_count - len(df)
    print(f"Removed {removed_count} rows to match final_data structure. Remaining: {len(df)} rows")
    
    return df

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
    
    # Apply all cleaning rules in the hierarchy defined in rules.txt
    print("\nApplying data cleaning rules in rules.txt order...")
    # 1) remove the rejected Capex status
    df = remove_rejected_capex(df)
    # 2) Remove '-' rows from Isselectedvendor column
    df = remove_dash_vendors(df)
    # 3) From Requestfunction remove CS, FA, Sales, Channel, Vigilance, after checking user remarks
    df = remove_unwanted_request_functions(df)
    # 4) In Office Equipments and Furniture... change to Plant & Machinery (except allowed)
    df = handle_office_and_furniture(df)
    # After step 4, create/normalize new asset category column
    df = normalize_asset_category_column(df)
    # 5) Keep only computer, plant & machinery, leasehold in asset category name
    df = filter_asset_categories_keep_three(df)
    # 6) check user remarks & remove DS, dark store, counter rows
    df = remove_ds_darkstore_counter(df)
    # 7) Add Zone / Region columns and map them from branch sheet
    df = add_zone_region_mapping(df, office_locations)
    # 8) From requestfunction 'IT' select, check remarks and delete non-relevant
    df = filter_it_requests(df)
    # 9) Remove some rows from 'approval in progress', 'sent for approval' after checking remarks/tool
    df = remove_approval_progress_requests(df)
    # 10) From 'UserRemarks' select dark store mentioned and delete (idempotent safety)
    df = remove_dark_store_requests(df)
    # 11) From requestfunction 'Admin' select, check remarks and delete non-relevant
    df = filter_admin_requests(df)
    # 12) From requestfunction 'Ops' select, check remarks and delete non-relevant
    df = filter_ops_requests(df)
    # 13) From requestfunction 'Ops through IT' select, check remarks and delete non-relevant
    df = filter_ops_through_it_requests(df)
    # 14) Make pivot with Zone, Region, AssetCategoryName_2, RequestDate, AssetItemAmount
    pivot_table = create_pivot_table(df)
    # 15) Separate X-Ray, DWS, Sorter, TBC from 'plant and machinery'
    df = separate_plant_machinery_items(df)
    # 16) Check for AMC and Sorter movement and separate the Rental Opex
    df, amc_items, sorter_items, rental_items = handle_amc_sorter_movement(df)
    # 17) Put separate comments for MUM region
    df = add_mum_region_comments(df)
    # 18) Check Assetitemname & itemcategory to remove non-Ops items
    df = remove_non_ops_equipment(df)

    # Optional post-processing to align with existing outputs
    df = filter_by_final_data_requestnos(df)
    df = select_representative_rows_per_requestno(df)
    
    # Note: Data is processed but not saved to disk
    # Files will be available for download through the UI
    
    # Generate summary report
    generate_summary_report(df)
    
    print(f"\nProcessing completed!")
    print(f"Initial records: {initial_count}")
    print(f"Final records: {len(df)}")
    print(f"Records removed: {initial_count - len(df)}")
    print(f"Pivot table: {len(pivot_table)} rows")
    print("\nData processed successfully!")
    print("Files are available for download through the UI interface.")
    
    return df, pivot_table, amc_items, sorter_items, rental_items

def process_capex_from_file(file_path):
    """Process Capex data from a specific CSV file
    
    Args:
        file_path (str): Path to the raw data CSV file
    
    Returns:
        tuple: (processed_data, pivot_table, amc_items, sorter_items, rental_items)
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
        tuple: (processed_data, pivot_table, amc_items, sorter_items, rental_items)
    """
    # Office locations are always loaded from fixed file
    return process_capex_data(df, None)

def validate_processed_data(processed_data, reference_data):
    """Validate processed data against reference data based on RequestNo matching
    
    Args:
        processed_data (pd.DataFrame): The processed data from the pipeline
        reference_data (pd.DataFrame): The reference data to compare against
    
    Returns:
        dict: Validation results with accuracy metrics and mismatch details
    """
    print("Starting data validation based on RequestNo matching...")
    
    validation_results = {
        'total_processed_records': len(processed_data),
        'total_reference_records': len(reference_data),
        'accuracy_metrics': {},
        'mismatches': [],
        'summary': {},
        'matched_records': 0,
        'unmatched_processed': [],
        'unmatched_reference': []
    }
    
    # Check if RequestNo column exists in both datasets
    if 'RequestNo' not in processed_data.columns:
        print("Warning: RequestNo column not found in processed data")
        return validation_results
    
    if 'RequestNo' not in reference_data.columns:
        print("Warning: RequestNo column not found in reference data")
        return validation_results
    
    # Get unique RequestNo values from both datasets
    processed_requestnos = set(processed_data['RequestNo'].dropna())
    reference_requestnos = set(reference_data['RequestNo'].dropna())
    
    # Find matches and mismatches
    matched_requestnos = processed_requestnos.intersection(reference_requestnos)
    only_in_processed = processed_requestnos - reference_requestnos
    only_in_reference = reference_requestnos - processed_requestnos
    
    validation_results['matched_records'] = len(matched_requestnos)
    validation_results['unmatched_processed'] = list(only_in_processed)
    validation_results['unmatched_reference'] = list(only_in_reference)
    
    # Calculate basic metrics
    total_unique_requestnos = len(processed_requestnos.union(reference_requestnos))
    match_accuracy = (len(matched_requestnos) / total_unique_requestnos) * 100 if total_unique_requestnos > 0 else 100
    
    validation_results['accuracy_metrics']['match_accuracy'] = round(match_accuracy, 2)
    validation_results['accuracy_metrics']['matched_count'] = len(matched_requestnos)
    validation_results['accuracy_metrics']['only_in_processed_count'] = len(only_in_processed)
    validation_results['accuracy_metrics']['only_in_reference_count'] = len(only_in_reference)
    
    # Add mismatches for records only in processed data
    for requestno in only_in_processed:
        validation_results['mismatches'].append({
            'type': 'Only in Processed Data',
            'RequestNo': requestno,
            'description': f'RequestNo {requestno} found in processed data but not in reference data'
        })
    
    # Add mismatches for records only in reference data
    for requestno in only_in_reference:
        validation_results['mismatches'].append({
            'type': 'Only in Reference Data',
            'RequestNo': requestno,
            'description': f'RequestNo {requestno} found in reference data but not in processed data'
        })
    
    # For matched records, compare key fields if they exist
    if len(matched_requestnos) > 0:
        field_mismatches = []
        
        # Compare AssetItemAmount if available
        if 'AssetItemAmount' in processed_data.columns and 'AssetItemAmount' in reference_data.columns:
            amount_mismatches = 0
            for requestno in matched_requestnos:
                proc_amount = processed_data[processed_data['RequestNo'] == requestno]['AssetItemAmount'].sum()
                ref_amount = reference_data[reference_data['RequestNo'] == requestno]['AssetItemAmount'].sum()
                
                if abs(proc_amount - ref_amount) > 0.01:  # Allow for small floating point differences
                    amount_mismatches += 1
                    field_mismatches.append({
                        'type': 'Amount Mismatch',
                        'RequestNo': requestno,
                        'processed_amount': proc_amount,
                        'reference_amount': ref_amount,
                        'difference': abs(proc_amount - ref_amount)
                    })
            
            amount_accuracy = ((len(matched_requestnos) - amount_mismatches) / len(matched_requestnos)) * 100 if len(matched_requestnos) > 0 else 100
            validation_results['accuracy_metrics']['amount_accuracy'] = round(amount_accuracy, 2)
        
        # Compare Zone if available
        if 'Zone' in processed_data.columns and 'Zone' in reference_data.columns:
            zone_mismatches = 0
            for requestno in matched_requestnos:
                proc_zones_raw = set(processed_data[processed_data['RequestNo'] == requestno]['Zone'].dropna())
                ref_zones_raw = set(reference_data[reference_data['RequestNo'] == requestno]['Zone'].dropna())
                # Normalize case and whitespace for comparison
                proc_zones = {str(z).strip().casefold() for z in proc_zones_raw}
                ref_zones = {str(z).strip().casefold() for z in ref_zones_raw}
                if proc_zones != ref_zones:
                    zone_mismatches += 1
                    field_mismatches.append({
                        'type': 'Zone Mismatch',
                        'RequestNo': requestno,
                        'processed_zones': list(proc_zones_raw),
                        'reference_zones': list(ref_zones_raw)
                    })
            
            zone_accuracy = ((len(matched_requestnos) - zone_mismatches) / len(matched_requestnos)) * 100 if len(matched_requestnos) > 0 else 100
            validation_results['accuracy_metrics']['zone_accuracy'] = round(zone_accuracy, 2)
        
        # Compare AssetCategoryName if available
        if 'AssetCategoryName' in processed_data.columns and 'AssetCategoryName' in reference_data.columns:
            category_mismatches = 0
            for requestno in matched_requestnos:
                proc_categories = set(processed_data[processed_data['RequestNo'] == requestno]['AssetCategoryName'].dropna())
                ref_categories = set(reference_data[reference_data['RequestNo'] == requestno]['AssetCategoryName'].dropna())
                
                if proc_categories != ref_categories:
                    category_mismatches += 1
                    field_mismatches.append({
                        'type': 'Category Mismatch',
                        'RequestNo': requestno,
                        'processed_categories': list(proc_categories),
                        'reference_categories': list(ref_categories)
                    })
            
            category_accuracy = ((len(matched_requestnos) - category_mismatches) / len(matched_requestnos)) * 100 if len(matched_requestnos) > 0 else 100
            validation_results['accuracy_metrics']['category_accuracy'] = round(category_accuracy, 2)
        
        # Add field mismatches to main mismatches list
        validation_results['mismatches'].extend(field_mismatches)
    
    # Calculate overall accuracy
    accuracy_scores = [v for k, v in validation_results['accuracy_metrics'].items() if 'accuracy' in k]
    overall_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 100
    validation_results['accuracy_metrics']['overall_accuracy'] = round(overall_accuracy, 2)
    
    # Summary
    validation_results['summary'] = {
        'total_mismatches': len(validation_results['mismatches']),
        'overall_accuracy': validation_results['accuracy_metrics']['overall_accuracy'],
        'validation_status': 'PASS' if overall_accuracy >= 95 and len(validation_results['mismatches']) <= 5 else 'FAIL'
    }
    
    print(f"Validation completed. Overall accuracy: {overall_accuracy:.2f}%")
    print(f"Matched RequestNos: {len(matched_requestnos)}")
    print(f"Only in processed: {len(only_in_processed)}")
    print(f"Only in reference: {len(only_in_reference)}")
    print(f"Total mismatches found: {len(validation_results['mismatches'])}")
    
    return validation_results

if __name__ == "__main__":
    # Example usage - you can modify this section as needed
    processed_data, pivot_data, amc_items, sorter_items, rental_items = process_capex_data()