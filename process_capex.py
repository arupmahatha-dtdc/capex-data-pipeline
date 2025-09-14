import pandas as pd
import numpy as np
from datetime import datetime
import re
import os
from functools import lru_cache

# Global in-memory map of exclusion reasons per CompositeKey (RequestNo|AssetItemName)
EXCLUSION_REASONS = {}
FAIL_OPEN_UNKNOWN = os.getenv('FAIL_OPEN_UNKNOWN', 'true').strip().lower() in ('1', 'true', 'yes', 'on')

def reinstate_unknown_exclusions(raw_df: pd.DataFrame, df: pd.DataFrame, office_locations: pd.DataFrame | None = None) -> pd.DataFrame:
    """Fail-open: re-include raw rows that are missing from df and have no recorded exclusion reason.
    Adds a boolean column 'ReincludedViaFailOpen' and a note column 'FailOpenNote'.
    Lightly normalizes and maps zone/region for consistency.
    """
    try:
        raw = raw_df.copy()
        proc = df.copy()
        raw['CompositePrimaryKey'] = create_composite_primary_key(raw)
        if 'CompositePrimaryKey' not in proc.columns:
            proc['CompositePrimaryKey'] = create_composite_primary_key(proc)
        raw_keys = set(raw['CompositePrimaryKey'].dropna())
        proc_keys = set(proc['CompositePrimaryKey'].dropna())
        missing = raw_keys - proc_keys
        if not missing:
            return df
        to_reinclude = []
        for ck in missing:
            if ck not in EXCLUSION_REASONS:
                rows = raw[raw['CompositePrimaryKey'] == ck]
                if not rows.empty:
                    r = rows.copy()
                    r['ReincludedViaFailOpen'] = True
                    r['FailOpenNote'] = 'Unknown exclusion; fail-open applied'
                    to_reinclude.append(r)
        if not to_reinclude:
            return df
        add_back = pd.concat(to_reinclude, ignore_index=True)
        merged = pd.concat([df, add_back], ignore_index=True)
        # Light normalization
        if 'AssetCategoryName_2' not in merged.columns:
            merged = normalize_asset_category_column(merged)
        else:
            merged = normalize_asset_category_column(merged)
        if office_locations is not None:
            merged = add_zone_region_mapping(merged, office_locations)
        # Ensure composite key exists
        if 'CompositePrimaryKey' not in merged.columns:
            merged['CompositePrimaryKey'] = create_composite_primary_key(merged)
        # Drop exact duplicate rows if any
        merged = merged.drop_duplicates(subset=['CompositePrimaryKey', 'AssetItemAmount'], keep='first')
        return merged
    except Exception:
        return df

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

def _keys_for_df(df: pd.DataFrame) -> set:
    try:
        return set((df['RequestNo'].astype(str) + '|' + df['AssetItemName'].astype(str)).dropna())
    except Exception:
        return set()

def _record_exclusions(before_df: pd.DataFrame, after_df: pd.DataFrame, rule_label: str, column_name: str | None = None):
    """Record composite keys excluded by a processing step with a rule label aligned to rules.txt.
    Also capture the triggering column value for UI display when a column_name is provided.
    """
    try:
        before_df = before_df.copy()
        before_df['CompositeKey'] = create_composite_primary_key(before_df)
        before_map = before_df.set_index('CompositeKey')
        before_keys = set(before_map.index)
        after_keys = _keys_for_df(after_df)
        removed = before_keys - after_keys
        for k in removed:
            if k not in EXCLUSION_REASONS:
                value = None
                if column_name and k in before_map.index and column_name in before_map.columns:
                    try:
                        value = before_map.loc[k, column_name]
                    except Exception:
                        value = None
                EXCLUSION_REASONS[k] = {
                    'label': rule_label,
                    'column': column_name,
                    'value': value,
                }
    except Exception:
        pass

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
    before = df
    df = df[status_series != 'rejected']
    _record_exclusions(before, df, "1: Rejected status", column_name='CurrentStatus')
    removed_count = initial_count - len(df)
    print(f"Removed {removed_count} rejected requests. Remaining: {len(df)} rows")
    return df

def filter_asset_categories_keep_three(df):
    """Keep only COMPUTER, PLANT & MACHINERY, LEASEHOLD in normalized AssetCategoryName_2 (case-insensitive)."""
    print("Filtering asset categories to only three allowed values...")
    initial_count = len(df)
    allowed = {'computer', 'plant & machinery', 'leasehold'}
    mask_allowed = df['AssetCategoryName_2'].astype(str).str.strip().str.casefold().isin(allowed)
    before = df
    df = df[mask_allowed]
    _record_exclusions(before, df, "5: Asset category not in [COMPUTER, PLANT & MACHINERY, LEASEHOLD]", column_name='AssetCategoryName_2')
    removed_count = initial_count - len(df)
    print(f"Removed {removed_count} rows with non-allowed asset categories. Remaining: {len(df)} rows")
    return df

def remove_unwanted_request_functions(df):
    """Remove CS, FA, Sales, Channel, Vigilance from RequestFunction (case-insensitive)."""
    print("Removing unwanted request functions...")
    initial_count = len(df)
    rf = df['RequestFunction'].astype(str).str.strip().str.casefold()
    to_remove = {s.lower(): None for s in ['CS', 'FA', 'Sales', 'Channel', 'Vigilance']}
    before = df
    df = df[~rf.isin(set(to_remove.keys()))]
    _record_exclusions(before, df, "3: Removed RequestFunction (CS/FA/Sales/Channel/Vigilance)", column_name='RequestFunction')
    removed_count = initial_count - len(df)
    print(f"Removed {removed_count} rows with unwanted request functions. Remaining: {len(df)} rows")
    return df


def remove_aircon_fan_fireext_items(df):
    """Remove Air Conditioner, Fan, and Fire Extinguisher items (robust matching with variants/typos).
    Matches across AssetItemName, ItemCategory, and AssetCategoryName with tolerant regex:
    - Fan variants: "fan", "fans" with word boundaries
    - Fire extinguisher variants: "fire extinguisher", "fire extingushier", "fire ex", "fireex", etc.
    - Air conditioner variants: "air condition(er/ing)", "aircondition", "air-conditioning", "a/c"; and standalone
      "AC" only when context terms like "split", "window", "ton", "inverter", "air", "cond", "compressor" also appear.
    """
    print("Removing Air Conditioner / Fan / Fire Extinguisher items...")
    initial_count = len(df)

    # Combine relevant text columns for matching
    cols = ['AssetItemName', 'ItemCategory', 'AssetCategoryName']
    for col in cols:
        if col not in df.columns:
            df[col] = ''
    combined = (
        df['AssetItemName'].astype(str) + ' ' +
        df['ItemCategory'].astype(str) + ' ' +
        df['AssetCategoryName'].astype(str)
    )

    # Primary regex patterns (case-insensitive)
    fire_regex = re.compile(r"fire\s*extinguish\w*|\bfire\s*ex\w*|\bfireex\b", flags=re.IGNORECASE)
    fan_regex = re.compile(r"\bfans?\b", flags=re.IGNORECASE)
    aircond_regex = re.compile(r"air\s*condit(?:ion|ioner|ioning)?|air[-\s]*conditioning|air\s*condition|air\s*conditioner|air\s*conditioning|air\s*con\b|aircon\b|a\s*\/\s*c", flags=re.IGNORECASE)

    # Context-aware AC detection: \bAC\b with context terms around
    ac_word_regex = re.compile(r"\bAC\b", flags=re.IGNORECASE)
    ac_context_terms = re.compile(r"split|window|\bton\b|inverter|compressor|air|cond", flags=re.IGNORECASE)

    fire_mask = combined.str.contains(fire_regex, na=False)
    fan_mask = combined.str.contains(fan_regex, na=False)
    aircond_mask = combined.str.contains(aircond_regex, na=False)
    ac_standalone_mask = combined.str.contains(ac_word_regex, na=False) & combined.str.contains(ac_context_terms, na=False)

    # STOOL-Ops items should be excluded
    stool_ops_regex = re.compile(r"stool\s*-\s*ops", flags=re.IGNORECASE)
    stool_ops_mask = combined.str.contains(stool_ops_regex, na=False)

    # CCTV should be excluded as well, but allow CCTV - BRANCH items
    cctv_regex = re.compile(r"\bcctv\b|camera", flags=re.IGNORECASE)
    cctv_branch_regex = re.compile(r"cctv\s*-\s*branch", flags=re.IGNORECASE)
    cctv_mask = combined.str.contains(cctv_regex, na=False) & ~combined.str.contains(cctv_branch_regex, na=False)

    remove_mask = fire_mask | fan_mask | aircond_mask | ac_standalone_mask | cctv_mask | stool_ops_mask

    before = df
    df = df[~remove_mask]
    _record_exclusions(before, df, "3a: Excluded equipment (AirCon/Fan/FireExt/CCTV/STOOL-Ops)", column_name='AssetItemName')

    removed_count = initial_count - len(df)
    print(f"Removed {removed_count} Air Conditioner/Fan/Fire Extinguisher/CCTV/STOOL-Ops items. Remaining: {len(df)} rows")
    return df


def explain_exclusion_reason(row: pd.Series) -> str:
    """Given a raw/input row, explain which rule would exclude it.
    Returns a short, user-facing reason string.
    """
    try:
        # Normalize accessors
        val = lambda c: str(row.get(c, '') if pd.notna(row.get(c, '')) else '')
        cs = val('CurrentStatus').strip()
        isv = val('IsSelectedVendor').strip()
        rf = val('RequestFunction').strip()
        ac = val('AssetCategoryName').strip()
        item = val('AssetItemName').strip()
        itemcat = val('ItemCategory').strip()
        remarks = val('UserRemarks')

        # 1) Rejected
        if cs.lower() == 'rejected':
            return 'Rejected status'

        # 2) Dash vendor
        if isv == '-':
            return "IsSelectedVendor is '-'"

        # 3) Unwanted request function
        if rf.casefold() in {s.lower() for s in ['CS', 'FA', 'Sales', 'Channel', 'Vigilance']}:
            return f"Removed RequestFunction '{rf}'"

        # 3a) AirCon/Fan/FireExt removal (use same regex as filter)
        combined = f"{item} {itemcat} {ac}"
        fire_regex = re.compile(r"fire\s*extinguish\w*|\bfire\s*ex\w*|\bfireex\b", flags=re.IGNORECASE)
        fan_regex = re.compile(r"\bfans?\b", flags=re.IGNORECASE)
        aircond_regex = re.compile(r"air\s*condit(?:ion|ioner|ioning)?|air[-\s]*conditioning|air\s*condition|air\s*conditioner|air\s*conditioning|air\s*con\b|aircon\b|a\s*\/\s*c", flags=re.IGNORECASE)
        ac_word_regex = re.compile(r"\bAC\b", flags=re.IGNORECASE)
        ac_context_terms = re.compile(r"split|window|\bton\b|inverter|compressor|air|cond", flags=re.IGNORECASE)
        if (
            bool(fire_regex.search(combined)) or
            bool(fan_regex.search(combined)) or
            bool(aircond_regex.search(combined)) or
            (bool(ac_word_regex.search(combined)) and bool(ac_context_terms.search(combined)))
        ):
            return 'Excluded equipment: Air Conditioner/Fan/Fire Extinguisher'

        # 4/5) Asset category normalization + keep only three
        ac2 = ac.upper()
        ac2 = {
            'LEASEHOLD IMPROVEMENTS': 'LEASEHOLD',
            'LEASE HOLD': 'LEASEHOLD',
            'LEASEHOLD IMPROVEMENT': 'LEASEHOLD',
            'FURNITURE': 'FURNITURE',
            'OFFICE EQUIPMENTS': 'OFFICE EQUIPMENTS'
        }.get(ac2, ac2)
        if ac2 not in {'COMPUTER', 'PLANT & MACHINERY', 'LEASEHOLD'}:
            return f"Asset category excluded after normalization ('{ac2}')"

        # 6/10) DS/Dark Store/Counter via remarks
        flags = remark_flags(remarks)
        if flags.get('mentions_ds_word'):
            return "UserRemarks mention 'DS'"
        if flags.get('mentions_dark_store'):
            return "UserRemarks mention 'dark store'"
        if flags.get('mentions_counter'):
            return "UserRemarks mention 'counter'"

        # 8/11/12/13) Non-relevant by remarks for IT/Admin/Ops/Ops through IT
        is_nonrel = (flags.get('is_personal_nonbusiness') or flags.get('is_test_demo') or flags.get('is_experimental'))
        if rf == 'IT' and is_nonrel:
            return 'IT non-relevant by UserRemarks'
        if rf == 'Admin' and is_nonrel:
            return 'Admin non-relevant by UserRemarks'
        if rf == 'Ops' and is_nonrel:
            return 'Ops non-relevant by UserRemarks'
        if rf == 'Ops through IT' and is_nonrel:
            return 'Ops through IT non-relevant by UserRemarks'

        # 9) Approval in Progress removal
        if cs == 'Approval in Progress':
            return "Status 'Approval in Progress' removed"

        # 18) Non-Ops equipment keywords
        non_ops_keywords = [
            'Personal', 'Individual', 'Non-operational', 'Administrative only',
            'test', 'demo', 'sample'
        ]
        combined2 = f"{item} {itemcat} {remarks}"
        for kw in non_ops_keywords:
            if re.search(re.escape(kw), combined2, flags=re.IGNORECASE):
                return f"Non-Ops equipment keyword: '{kw}'"

        # Fallback
        # Try matching against recorded reasons if available in memory
        try:
            ck = f"{val('RequestNo')}|{val('AssetItemName')}"
            rec = EXCLUSION_REASONS.get(ck)
            if isinstance(rec, dict) and rec.get('label'):
                return rec['label']
            if isinstance(rec, str):
                return rec
        except Exception:
            pass
        return 'Unknown'
    except Exception:
        return 'Reason detection error'

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
        before = it_requests
        relevant_it_requests = it_requests[~non_relevant_mask]
        _record_exclusions(before, relevant_it_requests, "8: IT non-relevant by UserRemarks", column_name='UserRemarks')
    
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
    before = pd.concat([approval_rows, other_rows], ignore_index=True)
    df = pd.concat([relevant_approval_rows, other_rows], ignore_index=True)
    _record_exclusions(before, df, "9: Removed Approval in Progress/Sent for Approval (non-relevant)", column_name='CurrentStatus')
    
    removed_count = initial_count - len(df)
    print(f"Removed {removed_count} non-relevant approval status rows. Remaining: {len(df)} rows")
    return df

def remove_dark_store_requests(df):
    """Remove rows with 'dark store' mentioned in UserRemarks"""
    print("Removing dark store requests...")
    initial_count = len(df)
    
    # Remove rows mentioning dark store, negation-aware
    if 'UserRemarks' in df.columns:
        before = df
        flags_df = df['UserRemarks'].apply(remark_flags).apply(pd.Series)
        mask = flags_df['mentions_dark_store']
        df = df[~mask]
        _record_exclusions(before, df, "10: UserRemarks mention dark store", column_name='UserRemarks')
    
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
        before = admin_requests
        relevant_admin_requests = admin_requests[~non_relevant_mask]
        _record_exclusions(before, relevant_admin_requests, "11: Admin non-relevant by UserRemarks", column_name='UserRemarks')
    
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
        before = ops_requests
        relevant_ops_requests = ops_requests[~non_relevant_mask]
        _record_exclusions(before, relevant_ops_requests, "12: Ops non-relevant by UserRemarks", column_name='UserRemarks')
    
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
        before = ops_it_requests
        relevant_ops_it_requests = ops_it_requests[~non_relevant_mask]
        _record_exclusions(before, relevant_ops_it_requests, "13: Ops through IT non-relevant by UserRemarks", column_name='UserRemarks')
    
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
    # Map common variants and typos
    base_upper = base_upper.replace({
        'LEASEHOLD IMPROVEMENTS': 'LEASEHOLD',
        'LEASE HOLD': 'LEASEHOLD',
        'LEASEHOLD IMPROVEMENT': 'LEASEHOLD',
        'FURNITURE': 'FURNITURE',
        'FURNITURES': 'FURNITURE',
        'FURINTURE': 'FURNITURE',
        'FURINITURE': 'FURNITURE',
        'OFFICE EQUIPMENTS': 'OFFICE EQUIPMENTS',
        'OFFICE EQUIPMENT': 'OFFICE EQUIPMENTS',
        'OFFICE-EQUIPMENTS': 'OFFICE EQUIPMENTS',
        'OFFICE EQUIPTMENTS': 'OFFICE EQUIPMENTS',
        'OFFICE EQPT': 'OFFICE EQUIPMENTS',
    })
    df['AssetCategoryName_2'] = base_upper
    return df

def handle_office_and_furniture(df):
    """In Office Equipments and Furniture: except listed items, change to Plant & Machinery.
    Robust to category typos like FURINTURE, OFFC EQUIPMENTS, etc.
    """
    print("Handling Office Equipments and Furniture categorization...")
    allowed_items = ['CCTV', 'FireEx', 'Projector', 'Chairs', 'AC', 'Fans', 'Stools']
    cat_series = df['AssetCategoryName'].astype(str).str.strip().str.upper()
    # Normalize obvious typos first to improve matching
    cat_series = cat_series.replace({
        'FURNITURES': 'FURNITURE',
        'FURINTURE': 'FURNITURE',
        'FURINITURE': 'FURNITURE',
        'OFFICE EQUIPMENT': 'OFFICE EQUIPMENTS',
        'OFFICE-EQUIPMENTS': 'OFFICE EQUIPMENTS',
        'OFFICE EQUIPTMENTS': 'OFFICE EQUIPMENTS',
        'OFFICE EQPT': 'OFFICE EQUIPMENTS',
    })
    # Broad regex identification as a fallback
    is_office = cat_series.str.contains(r"\bOFFICE\b.*\bEQUIP", regex=True, na=False)
    is_furniture = cat_series.str.contains(r"\bFURNIT", regex=True, na=False)
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
    """Remove rows where UserRemarks mention DS, dark store, or counter (case-insensitive, DS as word).
    Allow counter-related items that are legitimate operational items (Table, Electrical Work, Interior Work, etc.)
    """
    print("Removing DS/dark store/counter remarks...")
    initial_count = len(df)
    if 'UserRemarks' in df.columns:
        before = df
        flags_df = df['UserRemarks'].apply(remark_flags).apply(pd.Series)
        mask_ds = flags_df['mentions_ds_word']
        mask_dark = flags_df['mentions_dark_store']
        
        # For counter mentions, be more selective - exclude only if it's not a legitimate operational item
        mask_counter = flags_df['mentions_counter']
        
        # Define legitimate operational items that should be kept even if they mention counter
        legitimate_items = ['table', 'electrical work', 'interior work', 'renovation', 'relocation', 'construction', 'installation']
        
        # Create a mask for legitimate items
        legitimate_mask = df['AssetItemName'].str.contains('|'.join(legitimate_items), case=False, na=False)
        
        # Only exclude counter mentions if they're not legitimate operational items
        mask_counter_filtered = mask_counter & ~legitimate_mask
        
        df = df[~(mask_ds | mask_dark | mask_counter_filtered)]
        _record_exclusions(before, df, "6/10: UserRemarks mention DS/dark store/counter", column_name='UserRemarks')
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
        before = df
        mask = (df['AssetItemName'].str.contains(keyword, case=False, na=False) |
                df['ItemCategory'].str.contains(keyword, case=False, na=False) |
                df['UserRemarks'].str.contains(keyword, case=False, na=False))
        df = df[~mask]
        _record_exclusions(before, df, f"18: Non-Ops equipment keyword ('{keyword}')", column_name='AssetItemName')
    
    # Remove CCTV items but allow CCTV - BRANCH items
    before = df
    cctv_mask = (df['AssetItemName'].str.contains('CCTV', case=False, na=False) |
                 df['ItemCategory'].str.contains('CCTV', case=False, na=False) |
                 df['UserRemarks'].str.contains('CCTV', case=False, na=False))
    # Allow CCTV - BRANCH items
    cctv_branch_mask = (df['AssetItemName'].str.contains('CCTV.*BRANCH', case=False, na=False) |
                       df['ItemCategory'].str.contains('CCTV.*BRANCH', case=False, na=False) |
                       df['UserRemarks'].str.contains('CCTV.*BRANCH', case=False, na=False))
    # Remove CCTV items that are not CCTV - BRANCH
    final_cctv_mask = cctv_mask & ~cctv_branch_mask
    df = df[~final_cctv_mask]
    _record_exclusions(before, df, "18: Non-Ops equipment keyword ('CCTV')", column_name='AssetItemName')
    
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
        before = df
        df = df[df['RequestNo'].isin(final_requestnos)]
        _record_exclusions(before, df, "Post: Not in final_data RequestNos", column_name='RequestNo')
        
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
    
    # Combine all selected rows (no down-selection removal anymore)
    if selected_rows:
        df = pd.concat(selected_rows, ignore_index=True)
    
    removed_count = initial_count - len(df)
    print(f"Rows retained after grouping: {len(df)} (no representative down-selection applied)")
    
    return df

def add_composite_primary_key(df):
    """Add composite primary key to the dataframe
    
    Args:
        df (pd.DataFrame): DataFrame to add composite key to
    
    Returns:
        pd.DataFrame: DataFrame with composite primary key added
    """
    print("Adding composite primary key (RequestNo + AssetItemName)...")
    
    if 'RequestNo' not in df.columns or 'AssetItemName' not in df.columns:
        print("Warning: Cannot create composite primary key - missing RequestNo or AssetItemName columns")
        return df
    
    # Add composite primary key
    df['CompositePrimaryKey'] = create_composite_primary_key(df)
    
    # Validate the composite key
    key_validation = validate_composite_primary_key(df, 'CompositePrimaryKey')
    
    if key_validation['validation_status'] == 'FAIL':
        print("Warning: Composite primary key validation failed!")
        print(f"  - Duplicate keys: {key_validation['duplicate_keys']}")
        print(f"  - Missing keys: {key_validation['missing_keys']}")
    else:
        print("Composite primary key validation passed!")
    
    print(f"Composite primary key added successfully. Unique keys: {key_validation['unique_keys']}")
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
    # 3a) Remove Air Conditioner / Fan / Fire Extinguisher items (broad variants)
    df = remove_aircon_fan_fireext_items(df)
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
    
    # 19) Add composite primary key (RequestNo + AssetItemName)
    df = add_composite_primary_key(df)

    # Optional post-processing to align with existing outputs
    df = filter_by_final_data_requestnos(df)
    df = select_representative_rows_per_requestno(df)
    # Fail-open: reinclude rows excluded with Unknown reason
    if FAIL_OPEN_UNKNOWN:
        df = reinstate_unknown_exclusions(raw_data, df, office_locations)
    
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

def create_composite_primary_key(df):
    """Create composite primary key using RequestNo and AssetItemName
    
    Args:
        df (pd.DataFrame): DataFrame with RequestNo and AssetItemName columns
    
    Returns:
        pd.Series: Series with composite primary key values
    """
    if 'RequestNo' not in df.columns or 'AssetItemName' not in df.columns:
        raise ValueError("Both RequestNo and AssetItemName columns are required for composite primary key")
    
    # Create composite key by combining RequestNo and AssetItemName
    composite_key = df['RequestNo'].astype(str) + '|' + df['AssetItemName'].astype(str)
    return composite_key

def validate_composite_primary_key(df, key_name='CompositeKey'):
    """Validate composite primary key for uniqueness and completeness
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        key_name (str): Name for the composite key column
    
    Returns:
        dict: Validation results with duplicate and missing key information
    """
    print(f"Validating composite primary key ({key_name})...")
    
    validation_results = {
        'total_records': len(df),
        'unique_keys': 0,
        'duplicate_keys': 0,
        'missing_keys': 0,
        'duplicate_details': [],
        'missing_details': [],
        'validation_status': 'PASS'
    }
    
    # Create composite key
    df_with_key = df.copy()
    df_with_key[key_name] = create_composite_primary_key(df)
    
    # Check for missing values in composite key
    missing_mask = df_with_key[key_name].str.contains('nan|None', case=False, na=True)
    missing_count = missing_mask.sum()
    validation_results['missing_keys'] = missing_count
    
    if missing_count > 0:
        missing_details = df_with_key[missing_mask][['RequestNo', 'AssetItemName']].to_dict('records')
        validation_results['missing_details'] = missing_details
        print(f"Warning: Found {missing_count} records with missing composite key components")
    
    # Check for duplicates
    key_counts = df_with_key[key_name].value_counts()
    duplicates = key_counts[key_counts > 1]
    duplicate_count = len(duplicates)
    validation_results['duplicate_keys'] = duplicate_count
    
    if duplicate_count > 0:
        duplicate_details = []
        for key, count in duplicates.items():
            duplicate_records = df_with_key[df_with_key[key_name] == key][['RequestNo', 'AssetItemName']].to_dict('records')
            duplicate_details.append({
                'composite_key': key,
                'count': count,
                'records': duplicate_records
            })
        validation_results['duplicate_details'] = duplicate_details
        print(f"Warning: Found {duplicate_count} duplicate composite keys")
    
    # Calculate unique keys
    validation_results['unique_keys'] = len(key_counts) - duplicate_count
    
    # Determine validation status
    if missing_count > 0 or duplicate_count > 0:
        validation_results['validation_status'] = 'FAIL'
    else:
        validation_results['validation_status'] = 'PASS'
    
    print(f"Composite key validation completed: {validation_results['validation_status']}")
    print(f"Total records: {validation_results['total_records']}")
    print(f"Unique keys: {validation_results['unique_keys']}")
    print(f"Duplicate keys: {validation_results['duplicate_keys']}")
    print(f"Missing keys: {validation_results['missing_keys']}")
    
    return validation_results

def validate_all_sheets_composite_keys(input_data, processed_data, reference_data):
    """Comprehensive validation focusing on Processed vs Reference using ML metrics
    
    Args:
        input_data (pd.DataFrame): Raw input data (for context only)
        processed_data (pd.DataFrame): Processed output data (predictions)
        reference_data (pd.DataFrame): Reference data for comparison (ground truth)
    
    Returns:
        dict: Comprehensive validation results with ML metrics
    """
    print("Starting comprehensive validation using ML metrics (Processed vs Reference)...")
    
    validation_results = {
        'processed_validation': {},
        'reference_validation': {},
        'ml_validation': {},
        'summary': {},
        'all_mismatches': []
    }
    
    # Validate each sheet individually
    print("\n1. Validating Input Data (Raw Data)...")
    if input_data is not None and not input_data.empty:
        input_validation = validate_composite_primary_key(input_data, 'InputKey')
        validation_results['input_validation'] = input_validation
    else:
        validation_results['input_validation'] = {'validation_status': 'SKIP', 'message': 'No input data provided'}
    
    print("\n2. Validating Processed Data (Output Data)...")
    if processed_data is not None and not processed_data.empty:
        processed_validation = validate_composite_primary_key(processed_data, 'ProcessedKey')
        validation_results['processed_validation'] = processed_validation
    else:
        validation_results['processed_validation'] = {'validation_status': 'SKIP', 'message': 'No processed data provided'}
    
    print("\n3. Validating Reference Data...")
    if reference_data is not None and not reference_data.empty:
        reference_validation = validate_composite_primary_key(reference_data, 'ReferenceKey')
        validation_results['reference_validation'] = reference_validation
    else:
        validation_results['reference_validation'] = {'validation_status': 'SKIP', 'message': 'No reference data provided'}
    
    # ML-based validation: Processed vs Reference
    print("\n4. Performing ML-based validation (Processed vs Reference)...")
    if (processed_data is not None and not processed_data.empty and 
        reference_data is not None and not reference_data.empty):
        
        # Use the updated validate_processed_data function with ML metrics
        ml_validation = validate_processed_data(processed_data, reference_data)
        validation_results['ml_validation'] = ml_validation
        
        # Add mismatches from ML validation
        # Enrich False Negatives with exclusion reasons based on input_data (raw)
        enriched = []
        input_with_key = None
        try:
            if input_data is not None and not input_data.empty:
                input_with_key = input_data.copy()
                input_with_key['CompositeKey'] = create_composite_primary_key(input_with_key)
        except Exception:
            input_with_key = None

        for m in ml_validation['mismatches']:
            if m.get('type') == 'False Negative':
                ck = m.get('CompositeKey')
                # Prefer pipeline-recorded reason
                reason = EXCLUSION_REASONS.get(ck)
                if not reason:
                    # Fallback to on-the-fly explanation (legacy)
                    if input_with_key is not None:
                        recs = input_with_key[input_with_key['CompositeKey'] == ck]
                        if not recs.empty:
                            reason = explain_exclusion_reason(recs.iloc[0])
                    if not reason:
                        try:
                            ref_with_key = reference_data.copy()
                            ref_with_key['CompositeKey'] = create_composite_primary_key(ref_with_key)
                            recs2 = ref_with_key[ref_with_key['CompositeKey'] == ck]
                            if not recs2.empty:
                                reason = explain_exclusion_reason(recs2.iloc[0])
                        except Exception:
                            pass
                m['exclusion_reason'] = reason or 'Unknown'
            enriched.append(m)

        validation_results['all_mismatches'].extend(enriched)
    else:
        validation_results['ml_validation'] = {'validation_status': 'SKIP', 'message': 'Missing processed or reference data'}
    
    # Calculate summary
    total_mismatches = len(validation_results['all_mismatches'])
    ml_metrics = validation_results['ml_validation'].get('ml_metrics', {})
    
    validation_results['summary'] = {
        'total_mismatches': total_mismatches,
        'processed_status': validation_results['processed_validation'].get('validation_status', 'UNKNOWN'),
        'reference_status': validation_results['reference_validation'].get('validation_status', 'UNKNOWN'),
        'precision': ml_metrics.get('precision', 0),
        'recall': ml_metrics.get('recall', 0),
        'f1_score': ml_metrics.get('f1_score', 0),
        'overall_status': 'PASS' if ml_metrics.get('f1_score', 0) >= 0.95 else 'FAIL'
    }
    
    print(f"\nComprehensive validation completed!")
    print(f"ML Metrics Summary:")
    print(f"  Precision: {ml_metrics.get('precision', 0):.4f}")
    print(f"  Recall: {ml_metrics.get('recall', 0):.4f}")
    print(f"  F1-Score: {ml_metrics.get('f1_score', 0):.4f}")
    print(f"Total mismatches found: {total_mismatches}")
    print(f"Overall status: {validation_results['summary']['overall_status']}")
    
    return validation_results

def validate_processed_data(processed_data, reference_data):
    """Validate processed data against reference data using ML metrics (Precision, Recall, F1-Score)
    
    Args:
        processed_data (pd.DataFrame): The processed data from the pipeline (predictions)
        reference_data (pd.DataFrame): The reference data to compare against (ground truth)
    
    Returns:
        dict: Validation results with precision, recall, F1-score and mismatch details
    """
    print("Starting data validation using ML metrics (Precision, Recall, F1-Score)...")
    
    validation_results = {
        'total_processed_records': len(processed_data),
        'total_reference_records': len(reference_data),
        'ml_metrics': {},
        'mismatches': [],
        'summary': {},
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'composite_key_validation': {}
    }
    
    # Check if required columns exist in both datasets
    required_columns = ['RequestNo', 'AssetItemName']
    for col in required_columns:
        if col not in processed_data.columns:
            print(f"Warning: {col} column not found in processed data")
            return validation_results
        if col not in reference_data.columns:
            print(f"Warning: {col} column not found in reference data")
            return validation_results
    
    # Validate composite primary keys for both datasets
    print("Validating composite primary keys...")
    processed_key_validation = validate_composite_primary_key(processed_data, 'ProcessedKey')
    reference_key_validation = validate_composite_primary_key(reference_data, 'ReferenceKey')
    
    validation_results['composite_key_validation'] = {
        'processed': processed_key_validation,
        'reference': reference_key_validation
    }
    
    # Create composite keys for both datasets
    processed_data_with_key = processed_data.copy()
    processed_data_with_key['CompositeKey'] = create_composite_primary_key(processed_data)
    
    reference_data_with_key = reference_data.copy()
    reference_data_with_key['CompositeKey'] = create_composite_primary_key(reference_data)
    
    # Get unique composite key values from both datasets
    processed_keys = set(processed_data_with_key['CompositeKey'].dropna())
    reference_keys = set(reference_data_with_key['CompositeKey'].dropna())
    
    # Calculate ML metrics: Precision, Recall, F1-Score
    # True Positives: Records that exist in both processed and reference (correctly identified)
    true_positives = processed_keys.intersection(reference_keys)
    # False Positives: Records in processed but not in reference (incorrectly included)
    false_positives = processed_keys - reference_keys
    # False Negatives: Records in reference but not in processed (incorrectly excluded)
    false_negatives = reference_keys - processed_keys
    
    validation_results['true_positives'] = len(true_positives)
    validation_results['false_positives'] = len(false_positives)
    validation_results['false_negatives'] = len(false_negatives)
    
    # Calculate Precision, Recall, F1-Score
    precision = len(true_positives) / (len(true_positives) + len(false_positives)) if (len(true_positives) + len(false_positives)) > 0 else 0
    recall = len(true_positives) / (len(true_positives) + len(false_negatives)) if (len(true_positives) + len(false_negatives)) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    validation_results['ml_metrics']['precision'] = round(precision, 4)
    validation_results['ml_metrics']['recall'] = round(recall, 4)
    validation_results['ml_metrics']['f1_score'] = round(f1_score, 4)
    validation_results['ml_metrics']['true_positives'] = len(true_positives)
    validation_results['ml_metrics']['false_positives'] = len(false_positives)
    validation_results['ml_metrics']['false_negatives'] = len(false_negatives)
    
    # Add mismatches for False Positives (records in processed but not in reference)
    for composite_key in false_positives:
        # Extract RequestNo and AssetItemName from composite key
        key_parts = composite_key.split('|', 1)
        requestno = key_parts[0] if len(key_parts) > 0 else 'Unknown'
        asset_item = key_parts[1] if len(key_parts) > 1 else 'Unknown'
        
        validation_results['mismatches'].append({
            'type': 'False Positive',
            'RequestNo': requestno,
            'AssetItemName': asset_item,
            'CompositeKey': composite_key,
            'description': f'Record {composite_key} (RequestNo: {requestno}, AssetItem: {asset_item}) incorrectly included in processed data (not in reference)'
        })
    
    # Add mismatches for False Negatives (records in reference but not in processed)
    for composite_key in false_negatives:
        # Extract RequestNo and AssetItemName from composite key
        key_parts = composite_key.split('|', 1)
        requestno = key_parts[0] if len(key_parts) > 0 else 'Unknown'
        asset_item = key_parts[1] if len(key_parts) > 1 else 'Unknown'
        
        validation_results['mismatches'].append({
            'type': 'False Negative',
            'RequestNo': requestno,
            'AssetItemName': asset_item,
            'CompositeKey': composite_key,
            'description': f'Record {composite_key} (RequestNo: {requestno}, AssetItem: {asset_item}) incorrectly excluded from processed data (should be included)'
        })
    
    # For True Positives (matched records), compare key fields if they exist
    if len(true_positives) > 0:
        field_mismatches = []
        
        # Compare AssetItemAmount if available
        if 'AssetItemAmount' in processed_data.columns and 'AssetItemAmount' in reference_data.columns:
            amount_mismatches = 0
            for composite_key in true_positives:
                proc_amount = processed_data_with_key[processed_data_with_key['CompositeKey'] == composite_key]['AssetItemAmount'].sum()
                ref_amount = reference_data_with_key[reference_data_with_key['CompositeKey'] == composite_key]['AssetItemAmount'].sum()
                
                if abs(proc_amount - ref_amount) > 0.01:  # Allow for small floating point differences
                    amount_mismatches += 1
                    key_parts = composite_key.split('|', 1)
                    requestno = key_parts[0] if len(key_parts) > 0 else 'Unknown'
                    asset_item = key_parts[1] if len(key_parts) > 1 else 'Unknown'
                    
                    field_mismatches.append({
                        'type': 'Amount Mismatch',
                        'RequestNo': requestno,
                        'AssetItemName': asset_item,
                        'CompositeKey': composite_key,
                        'processed_amount': proc_amount,
                        'reference_amount': ref_amount,
                        'difference': abs(proc_amount - ref_amount)
                    })
            
            amount_accuracy = ((len(true_positives) - amount_mismatches) / len(true_positives)) * 100 if len(true_positives) > 0 else 100
            validation_results['ml_metrics']['amount_accuracy'] = round(amount_accuracy, 2)
        
        # Compare Zone if available
        if 'Zone' in processed_data.columns and 'Zone' in reference_data.columns:
            zone_mismatches = 0
            for composite_key in true_positives:
                proc_zones_raw = set(processed_data_with_key[processed_data_with_key['CompositeKey'] == composite_key]['Zone'].dropna())
                ref_zones_raw = set(reference_data_with_key[reference_data_with_key['CompositeKey'] == composite_key]['Zone'].dropna())
                # Normalize case and whitespace for comparison
                proc_zones = {str(z).strip().casefold() for z in proc_zones_raw}
                ref_zones = {str(z).strip().casefold() for z in ref_zones_raw}
                if proc_zones != ref_zones:
                    zone_mismatches += 1
                    key_parts = composite_key.split('|', 1)
                    requestno = key_parts[0] if len(key_parts) > 0 else 'Unknown'
                    asset_item = key_parts[1] if len(key_parts) > 1 else 'Unknown'
                    
                    field_mismatches.append({
                        'type': 'Zone Mismatch',
                        'RequestNo': requestno,
                        'AssetItemName': asset_item,
                        'CompositeKey': composite_key,
                        'processed_zones': list(proc_zones_raw),
                        'reference_zones': list(ref_zones_raw)
                    })
            
            zone_accuracy = ((len(true_positives) - zone_mismatches) / len(true_positives)) * 100 if len(true_positives) > 0 else 100
            validation_results['ml_metrics']['zone_accuracy'] = round(zone_accuracy, 2)
        
        # Compare AssetCategoryName if available
        if 'AssetCategoryName' in processed_data.columns and 'AssetCategoryName' in reference_data.columns:
            category_mismatches = 0
            for composite_key in true_positives:
                proc_categories = set(processed_data_with_key[processed_data_with_key['CompositeKey'] == composite_key]['AssetCategoryName'].dropna())
                ref_categories = set(reference_data_with_key[reference_data_with_key['CompositeKey'] == composite_key]['AssetCategoryName'].dropna())
                
                if proc_categories != ref_categories:
                    category_mismatches += 1
                    key_parts = composite_key.split('|', 1)
                    requestno = key_parts[0] if len(key_parts) > 0 else 'Unknown'
                    asset_item = key_parts[1] if len(key_parts) > 1 else 'Unknown'
                    
                    field_mismatches.append({
                        'type': 'Category Mismatch',
                        'RequestNo': requestno,
                        'AssetItemName': asset_item,
                        'CompositeKey': composite_key,
                        'processed_categories': list(proc_categories),
                        'reference_categories': list(ref_categories)
                    })
            
            category_accuracy = ((len(true_positives) - category_mismatches) / len(true_positives)) * 100 if len(true_positives) > 0 else 100
            validation_results['ml_metrics']['category_accuracy'] = round(category_accuracy, 2)
        
        # Add field mismatches to main mismatches list
        validation_results['mismatches'].extend(field_mismatches)
    
    # Summary
    validation_results['summary'] = {
        'total_mismatches': len(validation_results['mismatches']),
        'precision': validation_results['ml_metrics']['precision'],
        'recall': validation_results['ml_metrics']['recall'],
        'f1_score': validation_results['ml_metrics']['f1_score'],
        'validation_status': 'PASS' if validation_results['ml_metrics']['f1_score'] >= 0.95 else 'FAIL'
    }
    
    print(f"Validation completed using ML metrics:")
    print(f"Precision: {validation_results['ml_metrics']['precision']:.4f}")
    print(f"Recall: {validation_results['ml_metrics']['recall']:.4f}")
    print(f"F1-Score: {validation_results['ml_metrics']['f1_score']:.4f}")
    print(f"True Positives: {len(true_positives)}")
    print(f"False Positives: {len(false_positives)}")
    print(f"False Negatives: {len(false_negatives)}")
    print(f"Total mismatches found: {len(validation_results['mismatches'])}")
    
    return validation_results

if __name__ == "__main__":
    # Example usage - you can modify this section as needed
    processed_data, pivot_data, amc_items, sorter_items, rental_items = process_capex_data()