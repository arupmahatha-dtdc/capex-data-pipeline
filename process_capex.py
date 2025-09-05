import pandas as pd

def process_capex(raw_file: str, branch_map_file: str = None):
    df = pd.read_csv(raw_file)

    # Load branch mapping if provided
    if branch_map_file:
        branch_map = pd.read_csv(branch_map_file)
    else:
        branch_map = None

    # 1. Remove rejected Capex
    df = df[df["CurrentStatus"].str.lower() != "rejected"]

    # 2. Remove '-' in IsSelectedVendor
    df = df[df["IsSelectedVendor"] != "-"]

    # 3. Remove some RequestFunction
    remove_funcs = ["CS", "FA", "Sales", "Channel", "Vigilance"]
    df = df[~df["RequestFunction"].isin(remove_funcs)]

    # 4. Keep only categories
    keep_categories = ["Computer", "Plant & Machinery", "Leasehold"]
    df = df[df["AssetCategoryName"].isin(keep_categories)]

    # 5. Office Equipments & Furniture handling
    office_equips = ["Office Equipment", "Furniture"]
    exclude_items = ["CCTV", "FireEx", "Projector", "Chairs", "AC", "Fans", "Stools"]
    mask = df["AssetCategoryName"].isin(office_equips)
    mask_exclude = df["AssetItemName"].isin(exclude_items)
    df.loc[mask & ~mask_exclude, "AssetCategoryName"] = "Plant & Machinery"

    # 6. Add Zone / Region
    if branch_map is not None:
        df = df.merge(branch_map, on="BranchCode", how="left")

    # 7–12. Relevance filter based on remarks
    def check_relevance(row):
        remarks = str(row["UserRemarks"]).lower()
        if "not relevant" in remarks:
            return False
        return True
    df = df[df.apply(check_relevance, axis=1)]

    # 8. Remove specific statuses
    exclude_status = ["approval in progress", "sent for approval"]
    df = df[~df["CurrentStatus"].str.lower().isin(exclude_status)]

    # 9. Remove remarks containing dark store
    df = df[~df["UserRemarks"].str.lower().str.contains("dark store", na=False)]

    # 16. Comments for Mumbai
    if "Region" in df.columns:
        df.loc[df["Region"].str.upper() == "MUM", "Comments"] = "Capex raised centrally for Pan-India"

    # 17. Remove CCTV and other excluded items
    exclude_items2 = ["CCTV", "Other"]
    df = df[~df["AssetItemName"].isin(exclude_items2)]

    return df