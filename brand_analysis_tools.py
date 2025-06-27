import pandas as pd
import numpy as np
import json
from datetime import timedelta
from langchain.tools import tool

# --- Constants (can be used by tools) ---
FOCUS_RETAILER_ID = 1
FOCUS_BRAND_OWNER = ['POST HOLDINGS INC']
FOCUS_SUPERCATEGORY = ['Total Pet']
COMPETITOR_CONTRIBUTION_THRESHOLD = 0.05 # Competitor must contribute >5% of category change

# --- In-memory cache for the dataframe ---
_df_cache = None

def _load_and_prepare_data():
    """Loads and prepares the base dataframe."""
    global _df_cache
    if _df_cache is not None:
        return _df_cache
    
    try:
        df = pd.read_csv('stackline_sales.csv')
    except FileNotFoundError:
        print("WARNING: 'stackline_sales.csv' not found. Cannot proceed.")
        return pd.DataFrame()
    
    df['Week Ending'] = pd.to_datetime(df['Week Ending'])
    df = df.dropna(subset=['Organic Traffic'])
    
    # Apply initial filters
    df = df[df['Retailer ID'] == FOCUS_RETAILER_ID].copy()
    if FOCUS_SUPERCATEGORY and 'PCB_Supercategory' in df.columns:
        df = df[df['PCB_Supercategory'].isin(FOCUS_SUPERCATEGORY)].copy()

    _df_cache = df.copy()
    return _df_cache

def _get_brand_level_analysis_df():
    """
    A helper function that performs the main brand-level aggregation and delta calculations.
    This prepares the data that all tools will query using a robust aggregation method.
    """
    df = _load_and_prepare_data()
    if df.empty:
        return pd.DataFrame()

    most_recent_date = df['Week Ending'].max()
    periods = {
        'L1': (most_recent_date, most_recent_date), 'P1': (most_recent_date - timedelta(weeks=1), most_recent_date - timedelta(weeks=1)), 'Y1': (most_recent_date - timedelta(weeks=52), most_recent_date - timedelta(weeks=52)),
        'L4': (most_recent_date - timedelta(weeks=3), most_recent_date), 'P4': (most_recent_date - timedelta(weeks=7), most_recent_date - timedelta(weeks=4)), 'Y4': (most_recent_date - timedelta(weeks=55), most_recent_date - timedelta(weeks=52)), 'PP4': (most_recent_date - timedelta(weeks=11), most_recent_date - timedelta(weeks=8)),
        'L13': (most_recent_date - timedelta(weeks=12), most_recent_date), 'P13': (most_recent_date - timedelta(weeks=25), most_recent_date - timedelta(weeks=13)), 'Y13': (most_recent_date - timedelta(weeks=64), most_recent_date - timedelta(weeks=52)),
        'L26': (most_recent_date - timedelta(weeks=25), most_recent_date), 'P26': (most_recent_date - timedelta(weeks=51), most_recent_date - timedelta(weeks=26)), 'Y26': (most_recent_date - timedelta(weeks=77), most_recent_date - timedelta(weeks=52)),
    }

    all_periods_agg = []
    agg_cols = {'Retail Sales': 'sum', 'Units Sold': 'sum', 'In-Stock Rate': 'mean', 'Weeks On-Hand': 'mean', 'Buy Box - Rate': 'mean', 'Total Traffic': 'sum', 'Paid Ad Spend': 'sum', 'Retail Price': 'mean'}
    
    for period_name, (start_date, end_date) in periods.items():
        period_df = df[(df['Week Ending'] >= start_date) & (df['Week Ending'] <= end_date)]
        period_agg = period_df.groupby(['Brand Owner', 'Brand', 'PCB_Category', 'PCB_Supercategory']).agg(agg_cols).reset_index()
        period_agg['Period'] = period_name
        all_periods_agg.append(period_agg)

    long_format_df = pd.concat(all_periods_agg, ignore_index=True)
    
    pivot_df = long_format_df.pivot_table(index=['Brand Owner', 'Brand', 'PCB_Category', 'PCB_Supercategory'], columns='Period', values=agg_cols.keys())
    pivot_df.columns = ['_'.join(col).strip() for col in pivot_df.columns.values]
    pivot_df.fillna(0, inplace=True)
    
    # Standardize all column names ONCE to use underscores for consistency
    sanitized_columns = {col: col.replace(' ', '_') for col in pivot_df.columns}
    pivot_df.rename(columns=sanitized_columns, inplace=True)

    # --- Calculate All Deltas, Heuristics, and Contributions with Safe Division ---
    causal_metrics = ['Total_Traffic', 'Paid_Ad_Spend', 'Buy_Box_-_Rate']
    for p in ['1', '4', '13', '26']:
        for t in ['P', 'Y']:
            pivot_df[f'L{p}_vs_{t}{p}_Sales_Abs_Chg'] = pivot_df.get(f'Retail_Sales_L{p}', 0) - pivot_df.get(f'Retail_Sales_{t}{p}', 0)
            pivot_df[f'L{p}_vs_{t}{p}_Units_Abs_Chg'] = pivot_df.get(f'Units_Sold_L{p}', 0) - pivot_df.get(f'Units_Sold_{t}{p}', 0)
            
            numerator_sales = pivot_df[f'L{p}_vs_{t}{p}_Sales_Abs_Chg']
            denominator_sales = pivot_df.get(f'Retail_Sales_{t}{p}', 0)
            pivot_df[f'L{p}_vs_{t}{p}_Sales_Pct_Chg'] = np.where(denominator_sales != 0, numerator_sales / denominator_sales, 0)
            
            for metric_name_key in causal_metrics:
                numerator = pivot_df.get(f'{metric_name_key}_L{p}', 0) - pivot_df.get(f'{metric_name_key}_{t}{p}', 0)
                denominator = pivot_df.get(f'{metric_name_key}_{t}{p}', 0)
                pivot_df[f'{metric_name_key}_L{p}_vs_{t}{p}_Pct_Chg'] = np.where(denominator != 0, numerator / denominator, 0)

        if p in ['4', '13', '26']:
            pivot_df[f'L{p}_Price_Effect'] = (pivot_df.get(f'Retail_Price_L{p}', 0) - pivot_df.get(f'Retail_Price_P{p}', 0)) * pivot_df.get(f'Units_Sold_L{p}', 0)
            pivot_df[f'L{p}_Volume_Effect'] = (pivot_df.get(f'Units_Sold_L{p}', 0) - pivot_df.get(f'Units_Sold_P{p}', 0)) * pivot_df.get(f'Retail_Price_P{p}', 0)
            
            pivot_df[f'L{p}_Price_Effect_Pct_of_Chg'] = np.where(pivot_df[f'L{p}_vs_P{p}_Sales_Abs_Chg'] != 0, pivot_df[f'L{p}_Price_Effect'] / pivot_df[f'L{p}_vs_P{p}_Sales_Abs_Chg'], 0)
            pivot_df[f'L{p}_Volume_Effect_Pct_of_Chg'] = np.where(pivot_df[f'L{p}_vs_P{p}_Sales_Abs_Chg'] != 0, pivot_df[f'L{p}_Volume_Effect'] / pivot_df[f'L{p}_vs_P{p}_Sales_Abs_Chg'], 0)

    pivot_df['P4_vs_PP4_Sales_Abs_Chg'] = pivot_df.get('Retail_Sales_P4', 0) - pivot_df.get('Retail_Sales_PP4', 0)
    
    owner_change = pivot_df.groupby('Brand Owner')['L4_vs_P4_Sales_Abs_Chg'].transform('sum')
    category_change = pivot_df.groupby('PCB_Category')['L4_vs_P4_Sales_Abs_Chg'].transform('sum')
    pivot_df['Contribution_To_Owner_Chg'] = np.where(owner_change != 0, pivot_df['L4_vs_P4_Sales_Abs_Chg'] / owner_change, 0)
    pivot_df['Contribution_To_Cat_Chg'] = np.where(category_change != 0, pivot_df['L4_vs_P4_Sales_Abs_Chg'] / category_change, 0)
    
    return pivot_df.reset_index()


@tool
def get_category_health() -> str:
    """Provides a high-level health check on focus supercategories across L4 and L13 time periods."""
    df = _get_brand_level_analysis_df()
    if df.empty: return "[]"
    category_health = df.groupby('PCB_Supercategory').sum(numeric_only=True)
    report = []
    for cat, row in category_health.iterrows():
        report.append({"Supercategory": cat, "L4 vs P4 Sales Change": f"${row.get('L4_vs_P4_Sales_Abs_Chg', 0):,.0f}", "L4 vs Y4 Sales Change": f"${row.get('L4_vs_Y4_Sales_Abs_Chg', 0):,.0f}"})
    return json.dumps(report)

@tool
def get_performance_and_contribution_summary() -> str:
    """Provides a summary of own-brand performance (L1, L4, L13), contribution to change, and identifies major competitor movements based on their contribution to category change."""
    df = _get_brand_level_analysis_df()
    if df.empty: return "{}"
    focus_df = df[df['Brand Owner'].isin(FOCUS_BRAND_OWNER)].copy()
    focus_summary = focus_df[['Brand', 'PCB_Category', 'L1_vs_P1_Sales_Abs_Chg', 'L4_vs_P4_Sales_Abs_Chg', 'L13_vs_P13_Sales_Abs_Chg', 'L26_vs_P26_Sales_Abs_Chg', 'L4_vs_Y4_Sales_Abs_Chg', 'P4_vs_PP4_Sales_Abs_Chg']].sort_values(by='L4_vs_P4_Sales_Abs_Chg', key=abs, ascending=False)
    comp_df = df[~df['Brand Owner'].isin(FOCUS_BRAND_OWNER)].copy()
    significant_comps = comp_df[comp_df['Contribution_To_Cat_Chg'].abs() >= COMPETITOR_CONTRIBUTION_THRESHOLD].sort_values(by='L4_vs_P4_Sales_Abs_Chg', key=abs, ascending=False).head(5)
    comp_summary = significant_comps[['Brand', 'PCB_Category', 'L4_vs_P4_Sales_Abs_Chg', 'L13_vs_P13_Sales_Abs_Chg', 'Contribution_To_Cat_Chg']]
    return json.dumps({"focus_brand_summary": focus_summary.to_dict(orient='records'),"competitor_summary": comp_summary.to_dict(orient='records')})

@tool
def get_brand_and_competitor_diagnostics(brand: str, category: str) -> str:
    """Provides detailed metrics for a focus brand and its key competitors, including a pre-calculated causal summary for the focus brand."""
    df = _get_brand_level_analysis_df()
    if df.empty: return "{}"

    brand_data = df[(df['Brand'] == brand) & (df['PCB_Category'] == category)]
    
    if brand_data.empty:
        return json.dumps({"focus_brand_diagnostics": {}, "competitor_details": []})

    brand_row = brand_data.iloc[0]

    causal_factors = {}
    metrics_to_summarize = {
        'Total_Traffic': ('', ',.0f'), 
        'Paid_Ad_Spend': ('$', ',.0f'), 
        'Buy_Box_-_Rate': ('', '.1f'), 
    }
    for metric_key, (prefix, f_str) in metrics_to_summarize.items():
        metric_display_name = metric_key.replace('_', ' ')
        causal_factors[metric_display_name] = {}
        for p in ['4', '13', '26']:
            current_val = brand_row.get(f'{metric_key}_L{p}', 0)
            pct_chg_vs_prev = brand_row.get(f'{metric_key}_L{p}_vs_P{p}_Pct_Chg', 0)
            pct_chg_vs_ya = brand_row.get(f'{metric_key}_L{p}_vs_Y{p}_Pct_Chg', 0)
            interpretation = "Increased" if pct_chg_vs_prev > 0 else "Decreased"
            
            causal_factors[metric_display_name][f'L{p}'] = {
                'Current Period Value': f"{prefix}{current_val:{f_str}}",
                '% Chg vs Prev': f"{pct_chg_vs_prev:.1%}",
                '% Chg vs YA': f"{pct_chg_vs_ya:.1%}",
                'Interpretation': interpretation
            }

    focus_brand_report = {
        'Price_Volume_Decomposition': {
            f'L{p}': {
                'Total Change': f"${brand_row.get(f'L{p}_vs_P{p}_Sales_Abs_Chg', 0):,.0f}",
                'Price Effect': f"${brand_row.get(f'L{p}_Price_Effect', 0):,.0f}",
                'Volume Effect': f"${brand_row.get(f'L{p}_Volume_Effect', 0):,.0f}",
                'Price Effect % of Change': f"{brand_row.get(f'L{p}_Price_Effect_Pct_of_Chg', 0):.1%}",
                'Volume Effect % of Change': f"{brand_row.get(f'L{p}_Volume_Effect_Pct_of_Chg', 0):.1%}"
            } for p in ['4', '13', '26']
        },
        'Causal_Factors': causal_factors
    }

    comp_df = df[(df['PCB_Category'] == category) & (~df['Brand Owner'].isin(FOCUS_BRAND_OWNER))].copy()
    comp_df = comp_df.sort_values(by='Contribution_To_Cat_Chg', key=abs, ascending=False).head(5)
    
    comp_report_list = []
    for _, row in comp_df.iterrows():
        comp_report_list.append({
            'Brand': row['Brand'],
            'L4 Sales': f"${row.get('Retail_Sales_L4', 0):,.0f}",
            'L4 vs P4 Sales % Chg': f"{row.get('L4_vs_P4_Sales_Pct_Chg', 0):.1%}",
            'L4 vs Y4 Sales % Chg': f"{row.get('L4_vs_Y4_Sales_Pct_Chg', 0):.1%}",
            'L4 Contribution to Category Change': f"{row.get('Contribution_To_Cat_Chg', 0):.1%}",
        })

    return json.dumps({
        "focus_brand_diagnostics": focus_brand_report,
        "competitor_details": comp_report_list
    })

if __name__ == '__main__':
    print("--- Testing Tool with Corrected Aggregation: get_performance_and_contribution_summary ---")
    summary = get_performance_and_contribution_summary.invoke({})
    print(json.dumps(json.loads(summary), indent=2))
    
    print("\n--- Testing Tool with ENRICHED Causal and Competitor Summary ---")
    summary_data = json.loads(summary)
    if summary_data['focus_brand_summary']:
        first_brand = summary_data['focus_brand_summary'][0]
        diagnostics = get_brand_and_competitor_diagnostics.invoke({
            "brand": first_brand['Brand'], 
            "category": first_brand['PCB_Category']
        })
        print(f"running deep dive on {first_brand['Brand']} / {first_brand['PCB_Category']} ")
        print(json.dumps(json.loads(diagnostics), indent=2))
