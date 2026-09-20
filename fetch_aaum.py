import pandas as pd
import time
from amfipy import AMFIClient

# Pulls every AMC's quarterly AAUM data from AMFI for the last few years
def fetch_aaum_history(num_years=3):
    client = AMFIClient()
    fys = client.aum.financial_years()
    years_to_pull = fys[:num_years]

    # Build a list of every (year, quarter) combo we need to fetch
    periods_to_fetch = []
    for fy in years_to_pull:
        fy_periods = client.aum.periods(fy_id=fy['id'])['periods']
        for period in fy_periods:
            periods_to_fetch.append({
                "fy_id": fy['id'],
                "fy_label": fy['financial_year'],
                "period_id": period['id'],
                "period_label": period['period'],
            })

    # Fetch each quarter one by one, skipping any that fail
    all_rows = []
    for p in periods_to_fetch:
        print(f"Fetching: {p['fy_label']} | {p['period_label']}")
        try:
            data = client.aum.average_aum_fundwise(fy_id=p['fy_id'], period_id=p['period_id'])
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        for record in data:
            all_rows.append({
                "amc_name": record["MutualFundName"],
                "aaum_excl_fof": record["averageAUM"]["average_aum_excluding_domestic_including_overseas"],
                "aaum_fof_domestic": record["averageAUM"]["average_aum_fund_of_funds_domestic"],
                "period_label": p["period_label"],
                "fy_label": p["fy_label"],
            })
        time.sleep(1)  # be polite to AMFI's server

    return pd.DataFrame(all_rows)


# Adds quarter sorting, total AAUM, market share %, and growth rates
def add_metrics(history_df):
    # Maps quarter labels to Q1-Q4 codes for correct chronological sorting
    quarter_order_map = {
        "April - June": "Q1",
        "July - September": "Q2",
        "October - December": "Q3",
        "January - March": "Q4",
    }

    def make_sortable_quarter(row):
        for text, qcode in quarter_order_map.items():
            if row["period_label"].startswith(text):
                fy_start_year = row["fy_label"].split()[1]  # use FY start year so Jan-Mar sorts correctly
                return f"{fy_start_year}-{qcode}"
        return None

    history_df["quarter_id"] = history_df.apply(make_sortable_quarter, axis=1)
    history_df["total_aaum"] = history_df["aaum_excl_fof"] + history_df["aaum_fof_domestic"]

    # Each AMC's share of the total industry AAUM, per quarter
    history_df["market_share_pct"] = history_df.groupby("quarter_id")["total_aaum"].transform(
        lambda x: x / x.sum() * 100
    )

    history_df = history_df.sort_values(["amc_name", "quarter_id"]).reset_index(drop=True)

    # % change vs previous quarter, and vs same quarter last year
    history_df["qoq_growth_pct"] = history_df.groupby("amc_name")["total_aaum"].pct_change() * 100
    history_df["yoy_growth_pct"] =
