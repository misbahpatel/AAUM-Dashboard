import pandas as pd
import time
from amfipy import AMFIClient

# Pulls every AMC's quarterly AAUM data from AMFI for the last few years
def fetch_aaum_history(num_years=3):
    client = AMFIClient()
    fys = client.aum.financial_years()
    years_to_pull = fys[:num_years]

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
        time.sleep(1)

    return pd.DataFrame(all_rows)


# Adds quarter sorting, market share, growth, scale tier, peer rank/quartile, and concentration metrics
def add_metrics(history_df):
    # --- Sortable quarter label (uses the FY's start year so Jan-Mar sorts correctly after Oct-Dec) ---
    quarter_order_map = {
        "April - June": "Q1",
        "July - September": "Q2",
        "October - December": "Q3",
        "January - March": "Q4",
    }

    def make_sortable_quarter(row):
        for text, qcode in quarter_order_map.items():
            if row["period_label"].startswith(text):
                fy_start_year = row["fy_label"].split()[1]
                return f"{fy_start_year}-{qcode}"
        return None

    history_df["quarter_id"] = history_df.apply(make_sortable_quarter, axis=1)

    # --- Total AAUM per AMC per quarter (values are in ₹ Lakhs, as given by AMFI) ---
    history_df["total_aaum"] = history_df["aaum_excl_fof"] + history_df["aaum_fof_domestic"]

    # --- Market share %, based on the FULL industry total for that quarter ---
    history_df["market_share_pct"] = history_df.groupby("quarter_id")["total_aaum"].transform(
        lambda x: x / x.sum() * 100
    )

    history_df = history_df.sort_values(["amc_name", "quarter_id"]).reset_index(drop=True)

    # --- QoQ and YoY growth ---
    history_df["qoq_growth_pct"] = history_df.groupby("amc_name")["total_aaum"].pct_change() * 100
    history_df["yoy_growth_pct"] = history_df.groupby("amc_name")["total_aaum"].pct_change(periods=4) * 100

    # --- Scale tier: based on % of industry AAUM, recalculated every quarter (self-adjusting) ---
    def assign_scale_tier(share_pct):
        if share_pct >= 3:
            return "Large"
        elif share_pct >= 0.5:
            return "Mid-sized"
        else:
            return "Emerging/Smaller"

    history_df["scale_tier"] = history_df["market_share_pct"].apply(assign_scale_tier)

    # --- Peer-group rank: rank by AAUM within the same scale tier, same quarter ---
    history_df["peer_rank"] = history_df.groupby(["quarter_id", "scale_tier"])["total_aaum"].rank(
        ascending=False, method="min"
    ).astype(int)

    # --- Peer-group rank change vs previous quarter ---
    history_df = history_df.sort_values(["amc_name", "quarter_id"]).reset_index(drop=True)
    history_df["prev_peer_rank"] = history_df.groupby("amc_name")["peer_rank"].shift(1)
    history_df["peer_rank_change"] = history_df["prev_peer_rank"] - history_df["peer_rank"]
    history_df = history_df.drop(columns=["prev_peer_rank"])

    # --- Peer-group quartile: quartile within the same scale tier, same quarter ---
    def assign_quartile(group):
        if len(group) < 4:
            return pd.Series([None] * len(group), index=group.index)
        return pd.qcut(group.rank(method="first"), 4, labels=[4, 3, 2, 1])

    history_df["peer_quartile"] = (
        history_df.groupby(["quarter_id", "scale_tier"])["total_aaum"]
        .apply(assign_quartile)
        .reset_index(level=[0, 1], drop=True)
    )

    # --- Market share change in basis points ---
    history_df["market_share_change_bps"] = history_df.groupby("amc_name")["market_share_pct"].diff() * 100

    # --- Top-5 / Top-10 concentration ratio, based on the full industry ---
    def concentration_ratios(group):
        sorted_group = group.sort_values("total_aaum", ascending=False)
        top5_share = sorted_group["market_share_pct"].iloc[:5].sum()
        top10_share = sorted_group["market_share_pct"].iloc[:10].sum()
        return pd.Series({"top5_concentration_pct": top5_share, "top10_concentration_pct": top10_share})

    concentration = history_df.groupby("quarter_id").apply(concentration_ratios).reset_index()
    history_df = history_df.merge(concentration, on="quarter_id", how="left")

    history_df = history_df.sort_values(["amc_name", "quarter_id"]).reset_index(drop=True)

    return history_df


if __name__ == "__main__":
    df = fetch_aaum_history(num_years=3)
    df = add_metrics(df)
    df.to_csv("aaum_history.csv", index=False)
    print(f"Done. Saved {len(df)} rows to aaum_history.csv")
