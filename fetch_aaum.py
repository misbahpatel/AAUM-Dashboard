import pandas as pd
import time
from amfipy import AMFIClient

# Pulls every AMC's quarterly AAUM data from AMFI for the last few years.
# Also returns the full list of quarters we INTENDED to fetch, so later steps
# can tell the difference between "no data because it doesn't exist yet" and
# "no data because this quarter's fetch failed" — both are real gaps that
# should be treated as missing, not silently skipped over.
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
                # AMFI's own complete AAUM figure — already excludes domestic Fund-of-Funds
                # to avoid double-counting (a domestic FoF invests in other domestic schemes,
                # whose AUM is already counted here), while including overseas FoF.
                "aaum_total": record["averageAUM"]["average_aum_excluding_domestic_including_overseas"],
                # Kept separately for reference — NOT added into the total, to avoid double-counting.
                "aaum_fof_domestic": record["averageAUM"]["average_aum_fund_of_funds_domestic"],
                "period_label": p["period_label"],
                "fy_label": p["fy_label"],
            })
        time.sleep(1)

    # The full set of quarters we meant to pull, in canonical "period_label | fy_label" form,
    # regardless of whether the fetch for that quarter actually succeeded.
    expected_periods = [(p["period_label"], p["fy_label"]) for p in periods_to_fetch]

    return pd.DataFrame(all_rows), expected_periods


def assign_scale_tier(share_pct):
    if pd.isna(share_pct):
        return None
    if share_pct >= 3:
        return "Large"
    elif share_pct >= 0.5:
        return "Mid-sized"
    else:
        return "Emerging"


def make_sortable_quarter(period_label, fy_label):
    quarter_order_map = {
        "April - June": "Q1",
        "July - September": "Q2",
        "October - December": "Q3",
        "January - March": "Q4",
    }
    for text, qcode in quarter_order_map.items():
        if period_label.startswith(text):
            try:
                fy_start_year = fy_label.split()[1]  # e.g. "April 2025 - March 2026" -> "2025"
                return f"{fy_start_year}-{qcode}"
            except IndexError:
                return None
    return None


def flag_suspicious_amc_changes(history_df):
    """
    Cheap detector for possible AMC renames: an AMC that was present in the
    previous quarter but is missing in the current one, in the same quarter
    that a brand-new amc_name appears for the first time. Doesn't fix
    anything — the pipeline still treats a rename as "one fund vanished,
    another appeared" — this just prints a warning so a rename doesn't pass
    unnoticed. If one is confirmed, patch it manually with a RENAME_MAP
    applied to amc_name right after fetch, before any metrics are computed.
    """
    quarters = sorted(history_df["quarter_id"].unique())
    for i in range(1, len(quarters)):
        prev_q, curr_q = quarters[i - 1], quarters[i]
        prev_amcs = set(history_df.loc[history_df["quarter_id"] == prev_q, "amc_name"])
        curr_amcs = set(history_df.loc[history_df["quarter_id"] == curr_q, "amc_name"])

        vanished = prev_amcs - curr_amcs
        new_entrants = curr_amcs - prev_amcs

        if vanished and new_entrants:
            print(f"NOTE ({curr_q}): {len(vanished)} AMC(s) vanished and "
                  f"{len(new_entrants)} new name(s) appeared — check for a rename:")
            print(f"  Vanished: {sorted(vanished)}")
            print(f"  New:      {sorted(new_entrants)}")


def add_metrics(history_df, expected_periods):
    # --- Sortable quarter label ---
    history_df["quarter_id"] = history_df.apply(
        lambda row: make_sortable_quarter(row["period_label"], row["fy_label"]), axis=1
    )

    unparsed = history_df["quarter_id"].isna()
    if unparsed.any():
        print(f"WARNING: dropping {unparsed.sum()} row(s) with an unrecognized quarter/FY label format")
    history_df = history_df[~unparsed].reset_index(drop=True)

    # --- Safety net: drop any row that looks like an industry "Total" summary, not a real AMC ---
    is_total_row = history_df["amc_name"].str.contains("total", case=False, na=False)
    if is_total_row.any():
        print(f"Dropping {is_total_row.sum()} row(s) that look like a summary/total row, not a real AMC")
    history_df = history_df[~is_total_row].reset_index(drop=True)

    # --- Safety net: drop exact duplicate (amc_name, quarter_id) rows, if the API ever double-returns one ---
    dupes = history_df.duplicated(subset=["amc_name", "quarter_id"], keep="first")
    if dupes.any():
        print(f"WARNING: dropping {dupes.sum()} duplicate (AMC, quarter) row(s)")
    history_df = history_df[~dupes].reset_index(drop=True)

    # --- Safety net: catch missing/NaN/non-numeric AAUM before it poisons % and rank calculations ---
    # A NaN here means AMFI returned no usable figure for that fund/quarter (e.g. a newly
    # registered AMC with an incomplete record) — this is NOT the same as a genuine zero AAUM.
    # Left in, NaN silently fails every threshold check (NaN >= 3 is False, NaN >= 0.5 is False),
    # so assign_scale_tier() would default it into "Emerging" not because the fund is actually
    # small, but because its size is simply unknown. Drop it explicitly instead, with a visible
    # warning, rather than let it masquerade as a real data point.
    history_df["aaum_total"] = pd.to_numeric(history_df["aaum_total"], errors="coerce")

    missing_aaum = history_df["aaum_total"].isna()
    if missing_aaum.any():
        affected = sorted(history_df.loc[missing_aaum, "amc_name"].unique())
        print(f"WARNING: dropping {missing_aaum.sum()} row(s) with missing/NaN aaum_total "
              f"(AMCs: {affected})")
    history_df = history_df[~missing_aaum].reset_index(drop=True)

    # --- Market share %, based on the full industry total for that quarter ---
    history_df["market_share_pct"] = history_df.groupby("quarter_id")["aaum_total"].transform(
        lambda x: x / x.sum() * 100
    )

    # --- Scale tier, based on % of industry AAUM, recalculated every quarter ---
    history_df["scale_tier"] = history_df["market_share_pct"].apply(assign_scale_tier)

    # --- Drop Emerging-tier AMCs with zero AAUM: not-yet-operational funds, not genuinely "smallest" ---
    zero_emerging = (history_df["scale_tier"] == "Emerging") & (history_df["aaum_total"] == 0)
    if zero_emerging.any():
        print(f"Dropping {zero_emerging.sum()} Emerging-tier row(s) with zero AAUM (fund not yet operational)")
    history_df = history_df[~zero_emerging].reset_index(drop=True)

    # --- Peer rank within the same scale tier, same quarter ---
    history_df["peer_rank"] = history_df.groupby(["quarter_id", "scale_tier"])["aaum_total"].rank(
        ascending=False, method="min"
    ).astype(int)

    # --- Tier percentile: where an AMC sits within its own tier, normalized 0-100 regardless of tier size ---
    history_df["tier_percentile"] = (
        history_df.groupby(["quarter_id", "scale_tier"])["aaum_total"].rank(pct=True) * 100
    )

    # --- Gap-aware change metrics ---
    # Build the canonical quarter grid from what we INTENDED to fetch (expected_periods),
    # not from what's actually present in the data. This means even a quarter that failed
    # to fetch ENTIRELY still shows up as a real, visible gap — not silently skipped,
    # which would otherwise make QoQ/YoY growth quietly compare across a bigger time gap
    # than "one quarter" or "one year" without any indication anything was missing.
    canonical_quarters = sorted(set(
        q for q in (make_sortable_quarter(pl, fyl) for pl, fyl in expected_periods) if q is not None
    ))

    all_quarters = sorted(set(canonical_quarters) | set(history_df["quarter_id"].unique()))
    full_index = pd.MultiIndex.from_product(
        [history_df["amc_name"].unique(), all_quarters], names=["amc_name", "quarter_id"]
    )

    grid = (
        history_df.set_index(["amc_name", "quarter_id"])[
            ["aaum_total", "market_share_pct", "peer_rank", "scale_tier"]
        ]
        .reindex(full_index)
        .reset_index()
        .sort_values(["amc_name", "quarter_id"])
    )

    grid["qoq_growth_pct"] = grid.groupby("amc_name")["aaum_total"].pct_change(fill_method=None) * 100
    grid["yoy_growth_pct"] = grid.groupby("amc_name")["aaum_total"].pct_change(periods=4, fill_method=None) * 100
    grid["market_share_change_bps"] = grid.groupby("amc_name")["market_share_pct"].diff() * 100

    grid["prev_peer_rank"] = grid.groupby("amc_name")["peer_rank"].shift(1)
    grid["prev_scale_tier"] = grid.groupby("amc_name")["scale_tier"].shift(1)

    # peer_rank is only comparable across quarters when the AMC stayed in the same tier.
    # If it moved tiers (e.g. Mid-sized -> Large), rank 3 in one tier and rank 45 in another
    # aren't on the same scale, so a naive subtraction would misreport a tier promotion as a
    # huge fall in standing. Only compute a real change when the tier is unchanged; otherwise
    # leave it blank and flag the tier move explicitly instead.
    same_tier = grid["scale_tier"] == grid["prev_scale_tier"]
    grid["peer_rank_change"] = pd.array([pd.NA] * len(grid), dtype="Int64")
    grid.loc[same_tier, "peer_rank_change"] = (
        grid.loc[same_tier, "prev_peer_rank"] - grid.loc[same_tier, "peer_rank"]
    ).astype("Int64")

    tier_order = ["Emerging", "Mid-sized", "Large"]
    grid["tier_changed"] = grid["prev_scale_tier"].notna() & ~same_tier
    grid["tier_change_direction"] = None
    grid.loc[grid["tier_changed"], "tier_change_direction"] = grid.loc[grid["tier_changed"]].apply(
        lambda r: "promoted" if tier_order.index(r["scale_tier"]) > tier_order.index(r["prev_scale_tier"])
                  else "demoted",
        axis=1,
    )

    change_cols = grid[[
        "amc_name", "quarter_id", "qoq_growth_pct", "yoy_growth_pct",
        "market_share_change_bps", "peer_rank_change", "tier_changed", "tier_change_direction"
    ]]
    history_df = history_df.merge(change_cols, on=["amc_name", "quarter_id"], how="left")

    # --- Top-5 / Top-10 concentration ratio, based on the full industry ---
    def concentration_ratios(group):
        sorted_group = group.sort_values("aaum_total", ascending=False)
        return pd.Series({
            "top5_concentration_pct": sorted_group["market_share_pct"].iloc[:5].sum(),
            "top10_concentration_pct": sorted_group["market_share_pct"].iloc[:10].sum(),
        })

    concentration = history_df.groupby("quarter_id").apply(concentration_ratios).reset_index()
    history_df = history_df.merge(concentration, on="quarter_id", how="left")

    history_df = history_df.sort_values(["amc_name", "quarter_id"]).reset_index(drop=True)

    # --- Detect possible AMC renames (vanish + new-entrant in the same quarter) ---
    # Doesn't fix anything — just prints a warning so a rename doesn't pass unnoticed.
    flag_suspicious_amc_changes(history_df)

    # --- Final rounding pass, applied only now so every calculation above used full precision ---
    round_2dp_columns = [
        "aaum_total", "aaum_fof_domestic", "market_share_pct",
        "qoq_growth_pct", "yoy_growth_pct", "market_share_change_bps",
        "top5_concentration_pct", "top10_concentration_pct", "tier_percentile",
    ]
    history_df[round_2dp_columns] = history_df[round_2dp_columns].round(2)

    return history_df


if __name__ == "__main__":
    df, expected_periods = fetch_aaum_history(num_years=3)
    df = add_metrics(df, expected_periods)
    df.to_csv("aaum_history.csv", index=False)
    print(f"Done. Saved {len(df)} rows to aaum_history.csv")
