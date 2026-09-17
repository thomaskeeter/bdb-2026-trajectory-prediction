"""
Quick first look at one week of NFL Big Data Bowl 2026 tracking data.
Just loads and describes the data -- no feature building yet.
"""
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

DATA_PATH = "data/input_2023_w01.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    print("=" * 80)
    print(f"Loaded: {DATA_PATH}")
    print("=" * 80)

    print("\n--- Columns ---")
    print(list(df.columns))

    print("\n--- Dtypes ---")
    print(df.dtypes)

    print("\n--- Row count ---")
    print(f"{len(df):,}")

    print("\n--- Unique values in 'event' column ---")
    if "event" in df.columns:
        print(df["event"].value_counts(dropna=False))
    else:
        print("No 'event' column found.")

    print("\n--- Sample play: all frames for one game_id/play_id ---")
    first_game_id = df["game_id"].iloc[0]
    first_play_id = df.loc[df["game_id"] == first_game_id, "play_id"].iloc[0]
    play_df = df[(df["game_id"] == first_game_id) & (df["play_id"] == first_play_id)]
    print(f"game_id={first_game_id}, play_id={first_play_id}, total rows for this play={len(play_df)}")
    print(play_df.head(5))


if __name__ == "__main__":
    main()
