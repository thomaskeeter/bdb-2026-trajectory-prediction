"""
Spot-check Stage 0 output against the raw CSVs for a handful of hand-picked
examples: trace a raw row -> the final self_seq/context_seq/target_seq arrays
and print both side by side so the transforms can be eyeballed.
"""

import numpy as np
import pandas as pd
import torch

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

WEEK = "01"
examples = torch.load(f"data/processed/week{WEEK}_examples.pt", weights_only=False)
raw_in = pd.read_csv(f"data/train/input_2023_w{WEEK}.csv")
raw_out = pd.read_csv(f"data/train/output_2023_w{WEEK}.csv")

by_key = {(e["game_id"], e["play_id"], e["nfl_id"]): e for e in examples}


def spot_check(game_id, play_id, nfl_id, label):
    print("=" * 100)
    print(f"CASE: {label}  (game={game_id}, play={play_id}, nfl_id={nfl_id})")
    print("=" * 100)

    ex = by_key[(game_id, play_id, nfl_id)]
    raw_player = raw_in[
        (raw_in.game_id == game_id) & (raw_in.play_id == play_id) & (raw_in.nfl_id == nfl_id)
    ].sort_values("frame_id")

    direction = raw_player["play_direction"].iloc[0]
    role = raw_player["player_role"].iloc[0]
    print(f"play_direction={direction}  player_role={role}  num_frames_output={ex['static_feats'][2].item():.0f}")
    print(f"raw frame count={len(raw_player)}  ->  input_len (post-cap)={ex['input_len']}")
    print()

    # --- raw vs standardized x, first and last kept frame ---
    kept_raw = raw_player.iloc[-ex["input_len"]:]
    print("--- self_seq: raw vs standardized, first and last kept frame ---")
    for pos in [0, -1]:
        raw_row = kept_raw.iloc[pos]
        arr_row = ex["self_seq"][pos].numpy()
        print(
            f"  frame_id={int(raw_row.frame_id):>3}  "
            f"raw(x={raw_row.x:7.2f}, y={raw_row.y:6.2f}, dir={raw_row.dir:6.2f}, o={raw_row.o:6.2f})  "
            f"-> self_seq[x={arr_row[0]:7.2f}, y={arr_row[1]:6.2f}, "
            f"dx_ball={arr_row[2]:7.2f}, dy_ball={arr_row[3]:6.2f}, dx_los={arr_row[4]:7.2f}, "
            f"s={arr_row[5]:5.2f}, a={arr_row[6]:5.2f}, "
            f"sin_dir={arr_row[7]:6.3f}, cos_dir={arr_row[8]:6.3f}, sin_o={arr_row[9]:6.3f}, cos_o={arr_row[10]:6.3f}]"
        )

    # manual re-derivation to confirm math by hand, using the LAST kept frame
    raw_last = kept_raw.iloc[-1]
    is_left = direction == "left"
    x_std_expected = (120.0 - raw_last.x) if is_left else raw_last.x
    dir_std_expected = ((360.0 - raw_last.dir) % 360.0) if is_left else raw_last.dir
    ball_land_x_std_expected = (120.0 - raw_player.ball_land_x.iloc[0]) if is_left else raw_player.ball_land_x.iloc[0]
    los_std_expected = (120.0 - raw_player.absolute_yardline_number.iloc[0]) if is_left else raw_player.absolute_yardline_number.iloc[0]

    arr_last = ex["self_seq"][-1].numpy()
    print()
    print("  hand-derived check (last kept frame):")
    print(f"    x_std expected={x_std_expected:.2f}   actual={arr_last[0]:.2f}   match={np.isclose(x_std_expected, arr_last[0])}")
    print(f"    dx_ball expected={ball_land_x_std_expected - x_std_expected:.2f}   actual={arr_last[2]:.2f}   match={np.isclose(ball_land_x_std_expected - x_std_expected, arr_last[2])}")
    print(f"    dx_los expected={x_std_expected - los_std_expected:.2f}   actual={arr_last[4]:.2f}   match={np.isclose(x_std_expected - los_std_expected, arr_last[4])}")
    dir_rad = np.radians(dir_std_expected)
    print(f"    sin_dir expected={np.sin(dir_rad):.3f}   actual={arr_last[7]:.3f}   match={np.isclose(np.sin(dir_rad), arr_last[7])}")
    print(f"    cos_dir expected={np.cos(dir_rad):.3f}   actual={arr_last[8]:.3f}   match={np.isclose(np.cos(dir_rad), arr_last[8])}")

    # --- context ---
    print()
    n_context = ex["context_mask"].sum().item()
    print(f"--- context: {n_context} / 16 real players (mask sum) ---")
    other_ids_raw = raw_in[
        (raw_in.game_id == game_id) & (raw_in.play_id == play_id) & (raw_in.nfl_id != nfl_id)
    ]["nfl_id"].nunique()
    print(f"    raw distinct other players on this play: {other_ids_raw}  (expect context real-count == min(that, 16))")

    # spot-check one real context slot's relative position at last frame
    real_slots = ex["context_mask"].nonzero().squeeze(-1).tolist()
    if isinstance(real_slots, int):
        real_slots = [real_slots]
    if real_slots:
        slot = real_slots[0]
        ctx_last = ex["context_seq"][-1, slot].numpy()
        print(f"    slot {slot}: dx_to_target={ctx_last[0]:.2f}, dy_to_target={ctx_last[1]:.2f}, role_onehot={ctx_last[8:12]}")

    # --- target_seq ---
    print()
    raw_target = raw_out[
        (raw_out.game_id == game_id) & (raw_out.play_id == play_id) & (raw_out.nfl_id == nfl_id)
    ].sort_values("frame_id")
    print(f"--- target_seq: output_len={ex['output_len']}  raw output rows={len(raw_target)} ---")
    for pos in [0, -1]:
        raw_row = raw_target.iloc[pos]
        arr_row = ex["target_seq"][pos].numpy()
        x_std_expected = (120.0 - raw_row.x) if is_left else raw_row.x
        print(
            f"  frame_id={int(raw_row.frame_id):>3}  raw(x={raw_row.x:7.2f}, y={raw_row.y:6.2f})  "
            f"-> target_seq[x_std={arr_row[0]:7.2f}, y={arr_row[1]:6.2f}]  "
            f"(expected x_std={x_std_expected:.2f}, match={np.isclose(x_std_expected, arr_row[0])})"
        )
    print()


# Case 1: 'right' direction play -- no mirroring, easiest to eyeball
spot_check(2023090700, 101, by_key_lookup_role := raw_in[(raw_in.game_id==2023090700)&(raw_in.play_id==101)&(raw_in.player_role=="Targeted Receiver")].nfl_id.iloc[0], "right-direction play, Targeted Receiver")

# Case 2: 'left' direction play -- exercises the mirroring math
tr_left = raw_in[(raw_in.game_id==2023090700)&(raw_in.play_id==194)&(raw_in.player_role=="Targeted Receiver")].nfl_id.iloc[0]
spot_check(2023090700, 194, tr_left, "left-direction play, Targeted Receiver (mirroring)")

# Case 3: a truncated example (input_len == 60, most-recent-frames kept)
spot_check(2023090700, 2906, 42460, "truncated example (62 raw frames -> 60 kept)")

# Case 4: a predicted Defensive Coverage player (context/role one-hot path)
spot_check(2023090700, 101, 46137, "predicted Defensive Coverage player")
