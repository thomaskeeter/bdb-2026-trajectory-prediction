"""
Stage 0 preprocessing for the BDB2026 trajectory-prediction task.

Reads one week's paired input_2023_wXX.csv / output_2023_wXX.csv, and for every
player with player_to_predict == True builds a fixed-schema example:

  self_seq      (input_len<=60, F_SELF)   target player's own per-frame features
  context_seq   (input_len<=60, 16, F_CTX) up to 16 other players' per-frame features
  context_mask  (16,) bool                 which of the 16 context slots are real
  static_feats  (F_STATIC,)                non-time-varying features
  target_seq    (output_len, 2)            [x_std, y] labels, post-throw
  input_len     int                        real (post-cap) input length
  output_len    int                        real output length (== num_frames_output)

Coordinate standardization: every x-coordinate (player x, ball_land_x,
absolute_yardline_number) is mirrored so the offense always drives in the +x
direction, using the play_direction column. dir/o are mirrored to match
(dir_std = (360 - dir) % 360 when play_direction == 'left', confirmed against
raw frame-to-frame displacement vectors -- see conversation notes). target_seq
is stored in this SAME standardized coordinate frame, and must be un-mirrored
at inference/scoring time if the raw field frame is needed.

No z-score normalization is applied here -- that requires training-set-only
statistics, which we don't want to compute from a single week. This script
just builds the raw (but standardized/derived) feature arrays; normalization
is a separate step once we've settled the train/val split and are running the
full 18-week pass.
"""

import numpy as np
import pandas as pd
import torch

INPUT_MAX_LEN = 60
MAX_CONTEXT_PLAYERS = 16
VAL_WEEKS = {17, 18}

ROLE_CATEGORIES = ["Passer", "Other Route Runner", "Targeted Receiver", "Defensive Coverage"]


def load_week(week: str, data_dir: str = "data/train"):
    inp = pd.read_csv(f"{data_dir}/input_2023_w{week}.csv")
    out = pd.read_csv(f"{data_dir}/output_2023_w{week}.csv")
    return inp, out


def standardize_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror x-coordinates and dir/o angles so offense always drives toward +x."""
    df = df.copy()
    is_left = df["play_direction"] == "left"

    df["x_std"] = np.where(is_left, 120.0 - df["x"], df["x"])
    df["ball_land_x_std"] = np.where(is_left, 120.0 - df["ball_land_x"], df["ball_land_x"])
    df["los_x_std"] = np.where(is_left, 120.0 - df["absolute_yardline_number"], df["absolute_yardline_number"])
    df["dir_std"] = np.where(is_left, (360.0 - df["dir"]) % 360.0, df["dir"])
    df["o_std"] = np.where(is_left, (360.0 - df["o"]) % 360.0, df["o"])

    return df


def role_one_hot(role: str) -> np.ndarray:
    vec = np.zeros(len(ROLE_CATEGORIES), dtype=np.float32)
    if role in ROLE_CATEGORIES:
        vec[ROLE_CATEGORIES.index(role)] = 1.0
    return vec


def build_self_features(player_frames: pd.DataFrame, ball_land_x_std: float, ball_land_y: float) -> np.ndarray:
    """(T, F_SELF) for the target player's own sequence, standardized frame."""
    x = player_frames["x_std"].to_numpy()
    y = player_frames["y"].to_numpy()
    dir_rad = np.radians(player_frames["dir_std"].to_numpy())
    o_rad = np.radians(player_frames["o_std"].to_numpy())

    feats = np.stack(
        [
            x,
            y,
            ball_land_x_std - x,
            ball_land_y - y,
            x - player_frames["los_x_std"].to_numpy(),
            player_frames["s"].to_numpy(),
            player_frames["a"].to_numpy(),
            np.sin(dir_rad),
            np.cos(dir_rad),
            np.sin(o_rad),
            np.cos(o_rad),
        ],
        axis=1,
    ).astype(np.float32)
    return feats


F_SELF = 11  # x, y, dx_ball, dy_ball, dx_los, s, a, sin_dir, cos_dir, sin_o, cos_o


def build_context_features(other_frames: pd.DataFrame, target_x_std: np.ndarray, target_y: np.ndarray) -> np.ndarray:
    """(T, F_CTX) for one context player, positions relative to the target player."""
    x = other_frames["x_std"].to_numpy()
    y = other_frames["y"].to_numpy()
    dir_rad = np.radians(other_frames["dir_std"].to_numpy())
    o_rad = np.radians(other_frames["o_std"].to_numpy())
    role_vec = role_one_hot(other_frames["player_role"].iloc[0])
    role_block = np.tile(role_vec, (len(other_frames), 1))

    feats = np.concatenate(
        [
            np.stack(
                [
                    x - target_x_std,
                    y - target_y,
                    other_frames["s"].to_numpy(),
                    other_frames["a"].to_numpy(),
                    np.sin(dir_rad),
                    np.cos(dir_rad),
                    np.sin(o_rad),
                    np.cos(o_rad),
                ],
                axis=1,
            ),
            role_block,
        ],
        axis=1,
    ).astype(np.float32)
    return feats


F_CTX = 12  # dx, dy, s, a, sin_dir, cos_dir, sin_o, cos_o, + 4 role one-hot


def build_examples(inp: pd.DataFrame, out: pd.DataFrame, week: int):
    inp = standardize_direction(inp)
    examples = []

    grouped = inp.groupby(["game_id", "play_id"])
    for (game_id, play_id), play_df in grouped:
        ball_land_x_std = play_df["ball_land_x_std"].iloc[0]
        ball_land_y = play_df["ball_land_y"].iloc[0]

        predicted_players = play_df[play_df["player_to_predict"]]["nfl_id"].unique()
        if len(predicted_players) == 0:
            continue

        all_players_frames = {
            nfl_id: g.sort_values("frame_id") for nfl_id, g in play_df.groupby("nfl_id")
        }

        for target_id in predicted_players:
            target_frames = all_players_frames[target_id]
            target_role = target_frames["player_role"].iloc[0]
            num_frames_output = int(target_frames["num_frames_output"].iloc[0])

            self_seq_full = build_self_features(target_frames, ball_land_x_std, ball_land_y)
            real_len = self_seq_full.shape[0]
            input_len = min(real_len, INPUT_MAX_LEN)
            self_seq = self_seq_full[-input_len:]  # truncate to most recent frames

            target_x_std = target_frames["x_std"].to_numpy()[-input_len:]
            target_y = target_frames["y"].to_numpy()[-input_len:]

            context_ids = [pid for pid in all_players_frames if pid != target_id]
            context_seq = np.zeros((input_len, MAX_CONTEXT_PLAYERS, F_CTX), dtype=np.float32)
            context_mask = np.zeros(MAX_CONTEXT_PLAYERS, dtype=bool)

            for slot, other_id in enumerate(context_ids[:MAX_CONTEXT_PLAYERS]):
                other_frames = all_players_frames[other_id]
                # align to the same (truncated) frame range as the target player
                other_frames = other_frames[other_frames["frame_id"].isin(target_frames["frame_id"].iloc[-input_len:])]
                if len(other_frames) != input_len:
                    # a context player missing frames within this window -- skip, leave slot masked out
                    continue
                context_seq[:, slot, :] = build_context_features(other_frames, target_x_std, target_y)
                context_mask[slot] = True

            static_feats = np.concatenate(
                [
                    [ball_land_x_std, ball_land_y, float(num_frames_output)],
                    role_one_hot(target_role),
                ]
            ).astype(np.float32)

            target_rows = out[
                (out.game_id == game_id) & (out.play_id == play_id) & (out.nfl_id == target_id)
            ].sort_values("frame_id")
            # standardize target coordinates the same way as the input frame
            is_left = play_df["play_direction"].iloc[0] == "left"
            t_x = target_rows["x"].to_numpy()
            t_y = target_rows["y"].to_numpy()
            t_x_std = (120.0 - t_x) if is_left else t_x
            target_seq = np.stack([t_x_std, t_y], axis=1).astype(np.float32)
            output_len = target_seq.shape[0]

            assert output_len == num_frames_output, (
                f"output_len {output_len} != num_frames_output {num_frames_output} "
                f"for game={game_id} play={play_id} nfl_id={target_id}"
            )

            examples.append(
                {
                    "game_id": int(game_id),
                    "play_id": int(play_id),
                    "nfl_id": int(target_id),
                    "week": week,
                    "split": "val" if week in VAL_WEEKS else "train",
                    "self_seq": torch.from_numpy(self_seq),
                    "context_seq": torch.from_numpy(context_seq),
                    "context_mask": torch.from_numpy(context_mask),
                    "static_feats": torch.from_numpy(static_feats),
                    "target_seq": torch.from_numpy(target_seq),
                    "input_len": input_len,
                    "output_len": output_len,
                }
            )

    return examples


def main(week: str = "01", data_dir: str = "data/train", out_dir: str = "data/processed"):
    inp, out = load_week(week, data_dir)
    examples = build_examples(inp, out, week=int(week))

    out_path = f"{out_dir}/week{week}_examples.pt"
    torch.save(examples, out_path)

    index = pd.DataFrame(
        [
            {
                "game_id": e["game_id"],
                "play_id": e["play_id"],
                "nfl_id": e["nfl_id"],
                "week": e["week"],
                "split": e["split"],
                "input_len": e["input_len"],
                "output_len": e["output_len"],
            }
            for e in examples
        ]
    )
    index_path = f"{out_dir}/week{week}_index.csv"
    index.to_csv(index_path, index=False)

    print(f"week {week}: {len(examples)} examples -> {out_path}")
    print(f"index -> {index_path}")
    print(index[["input_len", "output_len"]].describe())


def main_all_weeks(data_dir: str = "data/train", out_dir: str = "data/processed"):
    """Run Stage 0 for weeks 01-18 and build one combined example index."""
    all_index_rows = []
    total_examples = 0
    issues = []

    for week_num in range(1, 19):
        week = f"{week_num:02d}"
        try:
            inp, out = load_week(week, data_dir)
        except FileNotFoundError as e:
            issues.append(f"week {week}: missing file -- {e}")
            continue

        expected_self_cols = {"x", "y", "s", "a", "dir", "o", "player_role", "player_to_predict",
                               "frame_id", "game_id", "play_id", "nfl_id", "play_direction",
                               "absolute_yardline_number", "num_frames_output",
                               "ball_land_x", "ball_land_y"}
        missing_cols = expected_self_cols - set(inp.columns)
        if missing_cols:
            issues.append(f"week {week}: input missing columns {missing_cols}")
            continue
        missing_out_cols = {"game_id", "play_id", "nfl_id", "frame_id", "x", "y"} - set(out.columns)
        if missing_out_cols:
            issues.append(f"week {week}: output missing columns {missing_out_cols}")
            continue

        n_before = len(inp)
        inp = inp.dropna(subset=["x", "y", "s", "a", "dir", "o"])
        n_dropped = n_before - len(inp)
        if n_dropped:
            issues.append(f"week {week}: dropped {n_dropped} input rows with NaN in core kinematic columns")

        try:
            examples = build_examples(inp, out, week=week_num)
        except AssertionError as e:
            issues.append(f"week {week}: assertion failed during build_examples -- {e}")
            continue

        out_path = f"{out_dir}/week{week}_examples.pt"
        torch.save(examples, out_path)
        total_examples += len(examples)

        for e in examples:
            all_index_rows.append({
                "game_id": e["game_id"], "play_id": e["play_id"], "nfl_id": e["nfl_id"],
                "week": e["week"], "split": e["split"],
                "input_len": e["input_len"], "output_len": e["output_len"],
            })

        print(f"week {week}: {len(examples)} examples -> {out_path}")

    index = pd.DataFrame(all_index_rows)
    index_path = f"{out_dir}/full_index.csv"
    index.to_csv(index_path, index=False)

    print()
    print(f"TOTAL examples across all weeks: {total_examples}")
    print(f"combined index -> {index_path}")
    print()
    print("split counts:")
    print(index["split"].value_counts())
    print()
    print("weeks per split:")
    print(index.groupby("split")["week"].unique())

    if issues:
        print()
        print(f"ISSUES FLAGGED ({len(issues)}):")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print()
        print("No issues flagged -- all 18 weeks processed cleanly with no schema drift or malformed rows detected.")

    return index, issues


if __name__ == "__main__":
    main()
