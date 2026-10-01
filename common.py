import numpy as np, pandas as pd
CLS = {"FF":0,"FA":0,"SI":1,"FC":2,"SL":3,"ST":4,"SV":4,"CU":5,"KC":5,"CS":5,"CH":6,"FS":7,"FO":7}
NAME = ["FF","SI","FC","SL","ST","CU","CH","FS"]
import os, glob
HERE = os.path.dirname(os.path.abspath(__file__))
def read_snapshot():
    files = sorted(glob.glob(os.path.join(HERE, "data", "statcast_20??.csv.gz")))
    if not files: raise FileNotFoundError("data/statcast_YYYY.csv.gz missing; run fetch_full.py then snapshot.py")
    return pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
def load(min_n=250):
    df = read_snapshot()
    df = df[df.game_date.astype(str).str.len() == 10].copy()
    df["season"] = df.game_date.str[:4].astype(int)
    df["x"] = df.pitch_type.map(CLS)
    df = df.dropna(subset=["x", "balls", "strikes"]).copy()
    df["x"] = df.x.astype(np.int8)
    df = df[(df.balls <= 3) & (df.strikes <= 2)]
    df = df.drop_duplicates(["game_pk", "at_bat_number", "pitch_number"])
    df = df.sort_values(["pitcher", "season", "game_date", "game_pk", "at_bat_number", "pitch_number"], kind="stable")
    n = df.groupby(["pitcher", "season"]).x.transform("size")
    df = df[n >= min_n].reset_index(drop=True)
    df["count"] = (df.balls * 3 + df.strikes).astype(np.int8)           # 12 counts
    df["hand"] = (df.stand == "L").astype(np.int8)
    newpa = (df.pitcher.ne(df.pitcher.shift()) | df.season.ne(df.season.shift()) |
             df.game_pk.ne(df.game_pk.shift()) | df.at_bat_number.ne(df.at_bat_number.shift()))
    df["prev"] = np.where(newpa, 8, df.x.shift(fill_value=0)).astype(np.int8)       # 8 = first pitch of the PA
    z = df.zone.fillna(0).astype(int)
    pz = np.where(z.between(1, 9), 1, np.where(z >= 11, 2, 0))                      # 1 in zone, 2 out of zone
    df["prevzone"] = np.where(newpa, 0, pd.Series(pz).shift(fill_value=0)).astype(np.int8)
    return df
