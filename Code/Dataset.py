import csv, os

pairs = [(26,20),(63,18),(19,19),(66,85),(40,40),(49,49),(8,8),(69,71),(39,39),(82,48),
         (72,72),(66,62),(25,9),(41,3),(16,75),(18,18),(22,14),(42,42),(36,52),(34,34),
         (53,39),(54,7),(51,28),(76,64),(64,15),(26,48),(16,16),(44,13),(25,14),(55,11),
         (49,49),(24,24),(44,30),(42,3),(27,47),(28,28),(2,2)]

assert len(pairs) == 37

out_dir = r"C:\abhi\AMA_ACADEMIC_PROJECT\outputs\dataset"
os.makedirs(out_dir, exist_ok=True)
path = os.path.join(out_dir, "uefa_champions_league_data.csv")

with open(path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["match", "X1_first_kick_goal_min", "X2_first_home_goal_min", "tie"])
    for i, (x1, x2) in enumerate(pairs, 1):
        w.writerow([i, x1, x2, int(x1 == x2)])

n_tie = sum(a == b for a, b in pairs)
print("saved:", path)
print("rows:", len(pairs), "| ties:", n_tie, "| X1<X2:", sum(a < b for a, b in pairs),
      "| X1>X2:", sum(a > b for a, b in pairs))