from pprint import pprint

from utils.load_data import *
from collections import defaultdict

all_df = load_data("datasets/train.csv")

cat = defaultdict(lambda: defaultdict(set))

for i in all_df.iloc():
    cat[i["cat1"]][i["cat2"]].add(i["cat3"])

# pprint(cat)

for i in cat:
    print(i)
    for j in cat[i]:
        print(">>",j)
        print(">>>>",cat[i][j])