"""Q6. Market Basket Analysis using Association Rules (Apriori algorithm).

Implemented from scratch with only the standard library + pandas, so no extra
package (such as mlxtend) is needed.
Run: python 06_market_basket_analysis.py
"""
from itertools import combinations
import pandas as pd

# 1. Transactions
transactions = [
    ["bread", "milk", "butter"],
    ["bread", "diapers", "beer", "eggs"],
    ["milk", "diapers", "beer", "cola"],
    ["bread", "milk", "diapers", "beer"],
    ["bread", "milk", "diapers", "cola"],
    ["bread", "butter", "milk"],
    ["milk", "diapers", "beer"],
    ["bread", "milk", "eggs"],
    ["bread", "butter"],
    ["milk", "diapers", "bread", "butter"],
]
MIN_SUPPORT = 0.3
MIN_CONFIDENCE = 0.6
N = len(transactions)
tx_sets = [set(t) for t in transactions]


def support(itemset):
    """Fraction of transactions that contain every item of `itemset`."""
    return sum(1 for t in tx_sets if set(itemset) <= t) / N


# 2. Apriori: grow frequent itemsets level by level
items = sorted({i for t in transactions for i in t})
current = [frozenset([i]) for i in items if support([i]) >= MIN_SUPPORT]
frequent = {}
k = 1
while current:
    for s in current:
        frequent[s] = support(s)
    k += 1
    candidates = {a | b for a in current for b in current if len(a | b) == k}
    current = [c for c in candidates if support(c) >= MIN_SUPPORT]

freq_df = pd.DataFrame(
    [(", ".join(sorted(s)), sup) for s, sup in frequent.items()],
    columns=["itemset", "support"],
).sort_values("support", ascending=False)
print(f"Frequent itemsets (min support = {MIN_SUPPORT}):\n")
print(freq_df.to_string(index=False))

# 3. Generate association rules A -> B
rules = []
for itemset, sup in frequent.items():
    if len(itemset) < 2:
        continue
    for r in range(1, len(itemset)):
        for ante in combinations(itemset, r):
            ante = frozenset(ante)
            cons = itemset - ante
            conf = sup / support(ante)
            lift = conf / support(cons)
            if conf >= MIN_CONFIDENCE:
                rules.append((", ".join(sorted(ante)), ", ".join(sorted(cons)),
                              round(sup, 2), round(conf, 2), round(lift, 2)))

rules_df = pd.DataFrame(rules, columns=["antecedent", "consequent", "support", "confidence", "lift"])
rules_df = rules_df.sort_values("lift", ascending=False)
print(f"\nAssociation rules (min confidence = {MIN_CONFIDENCE}):\n")
print(rules_df.to_string(index=False))

# 4. Strong rules: positive association (lift > 1) and high confidence
strong = rules_df[(rules_df["lift"] > 1) & (rules_df["confidence"] >= 0.7)]
print("\nStrong rules (lift > 1 and confidence >= 0.7):\n")
print(strong.to_string(index=False))
print("\nInterpretation: e.g. 'diapers -> beer' with lift > 1 means customers who buy "
      "diapers are more likely than average to also buy beer.")
