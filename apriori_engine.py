"""Implementasi Algoritma Apriori + Association Rules.

Dibangun dari dasar (tanpa library eksternal) agar perhitungan support,
confidence, dan lift transparan dan dapat dipertanggungjawabkan.

Input : matriks transaksi x produk (binary DataFrame).
Output: frequent itemsets + association rules (support, confidence, lift).
"""

from __future__ import annotations

from itertools import combinations

import pandas as pd


# ---------------------------------------------------------------------------
# Frequent itemsets
# ---------------------------------------------------------------------------

def frequent_itemsets(matriks: pd.DataFrame, min_support: float = 0.02,
                      min_len: int = 1, max_len: int = 3) -> pd.DataFrame:
    """Apriori: cari frequent itemsets dengan minimum support tertentu.

    Parameters
    ----------
    matriks : pd.DataFrame
        Matriks binary transaksi x produk (1 = produk dibeli).
    min_support : float
        Support minimum (proporsi transaksi).
    max_len : int
        Ukuran itemset maksimum.

    Returns
    -------
    pd.DataFrame dengan kolom: items, support, length.
    """
    n_trans = float(len(matriks))
    if n_trans == 0:
        return pd.DataFrame(columns=["items", "support", "length"])

    kolom = list(matriks.columns)
    data = matriks.values.astype(bool)

    # hitung kemunculan tiap item (L1)
    counts: dict[frozenset, int] = {}
    for i, col in enumerate(kolom):
        c = int(data[:, i].sum())
        if c / n_trans >= min_support:
            counts[frozenset([col])] = c

    hasil = []
    for items, c in counts.items():
        hasil.append({"items": items, "support": c / n_trans})

    # iterasi k >= 2
    prev_frequent = sorted(counts.keys(), key=lambda x: sorted(x))
    k = 2
    while k <= max_len and prev_frequent:
        # join: gabung itemset yang berbagi k-2 item pertama
        candidates = set()
        prev_list = sorted(prev_frequent, key=lambda x: sorted(x))
        for i in range(len(prev_list)):
            for j in range(i + 1, len(prev_list)):
                a, b = prev_list[i], prev_list[j]
                union = a | b
                if len(union) == k:
                    # pruning: semua subset harus frequent
                    if all(frozenset(sub) in counts
                           for sub in combinations(union, k - 1)):
                        candidates.add(union)

        # hitung support kandidat
        new_counts = {}
        col_idx = {c: idx for idx, c in enumerate(kolom)}
        for cand in candidates:
            idxs = [col_idx[it] for it in cand]
            mask = data[:, idxs].all(axis=1)
            c = int(mask.sum())
            if c / n_trans >= min_support:
                new_counts[cand] = c

        for items, c in new_counts.items():
            hasil.append({"items": items, "support": c / n_trans})

        counts.update(new_counts)
        prev_frequent = sorted(new_counts.keys(), key=lambda x: sorted(x))
        k += 1

    df = pd.DataFrame(hasil)
    if df.empty:
        return pd.DataFrame(columns=["items", "support", "length"])
    df["length"] = df["items"].apply(len)
    df = df[df["length"] >= min_len]
    df = df.sort_values("support", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Association rules
# ---------------------------------------------------------------------------

def association_rules(freq: pd.DataFrame, n_trans: int,
                      min_confidence: float = 0.2,
                      min_lift: float = 1.0) -> pd.DataFrame:
    """Bentuk aturan asosiasi X -> Y dari frequent itemsets.

    Metrik:
    - support(X->Y)  = transaksi berisi X dan Y / total transaksi
    - confidence(X->Y) = support(X dan Y) / support(X)
    - lift(X->Y)     = confidence(X->Y) / support(Y)
    """
    sup_map = {frozenset(r["items"]): r["support"] for _, r in freq.iterrows()}

    rules = []
    for _, row in freq.iterrows():
        items = frozenset(row["items"])
        if len(items) < 2:
            continue
        support_xy = row["support"]
        for r in range(1, len(items)):
            for antecedent in combinations(items, r):
                antecedent = frozenset(antecedent)
                consequent = items - antecedent
                if not consequent:
                    continue
                sup_ant = sup_map.get(antecedent)
                sup_con = sup_map.get(consequent)
                if sup_ant is None or sup_con is None or sup_ant == 0:
                    continue
                confidence = support_xy / sup_ant
                if confidence < min_confidence:
                    continue
                lift = confidence / sup_con if sup_con else 0
                if lift < min_lift:
                    continue
                rules.append({
                    "antecedents": sorted(antecedent),
                    "consequents": sorted(consequent),
                    "antecedent_str": ", ".join(sorted(antecedent)),
                    "consequent_str": ", ".join(sorted(consequent)),
                    "support": round(support_xy, 4),
                    "confidence": round(confidence, 4),
                    "lift": round(lift, 4),
                    "leverage": round(support_xy - sup_ant * sup_con, 4),
                })

    df = pd.DataFrame(rules)
    if df.empty:
        return df
    # peringkat: kombinasi lift tertinggi + confidence + support
    df["skor"] = (df["lift"] * 0.5 + df["confidence"] * 0.3 + df["support"] * 0.2)
    df = df.sort_values("skor", ascending=False).reset_index(drop=True)
    return df


def jalankan_apriori(matriks: pd.DataFrame, min_support: float,
                     min_confidence: float, min_lift: float = 1.0,
                     max_len: int = 3) -> dict:
    """Pipeline lengkap Apriori."""
    freq = frequent_itemsets(matriks, min_support=min_support, max_len=max_len)
    rules = association_rules(freq, n_trans=len(matriks),
                              min_confidence=min_confidence, min_lift=min_lift)
    return {
        "frequent_itemsets": freq,
        "rules": rules,
        "n_transaksi": len(matriks),
        "min_support": min_support,
        "min_confidence": min_confidence,
        "min_lift": min_lift,
    }
