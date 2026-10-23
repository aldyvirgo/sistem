"""Segmentasi pelanggan dengan Algoritma K-Means.

Mencakup:
- standardisasi fitur (StandardScaler manual via numpy)
- Elbow Method (inertia) dan Silhouette Score untuk k kandidat
- pelatihan K-Means final + penamaan label segmen berbasis profil RFM
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Persiapan fitur
# ---------------------------------------------------------------------------

def siapkan_fitur(rfm: pd.DataFrame, kolom: list[str] | None = None
                  ) -> tuple[np.ndarray, StandardScaler, list[str]]:
    """Standardisasi fitur RFM untuk K-Means.

    Alasan standardisasi: K-Means memakai jarak Euclidean, sehingga fitur
    dengan skala berbeda (Monetary ratusan ribu vs Recency ratusan hari)
    akan mendominasi. StandardScaler menstandarkan ke mean=0, std=1.
    """
    if kolom is None:
        kolom = ["Recency", "Frequency", "Monetary"]
    X = rfm[kolom].values.astype(float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler, kolom


# ---------------------------------------------------------------------------
# Evaluasi jumlah klaster
# ---------------------------------------------------------------------------

def evaluasi_k(X: np.ndarray, k_range=range(2, 9), random_state: int = 42) -> pd.DataFrame:
    """Hitung Inertia (Elbow) dan Silhouette Score untuk tiap k."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels) if len(set(labels)) > 1 else np.nan
        rows.append({
            "k": k,
            "inertia": round(float(km.inertia_), 2),
            "silhouette": round(float(sil), 4),
            "n_iter": int(km.n_iter_),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pelatihan final
# ---------------------------------------------------------------------------

def latih_kmeans(X: np.ndarray, k: int, random_state: int = 42) -> tuple[KMeans, np.ndarray]:
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(X)
    return km, labels


def profil_segmen(rfm: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Profil rata-rata/median RFM tiap klaster + jumlah pelanggan."""
    d = rfm.copy()
    d["Cluster"] = labels
    prof = d.groupby("Cluster").agg(
        jumlah_pelanggan=("Customer_ID", "count"),
        recency_mean=("Recency", "mean"),
        frequency_mean=("Frequency", "mean"),
        monetary_mean=("Monetary", "mean"),
        recency_median=("Recency", "median"),
        frequency_median=("Frequency", "median"),
        monetary_median=("Monetary", "median"),
        monetary_total=("Monetary", "sum"),
    ).round(2).reset_index()
    return prof


def label_segmen(prof: pd.DataFrame) -> pd.DataFrame:
    """Memberi label berbasis karakteristik data, bukan tebakan.

    Kandidat label mengikuti arahan soal: Loyal, New/Potensial, Churn Risk.
    Aturan penamaan (berbasis profil RFM relatif antar-klaster):
    - Frequency tertinggi + Monetary tinggi      -> Loyal Customers
    - Recency terbaru (terkecil) + Frequency rendah -> New Customers
    - Recency besar + Frequency rendah           -> Churn Risk
    - selain itu, Frequency/Monetary menengah    -> Potential / Hibernating
    """
    p = prof.copy()
    freq_rank = p["frequency_mean"].rank(ascending=False)
    mon_rank = p["monetary_mean"].rank(ascending=False)
    rec_rank = p["recency_mean"].rank(ascending=False)  # besar = lama

    labels = []
    for _, r in p.iterrows():
        f, m, rc = r["frequency_mean"], r["monetary_mean"], r["recency_mean"]
        if f >= p["frequency_mean"].quantile(0.66) and m >= p["monetary_mean"].quantile(0.66):
            labels.append("Loyal Customers")
        elif rc <= p["recency_mean"].quantile(0.33) and f <= p["frequency_mean"].quantile(0.5):
            labels.append("New Customers")
        elif rc >= p["recency_mean"].quantile(0.66):
            labels.append("Churn Risk")
        elif f >= p["frequency_mean"].quantile(0.5):
            labels.append("Potential Loyal")
        else:
            labels.append("Hibernating")
    p["label_segmen"] = labels
    return p
