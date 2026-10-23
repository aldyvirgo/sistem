"""Modul preprocessing & EDA untuk sistem BI UTS.

Fungsi:
- memuat sheet Data_Raw, Data_Produk, Data_Pelanggan
- audit kualitas data (missing, duplikat, outlier, inkonsistensi teks)
- pembersihan & transformasi (Data_Clean)
- feature engineering: basket_size, RFM, matriks transaksi-produk
"""

from __future__ import annotations

import os
import pandas as pd
import numpy as np

from .config import DATA_PATH, OUTPUT_DIR

SHEET_RAW = "Data_Raw"
SHEET_PRODUK = "Data_Produk"
SHEET_PELANGGAN = "Data_Pelanggan"


# ---------------------------------------------------------------------------
# Pemuatan data
# ---------------------------------------------------------------------------

def load_raw() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_RAW)
    return df


def load_produk() -> pd.DataFrame:
    return pd.read_excel(DATA_PATH, sheet_name=SHEET_PRODUK)


def load_pelanggan() -> pd.DataFrame:
    return pd.read_excel(DATA_PATH, sheet_name=SHEET_PELANGGAN)


# ---------------------------------------------------------------------------
# Audit kualitas data
# ---------------------------------------------------------------------------

def audit_kualitas(df: pd.DataFrame) -> dict:
    """Menghitung berbagai metrik kualitas data mentah."""
    hasil = {}
    hasil["jumlah_baris_awal"] = int(len(df))
    hasil["jumlah_kolom"] = int(df.shape[1])
    hasil["duplikat_eksekak"] = int(df.duplicated().sum())
    hasil["missing_total"] = int(df.isna().sum().sum())
    hasil["missing_per_kolom"] = df.isna().sum().to_dict()

    # outlier numerik
    qty_invalid = int(((df["Quantity"] < 0) | (df["Quantity"] == 0) | (df["Quantity"] >= 100)).sum())
    harga_invalid = int(((df["Unit_Price"] <= 0) | (df["Unit_Price"] > 500000)).sum())
    hasil["quantity_invalid"] = qty_invalid
    hasil["harga_invalid"] = harga_invalid

    # inkonsistensi teks kategori
    cat = df["Category"].dropna().astype(str)
    hasil["kategori_inkonsisten"] = int((cat != cat.str.strip().str.title().replace(
        {"Semabko": "Sembako"})).sum())

    # kategori mentah berbeda setelah normalisasi
    raw_low = df["Category"].dropna().astype(str).str.strip().str.lower()
    hasil["kategori_variasi_huruf"] = int((df["Category"].dropna().astype(str) != raw_low.str.title()).sum())

    pm = df["Payment_Method"].dropna().astype(str)
    hasil["payment_inkonsisten"] = int((pm != pm.str.strip().str.title()).sum())

    # baris Line_Total tidak konsisten dengan Quantity*Unit_Price*(1-Diskon)
    calc = df["Quantity"] * df["Unit_Price"] * (1 - df["Discount_Rate"])
    rel = (df["Line_Total"] - calc).abs() / calc.replace(0, np.nan)
    hasil["line_total_tidak_konsisten"] = int((rel > 0.01).sum())

    hasil["transaksi_unik"] = int(df["Transaction_ID"].nunique())
    hasil["pelanggan_unik"] = int(df["Customer_ID"].nunique(dropna=True))
    hasil["produk_unik"] = int(df["Product_ID"].nunique(dropna=True))
    hasil["tanggal_min"] = df["Transaction_Date"].min()
    hasil["tanggal_maks"] = df["Transaction_Date"].max()
    return hasil


# ---------------------------------------------------------------------------
# Pembersihan & transformasi
# ---------------------------------------------------------------------------

def bersihkan_data(df: pd.DataFrame, produk: pd.DataFrame) -> pd.DataFrame:
    """Membersihkan Data_Raw menjadi Data_Clean.

    Aturan:
    1. Normalisasi teks (hapus spasi, kapitalisasi konsisten).
    2. Perbaiki kategori mengikuti katalog produk (tulis benar).
    3. Tangani missing Customer_ID / Region / Customer_Type / Payment_Method.
    4. Buang baris duplikat persis.
    5. Perbaiki outlier Quantity (<=0 atau >=100) & Unit_Price (<=0 atau >500000)
       dengan imputasi median harga katalog / median qty.
    6. Hitung ulang Line_Total = Quantity * Unit_Price * (1 - Discount_Rate).
    7. Drop baris yang tetap tidak valid (Quantity/Unit_Price kosong total).
    """
    d = df.copy()

    # 1. normalisasi teks dasar
    for col in ["Transaction_ID", "Customer_ID", "Region", "Customer_Type",
                "Product_ID", "Product_Name", "Category", "Payment_Method"]:
        d[col] = d[col].astype("string").str.strip()

    # 2. kategori & payment_Method konsisten huruf
    d["Category"] = d["Category"].str.title()
    d["Payment_Method"] = d["Payment_Method"].str.title()
    d["Region"] = d["Region"].str.title()
    d["Customer_Type"] = d["Customer_Type"].str.title()

    # kategori mengikuti katalog produk
    map_kategori = dict(zip(produk["Product_ID"], produk["Category"]))
    d["Category"] = d.apply(
        lambda r: map_kategori.get(r["Product_ID"], r["Category"])
        if pd.notna(r["Product_ID"]) else r["Category"], axis=1)

    # payment_Method seragam (Qris -> QRIS)
    d["Payment_Method"] = d["Payment_Method"].replace({"Qris": "QRIS"})

    # 3. missing Customer_ID / Region / Customer_Type
    d["Customer_ID"] = d["Customer_ID"].fillna("UNKNOWN")
    d["Region"] = d["Region"].fillna("Tidak Diketahui")
    d["Customer_Type"] = d["Customer_Type"].fillna("Tidak Diketahui")
    d["Payment_Method"] = d["Payment_Method"].fillna("Tidak Diketahui")

    # 4. drop duplikat persis (ulangi setelah normalisasi agar terdeteksi)
    d = d.drop_duplicates().copy()

    # 5a. outlier Quantity
    med_qty = d.loc[(d["Quantity"] > 0) & (d["Quantity"] < 100), "Quantity"].median()
    mask_qty_bad = (d["Quantity"].isna()) | (d["Quantity"] <= 0) | (d["Quantity"] >= 100)
    d.loc[mask_qty_bad, "Quantity"] = med_qty

    # 5b. outlier Unit_Price -> ganti dengan harga katalog
    map_harga = dict(zip(produk["Product_ID"], produk["Reference_Unit_Price_Rp"]))
    mask_harga_bad = (d["Unit_Price"].isna()) | (d["Unit_Price"] <= 0) | (d["Unit_Price"] > 500000)
    d.loc[mask_harga_bad, "Unit_Price"] = d.loc[mask_harga_bad, "Product_ID"].map(map_harga)

    # jika masih ada yang kosong (produk tidak dikenal), isi median global
    med_harga = d.loc[~mask_harga_bad, "Unit_Price"].median()
    d["Unit_Price"] = d["Unit_Price"].fillna(med_harga)

    # 6. hitung ulang Line_Total
    d["Line_Total"] = (d["Quantity"] * d["Unit_Price"] * (1 - d["Discount_Rate"])).round(0)

    # 7. drop baris yang tetap tidak valid
    d = d.dropna(subset=["Quantity", "Unit_Price", "Transaction_Date", "Product_ID"]).copy()

    d = d.reset_index(drop=True)
    return d


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def tambah_basket_size(df: pd.DataFrame) -> pd.DataFrame:
    """Menambah kolom basket_size: jumlah produk unik per Transaction_ID."""
    d = df.copy()
    basket = d.groupby("Transaction_ID")["Product_ID"].nunique().rename("basket_size")
    d = d.merge(basket, on="Transaction_ID", how="left")
    return d


def buat_rfm(df: pd.DataFrame, tanggal_acuan: pd.Timestamp | None = None) -> pd.DataFrame:
    """Membentuk fitur Recency, Frequency, Monetary per pelanggan.

    Recency : hari sejak transaksi terakhir terhadap tanggal acuan.
    Frequency: jumlah transaksi unik.
    Monetary : total penjualan bersih (Line_Total).
    """
    d = df[df["Customer_ID"] != "UNKNOWN"].copy()
    if tanggal_acuan is None:
        tanggal_acuan = d["Transaction_Date"].max() + pd.Timedelta(days=1)

    g = d.groupby("Customer_ID").agg(
        Recency=("Transaction_Date", lambda s: (tanggal_acuan - s.max()).days),
        Frequency=("Transaction_ID", "nunique"),
        Monetary=("Line_Total", "sum"),
    ).reset_index()

    g["Monetary"] = g["Monetary"].round(0)
    return g, tanggal_acuan


def buat_matriks_transaksi(df: pd.DataFrame) -> pd.DataFrame:
    """Matriks transaksi x produk (binary) untuk Apriori.

    Satu baris = satu Transaction_ID, kolom = Product_ID, nilai 1 jika dibeli.
    """
    d = df.copy()
    d["flag"] = 1
    mtx = d.pivot_table(index="Transaction_ID", columns="Product_ID",
                        values="flag", aggfunc="max", fill_value=0)
    mtx = mtx.astype(int)
    return mtx


def simpan_hasil(df: pd.DataFrame, nama: str) -> str:
    path = os.path.join(OUTPUT_DIR, nama)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path
