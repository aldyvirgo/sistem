"""Modul pembuatan laporan PDF eksekutif dari hasil analisis.

Menghasilkan satu berkas PDF berisi identitas mahasiswa, ringkasan KPI,
ringkasan preprocessing, aturan Apriori teratas, profil segmen K-Means,
dan rekomendasi keputusan.
"""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
from fpdf import FPDF

FONT_DIR = r"C:\Windows\Fonts"


class LaporanPDF(FPDF):
    def __init__(self, nama_mahasiswa: str, nim: str, prodi: str, semester: str):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.mahasiswa = nama_mahasiswa
        self.nim = nim
        self.prodi = prodi
        self.semester = semester

        self.add_font("Georgia", "", os.path.join(FONT_DIR, "georgia.ttf"))
        self.add_font("Georgia", "B", os.path.join(FONT_DIR, "georgiab.ttf"))
        self.add_font("Calibri", "", os.path.join(FONT_DIR, "calibri.ttf"))
        self.add_font("Calibri", "B", os.path.join(FONT_DIR, "calibrib.ttf"))
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(18, 16, 18)

    # --- header/footer ---
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Calibri", "", 8)
        self.set_text_color(110, 110, 110)
        self.cell(0, 6, f"Laporan Analisis BI  |  {self.mahasiswa}  ({self.nim})",
                  align="L")
        self.ln(8)

    def footer(self):
        self.set_y(-14)
        self.set_font("Calibri", "", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, f"Halaman {self.page_no()}", align="C")

    # --- helper ---
    def judul_bab(self, teks: str):
        self.set_font("Georgia", "B", 14)
        self.set_text_color(31, 90, 70)      # hijau sage gelap
        self.cell(0, 9, teks, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(201, 169, 97)    # emas lembut
        self.set_line_width(0.5)
        self.line(self.l_margin, self.get_y() + 1, self.w - self.r_margin, self.get_y() + 1)
        self.ln(5)

    def subjudul(self, teks: str):
        self.set_font("Calibri", "B", 11)
        self.set_text_color(60, 70, 68)
        self.cell(0, 7, teks, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def paragraf(self, teks: str):
        self.set_font("Calibri", "", 10)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 5.5, teks)
        self.ln(2)

    def butir(self, teks: str):
        self.set_font("Calibri", "", 10)
        self.set_text_color(50, 50, 50)
        self.set_x(self.l_margin + 3)
        self.multi_cell(0, 5.5, f"-  {teks}")

    def tabel(self, headers, rows, lebar):
        self.set_font("Calibri", "B", 9)
        self.set_fill_color(31, 90, 70)
        self.set_text_color(255, 255, 255)
        for h, w in zip(headers, lebar):
            self.cell(w, 7, str(h), border=1, fill=True, align="C")
        self.ln()
        self.set_font("Calibri", "", 9)
        self.set_text_color(40, 40, 40)
        fill = False
        for row in rows:
            self.set_fill_color(244, 246, 245)
            # periksa tinggi baris (multi_cell sementara)
            x0, y0 = self.get_x(), self.get_y()
            tinggi = 6
            for val, w in zip(row, lebar):
                if y0 + tinggi > self.h - 18:
                    self.add_page()
                    x0, y0 = self.l_margin, self.get_y()
                self.cell(w, 6.5, str(val)[:60], border=1, fill=fill, align="L")
            self.ln()
            fill = not fill
        self.ln(3)


def rib(x) -> str:
    """Kelompokkan ribuan gaya Indonesia: 696085375 -> 696.085.375."""
    return f"{int(round(float(x))):,}".replace(",", ".")


def des(x, n: int = 2) -> str:
    """Desimal gaya Indonesia: pemisah ribuan titik, desimal koma."""
    s = f"{float(x):,.{n}f}"
    s = s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")
    return s


def bersih_teks(s: str) -> str:
    """Ganti karakter non-Latin standar agar font PDF aman."""
    if not isinstance(s, str):
        s = str(s)
    ganti = {
        "—": "-", "–": "-", "→": "->", "×": "x", "·": "-",
        "\u2013": "-", "\u2014": "-", "\u2192": "->",
        "\u00d7": "x", "\u2022": "-", "\u2018": "'",
        "\u2019": "'", "\u201c": '"', "\u201d": '"',
    }
    for a, b in ganti.items():
        s = s.replace(a, b)
    return s


def buat_laporan(path_keluar: str, *, identitas: dict,
                 kpi: dict, ringkas_preprocessing: pd.DataFrame,
                 rules_top: pd.DataFrame, prof_segmen: pd.DataFrame,
                 rekomendasi: pd.DataFrame, temuan: list[str]) -> str:
    pdf = LaporanPDF(identitas["nama"], identitas["nim"],
                     identitas["prodi"], identitas["semester"])

    # ---------- Sampul ----------
    pdf.add_page()
    pdf.set_fill_color(250, 248, 243)   # krem lembut
    pdf.rect(0, 0, 210, 297, "F")
    pdf.set_fill_color(31, 90, 70)
    pdf.rect(0, 0, 210, 42, "F")
    pdf.set_fill_color(201, 169, 97)
    pdf.rect(0, 42, 210, 2.5, "F")

    pdf.set_y(14)
    pdf.set_font("Georgia", "B", 20)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, "UJIAN TENGAH SEMESTER", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Georgia", "", 15)
    pdf.cell(0, 9, "Business Intelligence", align="C",
             new_x="LMARGIN", new_y="NEXT")

    pdf.set_y(62)
    pdf.set_font("Georgia", "B", 17)
    pdf.set_text_color(31, 90, 70)
    pdf.cell(0, 10, "Studi Kasus Analisis Data Transaksi Toko", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Calibri", "", 12)
    pdf.set_text_color(90, 95, 92)
    pdf.cell(0, 8, "Analisis Data Transaksi Toko", align="C",
             new_x="LMARGIN", new_y="NEXT")

    # kartu identitas
    pdf.set_y(100)
    pdf.set_x(45)
    pdf.set_fill_color(255, 255, 255)
    pdf.set_draw_color(201, 169, 97)
    pdf.set_line_width(0.4)
    pdf.rect(45, 100, 120, 68, "DF")

    pdf.set_y(106)
    pdf.set_font("Georgia", "B", 13)
    pdf.set_text_color(31, 90, 70)
    pdf.cell(0, 8, "Identitas Mahasiswa", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    for label, nilai in [("Nama", identitas["nama"]),
                         ("NIM", identitas["nim"]),
                         ("Program Studi", identitas["prodi"]),
                         ("Semester", identitas["semester"]),
                         ("Mata Kuliah", "Business Intelligence")]:
        pdf.set_x(52)
        pdf.set_font("Calibri", "B", 11)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(42, 8, label + " :")
        pdf.set_font("Calibri", "", 11)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(0, 8, bersih_teks(nilai), new_x="LMARGIN", new_y="NEXT")

    pdf.set_y(180)
    pdf.set_font("Calibri", "", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, f"Laporan dibuat pada {datetime.now():%d %B %Y}",
             align="C", new_x="LMARGIN", new_y="NEXT")

    # ---------- Ringkasan eksekutif ----------
    pdf.add_page()
    pdf.judul_bab("1. Ringkasan Eksekutif")
    pdf.paragraf(
        "Laporan ini menyajikan hasil analisis data transaksi toko sepanjang 2025, "
        "mulai dari pemeriksaan kualitas data, pembersihan, market basket analysis "
        "(Apriori), segmentasi pelanggan (K-Means), hingga rekomendasi keputusan "
        "bisnis yang dapat diterapkan manajemen.")

    pdf.subjudul("KPI Utama")
    pdf.tabel(["Metrik", "Nilai"],
              [["Total Penjualan", kpi["total_penjualan"]],
               ["Jumlah Transaksi", kpi["transaksi"]],
               ["Pelanggan Unik", kpi["pelanggan"]],
               ["Rata-rata basket_size", kpi["basket"]],
               ["Produk Unik", kpi["produk"]]],
              [80, 90])

    pdf.subjudul("Temuan Kunci")
    for t in temuan:
        pdf.butir(bersih_teks(t))

    # ---------- Preprocessing ----------
    pdf.add_page()
    pdf.judul_bab("2. Data Preprocessing")
    pdf.paragraf(
        "Data mentah (18.513 baris) dibersihkan: duplikat dihapus, sel kosong diimputasi, "
        "outlier dinetralkan, teks diseragamkan, dan Line_Total dihitung ulang. "
        "Tabel berikut merangkum perubahan sebelum dan sesudah.")
    rows = []
    for _, r in ringkas_preprocessing.iterrows():
        rows.append([r["Aspek"], r["Sebelum"], r["Sesudah"]])
    pdf.tabel(["Aspek", "Sebelum", "Sesudah"], rows, [60, 55, 55])

    # ---------- Apriori ----------
    pdf.add_page()
    pdf.judul_bab("3. Market Basket Analysis (Apriori)")
    pdf.paragraf(
        "Aturan asosiasi diperoleh dengan minimum support 2%, confidence 20%, "
        "dan lift > 1. Lift > 1 menandakan produk saling menarik untuk dibeli bersama.")
    rows = []
    for _, r in rules_top.iterrows():
        rows.append([bersih_teks(r["Aturan"]),
                     des(r['support'] * 100, 2) + "%",
                     des(r['confidence'] * 100, 1) + "%",
                     des(r['lift'], 2)])
    pdf.tabel(["Aturan (X -> Y)", "Support", "Confidence", "Lift"],
              rows, [88, 30, 32, 25])

    # ---------- K-Means ----------
    pdf.add_page()
    pdf.judul_bab("4. Segmentasi Pelanggan (K-Means)")
    pdf.paragraf(
        "Segmentasi memakai fitur Recency, Frequency, Monetary yang distandardisasi. "
        "Jumlah klaster optimal ditentukan lewat Elbow Method dan Silhouette Score.")
    rows = []
    for _, r in prof_segmen.iterrows():
        rows.append([bersih_teks(r["label_segmen"]),
                     rib(r["jumlah_pelanggan"]),
                     rib(r['recency_mean']),
                     des(r['frequency_mean'], 1),
                     rib(r['monetary_mean'])])
    pdf.tabel(["Segmen", "Jumlah", "Recency", "Frequency", "Monetary"],
              rows, [50, 25, 30, 30, 35])

    # ---------- Rekomendasi ----------
    pdf.add_page()
    pdf.judul_bab("5. Rekomendasi Keputusan Bisnis")
    rows = []
    for _, r in rekomendasi.iterrows():
        rows.append([str(r["Prioritas"]), bersih_teks(r["Tindakan"]),
                     bersih_teks(r["Urgensi"]), bersih_teks(r["Kemudahan"])])
    pdf.tabel(["No", "Tindakan", "Urgensi", "Kemudahan"], rows, [15, 105, 28, 30])

    pdf.ln(2)
    pdf.paragraf(
        "Catatan: bagian estimasi dampak bersifat proyeksi berbasis asumsi analitis, "
        "bukan fakta terukur. Dampak aktual baru dapat diketahui setelah implementasi "
        "dan pengukuran KPI pasca-penerapan.")

    pdf.output(path_keluar)
    return path_keluar
