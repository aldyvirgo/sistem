"""Sistem Business Intelligence - Studi Kasus Analisis Data Transaksi Toko.

Aplikasi ini mengintegrasikan seluruh tahap UTS:
  Tahap 1 : Business Understanding & EDA
  Tahap 2 : Data Preprocessing & Feature Engineering
  Tahap 3 : Apriori / Market Basket Analysis
  Tahap 4 : K-Means / Segmentasi Pelanggan
  Tahap 5 : Dashboard Business Intelligence
  Tahap 6 : Rekomendasi Keputusan Bisnis

Tema visual "Bright Cloud" (terang & cerah) dengan font Plus Jakarta Sans.
"""

from __future__ import annotations

import base64
import io
import os
import sys
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import data_pipeline as dp
from core import apriori_engine as ae
from core import segmentasi as seg
from core import report_pdf as rp
from core.config import OUTPUT_DIR

# ---------------------------------------------------------------------------
# Konfigurasi halaman & tema
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="NusaRetail Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
with open(os.path.join(ASSETS, "style.css")) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

PALETTE = ["#0EA5E9", "#10B981", "#D97706", "#F59E0B",
           "#6366F1", "#14B8A6", "#EF4444", "#8B5CF6",
           "#EC4899", "#84CC16"]

FONT_SANS = "Plus Jakarta Sans, Segoe UI, sans-serif"
FONT_MONO = "JetBrains Mono, Consolas, monospace"

PLOTLY_TEMPLATE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(241,245,249,0.55)",
    font=dict(family=FONT_SANS, color="#1E293B", size=12),
    title=dict(font=dict(family=FONT_SANS, color="#0F172A", size=15, weight=700),
               x=0.01, xanchor="left"),
    xaxis=dict(gridcolor="rgba(148,163,184,0.28)", zerolinecolor="rgba(14,165,233,0.3)",
               tickfont=dict(size=11, color="#475569"), tickformat=",.0f",
               linecolor="#CBD5E1", mirror=True),
    yaxis=dict(gridcolor="rgba(148,163,184,0.28)", zerolinecolor="rgba(14,165,233,0.3)",
               tickfont=dict(size=11, color="#475569"), tickformat=",.0f",
               linecolor="#CBD5E1", mirror=True),
    legend=dict(bgcolor="rgba(255,255,255,0)", font=dict(color="#334155", size=11),
                orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=70, r=30, t=64, b=64),
    height=400,
    separators=",.",
    hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#0EA5E9",
                    font=dict(family=FONT_SANS, size=12, color="#0F172A")),
    colorway=PALETTE,
)
px.defaults.template = None
px.defaults.color_discrete_sequence = PALETTE


def gaya(fig: go.Figure, tinggi: int | None = None,
         mr: int | None = None, ml: int | None = None,
         mb: int | None = None, mt: int | None = None) -> go.Figure:
    fig.update_layout(template=None, **PLOTLY_TEMPLATE)
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    if tinggi:
        fig.update_layout(height=tinggi)
    m = {}
    if mr is not None:
        m["r"] = mr
    if ml is not None:
        m["l"] = ml
    if mb is not None:
        m["b"] = mb
    if mt is not None:
        m["t"] = mt
    if m:
        fig.update_layout(margin=m)
    return fig


# ---------------------------------------------------------------------------
# Format angka gaya Indonesia: pemisah ribuan titik ("1.000.000"),
# desimal koma ("3,05"). Berlaku untuk KPI, tabel, teks, label, hover & PDF.
# ---------------------------------------------------------------------------

def _rib(n: int) -> str:
    """Kelompokkan ribuan dengan titik: 696085375 -> 696.085.375."""
    return f"{n:,}".replace(",", ".")


def fmt_rupiah(x: float) -> str:
    return "Rp " + _rib(int(round(x)))


def fmt_angka(x: float) -> str:
    return _rib(int(round(x)))


def label_periode() -> str:
    """Label periode data aktif, mis. 'Jan 2025 – Des 2025'."""
    if not data_siap:
        return "-"
    return (f"{clean['Transaction_Date'].min():%b %Y} – "
            f"{clean['Transaction_Date'].max():%b %Y}")


def fmt_des(x: float, n: int = 2) -> str:
    s = f"{float(x):,.{n}f}"          # gaya AS: 1,234.56
    s = s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")
    return s


def fmt_pct(x: float, n: int = 2) -> str:
    return fmt_des(x, n) + "%"


def ticks_ribuan(vals, maks: int = 4, nol: bool = True) -> tuple[list, list]:
    """(tickvals, ticktext) gaya Indonesia dengan jumlah tick terbatas.

    Dipakai pada sumbu nilai uang/angka besar agar label panjang
    (mis. 60.000.000) tidak bertumpuk di lebar separuh layar.
    """
    import math
    hi = float(np.nanmax(vals)) if len(vals) else 0.0
    lo = 0.0 if nol else float(np.nanmin(vals))
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    span = hi - lo
    kasar = span / max(maks, 1)
    p = 10.0 ** math.floor(math.log10(kasar)) if kasar > 0 else 1.0
    step = 10.0 * p
    for m in (1, 2, 5):
        if kasar <= m * p * (1 + 1e-9):
            step = m * p
            break
    start = math.ceil(lo / step) * step
    tv, v = [], start
    while v <= hi + 1e-6 * step:
        tv.append(round(v, 6))
        v += step
    if not tv:
        tv, v = [lo, hi], hi
    tt = [_rib(int(round(t))) if abs(t - round(t)) < 1e-9 else fmt_des(t, 2)
          for t in tv]
    return tv, tt


# ---------------------------------------------------------------------------
# Komponen tampilan
# ---------------------------------------------------------------------------

def kartu_kpi(items: list[tuple[str, str, str]]) -> None:
    """Kartu KPI berwarna. items = [(label, nilai, warna_hex), ...]."""
    cells = "".join(
        f'<div class="kpi" style="--k:{w}"><div class="v">{v}</div>'
        f'<div class="l">{l}</div></div>'
        for l, v, w in items)
    st.markdown(f'<div class="kpi-grid">{cells}</div>', unsafe_allow_html=True)


def tabel_html(df: pd.DataFrame, num_cols=(), wrap_cols=()) -> str:
    """Tabel HTML rapi; nilai diasumsikan string berformat (tanpa titik)."""
    cols = list(df.columns)
    th = "".join(f'<th class="{"num" if c in num_cols else ""}">{c}</th>'
                 for c in cols)
    rows = []
    for _, r in df.iterrows():
        tds = "".join(
            f'<td class="{("num" if c in num_cols else ("wrap" if c in wrap_cols else ""))}">'
            f"{r[c]}</td>"
            for c in cols)
        rows.append(f"<tr>{tds}</tr>")
    return ('<div class="tbl-wrap"><table class="tbl"><thead>'
            f"<tr>{th}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")


def hero(judul: str, sub: str, warna: str = "#0EA5E9") -> None:
    st.markdown(
        f'<div class="hero" style="--acc:{warna}"><h1>{judul}</h1>'
        f"<p>{sub}</p></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Pemrosesan data hasil UNGGAHAN MANUAL
# ---------------------------------------------------------------------------
# Data tidak ditanam di dalam aplikasi. Pengguna mengunggah berkas
# (.xlsx / .csv) pada halaman "Unggah Data", lalu menekan tombol Proses.
# Seluruh hasil analisis baru dihitung dan baru tampil setelah proses selesai.

KOLOM_RAW = [
    "Transaction_ID", "Transaction_Date", "Customer_ID", "Region",
    "Customer_Type", "Product_ID", "Product_Name", "Category", "Quantity",
    "Unit_Price", "Discount_Rate", "Line_Total", "Payment_Method",
]
KOLOM_PRODUK = ["Product_ID", "Product_Name", "Category",
                "Reference_Unit_Price_Rp"]
KOLOM_ANGKA = ("Quantity", "Unit_Price", "Discount_Rate", "Line_Total")


def _baca_csv(bites: bytes) -> pd.DataFrame:
    """Baca CSV dengan pemisah kolom yang otomatis ditebak."""
    for sep in (",", ";", "\t", "|"):
        df = pd.read_csv(io.BytesIO(bites), sep=sep)
        if df.shape[1] > 1:
            return df
    return pd.read_csv(io.BytesIO(bites))


def _nama_tabel(sheet: str) -> str:
    return str(sheet).strip().lower().replace(" ", "_")


def baca_unggahan(kecil, sheets: list[str] | None = None
                  ) -> list[tuple[str, pd.DataFrame]]:
    """Ubah satu berkas unggahan menjadi daftar (nama_tabel, dataframe).

    Mendukung CSV (satu tabel) serta Excel (boleh banyak sheet, misal
    Data_Raw, Data_Produk, Data_Pelanggan dalam satu berkas). Bila argumen
    ``sheets`` diberikan, isinya akan diisi nama-nama sheet yang terbaca.
    """
    if kecil is None:
        return []
    nama = (kecil.name or "").lower()
    isi = kecil.getvalue()
    try:
        if nama.endswith(".csv"):
            return [("data_raw", _baca_csv(isi))]
        if nama.endswith((".xlsx", ".xls")):
            xls = pd.ExcelFile(io.BytesIO(isi))
            if sheets is not None:
                sheets.extend(xls.sheet_names)
            return [(_nama_tabel(sh), pd.read_excel(xls, sheet_name=sh))
                    for sh in xls.sheet_names]
    except Exception as e:
        st.error(f"Gagal membaca berkas **{kecil.name}** — {e}")
        return []
    st.error(f"Format berkas **{kecil.name}** belum didukung. "
             "Gunakan berkas .xlsx, .xls, atau .csv.")
    return []


def _pilih(tabel: dict, *kandidat: str) -> pd.DataFrame | None:
    """Ambil tabel pertama yang cocok dari hasil unggahan."""
    for k in kandidat:
        d = tabel.get(k)
        if isinstance(d, pd.DataFrame) and not d.empty:
            return d
    return None


def _cocokkan_tipe(raw: pd.DataFrame) -> pd.DataFrame:
    """Paksa tipe data mentah agar sesuai kebutuhan pipeline."""
    d = raw.copy()
    d["Transaction_Date"] = pd.to_datetime(d["Transaction_Date"],
                                           errors="coerce")
    for c in KOLOM_ANGKA:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def _produk_dari_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Katalog produk sederhana bila berkas produk tidak diunggah."""
    g = (raw.dropna(subset=["Product_ID"])
         .groupby("Product_ID", as_index=False)
         .agg(Product_Name=("Product_Name", "first"),
              Category=("Category", "first"),
              Reference_Unit_Price_Rp=("Unit_Price", "median")))
    return g.reindex(columns=KOLOM_PRODUK)


def _pelanggan_dari_clean(clean: pd.DataFrame) -> pd.DataFrame:
    """Tabel pelanggan ringkas bila berkas pelanggan tidak diunggah."""
    d = clean[clean["Customer_ID"] != "UNKNOWN"]
    return (d.groupby("Customer_ID", as_index=False)
            .agg(Region=("Region", "first"),
                 Customer_Type=("Customer_Type", "first"),
                 Transaksi=("Transaction_ID", "nunique"),
                 Total_Belanja=("Line_Total", "sum")))


def jalankan_pipeline(raw: pd.DataFrame, produk, pelanggan) -> dict:
    """Seluruh tahap analisis: audit -> bersih -> RFM -> matriks transaksi."""
    if produk is None or produk.empty or not set(KOLOM_PRODUK) <= set(produk.columns):
        produk = _produk_dari_raw(raw)
    audit = dp.audit_kualitas(raw)
    clean = dp.bersihkan_data(raw, produk)
    clean = dp.tambah_basket_size(clean)
    rfm, tgl_acuan = dp.buat_rfm(clean)
    matriks = dp.buat_matriks_transaksi(clean)
    if pelanggan is None or pelanggan.empty:
        pelanggan = _pelanggan_dari_clean(clean)
    return dict(raw=raw, produk=produk, pelanggan=pelanggan, audit=audit,
                clean=clean, rfm=rfm, tgl_acuan=tgl_acuan, matriks=matriks)


@st.cache_data(show_spinner=False)
def jalankan_apriori_cached(matriks_key, min_sup, min_conf, min_lift):
    return ae.jalankan_apriori(st.session_state["matriks"], min_sup, min_conf, min_lift)


@st.cache_data(show_spinner=False)
def jalankan_kmeans_cached(rfm_key, k, random_state=42):
    rfm = st.session_state["rfm"]
    Xs, scaler, kolom = seg.siapkan_fitur(rfm)
    ev = seg.evaluasi_k(Xs)
    km, labels = seg.latih_kmeans(Xs, k, random_state)
    prof = seg.label_segmen(seg.profil_segmen(rfm, labels))
    return Xs, ev, labels, prof


# ---------------------------------------------------------------------------
# Status data & variabel halaman
# ---------------------------------------------------------------------------
# Seluruh halaman analisis baru memakai data setelah unggahan diproses.

data_siap = bool(st.session_state.get("data_siap", False))

if data_siap:
    raw = st.session_state["raw"]
    produk = st.session_state["produk"]
    pelanggan = st.session_state["pelanggan"]
    audit = st.session_state["audit"]
    clean = st.session_state["clean"]
    rfm = st.session_state["rfm"]
    tgl_acuan = st.session_state["tgl_acuan"]
    matriks = st.session_state["matriks"]
    nama_produk = dict(zip(produk["Product_ID"], produk["Product_Name"]))
else:
    raw = produk = pelanggan = audit = clean = rfm = matriks = None
    tgl_acuan = None
    nama_produk = {}


def label_item(items) -> str:
    if isinstance(items, str):
        items = [x.strip() for x in items.split(",")]
    return ", ".join(nama_produk.get(i, i) for i in items)


def pendek(teks: str, maks: int = 22) -> str:
    return teks if len(teks) <= maks else teks[: maks - 1] + "…"


# ---------------------------------------------------------------------------
# Identitas & Cover
# ---------------------------------------------------------------------------

IDENTITAS = {
    "nama": "Yohanes Zakharias Aldy",
    "nim": "51230170",
    "prodi": "Teknologi Informasi",
    "semester": "7",
    "kegiatan": "Ujian Tengah Semester",
    "mata_kuliah": "Business Intelligence",
}

# Kredensial login (untuk keperluan tugas/demo)
KREDENSIAL = {
    "username": "yohanes",
    "password": "zakharias51230170",
}


def cari_foto() -> str | None:
    root = os.path.dirname(os.path.abspath(__file__))
    for nama in ("pas foto", "pas_foto", "pasfoto"):
        for ext in ("png", "jpg", "jpeg", "webp"):
            p = os.path.join(root, f"{nama}.{ext}")
            if os.path.exists(p):
                return p
    for ext in ("png", "jpg", "jpeg", "webp"):
        p = os.path.join(ASSETS, f"foto.{ext}")
        if os.path.exists(p):
            return p
    return None


def foto_base64(sisi: int = 84) -> str | None:
    p = cari_foto()
    if not p:
        return None
    with open(p, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


def _isi_akun_demo() -> None:
    """Isi kolom login dengan akun demo.

    Berjalan sebagai callback tombol (sebelum widget dirender),
    sehingga aman memodifikasi nilai widget di dalamnya.
    """
    st.session_state["in_user"] = KREDENSIAL["username"]
    st.session_state["in_pass"] = KREDENSIAL["password"]


# Pesan galat yang makin ramah seiring banyaknya percobaan gagal.
PESAN_LOGIN = (
    "Ups, kombinasi **username** dan **password** belum tepat. Silakan coba lagi.",
    "Masih belum cocok. Periksa kembali huruf besar/kecilnya dan pastikan "
    "tidak ada spasi berlebih di awal atau akhir.",
    "Belum berhasil juga? Tenang — gunakan tombol **Akun demo** di atas untuk "
    "mengisi kredensial secara otomatis.",
)


def tampilkan_login() -> None:
    """Halaman masuk: bar atas, panel identitas (kiri) & kartu formulir (kanan)."""
    foto = cari_foto()
    if foto:
        with open(foto, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        ext = foto.rsplit(".", 1)[-1].lower()
        mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}[ext]
        foto_login = (f'<img class="login-foto" alt="Foto profil" '
                      f'src="data:image/{mime};base64,{b64}">')
    else:
        foto_login = '<div class="login-foto-mono">YZ</div>'

    meta = "".join(
        f'<div class="lm-row"><span>{k}</span><b>{v}</b></div>'
        for k, v in (("Mahasiswa", IDENTITAS["nama"]),
                     ("NIM", IDENTITAS["nim"]),
                     ("Program Studi", IDENTITAS["prodi"]),
                     ("Mata Kuliah", IDENTITAS["mata_kuliah"])))

    fitur = "".join(
        f'<div class="lf-item"><i>&#10003;</i><span>{t}</span></div>'
        for t in ("Market Basket Analysis &mdash; Apriori",
                  "Segmentasi Pelanggan &mdash; K-Means",
                  "Dashboard &amp; ekspor laporan PDF"))

    # --- Bar atas ---------------------------------------------------------
    st.markdown(
        """
<div class="login-topbar">
  <div class="lt-left">
    <span class="lt-mark">&#9670;</span>
    <b>NusaRetail Intelligence</b>
    <span class="lt-tag">Business Intelligence</span>
  </div>
  <div class="lt-right"><span class="lt-dot"></span> Sistem siap digunakan</div>
</div>
""",
        unsafe_allow_html=True,
    )

    kiri, kanan = st.columns([1.05, 1], gap="medium", vertical_alignment="center")

    # --- Panel kiri: identitas & nilai aplikasi ---------------------------
    with kiri:
        st.markdown(f"""
<div class="login-card">
  <div class="lc-head">
    <div class="lc-mark">&#9670;</div>
    <div class="lc-head-txt">
      <div class="login-brand">Business Intelligence</div>
      <div class="lc-head-sub">NusaRetail Analytics Platform</div>
    </div>
  </div>
  {foto_login}
  <div class="login-title">NusaRetail Intelligence</div>
  <div class="login-sub">Sistem terintegrasi analitika ritel &mdash; mengubah data
    transaksi menjadi keputusan bisnis yang lebih tepat.</div>
  <div class="login-feat">{fitur}</div>
  <div class="login-meta">{meta}</div>
  <div class="login-stats">
    <div class="ls"><div class="n">6</div><div class="l">Tahap Analisis</div></div>
    <div class="ls"><div class="n">2</div><div class="l">Metode Data Mining</div></div>
  </div>
</div>""", unsafe_allow_html=True)

    # --- Panel kanan: kartu formulir -------------------------------------
    with kanan:
        with st.form("form_login", clear_on_submit=False, border=False):
            st.markdown(
                '<div class="login-form-mark"></div>'
                '<div class="login-eyebrow"><span class="le-dot"></span> Akses Sistem</div>'
                '<div class="login-h1">Selamat datang kembali</div>'
                '<div class="login-desc">Masuk, unggah data transaksi Anda, lalu '
                'biarkan sistem menghitung analisis market basket dan '
                'segmentasi pelanggan.</div>',
                unsafe_allow_html=True,
            )
            user = st.text_input(
                "Username", key="in_user", autocomplete="username",
                placeholder="cth. yohanes", icon=":material/person_outline:",
                help="Gunakan akun yang terdaftar pada sistem.")
            pw = st.text_input(
                "Password", key="in_pass", type="password",
                autocomplete="current-password", placeholder="Masukkan password",
                icon=":material/lock_outline:",
                help="Perhatikan huruf besar dan kecilnya.")

            st.markdown(f"""
<div class="login-cred">
  <div class="lc-title">&#9432; Akun demo untuk keperluan penilaian</div>
  <div class="lc-row"><span>Username</span><code>{KREDENSIAL['username']}</code></div>
  <div class="lc-row"><span>Password</span><code>{KREDENSIAL['password']}</code></div>
</div>""", unsafe_allow_html=True)

            a1, a2 = st.columns([3, 2], gap="small")
            with a1:
                tombol_masuk = st.form_submit_button(
                    "Masuk ke Dashboard", type="primary", width="stretch",
                    icon=":material/login:")
            with a2:
                st.form_submit_button(
                    "Akun demo", width="stretch", icon=":material/auto_awesome:",
                    on_click=_isi_akun_demo)

            st.caption("Belum punya akun? Gunakan akun demo di atas. "
                       "Tekan Enter untuk masuk lebih cepat.")

            if tombol_masuk:
                u = (user or "").strip()
                p = pw or ""
                if not u or not p:
                    st.warning("Username dan password belum diisi. Silakan lengkapi "
                               "keduanya atau gunakan tombol **Akun demo**.")
                elif u == KREDENSIAL["username"] and p == KREDENSIAL["password"]:
                    st.session_state["percobaan_gagal"] = 0
                    with st.spinner("Memverifikasi akses..."):
                        time.sleep(0.8)
                    st.session_state["sudah_masuk"] = True
                    st.session_state["user_login"] = u
                    st.rerun()
                else:
                    n = st.session_state.get("percobaan_gagal", 0) + 1
                    st.session_state["percobaan_gagal"] = n
                    st.error(PESAN_LOGIN[min(n, len(PESAN_LOGIN)) - 1])

    st.markdown(
        f"""
<div class="login-foot">
  Semester {IDENTITAS['semester']} &middot; UTS Business Intelligence &middot;
  {IDENTITAS['nama']} &middot; {IDENTITAS['nim']}
</div>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Gerbang login
# ---------------------------------------------------------------------------

if not st.session_state.get("sudah_masuk", False):
    tampilkan_login()
    st.stop()

# ---------------------------------------------------------------------------
# Topbar: identitas pengguna di bagian atas halaman
# ---------------------------------------------------------------------------

b64_topbar = foto_base64(84)
if b64_topbar:
    foto_topbar = (f'<img class="tb-foto" alt="Foto profil" '
                   f'src="data:image/jpeg;base64,{b64_topbar}">')
else:
    foto_topbar = '<div class="tb-foto tb-foto-mono">YZ</div>'

if data_siap:
    status_data = ('<span class="tb-dot tb-on"></span>'
                   f"Data aktif &middot; {fmt_angka(len(clean))} baris")
else:
    status_data = '<span class="tb-dot tb-off"></span> Belum ada data'

st.markdown(
    f"""
<div class="topbar">
  <div class="tb-left">
    <div class="tb-brand">NusaRetail Intelligence</div>
    <div class="tb-sub">Business Intelligence Suite</div>
  </div>
  <div class="tb-right">
    <div class="tb-status">{status_data}</div>
    <div class="tb-user">
      {foto_topbar}
      <div class="tb-user-txt">
        <div class="tb-user-name">{IDENTITAS['nama']}</div>
        <div class="tb-user-meta">NIM {IDENTITAS['nim']} &middot;
          Semester {IDENTITAS['semester']}</div>
      </div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

NAV = [
    "00 · Unggah Data",
    "01 · Ringkasan",
    "02 · EDA & Pemahaman",
    "03 · Preprocessing",
    "04 · Apriori",
    "05 · K-Means",
    "06 · Dashboard",
    "07 · Rekomendasi",
]

NAV_UNGAH = NAV[0]

# Info ramah untuk tiap menu: (judul pendek, deskripsi singkat)
NAV_INFO = {
    "00 · Unggah Data": ("Unggah Data",
                         "Unggah berkas transaksi, lalu jalankan proses analisis."),
    "01 · Ringkasan": ("Ringkasan Eksekutif",
                       "Gambaran umum hasil analisis & peta jalan tahapan."),
    "02 · EDA & Pemahaman": ("EDA & Pemahaman",
                             "Profil data, statistik deskriptif & kualitas data."),
    "03 · Preprocessing": ("Preprocessing",
                           "Pembersihan data, fitur engineering, RFM & basket."),
    "04 · Apriori": ("Apriori",
                     "Market basket: itemset sering & aturan asosiasi."),
    "05 · K-Means": ("Segmentasi K-Means",
                     "Kelompokkan pelanggan berbasis RFM (Elbow & Silhouette)."),
    "06 · Dashboard": ("Dashboard BI",
                       "KPI, tren, filter interaktif & hasil Apriori x Segmen."),
    "07 · Rekomendasi": ("Rekomendasi",
                         "Strategi preskriptif, prioritas & estimasi dampak."),
}

def bagian_ekspor() -> None:
    """Blok ekspor di sidebar: PDF laporan & unduhan CSV (data harus siap)."""
    st.markdown('<div class="side-sub">Ekspor Laporan</div>',
                unsafe_allow_html=True)
    if st.session_state.get("pdf_bytes"):
        st.download_button("Unduh Laporan PDF", st.session_state["pdf_bytes"],
                           "Laporan_Analisis_BI.pdf", "application/pdf",
                           width="stretch")

    with st.expander("Unduh Data (CSV)"):
        st.download_button("Data Bersih", clean.to_csv(index=False).encode("utf-8-sig"),
                           "Data_Clean.csv", "text/csv", width="stretch")
        st.download_button("RFM Pelanggan", rfm.to_csv(index=False).encode("utf-8-sig"),
                           "RFM_Pelanggan.csv", "text/csv", width="stretch")
        res_def = ae.jalankan_apriori(matriks, 0.02, 0.2, 1.05)
        if not res_def["rules"].empty:
            rr = res_def["rules"].head(20).copy()
            rr["Aturan"] = rr.apply(
                lambda r: f"{label_item(r['antecedents'])} → "
                          f"{label_item(r['consequents'])}", axis=1)
            st.download_button("Aturan Apriori",
                               rr.to_csv(index=False).encode("utf-8-sig"),
                               "Aturan_Apriori.csv", "text/csv",
                               width="stretch")


def bagian_pdf() -> None:
    """Menyusun laporan PDF dari data yang sedang aktif."""
    with st.spinner("Menyusun laporan PDF..."):
        try:
            res = ae.jalankan_apriori(matriks, 0.02, 0.2, 1.05)
            rules = res["rules"]
            if not rules.empty:
                rules = rules.copy()
                rules["Aturan"] = rules.apply(
                    lambda r: f"{label_item(r['antecedents'])} → "
                              f"{label_item(r['consequents'])}", axis=1)
            _, _, _, prof = jalankan_kmeans_cached("rfm", 3)
            path_pdf = rp.buat_laporan(
                os.path.join(OUTPUT_DIR, "Laporan_Analisis_BI.pdf"),
                identitas=IDENTITAS,
                kpi={
                    "total_penjualan": fmt_rupiah(clean["Line_Total"].sum()),
                    "transaksi": fmt_angka(clean["Transaction_ID"].nunique()),
                    "pelanggan": fmt_angka(
                        clean.loc[clean.Customer_ID != "UNKNOWN",
                                  "Customer_ID"].nunique()),
                    "basket": (fmt_des(clean.groupby('Transaction_ID')
                                        ['Product_ID'].nunique().mean())
                               + " produk"),
                    "produk": fmt_angka(clean["Product_ID"].nunique()),
                },
                ringkas_preprocessing=pd.DataFrame([
                    {"Aspek": "Jumlah baris",
                     "Sebelum": fmt_angka(audit["jumlah_baris_awal"]),
                     "Sesudah": fmt_angka(len(clean))},
                    {"Aspek": "Sel kosong",
                     "Sebelum": fmt_angka(audit["missing_total"]),
                     "Sesudah": "0"},
                    {"Aspek": "Baris duplikat",
                     "Sebelum": fmt_angka(audit["duplikat_eksekak"]),
                     "Sesudah": "0"},
                    {"Aspek": "Qty tidak valid",
                     "Sebelum": fmt_angka(audit["quantity_invalid"]),
                     "Sesudah": "0"},
                    {"Aspek": "Harga tidak wajar",
                     "Sebelum": fmt_angka(audit["harga_invalid"]),
                     "Sesudah": "0"},
                    {"Aspek": "Line_Total tak konsisten",
                     "Sebelum": fmt_angka(audit["line_total_tidak_konsisten"]),
                     "Sesudah": "Konsisten"},
                ]),
                rules_top=rules.head(10) if not rules.empty else pd.DataFrame(
                    columns=["Aturan", "support", "confidence", "lift"]),
                prof_segmen=prof,
                rekomendasi=pd.DataFrame([
                    {"Prioritas": 1,
                     "Tindakan": "Bundling & penempatan rak (produk lift tertinggi)",
                     "Urgensi": "Tinggi", "Kemudahan": "Mudah"},
                    {"Prioritas": 2,
                     "Tindakan": "Win-back campaign segmen Churn Risk",
                     "Urgensi": "Tinggi", "Kemudahan": "Sedang"},
                    {"Prioritas": 3,
                     "Tindakan": "Program loyalitas Loyal Customers",
                     "Urgensi": "Sedang", "Kemudahan": "Sedang"},
                    {"Prioritas": 4,
                     "Tindakan": "Cross-sell onboarding New Customers",
                     "Urgensi": "Sedang", "Kemudahan": "Mudah"},
                ]),
                temuan=[
                    f"Data mentah {fmt_angka(audit['jumlah_baris_awal'])} baris dibersihkan "
                    f"menjadi {fmt_angka(len(clean))} baris; {fmt_angka(audit['duplikat_eksekak'])} duplikat "
                    f"dan {fmt_angka(audit['missing_total'])} sel kosong ditangani.",
                    "Aturan asosiasi terkuat ditemukan antar produk dengan lift "
                    "jauh di atas 1 (lihat bab Apriori).",
                    f"Segmentasi K-Means menghasilkan segmen Loyal, Potential, dan "
                    f"Churn Risk berdasarkan profil RFM.",
                    "Rata-rata basket_size sekitar 3 produk per transaksi - peluang "
                    "cross-selling.",
                ])
            with open(path_pdf, "rb") as fh:
                st.session_state["pdf_bytes"] = fh.read()
            st.success("Laporan PDF siap diunduh.")
        except Exception as e:
            st.error(f"Gagal membuat PDF: {e}")


# ---------------------------------------------------------------------------
# Sidebar kiri: menu navigasi utama, info data, ekspor laporan, sesi
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div class="side-menu-title">Menu Analisis</div>',
        unsafe_allow_html=True,
    )

    # Terapkan permintaan pindah halaman (mis. setelah berkas selesai
    # diproses). Dilakukan di sini karena nilai radio widget tidak boleh
    # diubah setelah widget dibuat.
    tujuan = st.session_state.pop("nav_tujuan", None)
    if tujuan in NAV:
        st.session_state["nav_utama"] = tujuan

    # Bila belum ada data diproses, paksa pengguna ke halaman Unggah Data.
    if not data_siap and st.session_state.get("nav_utama") != NAV_UNGAH:
        st.session_state["nav_utama"] = NAV_UNGAH

    def _label_nav(op: str) -> str:
        return NAV_INFO[op][0]

    halaman = st.radio("Menu analisis", NAV, key="nav_utama",
                       label_visibility="collapsed", format_func=_label_nav)

    judul_pilih, desc_pilih = NAV_INFO[halaman]
    st.markdown(
        f"""
<div class="side-menu-desc">
  <div class="smd-title">{judul_pilih}</div>
  <div class="smd-text">{desc_pilih}</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="side-sep"></div>', unsafe_allow_html=True)

    if data_siap:
        n_trx = clean["Transaction_ID"].nunique()
        n_pel = clean.loc[clean.Customer_ID != "UNKNOWN", "Customer_ID"].nunique()
        st.markdown(
            f"""
<div class="side-info">
  Periode data: <b>{clean['Transaction_Date'].min():%b %Y} &ndash;
  {clean['Transaction_Date'].max():%b %Y}</b><br>
  Baris bersih: <b>{fmt_angka(len(clean))}</b><br>
  Transaksi: <b>{fmt_angka(n_trx)}</b> &middot;
  Pelanggan: <b>{fmt_angka(n_pel)}</b><br>
  Total penjualan: <b>{fmt_rupiah(clean['Line_Total'].sum())}</b>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
<div class="side-hint">
  <div class="sh-title">Data belum tersedia</div>
  <div class="sh-text">Buka menu <b>Unggah Data</b>, unggah berkas transaksi,
    lalu tekan <b>Proses Data</b>. Hasil analisis tampil setelah proses selesai.</div>
</div>
""",
            unsafe_allow_html=True,
        )

    if data_siap:
        if st.button("Buat Laporan PDF", width="stretch",
                     icon=":material/picture_as_pdf:",
                     help="Susun laporan analisis menjadi berkas PDF"):
            bagian_pdf()
        bagian_ekspor()

    st.markdown('<div class="side-sep"></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-sub">Sesi &amp; Akun</div>',
                unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Logout", width="stretch", icon=":material/logout:",
                     help="Keluar dari sistem"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
    with col_b:
        if st.button("Reset Cache", width="stretch", icon=":material/refresh:",
                     help="Hapus cache data & resource"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Cache dibersihkan.")
            st.rerun()


# ---------------------------------------------------------------------------
# Halaman 00 - Unggah & Proses Data
# ---------------------------------------------------------------------------

def _reset_data() -> None:
    """Kosongkan data hasil unggahan beserta berkas yang tersimpan di widget."""
    for k in ("data_siap", "raw", "produk", "pelanggan", "audit", "clean",
              "rfm", "tgl_acuan", "matriks", "pdf_bytes",
              "up_raw", "up_transaksi", "up_produk", "up_pelanggan"):
        st.session_state.pop(k, None)
    st.cache_data.clear()
    st.session_state["nav_utama"] = NAV_UNGAH


KARTU_FORMAT = tabel_html(pd.DataFrame([
    {"Sheet / Tabel": "Data_Raw (wajib)",
     "Kolom minimal": ", ".join(KOLOM_RAW)},
    {"Sheet / Tabel": "Data_Produk (opsional)",
     "Kolom minimal": ", ".join(KOLOM_PRODUK)},
    {"Sheet / Tabel": "Data_Pelanggan (opsional)",
     "Kolom minimal": "Customer_ID, Region, Customer_Type (bebas)"},
]), wrap_cols=["Kolom minimal"])


def _baris_sheet(judul: str, df, wajib: bool) -> str:
    """Satu baris status sheet: ditemukan (jumlah baris) atau belum ada."""
    if df is None:
        if wajib:
            kelas, tanda = "us-miss", "&#10007;"
            ket = "tidak ditemukan"
        else:
            kelas, tanda = "us-opt", "&#9675;"
            ket = "opsional — dibuat otomatis dari data transaksi"
    else:
        kelas, tanda = "us-ok", "&#10003;"
        ket = f"{fmt_angka(len(df))} baris &middot; {df.shape[1]} kolom"
    return (f'<div class="us-row {kelas}"><i>{tanda}</i>'
            f'<span>{judul}</span><b>{ket}</b></div>')


def _header_langkah(nomor: str, judul: str, teks: str,
                    badge: str = "", badge_kelas: str = "hb-sky") -> str:
    """Judul seksi berpenomoran agar pengguna tahu sedang di langkah ke berapa."""
    return f"""
<div class="up-head">
  <div class="uh-num">{nomor}</div>
  <div class="uh-txt">
    <div class="uh-title">{judul}</div>
    <div class="uh-text">{teks}</div>
  </div>
  {'<span class="uh-badge ' + badge_kelas + '">' + badge + '</span>' if badge else ''}
</div>"""


# Alur singkat 3 langkah (selalu tampil di atas agar tidak membingungkan).
KARTU_ALUR = """
<div class="up-alur">
  <div class="al-step"><i>1</i><div><b>Unggah</b><span>Satu berkas Excel</span></div></div>
  <div class="al-line"></div>
  <div class="al-step"><i>2</i><div><b>Periksa</b><span>Sheet terdeteksi</span></div></div>
  <div class="al-line"></div>
  <div class="al-step"><i>3</i><div><b>Proses</b><span>Hitung semua analisis</span></div></div>
</div>"""

# Gambar struktur berkas Excel: satu file dengan tiga sheet di dalamnya.
KARTU_STRUKTUR = """
<div class="xl">
  <div class="xl-bar">
    <span class="xl-d"></span><span class="xl-d"></span><span class="xl-d"></span>
    <span class="xl-name">berkas-analisis.xlsx</span>
  </div>
  <div class="xl-tabs">
    <span class="xl-tab is-on">Data_Raw</span>
    <span class="xl-tab">Data_Produk</span>
    <span class="xl-tab">Data_Pelanggan</span>
  </div>
  <div class="xl-grid">
    <div class="xl-cell is-on">
      <b>Data_Raw</b><span>Data transaksi penjualan</span><i>wajib ada</i>
    </div>
    <div class="xl-cell">
      <b>Data_Produk</b><span>Katalog / daftar produk</span><i>opsional</i>
    </div>
    <div class="xl-cell">
      <b>Data_Pelanggan</b><span>Profil &amp; demografi pelanggan</span><i>opsional</i>
    </div>
  </div>
</div>"""


def _kartu_berkas(nama: str, ukuran: int, n_sheet: int) -> str:
    """Bannner hijau: berkas sudah terbaca dengan baik."""
    mb = ukuran / (1024 * 1024)
    uk = f"{mb:,.1f} MB" if mb >= 1 else f"{ukuran / 1024:,.0f} KB"
    return f"""
<div class="up-file">
  <i>&#10003;</i>
  <div class="uf-txt">
    <b>{nama}</b>
    <span>{uk} &middot; {fmt_angka(n_sheet)} sheet terbaca &middot; siap diproses</span>
  </div>
</div>"""


KARTU_KOSONG = """
<div class="up-empty">
  <i>&#9675;</i>
  <div><b>Belum ada berkas yang diunggah.</b>
  <span>Sheet yang terbaca dari berkas Anda akan tampil di sini.</span></div>
</div>"""

# Ringkasan singkat isi proses, dipasang di atas tombol Proses.
KARTU_ISI_PROSES = """
<div class="up-what">
  <div class="uw-title">Yang akan dihitung</div>
  <div class="uw-chips">
    <span>Audit &amp; pembersihan data</span>
    <span>Fitur basket size</span>
    <span>Segmentasi RFM</span>
    <span>Market basket Apriori</span>
    <span>K-Means pelanggan</span>
    <span>Dashboard &amp; ekspor PDF</span>
  </div>
</div>"""


def halaman_unggah() -> None:
    hero("Unggah & Proses Data",
         "Cukup <b>satu berkas Excel</b>: unggah, periksa sheet yang terbaca, "
         "lalu tekan <b>Proses Data</b>. Seluruh hasil analisis tampil setelah "
         "proses selesai.", "#0EA5E9")

    st.markdown(KARTU_ALUR, unsafe_allow_html=True)

    if data_siap:
        st.markdown(
            f"""
<div class="up-status">
  <div class="us-title">Data aktif sedang dipakai</div>
  <div class="us-text">{fmt_angka(len(clean))} baris bersih &middot;
    {fmt_angka(clean['Transaction_ID'].nunique())} transaksi &middot;
    {fmt_angka(clean['Product_ID'].nunique())} produk &middot;
    periode {clean['Transaction_Date'].min():%b %Y} &ndash;
    {clean['Transaction_Date'].max():%b %Y}. Unggah berkas baru di bawah untuk
    mengganti dan menghitung ulang seluruh analisis.</div>
</div>
""",
            unsafe_allow_html=True,
        )

    # --- Langkah 1: unggah satu berkas Excel ----------------------------
    st.markdown(_header_langkah(
        "1", "Unggah berkas Excel",
        "Satu berkas ini sudah memuat ketiga sheet yang dibutuhkan sistem.",
        "WAJIB", "hb-red"),
        unsafe_allow_html=True)
    st.markdown(KARTU_STRUKTUR, unsafe_allow_html=True)

    f_utama = st.file_uploader(
        "Pilih berkas Excel analisis Anda", type=["xlsx", "xls"], key="up_raw",
        help="Satu berkas Excel berisi sheet Data_Raw (wajib), "
             "Data_Produk, dan Data_Pelanggan (opsional).")

    with st.expander(
            "Opsi lanjutan — lewati bagian ini bila berkas Excel Anda sudah lengkap"):
        f_transaksi = st.file_uploader(
            "Data transaksi (opsional)", type=["csv", "xlsx", "xls"],
            key="up_transaksi",
            help="Dipakai bila tidak ada sheet Data_Raw di berkas Excel utama.")
        f_prod = st.file_uploader(
            "Katalog produk (opsional)", type=["csv", "xlsx", "xls"],
            key="up_produk",
            help="Kolom: Product_ID, Product_Name, Category, Reference_Unit_Price_Rp.")
        f_pel = st.file_uploader(
            "Data pelanggan (opsional)", type=["csv", "xlsx", "xls"],
            key="up_pelanggan",
            help="Kolom bebas, minimal Customer_ID.")

    tabel: dict[str, pd.DataFrame] = {}
    sheet_utama: list[str] = []        # nama sheet pada berkas Excel utama
    for kecil, kotak in ((f_utama, sheet_utama), (f_transaksi, None),
                         (f_prod, None), (f_pel, None)):
        for nama, df in baca_unggahan(kecil, sheets=kotak):
            tabel.setdefault(nama, df)

    raw_up = _pilih(tabel, "data_raw", "data_transaksi", "transaksi",
                    "data_utama", "data_bersih")
    if raw_up is None and len(tabel) == 1:
        raw_up = next(iter(tabel.values()))
    produk_up = _pilih(tabel, "data_produk", "produk", "katalog_produk")
    pelanggan_up = _pilih(tabel, "data_pelanggan", "pelanggan")

    hilang = list(KOLOM_RAW)
    if raw_up is not None:
        hilang = [c for c in KOLOM_RAW if c not in raw_up.columns]

    # --- Langkah 2: periksa sheet yang terbaca ---------------------------
    st.markdown(_header_langkah(
        "2", "Periksa sheet yang terbaca",
        "Sistem membaca isi berkas Anda dan melaporkan sheet yang ditemukan.",
        "otomatis", "hb-sky"),
        unsafe_allow_html=True)

    if f_utama is None and not tabel:
        st.markdown(KARTU_KOSONG, unsafe_allow_html=True)
    else:
        if f_utama is not None:
            st.markdown(_kartu_berkas(f_utama.name,
                                      len(f_utama.getvalue()),
                                      len(sheet_utama)),
                        unsafe_allow_html=True)

        # Kartu pemeriksaan sheet: mana yang terbaca dari berkas Excel.
        lain = sorted(k for k in tabel
                      if k not in ("data_raw", "data_produk", "data_pelanggan"))
        st.markdown(
            '<div class="up-sheet">'
            '<div class="uss-title">Hasil pemeriksaan sheet</div>'
            + _baris_sheet("Data_Raw &mdash; data transaksi", raw_up, True)
            + _baris_sheet("Data_Produk &mdash; katalog produk", produk_up, False)
            + _baris_sheet("Data_Pelanggan &mdash; data pelanggan",
                           pelanggan_up, False)
            + (f'<div class="uss-note">Sheet lain yang diabaikan: '
               f'{", ".join(lain)}</div>' if lain else '')
            + '</div>',
            unsafe_allow_html=True,
        )

        if f_utama is not None and raw_up is None:
            st.error("Berkas terunggah, tetapi sheet **Data_Raw** tidak ditemukan. "
                     "Pastikan berkas Excel memuat sheet bernama Data_Raw "
                     "(lihat gambar struktur berkas di langkah 1).")
        elif raw_up is not None and hilang:
            st.error("Kolom wajib belum lengkap pada sheet Data_Raw: "
                     + ", ".join(f"`{c}`" for c in hilang))
            st.markdown("#### Format kolom yang diharapkan\n" + KARTU_FORMAT,
                        unsafe_allow_html=True)

        if raw_up is not None and not hilang:
            uji = _cocokkan_tipe(raw_up)
            n_tgl = int(uji["Transaction_Date"].isna().sum())
            kartu_kpi([
                ("Baris Data", fmt_angka(len(raw_up)), "#0EA5E9"),
                ("Kolom", fmt_angka(raw_up.shape[1]), "#14B8A6"),
                ("Transaksi Unik", fmt_angka(uji["Transaction_ID"].nunique()), "#D97706"),
                ("Produk Unik", fmt_angka(uji["Product_ID"].nunique()), "#6366F1"),
            ])
            if n_tgl:
                st.warning(f"{fmt_angka(n_tgl)} baris memiliki tanggal yang tidak "
                           "terbaca dan akan dibuang saat proses pembersihan.")
            st.markdown("#### Pratinjau 10 baris pertama")
            st.dataframe(raw_up.head(10), hide_index=True)

    # --- Langkah 3: proses data ------------------------------------------
    st.markdown(_header_langkah(
        "3", "Proses data Anda",
        "Satu klik untuk menghitung seluruh analisis dari berkas di atas.",
        "", "hb-sky"),
        unsafe_allow_html=True)
    st.markdown(KARTU_ISI_PROSES, unsafe_allow_html=True)

    siap_proses = raw_up is not None and not hilang
    tombol_proses = st.button(
        "Proses Data Sekarang", type="primary", width="stretch",
        icon=":material/play_arrow:",
        disabled=not siap_proses,
        help="Bersihkan data & jalankan seluruh tahap analisis.")

    if not siap_proses:
        st.caption(
            "Tombol ini aktif setelah berkas Excel terunggah dan sheet "
            "**Data_Raw** terbaca lengkap di langkah 2.")
    else:
        st.caption(
            "Perkiraan waktu proses: beberapa detik, lalu aplikasi otomatis "
            "pindah ke menu Ringkasan.")

    if tombol_proses:
        with st.spinner("Membaca berkas & menghitung seluruh tahap analisis..."):
            try:
                hasil = jalankan_pipeline(_cocokkan_tipe(raw_up),
                                          produk_up, pelanggan_up)
            except Exception as e:
                st.error(f"Proses gagal dijalankan: {e}")
                st.stop()
            st.session_state.update(hasil)
            st.session_state["data_siap"] = True
            st.session_state["pdf_bytes"] = None
            st.cache_data.clear()          # buang hasil analisis berkas lama
            # Radio menu sudah dibuat di sidebar, jadi tujuan perpindahan
            # disimpan sebagai state biasa (dipakai sidebar pada run berikut).
            st.session_state["nav_tujuan"] = "01 · Ringkasan"
        st.success("Data berhasil diproses. Seluruh hasil analisis siap ditampilkan.")
        time.sleep(0.8)
        st.rerun()

    if data_siap:
        st.markdown('<div class="up-sep"></div>', unsafe_allow_html=True)
        st.button("Hapus data & unggah berkas baru", width="stretch",
                  icon=":material/delete_outline:", on_click=_reset_data,
                  help="Kosongkan sesi lalu unggah berkas yang berbeda.")


# ---------------------------------------------------------------------------
# Halaman 01 - Ringkasan
# ---------------------------------------------------------------------------

def halaman_ringkasan():
    hero("Ringkasan Eksekutif",
         "Gambaran menyeluruh hasil analisis: EDA, preprocessing, Apriori, "
         "K-Means, hingga rekomendasi bisnis.", "#0EA5E9")

    kartu_kpi([
        ("Total Penjualan", fmt_rupiah(clean["Line_Total"].sum()), "#0EA5E9"),
        ("Transaksi Unik", fmt_angka(clean["Transaction_ID"].nunique()), "#D97706"),
        ("Pelanggan Unik", fmt_angka(clean.loc[clean.Customer_ID != 'UNKNOWN',
                                               'Customer_ID'].nunique()), "#6366F1"),
        ("Produk", fmt_angka(clean["Product_ID"].nunique()), "#F59E0B"),
    ])

    st.markdown("### Peta Jalan Analisis")
    peta = [
        ("Tahap 1 - EDA", "Memahami masalah bisnis, eksplorasi & temuan kualitas data.", "10%"),
        ("Tahap 2 - Preprocessing", "Pembersihan, transformasi, RFM & basket_size.", "25%"),
        ("Tahap 3 - Apriori", "Market basket: frequent itemsets & aturan asosiasi.", "20%"),
        ("Tahap 4 - K-Means", "Segmentasi pelanggan berbasis RFM, Elbow & Silhouette.", "20%"),
        ("Tahap 5 - Dashboard BI", "KPI, tren, integrasi Apriori x segmen, filter interaktif.", "15%"),
        ("Tahap 6 - Rekomendasi", "Strategi preskriptif, prioritas & estimasi dampak.", "10%"),
    ]
    cols = st.columns(2)
    for i, (judul, isi, bobot) in enumerate(peta):
        with cols[i % 2].expander(f"{judul}  -  Bobot {bobot}"):
            st.write(isi)

    st.markdown("### Tren Penjualan & Produk Teratas")
    v1, v2 = st.columns(2)
    bulanan = (clean.set_index("Transaction_Date")
               .resample("ME")["Line_Total"].sum().reset_index())
    bulanan["Bulan"] = bulanan["Transaction_Date"].dt.strftime("%b")
    with v1:
        fig = px.line(bulanan, x="Bulan", y="Line_Total", markers=True,
                      title=f"Tren Penjualan Bulanan ({label_periode()})",
                      color_discrete_sequence=["#10B981"])
        fig.update_traces(line=dict(width=3),
                          hovertemplate="%{x}: Rp %{y:,.0f}<extra></extra>")
        tv, tt = ticks_ribuan(bulanan["Line_Total"], maks=4)
        fig.update_yaxes(tickvals=tv, ticktext=tt, title_text="Penjualan (Rp)")
        gaya(fig, tinggi=400, mb=70)
        st.plotly_chart(fig, use_container_width=True)
    with v2:
        tp = (clean.groupby("Product_Name")["Line_Total"].sum()
              .sort_values(ascending=False).head(8).reset_index())
        tp["LabelNilai"] = tp["Line_Total"].map(fmt_angka)
        fig = px.bar(tp, x="Line_Total", y="Product_Name", orientation="h",
                     text="LabelNilai", title="8 Produk Penjualan Tertinggi",
                     labels={"Line_Total": "Penjualan (Rp)", "Product_Name": "Produk"},
                     color_discrete_sequence=["#D97706"])
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        tv, tt = ticks_ribuan(tp["Line_Total"], maks=3)
        fig.update_xaxes(tickvals=tv, ticktext=tt, range=[0, tp["Line_Total"].max() * 1.22])
        fig.update_traces(textposition="outside", textfont=dict(size=10),
                          hovertemplate="%{y}<br>Rp %{x:,.0f}<extra></extra>")
        gaya(fig, tinggi=400, mr=100, ml=110)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Ringkasan Temuan Kunci")
    res = ae.jalankan_apriori(matriks, 0.02, 0.2, 1.05)
    rules = res["rules"]
    top_rule = "belum cukup data"
    if not rules.empty:
        r0 = rules.iloc[0]
        top_rule = (f"pasangan **{label_item(r0['antecedents'])} → "
                    f"{label_item(r0['consequents'])}** (lift {fmt_des(r0['lift'])}, "
                    f"confidence {fmt_pct(r0['confidence'] * 100, 1)})")
    _, _, _, prof = jalankan_kmeans_cached("rfm", 3)
    seg_count = prof["label_segmen"].value_counts().index[0]
    st.markdown(
        f"""
- **Kualitas data:** {fmt_angka(audit['duplikat_eksekak'])} baris duplikat, {fmt_angka(audit['missing_total'])} sel kosong,
  {fmt_angka(audit['quantity_invalid'])} nilai Quantity tidak valid, dan {fmt_angka(audit['harga_invalid'])} harga tidak wajar
  ditemukan pada {fmt_angka(audit['jumlah_baris_awal'])} baris data mentah - semuanya telah ditangani.
- **Pola belanja:** aturan paling menarik adalah {top_rule}.
- **Segmentasi:** k terbaik menurut Silhouette = **{int(ev_best_k())} klaster**; segmen terbesar **{seg_count}**.
- **Implikasi:** hasil analisis menjadi dasar rekomendasi bundling, penempatan rak, dan retensi pelanggan.
        """
    )


def ev_best_k() -> int:
    _, ev, _, _ = jalankan_kmeans_cached("rfm", 3)
    return int(ev.sort_values("silhouette", ascending=False).iloc[0]["k"])


# ---------------------------------------------------------------------------
# Halaman 01 - EDA
# ---------------------------------------------------------------------------

def halaman_eda():
    hero("01 - Business Understanding & EDA",
         "Memahami permasalahan bisnis dan mengeksplorasi karakteristik data transaksi.")

    st.markdown("### 1a. Permasalahan Bisnis & Pertanyaan Analitis")
    st.markdown(
        """
Perusahaan ritel hanya memakai data transaksi untuk **laporan rutin**, belum untuk memahami
pola belanja, kombinasi produk, segmentasi pelanggan, dan strategi pemasaran. Data juga berpotensi
bermasalah (missing, duplikat, outlier, format tidak konsisten).

**Pertanyaan bisnis yang dirumuskan:**
1. **Produk apa yang sering dibeli bersama?** - untuk bundling & penempatan rak (Apriori).
2. **Segmen pelanggan mana yang paling bernilai dan mana yang berisiko hilang?** - retensi (K-Means).
3. **Kapan penjualan meningkat & produk mana yang mendominasi?** - perencanaan stok & promosi (EDA).
4. **Bagaimana karakteristik transaksi (basket_size) memengaruhi nilai penjualan?** - strategi cross-sell.
        """
    )

    st.markdown("### 1b. Profil Data (Sebelum Pembersihan)")
    kartu_kpi([
        ("Baris x Kolom",
         f"{fmt_angka(audit['jumlah_baris_awal'])} x {fmt_angka(audit['jumlah_kolom'])}",
         "#6366F1"),
        ("Transaksi Unik", fmt_angka(audit["transaksi_unik"]), "#0EA5E9"),
        ("Pelanggan Unik", fmt_angka(audit["pelanggan_unik"]), "#D97706"),
        ("Rentang Tanggal",
         f"{audit['tanggal_min']:%d/%m/%y} - {audit['tanggal_maks']:%d/%m/%y}", "#F59E0B"),
    ])

    st.markdown("### 1c. Statistik Deskriptif")
    desc = raw[["Quantity", "Unit_Price", "Discount_Rate", "Line_Total"]].describe().T
    desc = desc.rename(columns={"count": "Jumlah", "mean": "Rata-rata", "std": "Std",
                                "min": "Min", "25%": "Q1", "50%": "Median",
                                "75%": "Q3", "max": "Max"})
    desc_fmt = desc.copy()
    for c in desc_fmt.columns:
        if c == "Jumlah":
            desc_fmt[c] = desc[c].map(fmt_angka)
        elif c == "Std":
            desc_fmt[c] = desc[c].map(lambda v: fmt_des(v, 1))
        else:
            desc_fmt[c] = desc[c].map(lambda v: fmt_des(v, 1))
    desc_fmt.insert(0, "Statistik", desc_fmt.index)
    st.markdown(tabel_html(desc_fmt.reset_index(drop=True),
                           num_cols=["Jumlah", "Rata-rata", "Std", "Min", "Q1",
                                     "Median", "Q3", "Max"]),
                unsafe_allow_html=True)

    st.markdown("### 1d. Kualitas Data (Missing, Duplikat, Outlier)")
    q1, q2 = st.columns(2)
    miss = pd.Series(audit["missing_per_kolom"])
    miss = miss[miss > 0].sort_values(ascending=False)
    with q1:
        md = pd.DataFrame({"Kolom": miss.index, "Jumlah": miss.values,
                           "Label": [fmt_angka(v) for v in miss.values]})
        fig = px.bar(md, x="Jumlah", y="Kolom", orientation="h", text="Label",
                     title="Missing Values per Kolom",
                     color_discrete_sequence=["#F59E0B"])
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        vmax = float(md["Jumlah"].max())
        fig.update_xaxes(tickvals=ticks_ribuan(md["Jumlah"], maks=4)[0],
                         ticktext=ticks_ribuan(md["Jumlah"], maks=4)[1],
                         range=[0, vmax * 1.25])
        fig.update_traces(textposition="outside", textfont=dict(size=10),
                          hovertemplate="%{y}: %{x} sel<extra></extra>")
        gaya(fig, tinggi=390, mr=70, ml=130)
        st.plotly_chart(fig, use_container_width=True)
    with q2:
        masalah = pd.DataFrame({
            "Jenis": ["Duplikat Baris", "Sel Kosong", "Qty Tidak Valid",
                      "Harga Tidak Wajar", "Line_Total Tak Konsisten"],
            "Jumlah": [audit["duplikat_eksekak"], audit["missing_total"],
                       audit["quantity_invalid"], audit["harga_invalid"],
                       audit["line_total_tidak_konsisten"]],
        })
        masalah["Label"] = masalah["Jumlah"].map(fmt_angka)
        fig = px.bar(masalah, x="Jumlah", y="Jenis", orientation="h", text="Label",
                     title="Rekap Masalah Kualitas Data",
                     color_discrete_sequence=["#D97706"])
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        vmax = float(masalah["Jumlah"].max())
        fig.update_xaxes(tickvals=ticks_ribuan(masalah["Jumlah"], maks=4)[0],
                         ticktext=ticks_ribuan(masalah["Jumlah"], maks=4)[1],
                         range=[0, vmax * 1.25])
        fig.update_traces(textposition="outside", textfont=dict(size=10),
                          hovertemplate="%{y}: %{x} baris<extra></extra>")
        gaya(fig, tinggi=390, mr=70, ml=165)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 1e. Visualisasi Awal")
    v1, v2 = st.columns(2)
    bulanan = (clean.set_index("Transaction_Date")
               .resample("ME")["Line_Total"].sum().reset_index())
    bulanan["Bulan"] = bulanan["Transaction_Date"].dt.strftime("%b")
    with v1:
        fig = px.area(bulanan, x="Bulan", y="Line_Total",
                      title=f"Tren Penjualan Bulanan ({label_periode()})",
                      color_discrete_sequence=["#10B981"])
        fig.update_traces(line=dict(width=3),
                          hovertemplate="%{x}: Rp %{y:,.0f}<extra></extra>")
        tv, tt = ticks_ribuan(bulanan["Line_Total"], maks=4)
        fig.update_yaxes(tickvals=tv, ticktext=tt, title_text="Penjualan (Rp)")
        gaya(fig, tinggi=400, mb=70)
        st.plotly_chart(fig, use_container_width=True)
    with v2:
        top_produk = (clean.groupby("Product_Name")["Line_Total"].sum()
                      .sort_values(ascending=False).head(8).reset_index())
        top_produk["LabelNilai"] = top_produk["Line_Total"].map(fmt_angka)
        fig = px.bar(top_produk, x="Line_Total", y="Product_Name", orientation="h",
                     text="LabelNilai", title="8 Produk Penjualan Tertinggi",
                     labels={"Line_Total": "Penjualan (Rp)", "Product_Name": "Produk"},
                     color_discrete_sequence=["#6366F1"])
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        tv, tt = ticks_ribuan(top_produk["Line_Total"], maks=3)
        fig.update_xaxes(tickvals=tv, ticktext=tt,
                         range=[0, top_produk["Line_Total"].max() * 1.22])
        fig.update_traces(textposition="outside", textfont=dict(size=10),
                          hovertemplate="%{y}<br>Rp %{x:,.0f}<extra></extra>")
        gaya(fig, tinggi=400, mr=100, ml=110)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 1f. Temuan Awal (min. 3)")
    st.markdown(
        f"""
1. **Pola penjualan relatif stabil sepanjang {label_periode()}** dengan sedikit kenaikan menjelang akhir periode,
   menandakan kebutuhan promosi terjadwal pada periode tertentu.
2. **Konsentrasi produk** - beberapa produk berkontribusi besar pada pendapatan; produk harga
   lebih tinggi (kategori Sembako & Elektronik) mendominasi nilai, bukan volume transaksi.
3. **Masalah kualitas nyata & terukur** - ditemukan {fmt_angka(audit['duplikat_eksekak'])} duplikat,
   {fmt_angka(audit['missing_total'])} sel kosong, dan outlier ekstrem (Quantity 999, harga 999999)
   yang jika dibiarkan akan menaikkan nilai penjualan secara artifisial.
4. **Basket multi-item umum** - rata-rata tiap transaksi memuat sekitar 3 produk berbeda,
   peluang nyata untuk cross-selling via bundling.
        """
    )


# ---------------------------------------------------------------------------
# Halaman 02 - Preprocessing
# ---------------------------------------------------------------------------

def halaman_preprocessing():
    hero("02 - Data Preprocessing & Feature Engineering",
         "Pembersihan transparan, transformasi format, penanganan outlier, RFM & basket_size.")

    st.markdown("### 2a. Ringkasan Perubahan Data (Sebelum → Sesudah)")
    before = audit["jumlah_baris_awal"]
    after = len(clean)
    ringkasan = pd.DataFrame([
        {"Aspek": "Jumlah baris", "Sebelum": fmt_angka(before),
         "Sesudah": fmt_angka(after),
         "Aksi": f"Menghapus {fmt_angka(before - after)} baris (duplikat + invalid)"},
        {"Aspek": "Sel kosong", "Sebelum": fmt_angka(audit["missing_total"]),
         "Sesudah": "0",
         "Aksi": "Imputasi median/harga katalog + modus kategori"},
        {"Aspek": "Baris duplikat", "Sebelum": fmt_angka(audit["duplikat_eksekak"]),
         "Sesudah": "0", "Aksi": "Drop duplikat persis setelah normalisasi teks"},
        {"Aspek": "Qty tidak valid", "Sebelum": fmt_angka(audit["quantity_invalid"]),
         "Sesudah": "0", "Aksi": "Nilai <=0 atau >=100 diganti median qty"},
        {"Aspek": "Harga tidak wajar", "Sebelum": fmt_angka(audit["harga_invalid"]),
         "Sesudah": "0", "Aksi": "Diganti harga referensi katalog produk"},
        {"Aspek": "Inkonsistensi teks", "Sebelum": "Kategori & pembayaran campur",
         "Sesudah": "Seragam (Title Case + QRIS)",
         "Aksi": "Normalisasi huruf/spasi & mapping kategori ke katalog"},
        {"Aspek": "Line_Total", "Sebelum": "16 baris tidak konsisten",
         "Sesudah": "Konsisten", "Aksi": "Dihitung ulang Qty x Harga x (1 - Diskon)"},
    ])
    st.markdown(tabel_html(ringkasan, num_cols=["Sebelum", "Sesudah"],
                           wrap_cols=["Aksi"]), unsafe_allow_html=True)

    st.markdown("### 2b. Alasan Metode Penanganan")
    st.markdown(
        """
- **Missing Customer_ID** → diisi `UNKNOWN` (bukan dihapus) agar baris transaksi tetap
  berkontribusi pada analisis produk/Apriori; pelanggan tak dikenal dikeluarkan dari RFM.
- **Duplikat** → dihapus karena mewakili pencatatan ganda, bukan transaksi nyata.
- **Outlier Quantity/harga ekstrem** → diimputasi (bukan dibuang) karena tetap mewakili
  produk nyata; imputasi median/harga katalog menjaga distribusi tanpa memicu bias nilai.
- **Kategori/pembayaran campur huruf** → diseragamkan agar agregasi tidak terpecah.
- **Line_Total** → dihitung ulang agar konsisten secara finansial.
        """
    )

    st.markdown("### 2c. Fitur Engineering")
    bs = (clean.groupby("Transaction_ID")["Product_ID"].nunique()
          .rename("basket_size").reset_index())
    kartu_kpi([
        ("Rata-rata basket_size", fmt_des(bs["basket_size"].mean()) + " produk", "#0EA5E9"),
        ("Median basket_size", fmt_angka(bs["basket_size"].median()) + " produk", "#D97706"),
        ("Maks basket_size", fmt_angka(bs["basket_size"].max()) + " produk", "#6366F1"),
        ("Pelanggan ter-RFM", fmt_angka(len(rfm)), "#F59E0B"),
    ])

    bs_cnt = bs["basket_size"].value_counts().sort_index().reset_index()
    bs_cnt.columns = ["basket_size", "Jumlah"]
    bs_cnt["Label"] = bs_cnt["Jumlah"].map(fmt_angka)
    fig = px.bar(bs_cnt, x="basket_size", y="Jumlah", text="Label",
                 title="Distribusi basket_size per Transaksi",
                 color_discrete_sequence=["#6366F1"])
    fig.update_traces(textposition="outside", textfont=dict(size=10),
                      hovertemplate="basket_size %{x}: %{y:,.0f} transaksi<extra></extra>")
    fig.update_xaxes(dtick=1)
    tv, tt = ticks_ribuan(bs_cnt["Jumlah"], maks=4)
    fig.update_yaxes(tickvals=tv, ticktext=tt,
                     range=[0, float(bs_cnt["Jumlah"].max()) * 1.18])
    gaya(fig, tinggi=370, mt=64)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Recency · Frequency · Monetary (per Pelanggan)")
    st.caption(f"**Tanggal acuan** = {tgl_acuan:%d %b %Y} (satu hari setelah transaksi "
               f"maksimum, sesuai rekomendasi Info_Dataset).")
    rfm_show = rfm.head(15).copy()
    rfm_tabel = pd.DataFrame({
        "Customer_ID": rfm_show["Customer_ID"],
        "Recency (hari)": rfm_show["Recency"].map(fmt_angka),
        "Frequency": rfm_show["Frequency"].map(fmt_angka),
        "Monetary": rfm_show["Monetary"].map(fmt_rupiah),
    })
    st.markdown(tabel_html(rfm_tabel, num_cols=["Recency (hari)", "Frequency",
                                                "Monetary"]),
                unsafe_allow_html=True)

    st.markdown("#### Matriks Transaksi × Produk (untuk Apriori)")
    st.caption(f"Bentuk one-hot: {fmt_angka(matriks.shape[0])} transaksi x "
               f"{fmt_angka(matriks.shape[1])} produk. "
               f"Nilai 1 = produk dibeli dalam transaksi tersebut.")
    st.dataframe(matriks.iloc[:8, :10], use_container_width=True, hide_index=True)

    st.markdown("### 2d. Strategi untuk Data Besar")
    st.markdown(
        f"""
- **Sampling stratified** - untuk eksplorasi cepat, ambil sampel berimbang per bulan/kategori
  agar pola musiman tidak terdistorsi.
- **Chunking** - baca & proses file dalam potongan (misal 50000 baris) untuk memori terbatas,
  lalu agregasikan hasil tiap chunk.
- **Agregasi awal** - untuk Apriori/K-Means, data langsung diagregat ke level transaksi-item
  dan pelanggan-RFM sehingga volume turun drastis ({fmt_angka(18513)} baris menjadi {fmt_angka(5998)} basket
  dan {fmt_angka(1480)} pelanggan).
        """
    )


# ---------------------------------------------------------------------------
# Halaman 03 - Apriori
# ---------------------------------------------------------------------------

def halaman_apriori():
    hero("03 - Apriori / Market Basket Analysis",
         "Menemukan kombinasi produk yang sering dibeli bersama beserta metrik aturan asosiasi.")

    st.markdown("### 3a. Parameter Minimum Support & Confidence")
    st.caption(
        "Support maksimum pasangan produk pada data ini hanya sekitar 4% (banyak produk, "
        "transaksi menyebar), sehingga ambang 5% akan menghasilkan nol aturan. Support 2% "
        "dipilih agar pola signifikan tertangkap tanpa aturan sampah; confidence 20% dan "
        "lift > 1 menjaga relevansi serta menghindari aturan kebetulan.")
    csup, cconf, clift = st.columns(3)
    min_sup = csup.slider("Minimum Support", 0.005, 0.05, 0.02, 0.005, format="%.3f")
    min_conf = cconf.slider("Minimum Confidence", 0.1, 0.6, 0.2, 0.05, format="%.2f")
    min_lift = clift.slider("Minimum Lift", 1.0, 3.0, 1.05, 0.05, format="%.2f")

    res = ae.jalankan_apriori(matriks, min_sup, min_conf, min_lift)
    freq = res["frequent_itemsets"]
    rules = res["rules"].copy()

    kartu_kpi([
        ("Frequent Itemsets", str(len(freq)), "#0EA5E9"),
        ("Aturan Asosiasi", str(len(rules)), "#F59E0B"),
        ("Basis Transaksi", fmt_angka(res["n_transaksi"]), "#D97706"),
    ])

    if rules.empty:
        st.warning("Tidak ada aturan yang memenuhi ambang. Turunkan minimum support/confidence.")
        return

    rules["Aturan"] = rules.apply(
        lambda r: f"{label_item(r['antecedents'])} → {label_item(r['consequents'])}", axis=1)

    st.markdown("### 3b. Frequent Itemsets Teratas")
    fi = freq.head(12).copy()
    fi_tabel = pd.DataFrame({
        "Itemset": fi["items"].apply(label_item),
        "Ukuran": fi["length"].map(str),
        "Support": fi["support"].map(lambda v: fmt_pct(v * 100)),
    })
    st.markdown(tabel_html(fi_tabel, num_cols=["Ukuran", "Support"],
                           wrap_cols=["Itemset"]), unsafe_allow_html=True)

    st.markdown("### 3c. Aturan Asosiasi Terbaik (maks. 10)")
    st.caption("Peringkat berbasis skor gabungan = 0,5 x lift + 0,3 x confidence + "
               "0,2 x support; lift > 1 berarti pembelian produk Y meningkat ketika X ikut dibeli.")
    top10 = rules.head(10).copy()
    top10_tabel = pd.DataFrame({
        "Aturan (X → Y)": top10["Aturan"],
        "Support": top10["support"].map(lambda v: fmt_pct(v * 100)),
        "Confidence": top10["confidence"].map(lambda v: fmt_pct(v * 100, 1)),
        "Lift": top10["lift"].map(lambda v: fmt_des(v)),
        "Leverage": top10["leverage"].map(lambda v: fmt_des(v, 4)),
    })
    st.markdown(tabel_html(top10_tabel, num_cols=["Support", "Confidence", "Lift",
                                                  "Leverage"],
                           wrap_cols=["Aturan (X → Y)"]), unsafe_allow_html=True)

    st.markdown("### 3d. Visualisasi Aturan")
    st.caption(
        "Cara baca: sumbu X = Support (seberapa sering kombinasi muncul), sumbu Y = "
        "Confidence (seberapa yakin aturan berlaku), ukuran & warna titik = Lift "
        "(makin besar/makin gelap = kaitan makin kuat). Nomor pada titik sesuai "
        "daftar pada kotak sorot di bawah; arahkan kursor untuk detail lengkap.")

    d = top10.copy().reset_index(drop=True)
    d["sx"] = d["support"] * 100
    d["cy"] = d["confidence"] * 100
    d["no"] = [str(i + 1) for i in range(len(d))]
    d["info"] = d.apply(
        lambda r: (f"<b>{r['Aturan']}</b><br>"
                   f"Support {fmt_pct(r['support'] * 100)}<br>"
                   f"Confidence {fmt_pct(r['confidence'] * 100, 1)}<br>"
                   f"Lift {fmt_des(r['lift'])}"), axis=1)

    fig = px.scatter(d, x="sx", y="cy", size="lift", color="lift",
                     text="no", custom_data=["info"],
                     color_continuous_scale=["#10B981", "#D97706"],
                     size_max=30,
                     title="Peta Aturan: Support x Confidence (ukuran & warna = Lift)")
    fig.update_traces(
        textposition="middle center",
        textfont=dict(size=11, color="#ffffff", family=FONT_SANS),
        hovertemplate="%{customdata[0]}<extra></extra>",
        marker=dict(line=dict(width=1.5, color="#ffffff")))
    fig.update_xaxes(title_text="Support (%)", tickformat=",.2f", ticksuffix="%")
    fig.update_yaxes(title_text="Confidence (%)", tickformat=",.0f", ticksuffix="%")
    fig.update_layout(coloraxis_colorbar=dict(title="Lift"))
    gaya(fig, tinggi=470, mr=40)

    opsi = ["- Semua aturan -"] + [
        f"{i}. {pendek(r['Aturan'], 46)}" for i, r in d.iterrows()]
    pilih = st.selectbox("Sorot aturan pada peta", opsi)
    if pilih != opsi[0]:
        i = int(pilih.split(".", 1)[0]) - 1
        row = d.iloc[i]
        fig.add_trace(go.Scatter(
            x=[row["sx"]], y=[row["cy"]], mode="markers+text",
            marker=dict(size=30, symbol="circle-open",
                        line=dict(width=3, color="#F59E0B")),
            text=[row["no"]], textposition="middle center",
            textfont=dict(size=11, color="#F59E0B"),
            name="Sorotan", hoverinfo="skip", showlegend=False))
        dy = -78 if row["cy"] >= float(np.median(d["cy"])) else 78
        dx = 70 if row["sx"] <= float(np.median(d["sx"])) else -70
        teks_sorot = pilih.split(".", 1)[1].strip()
        fig.add_annotation(
            x=row["sx"], y=row["cy"], text=teks_sorot,
            showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=1.6,
            arrowcolor="#F59E0B", ax=dx, ay=dy,
            font=dict(size=11, color="#0F172A", family=FONT_SANS),
            bgcolor="rgba(255,255,255,.95)", bordercolor="#F59E0B",
            borderwidth=1, borderpad=5, align="left")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 3e. Heatmap Lift Antar Produk")
    idx_lbl = list(dict.fromkeys(
        pendek(label_item(r["antecedents"]), 24) for _, r in top10.iterrows()))
    col_lbl = list(dict.fromkeys(
        pendek(label_item(r["consequents"]), 24) for _, r in top10.iterrows()))
    z = np.full((len(idx_lbl), len(col_lbl)), np.nan)
    teks = np.full((len(idx_lbl), len(col_lbl)), "", dtype=object)
    for _, r in top10.iterrows():
        i = idx_lbl.index(pendek(label_item(r["antecedents"]), 24))
        j = col_lbl.index(pendek(label_item(r["consequents"]), 24))
        z[i, j] = r["lift"]
        teks[i, j] = fmt_des(r["lift"])
    hm = go.Figure(go.Heatmap(
        z=z, x=col_lbl, y=idx_lbl, text=teks, texttemplate="{text}",
        colorscale=[[0, "#F0F9FF"], [0.5, "#FDE68A"], [1, "#D97706"]],
        colorbar=dict(title="Lift"), hovertemplate=
        "%{y} → %{x}<br>Lift %{z:.2f}<extra></extra>"))
    hm.update_layout(title="Lift Antar Produk (Aturan Teratas)")
    gaya(hm, tinggi=max(420, 46 * len(idx_lbl)))
    hm.update_yaxes(autorange="reversed")
    st.plotly_chart(hm, use_container_width=True)

    st.session_state["rules"] = rules
    st.session_state["freq"] = freq


# ---------------------------------------------------------------------------
# Halaman 04 - K-Means
# ---------------------------------------------------------------------------

def halaman_kmeans():
    hero("04 - Segmentasi Pelanggan (K-Means)",
         "Segmentasi berbasis fitur RFM dengan evaluasi Elbow & Silhouette.")

    st.markdown("### 4a. Fitur yang Digunakan")
    st.markdown(
        """
- **Recency** - hari sejak transaksi terakhir (makin kecil = makin aktif baru-baru ini).
- **Frequency** - jumlah transaksi unik (loyalitas).
- **Monetary** - total belanja bersih (nilai pelanggan).

Fitur distandardisasi (StandardScaler: mean 0, std 1) karena K-Means berbasis jarak Euclidean;
tanpa scaling, Monetary (ratusan ribu) akan mendominasi Recency (ratusan hari).
        """
    )

    st.markdown("### 4b. Elbow Method & Silhouette Score")
    Xs, ev, labels, prof = jalankan_kmeans_cached("rfm", 3)
    e1, e2 = st.columns(2)
    ev_i = ev.copy()
    ev_i["inertia_lbl"] = ev_i["inertia"].map(fmt_angka)
    ev_i["sil_lbl"] = ev_i["silhouette"].map(lambda v: fmt_des(v, 3))
    with e1:
        fig = px.line(ev_i, x="k", y="inertia", text="inertia_lbl", markers=True,
                      title="Elbow Method (Inertia)", color_discrete_sequence=["#10B981"])
        fig.update_traces(mode="lines+markers+text", line=dict(width=3),
                          textposition="top center", textfont=dict(size=10),
                          hovertemplate="k=%{x}<br>Inertia %{y:,.0f}<extra></extra>")
        tv, tt = ticks_ribuan(ev_i["inertia"], maks=4)
        fig.update_xaxes(dtick=1, title_text="Jumlah klaster (k)")
        fig.update_yaxes(tickvals=tv, ticktext=tt)
        gaya(fig, tinggi=390, mt=74)
        st.plotly_chart(fig, use_container_width=True)
    with e2:
        fig = px.line(ev_i, x="k", y="silhouette", text="sil_lbl", markers=True,
                      title="Silhouette Score", color_discrete_sequence=["#D97706"])
        fig.update_traces(mode="lines+markers+text", line=dict(width=3),
                          textposition="top center", textfont=dict(size=10),
                          hovertemplate="k=%{x}<br>Skor %{y:.3f}<extra></extra>")
        fig.update_xaxes(dtick=1, title_text="Jumlah klaster (k)")
        fig.update_yaxes(tickformat=",.3f")
        gaya(fig, tinggi=390, mt=74)
        st.plotly_chart(fig, use_container_width=True)

    ev_tabel = pd.DataFrame({
        "k": ev["k"].map(str),
        "Inertia": ev["inertia"].map(fmt_angka),
        "Silhouette": ev["silhouette"].map(lambda v: fmt_des(v, 3)),
        "Iterasi": ev["n_iter"].map(str),
    })
    st.markdown(tabel_html(ev_tabel, num_cols=["k", "Inertia", "Silhouette", "Iterasi"]),
                unsafe_allow_html=True)

    best_k = int(ev.sort_values("silhouette", ascending=False).iloc[0]["k"])
    skor_best = float(ev.sort_values("silhouette", ascending=False).iloc[0]["silhouette"])
    st.info(f"**k terbaik menurut Silhouette = {best_k}** (skor {fmt_des(skor_best, 3)}). "
            f"Elbow menunjukkan tekukan inertia sekitar k={best_k}-4; dipilih k={best_k} "
            f"karena seimbang antara kepadatan klaster dan kemudahan interpretasi bisnis.")

    st.markdown("### 4c. Jumlah Klaster Final")
    k_final = st.select_slider("Jumlah klaster (k)", options=list(range(2, 9)),
                               value=best_k)
    Xs, ev2, labels, prof = jalankan_kmeans_cached("rfm", int(k_final))

    st.markdown("### 4d. Profil Klaster")
    prof_tabel = pd.DataFrame({
        "Klaster": prof["Cluster"].map(str),
        "Label Segmen": prof["label_segmen"],
        "Jumlah Pelanggan": prof["jumlah_pelanggan"].map(fmt_angka),
        "Recency Rata-rata (hari)": prof["recency_mean"].map(lambda v: fmt_des(v, 1)),
        "Frequency Rata-rata": prof["frequency_mean"].map(lambda v: fmt_des(v, 1)),
        "Monetary Rata-rata": prof["monetary_mean"].map(fmt_rupiah),
        "Total Moneter": prof["monetary_total"].map(fmt_rupiah),
    })
    st.markdown(tabel_html(prof_tabel,
                           num_cols=["Klaster", "Jumlah Pelanggan",
                                     "Recency Rata-rata (hari)", "Frequency Rata-rata",
                                     "Monetary Rata-rata", "Total Moneter"],
                           wrap_cols=["Label Segmen"]), unsafe_allow_html=True)

    st.markdown("### 4e. Visualisasi Segmen")
    rfm_l = rfm.copy()
    rfm_l["Cluster"] = labels
    rfm_l["Label"] = prof.set_index("Cluster")["label_segmen"].reindex(labels).values

    v1, v2 = st.columns(2)
    tv_r, tt_r = ticks_ribuan(rfm_l["Monetary"], maks=4)
    with v1:
        fig = px.scatter(rfm_l, x="Recency", y="Monetary", color="Label",
                         size="Frequency", hover_data=["Customer_ID", "Frequency"],
                         title="Segmen: Recency x Monetary (ukuran = Frequency)",
                         color_discrete_sequence=PALETTE)
        fig.update_yaxes(tickvals=tv_r, ticktext=tt_r)
        fig.update_traces(hovertemplate=
                          "%{customdata[0]}<br>Recency %{x:,.0f} hari<br>"
                          "Rp %{y:,.0f}<extra></extra>")
        gaya(fig, tinggi=460, mr=80)
        st.plotly_chart(fig, use_container_width=True)
    with v2:
        fig = px.box(rfm_l, x="Label", y="Monetary", color="Label",
                     title="Distribusi Monetary per Segmen",
                     color_discrete_sequence=PALETTE)
        fig.update_yaxes(tickvals=tv_r, ticktext=tt_r)
        fig.update_traces(hovertemplate="Rp %{y:,.0f}<extra></extra>")
        gaya(fig, tinggi=460)
        st.plotly_chart(fig, use_container_width=True)

    st.session_state["labels"] = labels
    st.session_state["prof"] = prof
    st.session_state["rfm_l"] = rfm_l


# ---------------------------------------------------------------------------
# Halaman 05 - Dashboard BI
# ---------------------------------------------------------------------------

def halaman_dashboard():
    hero("05 - Dashboard Business Intelligence",
         "Tampilan eksekutif dengan KPI, tren, hasil Apriori & K-Means, serta filter interaktif.",
         "#0EA5E9")

    st.markdown('<div class="fb-title-inline">Filter Analisis</div>',
                unsafe_allow_html=True)
    fa, fb, fc = st.columns(3)
    t_min = clean["Transaction_Date"].min().date()
    t_max = clean["Transaction_Date"].max().date()
    rentang = fa.slider("Periode transaksi", t_min, t_max, (t_min, t_max),
                        format="YYYY-MM-DD")
    kategori_list = sorted(clean["Category"].dropna().unique())
    kategori = fb.multiselect("Kategori produk", kategori_list, default=kategori_list)

    labels = st.session_state.get("labels", None)
    if labels is None:
        _, _, labels, _ = jalankan_kmeans_cached("rfm", 3)
    prof = st.session_state.get("prof", None)
    if prof is None:
        _, _, _, prof = jalankan_kmeans_cached("rfm", 3)
    seg_map = prof.set_index("Cluster")["label_segmen"].to_dict()
    cust_seg = {cid: seg_map.get(lab, str(lab))
                for cid, lab in zip(rfm["Customer_ID"], labels)}
    seg_options = ["Semua"] + sorted(set(cust_seg.values()))
    seg_pilih = fc.multiselect("Segmen pelanggan", seg_options, default=["Semua"])

    mask = ((clean["Transaction_Date"].dt.date >= rentang[0]) &
            (clean["Transaction_Date"].dt.date <= rentang[1]) &
            (clean["Category"].isin(kategori)))
    dff = clean[mask]
    if seg_pilih and "Semua" not in seg_pilih:
        cust_ok = {c for c, s in cust_seg.items() if s in seg_pilih}
        dff = dff[dff["Customer_ID"].isin(cust_ok)]

    st.markdown("### KPI Utama")
    bs_mean = dff.groupby("Transaction_ID")["Product_ID"].nunique().mean()
    kartu_kpi([
        ("Total Penjualan", fmt_rupiah(dff["Line_Total"].sum()), "#0EA5E9"),
        ("Jumlah Transaksi", fmt_angka(dff["Transaction_ID"].nunique()), "#D97706"),
        ("Pelanggan Unik", fmt_angka(dff.loc[dff.Customer_ID != "UNKNOWN",
                                             "Customer_ID"].nunique()), "#6366F1"),
        ("Rata-rata basket_size", (fmt_des(bs_mean) if not np.isnan(bs_mean)
                                   else "0") + " produk", "#F59E0B"),
    ])

    st.markdown("### Analisis Penjualan")
    a1, a2 = st.columns(2)
    bul = (dff.set_index("Transaction_Date").resample("ME")["Line_Total"].sum().reset_index())
    bul["Bulan"] = bul["Transaction_Date"].dt.strftime("%b")
    with a1:
        fig = px.line(bul, x="Bulan", y="Line_Total", markers=True,
                      title=f"Tren Penjualan Bulanan ({label_periode()})",
                      color_discrete_sequence=["#10B981"])
        fig.update_traces(line=dict(width=3),
                          hovertemplate="%{x}: Rp %{y:,.0f}<extra></extra>")
        tv, tt = ticks_ribuan(bul["Line_Total"], maks=4)
        fig.update_yaxes(tickvals=tv, ticktext=tt, title_text="Penjualan (Rp)")
        gaya(fig, tinggi=400, mb=70)
        st.plotly_chart(fig, use_container_width=True)
    with a2:
        tp = (dff.groupby("Product_Name")["Line_Total"].sum()
              .sort_values(ascending=False).head(8).reset_index())
        tp["LabelNilai"] = tp["Line_Total"].map(fmt_angka)
        fig = px.bar(tp, x="Line_Total", y="Product_Name", orientation="h",
                     text="LabelNilai", title="8 Produk Penjualan Tertinggi",
                     color_discrete_sequence=["#D97706"])
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        tv, tt = ticks_ribuan(tp["Line_Total"], maks=3)
        fig.update_xaxes(tickvals=tv, ticktext=tt,
                         range=[0, float(tp["Line_Total"].max()) * 1.22])
        fig.update_traces(textposition="outside", textfont=dict(size=10),
                          hovertemplate="%{y}<br>Rp %{x:,.0f}<extra></extra>")
        gaya(fig, tinggi=400, mr=100, ml=110)
        st.plotly_chart(fig, use_container_width=True)

    a3, a4 = st.columns(2)
    kat = (dff.groupby("Category")["Line_Total"].sum()
           .sort_values(ascending=False).reset_index())
    total_kat = kat["Line_Total"].sum()
    with a3:
        kat["Pct"] = [fmt_des(v / total_kat * 100, 1) + "%"
                      for v in kat["Line_Total"]]
        kat["Info"] = [f"{kat['Category'][i]}<br>Penjualan Rp {_rib(int(kat['Line_Total'][i]))} "
                       f"({kat['Pct'][i]})" for i in kat.index]
        fig = px.bar(kat.sort_values("Line_Total", ascending=True),
                     x="Line_Total", y="Category", orientation="h", text="Pct",
                     custom_data=["Info"],
                     title="Distribusi Penjualan per Kategori",
                     color_discrete_sequence=["#10B981"])
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        tv, tt = ticks_ribuan(kat["Line_Total"], maks=3)
        fig.update_xaxes(tickvals=tv, ticktext=tt,
                         range=[0, float(kat["Line_Total"].max()) * 1.30])
        fig.update_traces(textposition="outside", textfont=dict(size=10),
                          hovertemplate="%{customdata[0]}<extra></extra>")
        gaya(fig, tinggi=430, mr=70, ml=120)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Total penjualan terfilter: <b>{fmt_rupiah(total_kat)}</b> "
                   f"dari {fmt_angka(kat.shape[0])} kategori; persen = pangsa "
                   f"terhadap total.", unsafe_allow_html=True)
    with a4:
        pm = (dff.groupby("Payment_Method")["Line_Total"].sum()
              .sort_values(ascending=False).reset_index())
        pm["LabelNilai"] = pm["Line_Total"].map(fmt_angka)
        fig = px.bar(pm, x="Line_Total", y="Payment_Method", orientation="h",
                     text="LabelNilai", title="Penjualan per Metode Bayar",
                     color_discrete_sequence=["#6366F1"])
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        tv, tt = ticks_ribuan(pm["Line_Total"], maks=3)
        fig.update_xaxes(tickvals=tv, ticktext=tt,
                         range=[0, float(pm["Line_Total"].max()) * 1.32])
        fig.update_traces(textposition="outside", textfont=dict(size=10),
                          hovertemplate="%{y}<br>Rp %{x:,.0f}<extra></extra>")
        gaya(fig, tinggi=430, mr=115, ml=125)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Hasil Apriori")
    rules = st.session_state.get("rules", None)
    if rules is None or rules.empty:
        rules = ae.jalankan_apriori(matriks, 0.02, 0.2, 1.05)["rules"].copy()
    t10 = rules.head(10).copy()
    t10["Aturan"] = t10.apply(lambda r: f"{label_item(r['antecedents'])} → "
                                        f"{label_item(r['consequents'])}", axis=1)
    t10_tabel = pd.DataFrame({
        "Aturan (X → Y)": t10["Aturan"],
        "Support": t10["support"].map(lambda v: fmt_pct(v * 100)),
        "Confidence": t10["confidence"].map(lambda v: fmt_pct(v * 100, 1)),
        "Lift": t10["lift"].map(lambda v: fmt_des(v)),
    })
    st.markdown(tabel_html(t10_tabel, num_cols=["Support", "Confidence", "Lift"],
                           wrap_cols=["Aturan (X → Y)"]), unsafe_allow_html=True)

    st.markdown("### Hasil K-Means")
    m1, m2 = st.columns(2)
    with m1:
        p1 = prof.copy().sort_values("jumlah_pelanggan", ascending=True)
        p1["LabelNilai"] = p1["jumlah_pelanggan"].map(fmt_angka)
        fig = px.bar(p1, x="jumlah_pelanggan", y="label_segmen", orientation="h",
                     text="LabelNilai", title="Jumlah Pelanggan per Segmen",
                     color_discrete_sequence=["#0EA5E9"])
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        tv, tt = ticks_ribuan(p1["jumlah_pelanggan"], maks=4)
        fig.update_xaxes(tickvals=tv, ticktext=tt,
                         range=[0, float(p1["jumlah_pelanggan"].max()) * 1.28])
        fig.update_traces(textposition="outside", textfont=dict(size=10),
                          hovertemplate="%{y}: %{x:,.0f} pelanggan<extra></extra>")
        gaya(fig, tinggi=390, mr=85, ml=135)
        st.plotly_chart(fig, use_container_width=True)
    with m2:
        p2 = prof.copy().sort_values("monetary_mean", ascending=True)
        p2["LabelNilai"] = p2["monetary_mean"].map(fmt_angka)
        fig = px.bar(p2, x="monetary_mean", y="label_segmen", orientation="h",
                     text="LabelNilai", title="Rata-rata Monetary per Segmen (Rp)",
                     color_discrete_sequence=["#D97706"])
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        tv, tt = ticks_ribuan(p2["monetary_mean"], maks=3)
        fig.update_xaxes(tickvals=tv, ticktext=tt,
                         range=[0, float(p2["monetary_mean"].max()) * 1.34])
        fig.update_traces(textposition="outside", textfont=dict(size=10),
                          hovertemplate="%{y}<br>Rp %{x:,.0f}<extra></extra>")
        gaya(fig, tinggi=390, mr=105, ml=135)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Bonus · Pola Produk per Segmen")
    st.markdown(
        """
Bagian ini **menggabungkan hasil Apriori x K-Means**: menampilkan seberapa kuat satu
produk "dimiliki" oleh satu segmen pelanggan - dasar untuk menyesuaikan promosi per segmen.

**Cara membaca heatmap di bawah:**
- **Baris** = enam produk terlaris keseluruhan; **kolom** = segmen pelanggan
  (Loyal, Potential Loyal, Churn Risk), diurutkan dari segmen terbesar.
- **Warna & angka sel** = pangsa penjualan produk tersebut *di dalam segmen* (%);
  makin gelap makin dominan.
- **Arahkan kursor** ke sel untuk melihat nilai rupiah absolutnya.
- **Filter periode, kategori, dan segmen** di bagian atas dashboard dipakai ulang
  di sini, sehingga peta bisa dijelajah sesuai kebutuhan.
        """
    )
    if labels is not None and not dff.empty:
        cust_seg2 = dict(zip(rfm["Customer_ID"], labels))
        dseg = dff[dff["Customer_ID"].isin(cust_seg2)].copy()
        dseg["Segmen"] = dseg["Customer_ID"].map(cust_seg2)
        seg_map2 = prof.set_index("Cluster")["label_segmen"].to_dict()
        dseg["NamaSegmen"] = dseg["Segmen"].map(seg_map2)
        produk_seg = (dseg.groupby(["NamaSegmen", "Product_Name"])["Line_Total"].sum()
                      .reset_index())
        pivot = produk_seg.pivot_table(index="Product_Name", columns="NamaSegmen",
                                        values="Line_Total", aggfunc="sum",
                                        fill_value=0)
        urut_seg = [s for s in (prof.sort_values("jumlah_pelanggan",
                                                  ascending=False)["label_segmen"]
                                .tolist()) if s in pivot.columns]
        pivot = pivot.reindex(columns=urut_seg)
        if pivot.empty or not urut_seg:
            st.info("Tidak ada data produk/segmen untuk filter saat ini - longgarkan "
                    "filter di atas untuk melihat pola ini.")
        else:
            top_produk = (pivot.sum(axis=1).sort_values(ascending=False)
                          .head(6).index.tolist())
            pivot = pivot.reindex(top_produk)
            tot_kolom = pivot.sum(axis=0).replace(0, np.nan)
            share = (pivot.div(tot_kolom, axis=1) * 100).fillna(0)

            teks_pct = [[fmt_des(v, 1) + "%" for v in row] for row in share.values]
            rp_cell = [[_rib(int(v)) for v in row] for row in pivot.values]

            fig = go.Figure(go.Heatmap(
                z=share.values, x=urut_seg, y=top_produk,
                text=teks_pct, texttemplate="{text}", textfont=dict(size=11),
                customdata=rp_cell,
                colorscale=[[0, "#F0F9FF"], [0.55, "#7DD3FC"], [1, "#0EA5E9"]],
                colorbar=dict(title="% segmen"),
                hovertemplate=("%{y}<br>%{x}<br>Penjualan Rp %{customdata[0]}"
                               "<br>Pangsa %{z:,.1f}%<extra></extra>")))
            fig.update_layout(title="Pangsa Produk dalam Belanja Tiap Segmen (%)")
            fig.update_yaxes(autorange="reversed")
            gaya(fig, tinggi=max(430, 46 * len(top_produk) + 205),
                 ml=130, mr=40, mb=74)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Insight per Segmen")
            AKSI_SEG = {
                "Loyal Customers": ("Pertahankan: reward, early access, dan jaga "
                                    "ketersediaan produk andalan."),
                "Potential Loyal": ("Kembangkan: bundling & upsell ke produk "
                                    "ber-lift tinggi (lihat peta aturan Apriori)."),
                "Churn Risk": ("Menangkan kembali: diskon personal, pengingat, dan "
                               "penawaran khusus."),
                "New Customers": ("Onboarding: cross-sell produk populer sejak "
                                  "transaksi awal."),
            }
            kartu = []
            for i, nama_seg in enumerate(urut_seg):
                sub = produk_seg[produk_seg["NamaSegmen"] == nama_seg].sort_values(
                    "Line_Total", ascending=False)
                if sub.empty:
                    continue
                tot_seg = float(sub["Line_Total"].sum())
                r1 = sub.iloc[0]
                p1 = float(r1["Line_Total"]) / tot_seg * 100 if tot_seg else 0.0
                andalan = (f"Produk andalan: <b>{r1['Product_Name']}</b> "
                           f"({fmt_des(p1, 1)}% dari belanja segmen)")
                if len(sub) > 1:
                    r2 = sub.iloc[1]
                    p2 = float(r2["Line_Total"]) / tot_seg * 100 if tot_seg else 0.0
                    andalan += (f", disusul <b>{r2['Product_Name']}</b> "
                                f"({fmt_des(p2, 1)}%)")
                andalan += "."
                warna = PALETTE[i % len(PALETTE)]
                aksi = AKSI_SEG.get(nama_seg,
                                    "Kembangkan potensi lewat promosi bertarget.")
                kartu.append(
                    f'<div class="kpi" style="--k:{warna}">'
                    f'<div class="v" style="font-size:1.02rem">{nama_seg}</div>'
                    f'<div class="l" style="text-transform:none;letter-spacing:.01em;'
                    f'line-height:1.6;margin-top:.4rem">{andalan} '
                    f"Total belanja segmen <b>{fmt_rupiah(tot_seg)}</b>. "
                    f"{aksi}</div></div>")
            if kartu:
                st.markdown(f'<div class="kpi-grid">{"".join(kartu)}</div>',
                            unsafe_allow_html=True)
    else:
        st.info("Tidak ada data pelanggan untuk filter saat ini - longgarkan filter "
                "di atas untuk melihat pola produk per segmen.")


# ---------------------------------------------------------------------------
# Halaman 06 - Rekomendasi
# ---------------------------------------------------------------------------

def halaman_rekomendasi():
    hero("06 - Rekomendasi Keputusan Bisnis",
         "Strategi preskriptif berbasis bukti data, prioritas, estimasi dampak & evaluasi.")

    rules = st.session_state.get("rules", None)
    if rules is None or rules.empty:
        rules = ae.jalankan_apriori(matriks, 0.02, 0.2, 1.05)["rules"].copy()
    prof = st.session_state.get("prof", None)
    if prof is None:
        _, _, _, prof = jalankan_kmeans_cached("rfm", 3)

    st.markdown("### 6a. Rekomendasi Berdasarkan Apriori (Bundling & Penempatan Rak)")
    rec_rows = []
    for i, (_, r) in enumerate(rules.head(5).iterrows(), 1):
        a = label_item(r["antecedents"])
        c = label_item(r["consequents"])
        rec_rows.append({
            "No": str(i),
            "Kombinasi Produk": f"{a} + {c}",
            "Bukti Data": (f"lift {fmt_des(r['lift'])}, "
                           f"confidence {fmt_pct(r['confidence'] * 100, 1)}, "
                           f"support {fmt_pct(r['support'] * 100)}"),
            "Rekomendasi": f"Tempatkan {a} dan {c} berdekatan / tawarkan bundling hemat.",
        })
    st.markdown(tabel_html(pd.DataFrame(rec_rows), num_cols=["No"],
                           wrap_cols=["Kombinasi Produk", "Bukti Data", "Rekomendasi"]),
                unsafe_allow_html=True)

    st.markdown("### 6b. Rekomendasi Retensi Berdasarkan Segmentasi")
    seg_rows = []
    for _, r in prof.iterrows():
        lab = r["label_segmen"]
        if lab == "Loyal Customers":
            strat = "Program loyalitas, early-access promo, referral reward."
            prior = "Pertahankan (nilai tinggi)"
        elif lab == "Churn Risk":
            strat = "Win-back campaign, diskon personal, penawaran kembali."
            prior = "Tinggi - cegah churn"
        elif lab == "New Customers":
            strat = "Onboarding offer, cross-sell produk populer, edukasi katalog."
            prior = "Bangun kebiasaan belanja"
        else:
            strat = "Upsell bundling & pengingat promo berkala."
            prior = "Kembangkan potensi"
        seg_rows.append({
            "Segmen": lab,
            "Jumlah Pelanggan": fmt_angka(r["jumlah_pelanggan"]),
            "Karakteristik": (f"Recency {fmt_des(r['recency_mean'], 0)} hari, "
                              f"Frequency {fmt_des(r['frequency_mean'], 1)}, "
                              f"Monetary {fmt_rupiah(r['monetary_mean'])}"),
            "Strategi": strat,
            "Prioritas": prior,
        })
    st.markdown(tabel_html(pd.DataFrame(seg_rows),
                           num_cols=["Jumlah Pelanggan"],
                           wrap_cols=["Karakteristik", "Strategi", "Prioritas"]),
                unsafe_allow_html=True)

    st.markdown("### 6c. Prioritas Tindakan (Urgensi × Manfaat × Biaya × Kemudahan)")
    prioritas = pd.DataFrame([
        {"Prioritas": "1", "Tindakan": "Bundling & penempatan rak (produk lift tertinggi)",
         "Urgensi": "Tinggi", "Potensi Manfaat": "Sedang-Tinggi", "Biaya": "Rendah",
         "Kemudahan": "Mudah",
         "Alasan": "Memanfaatkan pola belanja nyata; hanya butuh penataan rak & label promo."},
        {"Prioritas": "2", "Tindakan": "Win-back campaign untuk segmen Churn Risk",
         "Urgensi": "Tinggi", "Potensi Manfaat": "Tinggi", "Biaya": "Sedang",
         "Kemudahan": "Sedang",
         "Alasan": "Segmen recency besar & frekuensi rendah berisiko hilang namun masih bisa diaktifkan."},
        {"Prioritas": "3", "Tindakan": "Program loyalitas untuk Loyal Customers",
         "Urgensi": "Sedang", "Potensi Manfaat": "Tinggi", "Biaya": "Sedang",
         "Kemudahan": "Sedang",
         "Alasan": "Melindungi segmen dengan kontribusi moneter terbesar."},
        {"Prioritas": "4", "Tindakan": "Cross-sell onboarding untuk New Customers",
         "Urgensi": "Sedang", "Potensi Manfaat": "Sedang", "Biaya": "Rendah",
         "Kemudahan": "Mudah",
         "Alasan": "Membangun kebiasaan belanja sejak awal hubungan."},
    ])
    st.markdown(tabel_html(prioritas, num_cols=["Prioritas"],
                           wrap_cols=["Tindakan", "Alasan", "Potensi Manfaat"]),
                unsafe_allow_html=True)

    st.markdown("### 6d. Estimasi Dampak, Asumsi & Cara Ukur")
    st.markdown(
        """
**Batas kejujuran analitis:** bagian berikut adalah **estimasi berbasis asumsi**, bukan fakta
terukur. Dampak aktual hanya diketahui setelah implementasi dan pengukuran lanjutan.

- **Asumsi:** kenaikan cross-sell 5-10% pada produk ber-lift tinggi bila bundling & rak
  diterapkan; recovery 15-20% pelanggan Churn Risk dari win-back campaign.
- **Indikator keberhasilan:** (1) rata-rata basket_size naik; (2) penjualan produk ber-lift
  tinggi meningkat; (3) jumlah pelanggan aktif pada segmen Churn Risk pulih; (4) pendapatan
  per pelanggan Loyal naik.
- **Cara mengukur:** bandingkan KPI pra vs pasca implementasi pada periode sama (misal 3 bulan),
  dengan kelompok kontrol toko bila memungkinkan; pantau bulanan melalui dashboard.
        """
    )

    st.markdown("### 6e. Rencana Implementasi")
    st.markdown(
        """
**Jangka pendek (0-1 bulan):**
- Penataan rak & display bundling untuk kombinasi lift tertinggi.
- Susun & jalankan promo win-back untuk pelanggan Churn Risk.

**Jangka menengah (1-3 bulan):**
- Luncurkan program loyalitas berjenjang berbasis Monetary.
- Otomatiskan rekomendasi produk per segmen di kasir/kanal pemasaran.
- Evaluasi bulanan KPI dashboard dan iterasi strategi.
        """
    )

    st.markdown("### 6f. KPI Evaluasi Keberhasilan")
    kpi_tabel = pd.DataFrame([
        {"KPI": "Rata-rata basket_size", "Baseline (data)": "3,05 produk",
         "Target": "Naik 5-10%", "Cara Ukur": "Rerata item unik/transaksi bulanan"},
        {"KPI": "Penjualan produk ber-lift tinggi", "Baseline (data)": "Lihat dashboard",
         "Target": "Naik 8-15%", "Cara Ukur": "Total penjualan produk tersebut/bulan"},
        {"KPI": "Pelanggan Churn Risk aktif kembali", "Baseline (data)": "Lihat dashboard",
         "Target": "Pulih 15-20%", "Cara Ukur": "Jumlah transaksi ulang segmen tersebut"},
        {"KPI": "Pendapatan per pelanggan Loyal", "Baseline (data)": "Lihat dashboard",
         "Target": "Naik 5-10%", "Cara Ukur": "Monetary rata-rata segmen loyal"},
    ])
    st.markdown(tabel_html(kpi_tabel, wrap_cols=["Cara Ukur", "Baseline (data)"]),
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

ROUTER = {
    "00 · Unggah Data": halaman_unggah,
    "01 · Ringkasan": halaman_ringkasan,
    "02 · EDA & Pemahaman": halaman_eda,
    "03 · Preprocessing": halaman_preprocessing,
    "04 · Apriori": halaman_apriori,
    "05 · K-Means": halaman_kmeans,
    "06 · Dashboard": halaman_dashboard,
    "07 · Rekomendasi": halaman_rekomendasi,
}

# Tanpa data hasil unggahan, seluruh halaman analisis dikunci dan pengguna
# diarahkan ke halaman Unggah Data.
if not data_siap and halaman != NAV_UNGAH:
    halaman = NAV_UNGAH

ROUTER[halaman]()
