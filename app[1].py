
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.graph_objects as go

st.set_page_config(page_title="Bottom Setup Scanner", page_icon="📉", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(ttl=300)
def get_prices(symbol, period="6mo", interval="1d"):
    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    needed = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in needed if c in df.columns]].copy()
    df = df.dropna(subset=["Close"])
    return df


def indicators(df):
    x = df.copy()

    # Moving averages
    x["MA20"] = x["Close"].rolling(20).mean()
    x["MA30"] = x["Close"].rolling(30).mean()

    # RSI(14)
    delta = x["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    x["RSI14"] = 100 - (100 / (1 + rs))

    # MACD(12,26,9)
    ema12 = x["Close"].ewm(span=12, adjust=False).mean()
    ema26 = x["Close"].ewm(span=26, adjust=False).mean()
    x["MACD"] = ema12 - ema26
    x["MACD_Signal"] = x["MACD"].ewm(span=9, adjust=False).mean()
    x["MACD_Hist"] = x["MACD"] - x["MACD_Signal"]

    # Volume
    x["VolAvg20"] = x["Volume"].rolling(20).mean()
    x["RVOL20"] = x["Volume"] / x["VolAvg20"]

    # 20-day bottom / support
    x["Low20"] = x["Low"].rolling(20).min()
    x["High20"] = x["High"].rolling(20).max()
    x["DistanceFromLow20Pct"] = (x["Close"] / x["Low20"] - 1) * 100
    x["Range20Pct"] = (x["High20"] / x["Low20"] - 1) * 100

    # Price change over the lookback
    x["Return20Pct"] = x["Close"].pct_change(20) * 100

    # A simple "base" measure: recent 5-day range compared with prior 20-day range
    x["Range5Pct"] = (
        x["High"].rolling(5).max() / x["Low"].rolling(5).min() - 1
    ) * 100
    x["RangeCompression"] = x["Range5Pct"] / x["Range20Pct"]

    return x


def get_chart_exchange(symbol, api_key):
    """
    Optional ChartExchange integration.
    If no API key is supplied, the scanner still works with Yahoo Finance data.
    The exact API endpoints can vary by account/API version, so failures are
    handled gracefully rather than breaking the scanner.
    """
    if not api_key:
        return {}

    result = {}
    headers = {"Authorization": f"Bearer {api_key}", "X-API-KEY": api_key}

    # Try common ChartExchange API routes. If the account does not have access,
    # we simply leave the field unavailable.
    endpoints = {
        "short_interest": f"https://chartexchange.com/api/v1/data/stocks/short-interest/{symbol}/",
        "borrow_fee": f"https://chartexchange.com/api/v1/data/stocks/borrow-fee/ib/{symbol}/",
    }

    for key, url in endpoints.items():
        try:
            r = requests.get(url, headers=headers, timeout=8)
            if r.ok:
                result[key] = r.json()
        except Exception:
            pass

    return result


def last_value(series):
    s = series.dropna()
    return float(s.iloc[-1]) if len(s) else np.nan


def evaluate(df, cfg):
    x = indicators(df)
    if len(x) < 35:
        return None, x

    r = x.iloc[-1]

    checks = {
        "هبوط آخر 20 يوم": r["Return20Pct"] <= cfg["max_return20"],
        "السعر تحت MA20": r["Close"] < r["MA20"],
        "السعر تحت MA30": r["Close"] < r["MA30"],
        "RSI منخفض": r["RSI14"] <= cfg["max_rsi"],
        "MACD إيجابي": r["MACD"] > r["MACD_Signal"],
        "فوليوم منخفض": r["RVOL20"] <= cfg["max_rvol"],
        "قريب من قاع 20 يوم": r["DistanceFromLow20Pct"] <= cfg["max_distance_low"],
        "تثبيت/ضغط النطاق": r["RangeCompression"] <= cfg["max_range_compression"],
    }

    score = int(sum(checks.values()))
    return {
        "Ticker": "",
        "Close": r["Close"],
        "Return20%": r["Return20Pct"],
        "RSI14": r["RSI14"],
        "MACD": r["MACD"],
        "MACD Signal": r["MACD_Signal"],
        "RVOL20": r["RVOL20"],
        "MA20": r["MA20"],
        "MA30": r["MA30"],
        "Dist Low20%": r["DistanceFromLow20Pct"],
        "Range20%": r["Range20Pct"],
        "Range Compression": r["RangeCompression"],
        "Score": score,
        "Max Score": len(checks),
        "Checks": checks,
    }, x


# -----------------------------
# UI
# -----------------------------
st.title("📉 Bottom Setup Scanner")
st.caption(
    "سكانر شخصي لاستراتيجية: هبوط 20 يوم → قرب القاع → فوليوم منخفض → RSI منخفض → "
    "MACD يتحسن/إيجابي → السعر تحت MA20 وMA30."
)

with st.sidebar:
    st.header("الإعدادات")

    symbols_text = st.text_area(
        "الأسهم",
        value="FTFT,PSIG,WFF,SCNI,RCON",
        help="افصل الأسهم بفاصلة أو اكتب كل سهم في سطر.",
    )

    st.subheader("شروط الاستراتيجية")

    max_return20 = st.number_input(
        "أقصى تغير خلال 20 يوم (%)",
        value=-10.0,
        step=1.0,
        help="مثال -10 يعني السهم يجب أن يكون هابطاً 10% أو أكثر خلال 20 جلسة.",
    )

    max_rsi = st.slider(
        "RSI14 أقل من أو يساوي",
        min_value=10,
        max_value=70,
        value=50,
    )

    max_rvol = st.number_input(
        "RVOL20 أقصى قيمة",
        min_value=0.10,
        max_value=2.00,
        value=0.70,
        step=0.05,
        help="0.70 = حجم التداول الحالي أقل من 70% من متوسط 20 يوم.",
    )

    max_distance_low = st.number_input(
        "الابتعاد عن قاع 20 يوم (%)",
        min_value=0.5,
        max_value=30.0,
        value=8.0,
        step=0.5,
    )

    max_range_compression = st.number_input(
        "ضغط النطاق الأقصى",
        min_value=0.1,
        max_value=1.0,
        value=0.60,
        step=0.05,
        help="كلما انخفضت القيمة كان النطاق الأخير أضيق مقارنة بنطاق 20 يوم.",
    )

    st.subheader("بيانات إضافية")
    api_key = st.text_input(
        "ChartExchange API Key (اختياري)",
        type="password",
        help="اتركه فارغاً إذا لم يكن لديك مفتاح. بيانات السعر والمؤشرات تعمل بدونه.",
    )

    scan = st.button("🔎 ابدأ الفحص", type="primary", use_container_width=True)


if scan:
    symbols = []
    for s in symbols_text.replace("\n", ",").split(","):
        s = s.strip().upper()
        if s and s not in symbols:
            symbols.append(s)

    cfg = {
        "max_return20": max_return20,
        "max_rsi": max_rsi,
        "max_rvol": max_rvol,
        "max_distance_low": max_distance_low,
        "max_range_compression": max_range_compression,
    }

    rows = []
    details = {}

    progress = st.progress(0)
    status = st.empty()

    for i, symbol in enumerate(symbols):
        status.write(f"جاري تحليل {symbol} ...")
        try:
            df = get_prices(symbol)
            if df.empty:
                continue

            result, calc = evaluate(df, cfg)
            if result is None:
                continue

            result["Ticker"] = symbol

            # Optional external short/borrow information
            ce = get_chart_exchange(symbol, api_key)
            result["ChartExchange"] = "متوفر" if ce else "غير مضاف/غير متاح"

            rows.append(result)
            details[symbol] = {
                "raw": df,
                "calc": calc,
                "ce": ce,
            }
        except Exception as e:
            st.warning(f"{symbol}: تعذر التحليل — {e}")

        progress.progress((i + 1) / len(symbols))

    status.empty()
    progress.empty()

    if not rows:
        st.error("ما حصلت بيانات كافية. تأكد من التكرز أو اتصال الإنترنت.")
    else:
        table = pd.DataFrame(rows)

        # Sort by score first, then by distance from the 20d low
        table = table.sort_values(
            ["Score", "Dist Low20%"],
            ascending=[False, True],
        )

        st.subheader("📊 نتائج السكانر")

        display_cols = [
            "Ticker", "Close", "Return20%", "RSI14", "RVOL20",
            "Dist Low20%", "MA20", "MA30", "MACD", "MACD Signal",
            "Range Compression", "Score", "Max Score", "ChartExchange"
        ]

        st.dataframe(
            table[display_cols].round(3),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ تحميل النتائج CSV",
            data=table[display_cols].to_csv(index=False).encode("utf-8-sig"),
            file_name="bottom_setup_scan.csv",
            mime="text/csv",
        )

        st.subheader("🔍 تفاصيل سهم")
        selected = st.selectbox("اختر السهم", table["Ticker"].tolist())

        item = details[selected]
        calc = item["calc"]
        latest = calc.iloc[-1]

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("السعر", f"{latest['Close']:.4f}")
        c2.metric("20D Return", f"{latest['Return20Pct']:.2f}%")
        c3.metric("RSI", f"{latest['RSI14']:.1f}")
        c4.metric("RVOL", f"{latest['RVOL20']:.2f}x")
        c5.metric("MACD Hist", f"{latest['MACD_Hist']:.4f}")
        c6.metric("من قاع 20D", f"{latest['DistanceFromLow20Pct']:.2f}%")

        checks = evaluate(item["raw"], cfg)[0]["Checks"]
        st.markdown("### شروط الاستراتيجية")
        check_cols = st.columns(4)
        for i, (name, passed) in enumerate(checks.items()):
            check_cols[i % 4].write(("✅ " if passed else "❌ ") + name)

        # Price + moving averages
        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=calc.index,
                open=calc["Open"],
                high=calc["High"],
                low=calc["Low"],
                close=calc["Close"],
                name="Price",
            )
        )
        fig.add_trace(
            go.Scatter(x=calc.index, y=calc["MA20"], name="MA20")
        )
        fig.add_trace(
            go.Scatter(x=calc.index, y=calc["MA30"], name="MA30")
        )
        fig.update_layout(
            title=f"{selected} — Price / MA20 / MA30",
            height=550,
            xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # RSI / MACD / Volume
        col1, col2 = st.columns(2)

        with col1:
            fig_rsi = go.Figure()
            fig_rsi.add_trace(
                go.Scatter(x=calc.index, y=calc["RSI14"], name="RSI14")
            )
            fig_rsi.add_hline(y=max_rsi, line_dash="dash")
            fig_rsi.update_layout(title="RSI14", height=350)
            st.plotly_chart(fig_rsi, use_container_width=True)

        with col2:
            fig_macd = go.Figure()
            fig_macd.add_trace(
                go.Scatter(x=calc.index, y=calc["MACD"], name="MACD")
            )
            fig_macd.add_trace(
                go.Scatter(x=calc.index, y=calc["MACD_Signal"], name="Signal")
            )
            fig_macd.add_hline(y=0, line_dash="dash")
            fig_macd.update_layout(title="MACD", height=350)
            st.plotly_chart(fig_macd, use_container_width=True)

        st.markdown("### آخر البيانات")
        st.dataframe(
            calc.tail(30).round(4),
            use_container_width=True,
        )

else:
    st.info(
        "اكتب التكرز واضغط «ابدأ الفحص». يمكنك تعديل حدود الاستراتيجية من القائمة الجانبية."
    )
