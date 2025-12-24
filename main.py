# main.py
from datetime import datetime
from typing import Dict, Any, List, Optional

import pandas as pd
import yfinance as yf
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select

from db import create_db_and_tables, get_session
from models import ApprovalRequest, Position, Trade

app = FastAPI(title="HITL Finance Agent API")

# If you use Next.js rewrites (/api -> backend), CORS matters less,
# but keeping this is fine for direct Swagger + local dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    # Do not crash the whole service if DB is temporarily unreachable.
    # We will still fail DB-backed endpoints with clear errors, but /health stays up.
    try:
        create_db_and_tables()
    except Exception as e:
        # Keep it visible in Render logs
        print(f"[startup] DB init failed: {repr(e)}")

@app.get("/health")
def health():
    return {"ok": True}


# -------------------------
# Helpers
# -------------------------
def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance can return:
      - SingleIndex columns: Open/High/Low/Close/Volume
      - MultiIndex columns: (Ticker, Field) or (Field, Ticker)
    This normalizes so that market endpoints can reliably access fields.
    """
    if df is None or df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = set(df.columns.get_level_values(0))
        lvl1 = set(df.columns.get_level_values(1))
        ohlcv = {"Open", "High", "Low", "Close", "Volume"}

        if ohlcv.issubset(lvl0):
            # ("Open","AAPL") style
            df.columns = df.columns.get_level_values(0)
        elif ohlcv.issubset(lvl1):
            # ("AAPL","Open") style
            df.columns = df.columns.get_level_values(1)
        else:
            df.columns = ["_".join(map(str, c)) for c in df.columns]

    return df


def _latest_close_for(data: pd.DataFrame, sym: str) -> float:
    """
    Robustly extract the latest close for a symbol from yfinance output.
    Supports:
      - MultiIndex columns (sym, "Close") or ("Close", sym)
      - SingleIndex columns when only one ticker returned
    Returns NaN if unavailable.
    """
    try:
        if data is None or data.empty:
            return float("nan")

        # Multi-ticker case typically MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            # Most common: (SYM, "Close")
            if (sym, "Close") in data.columns:
                s = data[(sym, "Close")].dropna()
                return float(s.iloc[-1]) if len(s) else float("nan")

            # Sometimes: ("Close", SYM)
            if ("Close", sym) in data.columns:
                s = data[("Close", sym)].dropna()
                return float(s.iloc[-1]) if len(s) else float("nan")

            # Fallback: try to find a "Close" column under sym by slicing
            try:
                sub = data.xs(sym, axis=1, level=0, drop_level=False)
                # sub columns might still be MultiIndex
                if isinstance(sub.columns, pd.MultiIndex) and (sym, "Close") in sub.columns:
                    s = sub[(sym, "Close")].dropna()
                    return float(s.iloc[-1]) if len(s) else float("nan")
            except Exception:
                pass

            return float("nan")

        # SingleIndex: assume "Close" exists
        if "Close" in data.columns:
            s = data["Close"].dropna()
            return float(s.iloc[-1]) if len(s) else float("nan")

        return float("nan")
    except Exception:
        return float("nan")


# -------------------------
# HITL Approvals API
# -------------------------
@app.post("/hitl/approvals", response_model=ApprovalRequest)
def create_approval(payload: Dict[str, Any], session: Session = Depends(get_session)):
    symbol = payload.get("symbol")
    recommendation = payload.get("recommendation_json")

    if not symbol or not isinstance(symbol, str):
        raise HTTPException(status_code=400, detail="symbol is required (string)")
    if recommendation is None or not isinstance(recommendation, dict):
        raise HTTPException(status_code=400, detail="recommendation_json is required (object)")

    approval = ApprovalRequest(
        status="PENDING",
        symbol=symbol.upper().strip(),
        recommendation_json=recommendation,
    )

    session.add(approval)
    session.commit()
    session.refresh(approval)
    return approval


@app.get("/hitl/approvals", response_model=List[ApprovalRequest])
def list_approvals(session: Session = Depends(get_session)):
    stmt = select(ApprovalRequest).order_by(ApprovalRequest.created_ts.desc())
    return session.exec(stmt).all()


@app.post("/hitl/approvals/{approval_id}/decision", response_model=ApprovalRequest)
def decide_approval(approval_id: int, payload: Dict[str, Any], session: Session = Depends(get_session)):
    decision = payload.get("decision")
    comment = payload.get("comment")

    if decision not in ("APPROVED", "REJECTED"):
        raise HTTPException(status_code=400, detail="decision must be APPROVED or REJECTED")

    approval = session.get(ApprovalRequest, approval_id)
    if not approval:
        raise HTTPException(status_code=404, detail="Approval not found")

    if approval.status != "PENDING":
        raise HTTPException(status_code=409, detail=f"Cannot decide approval in status {approval.status}")

    approval.status = decision
    approval.decided_ts = datetime.utcnow()
    if comment:
        approval.comment = str(comment)

    session.add(approval)
    session.commit()
    session.refresh(approval)
    return approval


# -------------------------
# Market Data API (OHLCV)
# -------------------------
@app.get("/market/ohlcv")
def get_ohlcv(symbol: str, period: str = "6mo", interval: str = "1d"):
    sym = symbol.upper().strip()

    df = yf.download(
        sym,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No data for symbol={sym}")

    df = _flatten_yf_columns(df)
    df = df.reset_index()

    time_col = "Date" if "Date" in df.columns else "Datetime"
    if time_col not in df.columns:
        raise HTTPException(status_code=500, detail="No Date/Datetime column found in data")

    ts_col = df[time_col]
    if isinstance(ts_col, pd.DataFrame):
        ts_col = ts_col.iloc[:, 0]
    ts_col = pd.to_datetime(ts_col, errors="coerce")

    def _col(name: str):
        if name not in df.columns:
            raise HTTPException(status_code=500, detail=f"Missing column: {name}")
        c = df[name]
        if isinstance(c, pd.DataFrame):
            c = c.iloc[:, 0]
        return c

    open_s = _col("Open")
    high_s = _col("High")
    low_s = _col("Low")
    close_s = _col("Close")
    vol_s = _col("Volume") if "Volume" in df.columns else pd.Series([0] * len(df))

    points = []
    for t, o, h, l, c, v in zip(ts_col, open_s, high_s, low_s, close_s, vol_s):
        if pd.isna(t):
            continue

        # Ensure t becomes ISO string
        if isinstance(t, pd.Timestamp):
            t_iso = t.to_pydatetime().isoformat()
        else:
            t_iso = pd.to_datetime(t).to_pydatetime().isoformat()

        points.append(
            {
                "t": t_iso,
                "o": float(o),
                "h": float(h),
                "l": float(l),
                "c": float(c),
                "v": int(v) if pd.notna(v) else 0,
            }
        )

    if not points:
        raise HTTPException(status_code=404, detail=f"No usable OHLCV points for symbol={sym}")

    return {"symbol": sym, "period": period, "interval": interval, "count": len(points), "points": points}


# -------------------------
# Portfolio (manual positions)
# -------------------------
@app.get("/portfolio/positions", response_model=List[Position])
def list_positions(session: Session = Depends(get_session)):
    stmt = select(Position).order_by(Position.symbol.asc())
    return session.exec(stmt).all()


@app.post("/portfolio/positions", response_model=Position)
def upsert_position(payload: Dict[str, Any], session: Session = Depends(get_session)):
    symbol = payload.get("symbol")
    qty = payload.get("qty")
    avg_price = payload.get("avg_price")

    if not symbol or not isinstance(symbol, str):
        raise HTTPException(status_code=400, detail="symbol is required (string)")
    try:
        qty_f = float(qty)
        avg_f = float(avg_price)
    except Exception:
        raise HTTPException(status_code=400, detail="qty and avg_price must be numbers")

    sym = symbol.upper().strip()

    existing = session.exec(select(Position).where(Position.symbol == sym)).first()
    if existing:
        existing.qty = qty_f
        existing.avg_price = avg_f
        session.add(existing)
        session.commit()
        session.refresh(existing)
        return existing

    pos = Position(symbol=sym, qty=qty_f, avg_price=avg_f)
    session.add(pos)
    session.commit()
    session.refresh(pos)
    return pos


@app.get("/portfolio/pnl")
def portfolio_pnl(session: Session = Depends(get_session)):
    """
    Position-based PnL:
    - Uses Position table (symbol, qty, avg_price)
    - Pulls latest close via yfinance
    - NEVER crashes if a ticker is missing from response
    """
    positions = session.exec(select(Position)).all()
    if not positions:
        return {"unrealized_pnl": 0.0, "gross_exposure": 0.0, "positions_count": 0, "positions": []}

    symbols = [p.symbol for p in positions]

    data = yf.download(
        tickers=" ".join(symbols),
        period="5d",
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
    )

    rows = []
    total_pnl = 0.0
    total_exposure = 0.0

    for p in positions:
        last = _latest_close_for(data, p.symbol)

        if pd.isna(last):
            rows.append(
                {
                    "symbol": p.symbol,
                    "qty": p.qty,
                    "avg_price": p.avg_price,
                    "last_price": None,
                    "unrealized_pnl": None,
                }
            )
            continue

        exposure = p.qty * last
        pnl = p.qty * (last - p.avg_price)

        total_exposure += exposure
        total_pnl += pnl

        rows.append(
            {
                "symbol": p.symbol,
                "qty": p.qty,
                "avg_price": p.avg_price,
                "last_price": float(last),
                "unrealized_pnl": float(pnl),
            }
        )

    return {
        "unrealized_pnl": float(total_pnl),
        "gross_exposure": float(total_exposure),
        "positions_count": len(rows),
        "positions": rows,
    }


# -------------------------
# Agent (creates approval request)
# -------------------------
@app.post("/agent/ask")
def agent_ask(payload: Dict[str, Any], session: Session = Depends(get_session)):
    question = payload.get("question")
    symbol = payload.get("symbol")

    if not question or not isinstance(question, str):
        raise HTTPException(status_code=400, detail="question is required (string)")
    if not symbol or not isinstance(symbol, str):
        raise HTTPException(status_code=400, detail="symbol is required (string)")

    sym = symbol.upper().strip()

    df = yf.download(sym, period="6mo", interval="1d", auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No data for symbol={sym}")

    df = _flatten_yf_columns(df)

    close = df["Close"].dropna()
    if len(close) < 1:
        raise HTTPException(status_code=404, detail=f"No Close data for symbol={sym}")

    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2]) if len(close) >= 2 else last_close
    change_1d = last_close - prev_close
    change_1d_pct = (change_1d / prev_close * 100.0) if prev_close != 0 else 0.0

    bias = "BUY" if change_1d_pct > 0 else "HOLD"
    risk = "Medium"
    time_horizon = "2-6 weeks"

    recommendation_json = {
        "symbol": sym,
        "question": question,
        "recommendation": bias,
        "time_horizon": time_horizon,
        "risk": risk,
        "market_snapshot": {
            "last_close": last_close,
            "change_1d": change_1d,
            "change_1d_pct": change_1d_pct,
        },
        "rationale": [
            "MVP agent uses 1-day momentum as a placeholder signal.",
            "This is decision support only; no auto-trading.",
        ],
        "next_checks": [
            "Review earnings calendar and guidance.",
            "Check broader market trend (SPY/QQQ).",
            "Confirm risk limits for position sizing.",
        ],
    }

    approval = ApprovalRequest(status="PENDING", symbol=sym, recommendation_json=recommendation_json)
    session.add(approval)
    session.commit()
    session.refresh(approval)

    answer = (
        f"Research memo created for {sym}. "
        f"Last close: {last_close:.2f} ({change_1d_pct:+.2f}% 1D). "
        f"Recommendation (MVP): {bias}. "
        f"Approval request created: {approval.id}."
    )

    return {"answer": answer, "recommendation_json": recommendation_json, "approval_id": approval.id}


# -------------------------
# Trades + trade-based PnL
# -------------------------
@app.get("/trades")
def list_trades(session: Session = Depends(get_session)):
    stmt = select(Trade).order_by(Trade.ts.desc(), Trade.id.desc())
    return session.exec(stmt).all()


@app.post("/trades")
def create_trade(payload: Dict[str, Any], session: Session = Depends(get_session)):
    symbol = payload.get("symbol")
    side = payload.get("side")
    qty = payload.get("qty")
    price = payload.get("price")
    note = payload.get("note", None)

    if not symbol or not isinstance(symbol, str):
        raise HTTPException(status_code=400, detail="symbol is required (string)")

    if not side or not isinstance(side, str):
        raise HTTPException(status_code=400, detail="side is required (BUY/SELL)")

    side_norm = side.strip().upper()
    if side_norm not in ("BUY", "SELL"):
        raise HTTPException(status_code=400, detail="side must be BUY or SELL")

    try:
        qty_f = float(qty)
        price_f = float(price)
    except Exception:
        raise HTTPException(status_code=400, detail="qty and price must be numbers")

    if qty_f <= 0:
        raise HTTPException(status_code=400, detail="qty must be > 0")
    if price_f <= 0:
        raise HTTPException(status_code=400, detail="price must be > 0")

    sym = symbol.strip().upper()

    trade = Trade(
        symbol=sym,
        side=side_norm,
        qty=qty_f,
        price=price_f,
        note=(str(note).strip() if note is not None and str(note).strip() else None),
    )

    session.add(trade)
    session.commit()
    session.refresh(trade)
    return trade


@app.get("/portfolio/pnl_trades")
def portfolio_pnl_trades(session: Session = Depends(get_session)):
    """
    Trade-based PnL (weighted average cost).
    """
    trades = session.exec(select(Trade).order_by(Trade.ts.asc(), Trade.id.asc())).all()

    if not trades:
        return {
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "net_pnl": 0.0,
            "gross_exposure": 0.0,
            "positions_count": 0,
            "positions": [],
            "method": "weighted_avg_cost",
        }

    state: Dict[str, Dict[str, float]] = {}

    def get_state(sym: str):
        if sym not in state:
            state[sym] = {"qty": 0.0, "avg_cost": 0.0, "realized_pnl": 0.0}
        return state[sym]

    for t in trades:
        sym = t.symbol.upper().strip()
        s = get_state(sym)

        qty = float(t.qty)
        price = float(t.price)

        if t.side.upper() == "BUY":
            new_qty = s["qty"] + qty
            if new_qty <= 0:
                s["qty"] = 0.0
                s["avg_cost"] = 0.0
            else:
                s["avg_cost"] = ((s["qty"] * s["avg_cost"]) + (qty * price)) / new_qty
                s["qty"] = new_qty

        elif t.side.upper() == "SELL":
            sell_qty = qty
            s["realized_pnl"] += sell_qty * (price - s["avg_cost"])
            s["qty"] -= sell_qty
            if abs(s["qty"]) < 1e-9:
                s["qty"] = 0.0
                s["avg_cost"] = 0.0

    open_symbols = [sym for sym, s in state.items() if abs(s["qty"]) > 1e-9]

    data = None
    if open_symbols:
        data = yf.download(
            tickers=" ".join(open_symbols),
            period="5d",
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="ticker",
        )

    rows = []
    total_realized = 0.0
    total_unrealized = 0.0
    total_exposure = 0.0

    for sym, s in state.items():
        qty = float(s["qty"])
        realized = float(s["realized_pnl"])
        total_realized += realized

        if abs(qty) < 1e-9:
            continue

        last = _latest_close_for(data, sym) if data is not None else float("nan")

        if pd.isna(last):
            rows.append(
                {
                    "symbol": sym,
                    "qty": qty,
                    "avg_cost": float(s["avg_cost"]),
                    "last_price": None,
                    "realized_pnl": realized,
                    "unrealized_pnl": None,
                }
            )
            continue

        exposure = qty * last
        unreal = qty * (last - float(s["avg_cost"]))

        total_exposure += exposure
        total_unrealized += unreal

        rows.append(
            {
                "symbol": sym,
                "qty": qty,
                "avg_cost": float(s["avg_cost"]),
                "last_price": float(last),
                "realized_pnl": realized,
                "unrealized_pnl": float(unreal),
            }
        )

    net = total_realized + total_unrealized

    return {
        "realized_pnl": float(total_realized),
        "unrealized_pnl": float(total_unrealized),
        "net_pnl": float(net),
        "gross_exposure": float(total_exposure),
        "positions_count": len(rows),
        "positions": rows,
        "method": "weighted_avg_cost",
    }

@app.delete("/portfolio/positions/{symbol}")
def delete_position(symbol: str, session: Session = Depends(get_session)):
    sym = symbol.upper().strip()
    existing = session.exec(select(Position).where(Position.symbol == sym)).first()
    if not existing:
        raise HTTPException(status_code=404, detail="Position not found")
    session.delete(existing)
    session.commit()
    return {"ok": True, "deleted": sym}

@app.delete("/trades/{trade_id}")
def delete_trade(trade_id: int, session: Session = Depends(get_session)):
    trade = session.get(Trade, trade_id)
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")
    session.delete(trade)
    session.commit()
    return {"ok": True, "deleted_id": trade_id}