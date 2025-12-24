# HITL Finance Agent
Human-in-the-Loop Decision Support System for Trading & Portfolio Analysis

---

## Overview

**HITL Finance Agent** is a full-stack, production-deployed **Human-in-the-Loop (HITL) financial decision support system**.

The project demonstrates how AI-style research generation, explicit human approvals, and persistent portfolio data can coexist safely in a real-world architecture **without automatic execution or black-box behavior**.

This system is intentionally designed to:

- Generate **recommendations, not trades**
- Require **explicit human approval**
- Preserve **auditability and transparency**
- Cleanly separate **frontend UI**, **backend APIs**, and **data storage**

It is well-suited for portfolio demonstrations, system-design interviews, and full-stack architecture discussions.

External market data is fetched read-only from public APIs and is **never persisted**.

---

## Core Principles

### Human-in-the-Loop (HITL)

- The agent **cannot execute trades**
- All recommendations must be:
  - Generated
  - Stored
  - Reviewed
  - Explicitly approved or rejected by a human

### Deterministic and Explainable

- Agent outputs include:
  - Rationale
  - Market snapshot
  - Risk and time horizon
- No autonomous or opaque decision-making

### Separation of Concerns

- **Frontend**: presentation and user interaction
- **Backend**: business logic and persistence
- **Database**: single source of truth

---

## Application Pages

| Page        | Purpose |
|------------|--------|
| Dashboard  | Portfolio overview and PnL |
| Trades     | Add and review trades |
| Market     | OHLCV candlestick charts |
| Agent Desk | Generate research memos |
| Approvals  | Human approval workflow |

---

## Backend Capabilities

### Portfolio
- Manual positions
- Real-time unrealized PnL
- Trade-based realized and unrealized PnL

### Trades
- BUY / SELL tracking
- Weighted average cost calculation
- Persistent trade history

### Market Data
- OHLCV candlestick data
- Fetched on demand
- Not stored server-side

### Agent (Decision Support)
- Generates structured research memos
- Produces recommendations
- Automatically creates approval requests

### HITL Approvals
- Pending / Approved / Rejected states
- Timestamped decisions
- Fully auditable history

---

## Key API Endpoints

### Health
GET /health

### Portfolio
GET /portfolio/positions
POST /portfolio/positions
DELETE /portfolio/positions/{symbol}

### PnL
GET /portfolio/pnl
GET /portfolio/pnl_trades

### Trades
GET /trades
POST /trades
DELETE /trades/{trade_id}

### Market
GET /market/ohlcv

### Agent
POST /agent/ask

### Approvals
GET /hitl/approvals
POST /hitl/approvals
POST /hitl/approvals/{id}/decision


Swagger documentation is available at `/docs`.

---

## Data Persistence & Sharing Behavior

### Where data is stored

- Trades, positions, approvals, and agent outputs are stored in **PostgreSQL (Supabase)**
- The database is the **single source of truth**

### Shared demo mode

- No authentication is enabled
- All users interact with the **same dataset**
- Data persists across sessions and users

### What is not stored

- Market prices
- Candlestick data
- External API responses

These are fetched fresh on demand.

---

## Frontend–Backend Integration

The frontend communicates with the backend using a single environment variable:

NEXT_PUBLIC_API_BASE=https://<backend-url>


This value is injected at build time in the hosting environment.

---

## Local Development (Optional)

### Backend

pip install -r requirements.txt
uvicorn main:app --reload

### Frontend
npm install
npm run dev


---

## Deployment Summary

### Frontend
- Hosted on **Vercel**
- Auto-deployed from GitHub
- Environment variables control API routing

### Backend
- Hosted on **Render**
- Auto-deployed from GitHub
- PostgreSQL hosted on **Supabase** using a session pooler

---

## Security & Scope Notes

- No authentication (by design)
- No automatic trade execution
- Decision support only
- Suitable for demos and interviews

---

## Future Enhancements

- Per-user or per-workspace data isolation
- Authentication and authorization
- Static demo mode (localStorage-only)
- Admin reset and seeding tools
- Risk rules and guardrails

---

## License

MIT
