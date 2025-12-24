from datetime import datetime
from typing import Optional, Dict, Any

from sqlmodel import SQLModel, Field, Column
from sqlalchemy import JSON


class ApprovalRequest(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    status: str = Field(default="PENDING", index=True)

    symbol: str = Field(index=True)

    recommendation_json: Dict[str, Any] = Field(sa_column=Column(JSON))

    created_ts: datetime = Field(default_factory=datetime.utcnow)
    decided_ts: Optional[datetime] = Field(default=None)

    comment: Optional[str] = Field(default=None)

class Position(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key = True)
    symbol: str = Field(index=True)
    qty: float
    avg_price: float
    created_ts: datetime = Field(default_factory=datetime.utcnow)

class Trade(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key= True)
    symbol: str = Field(index=True)
    side: str # Buy or sell
    qty: float
    price: float
    ts: datetime = Field(default_factory=datetime.utcnow, index=True)
    note: Optional[str] = None
   
    
