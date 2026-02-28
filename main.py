<<<<<<< HEAD
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Optional, List

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import jwt, JWTError
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, ConfigDict
from sqlalchemy.orm import Session

from db import SessionLocal, engine, Base
from models import User, Job, Invoice, Mileage, Expense


def _env(name: str, default: str) -> str:
    val = os.getenv(name)
    return val if val is not None and val.strip() else default


def _parse_cors_origins(raw: str) -> list[str]:
    raw = raw.strip()
    if raw == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]


app = FastAPI(title="JobFlow API")

cors_origins = _parse_cors_origins(_env("CORS_ORIGINS", "*"))
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)

SECRET_KEY = _env("SECRET_KEY", "CHANGE_ME_TO_A_LONG_RANDOM_SECRET")
ALGORITHM = _env("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_DAYS = int(_env("ACCESS_TOKEN_EXPIRE_DAYS", "30"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return pwd_context.verify(password, password_hash)


def create_access_token(user_id: int) -> str:
    expire = datetime.utcnow() + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    payload = {"sub": str(user_id), "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme),
) -> User:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = int(payload.get("sub"))
    except (JWTError, ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


class _ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class AuthSignupIn(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None


class AuthLoginIn(BaseModel):
    email: EmailStr
    password: str


class AuthOut(BaseModel):
    token: str
    userId: int


class JobCreateIn(BaseModel):
    title: str
    client_name: Optional[str] = None


class JobUpdateIn(BaseModel):
    title: Optional[str] = None
    client_name: Optional[str] = None
    status: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class JobOut(_ORMModel):
    id: int
    title: str
    client_name: Optional[str]
    status: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    created_at: datetime


class InvoiceCreateIn(BaseModel):
    job_id: Optional[int] = None
    amount: float
    note: Optional[str] = None
    status: Optional[str] = "draft"


class InvoiceUpdateIn(BaseModel):
    amount: Optional[float] = None
    note: Optional[str] = None
    status: Optional[str] = None
    job_id: Optional[int] = None


class InvoiceOut(_ORMModel):
    id: int
    job_id: Optional[int]
    amount: float
    note: Optional[str]
    status: str
    created_at: datetime


class MileageCreateIn(BaseModel):
    job_id: Optional[int] = None
    miles: float
    note: Optional[str] = None


class MileageOut(_ORMModel):
    id: int
    job_id: Optional[int]
    miles: float
    note: Optional[str]
    created_at: datetime


class ExpenseCreateIn(BaseModel):
    job_id: Optional[int] = None
    amount: float
    category: str
    note: Optional[str] = None


class ExpenseOut(_ORMModel):
    id: int
    job_id: Optional[int]
    amount: float
    category: str
    note: Optional[str]
    created_at: datetime


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/auth/signup", response_model=AuthOut)
def signup(data: AuthSignupIn, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == data.email.lower()).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already in use")

    user = User(email=data.email.lower(), password_hash=hash_password(data.password), full_name=data.full_name)
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(user.id)
    return {"token": token, "userId": user.id}


@app.post("/auth/login", response_model=AuthOut)
def login(data: AuthLoginIn, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email.lower()).first()
    if not user or not verify_password(data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token(user.id)
    return {"token": token, "userId": user.id}


@app.post("/auth/token")
def token_endpoint(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username.lower()).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token(user.id)
    return {"access_token": token, "token_type": "bearer"}


@app.delete("/users/me")
def delete_me(db: Session = Depends(get_db), me: User = Depends(get_current_user)):
    db.delete(me)
    db.commit()
    return {"ok": True}


@app.get("/jobs", response_model=List[JobOut])
def list_jobs(db: Session = Depends(get_db), me: User = Depends(get_current_user)):
    return db.query(Job).filter(Job.user_id == me.id).order_by(Job.id.desc()).all()


@app.post("/jobs", response_model=JobOut)
def create_job(data: JobCreateIn, db: Session = Depends(get_db), me: User = Depends(get_current_user)):
    job = Job(user_id=me.id, title=data.title, client_name=data.client_name, status="open")
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


@app.put("/jobs/{job_id}", response_model=JobOut)
def update_job(job_id: int, data: JobUpdateIn, db: Session = Depends(get_db), me: User = Depends(get_current_user)):
    job = db.query(Job).filter(Job.id == job_id, Job.user_id == me.id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    for k, v in data.model_dump(exclude_unset=True).items():
        setattr(job, k, v)

    db.commit()
    db.refresh(job)
    return job


@app.delete("/jobs/{job_id}")
def delete_job(job_id: int, db: Session = Depends(get_db), me: User = Depends(get_current_user)):
    job = db.query(Job).filter(Job.id == job_id, Job.user_id == me.id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    db.delete(job)
    db.commit()
    return {"ok": True}


def _validate_job_ownership(db: Session, me: User, job_id: Optional[int]) -> None:
    if job_id is None:
        return
    job = db.query(Job).filter(Job.id == job_id, Job.user_id == me.id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found for this user")


@app.get("/invoices", response_model=List[InvoiceOut])
def list_invoices(job_id: Optional[int] = None, db: Session = Depends(get_db), me: User = Depends(get_current_user)):
    q = db.query(Invoice).filter(Invoice.user_id == me.id)
    if job_id is not None:
        q = q.filter(Invoice.job_id == job_id)
    return q.order_by(Invoice.id.desc()).all()


@app.post("/invoices", response_model=InvoiceOut)
def create_invoice(data: InvoiceCreateIn, db: Session = Depends(get_db), me: User = Depends(get_current_user)):
    _validate_job_ownership(db, me, data.job_id)
    inv = Invoice(user_id=me.id, job_id=data.job_id, amount=data.amount, note=data.note, status=data.status or "draft")
    db.add(inv)
    db.commit()
    db.refresh(inv)
    return inv


@app.put("/invoices/{invoice_id}", response_model=InvoiceOut)
def update_invoice(invoice_id: int, data: InvoiceUpdateIn, db: Session = Depends(get_db), me: User = Depends(get_current_user)):
    inv = db.query(Invoice).filter(Invoice.id == invoice_id, Invoice.user_id == me.id).first()
    if not inv:
        raise HTTPException(status_code=404, detail="Invoice not found")

    payload = data.model_dump(exclude_unset=True)
    if "job_id" in payload:
        _validate_job_ownership(db, me, payload.get("job_id"))
    for k, v in payload.items():
        setattr(inv, k, v)

    db.commit()
    db.refresh(inv)
    return inv


@app.delete("/invoices/{invoice_id}")
def delete_invoice(invoice_id: int, db: Session = Depends(get_db), me: User = Depends(get_current_user)):
    inv = db.query(Invoice).filter(Invoice.id == invoice_id, Invoice.user_id == me.id).first()
    if not inv:
        raise HTTPException(status_code=404, detail="Invoice not found")
    db.delete(inv)
    db.commit()
    return {"ok": True}


@app.get("/mileage", response_model=List[MileageOut])
def list_mileage(job_id: Optional[int] = None, db: Session = Depends(get_db), me: User = Depends(get_current_user)):
    q = db.query(Mileage).filter(Mileage.user_id == me.id)
    if job_id is not None:
        q = q.filter(Mileage.job_id == job_id)
    return q.order_by(Mileage.id.desc()).all()


@app.post("/mileage", response_model=MileageOut)
def create_mileage(data: MileageCreateIn, db: Session = Depends(get_db), me: User = Depends(get_current_user)):
    _validate_job_ownership(db, me, data.job_id)
    item = Mileage(user_id=me.id, job_id=data.job_id, miles=data.miles, note=data.note)
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


class MileageUpdateIn(BaseModel):
    job_id: Optional[int] = None
    miles: Optional[float] = None
    note: Optional[str] = None


class ExpenseUpdateIn(BaseModel):
    job_id: Optional[int] = None
    amount: Optional[float] = None
    category: Optional[str] = None
    note: Optional[str] = None


@app.get("/mileage/{mileage_id}", response_model=MileageOut)
def get_mileage_entry(mileage_id: int, db: Session = Depends(get_db), me: User = Depends(get_current_user)):
    item = db.query(Mileage).filter(Mileage.id == mileage_id, Mileage.user_id == me.id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Mileage entry not found")
    return item


@app.put("/mileage/{mileage_id}", response_model=MileageOut)
def update_mileage(mileage_id: int, data: MileageUpdateIn, db: Session = Depends(get_db), me: User = Depends(get_current_user)):
    item = db.query(Mileage).filter(Mileage.id == mileage_id, Mileage.user_id == me.id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Mileage entry not found")
    for k, v in data.model_dump(exclude_unset=True).items():
        setattr(item, k, v)
    db.commit()
    db.refresh(item)
    return item


@app.put("/expenses/{expense_id}", response_model=ExpenseOut)
def update_expense(expense_id: int, data: ExpenseUpdateIn, db: Session = Depends(get_db), me: User = Depends(get_current_user)):
    item = db.query(Expense).filter(Expense.id == expense_id, Expense.user_id == me.id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Expense not found")
    for k, v in data.model_dump(exclude_unset=True).items():
        setattr(item, k, v)
    db.commit()
    db.refresh(item)
    return item



def delete_mileage(mileage_id: int, db: Session = Depends(get_db), me: User = Depends(get_current_user)):
    item = db.query(Mileage).filter(Mileage.id == mileage_id, Mileage.user_id == me.id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Mileage entry not found")
    db.delete(item)
    db.commit()
    return {"ok": True}


@app.get("/expenses", response_model=List[ExpenseOut])
def list_expenses(job_id: Optional[int] = None, db: Session = Depends(get_db), me: User = Depends(get_current_user)):
    q = db.query(Expense).filter(Expense.user_id == me.id)
    if job_id is not None:
        q = q.filter(Expense.job_id == job_id)
    return q.order_by(Expense.id.desc()).all()


@app.post("/expenses", response_model=ExpenseOut)
def create_expense(data: ExpenseCreateIn, db: Session = Depends(get_db), me: User = Depends(get_current_user)):
    _validate_job_ownership(db, me, data.job_id)
    item = Expense(user_id=me.id, job_id=data.job_id, amount=data.amount, category=data.category, note=data.note)
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


@app.get("/expenses/{expense_id}", response_model=ExpenseOut)
def get_expense(expense_id: int, db: Session = Depends(get_db), me: User = Depends(get_current_user)):
    item = db.query(Expense).filter(Expense.id == expense_id, Expense.user_id == me.id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Expense not found")
    return item


@app.delete("/expenses/{expense_id}")
def delete_expense(expense_id: int, db: Session = Depends(get_db), me: User = Depends(get_current_user)):
    item = db.query(Expense).filter(Expense.id == expense_id, Expense.user_id == me.id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Expense not found")
    db.delete(item)
    db.commit()
    return {"ok": True}


# ══════════════════════════════════════════════════════════════════════════════
# AI ANALYSIS ENDPOINTS
# All Claude API calls go through here — NEVER called directly from the app.
# ══════════════════════════════════════════════════════════════════════════════

import httpx
import json

ANTHROPIC_API_KEY = _env("ANTHROPIC_API_KEY", "")
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL = "claude-sonnet-4-6"

IRS_RATE = 0.67  # 2025 IRS standard mileage rate


def _gather_user_data(db: Session, me: User) -> dict:
    """Collect all data for a user and return a structured summary for the AI."""
    jobs = db.query(Job).filter(Job.user_id == me.id).all()
    invoices = db.query(Invoice).filter(Invoice.user_id == me.id).all()
    expenses = db.query(Expense).filter(Expense.user_id == me.id).all()
    mileage = db.query(Mileage).filter(Mileage.user_id == me.id).all()

    # Build per-job summary
    job_summaries = []
    for job in jobs:
        job_inv = [i for i in invoices if i.job_id == job.id]
        job_exp = [e for e in expenses if e.job_id == job.id]
        job_mil = [m for m in mileage if m.job_id == job.id]

        invoiced = sum(float(i.amount or 0) for i in job_inv)
        paid = sum(float(i.amount or 0) for i in job_inv if str(i.status or "").lower() == "paid")
        exp_total = sum(float(e.amount or 0) for e in job_exp)
        miles_total = sum(float(m.miles or 0) for m in job_mil)
        mile_deduction = miles_total * IRS_RATE

        expense_by_cat: dict[str, float] = {}
        for e in job_exp:
            k = str(e.category or "Other")
            expense_by_cat[k] = expense_by_cat.get(k, 0.0) + float(e.amount or 0)

        job_summaries.append({
            "id": job.id,
            "title": job.title,
            "client": job.client_name or "Unknown",
            "status": job.status or "open",
            "invoiced": round(invoiced, 2),
            "paid": round(paid, 2),
            "outstanding": round(invoiced - paid, 2),
            "expenses": round(exp_total, 2),
            "profit": round(invoiced - exp_total, 2),
            "miles": round(miles_total, 1),
            "mileage_deduction": round(mile_deduction, 2),
            "net_after_mileage": round(invoiced - exp_total + mile_deduction, 2),
            "expense_breakdown": expense_by_cat,
        })

    # Overall totals
    total_invoiced = sum(j["invoiced"] for j in job_summaries)
    total_paid = sum(j["paid"] for j in job_summaries)
    total_expenses = sum(j["expenses"] for j in job_summaries)
    total_miles = sum(j["miles"] for j in job_summaries)
    total_deduction = round(total_miles * IRS_RATE, 2)
    net_profit = round(total_paid - total_expenses, 2)

    return {
        "user": me.full_name or me.email,
        "job_count": len(jobs),
        "jobs": job_summaries,
        "totals": {
            "invoiced": round(total_invoiced, 2),
            "paid": round(total_paid, 2),
            "outstanding": round(total_invoiced - total_paid, 2),
            "expenses": round(total_expenses, 2),
            "net_profit": net_profit,
            "total_miles": round(total_miles, 1),
            "mileage_deduction": total_deduction,
            "net_after_mileage": round(net_profit + total_deduction, 2),
        },
    }


def _call_claude(system_prompt: str, user_message: str) -> str:
    """Make a synchronous call to the Claude API and return the text response."""
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=503, detail="AI service not configured. Set ANTHROPIC_API_KEY on the server.")

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": CLAUDE_MODEL,
        "max_tokens": 1024,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
    }
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.post(ANTHROPIC_URL, headers=headers, json=body)
            r.raise_for_status()
            data = r.json()
            return data["content"][0]["text"]
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="AI response timed out. Please try again.")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"AI service error: {e.response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


SYSTEM_PROMPT = """You are JobFlow AI — a sharp, plain-spoken financial advisor for contractors, 
freelancers, and tradespeople. You analyze real business data and give direct, actionable advice.

Rules:
- Be direct and conversational. No fluff, no corporate jargon.
- Use dollar amounts and percentages from the data. Be specific.
- Give concrete action steps, not vague suggestions.
- Flag problems clearly: underpricng, cash flow risks, missed deductions.
- Be encouraging but honest. If a number is bad, say so and explain why.
- Keep responses focused and scannable. Use short paragraphs.
- Always respond in valid JSON matching the schema requested.
"""


class AIAnalysisOut(BaseModel):
    type: str
    headline: str
    summary: str
    insights: list[dict]
    actions: list[str]
    raw_data_used: dict


class AIProposalOut(BaseModel):
    subject: str
    body: str
    estimated_value: Optional[float] = None


class AIMessageOut(BaseModel):
    message: str
    tone: str


class AIForecastOut(BaseModel):
    headline: str
    projection_30d: float
    projection_90d: float
    confidence: str
    factors: list[str]
    risks: list[str]


# ─── /ai/analyze ───────────────────────────────────────────────────────────────
@app.get("/ai/analyze", response_model=AIAnalysisOut)
def ai_analyze(db: Session = Depends(get_db), me: User = Depends(get_current_user)):
    """Full business analysis: profit, hourly rate, cash flow, missed deductions."""
    data = _gather_user_data(db, me)

    prompt = f"""Analyze this contractor's business data and return ONLY valid JSON.

Business data:
{json.dumps(data, indent=2)}

Return this exact JSON structure (no markdown, no preamble):
{{
  "headline": "One punchy sentence summarizing their financial health",
  "summary": "2-3 sentences giving the overall picture. Be specific with numbers.",
  "insights": [
    {{
      "category": "Profit Analysis",
      "icon": "💰",
      "finding": "What you found about profit margins",
      "detail": "Specific numbers and what they mean",
      "severity": "good|warning|critical"
    }},
    {{
      "category": "Pricing Analysis",
      "icon": "💲",
      "finding": "Are they underpricing jobs?",
      "detail": "Which jobs look underpricied and by how much",
      "severity": "good|warning|critical"
    }},
    {{
      "category": "Cash Flow",
      "icon": "🔄",
      "finding": "Outstanding invoices situation",
      "detail": "How much is unpaid, risk level",
      "severity": "good|warning|critical"
    }},
    {{
      "category": "Tax Deductions",
      "icon": "📋",
      "finding": "Mileage and expense deduction summary",
      "detail": "Estimated tax savings from deductions tracked",
      "severity": "good|warning|critical"
    }},
    {{
      "category": "Best & Worst Jobs",
      "icon": "📊",
      "finding": "Which jobs made the most and least profit",
      "detail": "Name the specific jobs and their profit numbers",
      "severity": "good|warning|critical"
    }}
  ],
  "actions": [
    "Specific action item 1 with a dollar amount or deadline",
    "Specific action item 2",
    "Specific action item 3",
    "Specific action item 4"
  ]
}}"""

    text = _call_claude(SYSTEM_PROMPT, prompt)

    # Strip any accidental markdown code fences
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI returned malformed response. Please try again.")

    return AIAnalysisOut(
        type="full_analysis",
        headline=parsed.get("headline", "Analysis complete"),
        summary=parsed.get("summary", ""),
        insights=parsed.get("insights", []),
        actions=parsed.get("actions", []),
        raw_data_used=data["totals"],
    )


# ─── /ai/forecast ───────────────────────────────────────────────────────────────
@app.get("/ai/forecast", response_model=AIForecastOut)
def ai_forecast(db: Session = Depends(get_db), me: User = Depends(get_current_user)):
    """30 and 90 day income forecast based on current pipeline."""
    data = _gather_user_data(db, me)

    prompt = f"""Based on this contractor's data, forecast their income.

Business data:
{json.dumps(data, indent=2)}

Return ONLY this JSON (no markdown):
{{
  "headline": "Short forecast summary sentence",
  "projection_30d": <number: estimated income next 30 days>,
  "projection_90d": <number: estimated income next 90 days>,
  "confidence": "low|medium|high",
  "factors": [
    "Factor driving the forecast (positive)",
    "Another factor"
  ],
  "risks": [
    "Risk that could hurt income",
    "Another risk"
  ]
}}

Base projections on: outstanding invoices likely to be paid, active job pipeline, historical patterns."""

    text = _call_claude(SYSTEM_PROMPT, prompt)
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI returned malformed response.")

    return AIForecastOut(
        headline=parsed.get("headline", "Forecast ready"),
        projection_30d=float(parsed.get("projection_30d", 0)),
        projection_90d=float(parsed.get("projection_90d", 0)),
        confidence=parsed.get("confidence", "medium"),
        factors=parsed.get("factors", []),
        risks=parsed.get("risks", []),
    )


# ─── /ai/proposal ───────────────────────────────────────────────────────────────
class ProposalRequest(BaseModel):
    job_type: str
    client_name: Optional[str] = None
    estimated_hours: Optional[float] = None
    estimated_materials: Optional[float] = None
    notes: Optional[str] = None


@app.post("/ai/proposal", response_model=AIProposalOut)
def ai_proposal(req: ProposalRequest, db: Session = Depends(get_db), me: User = Depends(get_current_user)):
    """Generate a professional job proposal/estimate."""
    data = _gather_user_data(db, me)

    # Find avg job value to calibrate pricing
    avg_job = 0.0
    if data["jobs"]:
        avg_job = sum(j["invoiced"] for j in data["jobs"]) / len(data["jobs"])

    prompt = f"""Write a professional job proposal for a contractor.

Contractor's average job value: ${avg_job:.2f}
New job request:
- Job type: {req.job_type}
- Client: {req.client_name or "the client"}
- Estimated hours: {req.estimated_hours or "not specified"}
- Estimated materials: ${req.estimated_materials or 0:.2f}
- Additional notes: {req.notes or "none"}

Return ONLY this JSON:
{{
  "subject": "Proposal email subject line",
  "body": "Full professional proposal text. Include: greeting, scope of work, what's included, timeline estimate, pricing breakdown, payment terms, call to action. Keep it professional but conversational. Use the client's name if given.",
  "estimated_value": <number: suggested total price for this job>
}}"""

    text = _call_claude(SYSTEM_PROMPT, prompt)
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI returned malformed response.")

    return AIProposalOut(
        subject=parsed.get("subject", "Job Proposal"),
        body=parsed.get("body", ""),
        estimated_value=parsed.get("estimated_value"),
    )


# ─── /ai/followup ───────────────────────────────────────────────────────────────
class FollowUpRequest(BaseModel):
    invoice_id: int
    tone: Optional[str] = "professional"  # professional | firm | friendly


@app.post("/ai/followup", response_model=AIMessageOut)
def ai_followup(req: FollowUpRequest, db: Session = Depends(get_db), me: User = Depends(get_current_user)):
    """Generate a client follow-up message for an unpaid invoice."""
    inv = db.query(Invoice).filter(Invoice.id == req.invoice_id, Invoice.user_id == me.id).first()
    if not inv:
        raise HTTPException(status_code=404, detail="Invoice not found")

    job = db.query(Job).filter(Job.id == inv.job_id, Job.user_id == me.id).first() if inv.job_id else None
    client = job.client_name if job and job.client_name else "there"
    job_title = job.title if job else "the job"
    days_old = (datetime.utcnow() - inv.created_at.replace(tzinfo=None)).days if inv.created_at else 14

    prompt = f"""Write a client follow-up message for an unpaid invoice.

Details:
- Amount owed: ${float(inv.amount or 0):.2f}
- Client name: {client}
- Job: {job_title}
- Invoice note: {inv.note or "N/A"}
- Days since invoiced: {days_old}
- Requested tone: {req.tone}

Return ONLY this JSON:
{{
  "message": "The complete follow-up message text. Should be appropriate for the tone and days overdue. Include the amount, a clear ask to pay, and a way to follow up. No subject line needed.",
  "tone": "{req.tone}"
}}"""

    text = _call_claude(SYSTEM_PROMPT, prompt)
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI returned malformed response.")

    return AIMessageOut(
        message=parsed.get("message", ""),
        tone=parsed.get("tone", req.tone),
    )


# ─── /ai/rate-check ─────────────────────────────────────────────────────────────
@app.get("/ai/rate-check")
def ai_rate_check(db: Session = Depends(get_db), me: User = Depends(get_current_user)):
    """Analyze pricing across jobs and suggest rate adjustments."""
    data = _gather_user_data(db, me)

    prompt = f"""Analyze this contractor's job pricing and identify whether they're undercharging.

Business data:
{json.dumps(data, indent=2)}

Return ONLY this JSON:
{{
  "verdict": "underpriced|fairly_priced|well_priced",
  "headline": "One punchy sentence about their pricing",
  "avg_job_value": <number>,
  "suggested_avg_job_value": <number>,
  "potential_annual_increase": <number: how much more they'd make per year if they raised rates>,
  "underpriced_jobs": [
    {{
      "job_title": "name",
      "current_profit": <number>,
      "estimated_market_value": <number>,
      "reason": "Why this job seems underpriced"
    }}
  ],
  "advice": "2-3 sentences of specific, actionable pricing advice"
}}"""

    text = _call_claude(SYSTEM_PROMPT, prompt)
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI returned malformed response.")

=======
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from uuid import uuid4
from datetime import datetime

app = FastAPI(title="JobFlow API")

# Allow your phone app to call your API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JobStatus = Literal["Pending", "Working", "Completed"]

# -----------------------
# Models
# -----------------------
class JobCreate(BaseModel):
    title: str
    company: str
    status: JobStatus = "Pending"
    revenue: float = Field(default=0.0, ge=0)

class Job(JobCreate):
    id: str
    created_at: str
    updated_at: str

class ExpenseCreate(BaseModel):
    job_id: Optional[str] = None
    description: str
    amount: float = Field(ge=0)
    created_at: Optional[str] = None  # optional override

class Expense(ExpenseCreate):
    id: str
    created_at: str

class TravelCreate(BaseModel):
    job_id: Optional[str] = None
    from_location: str
    to_location: str
    miles: float = Field(ge=0)
    created_at: Optional[str] = None

class Travel(TravelCreate):
    id: str
    created_at: str

class Summary(BaseModel):
    total_jobs: int
    total_revenue: float
    total_expenses: float
    total_profit: float
    total_miles: float

# -----------------------
# In-memory storage
# -----------------------
jobs: List[Job] = []
expenses: List[Expense] = []
travels: List[Travel] = []

def now_iso() -> str:
    return datetime.utcnow().isoformat()

# -----------------------
# Health
# -----------------------
@app.get("/")
def root():
    return {"message": "JobFlow backend is running"}

# -----------------------
# Jobs
# -----------------------
@app.get("/jobs", response_model=List[Job])
def get_jobs():
    # newest first
    return sorted(jobs, key=lambda j: j.created_at, reverse=True)

@app.post("/jobs", response_model=Job)
def add_job(job: JobCreate):
    new_job = Job(
        id=str(uuid4()),
        title=job.title,
        company=job.company,
        status=job.status,
        revenue=job.revenue,
        created_at=now_iso(),
        updated_at=now_iso(),
    )
    jobs.append(new_job)
    return new_job

@app.put("/jobs/{job_id}", response_model=Job)
def update_job(job_id: str, updated: JobCreate):
    for i, j in enumerate(jobs):
        if j.id == job_id:
            jobs[i] = Job(
                id=j.id,
                title=updated.title,
                company=updated.company,
                status=updated.status,
                revenue=updated.revenue,
                created_at=j.created_at,
                updated_at=now_iso(),
            )
            return jobs[i]
    raise HTTPException(status_code=404, detail="Job not found")

@app.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    for i, j in enumerate(jobs):
        if j.id == job_id:
            deleted = jobs.pop(i)
            return {"deleted": deleted}
    raise HTTPException(status_code=404, detail="Job not found")

# -----------------------
# Expenses
# -----------------------
@app.get("/expenses", response_model=List[Expense])
def get_expenses(job_id: Optional[str] = None):
    if job_id:
        return [e for e in expenses if e.job_id == job_id]
    return sorted(expenses, key=lambda e: e.created_at, reverse=True)

@app.post("/expenses", response_model=Expense)
def add_expense(exp: ExpenseCreate):
    created = exp.created_at or now_iso()
    new_exp = Expense(
        id=str(uuid4()),
        job_id=exp.job_id,
        description=exp.description,
        amount=exp.amount,
        created_at=created,
    )
    expenses.append(new_exp)
    return new_exp

@app.delete("/expenses/{expense_id}")
def delete_expense(expense_id: str):
    for i, e in enumerate(expenses):
        if e.id == expense_id:
            deleted = expenses.pop(i)
            return {"deleted": deleted}
    raise HTTPException(status_code=404, detail="Expense not found")

# -----------------------
# Travel
# -----------------------
@app.get("/travel", response_model=List[Travel])
def get_travel(job_id: Optional[str] = None):
    if job_id:
        return [t for t in travels if t.job_id == job_id]
    return sorted(travels, key=lambda t: t.created_at, reverse=True)

@app.post("/travel", response_model=Travel)
def add_travel(tr: TravelCreate):
    created = tr.created_at or now_iso()
    new_tr = Travel(
        id=str(uuid4()),
        job_id=tr.job_id,
        from_location=tr.from_location,
        to_location=tr.to_location,
        miles=tr.miles,
        created_at=created,
    )
    travels.append(new_tr)
    return new_tr

@app.delete("/travel/{travel_id}")
def delete_travel(travel_id: str):
    for i, t in enumerate(travels):
        if t.id == travel_id:
            deleted = travels.pop(i)
            return {"deleted": deleted}
    raise HTTPException(status_code=404, detail="Travel not found")

# -----------------------
# Summary
# -----------------------
@app.get("/summary", response_model=Summary)
def get_summary():
    total_revenue = sum(j.revenue for j in jobs)
    total_expenses = sum(e.amount for e in expenses)
    total_profit = total_revenue - total_expenses
    total_miles = sum(t.miles for t in travels)

    return Summary(
        total_jobs=len(jobs),
        total_revenue=round(total_revenue, 2),
        total_expenses=round(total_expenses, 2),
        total_profit=round(total_profit, 2),
        total_miles=round(total_miles, 2),
    )
>>>>>>> 70de87d9c82f50ccf8e2da2234c721fcdee3c164
