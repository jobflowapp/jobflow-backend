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