# backend/models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship
from sqlalchemy import event
from sqlalchemy.engine import Engine
import sqlite3

from db import Base


# ✅ SQLite must have foreign keys enabled for ON DELETE CASCADE to work
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    if isinstance(dbapi_connection, sqlite3.Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.close()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # RevenueCat subscription fields
    revenuecat_id = Column(String, nullable=True, index=True)
    subscription_tier = Column(String, nullable=False, default="free", server_default="free")

    jobs = relationship("Job", back_populates="user", cascade="all, delete-orphan", passive_deletes=True)


class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    title = Column(String, nullable=False)
    client_name = Column(String, nullable=True)
    status = Column(String, nullable=True)
    start_date = Column(String, nullable=True)
    end_date = Column(String, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="jobs")

    invoices = relationship(
        "Invoice",
        back_populates="job",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    expenses = relationship(
        "Expense",
        back_populates="job",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    mileage_entries = relationship(
        "Mileage",
        back_populates="job",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Invoice(Base):
    __tablename__ = "invoices"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    # ✅ cascade delete when job deleted
    job_id = Column(Integer, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=True, index=True)

    amount = Column(Float, nullable=False, default=0)
    note = Column(String, nullable=True)
    status = Column(String, nullable=True, default="unpaid")

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    job = relationship("Job", back_populates="invoices")


class Expense(Base):
    __tablename__ = "expenses"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    # ✅ cascade delete when job deleted
    job_id = Column(Integer, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=True, index=True)

    amount = Column(Float, nullable=False, default=0)
    category = Column(String, nullable=False, default="Other")
    note = Column(String, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    job = relationship("Job", back_populates="expenses")


class Mileage(Base):
    __tablename__ = "mileage"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    # ✅ cascade delete when job deleted
    job_id = Column(Integer, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=True, index=True)

    miles = Column(Float, nullable=False, default=0)
    note = Column(String, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    job = relationship("Job", back_populates="mileage_entries")
