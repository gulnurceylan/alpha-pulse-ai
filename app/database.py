from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/alphapulse"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class AnalysisReport(Base):
    __tablename__ = "analysis_reports"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), index=True, nullable=False)  # e.g., AAPL, THYAO.IS
    status = Column(String(20), default="PENDING", nullable=False)  # PENDING, COMPLETED, FAILED
    report = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
    print("✅ PostgreSQL connection successful and 'analysis_reports' table is ready!")