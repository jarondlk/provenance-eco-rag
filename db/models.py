"""
SQLAlchemy ORM models for all canonical tables, retrieval documents,
and cross-source links.

Uses pgvector for embedding storage and PostgreSQL tsvector for
full-text search.
"""
from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import DeclarativeBase

try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    Vector = None  # graceful fallback if pgvector not installed yet

import config


class Base(DeclarativeBase):
    pass


# -----------------------------------------------------------------------
# Layer 1: Provenance
# -----------------------------------------------------------------------
class ProvenanceRecord(Base):
    __tablename__ = "provenance_record"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_dataset = Column(String(64), nullable=False, index=True)
    source_file = Column(Text, nullable=False)
    sha256 = Column(String(64), nullable=False, unique=True)
    file_size_bytes = Column(Integer)
    ingested_at = Column(DateTime(timezone=True), server_default=func.now())
    processing_run = Column(String(64), index=True)
    notes = Column(Text)


# -----------------------------------------------------------------------
# Layer 3: Anchor events
# -----------------------------------------------------------------------
class AnchorEvent(Base):
    __tablename__ = "anchor_event"

    event_id = Column(String(128), primary_key=True)
    time_start = Column(String(32), index=True)
    time_end = Column(String(32))
    lat = Column(Float)
    lon = Column(Float)
    depth_min = Column(Float)
    depth_max = Column(Float)
    station_id = Column(String(32), index=True)
    sample_id = Column(String(64), index=True)
    bay_code = Column(String(4), index=True)
    source_types = Column(String(128))


# -----------------------------------------------------------------------
# Layer 3: CTD tables
# -----------------------------------------------------------------------
class CtdProfile(Base):
    __tablename__ = "ctd_profile"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sample_id = Column(String(64), nullable=False, index=True)
    ctd_date = Column(DateTime)
    depth_m = Column(Float)
    temperature = Column(Float)
    salinity = Column(Float)
    sigma_t = Column(Float)
    chl_a = Column(Float)
    chl_flu = Column(Float)
    do_percent = Column(Float)
    do_mg_l = Column(Float)
    turbidity = Column(Float)
    ec = Column(Float)
    ec25 = Column(Float)
    density = Column(Float)
    voltage = Column(Float)
    orp = Column(Float)
    ph = Column(Float)
    par = Column(Float)


class CtdSummary(Base):
    __tablename__ = "ctd_summary"

    sample_id = Column(String(64), primary_key=True)
    ctd_date = Column(DateTime)
    n_depth_points = Column(Integer)
    min_depth_m = Column(Float)
    max_depth_m = Column(Float)
    surface_temperature = Column(Float)
    bottom_temperature = Column(Float)
    mean_temperature = Column(Float)
    surface_salinity = Column(Float)
    bottom_salinity = Column(Float)
    mean_salinity = Column(Float)
    surface_do_percent = Column(Float)
    bottom_do_percent = Column(Float)
    mean_do_percent = Column(Float)
    surface_chl_a = Column(Float)
    bottom_chl_a = Column(Float)
    mean_chl_a = Column(Float)


# -----------------------------------------------------------------------
# Layer 3: Metagenome tables
# -----------------------------------------------------------------------
class MetagenomeSample(Base):
    __tablename__ = "metagenome_sample"

    sample_id = Column(String(64), primary_key=True)
    bay = Column(String(4), index=True)
    station_code = Column(String(16))
    sample_year_month = Column(String(8))
    n_runs = Column(Integer)
    first_run_date = Column(DateTime)
    last_run_date = Column(DateTime)
    sum_reads_gt1kb = Column(Float)
    sum_bases_gt1kb = Column(Float)
    has_kraken = Column(Boolean)
    has_metaeuk = Column(Boolean)
    has_ctd = Column(Boolean)
    top_kraken_genera_json = Column(Text)
    top_metaeuk_genera_json = Column(Text)
    top_upper_groups_json = Column(Text)


# -----------------------------------------------------------------------
# Layer 3: Remote sensing tables
# -----------------------------------------------------------------------
class SstPointObservation(Base):
    __tablename__ = "sst_point_observation"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file = Column(String(128))
    time_utc = Column(DateTime, index=True)
    time_jst = Column(DateTime)
    sst = Column(Float)
    nearest_lat = Column(Float)
    nearest_lon = Column(Float)


class SstDailySummary(Base):
    __tablename__ = "sst_daily_summary"

    date_jst = Column(String(16), primary_key=True)
    mean_sst = Column(Float)
    min_sst = Column(Float)
    max_sst = Column(Float)
    std_sst = Column(Float)
    n_files = Column(Integer)


# -----------------------------------------------------------------------
# Layer 4: Retrieval documents
# -----------------------------------------------------------------------
class RetrievalDocument(Base):
    __tablename__ = "retrieval_document"

    doc_id = Column(String(128), primary_key=True)
    source_type = Column(String(32), nullable=False, index=True)
    sample_id = Column(String(64), index=True)
    event_id = Column(String(128), index=True)
    time = Column(String(32), index=True)
    lat = Column(Float)
    lon = Column(Float)
    bay = Column(String(4), index=True)
    station = Column(String(32))
    title = Column(Text, nullable=False)
    text = Column(Text, nullable=False)
    text_tsv = Column(TSVECTOR)  # full-text search vector

    if Vector is not None:
        embedding = Column(Vector(config.EMBEDDING_DIM))

    __table_args__ = (
        Index("ix_retrieval_doc_text_fts", "text_tsv", postgresql_using="gin"),
    )


# -----------------------------------------------------------------------
# Layer 4: Cross-source links
# -----------------------------------------------------------------------
class CrossSourceLink(Base):
    __tablename__ = "cross_source_link"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_event_id = Column(String(128), nullable=False, index=True)
    target_event_id = Column(String(128), nullable=False, index=True)
    link_type = Column(String(32), nullable=False)
    distance_km = Column(Float)
    time_delta_days = Column(Float)
