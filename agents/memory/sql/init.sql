-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- ─────────────────────────────────────────
-- 1. IMMUTABLE LOG  (never delete, never update)
-- ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS memory_logs (
    id          SERIAL PRIMARY KEY,
    speaker     TEXT NOT NULL DEFAULT 'user',
    raw_text    TEXT NOT NULL,
    emotion     TEXT,                          -- from video/scene pipeline
    scene       TEXT,                          -- scene context
    metadata    JSONB DEFAULT '{}',            -- any extra multimodal data
    timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ─────────────────────────────────────────
-- 2. STATE MEMORY  (versioned, last-value-wins)
-- ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS state_memory (
    id          SERIAL PRIMARY KEY,
    entity      TEXT NOT NULL,                 -- 'neighbour', 'aleena', 'user'
    attribute   TEXT NOT NULL,                 -- 'relationship', 'age', 'mood'
    value       TEXT NOT NULL,
    confidence  FLOAT NOT NULL DEFAULT 1.0,
    emotion     TEXT,                          -- emotion when this was stated
    valid_from  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_to    TIMESTAMPTZ                    -- NULL = currently active
);

CREATE INDEX IF NOT EXISTS idx_state_active
    ON state_memory(entity, attribute)
    WHERE valid_to IS NULL;

-- ─────────────────────────────────────────
-- 3. BELIEF MEMORY  (subjective, versioned per observer+event)
-- ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS belief_memory (
    id              SERIAL PRIMARY KEY,
    observer        TEXT NOT NULL DEFAULT 'user',
    entity_or_event TEXT NOT NULL,             -- 'david_goal_123', 'project_launch'
    attribute       TEXT NOT NULL,             -- 'feeling_about', 'facial_expression'
    value           TEXT NOT NULL,
    confidence      FLOAT NOT NULL DEFAULT 0.8,
    valid_from      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_to        TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_belief_active
    ON belief_memory(observer, entity_or_event, attribute)
    WHERE valid_to IS NULL;

-- ─────────────────────────────────────────
-- 4. EVENT MEMORY  (immutable, things that happened)
-- ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS event_memory (
    id          SERIAL PRIMARY KEY,
    event_type  TEXT NOT NULL,                 -- 'meeting', 'argument', 'achievement'
    actor       TEXT,                          -- who did it
    description TEXT NOT NULL,
    emotion     TEXT,                          -- emotion at time of event
    scene       TEXT,
    embedding   VECTOR(768),                    -- nomic-embed-text embeddings
    timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_event_embedding
    ON event_memory USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- ─────────────────────────────────────────
-- 5. TOPIC FREQUENCY  (deterministic priority tracking)
-- ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS topic_frequency (
    topic       TEXT PRIMARY KEY,
    count       INTEGER NOT NULL DEFAULT 1,
    last_seen   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);