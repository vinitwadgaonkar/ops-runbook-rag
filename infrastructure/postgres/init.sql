-- Initialize pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create research database schema
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    metadata JSONB NOT NULL,
    document_type VARCHAR(50) NOT NULL,
    source_path TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Vector chunks table with pgvector support
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(3072),  -- OpenAI text-embedding-3-large
    metadata JSONB NOT NULL,
    chunk_index INTEGER,
    ts_vector tsvector,  -- For full-text search
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for vector similarity search
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS chunks_tsvector_idx ON chunks USING GIN (ts_vector);
CREATE INDEX IF NOT EXISTS chunks_metadata_idx ON chunks USING GIN (metadata);
CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);

-- Query history for feedback loop
CREATE TABLE IF NOT EXISTS queries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_text TEXT NOT NULL,
    context JSONB,
    retrieved_chunks UUID[],
    llm_response TEXT,
    latency_ms INTEGER,
    feedback_score INTEGER CHECK (feedback_score >= 1 AND feedback_score <= 5),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Research evaluation metrics
CREATE TABLE IF NOT EXISTS evaluation_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id UUID REFERENCES queries(id),
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    evaluation_type VARCHAR(50) NOT NULL,  -- 'automated' or 'human'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Ingestion job tracking
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    source_path TEXT,
    metadata JSONB,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Create function to update ts_vector automatically
CREATE OR REPLACE FUNCTION update_ts_vector() RETURNS TRIGGER AS $$
BEGIN
    NEW.ts_vector := to_tsvector('english', NEW.content);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for automatic ts_vector updates
CREATE TRIGGER update_chunks_ts_vector
    BEFORE INSERT OR UPDATE ON chunks
    FOR EACH ROW
    EXECUTE FUNCTION update_ts_vector();

-- Create function for temporal decay scoring
CREATE OR REPLACE FUNCTION temporal_decay_score(
    base_score FLOAT,
    days_old INTEGER,
    decay_lambda FLOAT DEFAULT 0.05
) RETURNS FLOAT AS $$
BEGIN
    RETURN base_score * exp(-decay_lambda * days_old);
END;
$$ LANGUAGE plpgsql;

-- Create view for research analytics
CREATE OR REPLACE VIEW retrieval_analytics AS
SELECT 
    DATE_TRUNC('day', q.created_at) as date,
    COUNT(*) as total_queries,
    AVG(q.latency_ms) as avg_latency_ms,
    AVG(q.feedback_score) as avg_feedback_score,
    COUNT(CASE WHEN q.feedback_score >= 4 THEN 1 END) as high_rating_count
FROM queries q
GROUP BY DATE_TRUNC('day', q.created_at)
ORDER BY date DESC;

-- Insert sample metadata for research
INSERT INTO documents (content, metadata, document_type, source_path) VALUES
('Sample runbook content', '{"service": "api-gateway", "severity": "critical", "component": "ingress"}', 'runbook', 'sample_data/runbooks/api-gateway.md'),
('Sample KB article', '{"service": "database", "severity": "medium", "component": "postgres"}', 'kb_article', 'sample_data/kb_articles/database-tuning.json')
ON CONFLICT DO NOTHING;
