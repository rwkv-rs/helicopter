SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS evaluation_result (
    id uuid PRIMARY KEY,
    publication_id text NOT NULL UNIQUE,
    content_digest text NOT NULL CHECK (content_digest ~ '^[0-9a-f]{64}$'),
    source_run_id text NOT NULL,
    source text NOT NULL,
    visibility text NOT NULL CHECK (visibility = 'non_official'),
    task_name text NOT NULL,
    task_config jsonb NOT NULL,
    artifact jsonb NOT NULL,
    model jsonb NOT NULL,
    benchmark jsonb NOT NULL,
    evaluation jsonb NOT NULL,
    comparisons jsonb NOT NULL,
    sampling_config jsonb NOT NULL,
    primary_metric text NOT NULL,
    aggregates jsonb NOT NULL,
    diagnostics jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS evaluation_sample (
    evaluation_id uuid NOT NULL REFERENCES evaluation_result(id) ON DELETE CASCADE,
    sample_index integer NOT NULL CHECK (sample_index >= 0),
    outcome text NOT NULL CHECK (
        outcome IN ('correct', 'incorrect', 'unanswered', 'undetermined')
    ),
    doc jsonb NOT NULL,
    metric jsonb NOT NULL,
    model_response jsonb NOT NULL,
    PRIMARY KEY (evaluation_id, sample_index)
);

CREATE INDEX IF NOT EXISTS evaluation_result_created_at_idx
    ON evaluation_result(created_at DESC);
CREATE INDEX IF NOT EXISTS evaluation_result_source_run_idx
    ON evaluation_result(source_run_id);
CREATE INDEX IF NOT EXISTS evaluation_sample_outcome_idx
    ON evaluation_sample(evaluation_id, outcome, sample_index);
"""
