SCHEMA_SQL = """
DO $$
BEGIN
    IF to_regclass('public.evaluation_result') IS NOT NULL THEN
        RAISE EXCEPTION
            'legacy evaluation_result schema detected; create a fresh Scoreboard database';
    END IF;
    IF to_regclass('public.evaluation_campaign') IS NOT NULL
       AND to_regclass('public.evaluation_schema_metadata') IS NULL THEN
        RAISE EXCEPTION
            'unversioned evaluation schema detected; create a fresh Scoreboard database';
    END IF;
END
$$;

CREATE TABLE IF NOT EXISTS evaluation_schema_metadata (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    contract_version integer NOT NULL
);
INSERT INTO evaluation_schema_metadata (singleton, contract_version)
VALUES (true, 4)
ON CONFLICT (singleton) DO NOTHING;
UPDATE evaluation_schema_metadata
SET contract_version = 4
WHERE singleton = true AND contract_version = 3;
DO $$
BEGIN
    IF (
        SELECT contract_version
        FROM evaluation_schema_metadata
        WHERE singleton = true
    ) <> 4 THEN
        RAISE EXCEPTION
            'unsupported evaluation schema version; create a fresh Scoreboard database';
    END IF;
END
$$;

CREATE TABLE IF NOT EXISTS evaluation_campaign (
    id uuid PRIMARY KEY,
    run_key text NOT NULL CHECK (run_key ~ '^[0-9a-f]{64}$'),
    status text NOT NULL CHECK (status IN ('incomplete', 'complete')),
    config_digest text NOT NULL CHECK (config_digest ~ '^[0-9a-f]{64}$'),
    registry_digest text NOT NULL CHECK (registry_digest ~ '^[0-9a-f]{64}$'),
    eval_contract_digest text NOT NULL CHECK (eval_contract_digest ~ '^[0-9a-f]{64}$'),
    lighteval_version text NOT NULL,
    evaluator_name text NOT NULL DEFAULT 'lighteval'
        CONSTRAINT evaluation_campaign_evaluator_name_check
        CHECK (evaluator_name IN ('lighteval', 'lm-eval')),
    configured_selectors jsonb NOT NULL,
    resolved_selectors jsonb NOT NULL,
    skipped_selectors jsonb NOT NULL,
    expected_tasks jsonb NOT NULL,
    publisher_principal text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz,
    CHECK (
        (status = 'incomplete' AND completed_at IS NULL)
        OR (status = 'complete' AND completed_at IS NOT NULL)
    )
);

ALTER TABLE evaluation_campaign
    ADD COLUMN IF NOT EXISTS evaluator_name text NOT NULL DEFAULT 'lighteval';
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid = 'evaluation_campaign'::regclass
          AND conname = 'evaluation_campaign_evaluator_name_check'
    ) THEN
        ALTER TABLE evaluation_campaign
            ADD CONSTRAINT evaluation_campaign_evaluator_name_check
            CHECK (evaluator_name IN ('lighteval', 'lm-eval'));
    END IF;
END
$$;

CREATE UNIQUE INDEX IF NOT EXISTS evaluation_campaign_run_key_idx
    ON evaluation_campaign(run_key);

CREATE TABLE IF NOT EXISTS evaluation_task (
    id uuid PRIMARY KEY,
    campaign_id uuid NOT NULL REFERENCES evaluation_campaign(id) ON DELETE CASCADE,
    task_identity text NOT NULL,
    content_digest text NOT NULL CHECK (content_digest ~ '^[0-9a-f]{64}$'),
    task jsonb NOT NULL,
    artifact jsonb NOT NULL,
    task_config jsonb NOT NULL,
    model jsonb NOT NULL,
    sampling_config jsonb NOT NULL,
    primary_metric text NOT NULL,
    aggregates jsonb NOT NULL,
    diagnostics jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (campaign_id, task_identity)
);

CREATE TABLE IF NOT EXISTS evaluation_sample (
    evaluation_id uuid NOT NULL REFERENCES evaluation_task(id) ON DELETE CASCADE,
    sample_index integer NOT NULL CHECK (sample_index >= 0),
    document_index integer NOT NULL CHECK (document_index >= 0),
    outcome text NOT NULL CHECK (
        outcome IN ('correct', 'incorrect', 'unanswered', 'undetermined')
    ),
    doc jsonb NOT NULL,
    metric jsonb NOT NULL,
    model_response jsonb NOT NULL,
    PRIMARY KEY (evaluation_id, sample_index)
);

CREATE INDEX IF NOT EXISTS evaluation_campaign_completed_idx
    ON evaluation_campaign(completed_at DESC)
    WHERE status = 'complete';
CREATE INDEX IF NOT EXISTS evaluation_task_created_idx
    ON evaluation_task(created_at DESC);
CREATE INDEX IF NOT EXISTS evaluation_sample_page_idx
    ON evaluation_sample(evaluation_id, sample_index);
CREATE INDEX IF NOT EXISTS evaluation_sample_document_idx
    ON evaluation_sample(evaluation_id, document_index);
"""
