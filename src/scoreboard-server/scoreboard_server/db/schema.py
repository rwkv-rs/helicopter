from __future__ import annotations

from tortoise import Tortoise


SCHEMA_SQL = """
CREATE INDEX IF NOT EXISTS idx_task_model ON task(model_id);
CREATE INDEX IF NOT EXISTS idx_task_benchmark ON task(benchmark_id);
CREATE INDEX IF NOT EXISTS idx_task_is_tmp_created_at ON task(is_tmp, created_at);
CREATE INDEX IF NOT EXISTS idx_task_status_created_at ON task(status, created_at);
CREATE INDEX IF NOT EXISTS idx_task_identity_lookup ON task(model_id, benchmark_id, evaluator, git_hash, config_path);
CREATE INDEX IF NOT EXISTS idx_completions_task ON completions(task_id);
CREATE INDEX IF NOT EXISTS idx_eval_completion ON eval(completions_id);
CREATE INDEX IF NOT EXISTS idx_checker_completion ON checker(completions_id);
CREATE INDEX IF NOT EXISTS idx_checker_needs_human_review ON checker(needs_human_review);
CREATE INDEX IF NOT EXISTS idx_scores_task ON scores(task_id);
CREATE INDEX IF NOT EXISTS idx_scheduler_lease_owner ON scheduler_lease(owner_id);
CREATE INDEX IF NOT EXISTS idx_scheduler_lease_until ON scheduler_lease(lease_until);
CREATE INDEX IF NOT EXISTS idx_scheduler_lease_node ON scheduler_lease(node_id);

CREATE OR REPLACE VIEW view_model_version AS
SELECT
    model_id,
    model_name,
    arch_version,
    data_version,
    num_params,
    concat_ws('_', arch_version, data_version, num_params) AS model_version
FROM model;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_benchmark_status') THEN
        ALTER TABLE benchmark ADD CONSTRAINT chk_benchmark_status
            CHECK (status IN ('Todo', 'Buggy', 'Low', 'DataSynthesizing', 'Completed'));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_task_status') THEN
        ALTER TABLE task ADD CONSTRAINT chk_task_status CHECK (status IN ('Running', 'Completed', 'Failed'));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_completions_sample_index') THEN
        ALTER TABLE completions ADD CONSTRAINT chk_completions_sample_index CHECK (sample_index >= 0);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_completions_avg_repeat_index') THEN
        ALTER TABLE completions ADD CONSTRAINT chk_completions_avg_repeat_index CHECK (avg_repeat_index >= 0);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_completions_pass_index') THEN
        ALTER TABLE completions ADD CONSTRAINT chk_completions_pass_index CHECK (pass_index >= 0);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_completions_status') THEN
        ALTER TABLE completions ADD CONSTRAINT chk_completions_status CHECK (status IN ('Running', 'Completed', 'Failed'));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_eval_answer_size') THEN
        ALTER TABLE eval ADD CONSTRAINT chk_eval_answer_size CHECK (char_length(answer) <= 65536);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_eval_ref_answer_size') THEN
        ALTER TABLE eval ADD CONSTRAINT chk_eval_ref_answer_size CHECK (char_length(ref_answer) <= 4096);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_eval_fail_reason_size') THEN
        ALTER TABLE eval ADD CONSTRAINT chk_eval_fail_reason_size CHECK (char_length(fail_reason) <= 2048);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_scores_cot_mode') THEN
        ALTER TABLE scores ADD CONSTRAINT chk_scores_cot_mode CHECK (cot_mode IN ('NoCoT', 'CoT'));
    END IF;
END $$;
"""


async def apply_schema_sql() -> None:
    connection = Tortoise.get_connection("default")
    await connection.execute_script(SCHEMA_SQL)
