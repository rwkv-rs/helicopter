from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_eval_has_one_product_entrypoint_and_no_public_task_selection() -> None:
    source = ROOT / "src/cli/helicopter_eval"
    assert source.is_dir()
    assert not (ROOT / "src/eval/lighteval/evaluate.py").exists()
    assert not list(ROOT.glob("src/**/lighteval/**/evaluate.py"))

    config = (source / "config.py").read_text(encoding="utf-8")
    assert 'frozenset({"schema_version", "weights"})' in config
    example = (ROOT / "configs/eval/lighteval.toml").read_text(encoding="utf-8")
    assert "schema_version = 1" in example
    assert "weights = [" in example
    assert example.count('"rwkv7/pth/') == 2
    for key in ("benchmarks =", "tasks =", "exclude =", "max_samples ="):
        assert key not in example


def test_eval_uses_http_publication_and_removed_classification_is_not_product_data() -> (
    None
):
    eval_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "src/cli/helicopter_eval").glob("*.py")
    )
    assert "asyncpg" not in eval_source
    assert "Content-Encoding" in eval_source
    assert "Idempotency-Key" in eval_source

    product_paths = [
        ROOT / "src/cli/helicopter_eval",
        ROOT / "src/scoreboard-server/scoreboard_server",
        ROOT / "src/scoreboard-client/app",
        ROOT / "src/scoreboard-client/components",
        ROOT / "src/scoreboard-client/lib",
    ]
    product_source = "\n".join(
        file.read_text(encoding="utf-8")
        for directory in product_paths
        for file in directory.rglob("*")
        if file.suffix in {".py", ".ts", ".tsx", ".css"}
    )
    for field in (
        "offi" + "cial",
        "non_" + "official",
        "trus" + "ted",
        "trust_" + "level",
        "visi" + "bility",
        "eligi" + "bility",
        "sani" + "ty",
        "compa" + "rable",
        "result_" + "level",
    ):
        assert field not in product_source
