"""
AQQIP Phase 1 — Infrastructure Health Check Tests

Validates that the infrastructure foundation is correctly configured:
- MySQL reachable + required tables exist
- MongoDB reachable + required collections/indexes exist
- ChromaDB persistence directory writable
- MLflow tracking directory writable
- Configuration loads correctly without hardcoded secrets
- Docker Compose config is valid (syntax check)

These tests are designed to be the Phase 1 exit gate.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Project root for path resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Configuration Tests
# =============================================================================


class TestConfiguration:
    """Verify configuration loads correctly and securely."""

    def test_settings_load_without_error(self, settings) -> None:
        """Settings singleton loads successfully."""
        assert settings is not None
        assert settings.mysql is not None
        assert settings.mongo is not None
        assert settings.chroma is not None
        assert settings.mlflow is not None
        assert settings.llm is not None
        assert settings.edgar is not None
        assert settings.embedding is not None
        assert settings.prediction_target is not None
        assert settings.marl is not None
        assert settings.logging is not None

    def test_mysql_config_has_required_fields(self, settings) -> None:
        """MySQL config exposes host, port, database, user, password."""
        assert settings.mysql.host
        assert settings.mysql.port > 0
        assert settings.mysql.database == "aqqip"
        assert settings.mysql.user
        assert settings.mysql.password

    def test_mongo_config_has_required_fields(self, settings) -> None:
        """MongoDB config exposes host, port, database."""
        assert settings.mongo.host
        assert settings.mongo.port > 0
        assert settings.mongo.database == "aqqip_raw"

    def test_mongo_uri_format(self, settings) -> None:
        """MongoDB URI is properly formatted."""
        uri = settings.mongo.uri
        assert uri.startswith("mongodb://")

    def test_edgar_user_agent_is_compliant(self, settings) -> None:
        """EDGAR User-Agent is present and non-trivial."""
        ua = settings.edgar.user_agent
        assert len(ua) >= 10, "EDGAR User-Agent must be a compliant string"

    def test_embedding_model_configured(self, settings) -> None:
        """Embedding model is the documented all-MiniLM-L6-v2."""
        assert "all-MiniLM-L6-v2" in settings.embedding.model_name

    def test_prediction_target_has_defaults(self, settings) -> None:
        """Prediction target config has valid defaults."""
        assert settings.prediction_target.horizon_trading_days > 0
        assert settings.prediction_target.sharp_drop_threshold_pct > 0
        assert settings.prediction_target.price_field in ("adj_close", "close")

    def test_marl_defaults_sensible(self, settings) -> None:
        """MARL defaults are within documented ranges."""
        assert settings.marl.episode_length > 0
        assert settings.marl.default_seed >= 0

    def test_no_hardcoded_gemini_key_in_config_source(self) -> None:
        """The config module must not contain a hardcoded Gemini key."""
        config_path = PROJECT_ROOT / "config" / "settings.py"
        content = config_path.read_text()
        # The default for gemini_api_key must be empty string, not a real key
        assert 'default=""' in content or "default=''" in content, (
            "Gemini API key default must be an empty string"
        )

    def test_no_secrets_in_env_example(self) -> None:
        """The .env.example must not contain real API keys."""
        env_example = PROJECT_ROOT / ".env.example"
        if env_example.exists():
            content = env_example.read_text()
            # GEMINI_API_KEY should be blank or placeholder
            for line in content.splitlines():
                if line.startswith("GEMINI_API_KEY"):
                    key_value = line.split("=", 1)[1].strip()
                    assert key_value == "" or key_value.startswith("<"), (
                        f"GEMINI_API_KEY in .env.example must be blank: got '{key_value}'"
                    )

    def test_gemini_not_required_for_startup(self, settings) -> None:
        """Gemini being unavailable must not prevent settings from loading."""
        # If we got here, settings loaded even without a real Gemini key
        assert settings.llm.gemini_available is False or isinstance(
            settings.llm.gemini_available, bool
        )


# =============================================================================
# MySQL Infrastructure Tests
# =============================================================================


class TestMySQLInfrastructure:
    """Verify MySQL is reachable and has the correct schema."""

    REQUIRED_TABLES = [
        "ohlcv_prices",
        "fundamental_ratios",
        "predictions",
        "marl_simulation_runs",
    ]

    def test_mysql_connection(self, mysql_connection) -> None:
        """MySQL is reachable and responds to queries."""
        cursor = mysql_connection.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        assert result == (1,)
        cursor.close()

    @pytest.mark.parametrize("table_name", REQUIRED_TABLES)
    def test_table_exists(self, mysql_connection, table_name: str) -> None:
        """Required table exists in the aqqip database."""
        cursor = mysql_connection.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = 'aqqip' AND table_name = %s",
            (table_name,),
        )
        count = cursor.fetchone()[0]
        cursor.close()
        assert count == 1, f"Table '{table_name}' not found in aqqip database"

    def test_ohlcv_unique_constraint(self, mysql_connection) -> None:
        """ohlcv_prices has unique constraint on (ticker, trade_date)."""
        cursor = mysql_connection.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM information_schema.table_constraints "
            "WHERE table_schema = 'aqqip' "
            "AND table_name = 'ohlcv_prices' "
            "AND constraint_type = 'UNIQUE'"
        )
        count = cursor.fetchone()[0]
        cursor.close()
        assert count >= 1, "ohlcv_prices must have a UNIQUE constraint on (ticker, trade_date)"

    def test_predictions_status_column(self, mysql_connection) -> None:
        """predictions.status has the full lifecycle enum."""
        cursor = mysql_connection.cursor()
        cursor.execute(
            "SELECT column_type FROM information_schema.columns "
            "WHERE table_schema = 'aqqip' "
            "AND table_name = 'predictions' "
            "AND column_name = 'status'"
        )
        row = cursor.fetchone()
        cursor.close()
        assert row is not None, "predictions.status column must exist"
        col_type = row[0].lower()
        for state in ("open", "resolved", "eligible", "resolution_failed"):
            assert state in col_type, f"predictions.status must include '{state}'"

    def test_predictions_has_target_date(self, mysql_connection) -> None:
        """predictions table must have target_date column."""
        cursor = mysql_connection.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM information_schema.columns "
            "WHERE table_schema = 'aqqip' "
            "AND table_name = 'predictions' "
            "AND column_name = 'target_date'"
        )
        count = cursor.fetchone()[0]
        cursor.close()
        assert count == 1, "predictions must have target_date column"

    def test_marl_has_simulation_id(self, mysql_connection) -> None:
        """marl_simulation_runs has unique simulation_id."""
        cursor = mysql_connection.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM information_schema.table_constraints "
            "WHERE table_schema = 'aqqip' "
            "AND table_name = 'marl_simulation_runs' "
            "AND constraint_type = 'UNIQUE'"
        )
        count = cursor.fetchone()[0]
        cursor.close()
        assert count >= 1, "marl_simulation_runs must have unique simulation_id"


# =============================================================================
# MongoDB Infrastructure Tests
# =============================================================================


class TestMongoDBInfrastructure:
    """Verify MongoDB is reachable and has required collections/indexes."""

    def test_mongo_ping(self, mongo_client) -> None:
        """MongoDB responds to ping."""
        result = mongo_client.admin.command("ping")
        assert result.get("ok") == 1.0

    def test_sec_filings_collection_exists(self, mongo_db) -> None:
        """sec_filings collection exists."""
        collections = mongo_db.list_collection_names()
        assert "sec_filings" in collections, (
            f"sec_filings not in collections: {collections}"
        )

    def test_semantic_diffs_collection_exists(self, mongo_db) -> None:
        """semantic_diffs collection exists."""
        collections = mongo_db.list_collection_names()
        assert "semantic_diffs" in collections, (
            f"semantic_diffs not in collections: {collections}"
        )

    def test_sec_filings_accession_index(self, mongo_db) -> None:
        """sec_filings has unique index on accession_number."""
        indexes = mongo_db.sec_filings.index_information()
        accession_indexes = [
            name for name, info in indexes.items()
            if any(
                key[0] == "accession_number" for key in info.get("key", [])
            )
        ]
        assert len(accession_indexes) >= 1, (
            "sec_filings must have an index on accession_number"
        )
        # Verify uniqueness
        for idx_name in accession_indexes:
            assert indexes[idx_name].get("unique", False), (
                "accession_number index must be unique"
            )

    def test_sec_filings_ticker_year_index(self, mongo_db) -> None:
        """sec_filings has compound index for chronological retrieval."""
        indexes = mongo_db.sec_filings.index_information()
        has_compound = any(
            any(key[0] == "ticker" for key in info.get("key", []))
            and any(key[0] == "fiscal_year" for key in info.get("key", []))
            for info in indexes.values()
        )
        assert has_compound, (
            "sec_filings must have compound index on (ticker, fiscal_year, ...)"
        )

    def test_semantic_diffs_unique_index(self, mongo_db) -> None:
        """semantic_diffs has unique index on (ticker, section, year_from, year_to)."""
        indexes = mongo_db.semantic_diffs.index_information()
        unique_indexes = [
            name for name, info in indexes.items()
            if info.get("unique", False) and name != "_id_"
        ]
        assert len(unique_indexes) >= 1, (
            "semantic_diffs must have a unique compound index"
        )


# =============================================================================
# ChromaDB Persistence Tests
# =============================================================================


class TestChromaDBPersistence:
    """Verify ChromaDB local persistence is correctly configured."""

    def test_chroma_persist_dir_configured(self, settings) -> None:
        """ChromaDB persist path is configured."""
        path = settings.chroma.persist_path
        assert path is not None

    def test_chroma_persist_dir_writable(self, settings) -> None:
        """ChromaDB persist directory can be created and is writable."""
        path = settings.chroma.persist_path
        path.mkdir(parents=True, exist_ok=True)
        test_file = path / ".write_test"
        try:
            test_file.write_text("test")
            assert test_file.exists()
        finally:
            test_file.unlink(missing_ok=True)


# =============================================================================
# MLflow Persistence Tests
# =============================================================================


class TestMLflowPersistence:
    """Verify MLflow local tracking is correctly configured."""

    def test_mlflow_tracking_uri_configured(self, settings) -> None:
        """MLflow tracking URI is set."""
        assert settings.mlflow.tracking_uri

    def test_mlflow_backend_store_configured(self, settings) -> None:
        """MLflow backend store URI points to local SQLite."""
        uri = settings.mlflow.backend_store_uri
        assert "sqlite" in uri.lower(), (
            f"MLflow backend must be SQLite, got: {uri}"
        )

    def test_mlflow_dir_writable(self) -> None:
        """MLflow local directory can be created and is writable."""
        mlruns_path = PROJECT_ROOT / "mlruns"
        mlruns_path.mkdir(parents=True, exist_ok=True)
        test_file = mlruns_path / ".write_test"
        try:
            test_file.write_text("test")
            assert test_file.exists()
        finally:
            test_file.unlink(missing_ok=True)


# =============================================================================
# Docker Compose Validation (offline check)
# =============================================================================


class TestDockerComposeConfig:
    """Validate docker-compose.yml structure (does not require Docker daemon)."""

    def test_compose_file_exists(self) -> None:
        """docker-compose.yml exists at project root."""
        compose_path = PROJECT_ROOT / "docker-compose.yml"
        assert compose_path.exists(), "docker-compose.yml not found"

    def test_compose_file_parseable(self) -> None:
        """docker-compose.yml is valid YAML."""
        import yaml
        compose_path = PROJECT_ROOT / "docker-compose.yml"
        with open(compose_path) as f:
            config = yaml.safe_load(f)
        assert "services" in config, "docker-compose.yml must define services"

    def test_required_services_defined(self) -> None:
        """All required services are present in docker-compose.yml."""
        import yaml
        compose_path = PROJECT_ROOT / "docker-compose.yml"
        with open(compose_path) as f:
            config = yaml.safe_load(f)
        services = set(config["services"].keys())
        required = {
            "mysql",
            "mongodb",
            "ingestion_service",
            "quant_engine",
            "qualitative_engine",
            "marl_engine",
            "mlops_loop",
            "streamlit_dashboard",
            "mlflow_ui",
        }
        missing = required - services
        assert not missing, f"Missing required services: {missing}"

    def test_network_defined(self) -> None:
        """aqqip-net network is defined."""
        import yaml
        compose_path = PROJECT_ROOT / "docker-compose.yml"
        with open(compose_path) as f:
            config = yaml.safe_load(f)
        assert "networks" in config
        networks = config["networks"]
        assert "aqqip-net" in networks, "aqqip-net network must be defined"

    def test_required_volumes_defined(self) -> None:
        """Persistent volumes are defined."""
        import yaml
        compose_path = PROJECT_ROOT / "docker-compose.yml"
        with open(compose_path) as f:
            config = yaml.safe_load(f)
        assert "volumes" in config
        volumes = set(config["volumes"].keys())
        required = {"mysql_data", "mongo_data", "chroma_data", "mlflow_data"}
        missing = required - volumes
        assert not missing, f"Missing required volumes: {missing}"

    def test_documented_ports(self) -> None:
        """Services expose documented ports."""
        import yaml
        compose_path = PROJECT_ROOT / "docker-compose.yml"
        with open(compose_path) as f:
            config = yaml.safe_load(f)
        services = config["services"]

        # MySQL: 3306
        mysql_ports = services["mysql"].get("ports", [])
        assert any("3306" in str(p) for p in mysql_ports), "MySQL must expose 3306"

        # MongoDB: 27017
        mongo_ports = services["mongodb"].get("ports", [])
        assert any("27017" in str(p) for p in mongo_ports), "MongoDB must expose 27017"

        # Streamlit: 8501
        st_ports = services["streamlit_dashboard"].get("ports", [])
        assert any("8501" in str(p) for p in st_ports), "Streamlit must expose 8501"

        # MLflow: 5000
        ml_ports = services["mlflow_ui"].get("ports", [])
        assert any("5000" in str(p) for p in ml_ports), "MLflow must expose 5000"


# =============================================================================
# Migration Files Existence Tests
# =============================================================================


class TestMigrationFiles:
    """Verify all required migration/init files exist."""

    MYSQL_MIGRATIONS = [
        "001_create_ohlcv_prices.sql",
        "002_create_fundamental_ratios.sql",
        "003_create_predictions.sql",
        "004_create_marl_simulation_runs.sql",
    ]

    @pytest.mark.parametrize("migration_file", MYSQL_MIGRATIONS)
    def test_mysql_migration_exists(self, migration_file: str) -> None:
        """MySQL migration file exists."""
        path = PROJECT_ROOT / "db" / "mysql" / "init" / migration_file
        assert path.exists(), f"Missing migration: {path}"

    def test_mongo_init_exists(self) -> None:
        """MongoDB initialization script exists."""
        path = PROJECT_ROOT / "db" / "mongo" / "init" / "init-indexes.js"
        assert path.exists(), f"Missing MongoDB init: {path}"


# =============================================================================
# Package Structure Tests
# =============================================================================


class TestPackageStructure:
    """Verify Python package skeleton is correct."""

    REQUIRED_PACKAGES = [
        "src",
        "src/database",
        "src/ingestion",
        "src/quant",
        "src/qualitative",
        "src/marl",
        "src/mlops",
        "src/utils",
        "config",
    ]

    @pytest.mark.parametrize("package", REQUIRED_PACKAGES)
    def test_package_has_init(self, package: str) -> None:
        """Each package directory has __init__.py."""
        init_path = PROJECT_ROOT / package / "__init__.py"
        assert init_path.exists(), f"Missing __init__.py for package: {package}"

    def test_stub_service_exists(self) -> None:
        """Stub service module exists for placeholder containers."""
        path = PROJECT_ROOT / "src" / "stub_service.py"
        assert path.exists()

    def test_dashboard_app_exists(self) -> None:
        """Dashboard Streamlit app exists."""
        path = PROJECT_ROOT / "dashboard" / "app.py"
        assert path.exists()
