"""Delta Lake ingestion for physics simulation results."""

import logging
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


class PhysicsDeltaStore:
    """Ingest physics simulation results into Delta Lake for feature store."""

    SCHEMA = {
        "run_id": "string",
        "batch_size": "int",
        "num_steps": "int",
        "hardware_used": "string",
        "timestamp": "string",
        "quality_score": "float",
        "mu_max_rel_dev": "float",
        "spin_norm_drift_percent": "float",
        "gates_passed": "boolean",
        "wall_clock_time_sec": "float",
        "estimated_cost_usd": "float",
        "particle_mass": "float",
        "edm_eta": "float",
    }

    def __init__(self, spark=None, table_path: str = "delta/gold/physics_simulations"):
        """
        Initialize physics Delta store.

        Args:
            spark: PySpark session (optional; uses lazy initialization)
            table_path: Path to Delta table
        """
        self.spark = spark
        self.table_path = table_path

    def _get_spark(self):
        """Lazy Spark initialization."""
        if self.spark is None:
            try:
                from pyspark.sql import SparkSession

                self.spark = (
                    SparkSession.builder.appName("PhysicsSimulationStore")
                    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
                    .config(
                        "spark.sql.catalog.spark_catalog",
                        "org.apache.spark.sql.delta.catalog.DeltaCatalog",
                    )
                    .getOrCreate()
                )
            except ImportError:
                logger.warning("PySpark not available; Delta ingestion disabled")
                self.spark = None
        return self.spark

    def ingest(self, run_id: str, results: dict, hardware_info: dict, wall_clock_time: float):
        """
        Ingest a simulation result into Delta Lake.

        Args:
            run_id: Unique simulation run ID
            results: Output from batch_simulate()
            hardware_info: Output from get_hardware_info()
            wall_clock_time: Execution time in seconds
        """
        spark = self._get_spark()
        if spark is None:
            logger.warning("Spark unavailable, skipping Delta ingestion")
            return

        now = datetime.now(UTC).isoformat()
        diagnostics = results.get("diagnostics", {})
        estimated_cost = (wall_clock_time / 3600.0) * 2.0  # $2/hr A100

        record = {
            "run_id": run_id,
            "batch_size": results.get("batch_size", 0),
            "num_steps": results.get("num_steps", 0),
            "hardware_used": hardware_info.get("device_name", "cpu"),
            "timestamp": now,
            "quality_score": results.get("quality_score", 0.0),
            "mu_max_rel_dev": diagnostics.get("mu_max_rel_dev", 0.0),
            "spin_norm_drift_percent": diagnostics.get("spin_norm_drift_percent", 0.0),
            "gates_passed": diagnostics.get("gates_passed", False),
            "wall_clock_time_sec": wall_clock_time,
            "estimated_cost_usd": estimated_cost,
            "particle_mass": results.get("particle_mass", 0.938),
            "edm_eta": results.get("edm_eta", 1e-3),
        }

        try:
            df = spark.createDataFrame([record])
            (
                df.write.format("delta")
                .mode("append")
                .option("mergeSchema", "true")
                .save(self.table_path)
            )
            logger.info(f"Ingested simulation {run_id} to {self.table_path}")
        except Exception as e:
            logger.error(f"Delta ingestion failed: {e}")

    def query_recent(self, limit: int = 100) -> list[dict]:
        """
        Query recent physics simulations.

        Returns:
            List of dicts with simulation metadata
        """
        spark = self._get_spark()
        if spark is None:
            logger.warning("Spark unavailable, cannot query Delta")
            return []

        try:
            df = spark.read.format("delta").load(self.table_path)
            df = df.orderBy("timestamp").tail(limit)
            return [row.asDict() for row in df.collect()]
        except Exception as e:
            logger.error(f"Delta query failed: {e}")
            return []

    def statistics(self) -> dict:
        """
        Get summary statistics from physics simulations.

        Returns:
            Dict with mean quality, pass rate, etc.
        """
        spark = self._get_spark()
        if spark is None:
            return {}

        try:
            df = spark.read.format("delta").load(self.table_path)
            stats = df.describe(
                ["quality_score", "wall_clock_time_sec", "estimated_cost_usd"]
            ).collect()
            return {
                row[0]: {s[0]: float(s[i]) if i > 0 else s[i] for i, s in enumerate(row)}
                for row in stats
            }
        except Exception as e:
            logger.error(f"Statistics query failed: {e}")
            return {}
