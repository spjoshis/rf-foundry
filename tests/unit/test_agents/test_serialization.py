"""Unit tests for model serialization utilities."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tradebox.agents.config import AgentConfig, PPOConfig, TrainingConfig
from tradebox.agents.serialization import (
    delete_model,
    get_model_info,
    list_saved_models,
    load_model_with_config,
    save_model_with_config,
)


class MockModel:
    """Mock SB3 model for testing serialization."""

    def __init__(self):
        self.num_timesteps = 10000
        self.observation_space = MagicMock()
        self.observation_space.__str__ = lambda self: "Box(-inf, inf, (100,), float32)"
        self.action_space = MagicMock()
        self.action_space.__str__ = lambda self: "Discrete(3)"

    def save(self, path: str) -> None:
        """Mock save method."""
        Path(f"{path}.zip").touch()


class TestSaveModelWithConfig:
    """Tests for save_model_with_config function."""

    def test_creates_model_file(self) -> None:
        """Test that model file is created."""
        model = MockModel()
        config = AgentConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_model"
            save_model_with_config(model, path, config)

            assert (Path(tmpdir) / "test_model.zip").exists()

    def test_creates_config_file(self) -> None:
        """Test that config file is created."""
        model = MockModel()
        config = AgentConfig(
            ppo=PPOConfig(learning_rate=0.0001),
            training=TrainingConfig(total_timesteps=500000),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_model"
            save_model_with_config(model, path, config)

            config_path = Path(tmpdir) / "test_model_config.json"
            assert config_path.exists()

            with open(config_path) as f:
                saved_config = json.load(f)

            assert saved_config["algorithm"] == "PPO"
            assert saved_config["ppo"]["learning_rate"] == 0.0001
            assert saved_config["training"]["total_timesteps"] == 500000

    def test_creates_metadata_file(self) -> None:
        """Test that metadata file is created."""
        model = MockModel()
        config = AgentConfig()
        metadata = {"final_sharpe": 1.5, "total_trades": 100}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_model"
            save_model_with_config(model, path, config, metadata)

            metadata_path = Path(tmpdir) / "test_model_metadata.json"
            assert metadata_path.exists()

            with open(metadata_path) as f:
                saved_metadata = json.load(f)

            assert "saved_at" in saved_metadata
            assert saved_metadata["num_timesteps"] == 10000
            assert saved_metadata["final_sharpe"] == 1.5
            assert saved_metadata["total_trades"] == 100

    def test_creates_parent_directories(self) -> None:
        """Test that parent directories are created."""
        model = MockModel()
        config = AgentConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "path" / "model"
            save_model_with_config(model, path, config)

            assert (path.parent / f"{path.name}.zip").exists()


class TestLoadModelWithConfig:
    """Tests for load_model_with_config function."""

    def test_loads_config(self) -> None:
        """Test that config is loaded correctly."""
        # Create config file manually
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_model"

            # Create model file
            (Path(tmpdir) / "test_model.zip").touch()

            # Create config file
            config = {
                "algorithm": "PPO",
                "ppo": {"learning_rate": 0.0005, "network_arch": [128, 128]},
                "training": {"total_timesteps": 100000},
            }
            with open(Path(tmpdir) / "test_model_config.json", "w") as f:
                json.dump(config, f)

            # Create metadata file
            metadata = {"num_timesteps": 50000}
            with open(Path(tmpdir) / "test_model_metadata.json", "w") as f:
                json.dump(metadata, f)

            # Mock the PPO.load function
            with patch("tradebox.agents.serialization.PPO") as MockPPO:
                MockPPO.load.return_value = MagicMock(num_timesteps=50000)

                model, loaded_config, loaded_metadata = load_model_with_config(path)

                assert loaded_config.algorithm == "PPO"
                assert loaded_config.ppo.learning_rate == 0.0005
                assert loaded_config.ppo.network_arch == [128, 128]
                assert loaded_config.training.total_timesteps == 100000
                assert loaded_metadata["num_timesteps"] == 50000

    def test_raises_for_missing_model(self) -> None:
        """Test that FileNotFoundError is raised for missing model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nonexistent"

            with pytest.raises(FileNotFoundError):
                load_model_with_config(path)


class TestGetModelInfo:
    """Tests for get_model_info function."""

    def test_returns_config_and_metadata(self) -> None:
        """Test that config and metadata are returned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_model"

            # Create model file
            (Path(tmpdir) / "test_model.zip").touch()

            # Create config file
            config = {"algorithm": "PPO", "ppo": {}, "training": {}}
            with open(Path(tmpdir) / "test_model_config.json", "w") as f:
                json.dump(config, f)

            # Create metadata file
            metadata = {"num_timesteps": 10000}
            with open(Path(tmpdir) / "test_model_metadata.json", "w") as f:
                json.dump(metadata, f)

            info = get_model_info(path)

            assert info["exists"] is True
            assert info["config"]["algorithm"] == "PPO"
            assert info["metadata"]["num_timesteps"] == 10000

    def test_returns_exists_false_for_missing_model(self) -> None:
        """Test exists is False when model file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_model"

            # Create only config file (no model)
            config = {"algorithm": "PPO"}
            with open(Path(tmpdir) / "test_model_config.json", "w") as f:
                json.dump(config, f)

            info = get_model_info(path)

            assert info["exists"] is False

    def test_raises_for_no_files(self) -> None:
        """Test raises when no config or metadata files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_model"

            with pytest.raises(FileNotFoundError):
                get_model_info(path)


class TestListSavedModels:
    """Tests for list_saved_models function."""

    def test_lists_models_in_directory(self) -> None:
        """Test that models in directory are listed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two model files
            for name in ["model_a", "model_b"]:
                (Path(tmpdir) / f"{name}.zip").touch()
                config = {"algorithm": "PPO"}
                with open(Path(tmpdir) / f"{name}_config.json", "w") as f:
                    json.dump(config, f)
                metadata = {"num_timesteps": 1000}
                with open(Path(tmpdir) / f"{name}_metadata.json", "w") as f:
                    json.dump(metadata, f)

            models = list_saved_models(tmpdir)

            assert len(models) == 2
            names = [m["name"] for m in models]
            assert "model_a" in names
            assert "model_b" in names

    def test_returns_empty_for_nonexistent_directory(self) -> None:
        """Test returns empty list for nonexistent directory."""
        models = list_saved_models("/nonexistent/path")

        assert models == []

    def test_returns_empty_for_empty_directory(self) -> None:
        """Test returns empty list for empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            models = list_saved_models(tmpdir)

            assert models == []


class TestDeleteModel:
    """Tests for delete_model function."""

    def test_deletes_all_files(self) -> None:
        """Test that all model files are deleted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_model"

            # Create model files
            (Path(tmpdir) / "test_model.zip").touch()
            (Path(tmpdir) / "test_model_config.json").touch()
            (Path(tmpdir) / "test_model_metadata.json").touch()

            delete_model(path)

            assert not (Path(tmpdir) / "test_model.zip").exists()
            assert not (Path(tmpdir) / "test_model_config.json").exists()
            assert not (Path(tmpdir) / "test_model_metadata.json").exists()

    def test_raises_for_missing_model(self) -> None:
        """Test raises FileNotFoundError for missing model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nonexistent"

            with pytest.raises(FileNotFoundError):
                delete_model(path)
