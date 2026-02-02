"""Model serialization utilities for saving and loading agents."""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from tradebox.agents.config import AgentConfig, PPOConfig, TrainingConfig


def save_model_with_config(
    model: BaseAlgorithm,
    path: Union[str, Path],
    config: AgentConfig,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save model with configuration and metadata.

    Creates:
    - {path}.zip: Model weights and optimizer state
    - {path}_config.json: Configuration dataclass as JSON
    - {path}_metadata.json: Training metadata

    Args:
        model: Trained Stable-Baselines3 model.
        path: Base path (without extension).
        config: Agent configuration dataclass.
        metadata: Additional metadata to save (e.g., final metrics, training time).

    Raises:
        IOError: If unable to write files.

    Example:
        >>> from stable_baselines3 import PPO
        >>> model = PPO("MlpPolicy", env)
        >>> model.learn(10000)
        >>> save_model_with_config(
        ...     model=model,
        ...     path="models/ppo_v1",
        ...     config=AgentConfig(),
        ...     metadata={"final_sharpe": 1.2, "total_trades": 150},
        ... )
        # Creates: models/ppo_v1.zip, models/ppo_v1_config.json, models/ppo_v1_metadata.json
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save model weights
    model_path = str(path)
    model.save(model_path)
    logger.info(f"Saved model weights to {model_path}.zip")

    # Save configuration
    config_path = path.parent / f"{path.name}_config.json"
    config_dict = {
        "algorithm": config.algorithm,
        "ppo": asdict(config.ppo),
        "training": asdict(config.training),
    }
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"Saved configuration to {config_path}")

    # Save metadata
    metadata_path = path.parent / f"{path.name}_metadata.json"
    full_metadata = {
        "saved_at": datetime.now().isoformat(),
        "num_timesteps": int(model.num_timesteps),
        "observation_space": str(model.observation_space),
        "action_space": str(model.action_space),
    }
    if metadata:
        full_metadata.update(metadata)

    with open(metadata_path, "w") as f:
        json.dump(full_metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")


def load_model_with_config(
    path: Union[str, Path],
    algorithm_class: Type[BaseAlgorithm] = PPO,
    env: Optional[Any] = None,
    device: str = "auto",
) -> Tuple[BaseAlgorithm, AgentConfig, Dict[str, Any]]:
    """
    Load model with configuration and metadata.

    Args:
        path: Base path (without extension).
        algorithm_class: Stable-Baselines3 algorithm class to use (PPO, SAC, etc.).
        env: Environment to attach (optional, can be None for inference).
        device: Device to load model on ('auto', 'cpu', 'cuda').

    Returns:
        Tuple of:
        - model: Loaded Stable-Baselines3 model
        - config: AgentConfig reconstructed from saved JSON
        - metadata: Training metadata dictionary

    Raises:
        FileNotFoundError: If model file doesn't exist.
        ValueError: If configuration is invalid.

    Example:
        >>> model, config, metadata = load_model_with_config("models/ppo_v1")
        >>> print(f"Model trained for {metadata['num_timesteps']} steps")
        >>> print(f"Learning rate: {config.ppo.learning_rate}")
    """
    path = Path(path)

    # Load model
    model_path = f"{path}.zip"
    if not Path(model_path).exists():
        # Try without .zip extension
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model_path = str(path)

    model = algorithm_class.load(model_path, env=env, device=device)
    logger.info(f"Loaded model from {model_path}")

    # Load configuration
    config_path = path.parent / f"{path.name}_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config_dict = json.load(f)
        config = AgentConfig(
            algorithm=config_dict.get("algorithm", "PPO"),
            ppo=PPOConfig(**config_dict.get("ppo", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
        )
        logger.info(f"Loaded configuration from {config_path}")
    else:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        config = AgentConfig()

    # Load metadata
    metadata_path = path.parent / f"{path.name}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        logger.info(f"Loaded metadata from {metadata_path}")
    else:
        logger.warning(f"Metadata file not found: {metadata_path}")
        metadata = {"num_timesteps": model.num_timesteps}

    return model, config, metadata


def get_model_info(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get model information without loading full model.

    This is faster than loading the full model when you only need
    to inspect configuration or metadata.

    Args:
        path: Path to saved model (without extension).

    Returns:
        Dictionary with:
        - config: Configuration dictionary
        - metadata: Training metadata dictionary
        - exists: Whether model file exists

    Raises:
        FileNotFoundError: If neither config nor metadata files exist.

    Example:
        >>> info = get_model_info("models/ppo_v1")
        >>> print(f"Learning rate: {info['config']['ppo']['learning_rate']}")
        >>> print(f"Timesteps: {info['metadata']['num_timesteps']}")
    """
    path = Path(path)

    result: Dict[str, Any] = {
        "path": str(path),
        "exists": Path(f"{path}.zip").exists() or path.exists(),
        "config": None,
        "metadata": None,
    }

    # Load configuration if exists
    config_path = path.parent / f"{path.name}_config.json"
    if config_path.exists():
        with open(config_path) as f:
            result["config"] = json.load(f)

    # Load metadata if exists
    metadata_path = path.parent / f"{path.name}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            result["metadata"] = json.load(f)

    if result["config"] is None and result["metadata"] is None:
        raise FileNotFoundError(
            f"No config or metadata files found for model at {path}"
        )

    return result


def list_saved_models(models_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    List all saved models in directory.

    Scans the directory for model files (.zip) and returns information
    about each model found.

    Args:
        models_dir: Directory containing saved models.

    Returns:
        List of dictionaries, each containing:
        - name: Model name (filename without extension)
        - path: Full path to model
        - config: Configuration (if available)
        - metadata: Metadata (if available)

    Example:
        >>> models = list_saved_models("models/")
        >>> for m in models:
        ...     print(f"{m['name']}: {m['metadata'].get('num_timesteps', 'N/A')} steps")
    """
    models_dir = Path(models_dir)

    if not models_dir.exists():
        logger.warning(f"Models directory does not exist: {models_dir}")
        return []

    models = []
    for model_file in models_dir.glob("*.zip"):
        model_name = model_file.stem
        model_path = models_dir / model_name

        try:
            info = get_model_info(model_path)
            models.append({
                "name": model_name,
                "path": str(model_path),
                "config": info.get("config"),
                "metadata": info.get("metadata"),
            })
        except FileNotFoundError:
            # Model file exists but no config/metadata
            models.append({
                "name": model_name,
                "path": str(model_path),
                "config": None,
                "metadata": None,
            })

    # Sort by save time if available
    models.sort(
        key=lambda m: m.get("metadata", {}).get("saved_at", ""),
        reverse=True,
    )

    logger.info(f"Found {len(models)} saved models in {models_dir}")
    return models


def delete_model(path: Union[str, Path]) -> None:
    """
    Delete saved model and associated files.

    Removes:
    - {path}.zip: Model weights
    - {path}_config.json: Configuration
    - {path}_metadata.json: Metadata

    Args:
        path: Base path of model to delete (without extension).

    Raises:
        FileNotFoundError: If model doesn't exist.

    Example:
        >>> delete_model("models/old_model")
        # Removes old_model.zip, old_model_config.json, old_model_metadata.json
    """
    path = Path(path)

    model_path = Path(f"{path}.zip")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Delete model file
    model_path.unlink()
    logger.info(f"Deleted model file: {model_path}")

    # Delete config if exists
    config_path = path.parent / f"{path.name}_config.json"
    if config_path.exists():
        config_path.unlink()
        logger.info(f"Deleted config file: {config_path}")

    # Delete metadata if exists
    metadata_path = path.parent / f"{path.name}_metadata.json"
    if metadata_path.exists():
        metadata_path.unlink()
        logger.info(f"Deleted metadata file: {metadata_path}")
