"""
Demo script for trade explainability.

This script demonstrates how to use the explainability module to understand
why the RL agent made specific trading decisions.

Usage:
    # Using Yahoo Finance data (downloads automatically)
    python examples/explain_trades.py --model models/ppo_best.zip --symbol RELIANCE.NS

    # Using local parquet file
    python examples/explain_trades.py --model models/ppo_best.zip --data data/eod/RELIANCE.NS_2020-01-01_2021-12-31.parquet

    # Specify date range for Yahoo data
    python examples/explain_trades.py --model models/ppo_best.zip --symbol RELIANCE.NS --start 2022-01-01 --end 2024-12-31
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tradebox.agents.ppo_agent import PPOAgent
from tradebox.explainability.trade_explainer import TradeExplainer
from tradebox.explainability.attention_viz import AttentionAnalyzer
from tradebox.explainability.text_generator import TradeExplainTextGenerator
from tradebox.env.trading_env import TradingEnv, EnvConfig
from tradebox.features.technical import TechnicalFeatures
from tradebox.data.loaders.yahoo_loader import YahooDataLoader


def load_agent_and_env(model_path: str, symbol: str = None, data_path: str = None,
                       start_date: str = "2022-01-01", end_date: str = "2024-12-31"):
    """
    Load trained agent and create environment.

    Args:
        model_path: Path to trained model
        symbol: Stock symbol (e.g., RELIANCE.NS) - downloads from Yahoo
        data_path: Path to local parquet/csv file (overrides symbol)
        start_date: Start date for Yahoo data
        end_date: End date for Yahoo data
    """
    print(f"Loading agent from {model_path}...")
    agent = PPOAgent.load(model_path, env=None)

    # Load data
    if data_path:
        print(f"Loading data from {data_path}...")
        if data_path.endswith('.parquet'):
            data = pd.read_parquet(data_path)
        elif data_path.endswith('.csv'):
            data = pd.read_csv(data_path, parse_dates=["Date"], index_col="Date")
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
    elif symbol:
        print(f"Downloading data for {symbol} from {start_date} to {end_date}...")
        cache_dir = 'cache'
        loader = YahooDataLoader(cache_dir=cache_dir, use_cache=True)
        data = loader.download(symbol, start_date, end_date)
        print(f"Loaded {len(data)} days of data")
    else:
        raise ValueError("Must provide either --symbol or --data")

    # Extract features
    print("Extracting technical features...")
    tech_features = TechnicalFeatures()
    features = tech_features.extract(data, fit_normalize=False)
    print(f"Extracted {len(features.columns)} features")

    # Create environment with appropriate settings for available data
    # Calculate max steps based on data length (leave some buffer)
    data_length = len(data)
    lookback_window = 60
    max_episode_steps = min(300, data_length - lookback_window - 10)  # Leave 10 bars buffer

    print(f"Data length: {data_length} bars")
    print(f"Max episode steps: {max_episode_steps}")

    env_config = EnvConfig(
        initial_capital=100000,
        lookback_window=lookback_window,
        max_episode_steps=max_episode_steps,
    )

    env = TradingEnv(data=data, features=features, config=env_config)

    return agent, env, features


def demonstrate_explainability(agent, env, features, output_dir: Path, num_steps: int = 5):
    """Demonstrate explainability features."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("TRADE EXPLAINABILITY DEMONSTRATION")
    print("=" * 80 + "\n")

    # Initialize explainer
    # Get feature names (exclude OHLCV columns)
    ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
    feature_names = [col for col in features.columns if col not in ohlcv_cols and col != "Date"]

    print(f"Detected {len(feature_names)} technical indicators")
    print(f"First 10: {feature_names[:10]}\n")

    explainer = TradeExplainer(agent, feature_names=feature_names)
    analyzer = AttentionAnalyzer()
    text_gen = TradeExplainTextGenerator()

    # Reset environment
    obs, _ = env.reset()

    # Run steps and explain each decision
    for step in range(num_steps):
        print(f"\n{'='*80}")
        print(f"STEP {step + 1}/{num_steps}")
        print(f"{'='*80}\n")

        # Get agent's action
        action, _states = agent.predict(obs, deterministic=True)

        # Generate explanation
        print("Generating explanation...")
        explanation = explainer.explain(obs, action=action, method="attention")

        # Print summary
        print(f"\n{explanation['summary']}\n")

        # Print detailed explanation
        detailed = text_gen.generate_detailed(explanation)
        print(detailed)

        # Get attention weights for visualization
        attention_weights = explainer.get_attention_weights(obs)

        if attention_weights is not None:
            # Create visualization
            print(f"\nGenerating visualization...")
            price_data = obs["price"]

            # Get top indicators for annotation
            top_indicators = explanation.get("indicator_analysis", {}).get("top_contributors", [])

            fig = analyzer.plot_attention_on_candlestick(
                price_data=price_data,
                attention_weights=attention_weights,
                action=explanation["action"],
                confidence=explanation["confidence"],
                top_indicators=top_indicators,
                save_path=output_dir / f"step_{step+1}_attention.png",
            )
            plt.close(fig)

            # Create attention heatmap
            fig_heatmap = analyzer.create_attention_heatmap(
                attention_weights,
                save_path=output_dir / f"step_{step+1}_heatmap.png",
            )
            plt.close(fig_heatmap)

            # Analyze attention distribution
            avg_attention = attention_weights.mean(axis=0)[-1, :]  # Last bar's attention
            attn_analysis = analyzer.analyze_attention_distribution(avg_attention)
            print(f"\nAttention Distribution Analysis:")
            print(f"  Pattern: {attn_analysis['pattern_type']}")
            print(f"  Critical Bars: {attn_analysis['critical_bars']}")
            print(f"  Entropy: {attn_analysis['entropy']:.3f}")
            print(f"  Max Attention: {attn_analysis['max_attention']:.3f}")
        else:
            print(f"\n⚠ Attention weights not available for this model")
            print(f"  This model may not have attention enabled or may use a different architecture.")
            print(f"  Explainability will be limited to indicator analysis only.")

        # Take step in environment
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print("\nEpisode ended. Resetting environment...")
            obs, _ = env.reset()

        print("\n")

    print(f"\nExplanation visualizations saved to: {output_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Explain RL trading agent decisions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (.zip file)",
    )

    # Data source: either symbol (Yahoo) or local file
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--symbol",
        type=str,
        help="Stock symbol to load from Yahoo Finance (e.g., RELIANCE.NS)",
    )
    data_group.add_argument(
        "--data",
        type=str,
        help="Path to local parquet/csv file",
    )

    # Date range for Yahoo data
    parser.add_argument(
        "--start",
        type=str,
        default="2022-01-01",
        help="Start date for Yahoo data (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2024-12-31",
        help="End date for Yahoo data (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="reports/explainability",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=5,
        help="Number of steps to explain (default: 5)",
    )

    args = parser.parse_args()

    try:
        # Load agent and environment
        agent, env, features = load_agent_and_env(
            args.model,
            symbol=args.symbol,
            data_path=args.data,
            start_date=args.start,
            end_date=args.end
        )

        # Demonstrate explainability
        demonstrate_explainability(agent, env, features, args.output, args.num_steps)

        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETE")
        print("=" * 80)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nMake sure you have:")
        print("  1. Trained a model (run training script first)")
        print("  2. Either:")
        print("     - Provide --symbol (e.g., RELIANCE.NS) to download from Yahoo")
        print("     - Or provide --data with path to local parquet/csv file")
        print("\nExamples:")
        print("  python examples/explain_trades.py --model models/ppo_best.zip --symbol RELIANCE.NS")
        print("  python examples/explain_trades.py --model models/ppo_best.zip --data data/eod/RELIANCE.NS_2020-01-01_2021-12-31.parquet")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
