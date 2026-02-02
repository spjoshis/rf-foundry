# Explainability Module

This module provides tools to understand and explain why the RL trading agent makes specific trading decisions.

## Features

### ✅ Phase 1 (Completed)
- **Attention Weight Visualization**: See which price bars the agent focused on before making decisions
- **Intermediate Activation Capture**: Extract embeddings from each encoder component
- **Text Summary Generation**: Human-readable explanations of trading decisions
- **Candlestick + Attention Overlay**: Visual representation of agent focus on price charts
- **Attention Heatmaps**: Multi-head attention visualization

## Quick Start

### 1. Basic Usage

```python
from tradebox.agents.ppo_agent import PPOAgent
from tradebox.explainability import TradeExplainer

# Load your trained agent
agent = PPOAgent.load("models/best_model.zip")

# Create explainer
explainer = TradeExplainer(agent, feature_names=your_feature_names)

# Explain a trade
explanation = explainer.explain(observation, action="BUY")

# Print summary
print(explanation['summary'])
# Output: "BUY executed with 87% confidence. Momentum pattern in bars 55-59. RSI oversold (35)."

# Get attention weights
attention = explainer.get_attention_weights(observation)
```

### 2. Visualization

```python
from tradebox.explainability import AttentionAnalyzer

analyzer = AttentionAnalyzer()

# Create attention visualization
fig = analyzer.plot_attention_on_candlestick(
    price_data=observation["price"],
    attention_weights=attention,
    action="BUY",
    confidence=0.87,
    save_path="reports/trade_explanation.png"
)
```

### 3. Detailed Report

```python
from tradebox.explainability import TradeExplainTextGenerator

generator = TradeExplainTextGenerator()

# Generate detailed multi-section report
detailed_report = generator.generate_detailed(explanation)
print(detailed_report)
```

## Components

### TradeExplainer
Main class that orchestrates explanation generation.

**Methods**:
- `explain(observation, action, method)`: Generate comprehensive explanation
- `get_attention_weights(observation)`: Extract raw attention weights

**Output**:
```python
{
    "action": "BUY",
    "confidence": 0.87,
    "price_pattern_analysis": {
        "attention_focus": {"primary_bars": [55, 56, 57, 58, 59], ...},
        "pattern_detected": "momentum_driven",
        ...
    },
    "indicator_analysis": {
        "top_contributors": [
            {"name": "RSI", "value": 35, "signal": "oversold", ...},
            ...
        ]
    },
    "portfolio_state": {...},
    "summary": "BUY executed with 87% confidence..."
}
```

### AttentionAnalyzer
Visualizes attention weights over candlestick charts.

**Methods**:
- `plot_attention_on_candlestick()`: Overlay attention on price chart
- `create_attention_heatmap()`: Multi-head attention heatmap
- `analyze_attention_distribution()`: Statistical analysis of attention patterns

### TradeExplainTextGenerator
Generates human-readable summaries.

**Methods**:
- `generate(explanation)`: Short summary
- `generate_detailed(explanation)`: Multi-section detailed report

## Examples

### Run Validation
Test all components without a trained model:
```bash
poetry run python examples/validate_explainability.py
```

### Explain Real Trades
Explain trades from a trained agent:

**Using Yahoo Finance data (downloads automatically):**
```bash
poetry run python examples/explain_trades.py \
    --model models/ppo_best.zip \
    --symbol RELIANCE.NS \
    --start 2022-01-01 \
    --end 2024-12-31
```

**Using local parquet file:**
```bash
poetry run python examples/explain_trades.py \
    --model models/ppo_best.zip \
    --data data/eod/RELIANCE.NS_2020-01-01_2021-12-31.parquet
```

## Architecture Integration

The explainability system integrates seamlessly with your existing architecture:

1. **PricePatternCNN** (`src/tradebox/models/trading_cnn_extractor.py`):
   - Added `capture_attention` parameter
   - Caches attention weights in `_attention_weights`

2. **TradingCNNExtractor**:
   - Added `capture_intermediates` parameter
   - Provides `get_intermediates()` method to access cached embeddings

3. **Zero Training Overhead**:
   - Explainability is only activated during inference
   - No impact on training speed or model performance

## Explanation Output Formats

### 1. Console Summary
```
BUY executed with 87% confidence. Momentum Driven pattern in bars 55-59.
RSI oversold (35), MACD bullish crossover (12.5). Cash: 100%.
```

### 2. Detailed Report
```
============================================================
BUY DECISION - Confidence: 87.0%
============================================================

PRICE PATTERN ANALYSIS:
  Pattern: Momentum Driven
  Focus Bars: 55, 56, 57, 58, 59
  Recent Focus: 65.0%

TOP INDICATORS:
  1. RSI: 35.20 (Oversold)
  2. MACD_diff: 12.50 (Bullish Crossover)
  3. BB_position: 0.15 (Near Lower Band)

PORTFOLIO STATE:
  Position: 0.0%
  Cash: 100.0%
============================================================
```

### 3. Visualizations
- **Candlestick + Attention**: Price chart with attention overlay showing which bars the agent focused on
- **Attention Heatmap**: Multi-head attention weights visualization
- **Attribution Charts** (Phase 2): Feature importance bar charts

## Performance

- **Attention extraction**: ~2ms overhead
- **Visualization generation**: ~50-100ms (offline)
- **Memory**: Minimal (caches are small CPU tensors)

## What's Next?

### Phase 2 (Coming Soon)
- ✨ Gradient-based attribution (Integrated Gradients, Saliency)
- ✨ Quantitative feature contribution scores
- ✨ Integration with Captum library

### Phase 3 (Future)
- ✨ SHAP values for deep analysis
- ✨ Interactive Streamlit dashboard
- ✨ Pattern aggregation across trades
- ✨ Backtesting report integration

## Validation Results

All components validated ✅:
```
✓ Attention capture working (4-head, 60x60)
✓ Intermediate embeddings working (price, indicators, portfolio)
✓ Attention visualization working
✓ Text generation working
```

See `reports/explainability_validation/` for sample outputs.

## Dependencies

New dependencies added (already installed):
- `captum>=0.8.0` - PyTorch interpretability toolkit
- `seaborn>=0.13.2` - Statistical visualizations

## API Reference

See docstrings in:
- `trade_explainer.py` - Main explainer class
- `attention_viz.py` - Visualization tools
- `text_generator.py` - Text summary generation

## Troubleshooting

**Q: Attention weights are None**
- Ensure model was created with `use_attention=True` (default)
- Set `capture_intermediates=True` when loading the model for inference

**Q: Visualizations not saving**
- Check output directory exists and has write permissions
- Ensure matplotlib backend is configured correctly

**Q: Feature names not showing**
- Pass `feature_names` list to TradeExplainer initialization
- Names should match the order of indicators in observation space

## Contributing

To extend the explainability module:
1. Add new attribution methods in `attribution.py` (create this file in Phase 2)
2. Add visualization types in `visualizers.py` (for custom charts)
3. Add text templates in `text_generator.py`

## License

Same as main TradeBox-RL project.
