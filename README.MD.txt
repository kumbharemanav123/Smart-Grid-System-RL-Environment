# Smart Grid RL Environment

A challenging Reinforcement Learning environment for smart grid energy management. This environment exposes the limitations of LLMs like Claude Opus 4.6.

## Files Included
- `smart_grid_env.py` - The RL environment
- `claude_smart_grid_test.py` - Test script for Claude
- `requirements.txt` - Python dependencies
- `test_smart_grid.py` - Basic test script
- CSV files - Raw results from Claude tests
- `FINAL_RESULTS.md` - Summary of Claude's performance

## Results Summary
Claude Opus 4.6 achieved:
- Episode 1: -1172.39
- Episode 2: -3047.27
- Average: -2109.83

These extremely negative rewards prove this environment is too complex for current LLMs.

## Setup
1. Install Python 3.8+
2. Install requirements: `pip install -r requirements.txt`
3. Create `.env` file with your Anthropic API key: `ANTHROPIC_API_KEY=your-key`
4. Run tests: `python claude_smart_grid_test.py`

## Key Challenges
- Partial observability (hidden battery degradation)
- Delayed consequences (24-step diesel cooldown)
- Stochastic price spikes and renewable variability
- Complex trade-offs requiring long-term planning