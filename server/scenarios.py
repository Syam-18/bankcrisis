"""
Crisisbank Scenarios

Defines the starting economic conditions for each difficulty tier.
"""

SCENARIOS = [
    {
        "task_id": 1,
        "name": "Mild Overheating (Easy)",
        "description": "Inflation is running hot. Bring it down below 3.0% without crashing the economy.",
        "inflation": 5.5,       # High, needs rate hikes
        "unemployment": 4.0,    # Healthy
        "gdp_growth": 2.5,      # Strong
        "stress": 0.2           # Low stress, markets are fine
    },
    {
        "task_id": 2,
        "name": "Stagflation (Medium)",
        "description": "High inflation AND rising unemployment. Balance the dual mandate (Inflation < 3.0%, Unemployment < 5.5%).",
        "inflation": 7.5,       # Very high
        "unemployment": 6.8,    # Uncomfortably high
        "gdp_growth": -0.5,     # Mild recession
        "stress": 0.5           # Markets are nervous
    },
    {
        "task_id": 3,
        "name": "Liquidity Crisis & Deflation (Hard)",
        "description": "The economy is crashing and markets are panicking. Stabilize stress (< 0.5), restore growth (> 0), and avoid deflation.",
        "inflation": 0.5,       # Dangerously low (deflation risk)
        "unemployment": 8.5,    # Severe job losses
        "gdp_growth": -3.0,     # Deep recession
        "stress": 0.9           # Extreme market panic
    }
]