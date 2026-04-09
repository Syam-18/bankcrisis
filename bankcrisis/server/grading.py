from dataclasses import dataclass
from typing import Dict, Any
MAX_STEPS = 20

@dataclass
class GradeResult:
    score: float  # 0.0 to 1.0
    success: bool
    reason: str

def grade(task_id: int, final_state: Dict[str, Any]) -> GradeResult:
    """
    Grade the agent's performance based on the task and final economic state.
    
    Args:
        task_id: The task level (1, 2, or 3)
        final_state: The final state dictionary containing economic indicators
    
    Returns:
        GradeResult with score, success boolean, and reason string
    """
    inflation = final_state.get("inflation", 0)
    unemployment = final_state.get("unemployment", 0)
    gdp_growth = final_state.get("gdp_growth", 0)
    market_stress = final_state.get("market_stress", 0)
    step = final_state.get("step", 0)
    max_steps = final_state.get("max_steps", MAX_STEPS)
    
    # Base score calculation similar to _compute_reward but final evaluation
    if task_id == 1:  # Inflation Control (easy)
        if inflation <= 2.5:
            score = 1.0
            success = True
            reason = f"Excellent! Inflation controlled at {inflation:.1f}%"
        elif inflation <= 4.0:
            score = 0.8
            success = True
            reason = f"Good. Inflation at {inflation:.1f}% is acceptable"
        elif inflation <= 6.0:
            score = 0.5
            success = False
            reason = f"Inflation at {inflation:.1f}% is above target range"
        else:
            score = 0.0
            success = False
            reason = f"Failed to control inflation at {inflation:.1f}%"
        # print(inflation)
            
    elif task_id == 2:  # Dual Mandate (medium)
        # Target: inflation 2-3%, unemployment 4.5-5.5%
        inf_target = 2.5
        unemp_target = 5.0
        
        inf_deviation = abs(inflation - inf_target) / 3.0
        unemp_deviation = abs(unemployment - unemp_target) / 5.0
        
        inf_score = max(0, 1.0 - inf_deviation)
        unemp_score = max(0, 1.0 - unemp_deviation)
        score = (inf_score + unemp_score) / 2.0
        
        success = score >= 0.7
        if success:
            reason = f"Successfully balanced inflation ({inflation:.1f}%) and unemployment ({unemployment:.1f}%)"
        else:
            reason = f"Missed dual mandate targets - inflation: {inflation:.1f}%, unemployment: {unemployment:.1f}%"
            
    elif task_id == 3:  # Crisis Stabilisation (hard)
        # Target: low stress (<0.3), positive growth, moderate inflation (<3%)
        stress_score = max(0, 1.0 - (market_stress / 0.5))
        growth_score = max(0, 1.0 + (gdp_growth / 3.0)) if gdp_growth < 0 else min(1.0, gdp_growth / 2.0)
        inflation_penalty = max(0, (inflation - 3.0) / 5.0) if inflation > 3.0 else 0
        
        base_score = (stress_score + growth_score) / 2.0
        score = max(0, base_score - inflation_penalty)
        
        success = (market_stress <= 0.4 and gdp_growth > -1.0 and inflation <= 5.0)
        if success:
            reason = f"Stabilized crisis! Stress: {market_stress:.2f}, Growth: {gdp_growth:.1f}%, Inflation: {inflation:.1f}%"
        else:
            reason = f"Crisis not stabilized - Stress: {market_stress:.2f}, Growth: {gdp_growth:.1f}%"
    else:
        score = 0.0
        success = False
        reason = f"Unknown task_id: {task_id}"
    
    # Bonus for completing all steps without catastrophic failure
    if step >= max_steps:
        score = min(1.0, score + 0.1)
        if success:
            reason += " (Completed full term)"
    
    # Catastrophic failure check
    if unemployment >= 20.0:
        score = 0.0
        success = False
        reason = "CATASTROPHIC FAILURE: Unemployment exceeded 20%"
    elif market_stress >= 1.0:
        score = 0.0
        success = False
        reason = "CATASTROPHIC FAILURE: Market stress reached 100%"
    
    return GradeResult(
        score=max(0.0, min(1.0, score)),
        success=success,
        reason=reason
    )