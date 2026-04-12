# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Bankcrisis Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from .scenarios import SCENARIOS
from .grading import grade

from bankcrisis.models import BankcrisisAction, BankcrisisObservation, BankcrisisState


class BankcrisisEnvironment(Environment):
    """
    A simple echo environment that echoes back messages.

    This environment is designed for testing the HTTP server infrastructure.
    It maintains minimal state and simply echoes back whatever message it receives.

    Example:
        >>> env = BankcrisisEnvironment()
        >>> obs = env.reset()
        >>> print(obs.echoed_message)  # "Bankcrisis environment ready!"
        >>>
        >>> obs = env.step(BankcrisisAction(message="Hello"))
        >>> print(obs.echoed_message)  # "Hello"
        >>> print(obs.message_length)  # 5
    """

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_level: int = 1):
        self._current_task_level = task_level
        self._state = None
        self._last_rate_change = 0.0
        
        # Define how many steps it takes for policy to hit the economy
        # Easy = 0 (instant), Medium = 1 step delay, Hard = 2 step delay
        self._policy_lag_steps = task_level - 1 
        self._policy_queue = []

    def reset(self):
        scenario = next((s for s in SCENARIOS if s.get("task_id") == self._current_task_level), SCENARIOS[0])
        
        self._current_task_id = scenario["task_id"]
        self._last_rate_change = 0.0
        
        # Reset the policy lag queue based on difficulty
        self._policy_lag_steps = self._current_task_level - 1
        # Pre-fill the queue with "zero-effect" actions so we can always pop from it
        self._policy_queue = [{"rate_effect": 0.0, "qe_effect": 0.0} for _ in range(self._policy_lag_steps + 1)]

        self._state = BankcrisisState(
            inflation=scenario["inflation"],
            unemployment=scenario["unemployment"],
            gdp_growth=scenario["gdp_growth"],
            interest_rate=2.0,
            market_stress=scenario["stress"],
            step=0,
            max_steps=scenario["max_steps"],
            episode_id=str(uuid4())
        )

        return self._observe()


    def step(self, action: BankcrisisAction) -> BankcrisisObservation: 
        if self._state is None:
            self.reset()
        s = self._state

        # 1. Calculate the raw effects of the CURRENT action
        current_rate_effect = action.rate_change / 100.0
        current_qe_effect = action.qe_amount / 100.0

        # 2. Push current effects into the back of the queue
        self._policy_queue.append({
            "rate_effect": current_rate_effect,
            "qe_effect": current_qe_effect
        })

        # 3. Pop the oldest policy effect from the front of the queue
        active_policy = self._policy_queue.pop(0)
        active_rate_effect = active_policy["rate_effect"]
        active_qe_effect = active_policy["qe_effect"]

        # 4. Update the Economy using the DELAYED effects
        # The economy only feels what the central bank did N steps ago
        s.inflation += 0.1 * active_qe_effect - 0.7 * active_rate_effect
        s.unemployment += -0.1 * active_qe_effect + 0.15 * active_rate_effect
        s.gdp_growth += 0.2 * active_qe_effect - 0.1 * active_rate_effect
        
        # Interest rates and guidance, however, update/react immediately
        s.interest_rate += current_rate_effect
        
        if action.guidance == "hawkish":
            s.market_stress -= 0.1
        elif action.guidance == "dovish":
            s.inflation += 0.1

        # Stress feedback (continuous loop)
        if s.market_stress > 0.7:
            excess = s.market_stress - 0.7
            s.gdp_growth -= excess * 1.0
            s.unemployment += excess * 0.5
        
        # Clamp Values to prevent catastrophic explosions
        s.inflation = max(-1.0, min(20.0, s.inflation))
        s.unemployment = max(0.0, min(30.0, s.unemployment))
        s.interest_rate = max(0.0, min(15.0, s.interest_rate))
        s.market_stress = max(0.0, min(1.0, s.market_stress))

        s.step += 1

        # Calculate STRICT 0.0 to 1.0 step reward (from previous code)
        base_reward = self._compute_reward()

        stability_penalty = 0.0
        stability_penalty += min(0.05, 0.01 * abs(action.rate_change)) 
        stability_penalty += min(0.05, 0.005 * abs(action.qe_amount))  
        if self._last_rate_change * action.rate_change < 0:
            stability_penalty += 0.10 

        self._last_rate_change = action.rate_change
        step_reward = max(0.0, min(1.0, base_reward - stability_penalty))

        # Check Boundaries
        catastrophic_failure = s.unemployment >= 20.0 or s.market_stress >= 1.0
        done = s.step >= s.max_steps or catastrophic_failure

        if catastrophic_failure:
            step_reward = 0.0 

        info = {}
        if done:
            result = grade(self._current_task_id, s.model_dump())
            info = {
                "score": max(0.0, min(1.0, float(result.score))), 
                "success": result.success,
                "reason": result.reason,
            }

        return self._observe(step_reward, done, info)

    def _compute_reward(self) -> float:
        """
        Task-aware dense reward bounded strictly between 0.0 and 1.0.
        """
        s = self._state
        task = self._current_task_id

        # Normalize stress and growth penalties to a 0-1 scale
        stress_penalty = s.market_stress * 0.2  # Max 0.2 deduction
        growth_penalty = min(0.2, max(0, -s.gdp_growth) * 0.05) # Max 0.2 deduction

        base_score = 0.0

        if task == 1:   # Inflation Control (easy)
            if s.inflation <= 3.0:
                base_score = 1.0
            elif s.inflation < 6.0:
                base_score = 1.0 - ((s.inflation - 3.0) / 3.0)

        elif task == 2:   # Dual Mandate (medium)
            inf_score = 1.0 if s.inflation <= 3.0 else max(0.0, 1.0 - ((s.inflation - 3.0) / 3.0))
            unemp_score = 1.0 if s.unemployment <= 5.5 else max(0.0, 1.0 - ((s.unemployment - 5.5) / 5.0))
            base_score = (inf_score * 0.5) + (unemp_score * 0.5)

        elif task == 3:   # Crisis Stabilisation (hard)
            stress_score = 1.0 if s.market_stress < 0.5 else max(0.0, 1.0 - ((s.market_stress - 0.5) / 0.5))
            growth_score = 1.0 if s.gdp_growth > 0 else max(0.0, 1.0 - (abs(s.gdp_growth) / 5.0))
            inf_score = 1.0 if s.inflation < 4.0 else max(0.0, 1.0 - ((s.inflation - 4.0) / 4.0))
            base_score = (stress_score + growth_score + inf_score) / 3.0
            
        # Apply global economic penalties and clamp
        final_score = max(0.0, min(1.0, base_score - stress_penalty - growth_penalty))
        return final_score

    @property
    def state(self):
        return self._state

    def get_state(self):
        return self._state

    def _observe(self, reward = 0.0, done = False, info = None) -> BankcrisisObservation:
        if info is None:
            info = {}
            
        s = self._state
        
        # Format the pending policies so the LLM knows what is coming
        pending_str = "None"
        if self._policy_lag_steps > 0:
            pending = [f"Rate: {p['rate_effect']*100:.0f}bps, QE: {p['qe_effect']*100:.0f}" 
            for p in self._policy_queue[:-1]] # exclude the one that just popped
            if pending:
                pending_str = " | ".join(pending)

        text = (
            f"Month {s.step}:\n"
            f"- Inflation: {s.inflation:.2f}%\n"
            f"- Unemployment: {s.unemployment:.2f}%\n"
            f"- GDP Growth: {s.gdp_growth:.2f}%\n"
            f"- Interest Rate: {s.interest_rate:.2f}%\n"
            f"- Market Stress: {s.market_stress:.2f}/1.0\n"
            f"- Pending Policy Effects (Hitting in next steps): {pending_str}\n"
        )
        
        return BankcrisisObservation(
            text=text,
            state=s.model_dump(),
            reward=reward,
            done=done,
            info={
                "reward": reward,
                "done": done,
                **info  
            }
        )