"""
Inference Script for Bankcrisis Environment
============================================
Uses OpenAI‑compatible client to control monetary policy.
Actions: rate_change (-25, 0, 25), qe_amount (0, 10, 20), guidance (hawkish/neutral/dovish)
Observation: inflation, unemployment, gdp_growth, interest_rate, market_stress, step, max_steps
Reward: dense, task‑aware (can be negative – but grader gives final score 0‑1)
STDOUT format matches the required specification.
"""

import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI
from bankcrisis.server.scenarios import SCENARIOS

from bankcrisis.models import BankcrisisAction
from bankcrisis.server.bankcrisis_environment import BankcrisisEnvironment

# Environment configuration (use your actual values)
API_KEY = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# Task configuration – change this depending on which task you evaluate
# (1 = Inflation Control, 2 = Dual Mandate, 3 = Crisis Stabilisation)
TASK_ID = int(os.getenv("BANKCRISIS_TASK", "1"))
BENCHMARK = os.getenv("BANKCRISIS_BENCHMARK", "bankcrisis_env")
MAX_STEPS = 15 if TASK_ID == 2 else 20 if TASK_ID == 3 else 15
TEMPERATURE = 0.3
MAX_TOKENS = 200

# For final score normalisation (score is already 0‑1 from grader, no need to rescale)
SUCCESS_SCORE_THRESHOLD = 0.8   # consider success if grader score >= 0.8
SYSTEM_PROMPT = textwrap.dedent(
    """
    ...
    RESPONSE FORMAT: You MUST respond with ONLY a JSON object, nothing else:
    {"rate_change": <-25|0|25>, "qe_amount": <0|10|20>, "guidance": "<hawkish|neutral|dovish>"}

    No explanation. No prose. Only the JSON object.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # reward_str = f"{reward:.2f}" if reward is not None else "None"

    # print(
    #     f"[STEP] step={step} action={action} reward={reward_str} done={done_val} error={error_val}", flush=True,
    # )
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(state: dict, step: int, last_reward: float) -> str:
    return textwrap.dedent(
        f"""
        Step: {step}/{state['max_steps']}
        Inflation: {state['inflation']:.2f}%
        Unemployment: {state['unemployment']:.2f}%
        GDP Growth: {state['gdp_growth']:.2f}%
        Interest Rate: {state['interest_rate']:.2f}%
        Market Stress: {state['market_stress']:.2f}
        Last reward: {last_reward:.2f}

        Choose your next policy action (rate_change, qe_amount, guidance).
        Based on the last reward you got change the parameters to attain a higher reward
        DECISION RULES:

        - Compare current state with previous step before choosing action
        - If inflation, unemployment, or stress are not improving, you MUST change your action
        - Never repeat the exact same action more than 2 times in a row
        - If the same action is repeated and results do not improve, choose a DIFFERENT action

        POLICY STYLE:

        - Prefer gradual adjustments over large changes
        - Maintain consistency, but adapt when progress stalls
        - Account for delayed effects (policy lag)

        IMPORTANT:

        - Doing nothing repeatedly is NOT a valid strategy
        - You must actively adjust policy when the economy is not improving
            """
    ).strip()


def parse_model_response(text: str) -> BankcrisisAction:
    """Extract action values from model output (expected JSON)."""
    import json
    import re

    # Try to find JSON block
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            rc = int(data.get("rate_change", 0))
            qe = int(data.get("qe_amount", 0))
            gu = data.get("guidance", "neutral")
            # Clamp to allowed values
            rc = max(-25, min(25, rc)) if rc in [-25, 0, 25] else 0
            qe = max(0, min(20, qe)) if qe in [0, 10, 20] else 0
            if gu not in ["hawkish", "neutral", "dovish"]:
                gu = "neutral"
            return BankcrisisAction(rate_change=rc, qe_amount=qe, guidance=gu)
        except:
            pass
    # Fallback: neutral action
    return BankcrisisAction(rate_change=0, qe_amount=0, guidance="neutral")


async def run_episode(env: BankcrisisEnvironment, client: OpenAI, task_id: int, max_steps: int) -> tuple:
    """Run one episode, return (success, steps, score, rewards list)."""
    result = env.reset()   # sync reset – returns observation
    state = result.state   # dict with current values
    step = 0
    rewards = []
    done = False

    # ← Maintain conversation history across steps
    conversation_history = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    for step in range(1, state['max_steps'] + 1):
        error = None 
        if done:
            break

        # Build prompt from current state
        user_prompt = build_user_prompt(state, step, rewards[-1] if rewards else 0.0)

        # ← Add current user message to history
        conversation_history.append({"role": "user", "content": user_prompt})

        # Get model action
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=conversation_history,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            raw = completion.choices[0].message.content or ""

            print(f"[RAW] step={step} output={raw[:200]}", flush=True)

            action = parse_model_response(raw)
            action_str = f"rate_change={action.rate_change}, qe_amount={action.qe_amount}, guidance={action.guidance}"

            conversation_history.append({"role": "assistant", "content": raw})

        except Exception as e:
            action = BankcrisisAction(rate_change=0, qe_amount=0, guidance="neutral")
            action_str = "fallback_neutral"
            error = str(e)
            conversation_history.append({"role": "assistant", "content": '{"rate_change": 0, "qe_amount": 0, "guidance": "neutral"}'})

        # Step environment
        obs = env.step(action)   # returns CrisisbankObservation
        reward = obs.reward
        done = obs.done
        state = obs.state
        rewards.append(reward)

        log_step(step=step, action=action_str, reward=reward, done=done, error=error)

        if done:
            break

    # Final score from grader (inside obs.info)
    final_score = obs.info.get("score", 0.0)
    success = final_score >= SUCCESS_SCORE_THRESHOLD

    return success, step, final_score, rewards


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Direct instantiation (since your env is synchronous and not containerised)
    env = BankcrisisEnvironment(task_level=TASK_ID)
    # Force the task ID by choosing appropriate scenario – or set via reset()?
    # The environment randomly picks scenario. For deterministic task, you could modify reset().
    # For now, we rely on env.reset() to select scenario with correct task_id.

    for task_id in [1, 2, 3]:
        env   = BankcrisisEnvironment(task_level=task_id)
        result = env.reset()
        MAX_STEPS = result.state["max_steps"]  # read per-task step count

        log_start(task=f"task{task_id}", env=BENCHMARK, model=MODEL_NAME)

        success, steps_taken, final_score, rewards = await run_episode(
            env, client, task_id, MAX_STEPS
        )
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())