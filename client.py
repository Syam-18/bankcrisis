# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bankcrisis Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import BankcrisisAction, BankcrisisObservation, BankcrisisAction


class BankcrisisEnv(
    EnvClient[BankcrisisAction, BankcrisisObservation, State]
):
    """
    Client for the Bankcrisis Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with BankcrisisEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(BankcrisisAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = BankcrisisEnv.from_docker_image("bankcrisis-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(BankcrisisAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: CrisisbankAction) -> dict:
        """
        Convert a CrisisbankAction into a JSON-serializable dictionary.
        This dictionary will be sent to the server's step endpoint.
        """
        return {
            "rate_change": action.rate_change,
            "qe_amount": action.qe_amount,
            "guidance": action.guidance,
        }

    def _parse_result(self, payload: dict) -> StepResult[CrisisbankObservation]:
        """
        Parse the server's response into a StepResult containing an observation.
        The payload is the JSON object returned by the server after a step.
        """
        obs_data = payload.get("observation", {})
        obs = CrisisbankObservation(
            text=obs_data.get("text", ""),
            state=obs_data.get("state", {}),
            info=obs_data.get("info", {}) 
        )
        return StepResult(
            observation=obs,
            reward=obs.info.get("reward", 0.0),  # Extract reward from info
            done=obs.info.get("done", False), 
        )

    def _parse_state(self, payload: dict) -> CrisisbankState:
        """
        Parse the server's state response into a CrisisbankState object.
        This is used when calling the state() method of the client.
        """
        return CrisisbankState(
            
            inflation=payload.get("inflation", 0.0),
            unemployment=payload.get("unemployment", 0.0),
            gdp_growth=payload.get("gdp_growth", 0.0),
            interest_rate=payload.get("interest_rate", 0.0),
            market_stress=payload.get("market_stress", 0.0),
            step=payload.get("step", 0),
            max_steps=payload.get("max_steps", 20),  # Provide a default if missing
            episode_id=payload.get('episode_id', '')
        )
