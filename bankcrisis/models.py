# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Bankcrisis Environment.

The bankcrisis environment is a simple test environment that echoes back messages.
"""

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field
from typing import Literal, Any



class BankcrisisAction(Action):
    rate_change: Literal[-25, 0, 25] = Field(..., description="Basis points change: -25, 0, +25")
    qe_amount: Literal[0.0, 10.0, 20.0] = Field(..., description="QE amount in billions: 0, 10, 20")
    guidance: Literal["hawkish", "dovish", "neutral"]


class BankcrisisState(State):
    inflation: float
    unemployment: float
    gdp_growth: float
    interest_rate: float
    market_stress: float
    step: int
    max_steps: int
    episode_id: str
    



class BankcrisisObservation(Observation):
    text: str
    state: dict
    info: dict[str, Any]
    reward: float
    done: bool