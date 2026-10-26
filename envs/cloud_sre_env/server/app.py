"""
FastAPI application for the Cloud SRE Environment.
Uses the OpenEnv create_fastapi_app helper.
"""

from openenv.core.env_server import create_fastapi_app

from ..models import SREAction, SREObservation, SREState
from .cloud_sre_environment import CloudSREEnvironment

# The OpenEnv server creates one environment instance per client session.
app = create_fastapi_app(
    CloudSREEnvironment, SREAction, SREObservation, state_cls=SREState
)
