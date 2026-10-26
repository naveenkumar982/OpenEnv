"""
Tests for the Cloud SRE Environment.

Validates all three tasks (easy, medium, hard) with both fixed
and seeded states, plus grading correctness.
"""

import pytest
from cloud_sre_env.models import SREAction
from cloud_sre_env.server.cloud_sre_environment import (
    CHAOS_EVENTS,
    CloudSREEnvironment,
    TASK_REGISTRY,
)


class TestEnvironmentBasics:
    """Test reset, step, state lifecycle."""

    def test_reset_returns_observation(self):
        env = CloudSREEnvironment()
        obs = env.reset(task_id="phantom_volume_cleanup")
        assert len(obs.resources) > 0
        assert obs.step_number == 0
        assert obs.task_description != ""

    def test_step_increments(self):
        env = CloudSREEnvironment()
        env.reset(task_id="phantom_volume_cleanup")
        obs = env.step(SREAction(command="wait"))
        assert obs.step_number == 1

    def test_state_returns_sre_state(self):
        env = CloudSREEnvironment()
        env.reset(task_id="phantom_volume_cleanup")
        state = env.state
        assert state.task_id == "phantom_volume_cleanup"
        assert state.episode_id != ""

    def test_max_steps_ends_episode(self):
        env = CloudSREEnvironment()
        env.reset(task_id="phantom_volume_cleanup")
        for _ in range(15):
            env.step(SREAction(command="wait"))
        assert env.state.done is True

    def test_invalid_task_raises(self):
        env = CloudSREEnvironment()
        with pytest.raises(ValueError, match="Unknown task"):
            env.reset(task_id="nonexistent_task")


class TestTask1PhantomVolumeCleanup:
    """Test the easy task — orphan volume termination."""

    def test_fixed_state_has_orphans(self):
        env = CloudSREEnvironment()
        obs = env.reset(task_id="phantom_volume_cleanup")
        orphans = [r for r in obs.resources if r.get("status") == "available"]
        assert len(orphans) == 3

    def test_perfect_score(self):
        env = CloudSREEnvironment()
        env.reset(task_id="phantom_volume_cleanup")
        for oid in ["ebs-orphan-001", "ebs-orphan-002", "ebs-orphan-003"]:
            env.step(SREAction(command="terminate", resource_id=oid))
        score, breakdown = env.grade()
        assert score == 1.0
        assert len(breakdown["orphans_terminated"]) == 3

    def test_penalty_for_wrong_termination(self):
        env = CloudSREEnvironment()
        env.reset(task_id="phantom_volume_cleanup")
        env.step(SREAction(command="terminate", resource_id="ec2-web-001"))
        score, breakdown = env.grade()
        assert score == 0.0
        assert len(breakdown["active_resources_terminated"]) == 1

    def test_seeded_state_is_different(self):
        env = CloudSREEnvironment()
        obs1 = env.reset(task_id="phantom_volume_cleanup", seed=42)
        obs2 = env.reset(task_id="phantom_volume_cleanup", seed=99)
        ids1 = {r["id"] for r in obs1.resources}
        ids2 = {r["id"] for r in obs2.resources}
        assert ids1 != ids2


class TestTask2LatencySpikeRemediation:
    """Test the medium task — RDS scaling."""

    def test_fixed_state_has_rds(self):
        env = CloudSREEnvironment()
        obs = env.reset(task_id="latency_spike_remediation")
        rds = [r for r in obs.resources if r.get("type") == "rds_database"]
        assert len(rds) >= 1

    def test_scaling_rds_scores(self):
        env = CloudSREEnvironment()
        env.reset(task_id="latency_spike_remediation")
        env.step(
            SREAction(
                command="scale",
                resource_id="rds-primary-001",
                params={"target_size": "db.t3.medium"},
            )
        )
        score, breakdown = env.grade()
        assert score >= 0.7
        assert breakdown["rds_scaled"] is True


class TestTask3NoisyNeighborIncident:
    """Test the hard task — rogue instance investigation."""

    def test_fixed_state_has_rogue_and_stopped_backend(self):
        env = CloudSREEnvironment()
        obs = env.reset(task_id="noisy_neighbor_incident")
        ids = {r["id"] for r in obs.resources}
        assert "ec2-rogue-test-001" in ids
        assert "ec2-backend-prod-001" in ids

    def test_perfect_incident_response(self):
        env = CloudSREEnvironment()
        env.reset(task_id="noisy_neighbor_incident")
        env.step(SREAction(command="inspect", resource_id="ec2-rogue-test-001"))
        env.step(SREAction(command="terminate", resource_id="ec2-rogue-test-001"))
        env.step(SREAction(command="reboot", resource_id="ec2-backend-prod-001"))
        score, breakdown = env.grade()
        assert score == 1.0
        assert breakdown["inspected_rogue"] is True
        assert breakdown["terminated_rogue"] is True
        assert breakdown["rebooted_backend"] is True
        assert breakdown["alerts_resolved"] is True


class TestAllTasksRegistered:
    """Ensure all tasks are in the registry."""

    def test_registry_has_all_tasks(self):
        assert "phantom_volume_cleanup" in TASK_REGISTRY
        assert "latency_spike_remediation" in TASK_REGISTRY
        assert "noisy_neighbor_incident" in TASK_REGISTRY


def test_step_rewards_and_terminal_flag_reach_client():
    from cloud_sre_env.client import CloudSREEnv
    from openenv.core.env_server.serialization import serialize_observation

    env = CloudSREEnvironment()
    env.reset(task_id="phantom_volume_cleanup")
    client = CloudSREEnv.__new__(CloudSREEnv)
    for step in range(15):
        obs = env.step(SREAction(command="wait"))
        result = client._parse_result(serialize_observation(obs))
        assert result.reward == pytest.approx(-0.01)
        assert result.observation.reward == result.reward
        assert result.done is (step == 14)
        assert result.observation.done == result.done
        assert result.observation.step_number == step + 1
    assert env.step(SREAction(command="wait")).reward == 0.0


def test_cost_drift_chaos_survives_state_recalculation():
    import random

    def event_for_seed(candidate):
        rng = random.Random(candidate * 1000 + 4)
        if rng.random() > 0.25:
            return None
        return rng.choice(CHAOS_EVENTS)["type"]

    seed = next(
        candidate
        for candidate in range(1000)
        if event_for_seed(candidate) == "cost_drift"
    )
    env = CloudSREEnvironment()
    env.reset(task_id="phantom_volume_cleanup", seed=seed)
    for _ in range(4):
        observation = env.step(SREAction(command="wait"))

    resource_cost = sum(r["cost_per_hour"] for r in observation.resources)
    assert observation.total_hourly_cost > resource_cost
    drifted_cost = observation.total_hourly_cost
    env._recalculate_state()
    assert env._build_observation().total_hourly_cost == drifted_cost


def test_scale_reward_uses_pre_mutation_cpu():
    env = CloudSREEnvironment()
    env.reset(task_id="latency_spike_remediation")
    observation = env.step(
        SREAction(
            command="scale",
            resource_id="rds-primary-001",
            params={"target_size": "db.t3.medium"},
        )
    )
    assert observation.reward == pytest.approx(0.13)
    rds = next(r for r in observation.resources if r["id"] == "rds-primary-001")
    assert rds["cpu_utilization"] == 45.0


def test_server_app_uses_environment_factory():
    from cloud_sre_env.server.app import app

    assert app.title == "OpenEnv Environment HTTP API"


def test_reset_preserves_requested_episode_id_and_seed():
    env = CloudSREEnvironment()
    first = env.reset(seed=42, episode_id="requested-episode")
    assert env.state.episode_id == "requested-episode"
    second = env.reset(seed=42)
    assert env.state.episode_id
    assert env.state.episode_id != "requested-episode"
    assert second.resources == first.resources
