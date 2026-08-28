import uuid
from unittest import mock
from unittest.mock import MagicMock

import pytest
from google.api_core.exceptions import AlreadyExists, NotFound, ServiceUnavailable
from google.cloud.batch_v1.types import Job, JobStatus, StatusEvent
from prefect_gcp.credentials import GcpCredentials
from prefect_gcp.models.batch import BatchJob
from prefect_gcp.workers.batch import (
    GCPBatchJobConfiguration,
    GCPBatchWorker,
    GCPBatchWorkerResult,
)

from prefect.client.schemas.objects import FlowRun
from prefect.exceptions import InfrastructureNotFound
from prefect.logging.loggers import PrefectLogAdapter
from prefect.utilities.dockerutils import get_prefect_image_name


class MockFlowRun:
    id = "test-id"
    name = "test-run"

    def model_dump(self, mode: str = "python") -> dict:
        return {"id": self.id, "name": self.name}


@pytest.fixture
def real_flow_run():
    """A real `FlowRun`, needed by tests that exercise `GCPBatchWorker.run()`
    end-to-end, since `get_flow_run_logger` accesses `deployment_id`/
    `parent_task_run_id`, which `MockFlowRun` doesn't have."""
    return FlowRun(id=uuid.uuid4(), flow_id=uuid.uuid4(), name="test-run")


@pytest.fixture
def job_body():
    return {
        "task_groups": [
            {
                "task_spec": {
                    "runnables": [{"container": {}}],
                    "compute_resource": {},
                    "environment": {"variables": {}},
                },
                "task_count": 1,
                "parallelism": 1,
            }
        ],
        "allocation_policy": {"instances": [{"policy": {}}]},
        "logs_policy": {"destination": "CLOUD_LOGGING"},
        "labels": {},
    }


@pytest.fixture
def gcp_batch_job_config(service_account_info, job_body):
    return GCPBatchJobConfiguration(
        name="my-job-name",
        job_body=job_body,
        credentials=GcpCredentials(service_account_info=service_account_info),
        region="us-central1",
        env={"ENV1": "VALUE1", "ENV2": "VALUE2"},
    )


def _job(state: "JobStatus.State", status_events=None, uid="job-uid-123") -> BatchJob:
    return BatchJob(
        Job(
            name="projects/my_project/locations/us-central1/jobs/my-job",
            uid=uid,
            status=JobStatus(state=state, status_events=status_events or []),
        )
    )


class TestGCPBatchJobConfiguration:
    def test_project(self, gcp_batch_job_config):
        assert gcp_batch_job_config.project == "my_project"

    def test_job_name(self, gcp_batch_job_config):
        job_name = gcp_batch_job_config.job_name
        assert job_name.startswith("my-job-name-")
        # cached
        assert gcp_batch_job_config.job_name == job_name

    def test_prepare_for_flow_run_populates_image(self, gcp_batch_job_config):
        gcp_batch_job_config.prepare_for_flow_run(flow_run=MockFlowRun())
        container = gcp_batch_job_config.job_body["task_groups"][0]["task_spec"][
            "runnables"
        ][0]["container"]
        assert container["image_uri"] == f"docker.io/{get_prefect_image_name()}"

    def test_prepare_for_flow_run_populates_command(self, gcp_batch_job_config):
        gcp_batch_job_config.prepare_for_flow_run(flow_run=MockFlowRun())
        container = gcp_batch_job_config.job_body["task_groups"][0]["task_spec"][
            "runnables"
        ][0]["container"]
        assert container["commands"] == ["prefect", "flow-run", "execute"]

    def test_prepare_for_flow_run_formats_string_command(self, gcp_batch_job_config):
        gcp_batch_job_config.job_body["task_groups"][0]["task_spec"]["runnables"][0][
            "container"
        ]["commands"] = "python my_script.py"
        gcp_batch_job_config.prepare_for_flow_run(flow_run=MockFlowRun())
        container = gcp_batch_job_config.job_body["task_groups"][0]["task_spec"][
            "runnables"
        ][0]["container"]
        assert container["commands"] == ["python", "my_script.py"]

    def test_prepare_for_flow_run_populates_env(self, gcp_batch_job_config):
        gcp_batch_job_config.prepare_for_flow_run(flow_run=MockFlowRun())
        env_vars = gcp_batch_job_config.job_body["task_groups"][0]["task_spec"][
            "environment"
        ]["variables"]
        assert env_vars["ENV1"] == "VALUE1"
        assert env_vars["ENV2"] == "VALUE2"

    def test_prepare_for_flow_run_populates_compute_resource(
        self, gcp_batch_job_config
    ):
        gcp_batch_job_config.cpu = "2000m"
        gcp_batch_job_config.memory = "2Gi"
        gcp_batch_job_config.prepare_for_flow_run(flow_run=MockFlowRun())
        compute_resource = gcp_batch_job_config.job_body["task_groups"][0][
            "task_spec"
        ]["compute_resource"]
        assert compute_resource["cpu_milli"] == 2000
        assert compute_resource["memory_mib"] == 2048

    def test_prepare_for_flow_run_populates_provisioning_model(
        self, gcp_batch_job_config
    ):
        gcp_batch_job_config.provisioning_model = "spot"
        gcp_batch_job_config.prepare_for_flow_run(flow_run=MockFlowRun())
        policy = gcp_batch_job_config.job_body["allocation_policy"]["instances"][0][
            "policy"
        ]
        assert policy["provisioning_model"] == "SPOT"

    def test_prepare_for_flow_run_populates_max_run_duration(
        self, gcp_batch_job_config
    ):
        gcp_batch_job_config.max_run_duration = 1800
        gcp_batch_job_config.prepare_for_flow_run(flow_run=MockFlowRun())
        task_spec = gcp_batch_job_config.job_body["task_groups"][0]["task_spec"]
        assert task_spec["max_run_duration"] == "1800s"

    def test_prepare_for_flow_run_populates_labels(self, gcp_batch_job_config):
        gcp_batch_job_config.prepare_for_flow_run(flow_run=MockFlowRun())
        labels = gcp_batch_job_config.job_body["labels"]
        assert labels["prefect-io-flow-run-id"] == "test-id"
        assert labels["prefect-io-flow-run-name"] == "test-run"

    def test_job_body_missing_required_components_raises(
        self, service_account_info
    ):
        with pytest.raises(ValueError, match="missing required components"):
            GCPBatchJobConfiguration(
                name="my-job-name",
                job_body={},
                credentials=GcpCredentials(service_account_info=service_account_info),
            )

    def test_job_body_incompatible_values_raises(self, service_account_info):
        with pytest.raises(ValueError, match="incompatible values"):
            GCPBatchJobConfiguration(
                name="my-job-name",
                job_body={
                    "task_groups": [{"task_spec": {"runnables": "not-a-list"}}]
                },
                credentials=GcpCredentials(service_account_info=service_account_info),
            )


class TestGCPBatchWorkerRun:
    async def test_run_succeeds_and_deletes_job(
        self, gcp_batch_job_config, mock_credentials, real_flow_run
    ):
        gcp_batch_job_config.prepare_for_flow_run(flow_run=real_flow_run)

        created_job = _job(JobStatus.State.STATE_UNSPECIFIED)
        running_job = _job(JobStatus.State.RUNNING)
        succeeded_job = _job(JobStatus.State.SUCCEEDED)

        fake_client = MagicMock()
        fake_client.create_job.return_value = created_job.job
        fake_client.get_job.side_effect = [running_job.job, succeeded_job.job]

        task_status = MagicMock()

        async with GCPBatchWorker("my-work-pool") as worker:
            with mock.patch.object(
                GCPBatchWorker, "_get_client", return_value=fake_client
            ):
                with mock.patch("prefect_gcp.workers.batch.time.sleep"):
                    with mock.patch.object(GCPBatchWorker, "_log_job_logs"):
                        result = await worker.run(
                            flow_run=real_flow_run,
                            configuration=gcp_batch_job_config,
                            task_status=task_status,
                        )

        assert isinstance(result, GCPBatchWorkerResult)
        assert result.status_code == 0
        task_status.started.assert_called_once_with(gcp_batch_job_config.job_name)
        assert fake_client.create_job.call_count == 1
        assert fake_client.get_job.call_count == 2
        assert fake_client.delete_job.call_count == 1

    async def test_run_keeps_job_when_keep_job_true(
        self, gcp_batch_job_config, mock_credentials, real_flow_run
    ):
        gcp_batch_job_config.keep_job = True
        gcp_batch_job_config.prepare_for_flow_run(flow_run=real_flow_run)

        created_job = _job(JobStatus.State.SUCCEEDED)
        fake_client = MagicMock()
        fake_client.create_job.return_value = created_job.job

        async with GCPBatchWorker("my-work-pool") as worker:
            with mock.patch.object(
                GCPBatchWorker, "_get_client", return_value=fake_client
            ):
                with mock.patch("prefect_gcp.workers.batch.time.sleep"):
                    with mock.patch.object(GCPBatchWorker, "_log_job_logs"):
                        await worker.run(
                            flow_run=real_flow_run, configuration=gcp_batch_job_config
                        )

        fake_client.delete_job.assert_not_called()

    async def test_run_reports_failure(
        self, gcp_batch_job_config, mock_credentials, real_flow_run
    ):
        gcp_batch_job_config.prepare_for_flow_run(flow_run=real_flow_run)

        created_job = _job(JobStatus.State.FAILED)
        fake_client = MagicMock()
        fake_client.create_job.return_value = created_job.job

        async with GCPBatchWorker("my-work-pool") as worker:
            with mock.patch.object(
                GCPBatchWorker, "_get_client", return_value=fake_client
            ):
                with mock.patch("prefect_gcp.workers.batch.time.sleep"):
                    with mock.patch.object(GCPBatchWorker, "_log_job_logs"):
                        result = await worker.run(
                            flow_run=real_flow_run, configuration=gcp_batch_job_config
                        )

        assert result.status_code == 1

    async def test_run_logs_preemption(self, gcp_batch_job_config, mock_credentials):
        gcp_batch_job_config.prepare_for_flow_run(flow_run=MockFlowRun())

        preempted_job = _job(
            JobStatus.State.FAILED,
            status_events=[StatusEvent(description="exit code 50001")],
        )
        fake_client = MagicMock()
        fake_client.create_job.return_value = preempted_job.job

        mock_logger = MagicMock(spec=PrefectLogAdapter)

        async with GCPBatchWorker("my-work-pool") as worker:
            with mock.patch.object(
                GCPBatchWorker, "_get_client", return_value=fake_client
            ):
                with mock.patch("prefect_gcp.workers.batch.time.sleep"):
                    with mock.patch.object(GCPBatchWorker, "_log_job_logs"):
                        with mock.patch.object(
                            worker, "get_flow_run_logger", return_value=mock_logger
                        ):
                            await worker.run(
                                flow_run=MockFlowRun(),
                                configuration=gcp_batch_job_config,
                            )

        assert mock_logger.error.call_count == 1
        assert "preempted" in mock_logger.error.call_args[0][0]


class TestGCPBatchWorkerWatchJob:
    def test_watch_job_returns_on_terminal_state(
        self, gcp_batch_job_config, mock_credentials
    ):
        fake_client = MagicMock()
        succeeded_job = _job(JobStatus.State.SUCCEEDED)
        fake_client.get_job.return_value = succeeded_job.job

        mock_logger = MagicMock(spec=PrefectLogAdapter)

        from prefect_gcp.settings import GCPBatchWorkerSettings

        with mock.patch("prefect_gcp.workers.batch.time.sleep"):
            result = GCPBatchWorker._watch_job(
                client=fake_client,
                job=_job(JobStatus.State.RUNNING),
                poll_interval=0,
                logger=mock_logger,
                settings=GCPBatchWorkerSettings(),
            )

        assert result.succeeded()

    def test_watch_job_raises_infrastructure_not_found_on_404(
        self, gcp_batch_job_config, mock_credentials
    ):
        fake_client = MagicMock()
        fake_client.get_job.side_effect = NotFound("deleted")

        mock_logger = MagicMock(spec=PrefectLogAdapter)

        from prefect_gcp.settings import GCPBatchWorkerSettings

        with mock.patch("prefect_gcp.workers.batch.time.sleep"):
            with pytest.raises(InfrastructureNotFound):
                GCPBatchWorker._watch_job(
                    client=fake_client,
                    job=_job(JobStatus.State.RUNNING),
                    poll_interval=0,
                    logger=mock_logger,
                    settings=GCPBatchWorkerSettings(),
                )

    def test_watch_job_retries_transient_error(
        self, gcp_batch_job_config, mock_credentials
    ):
        fake_client = MagicMock()
        succeeded_job = _job(JobStatus.State.SUCCEEDED)
        fake_client.get_job.side_effect = [
            ServiceUnavailable("transient"),
            succeeded_job.job,
        ]

        mock_logger = MagicMock(spec=PrefectLogAdapter)

        from prefect_gcp.settings import GCPBatchWorkerSettings

        with mock.patch("prefect_gcp.workers.batch.time.sleep"):
            result = GCPBatchWorker._watch_job(
                client=fake_client,
                job=_job(JobStatus.State.RUNNING),
                poll_interval=0,
                logger=mock_logger,
                settings=GCPBatchWorkerSettings(),
            )

        assert result.succeeded()
        assert fake_client.get_job.call_count == 2

    def test_watch_job_and_get_result_handles_deleted_job(
        self, gcp_batch_job_config, mock_credentials
    ):
        mock_logger = MagicMock(spec=PrefectLogAdapter)
        worker = GCPBatchWorker("my-work-pool")

        with mock.patch.object(
            worker, "_watch_job", side_effect=InfrastructureNotFound("deleted")
        ):
            result = worker._watch_job_and_get_result(
                client=MagicMock(),
                configuration=gcp_batch_job_config,
                job=_job(JobStatus.State.RUNNING),
                logger=mock_logger,
            )

        assert result.status_code == -1

    def test_watch_job_and_get_result_handles_deletion_in_progress(
        self, gcp_batch_job_config, mock_credentials
    ):
        worker = GCPBatchWorker("my-work-pool")
        fake_client = MagicMock()

        with mock.patch.object(
            worker,
            "_watch_job",
            return_value=_job(JobStatus.State.DELETION_IN_PROGRESS),
        ):
            with mock.patch.object(worker, "_log_job_logs"):
                result = worker._watch_job_and_get_result(
                    client=fake_client,
                    configuration=gcp_batch_job_config,
                    job=_job(JobStatus.State.RUNNING),
                    logger=MagicMock(spec=PrefectLogAdapter),
                )

        assert result.status_code == -1
        fake_client.delete_job.assert_not_called()


class TestGCPBatchWorkerCreateJobRetries:
    def test_create_job_retries_on_transient_error_then_succeeds(
        self, gcp_batch_job_config, mock_credentials
    ):
        gcp_batch_job_config.prepare_for_flow_run(flow_run=MockFlowRun())
        worker = GCPBatchWorker("my-work-pool")
        fake_client = MagicMock()
        succeeded_job = _job(JobStatus.State.SUCCEEDED)
        fake_client.create_job.side_effect = [
            ServiceUnavailable("transient"),
            succeeded_job.job,
        ]

        mock_logger = MagicMock(spec=PrefectLogAdapter)

        with mock.patch("prefect_gcp.workers.batch.time.sleep") as mock_sleep:
            job = worker._create_job(
                client=fake_client,
                configuration=gcp_batch_job_config,
                logger=mock_logger,
            )

        assert job.succeeded()
        assert fake_client.create_job.call_count == 2
        mock_sleep.assert_called_once()

    def test_create_job_does_not_retry_non_transient_error(
        self, gcp_batch_job_config, mock_credentials
    ):
        gcp_batch_job_config.prepare_for_flow_run(flow_run=MockFlowRun())
        worker = GCPBatchWorker("my-work-pool")
        fake_client = MagicMock()
        fake_client.create_job.side_effect = ValueError("bad request")

        with pytest.raises(ValueError):
            worker._create_job(
                client=fake_client,
                configuration=gcp_batch_job_config,
                logger=MagicMock(spec=PrefectLogAdapter),
            )

        assert fake_client.create_job.call_count == 1

    def test_create_job_recovers_from_already_exists(
        self, gcp_batch_job_config, mock_credentials
    ):
        gcp_batch_job_config.prepare_for_flow_run(flow_run=MockFlowRun())
        worker = GCPBatchWorker("my-work-pool")
        fake_client = MagicMock()
        existing_job = _job(JobStatus.State.RUNNING, uid="existing-uid")
        fake_client.create_job.side_effect = AlreadyExists("already exists")
        fake_client.get_job.return_value = existing_job.job

        job = worker._create_job(
            client=fake_client,
            configuration=gcp_batch_job_config,
            logger=MagicMock(spec=PrefectLogAdapter),
        )

        assert job.uid == "existing-uid"

    def test_create_job_wraps_not_found_with_helpful_message(
        self, gcp_batch_job_config, mock_credentials
    ):
        gcp_batch_job_config.prepare_for_flow_run(flow_run=MockFlowRun())
        worker = GCPBatchWorker("my-work-pool")
        fake_client = MagicMock()
        fake_client.create_job.side_effect = NotFound("no such region/project")

        with pytest.raises(RuntimeError, match="Confirm that region"):
            worker._create_job(
                client=fake_client,
                configuration=gcp_batch_job_config,
                logger=MagicMock(spec=PrefectLogAdapter),
            )


class TestGCPBatchWorkerKillInfrastructure:
    async def test_kill_infrastructure_deletes_job(
        self, gcp_batch_job_config, mock_credentials
    ):
        fake_client = MagicMock()

        async with GCPBatchWorker("my-work-pool") as worker:
            with mock.patch.object(
                GCPBatchWorker, "_get_client", return_value=fake_client
            ):
                await worker.kill_infrastructure(
                    infrastructure_pid="my-job-name-abc1234",
                    configuration=gcp_batch_job_config,
                )

        fake_client.delete_job.assert_called_once()

    async def test_kill_infrastructure_raises_not_found(
        self, gcp_batch_job_config, mock_credentials
    ):
        fake_client = MagicMock()
        fake_client.delete_job.side_effect = NotFound("not found")

        async with GCPBatchWorker("my-work-pool") as worker:
            with mock.patch.object(
                GCPBatchWorker, "_get_client", return_value=fake_client
            ):
                with pytest.raises(InfrastructureNotFound):
                    await worker.kill_infrastructure(
                        infrastructure_pid="my-job-name-abc1234",
                        configuration=gcp_batch_job_config,
                    )


class TestGCPBatchWorkerLogFetching:
    def test_log_job_logs_swallows_fetch_errors(
        self, gcp_batch_job_config, mock_credentials
    ):
        mock_logger = MagicMock(spec=PrefectLogAdapter)

        with mock.patch(
            "google.cloud.logging.Client", side_effect=RuntimeError("no creds")
        ):
            # Should not raise.
            GCPBatchWorker._log_job_logs(
                configuration=gcp_batch_job_config,
                job=_job(JobStatus.State.SUCCEEDED),
                logger=mock_logger,
            )

        mock_logger.warning.assert_called_once()

    def test_log_job_logs_streams_entries(self, gcp_batch_job_config, mock_credentials):
        mock_logger = MagicMock(spec=PrefectLogAdapter)
        fake_entry = MagicMock(payload="hello from batch")
        fake_logging_client = MagicMock()
        fake_logging_client.list_entries.return_value = [fake_entry]

        with mock.patch(
            "google.cloud.logging.Client", return_value=fake_logging_client
        ):
            GCPBatchWorker._log_job_logs(
                configuration=gcp_batch_job_config,
                job=_job(JobStatus.State.SUCCEEDED),
                logger=mock_logger,
            )

        mock_logger.info.assert_called_once()
        assert "hello from batch" in mock_logger.info.call_args[0][0]

    def test_log_job_logs_swallows_list_entries_errors(
        self, gcp_batch_job_config, mock_credentials
    ):
        mock_logger = MagicMock(spec=PrefectLogAdapter)
        fake_logging_client = MagicMock()
        fake_logging_client.list_entries.side_effect = RuntimeError("boom")

        with mock.patch(
            "google.cloud.logging.Client", return_value=fake_logging_client
        ):
            # Should not raise.
            GCPBatchWorker._log_job_logs(
                configuration=gcp_batch_job_config,
                job=_job(JobStatus.State.SUCCEEDED),
                logger=mock_logger,
            )

        mock_logger.warning.assert_called_once()
        assert "Failed to fetch logs" in mock_logger.warning.call_args[0][0]


class TestGCPBatchWorkerWatchJobAndGetResultDeleteFailure:
    def test_delete_failure_is_logged_but_does_not_raise(
        self, gcp_batch_job_config, mock_credentials
    ):
        worker = GCPBatchWorker("my-work-pool")
        fake_client = MagicMock()
        succeeded_job = _job(JobStatus.State.SUCCEEDED)

        mock_logger = MagicMock(spec=PrefectLogAdapter)

        with mock.patch.object(worker, "_watch_job", return_value=succeeded_job):
            with mock.patch.object(worker, "_log_job_logs"):
                with mock.patch.object(
                    BatchJob, "delete", side_effect=RuntimeError("boom")
                ):
                    result = worker._watch_job_and_get_result(
                        client=fake_client,
                        configuration=gcp_batch_job_config,
                        job=_job(JobStatus.State.RUNNING),
                        logger=mock_logger,
                    )

        assert result.status_code == 0
        mock_logger.critical.assert_called_once()
        assert "exception while deleting" in mock_logger.critical.call_args[0][0]
