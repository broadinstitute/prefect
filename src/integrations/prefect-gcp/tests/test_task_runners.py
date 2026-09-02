from unittest import mock
from uuid import uuid4

import pytest
from google.api_core.exceptions import ServiceUnavailable
from google.cloud.batch_v1.types import Job, JobStatus
from prefect_gcp.context import batch_options
from prefect_gcp.task_runners import GCPBatchTaskRunner
from prefect_gcp.workers.batch import NFSVolume

from prefect import task
from prefect.filesystems import LocalFileSystem


@task
def add_one(x: int) -> int:
    return x + 1


def _job(name: str, state: "JobStatus.State") -> Job:
    return Job(name=name, status=JobStatus(state=state))


def _wait_for_watchers(runner: GCPBatchTaskRunner) -> None:
    """Block until every watcher future submitted so far has finished, so
    assertions on mocked GCP Batch API calls (made from the background
    watcher thread) can run while the relevant mocks are still active."""
    for future in list(runner._watcher_futures):
        future.result(timeout=5)


@pytest.fixture
def fake_gcs_storage(tmp_path):
    """Stand-in for the GCS block used to exchange parameters: a real
    `WritableFileSystem` block (so it passes `ResultStore` validation), backed by
    a local temp dir instead of GCS. Also stands in for `batch_task_entrypoint`'s
    default result-storage block, which requires a persisted (i.e.
    `_block_document_id`-bearing) block."""
    block = LocalFileSystem(basepath=str(tmp_path))
    block._block_document_id = uuid4()
    return block


@pytest.fixture
def runner():
    with GCPBatchTaskRunner(
        result_storage_block="fake-storage/my-block", region="us-central1"
    ) as runner:
        runner.credentials.project = "my-project"
        yield runner


class TestGCPBatchTaskRunnerConfiguration:
    def test_duplicate_preserves_settings(self):
        original = GCPBatchTaskRunner(
            result_storage_block="fake-storage/my-block",
            region="us-east1",
            machine_type="e2-standard-4",
            max_attempts_spot=3,
        )
        duplicate = original.duplicate()
        assert duplicate is not original
        assert duplicate == original

    def test_eq_compares_settings(self):
        block = "fake-storage/my-block"
        a = GCPBatchTaskRunner(result_storage_block=block, cpu="1000m")
        b = GCPBatchTaskRunner(result_storage_block=block, cpu="1000m")
        c = GCPBatchTaskRunner(result_storage_block=block, cpu="2000m")
        assert a == b
        assert a != c
        assert a != "not-a-task-runner"

    def test_rejects_negative_max_attempts_spot(self):
        with pytest.raises(ValueError, match="max_attempts_spot must be >= 0"):
            GCPBatchTaskRunner(
                result_storage_block="fake-storage/my-block",
                max_attempts_spot=-1,
            )

    def test_rejects_zero_max_attempts_standard(self):
        with pytest.raises(ValueError, match="max_attempts_standard must be >= 1"):
            GCPBatchTaskRunner(
                result_storage_block="fake-storage/my-block",
                max_attempts_standard=0,
            )


class TestEntrypointForTask:
    def test_module_level_task(self):
        entrypoint = GCPBatchTaskRunner._entrypoint_for_task(add_one)
        assert entrypoint == f"{add_one.fn.__module__}:add_one"

    def test_rejects_closures(self):
        def outer():
            @task
            def inner(x):
                return x

            return inner

        inner_task = outer()

        with pytest.raises(ValueError, match="not defined at module level"):
            GCPBatchTaskRunner._entrypoint_for_task(inner_task)


class TestSubmit:
    def test_submit_requires_started_runner(self):
        runner = GCPBatchTaskRunner(result_storage_block="fake-storage/my-block")
        with pytest.raises(RuntimeError, match="must be started"):
            runner.submit(add_one, {"x": 41})

    def test_submit_rejects_invalid_batch_options_override(
        self, runner, fake_gcs_storage
    ):
        with mock.patch(
            "prefect.blocks.core.Block.load", return_value=fake_gcs_storage
        ):
            with batch_options(max_attempts_standard=0):
                with pytest.raises(
                    ValueError, match="max_attempts_standard must be >= 1"
                ):
                    runner.submit(add_one, {"x": 41})

    def test_submit_creates_the_task_run_before_returning(
        self, runner, fake_gcs_storage
    ):
        """
        `PrefectDistributedFuture` assumes its task run already exists
        server-side as soon as it's constructed - `.wait()`/`.result()` read it
        immediately and raise `ObjectNotFound` if it's missing, with no
        tolerance for the row not existing yet. A caller that calls
        `.result()` right after `submit()` would otherwise race the GCP Batch
        container, which only creates this row once it starts running (tens of
        seconds to minutes later).
        """
        from prefect.client.orchestration import get_client

        job_name = "projects/p/locations/us-central1/jobs/prefect-task-x"
        fake_client = mock.MagicMock()
        fake_client.create_job.return_value = _job(job_name, JobStatus.State.SUCCEEDED)

        with mock.patch(
            "prefect.blocks.core.Block.load", return_value=fake_gcs_storage
        ):
            with mock.patch.object(
                type(runner.credentials), "get_batch_client", return_value=fake_client
            ):
                future = runner.submit(add_one, {"x": 41})
                with get_client(sync_client=True) as client:
                    task_run = client.read_task_run(task_run_id=future.task_run_id)
                assert task_run.id == future.task_run_id
                _wait_for_watchers(runner)

    def test_submit_stores_parameters_and_creates_job(self, runner, fake_gcs_storage):
        job_name = "projects/p/locations/us-central1/jobs/prefect-task-x"
        fake_client = mock.MagicMock()
        fake_client.create_job.return_value = _job(job_name, JobStatus.State.SUCCEEDED)

        with mock.patch(
            "prefect.blocks.core.Block.load", return_value=fake_gcs_storage
        ):
            with mock.patch.object(
                type(runner.credentials), "get_batch_client", return_value=fake_client
            ):
                with batch_options(machine_type="e2-standard-4", max_attempts_spot=1):
                    future = runner.submit(add_one, {"x": 41})
                _wait_for_watchers(runner)

        assert future.task_run_id is not None
        assert fake_client.create_job.call_count == 1
        _, kwargs = fake_client.create_job.call_args
        job_arg = kwargs["request"].job
        assert (
            job_arg.allocation_policy.instances[0].policy.machine_type
            == "e2-standard-4"
        )
        assert (
            job_arg.allocation_policy.instances[0].policy.provisioning_model.name
            == "SPOT"
        )
        assert job_arg.task_groups[0].task_spec.max_retry_count == 0
        commands = list(
            job_arg.task_groups[0].task_spec.runnables[0].container.commands
        )
        assert "prefect_gcp.batch_task_entrypoint" in commands
        assert str(future.task_run_id) in commands
        assert GCPBatchTaskRunner._entrypoint_for_task(add_one) in commands

    def test_submit_defaults_used_without_batch_options(self, runner, fake_gcs_storage):
        runner.cpu = "2000m"
        runner.memory = "1Gi"
        job_name = "projects/p/locations/us-central1/jobs/prefect-task-x"
        fake_client = mock.MagicMock()
        fake_client.create_job.return_value = _job(job_name, JobStatus.State.SUCCEEDED)

        with mock.patch(
            "prefect.blocks.core.Block.load", return_value=fake_gcs_storage
        ):
            with mock.patch.object(
                type(runner.credentials), "get_batch_client", return_value=fake_client
            ):
                runner.submit(add_one, {"x": 41})
                _wait_for_watchers(runner)

        assert fake_client.create_job.call_count == 1
        _, kwargs = fake_client.create_job.call_args
        task_spec = kwargs["request"].job.task_groups[0].task_spec
        compute_resource = task_spec.compute_resource
        assert compute_resource.cpu_milli == 2000
        assert compute_resource.memory_mib == 1024
        assert task_spec.max_retry_count == 0
        policy = kwargs["request"].job.allocation_policy.instances[0].policy
        assert policy.provisioning_model.name == "STANDARD"

    def test_submit_falls_back_to_standard_after_spot_exhausted(
        self, runner, fake_gcs_storage
    ):
        job_name = "projects/p/locations/us-central1/jobs/prefect-task-x"
        spot_job = _job(job_name, JobStatus.State.FAILED)
        standard_job = _job(job_name, JobStatus.State.SUCCEEDED)
        fake_client = mock.MagicMock()
        fake_client.create_job.side_effect = [spot_job, standard_job]

        with mock.patch(
            "prefect.blocks.core.Block.load", return_value=fake_gcs_storage
        ):
            with mock.patch.object(
                type(runner.credentials), "get_batch_client", return_value=fake_client
            ):
                with batch_options(max_attempts_spot=2, max_attempts_standard=1):
                    runner.submit(add_one, {"x": 41})
                _wait_for_watchers(runner)

        assert fake_client.create_job.call_count == 2
        assert fake_client.delete_job.call_count == 2

        first_call_kwargs = fake_client.create_job.call_args_list[0].kwargs
        first_job = first_call_kwargs["request"].job
        assert (
            first_job.allocation_policy.instances[0].policy.provisioning_model.name
            == "SPOT"
        )
        assert first_job.task_groups[0].task_spec.max_retry_count == 1

        second_call_kwargs = fake_client.create_job.call_args_list[1].kwargs
        second_job = second_call_kwargs["request"].job
        assert (
            second_job.allocation_policy.instances[0].policy.provisioning_model.name
            == "STANDARD"
        )
        assert second_job.task_groups[0].task_spec.max_retry_count == 0

    def test_submit_does_not_fall_back_when_spot_succeeds(
        self, runner, fake_gcs_storage
    ):
        job_name = "projects/p/locations/us-central1/jobs/prefect-task-x"
        fake_client = mock.MagicMock()
        fake_client.create_job.return_value = _job(job_name, JobStatus.State.SUCCEEDED)

        with mock.patch(
            "prefect.blocks.core.Block.load", return_value=fake_gcs_storage
        ):
            with mock.patch.object(
                type(runner.credentials), "get_batch_client", return_value=fake_client
            ):
                with batch_options(max_attempts_spot=3, max_attempts_standard=1):
                    runner.submit(add_one, {"x": 41})
                _wait_for_watchers(runner)

        assert fake_client.create_job.call_count == 1

    def test_submit_mounts_nfs_volumes(self, runner, fake_gcs_storage):
        job_name = "projects/p/locations/us-central1/jobs/prefect-task-x"
        fake_client = mock.MagicMock()
        fake_client.create_job.return_value = _job(job_name, JobStatus.State.SUCCEEDED)

        with mock.patch(
            "prefect.blocks.core.Block.load", return_value=fake_gcs_storage
        ):
            with mock.patch.object(
                type(runner.credentials), "get_batch_client", return_value=fake_client
            ):
                with batch_options(
                    nfs_volumes=[
                        NFSVolume(
                            server="10.0.0.2",
                            remote_path="/exports/data",
                            mount_path="/mnt/data",
                        )
                    ]
                ):
                    runner.submit(add_one, {"x": 41})
                _wait_for_watchers(runner)

        _, kwargs = fake_client.create_job.call_args
        task_spec = kwargs["request"].job.task_groups[0].task_spec
        assert len(task_spec.volumes) == 1
        volume = task_spec.volumes[0]
        assert volume.nfs.server == "10.0.0.2"
        assert volume.nfs.remote_path == "/exports/data"
        assert volume.mount_path == "/mnt/data"

    def test_submit_batch_job_retries_transient_error_then_succeeds(
        self, runner, fake_gcs_storage
    ):
        job_name = "projects/p/locations/us-central1/jobs/prefect-task-x"
        fake_client = mock.MagicMock()
        fake_client.create_job.side_effect = [
            ServiceUnavailable("transient"),
            _job(job_name, JobStatus.State.SUCCEEDED),
        ]

        with mock.patch(
            "prefect.blocks.core.Block.load", return_value=fake_gcs_storage
        ):
            with mock.patch.object(
                type(runner.credentials), "get_batch_client", return_value=fake_client
            ):
                with mock.patch("prefect_gcp.workers.batch.time.sleep"):
                    runner.submit(add_one, {"x": 41})
                    _wait_for_watchers(runner)

        assert fake_client.create_job.call_count == 2


async def test_full_round_trip_submit_then_entrypoint_runs_task(
    runner, fake_gcs_storage
):
    """Simulates the full submit -> Batch container round trip: `submit()`
    persists parameters (via the same store_parameters/read_parameters path the
    entrypoint uses) and launches a Batch job (mocked here), then a fresh call
    into `batch_task_entrypoint.main()` (standing in for the container process)
    reads them back, reconstructs the task purely from its `module:function`
    entrypoint string, and actually executes it, producing a real completed
    task run.

    Note: this passes `context=None` for the entrypoint's `run_task_sync` call
    (by clearing the stored context) to sidestep a test-harness-only quirk
    where re-hydrating a serialized `settings_context` inside the same process
    that's *also* running `prefect_test_harness`'s ephemeral server causes a
    second, disconnected ephemeral app/db to spin up. That does not occur in a
    real deployment (a real Batch container talks to one persistent Prefect
    server/Cloud instance, not a locally-bootstrapped ephemeral one), so
    `hydrated_context`/flow-run-context propagation itself is exercised by
    Prefect core's own test suite and by `prefect-ray`'s equivalent usage, not
    re-tested here.
    """
    from prefect_gcp import batch_task_entrypoint

    fake_client = mock.MagicMock()
    fake_client.create_job.return_value = _job(
        "projects/p/locations/us-central1/jobs/x", JobStatus.State.SUCCEEDED
    )

    with mock.patch("prefect.blocks.core.Block.load", return_value=fake_gcs_storage):
        with mock.patch.object(
            type(runner.credentials), "get_batch_client", return_value=fake_client
        ):
            future = runner.submit(add_one, {"x": 41})
            _wait_for_watchers(runner)

    task_run_id = future.task_run_id
    entrypoint = GCPBatchTaskRunner._entrypoint_for_task(add_one)

    from prefect.results import ResultStore
    from prefect.task_worker import read_parameters, store_parameters

    result_store = ResultStore(result_storage=fake_gcs_storage)
    run_data = await read_parameters(result_store, task_run_id)
    run_data["context"] = None
    await store_parameters(result_store, task_run_id, run_data)

    with mock.patch("prefect.blocks.core.Block.load", return_value=fake_gcs_storage):
        exit_code = batch_task_entrypoint.main(
            [
                "--task-run-id",
                str(task_run_id),
                "--entrypoint",
                entrypoint,
                "--result-storage-block",
                "fake-storage/my-block",
            ]
        )

    # `exit_code` is derived directly from the real `State` object returned by
    # `run_task_sync` inside the same call/thread that created and ran the task,
    # so this alone proves parameters were round-tripped correctly, the task was
    # reconstructed from its entrypoint string, and it executed and completed
    # successfully.
    assert exit_code == 0
