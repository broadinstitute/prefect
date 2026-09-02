"""
A task runner that submits each task run as its own GCP Batch job, so a single
flow run can fan tasks out across heterogeneous machine types, CPU/memory shapes,
and spot/standard provisioning.

Because each `.submit()` provisions its own VM (tens of seconds to minutes of
cold-start), this fits coarse-grained, long-running, heterogeneous tasks rather
than fine-grained parallel task fan-out - use GCP Batch's native `TaskGroup`
parallelism (via the `GCPBatchWorker`) or a different task runner for that.

Example:
    Run one task on a large spot VM (falling back to standard if preempted too many
    times) and another on a small standard VM:
    ```python
    from prefect import flow, task
    from prefect_gcp.task_runners import GCPBatchTaskRunner
    from prefect_gcp.context import batch_options

    @task
    def process(x):
        return x + 1

    @flow(
        task_runner=GCPBatchTaskRunner(
            region="us-central1",
            result_storage_block="gcs-bucket/my-batch-task-storage",
        )
    )
    def my_flow():
        with batch_options(machine_type="n2-standard-8", max_attempts_spot=3):
            heavy = process.submit(42)
        with batch_options(cpu="1000m", memory="512Mi"):
            light = process.submit(1)
        return heavy.result(), light.result()
    ```
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import (
    Any,
    Iterable,
    Optional,
)
from uuid import uuid4

from typing_extensions import Self

from prefect.client.orchestration import get_client
from prefect.client.schemas.objects import RunInput
from prefect.context import FlowRunContext, serialize_context
from prefect.futures import PrefectDistributedFuture, PrefectFuture
from prefect.settings import get_current_settings
from prefect.states import Pending
from prefect.task_runners import TaskRunner
from prefect.tasks import Task
from prefect.utilities.asyncutils import run_coro_as_sync
from prefect.utilities.dockerutils import get_prefect_image_name
from prefect.utilities.engine import collect_task_run_inputs_sync, resolve_inputs_sync
from prefect_gcp.context import BatchOptionsContext
from prefect_gcp.credentials import GcpCredentials
from prefect_gcp.models.batch import BatchJob
from prefect_gcp.settings import GCPBatchWorkerSettings
from prefect_gcp.utilities import slugify_name
from prefect_gcp.workers.batch import (
    GCPBatchWorker,
    NFSVolume,
    _build_transient_retrying,
    _cpu_to_milli,
    _memory_to_mib,
    _nfs_volumes_to_job_body_volumes,
)

R = Any


def _validate_attempt_counts(
    max_attempts_spot: int, max_attempts_standard: int
) -> None:
    if max_attempts_spot < 0:
        raise ValueError(
            f"max_attempts_spot must be >= 0, got {max_attempts_spot!r}."
        )
    if max_attempts_standard < 1:
        raise ValueError(
            "max_attempts_standard must be >= 1, got "
            f"{max_attempts_standard!r}. At least one standard-VM attempt is "
            "required as a non-preemptible baseline."
        )


class GCPBatchTaskRunner(TaskRunner[PrefectDistributedFuture[R]]):
    """
    A task runner that submits each task run as its own GCP Batch job.

    Args:
        result_storage_block: The slug (`<block_type_slug>/<block_document_name>`)
            of a saved storage block (e.g. a `GcsBucket`) used to exchange task
            parameters between the submitting process and the Batch container. The
            block must be readable from both.
        region: The GCP region to run jobs in.
        credentials: The GCP credentials to use. If not provided, credentials will
            be inferred from the local environment.
        image: The container image to run. Must already contain the code defining
            any tasks submitted to this task runner. Defaults to the Prefect image.
        machine_type: A specific Compute Engine machine type to run tasks on (e.g.
            `e2-standard-4`). If not provided, GCP Batch chooses a machine type
            based on `cpu`/`memory`.
        cpu: The CPU to allocate to each task, in millicores (e.g. `1000m`).
        memory: The memory to allocate to each task (e.g. `512Mi`, `2Gi`).
        max_attempts_spot: The number of attempts to run a task on a spot
            (preemptible) VM before falling back to a standard VM. If `0`
            (the default), tasks skip spot entirely and run only on standard
            VMs.
        max_attempts_standard: The number of attempts to run a task on a
            standard VM, used either as the only attempts (if
            `max_attempts_spot` is `0`) or as a fallback after spot attempts
            are exhausted. Must be at least `1`, so every task always has at
            least one non-preemptible attempt.
        max_run_duration: The maximum number of seconds a task is allowed to run.
        service_account_email: The email of the service account to run VMs as.
        network: The VPC network to run VMs on.
        subnetwork: The subnetwork to run VMs on.
        boot_disk_size_gb: The boot disk size, in GB, for each VM.
        keep_job: Keep completed GCP Batch jobs on Google Cloud Platform, for
            debugging.
        nfs_volumes: NFS volumes to mount into each task's container. Each entry
            mounts the given remote path from an NFS server at the given mount
            path.

    Any of the options above (other than `result_storage_block`, `region`, and
    `credentials`) can be overridden per task submission using the
    [`batch_options`][prefect_gcp.context.batch_options] context manager.
    """

    def __init__(
        self,
        result_storage_block: str,
        region: str = "us-central1",
        credentials: Optional[GcpCredentials] = None,
        image: Optional[str] = None,
        machine_type: Optional[str] = None,
        cpu: str = "1000m",
        memory: str = "512Mi",
        max_attempts_spot: int = 0,
        max_attempts_standard: int = 1,
        max_run_duration: int = 3600,
        service_account_email: Optional[str] = None,
        network: Optional[str] = None,
        subnetwork: Optional[str] = None,
        boot_disk_size_gb: Optional[int] = None,
        keep_job: bool = False,
        nfs_volumes: Optional[list[NFSVolume]] = None,
    ):
        _validate_attempt_counts(max_attempts_spot, max_attempts_standard)

        self.result_storage_block = result_storage_block
        self.region = region
        self.credentials = credentials or GcpCredentials()
        self.image = image
        self.machine_type = machine_type
        self.cpu = cpu
        self.memory = memory
        self.max_attempts_spot = max_attempts_spot
        self.max_attempts_standard = max_attempts_standard
        self.max_run_duration = max_run_duration
        self.service_account_email = service_account_email
        self.network = network
        self.subnetwork = subnetwork
        self.boot_disk_size_gb = boot_disk_size_gb
        self.keep_job = keep_job
        self.nfs_volumes = list(nfs_volumes or [])

        self._init_kwargs = dict(
            result_storage_block=result_storage_block,
            region=region,
            credentials=credentials,
            image=image,
            machine_type=machine_type,
            cpu=cpu,
            memory=memory,
            max_attempts_spot=max_attempts_spot,
            max_attempts_standard=max_attempts_standard,
            max_run_duration=max_run_duration,
            service_account_email=service_account_email,
            network=network,
            subnetwork=subnetwork,
            boot_disk_size_gb=boot_disk_size_gb,
            keep_job=keep_job,
            nfs_volumes=nfs_volumes,
        )

        self._executor: Optional[ThreadPoolExecutor] = None
        self._watcher_futures: set[Future[None]] = set()

        super().__init__()

    def __enter__(self) -> Self:
        super().__enter__()
        self._executor = ThreadPoolExecutor(thread_name_prefix="gcp-batch-watcher")
        self._watcher_futures = set()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        super().__exit__(exc_type, exc_value, traceback)

    def duplicate(self) -> Self:
        """Return a new instance with the same settings as this one."""
        return type(self)(**self._init_kwargs)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, GCPBatchTaskRunner):
            return self._init_kwargs == other._init_kwargs
        return False

    def submit(
        self,
        task: "Task[Any, R]",
        parameters: dict[str, Any],
        wait_for: Optional[Iterable[PrefectFuture[Any]]] = None,
        dependencies: Optional[dict[str, set[RunInput]]] = None,
    ) -> PrefectDistributedFuture[R]:
        if not self._started:
            raise RuntimeError(
                "The task runner must be started before submitting work."
            )

        task_run_id = uuid4()

        task_inputs = {
            k: collect_task_run_inputs_sync(v, future_cls=PrefectDistributedFuture)
            for k, v in parameters.items()
        }

        # The Batch container starts fresh with no access to this process's
        # future objects, so any upstream futures/states embedded in the
        # parameters or wait_for must be resolved to concrete values now.
        parameters = resolve_inputs_sync(parameters)
        if wait_for:
            resolve_inputs_sync({"__wait_for__": list(wait_for)})

        entrypoint = self._entrypoint_for_task(task)

        context = serialize_context(
            asset_ctx_kwargs={
                "task": task,
                "task_run_id": task_run_id,
                "task_inputs": task_inputs,
                "copy_to_child_ctx": True,
            }
        )

        options = dict(BatchOptionsContext.get().current_batch_options)
        max_attempts_spot = options.get("max_attempts_spot", self.max_attempts_spot)
        max_attempts_standard = options.get(
            "max_attempts_standard", self.max_attempts_standard
        )
        _validate_attempt_counts(max_attempts_spot, max_attempts_standard)

        # `PrefectDistributedFuture` (returned below) assumes the task run it
        # points to already exists server-side as soon as it's returned - the
        # same contract `Task.apply_async` upholds by calling
        # `client.create_task_run` before constructing its future. Without
        # this, a caller that calls `.result()`/`.wait()` right after
        # `submit()` races the GCP Batch container (which otherwise creates
        # this row itself once it starts, tens of seconds to minutes later)
        # and gets `ObjectNotFound` instead of waiting.
        flow_run_context = FlowRunContext.get()
        with get_client(sync_client=True) as client:
            client.create_task_run(
                task=task,
                flow_run_id=(
                    flow_run_context.flow_run.id
                    if flow_run_context and flow_run_context.flow_run
                    else None
                ),
                dynamic_key=str(task_run_id),
                id=task_run_id,
                state=Pending(),
                task_inputs=task_inputs,
            )

        self._store_run_data(
            task_run_id=task_run_id,
            parameters=parameters,
            context=context,
            dependencies=dependencies,
        )

        assert self._executor is not None, (
            "The task runner must be started before submitting work."
        )
        future = self._executor.submit(
            self._run_and_maybe_fallback,
            task_run_id=task_run_id,
            entrypoint=entrypoint,
            task_name=task.name,
            options=options,
            max_attempts_spot=max_attempts_spot,
            max_attempts_standard=max_attempts_standard,
        )
        self._watcher_futures.add(future)
        future.add_done_callback(self._watcher_futures.discard)

        return PrefectDistributedFuture[R](task_run_id=task_run_id)

    def _run_and_maybe_fallback(
        self,
        task_run_id: Any,
        entrypoint: str,
        task_name: str,
        options: dict[str, Any],
        max_attempts_spot: int,
        max_attempts_standard: int,
    ) -> None:
        """
        Submit and watch a GCP Batch job for this task run, first on a spot VM
        (if `max_attempts_spot > 0`), falling back to a standard VM if the spot
        attempts are exhausted without succeeding.

        Runs on a background watcher thread so `submit()` can return
        immediately. If every attempt fails without the container ever
        starting the task, no terminal state is ever written for the task
        run, and the returned future's `.result()` will not resolve on its
        own - this is a known limitation, mitigated (not eliminated) by always
        requiring at least one standard-VM attempt.
        """
        client = self.credentials.get_batch_client()
        settings = GCPBatchWorkerSettings()
        keep_job = options.get("keep_job", self.keep_job)

        attempt_plan: list[tuple[str, int]] = []
        if max_attempts_spot > 0:
            attempt_plan.append(("SPOT", max_attempts_spot))
        attempt_plan.append(("STANDARD", max_attempts_standard))

        for i, (provisioning_model, attempts) in enumerate(attempt_plan):
            is_last_attempt_tier = i == len(attempt_plan) - 1
            job = None
            try:
                job = self._submit_batch_job(
                    client=client,
                    task_run_id=task_run_id,
                    entrypoint=entrypoint,
                    task_name=task_name,
                    options=options,
                    provisioning_model=provisioning_model,
                    max_retry_count=attempts - 1,
                )
                job = GCPBatchWorker._watch_job(
                    client=client,
                    job=job,
                    poll_interval=settings.job_watch_poll_interval_seconds,
                    logger=self.logger,
                    settings=settings,
                )
            except Exception:
                self.logger.exception(
                    "Error submitting/watching GCP Batch job for task run %s "
                    "on a %s VM",
                    task_run_id,
                    provisioning_model,
                )

            if job is not None:
                if job.was_preempted():
                    self.logger.warning(
                        "GCP Batch job %r for task run %s was preempted.",
                        job.name,
                        task_run_id,
                    )
                if not keep_job:
                    try:
                        BatchJob.delete(client=client, job_name=job.name)
                    except Exception as exc:
                        self.logger.critical(
                            "Received an exception while deleting the GCP "
                            "Batch job %r - %s",
                            job.name,
                            exc,
                        )
                if job.succeeded():
                    return

            if not is_last_attempt_tier:
                self.logger.warning(
                    "Task run %s exhausted its %s attempts on %s; falling "
                    "back to a standard VM.",
                    task_run_id,
                    attempts,
                    provisioning_model,
                )

        self.logger.error(
            "Task run %s exhausted all attempts (max_attempts_spot=%s, "
            "max_attempts_standard=%s) without succeeding.",
            task_run_id,
            max_attempts_spot,
            max_attempts_standard,
        )

    @staticmethod
    def _entrypoint_for_task(task: "Task[Any, R]") -> str:
        """
        Compute the `module:function` import path GCP Batch's container will use
        to reconstruct the task's function.
        """
        fn = task.fn
        if "<locals>" in fn.__qualname__:
            raise ValueError(
                f"Task {task.name!r} is not defined at module level, so it cannot "
                "be reconstructed inside a GCP Batch container. Define tasks "
                "submitted to GCPBatchTaskRunner as top-level functions in an "
                "importable module."
            )
        return f"{fn.__module__}:{fn.__qualname__}"

    def _store_run_data(
        self,
        task_run_id: Any,
        parameters: dict[str, Any],
        context: dict[str, Any],
        dependencies: Optional[dict[str, set[RunInput]]],
    ) -> None:
        """
        Persist the resolved parameters, serialized context, and dependencies for
        this task run to the shared result storage, so the Batch container can
        read them back by `task_run_id`.
        """
        from prefect.blocks.core import Block
        from prefect.results import ResultStore
        from prefect.task_worker import store_parameters

        block = Block.load(self.result_storage_block)
        result_store = ResultStore(result_storage=block)
        run_coro_as_sync(
            store_parameters(
                result_store,
                task_run_id,
                {
                    "parameters": parameters,
                    "context": context,
                    "dependencies": dependencies,
                },
            )
        )

    def _submit_batch_job(
        self,
        client: Any,
        task_run_id: Any,
        entrypoint: str,
        task_name: str,
        options: dict[str, Any],
        provisioning_model: str,
        max_retry_count: int,
    ) -> BatchJob:
        """
        Build and create the GCP Batch job that will run this single task.
        """
        job_body = self._build_job_body(
            task_run_id=task_run_id,
            entrypoint=entrypoint,
            options=options,
            provisioning_model=provisioning_model,
            max_retry_count=max_retry_count,
        )
        job_id = (
            f"{slugify_name(task_name, max_length=40)}-{uuid4().hex[:7]}"
            if slugify_name(task_name, max_length=40)
            else f"prefect-task-{uuid4().hex[:7]}"
        )

        retrying = _build_transient_retrying(
            max_attempts=3,
            initial_delay=1.0,
            max_delay=10.0,
            operation_label="creating job",
            logger=self.logger,
        )
        job = None
        for attempt in retrying:
            with attempt:
                job = BatchJob.create(
                    client=client,
                    project=self.credentials.project,
                    region=self.region,
                    job_id=job_id,
                    job=job_body,
                )
        assert job is not None
        return job

    def _build_job_body(
        self,
        task_run_id: Any,
        entrypoint: str,
        options: dict[str, Any],
        provisioning_model: str,
        max_retry_count: int,
    ) -> dict[str, Any]:
        """
        Build the GCP Batch `Job` request body for a single task run.
        """
        image = options.get("image", self.image) or (
            f"docker.io/{get_prefect_image_name()}"
        )
        cpu = options.get("cpu", self.cpu)
        memory = options.get("memory", self.memory)
        machine_type = options.get("machine_type", self.machine_type)
        max_run_duration = options.get("max_run_duration", self.max_run_duration)
        service_account_email = options.get(
            "service_account_email", self.service_account_email
        )
        network = options.get("network", self.network)
        subnetwork = options.get("subnetwork", self.subnetwork)
        boot_disk_size_gb = options.get("boot_disk_size_gb", self.boot_disk_size_gb)
        nfs_volumes = options.get("nfs_volumes", self.nfs_volumes)

        env = get_current_settings().to_environment_variables(exclude_unset=True)

        instance_policy: dict[str, Any] = {
            "provisioning_model": provisioning_model,
        }
        if machine_type:
            instance_policy["machine_type"] = machine_type
        if boot_disk_size_gb:
            instance_policy["boot_disk"] = {"size_gb": boot_disk_size_gb}

        allocation_policy: dict[str, Any] = {"instances": [{"policy": instance_policy}]}
        if service_account_email:
            allocation_policy["service_account"] = {"email": service_account_email}
        if network or subnetwork:
            allocation_policy["network"] = {
                "network_interfaces": [
                    {"network": network, "subnetwork": subnetwork}
                ],
            }

        task_spec: dict[str, Any] = {
            "runnables": [
                {
                    "container": {
                        "image_uri": image,
                        "commands": [
                            "python",
                            "-m",
                            "prefect_gcp.batch_task_entrypoint",
                            "--task-run-id",
                            str(task_run_id),
                            "--entrypoint",
                            entrypoint,
                            "--result-storage-block",
                            self.result_storage_block,
                        ],
                    },
                },
            ],
            "compute_resource": {
                "cpu_milli": _cpu_to_milli(cpu),
                "memory_mib": _memory_to_mib(memory),
            },
            "max_run_duration": f"{max_run_duration}s",
            "max_retry_count": max_retry_count,
            "environment": {"variables": env},
        }
        if nfs_volumes:
            task_spec["volumes"] = _nfs_volumes_to_job_body_volumes(nfs_volumes)

        return {
            "task_groups": [
                {
                    "task_spec": task_spec,
                    "task_count": 1,
                    "parallelism": 1,
                },
            ],
            "allocation_policy": allocation_policy,
            "logs_policy": {"destination": "CLOUD_LOGGING"},
            "labels": {"prefect-task-run-id": str(task_run_id)},
        }
