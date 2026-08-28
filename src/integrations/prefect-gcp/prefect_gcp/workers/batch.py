"""
The GCP Batch worker.

Runs each flow run as its own Google Cloud Batch job, giving access to
Batch's full machine-type catalog, custom CPU/memory shapes, and
spot/preemptible VMs.
"""

from __future__ import annotations

import re
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Final,
    Iterable,
    List,
    Literal,
    Optional,
    TypeVar,
)
from uuid import uuid4

from google.api_core.exceptions import (
    AlreadyExists,
    DeadlineExceeded,
    InternalServerError,
    NotFound,
    ServiceUnavailable,
    TooManyRequests,
)
from jsonpatch import JsonPatch
from pydantic import BaseModel, Field, PrivateAttr, field_validator
from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from prefect.exceptions import InfrastructureNotFound
from prefect.logging.loggers import PrefectLogAdapter
from prefect.utilities.asyncutils import run_sync_in_worker_thread
from prefect.utilities.dockerutils import get_prefect_image_name
from prefect.utilities.processutils import command_from_string
from prefect.workers.base import (
    BaseJobConfiguration,
    BaseVariables,
    BaseWorker,
    BaseWorkerResult,
)
from prefect_gcp.credentials import GcpCredentials
from prefect_gcp.models.batch import BatchJob
from prefect_gcp.settings import GCPBatchWorkerSettings
from prefect_gcp.utilities import merge_labels_for_gcp, slugify_name

if TYPE_CHECKING:
    from anyio.abc import TaskStatus
    from google.cloud.batch_v1 import BatchServiceClient

    from prefect.client.schemas.objects import FlowRun

_BATCH_JOB_NAME_MAX_LENGTH: Final[int] = 63
_BATCH_JOB_NAME_UUID_LENGTH: Final[int] = 7

_CPU_MILLI_PATTERN: Final[re.Pattern] = re.compile(r"^(\d+)m$")
_MEMORY_PATTERN: Final[re.Pattern] = re.compile(r"^(\d+)(G|Gi|M|Mi)$")

_TRANSIENT_EXCEPTIONS: Final[tuple[type[Exception], ...]] = (
    ServiceUnavailable,
    TooManyRequests,
    InternalServerError,
    DeadlineExceeded,
)


def _cpu_to_milli(cpu: str) -> int:
    """Convert a `"1000m"`-style CPU string to millicores for
    `ComputeResource.cpu_milli`."""
    match = _CPU_MILLI_PATTERN.match(cpu)
    if not match:
        raise ValueError(f"Invalid cpu value {cpu!r}; expected e.g. '1000m'")
    return int(match.group(1))


def _memory_to_mib(memory: str) -> int:
    """Convert a `"512Mi"`/`"2Gi"`/`"512M"`/`"2G"`-style memory string to
    mebibytes for `ComputeResource.memory_mib`."""
    match = _MEMORY_PATTERN.match(memory)
    if not match:
        raise ValueError(f"Invalid memory value {memory!r}; expected e.g. '512Mi'")
    quantity, unit = match.groups()
    quantity = int(quantity)
    if unit == "Mi":
        return quantity
    if unit == "Gi":
        return quantity * 1024
    if unit == "M":
        return round(quantity * 1_000_000 / (1024 * 1024))
    # unit == "G"
    return round(quantity * 1_000_000_000 / (1024 * 1024))


def _get_default_job_body_template() -> Dict[str, Any]:
    """
    Returns the default job body template for the GCP Batch worker.

    Field names match the Python attribute names of `google.cloud.batch_v1`
    types (snake_case), since the rendered `job_body` is passed directly to
    `batch_v1.types.Job(job_body)` when the job is submitted.
    """
    return {
        "task_groups": [
            {
                "task_spec": {
                    "runnables": [
                        {
                            "container": {
                                "image_uri": "{{ image }}",
                                "commands": "{{ command }}",
                            },
                        },
                    ],
                    "compute_resource": {},
                    "max_run_duration": "{{ max_run_duration }}",
                    "max_retry_count": "{{ max_retry_count }}",
                    "environment": {"variables": {}},
                },
                "task_count": 1,
                "parallelism": 1,
            },
        ],
        "allocation_policy": {
            "instances": [
                {
                    "policy": {
                        "machine_type": "{{ machine_type }}",
                        "provisioning_model": "{{ provisioning_model }}",
                        "boot_disk": {"size_gb": "{{ boot_disk_size_gb }}"},
                    },
                },
            ],
            "service_account": {"email": "{{ service_account_email }}"},
            "network": {
                "network_interfaces": [
                    {"network": "{{ network }}", "subnetwork": "{{ subnetwork }}"},
                ],
            },
            "location": {"allowed_locations": "{{ allowed_locations }}"},
        },
        "logs_policy": {"destination": "CLOUD_LOGGING"},
        "labels": {},
    }


def _get_base_job_body() -> Dict[str, Any]:
    """
    Returns the base job body for the GCP Batch worker's job body validation.
    """
    return {
        "task_groups": [
            {
                "task_spec": {
                    "runnables": [],
                },
            },
        ],
    }


class NFSVolume(BaseModel):
    """An NFS volume to mount into a GCP Batch task's container."""

    server: str = Field(description="The IP address or hostname of the NFS server.")
    remote_path: str = Field(
        description="The path exported by the NFS server to mount."
    )
    mount_path: str = Field(
        description="The path to mount the volume at inside the container."
    )


def _nfs_volumes_to_job_body_volumes(
    volumes: "Iterable[NFSVolume]",
) -> List[Dict[str, Any]]:
    return [
        {
            "nfs": {"server": v.server, "remote_path": v.remote_path},
            "mount_path": v.mount_path,
        }
        for v in volumes
    ]


class GCPBatchJobConfiguration(BaseJobConfiguration):
    """
    The configuration for the GCP Batch worker.

    The schema for this class is used to populate the `job_body` section of the
    default base job template.
    """

    credentials: GcpCredentials = Field(
        title="GCP Credentials",
        default_factory=GcpCredentials,
        description=(
            "The GCP Credentials used to connect to GCP Batch. "
            "If not provided credentials will be inferred from "
            "the local environment."
        ),
    )
    region: str = Field(
        default="us-central1",
        description="The region in which to run the GCP Batch job.",
    )
    job_body: Dict[str, Any] = Field(
        json_schema_extra=dict(template=_get_default_job_body_template()),
    )
    keep_job: bool = Field(
        default=False,
        title="Keep Job After Completion",
        description="Keep the completed GCP Batch job on Google Cloud Platform.",
    )
    cpu: str = Field(
        default="1000m",
        title="CPU",
        description=(
            "The CPU to allocate to each task, in millicores (e.g. '1000m' "
            "for one vCPU). Used to size the task's compute resource request; "
            "the actual VM shape is chosen by 'machine_type' if set, or "
            "inferred from 'cpu'/'memory' otherwise."
        ),
        pattern=r"^\d+m$",
    )
    memory: str = Field(
        default="512Mi",
        title="Memory",
        description=(
            "The memory to allocate to each task, along with its units, "
            "which could be: G, Gi, M, Mi."
        ),
        examples=["512Mi"],
        pattern=r"^\d+(?:G|Gi|M|Mi)$",
    )
    provisioning_model: Literal["standard", "spot"] = Field(
        default="standard",
        title="Provisioning Model",
        description=(
            "Whether to run the job on standard or spot (preemptible) VMs. "
            "Spot VMs are substantially cheaper but may be preempted; set "
            "'max_retry_count' above 0 to have Batch automatically retry a "
            "preempted task."
        ),
    )
    max_run_duration: int = Field(
        default=3600,
        gt=0,
        title="Max Run Duration",
        description=(
            "The maximum number of seconds a task is allowed to run before "
            "GCP Batch will actively try to mark it failed."
        ),
    )
    nfs_volumes: List[NFSVolume] = Field(
        default_factory=list,
        title="NFS Volumes",
        description=(
            "NFS volumes to mount into the job's containers. Each entry mounts "
            "the given remote path from an NFS server at the given mount path."
        ),
    )
    _job_name: Optional[str] = PrivateAttr(default=None)

    @property
    def project(self) -> str:
        """
        Returns the GCP project associated with the credentials.
        """
        return self.credentials.project

    @property
    def job_name(self) -> str:
        """
        Returns the name of the job, generating and caching one the first
        time it's accessed.
        """
        if self._job_name is None:
            base_job_name = slugify_name(
                self.name,
                max_length=_BATCH_JOB_NAME_MAX_LENGTH
                - 1
                - _BATCH_JOB_NAME_UUID_LENGTH,
            )
            self._job_name = (
                f"{base_job_name}-{uuid4().hex[:_BATCH_JOB_NAME_UUID_LENGTH]}"
            )

        return self._job_name

    def prepare_for_flow_run(
        self,
        flow_run,
        deployment=None,
        flow=None,
        work_pool=None,
        worker_name=None,
        worker_id=None,
    ):
        """
        Prepares the job configuration for a flow run.

        Ensures that necessary values are present in the job body and that the
        job body is valid.
        """
        super().prepare_for_flow_run(
            flow_run=flow_run,
            deployment=deployment,
            flow=flow,
            work_pool=work_pool,
            worker_name=worker_name,
            worker_id=worker_id,
        )

        self._populate_image_if_not_present()
        self._populate_or_format_command()
        self._populate_env()
        self._populate_compute_resource()
        self._populate_provisioning_model()
        self._populate_max_run_duration()
        self._populate_volumes()
        self._populate_labels()

    def _populate_image_if_not_present(self):
        """
        Populates the job body with the image if not present.
        """
        container = self.job_body["task_groups"][0]["task_spec"]["runnables"][0][
            "container"
        ]
        if "image_uri" not in container:
            container["image_uri"] = f"docker.io/{get_prefect_image_name()}"

    def _populate_or_format_command(self):
        """
        Populates the job body with the command if not present, or formats it
        from a string into the list of strings that GCP Batch expects.
        """
        container = self.job_body["task_groups"][0]["task_spec"]["runnables"][0][
            "container"
        ]
        command = container.get("commands")

        if command is None:
            container["commands"] = command_from_string(self._base_flow_run_command())
        elif isinstance(command, str):
            container["commands"] = command_from_string(command)

    def _populate_env(self):
        """
        Populates the job body with environment variables.
        """
        task_spec = self.job_body["task_groups"][0]["task_spec"]
        task_spec.setdefault("environment", {}).setdefault("variables", {}).update(
            self.env
        )

    def _populate_compute_resource(self):
        """
        Populates the job body's compute resource with the CPU/memory
        requested for each task, converted to the millicore/mebibyte units
        that GCP Batch's `ComputeResource` expects.
        """
        task_spec = self.job_body["task_groups"][0]["task_spec"]
        compute_resource = task_spec.setdefault("compute_resource", {})
        compute_resource["cpu_milli"] = _cpu_to_milli(self.cpu)
        compute_resource["memory_mib"] = _memory_to_mib(self.memory)

    def _populate_provisioning_model(self):
        """
        Populates the job body's allocation policy with the requested
        provisioning model, formatted as the uppercase enum name GCP Batch
        expects.
        """
        policy = self.job_body["allocation_policy"]["instances"][0]["policy"]
        policy["provisioning_model"] = self.provisioning_model.upper()

    def _populate_max_run_duration(self):
        """
        Populates the job body with the max run duration, formatted as a
        `Duration`-style string (e.g. `"3600s"`) as GCP Batch expects.
        """
        task_spec = self.job_body["task_groups"][0]["task_spec"]
        task_spec["max_run_duration"] = f"{self.max_run_duration}s"

    def _populate_volumes(self):
        """
        Populates the job body's task spec with any configured NFS volumes.
        """
        if not self.nfs_volumes:
            return
        task_spec = self.job_body["task_groups"][0]["task_spec"]
        task_spec.setdefault("volumes", []).extend(
            _nfs_volumes_to_job_body_volumes(self.nfs_volumes)
        )

    def _populate_labels(self):
        """
        Injects sanitized Prefect labels into the GCP Batch job body.
        """
        existing = self.job_body.get("labels", {})
        self.job_body["labels"] = merge_labels_for_gcp(self.labels, existing)

    # noinspection PyMethodParameters
    @field_validator("job_body")
    @classmethod
    def _ensure_job_includes_all_required_components(cls, value: Dict[str, Any]):
        """
        Ensures that the job body includes all required components.
        """
        patch = JsonPatch.from_diff(value, _get_base_job_body())

        missing_paths = sorted([op["path"] for op in patch if op["op"] == "add"])

        if missing_paths:
            raise ValueError(
                f"Job body is missing required components: {', '.join(missing_paths)}"
            )

        return value

    # noinspection PyMethodParameters
    @field_validator("job_body")
    @classmethod
    def _ensure_job_has_compatible_values(cls, value: Dict[str, Any]):
        """Ensure that the job body has compatible values."""
        patch = JsonPatch.from_diff(value, _get_base_job_body())
        incompatible = sorted(
            [
                f"{op['path']} must have value {op['value']!r}"
                for op in patch
                if op["op"] == "replace"
            ]
        )
        if incompatible:
            raise ValueError(
                "Job has incompatible values for the following attributes: "
                f"{', '.join(incompatible)}"
            )
        return value


class GCPBatchVariables(BaseVariables):
    """
    Default variables for the GCP Batch worker.

    The schema for this class is used to populate the `variables` section of the
    default base job template.
    """

    credentials: GcpCredentials = Field(
        title="GCP Credentials",
        default_factory=GcpCredentials,
        description=(
            "The GCP Credentials used to connect to GCP Batch. "
            "If not provided credentials will be inferred from "
            "the local environment."
        ),
    )
    region: str = Field(
        default="us-central1",
        description="The region in which to run the GCP Batch job.",
    )
    image: Optional[str] = Field(
        default="prefecthq/prefect:3-latest",
        title="Image Name",
        description=(
            "The image to use for the GCP Batch job. "
            "If not provided the default Prefect image will be used."
        ),
    )
    machine_type: Optional[str] = Field(
        default=None,
        title="Machine Type",
        description=(
            "The Compute Engine machine type to run the job on, e.g. "
            "'e2-standard-4'. If not provided, GCP Batch will choose a "
            "machine type based on 'cpu' and 'memory'."
        ),
        examples=["e2-standard-4"],
    )
    cpu: str = Field(
        default="1000m",
        title="CPU",
        description=(
            "The CPU to allocate to each task, in millicores (e.g. '1000m' "
            "for one vCPU)."
        ),
        pattern=r"^\d+m$",
    )
    memory: str = Field(
        default="512Mi",
        title="Memory",
        description=(
            "The memory to allocate to each task, along with its units, "
            "which could be: G, Gi, M, Mi."
        ),
        examples=["512Mi"],
        pattern=r"^\d+(?:G|Gi|M|Mi)$",
    )
    provisioning_model: Literal["standard", "spot"] = Field(
        default="standard",
        title="Provisioning Model",
        description=(
            "Whether to run the job on standard or spot (preemptible) VMs. "
            "Spot VMs are substantially cheaper but may be preempted; set "
            "'max_retry_count' above 0 to have Batch automatically retry a "
            "preempted task."
        ),
    )
    boot_disk_size_gb: Optional[int] = Field(
        default=None,
        title="Boot Disk Size (GB)",
        description=(
            "The size of the boot disk, in GB, to attach to each VM. If not "
            "provided, GCP Batch will use its default boot disk size."
        ),
    )
    service_account_email: Optional[str] = Field(
        default=None,
        title="Service Account Email",
        description=(
            "The email of the service account to run the job's VMs as. If "
            "not provided, the default Compute Engine service account is "
            "used."
        ),
        examples=["service-account@example.iam.gserviceaccount.com"],
    )
    network: Optional[str] = Field(
        default=None,
        title="Network",
        description="The VPC network to run the job's VMs on.",
    )
    subnetwork: Optional[str] = Field(
        default=None,
        title="Subnetwork",
        description="The subnetwork to run the job's VMs on.",
    )
    allowed_locations: Optional[List[str]] = Field(
        default=None,
        title="Allowed Locations",
        description=(
            "A list of zones/regions the job's VMs are allowed to run in, "
            "e.g. ['zones/us-central1-a']. Restricting this can slow down "
            "scheduling; leave unset to allow any zone in 'region'."
        ),
        examples=[["zones/us-central1-a"]],
    )
    max_run_duration: int = Field(
        default=3600,
        gt=0,
        title="Max Run Duration",
        description=(
            "The maximum number of seconds a task is allowed to run before "
            "GCP Batch will actively try to mark it failed."
        ),
    )
    max_retry_count: int = Field(
        default=0,
        title="Max Retry Count",
        description=(
            "The number of times GCP Batch should retry a failed task, e.g. "
            "after a spot VM preemption."
        ),
    )
    keep_job: bool = Field(
        default=False,
        title="Keep Job After Completion",
        description="Keep the completed GCP Batch job on Google Cloud Platform.",
    )
    nfs_volumes: List[NFSVolume] = Field(
        default_factory=list,
        title="NFS Volumes",
        description=(
            "NFS volumes to mount into the job's containers. Each entry mounts "
            "the given remote path from an NFS server at the given mount path."
        ),
    )


class GCPBatchWorkerResult(BaseWorkerResult):
    """
    The result of a GCP Batch worker job.
    """


def _is_transient_error(exc: Exception) -> bool:
    return isinstance(exc, _TRANSIENT_EXCEPTIONS)


_OperationLabel = Literal["creating job", "polling job status"]


def _build_transient_retrying(
    *,
    max_attempts: int,
    initial_delay: float,
    max_delay: float,
    operation_label: _OperationLabel,
    logger: PrefectLogAdapter,
) -> Retrying:
    """Build a Retrying that retries transient GCP Batch API errors with
    exponential jitter."""

    def _log_retry(retry_state) -> None:
        exc = retry_state.outcome.exception()
        delay = retry_state.next_action.sleep
        logger.warning(
            "Transient error (%s) when %s. Retrying in %.2fs... (Attempt %s/%s)",
            type(exc).__name__,
            operation_label,
            delay,
            retry_state.attempt_number,
            max_attempts,
        )

    return Retrying(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(initial=initial_delay, max=max_delay),
        retry=retry_if_exception(_is_transient_error),
        before_sleep=_log_retry,
        sleep=time.sleep,
    )


_ReadT = TypeVar("_ReadT")


def _read_with_retry(
    fn: Callable[[], _ReadT],
    *,
    logger: PrefectLogAdapter,
    settings: GCPBatchWorkerSettings | None = None,
) -> _ReadT:
    """Run an idempotent GCP Batch read (job status polling) through the
    shared transient-retry policy. Reads are side-effect free, so retrying
    never risks starting a duplicate job."""
    if settings is None:
        settings = GCPBatchWorkerSettings()
    retrying = _build_transient_retrying(
        max_attempts=settings.api_read_retry_max_attempts,
        initial_delay=settings.api_read_retry_initial_delay_seconds,
        max_delay=settings.api_read_retry_max_delay_seconds,
        operation_label="polling job status",
        logger=logger,
    )
    return retrying(fn)


class GCPBatchWorker(
    BaseWorker[GCPBatchJobConfiguration, GCPBatchVariables, GCPBatchWorkerResult]
):
    """
    The GCP Batch worker. Runs each flow run as its own Google Cloud Batch job.
    """

    type = "gcp-batch"
    job_configuration = GCPBatchJobConfiguration
    job_configuration_variables = GCPBatchVariables
    _description = "Execute flow runs as jobs on Google Cloud Batch. Requires a Google Cloud Platform account."  # noqa
    _display_name = "Google Cloud Batch"
    _documentation_url = "https://docs.prefect.io/integrations/prefect-gcp"
    _logo_url = "https://images.ctfassets.net/gm98wzqotmnx/4SpnOBvMYkHp6z939MDKP6/549a91bc1ce9afd4fb12c68db7b68106/social-icon-google-cloud-1200-630.png?h=250"  # noqa

    async def run(
        self,
        flow_run: "FlowRun",
        configuration: GCPBatchJobConfiguration,
        task_status: Optional["TaskStatus"] = None,
    ) -> GCPBatchWorkerResult:
        """
        Runs the flow run on GCP Batch and waits for it to complete.
        """
        logger = self.get_flow_run_logger(flow_run)

        client = self._get_client(configuration=configuration)

        job = await run_sync_in_worker_thread(
            self._create_job,
            client=client,
            configuration=configuration,
            logger=logger,
        )

        if task_status:
            task_status.started(configuration.job_name)

        return await run_sync_in_worker_thread(
            self._watch_job_and_get_result,
            client=client,
            configuration=configuration,
            job=job,
            logger=logger,
        )

    @staticmethod
    def _get_client(
        configuration: GCPBatchJobConfiguration,
    ) -> "BatchServiceClient":
        """
        Get an authenticated client for interacting with the GCP Batch API.
        """
        return configuration.credentials.get_batch_client()

    def _create_job(
        self,
        client: "BatchServiceClient",
        configuration: GCPBatchJobConfiguration,
        logger: PrefectLogAdapter,
    ) -> BatchJob:
        """
        Creates the GCP Batch job.

        Because the job's ID is client-specified (`configuration.job_name`),
        job creation is safe to retry: if a transient error is returned after
        the job was actually created on the server, the retry receives
        `AlreadyExists` and the existing job is fetched instead of creating a
        duplicate.
        """
        settings = GCPBatchWorkerSettings()
        retrying = _build_transient_retrying(
            max_attempts=settings.create_job_max_attempts,
            initial_delay=settings.create_job_initial_delay_seconds,
            max_delay=settings.create_job_max_delay_seconds,
            operation_label="creating job",
            logger=logger,
        )

        logger.info(f"Creating GCP Batch job {configuration.job_name!r}")

        try:
            for attempt in retrying:
                with attempt:
                    try:
                        return BatchJob.create(
                            client=client,
                            project=configuration.project,
                            region=configuration.region,
                            job_id=configuration.job_name,
                            job=configuration.job_body,
                        )
                    except AlreadyExists:
                        logger.warning(
                            "GCP Batch job %r already exists; a previous "
                            "attempt likely succeeded despite a transient "
                            "error. Using the existing job instead of "
                            "creating a duplicate.",
                            configuration.job_name,
                        )
                        return BatchJob.get(
                            client=client,
                            job_name=(
                                f"projects/{configuration.project}/locations/"
                                f"{configuration.region}/jobs/{configuration.job_name}"
                            ),
                        )
        except NotFound as exc:
            raise RuntimeError(
                f"Failed to create GCP Batch job. Confirm that region "
                f"{configuration.region!r} is valid and that "
                f"{configuration.project!r} is the correct GCP project. If "
                "your project ID is not correct, you are using a Credentials "
                "block with permissions for the wrong project."
            ) from exc

    def _watch_job_and_get_result(
        self,
        client: "BatchServiceClient",
        configuration: GCPBatchJobConfiguration,
        job: BatchJob,
        logger: PrefectLogAdapter,
    ) -> GCPBatchWorkerResult:
        """
        Watch a GCP Batch job until it reaches a terminal state and return
        the result.
        """
        settings = GCPBatchWorkerSettings()

        try:
            job = self._watch_job(
                client=client,
                job=job,
                poll_interval=settings.job_watch_poll_interval_seconds,
                logger=logger,
                settings=settings,
            )
        except InfrastructureNotFound:
            logger.info(
                f"GCP Batch job {configuration.job_name!r} was deleted. "
                "The flow run will be marked based on its current state."
            )
            return GCPBatchWorkerResult(
                identifier=configuration.job_name, status_code=-1
            )

        if job.is_being_deleted():
            logger.info(
                f"GCP Batch job {configuration.job_name!r} is being deleted. "
                "The flow run will be marked based on its current state."
            )
            return GCPBatchWorkerResult(
                identifier=configuration.job_name, status_code=-1
            )

        if job.succeeded():
            status_code = 0
            logger.info(f"GCP Batch job {configuration.job_name!r} succeeded")
        else:
            status_code = 1
            if job.was_preempted():
                logger.error(
                    f"GCP Batch job {configuration.job_name!r} failed - "
                    "one or more tasks were preempted (Spot VM reclaimed by "
                    "GCP) and exhausted their retries. Consider increasing "
                    "'max_retry_count' if using spot provisioning."
                )
            else:
                logger.error(f"GCP Batch job {configuration.job_name!r} failed")

        self._log_job_logs(configuration=configuration, job=job, logger=logger)

        if not configuration.keep_job:
            logger.info(
                f"Deleting completed GCP Batch job {configuration.job_name!r} "
                "from Google Cloud Batch..."
            )
            try:
                BatchJob.delete(client=client, job_name=job.name)
            except Exception as exc:
                logger.critical(
                    "Received an exception while deleting the GCP Batch job "
                    f"{configuration.job_name!r} - {exc}"
                )

        return GCPBatchWorkerResult(
            identifier=configuration.job_name,
            status_code=status_code,
        )

    @staticmethod
    def _watch_job(
        client: "BatchServiceClient",
        job: BatchJob,
        poll_interval: float,
        logger: PrefectLogAdapter,
        settings: GCPBatchWorkerSettings,
    ) -> BatchJob:
        """
        Poll a GCP Batch job's status until it is no longer running.

        Raises:
            InfrastructureNotFound: If the job is deleted while being watched
                (e.g., by `kill_infrastructure`).
        """
        while job.is_running():
            time.sleep(poll_interval)
            try:
                job = _read_with_retry(
                    lambda: BatchJob.get(client=client, job_name=job.name),
                    logger=logger,
                    settings=settings,
                )
            except NotFound as exc:
                raise InfrastructureNotFound(
                    f"GCP Batch job {job.name!r} was deleted."
                ) from exc

        return job

    @staticmethod
    def _log_job_logs(
        configuration: GCPBatchJobConfiguration,
        job: BatchJob,
        logger: PrefectLogAdapter,
    ) -> None:
        """
        Best-effort fetch of the job's task logs from Cloud Logging. Failures
        fetching logs never fail the flow run.
        """
        try:
            from google.cloud.logging import ASCENDING
            from google.cloud.logging import Client as LoggingClient
        except ModuleNotFoundError:
            logger.debug(
                "google-cloud-logging is not installed; skipping log retrieval "
                "for GCP Batch job %r. Install prefect-gcp with the 'batch' "
                "extra to enable this.",
                configuration.job_name,
            )
            return

        try:
            logging_client = LoggingClient(
                project=configuration.project,
                credentials=configuration.credentials.get_credentials_from_service_account(),
            )
            filter_ = (
                f'logName="projects/{configuration.project}/logs/batch_task_logs" '
                f'AND labels.job_uid="{job.uid}"'
            )
            for entry in logging_client.list_entries(
                filter_=filter_, order_by=ASCENDING
            ):
                logger.info(f"[GCP Batch] {entry.payload}")
        except Exception as exc:
            logger.warning(
                f"Failed to fetch logs for GCP Batch job "
                f"{configuration.job_name!r} from Cloud Logging: {exc}"
            )

    async def kill_infrastructure(
        self,
        infrastructure_pid: str,
        configuration: GCPBatchJobConfiguration,
        grace_seconds: int = 30,
    ) -> None:
        """
        Kill a GCP Batch job by deleting it.

        Args:
            infrastructure_pid: The job name.
            configuration: The job configuration used to connect to GCP.
            grace_seconds: Not used for GCP Batch (GCP handles graceful shutdown).

        Raises:
            InfrastructureNotFound: If the job doesn't exist.
        """
        job_name = infrastructure_pid
        await run_sync_in_worker_thread(self._delete_job, job_name, configuration)

    def _delete_job(
        self, job_name: str, configuration: GCPBatchJobConfiguration
    ) -> None:
        """
        Delete a GCP Batch job.

        Raises:
            InfrastructureNotFound: If the job doesn't exist.
        """
        client = self._get_client(configuration)
        full_name = (
            f"projects/{configuration.project}/locations/"
            f"{configuration.region}/jobs/{job_name}"
        )
        try:
            BatchJob.delete(client=client, job_name=full_name)
            self._logger.info(
                f"Deleted GCP Batch job {job_name!r} in project "
                f"{configuration.project!r} region {configuration.region!r}"
            )
        except NotFound as exc:
            raise InfrastructureNotFound(
                f"GCP Batch job {job_name!r} not found in project "
                f"{configuration.project!r} region {configuration.region!r}"
            ) from exc
