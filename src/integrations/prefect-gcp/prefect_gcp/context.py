"""
Context for per-task GCP Batch options, used by `GCPBatchTaskRunner`.
"""

from contextlib import contextmanager
from typing import Any, Dict, Generator

from pydantic import Field

from prefect.context import ContextModel, ContextVar


class BatchOptionsContext(ContextModel):
    """
    The context for GCP Batch task-level option management.

    Attributes:
        current_batch_options: The set of current batch options in the context.
    """

    current_batch_options: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def get(cls) -> "BatchOptionsContext":
        """
        Return an empty `BatchOptionsContext` instead of `None` if no context exists.
        """
        return cls.__var__.get(BatchOptionsContext())

    __var__: ContextVar = ContextVar("gcp_batch_options")


@contextmanager
def batch_options(**new_batch_options: Any) -> Generator[None, Dict[str, Any], None]:
    """
    Context manager to set per-task GCP Batch job options (machine type, CPU/memory,
    provisioning model, etc.) for tasks submitted to a `GCPBatchTaskRunner` within the
    block. If contexts are nested, new options are merged with options in the outer
    context. If a key is present in both, the new option will be used.

    Each option name matches a field on `GCPBatchTaskRunner` (e.g. `machine_type`,
    `cpu`, `memory`, `spot`, `service_account_email`), and overrides that task
    runner's default for tasks submitted within the block.

    Yields:
        The current set of batch options.

    Examples:
        Run one task on a large spot VM and another on a small standard VM:
        ```python
        from prefect import flow, task
        from prefect_gcp.task_runners import GCPBatchTaskRunner
        from prefect_gcp.context import batch_options

        @task
        def process(x):
            return x + 1

        @flow(task_runner=GCPBatchTaskRunner(region="us-central1"))
        def my_flow():
            with batch_options(machine_type="n2-standard-8", spot=True):
                process.submit(42)
            with batch_options(cpu="1000m", memory="512Mi"):
                process.submit(43)
        ```
    """
    current_batch_options = BatchOptionsContext.get().current_batch_options
    with BatchOptionsContext(
        current_batch_options={**current_batch_options, **new_batch_options}
    ):
        yield
