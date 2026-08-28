"""Thin wrappers around the GCP Batch API for use by the GCP Batch worker."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google.cloud.batch_v1 import BatchServiceClient
    from google.cloud.batch_v1.types import Job

# Exit code Batch reports on a task when its Spot VM is preempted. Used to
# distinguish preemption from a "real" task failure, since Batch surfaces
# preemption as an ordinary task failure rather than a distinct job state.
SPOT_PREEMPTION_EXIT_CODE = 50001


class BatchJob:
    """
    Wraps a `google.cloud.batch_v1.types.Job` resource with helpers for
    interpreting job state, used by the GCP Batch worker.
    """

    def __init__(self, job: "Job"):
        self.job = job

    @property
    def name(self) -> str:
        """The fully-qualified resource name of the job."""
        return self.job.name

    @property
    def uid(self) -> str:
        """The GCP-assigned unique ID of the job."""
        return self.job.uid

    def is_running(self) -> bool:
        """
        Whether the job is still in a non-terminal state.

        `STATE_UNSPECIFIED` is treated as non-terminal (still running): a
        just-created job may not have its status populated yet, and the
        worker should poll again rather than treat that as a terminal state.
        """
        from google.cloud.batch_v1.types import JobStatus

        return self.job.status.state in (
            JobStatus.State.STATE_UNSPECIFIED,
            JobStatus.State.QUEUED,
            JobStatus.State.SCHEDULED,
            JobStatus.State.RUNNING,
        )

    def is_being_deleted(self) -> bool:
        """Whether the job is in the process of being deleted."""
        from google.cloud.batch_v1.types import JobStatus

        return self.job.status.state == JobStatus.State.DELETION_IN_PROGRESS

    def succeeded(self) -> bool:
        """Whether the job reached a successful terminal state."""
        from google.cloud.batch_v1.types import JobStatus

        return self.job.status.state == JobStatus.State.SUCCEEDED

    def failed(self) -> bool:
        """Whether the job reached a failed terminal state."""
        from google.cloud.batch_v1.types import JobStatus

        return self.job.status.state == JobStatus.State.FAILED

    def was_preempted(self) -> bool:
        """
        Whether any status event on the job indicates a Spot VM preemption,
        as opposed to a genuine task failure.
        """
        for event in self.job.status.status_events:
            if str(SPOT_PREEMPTION_EXIT_CODE) in (event.description or ""):
                return True
        return False

    @classmethod
    def get(cls, client: "BatchServiceClient", job_name: str) -> "BatchJob":
        """
        Get a job from the GCP Batch API.

        Args:
            client: An authenticated Batch Service client.
            job_name: The fully-qualified resource name of the job, in the
                form `projects/{project}/locations/{region}/jobs/{job_id}`.
        """
        from google.cloud.batch_v1.types import GetJobRequest

        job = client.get_job(request=GetJobRequest(name=job_name))
        return cls(job)

    @staticmethod
    def create(
        client: "BatchServiceClient",
        project: str,
        region: str,
        job_id: str,
        job: "Job",
    ) -> "BatchJob":
        """
        Create a job with the GCP Batch API.

        Args:
            client: An authenticated Batch Service client.
            project: The GCP project ID.
            region: The GCP region to create the job in.
            job_id: The caller-assigned ID of the job to create.
            job: The job spec to create.
        """
        from google.cloud.batch_v1.types import CreateJobRequest

        request = CreateJobRequest(
            parent=f"projects/{project}/locations/{region}",
            job_id=job_id,
            job=job,
        )
        return BatchJob(client.create_job(request=request))

    @staticmethod
    def delete(client: "BatchServiceClient", job_name: str) -> None:
        """
        Delete a job with the GCP Batch API. Used both for cleanup after a
        completed job and to implement flow-run cancellation.

        Args:
            client: An authenticated Batch Service client.
            job_name: The fully-qualified resource name of the job to delete.
        """
        from google.cloud.batch_v1.types import DeleteJobRequest

        client.delete_job(request=DeleteJobRequest(name=job_name))
