import pytest
from google.cloud.batch_v1.types import Job, JobStatus, StatusEvent
from prefect_gcp.models.batch import SPOT_PREEMPTION_EXIT_CODE, BatchJob


def _job(state: "JobStatus.State", status_events=None) -> BatchJob:
    return BatchJob(
        Job(
            name="projects/p/locations/us-central1/jobs/my-job",
            uid="job-uid-123",
            status=JobStatus(state=state, status_events=status_events or []),
        )
    )


class TestBatchJob:
    @pytest.mark.parametrize(
        "state,expected",
        [
            (JobStatus.State.STATE_UNSPECIFIED, True),
            (JobStatus.State.QUEUED, True),
            (JobStatus.State.SCHEDULED, True),
            (JobStatus.State.RUNNING, True),
            (JobStatus.State.SUCCEEDED, False),
            (JobStatus.State.FAILED, False),
            (JobStatus.State.DELETION_IN_PROGRESS, False),
        ],
    )
    def test_is_running(self, state, expected):
        assert _job(state).is_running() is expected

    def test_is_being_deleted(self):
        assert _job(JobStatus.State.DELETION_IN_PROGRESS).is_being_deleted() is True
        assert _job(JobStatus.State.RUNNING).is_being_deleted() is False

    def test_succeeded(self):
        assert _job(JobStatus.State.SUCCEEDED).succeeded() is True
        assert _job(JobStatus.State.FAILED).succeeded() is False

    def test_failed(self):
        assert _job(JobStatus.State.FAILED).failed() is True
        assert _job(JobStatus.State.SUCCEEDED).failed() is False

    def test_was_preempted_true_when_status_event_mentions_exit_code(self):
        job = _job(
            JobStatus.State.FAILED,
            status_events=[
                StatusEvent(
                    description=(
                        f"Task failed due to Spot VM preemption, exit code "
                        f"{SPOT_PREEMPTION_EXIT_CODE}"
                    )
                )
            ],
        )
        assert job.was_preempted() is True

    def test_was_preempted_false_for_unrelated_failure(self):
        job = _job(
            JobStatus.State.FAILED,
            status_events=[StatusEvent(description="Task failed: exit code 1")],
        )
        assert job.was_preempted() is False

    def test_was_preempted_false_with_no_status_events(self):
        assert _job(JobStatus.State.FAILED).was_preempted() is False

    def test_name_and_uid_properties(self):
        job = _job(JobStatus.State.RUNNING)
        assert job.name == "projects/p/locations/us-central1/jobs/my-job"
        assert job.uid == "job-uid-123"


class TestBatchJobApiCalls:
    def test_create_builds_request_and_wraps_response(self):
        from unittest import mock

        from google.cloud.batch_v1.types import CreateJobRequest

        client = mock.MagicMock()
        created = Job(name="projects/p/locations/us-central1/jobs/my-job")
        client.create_job.return_value = created

        job_body = {"task_groups": [{"task_spec": {"runnables": []}}]}
        result = BatchJob.create(
            client=client,
            project="p",
            region="us-central1",
            job_id="my-job",
            job=job_body,
        )

        client.create_job.assert_called_once()
        (_, kwargs) = client.create_job.call_args
        request = kwargs["request"]
        assert isinstance(request, CreateJobRequest)
        assert request.parent == "projects/p/locations/us-central1"
        assert request.job_id == "my-job"
        assert result.name == created.name

    def test_get_builds_request(self):
        from unittest import mock

        from google.cloud.batch_v1.types import GetJobRequest

        client = mock.MagicMock()
        client.get_job.return_value = Job(
            name="projects/p/locations/us-central1/jobs/my-job"
        )

        job = BatchJob.get(
            client=client, job_name="projects/p/locations/us-central1/jobs/my-job"
        )

        client.get_job.assert_called_once()
        (_, kwargs) = client.get_job.call_args
        request = kwargs["request"]
        assert isinstance(request, GetJobRequest)
        assert request.name == "projects/p/locations/us-central1/jobs/my-job"
        assert job.name == "projects/p/locations/us-central1/jobs/my-job"

    def test_delete_builds_request(self):
        from unittest import mock

        from google.cloud.batch_v1.types import DeleteJobRequest

        client = mock.MagicMock()

        BatchJob.delete(
            client=client, job_name="projects/p/locations/us-central1/jobs/my-job"
        )

        client.delete_job.assert_called_once()
        (_, kwargs) = client.delete_job.call_args
        request = kwargs["request"]
        assert isinstance(request, DeleteJobRequest)
        assert request.name == "projects/p/locations/us-central1/jobs/my-job"
