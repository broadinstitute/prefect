from unittest import mock
from uuid import uuid4

from prefect_gcp.batch_task_entrypoint import main

from prefect import task
from prefect.filesystems import LocalFileSystem


@task
def add_one(x: int) -> int:
    return x + 1


def test_main_persists_result_so_a_later_process_can_fetch_it():
    """
    `main()`'s task run happens in a container process distinct from the one
    that later calls `.result()` on the submitting process's future, so the
    result must be persisted - the same requirement `TaskWorker` applies to
    tasks it runs remotely.
    """
    task_run_id = uuid4()
    captured_task = {}
    storage_block = LocalFileSystem()
    # `with_options(result_storage=...)` requires a block that's already
    # persisted server-side (as `Block.load` would return), not a bare
    # in-memory instance.
    storage_block._block_document_id = uuid4()

    def fake_run_task_sync(task, **kwargs):
        captured_task["task"] = task
        state = mock.MagicMock()
        state.is_completed.return_value = True
        return state

    with (
        mock.patch(
            "prefect.utilities.importtools.import_object",
            return_value=add_one,
        ),
        mock.patch(
            "prefect.blocks.core.Block.load", return_value=storage_block
        ),
        mock.patch(
            "prefect.task_worker.read_parameters",
            new=mock.AsyncMock(return_value={"parameters": {"x": 1}}),
        ),
        mock.patch(
            "prefect.task_engine.run_task_sync", side_effect=fake_run_task_sync
        ),
    ):
        exit_code = main(
            [
                "--task-run-id",
                str(task_run_id),
                "--entrypoint",
                "tests.test_batch_task_entrypoint:add_one",
                "--result-storage-block",
                "local-file-system/test",
            ]
        )

    assert exit_code == 0
    assert captured_task["task"].persist_result is True
    assert captured_task["task"].result_storage is storage_block
