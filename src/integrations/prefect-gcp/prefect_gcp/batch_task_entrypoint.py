"""
Container entrypoint for a single task run submitted by `GCPBatchTaskRunner`.

Invoked as `python -m prefect_gcp.batch_task_entrypoint` inside the GCP Batch job's
container. The image running this entrypoint must already contain the code defining
the task (the same requirement as any Prefect worker image running `prefect
flow-run execute`).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from uuid import UUID


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single Prefect task submitted by GCPBatchTaskRunner."
    )
    parser.add_argument(
        "--task-run-id",
        required=True,
        help="The ID of the task run to execute.",
    )
    parser.add_argument(
        "--entrypoint",
        required=True,
        help=(
            "The import path of the task function, in the form "
            "`module:function` or `./path/to/file.py:function`."
        ),
    )
    parser.add_argument(
        "--result-storage-block",
        required=True,
        help=(
            "The slug (`<block_type_slug>/<block_document_name>`) of the storage "
            "block used to exchange parameters between the submitting process and "
            "this container."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    from prefect.blocks.core import Block
    from prefect.results import ResultStore
    from prefect.task_engine import run_task_async, run_task_sync
    from prefect.task_worker import read_parameters
    from prefect.utilities.asyncutils import run_coro_as_sync
    from prefect.utilities.importtools import import_object

    task_run_id = UUID(args.task_run_id)
    task = import_object(args.entrypoint)

    block = Block.load(args.result_storage_block)
    result_store = ResultStore(result_storage=block)
    run_data = run_coro_as_sync(read_parameters(result_store, task_run_id))

    parameters = run_data.get("parameters", {})
    context = run_data.get("context")
    dependencies = run_data.get("dependencies")

    run_task_kwargs = dict(
        task=task,
        task_run_id=task_run_id,
        parameters=parameters,
        dependencies=dependencies,
        context=context,
        return_type="state",
    )

    if task.isasync:
        state = asyncio.run(run_task_async(**run_task_kwargs))
    else:
        state = run_task_sync(**run_task_kwargs)

    return 0 if state.is_completed() else 1


if __name__ == "__main__":
    sys.exit(main())
