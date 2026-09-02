import asyncio
import uuid

import pytest

from prefect import task
from prefect.task_engine import run_task_async
from prefect.task_runs import TaskRunWaiter


class TestTaskRunWaiter:
    @pytest.fixture(autouse=True)
    def teardown(self):
        yield

        TaskRunWaiter.instance().stop()

    def test_instance_returns_singleton(self):
        assert TaskRunWaiter.instance() is TaskRunWaiter.instance()

    def test_instance_returns_instance_after_stop(self):
        instance = TaskRunWaiter.instance()
        instance.stop()
        assert TaskRunWaiter.instance() is not instance

    @pytest.mark.timeout(20)
    @pytest.mark.usefixtures("use_hosted_api_server")
    async def test_wait_for_task_run(self, prefect_client, emitting_events_pipeline):
        """This test will fail with a timeout error if waiting is not working correctly."""

        @task
        async def test_task():
            await asyncio.sleep(1)

        task_run_id = uuid.uuid4()
        asyncio.create_task(run_task_async(task=test_task, task_run_id=task_run_id))

        await TaskRunWaiter.wait_for_task_run(task_run_id)

        await emitting_events_pipeline.process_events()

        task_run = await prefect_client.read_task_run(task_run_id)
        assert task_run.state.is_completed()

    @pytest.mark.usefixtures("use_hosted_api_server")
    async def test_wait_for_task_run_with_timeout(self):
        @task
        async def test_task():
            await asyncio.sleep(5)

        task_run_id = uuid.uuid4()
        run = asyncio.create_task(
            run_task_async(task=test_task, task_run_id=task_run_id)
        )

        await TaskRunWaiter.wait_for_task_run(task_run_id, timeout=1)

        # TaskRunWaiter stopped waiting before the task finished
        assert not run.done()
        await run

    @pytest.mark.timeout(20)
    @pytest.mark.usefixtures("use_hosted_api_server")
    async def test_non_singleton_mode(self, prefect_client, emitting_events_pipeline):
        waiter = TaskRunWaiter()
        assert waiter is not TaskRunWaiter.instance()

        @task
        async def test_task():
            await asyncio.sleep(1)

        task_run_id = uuid.uuid4()
        asyncio.create_task(run_task_async(task=test_task, task_run_id=task_run_id))

        await waiter.wait_for_task_run(task_run_id)

        await emitting_events_pipeline.process_events()

        task_run = await prefect_client.read_task_run(task_run_id)
        assert task_run.state.is_completed()

        waiter.stop()

    @pytest.mark.timeout(20)
    @pytest.mark.usefixtures("use_hosted_api_server")
    async def test_wait_for_task_run_handles_cache_hit(
        self, prefect_client, emitting_events_pipeline
    ):
        """A cache hit renames a `COMPLETED` task run's state to "Cached" rather
        than "Completed" - the event filter must still catch it, or callers waiting
        on the future (e.g. `PrefectDistributedFuture.wait_async`) hang forever."""

        @task(cache_key_fn=lambda *_, **__: "same-key", persist_result=True)
        async def cached_task():
            return 1

        first_run_id = uuid.uuid4()
        await run_task_async(task=cached_task, task_run_id=first_run_id)
        await emitting_events_pipeline.process_events()

        second_run_id = uuid.uuid4()
        asyncio.create_task(
            run_task_async(task=cached_task, task_run_id=second_run_id)
        )

        state = await TaskRunWaiter.wait_for_task_run(second_run_id)

        await emitting_events_pipeline.process_events()

        assert state is not None
        assert state.name == "Cached"
        assert state.is_completed()

        task_run = await prefect_client.read_task_run(second_run_id)
        assert task_run.state.is_completed()

    @pytest.mark.timeout(20)
    @pytest.mark.usefixtures("use_hosted_api_server")
    async def test_handles_concurrent_task_runs(
        self, prefect_client, emitting_events_pipeline
    ):
        @task
        async def fast_task():
            await asyncio.sleep(1)

        @task
        async def slow_task():
            await asyncio.sleep(5)

        task_run_id_1 = uuid.uuid4()
        task_run_id_2 = uuid.uuid4()

        asyncio.create_task(run_task_async(task=fast_task, task_run_id=task_run_id_1))
        asyncio.create_task(run_task_async(task=slow_task, task_run_id=task_run_id_2))

        await TaskRunWaiter.wait_for_task_run(task_run_id_1)

        await emitting_events_pipeline.process_events()

        task_run_1 = await prefect_client.read_task_run(task_run_id_1)
        task_run_2 = await prefect_client.read_task_run(task_run_id_2)

        assert task_run_1.state.is_completed()
        assert not task_run_2.state.is_completed()

        await TaskRunWaiter.wait_for_task_run(task_run_id_2)

        await emitting_events_pipeline.process_events()

        task_run_1 = await prefect_client.read_task_run(task_run_id_1)
        task_run_2 = await prefect_client.read_task_run(task_run_id_2)

        assert task_run_1.state.is_completed()
        assert task_run_2.state.is_completed()
