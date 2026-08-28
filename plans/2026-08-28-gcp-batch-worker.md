# GCP Batch worker + per-task heterogeneous compute

## Goal

Add a `GCPBatchWorker` to `prefect-gcp` that runs each flow run as a Google Cloud Batch job
(instead of Cloud Run v2), giving access to Batch's full machine-type catalog, custom
CPU/memory shapes, and spot/preemptible VMs at lower cost than Cloud Run.

Additionally, let users declare per-task compute requirements (machine type or custom
CPU+memory, spot vs standard) directly on `@task`, so a single flow run can fan tasks out
across heterogeneous infrastructure without pre-provisioning one work pool per shape.

Reference material already gathered:
- `docs/contribute/develop-a-new-worker-type.mdx` — worker contract.
- `src/prefect/workers/base.py` — `BaseWorker`, `BaseJobConfiguration`, `BaseVariables`, `BaseWorkerResult`.
- `src/integrations/prefect-gcp/prefect_gcp/workers/cloud_run_v2.py` + `models/cloud_run_v2.py` — the pattern to mirror (typed model wrappers, `prepare_for_flow_run` mutators, tenacity retry settings, credential handling).
- `src/integrations/prefect-gcp/prefect_gcp/credentials.py` — `GcpCredentials.get_credentials_from_service_account()`.
- `src/integrations/prefect-ray/prefect_ray/{context.py,task_runners.py}` — the *only* existing precedent for per-submission infra options (`remote_options()` context manager consumed inside `TaskRunner.submit`). This is the model for task-level Batch options, since `BaseWorker.run()` only ever receives a `FlowRun`, never individual task runs — there is no way to plumb `@task`-level machine type through the worker/job-configuration layer.
- `google-cloud-batch` Python API (`google.cloud.batch_v1`): `BatchServiceClient`, `Job`, `TaskGroup`, `TaskSpec`, `Runnable.Container`, `ComputeResource`, `AllocationPolicy`/`InstancePolicy`/`InstancePolicyOrTemplate`, `LogsPolicy`, plus Cloud Logging (`batch_task_logs`) for log retrieval and `delete_job` for cancellation.
- `github.com/acgourley/prefect-gcp-batch` — useful for API call shape, but runs the whole flow as a single Batch task; not a template for job-configuration structure or task-level parallelism.

## Decisions

- **Where this code lives**: added to the existing `prefect-gcp` package
  (`prefect_gcp/workers/batch.py`, `prefect_gcp/models/batch.py`) rather than a new
  `prefect-gcp-batch` package, reusing `GcpCredentials` and the existing
  `prefect.collections` entry point — mirrors how `cloud_run.py`/`cloud_run_v2.py`/`vertex.py`
  coexist today.
- **Scope of "per-task" support**: the task-runner piece (Phase 5) ships in v1, in the
  same PR/release as the worker (Phases 1–4), not as a fast-follow.
- **GPU/accelerator support**: out of scope for v1. No `Accelerator`/`install_gpu_drivers`
  fields on `GCPBatchJobConfiguration` or `batch_options()` yet — leave them out entirely
  rather than stubbing, and revisit as a later addition if there's demand.

## Phase 1 — Package scaffolding & credentials

- Add `prefect_gcp/models/batch.py`: thin typed wrappers around the raw
  `google.cloud.batch_v1` request/response types, following the `JobV2`/`ExecutionV2`
  pattern in `models/cloud_run_v2.py` (classmethod `create`/`get`/`delete`, `is_running()`,
  `succeeded()`, helpers to interpret `status.status_events` — notably distinguishing SPOT
  preemption, exit code `50001`, from a genuine task failure).
- Reuse `GcpCredentials` unchanged; add a `get_batch_client()`/`get_batch_async_client()`
  convenience on `GcpCredentials` (`credentials.py`) analogous to `get_job_service_client`,
  returning a `BatchServiceClient`/`BatchServiceAsyncClient` authorized via
  `get_credentials_from_service_account()`.
- Add `google-cloud-batch` and `google-cloud-logging` to `prefect-gcp`'s dependencies in
  `pyproject.toml`.

## Phase 2 — Job configuration model

`prefect_gcp/workers/batch.py`:

- `GCPBatchJobConfiguration(BaseJobConfiguration)`:
  - `credentials: GcpCredentials`
  - `region: str`, `project` (derived from credentials, like Cloud Run v2)
  - `image: str` (default `prefecthq/prefect:3-latest`), `command`/`args` (inherited/derived)
  - `env` (inherited) → mapped into `TaskSpec.environment.variables`
  - `machine_type: Optional[str]` (e.g. `"e2-standard-4"`) **or** `cpu`/`memory` custom shape
    (`ComputeResource.cpu_milli`, `memory_mib`) — validate that a custom shape isn't combined
    with a `machine_type` that already implies a fixed shape where that's invalid.
  - `provisioning_model: Literal["standard", "spot"]` → `InstancePolicy.provisioning_model`
  - `boot_disk_size_gb`, `service_account_email`, `network`, `subnetwork`,
    `allowed_locations: Optional[list[str]]` (zone restriction)
  - `max_run_duration` (seconds, → `TaskSpec.max_run_duration`), `max_retry_count`
  - `job_body: dict` — the actual `CreateJobRequest`-shaped payload, built the same way
    Cloud Run v2 builds `job_body`: a Jinja-templated default (`_get_default_job_body_template`)
    exposed via `json_schema_extra`, mutated in `prepare_for_flow_run` by small helper
    methods (`_populate_image_if_not_present`, `_populate_env`, `_populate_compute_resource`,
    `_populate_allocation_policy`, `_populate_labels`, `_populate_command`).
  - `keep_job: bool` (skip delete-on-completion, for debugging)
  - `job_name` property: Batch job IDs have their own charset/length rules — slugify +
    uuid suffix like Cloud Run v2's `job_name`.
- `GCPBatchVariables(BaseVariables)`: UI-facing schema mirroring the fields above
  (`image`, `region`, `machine_type`, `cpu`, `memory`, `provisioning_model`/`spot: bool`,
  `boot_disk_size_gb`, `service_account_email`, `network`, `subnetwork`, `max_run_duration`,
  `keep_job`).
- `GCPBatchWorkerResult(BaseWorkerResult)`.

## Phase 3 — Worker implementation

`GCPBatchWorker(BaseWorker)`:

- `type = "gcp-batch"`, `job_configuration = GCPBatchJobConfiguration`,
  `job_configuration_variables = GCPBatchVariables`, `_documentation_url`, `_logo_url`,
  `_description`.
- `run(flow_run, configuration, task_status=None)`:
  1. Build `CreateJobRequest` from `configuration.job_body`, submit via
     `BatchServiceClient.create_job` (wrapped in `run_sync_in_worker_thread`, retried with
     tenacity for transient 429/500/503 — reuse/extend the `CloudRunV2WorkerSettings`
     pattern as `GCPBatchWorkerSettings`, namespaced
     `PREFECT_INTEGRATIONS_GCP_BATCH_WORKER_*`).
  2. `task_status.started(job.name)` once creation succeeds.
  3. Poll `get_job(name=...)` on `poll_interval` until `status.state` is terminal
     (`SUCCEEDED`/`FAILED`/`DELETION_IN_PROGRESS` after a cancel). While polling, inspect
     `status.status_events` to log human-readable progress and to special-case SPOT
     preemption vs hard failure in the returned status code / log message.
  4. Retrieve logs from Cloud Logging (`logger("batch_task_logs")`, filtered by
     `labels.job_uid == job.uid`) and stream them through Prefect's logger, mirroring how
     Cloud Run v2 surfaces `execution.logUri`.
  5. Delete the job unless `keep_job`, return `GCPBatchWorkerResult(status_code=0|1, identifier=job.name)`.
- `kill_infrastructure(infrastructure_pid, configuration, grace_seconds=...)`: call
  `delete_job` on the stored job name; translate 404 into `InfrastructureNotFound`
  (same convention as Cloud Run v2's `_delete_job`).
- Error-formatting helpers for common misconfigurations (bad region/project, quota
  exceeded, invalid machine type) analogous to `_create_job_error`.

## Phase 4 — Base job template, docs, examples

- No separate template file needed (same as Cloud Run v2) — the default lives inline via
  `json_schema_extra`; verify `get_default_base_job_template()` renders a sane
  `variables`/`job_configuration` split.
- Register in `prefect_gcp/__init__.py` / `workers/__init__.py` so it self-registers via
  `@register_base_type` on import (same line pattern as the Cloud Run v2 import).
- Add a docs page under the GCP integration docs describing required IAM roles
  (`roles/batch.jobsEditor` + `iam.serviceAccounts.actAs` for the worker identity;
  `roles/batch.agentReporter` + workload permissions for the job's runtime service
  account) and APIs to enable (`batch.googleapis.com`, `logging.googleapis.com`).

## Phase 5 — Per-task heterogeneous compute (`@task`-level machine type / spot) — DONE

Workers only ever receive a `FlowRun` in `run()` — there is no path for per-`@task`
metadata to reach `BaseJobConfiguration`. Implemented as a dedicated `TaskRunner`,
following `prefect_ray`'s `RayTaskRunner` + `remote_options()` for the context-manager
ergonomics — but with one key difference discovered during implementation: **Ray ships
the live `Task` object to already-running worker processes in a cluster; GCP Batch has no
such cluster.** Each Batch job is a fresh container, so there is no way to "just run a
task" there — this required a small new "run one task standalone" mechanism (recombining
existing pieces, not a new bundling system):

- `prefect_gcp/context.py`: `BatchOptionsContext(ContextModel)` + `batch_options(**kwargs)`,
  a direct structural port of `prefect_ray/context.py`'s `RemoteOptionsContext`/
  `remote_options()`. Options are plain kwargs matching `GCPBatchTaskRunner` fields
  (`machine_type`, `cpu`, `memory`, `spot`, `service_account_email`, ...).
- `prefect_gcp/task_runners.py`: `GCPBatchTaskRunner(TaskRunner[PrefectDistributedFuture[R]])`.
  - Reuses core's `PrefectDistributedFuture` as-is (no custom future subclass needed) —
    it already implements exactly "poll/subscribe to a task run's state anywhere," via
    `TaskRunWaiter`'s event-based waiting, which is exactly what's needed once a Batch
    container reports task-run state back to the API on its own.
  - `submit()`: generates `task_run_id`, eagerly resolves any upstream `PrefectFuture`s
    embedded in `parameters`/`wait_for` via `resolve_inputs_sync` (the Batch container
    starts fresh with no access to this process's future objects — unlike Ray, which
    ships live futures into its object store), computes the task's `module:function`
    entrypoint string from `task.fn.__module__`/`__qualname__` (rejecting closures/locally
    -defined tasks with a clear error), captures `context = serialize_context(...)` for
    flow-run/tag linkage (same call Ray makes), persists
    `{parameters, context, dependencies}` to a user-provided storage block via
    `prefect.task_worker.store_parameters` (reusing that helper's existing
    `parameters/{id}` convention rather than inventing a new one), and creates the Batch
    job (a single-task job whose container command is the new entrypoint below). Returns
    `PrefectDistributedFuture(task_run_id=task_run_id)` immediately — no polling of the
    Batch API at all; `.result()`/`.wait()` come for free from core.
  - Job body built directly as a `google.cloud.batch_v1` proto-compatible dict (bypassing
    the work-pool `GCPBatchJobConfiguration`/Jinja-template machinery, which is designed
    for the work-pool/UI editing flow, not direct instantiation), reusing
    `_cpu_to_milli`/`_memory_to_mib`/`_build_transient_retrying` from `workers/batch.py`.
- `prefect_gcp/batch_task_entrypoint.py`: `python -m prefect_gcp.batch_task_entrypoint
  --task-run-id <id> --entrypoint "<module>:<func>" --result-storage-block "<slug>"` —
  the Batch container's command. Loads the storage block via `Block.load(slug)`, reads
  back `{parameters, context, dependencies}` via `prefect.task_worker.read_parameters`,
  reconstructs the task via `prefect.utilities.importtools.import_object(entrypoint)`,
  and calls `run_task_sync`/`run_task_async(task=..., task_run_id=..., parameters=...,
  context=..., return_type="state")` — the same engine entrypoints `TaskWorker` uses,
  which create the actual `TaskRun` row keyed by the given ID and report state to the API
  as they run. Exits `0`/`1` based on `state.is_completed()`.
- **Documented constraint**: the Batch image must already contain the code defining any
  submitted tasks (same requirement as any worker image running `prefect flow-run
  execute`), and tasks must be defined at module level (not closures) so
  `module:function` reconstruction works. This was the scope explicitly chosen over a
  cloudpickle-based task-bundling alternative.
- Cost/latency tradeoff documented: each `.submit()` provisions its own VM (tens of
  seconds to minutes of cold-start) — fits coarse-grained, long-running, heterogeneous
  tasks, not fine-grained fan-out (use the `GCPBatchWorker`'s native `task_count`/
  `parallelism`, or another task runner, for that).
- `docs/v3/advanced/submit-flows-directly-to-dynamic-infrastructure.mdx`'s
  decorator-per-subflow pattern is documented alongside as the zero-new-code alternative
  for teams fine with subflow-level (rather than task-level) granularity.
- Verified with a throwaway pytest suite: `batch_options()` overrides flow through to the
  constructed `Job` proto correctly; a full round trip (`submit()` → persisted blob →
  `batch_task_entrypoint.main()` standing in for the container → real executed task run)
  proves the mechanism end-to-end against the real `run_task_sync` engine.

## Phase 6 — Testing — DONE

Test files follow the repo's actual flat `tests/` convention (matching
`test_cloud_run_v2.py`/`test_cloud_run_worker_v2.py`), not the `tests/models/`,
`tests/workers/`, `tests/task_runners/` subdirectories originally sketched above:

- `tests/test_batch.py` (17 tests): pure `BatchJob` model logic (`is_running`,
  `is_being_deleted`, `succeeded`, `failed`, `was_preempted` against real
  `google.cloud.batch_v1.types.Job`/`JobStatus`/`StatusEvent` objects, no mocking) plus
  `create`/`get`/`delete` request-building tests against a mocked `BatchServiceClient`.
- `tests/test_batch_worker.py` (31 tests): `GCPBatchJobConfiguration` (`project`/`job_name`
  properties, every `prepare_for_flow_run` mutator, job-body validators), `GCPBatchWorker.run()`
  end-to-end (success/failure/preemption-logging/`keep_job`), `_watch_job` (terminal-state
  return, `NotFound` → `InfrastructureNotFound`, transient-error retry),
  `_watch_job_and_get_result` (deleted-job and `DELETION_IN_PROGRESS` handling, delete-failure
  logged but non-raising), `_create_job` (transient retry, non-transient passthrough,
  `AlreadyExists` recovery, `NotFound` wrapped with a helpful message), `kill_infrastructure`,
  and `_log_job_logs` (streams entries, swallows fetch errors).
- `tests/test_context.py` (6 tests): `BatchOptionsContext`/`batch_options()` scoping
  (nested/sequential `with` blocks), directly mirroring `prefect-ray`'s `test_context.py`.
- `tests/test_task_runners.py` (10 tests): `duplicate()`/`__eq__`, `_entrypoint_for_task`
  (module-level vs. closure rejection), `submit()` job-body construction with `batch_options()`
  overrides applied, `submit()` falling back to the task runner's own defaults (`cpu`/`memory`)
  when no override is given, `_submit_batch_job` transient-error retry, and the full
  submit → persisted blob → `batch_task_entrypoint.main()` round trip proving real task
  execution.

**Two real bugs caught by writing these tests** (both fixed in `task_runners.py`):
1. `submit()` merged `self._init_kwargs` (a frozen snapshot from `__init__`) into per-call
   options, so any attribute mutated on the runner instance after construction was silently
   ignored in favor of the stale init-time value.
2. `_build_job_body` never fell back to `self.machine_type`/`self.service_account_email`/
   `self.network`/`self.subnetwork` at all — those constructor defaults were dead code unless
   re-specified via `batch_options()` on every single call.

Coverage on the five new/changed modules: 96% overall (`context.py` 100%, `models/batch.py`
100%, `task_runners.py` 96%, `workers/batch.py` 96%, `batch_task_entrypoint.py` 94%) — well
above the package's configured `fail_under = 80`. Remaining gaps are dead-in-this-environment
branches (the `ModuleNotFoundError` guard for an extra that's always installed via the `dev`
dependency group; defensive `ValueError`s in the cpu/memory parsers that pydantic field
patterns already prevent from being reached) or trivial one-line delegations, not real
behavior. Full existing `prefect-gcp` suite (588 pre-existing + 63 new = 651 tests) passes,
both with and without `--extra batch` (the `dev` dependency group already installs
`google-cloud-batch`/`google-cloud-logging` unconditionally, matching how `aiplatform`/
`bigquery` extras are already handled in this package).

## Phase 7 — Release

- Update `prefect-gcp` changelog/README, bump version per integration release process.
- Confirm `prefect worker start --type gcp-batch` resolves via the existing
  `prefect_gcp` → `prefect.collections` entry point (no new entry point needed).
