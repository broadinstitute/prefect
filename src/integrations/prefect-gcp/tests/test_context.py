from prefect_gcp.context import BatchOptionsContext, batch_options


def test_batch_options_context():
    input_batch_options = {"machine_type": "e2-standard-4"}
    current_batch_options = BatchOptionsContext(
        current_batch_options=input_batch_options
    ).current_batch_options
    assert current_batch_options == input_batch_options


def test_batch_options_context_get():
    current_batch_options = BatchOptionsContext.get().current_batch_options
    assert current_batch_options == {}


def test_batch_options():
    with batch_options(machine_type="e2-standard-4", spot=True):
        current_batch_options = BatchOptionsContext.get().current_batch_options
        assert current_batch_options == {"machine_type": "e2-standard-4", "spot": True}


def test_batch_options_empty():
    with batch_options():
        current_batch_options = BatchOptionsContext.get().current_batch_options
        assert current_batch_options == {}


def test_batch_options_override():
    with batch_options(cpu="1000m", spot=False):
        with batch_options(cpu="2000m", spot=True):
            current_batch_options = BatchOptionsContext.get().current_batch_options
            assert current_batch_options == {"cpu": "2000m", "spot": True}


def test_batch_options_nested_merges_outer_keys():
    with batch_options(cpu="1000m", memory="512Mi"):
        with batch_options(cpu="2000m"):
            current_batch_options = BatchOptionsContext.get().current_batch_options
            assert current_batch_options == {"cpu": "2000m", "memory": "512Mi"}
        # Outer context is restored after the inner block exits.
        current_batch_options = BatchOptionsContext.get().current_batch_options
        assert current_batch_options == {"cpu": "1000m", "memory": "512Mi"}
