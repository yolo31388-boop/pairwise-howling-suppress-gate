import pytest
import media.howling_suppress_gate as mod

CASES = [
    ("adaptive_freq_track", (1,)),
    ("harmonic_structure_detect", (1,)),
    ("coeff_state_lock", (1,)),
    ("gate_threshold_track", (1,)),
    ("latency_by_block", (1,)),
    ("export_check", (1,)),
    ("process", (1,)),
    ("detect", (1,)),
    ("notch", (1,)),
    ("gate", (1,)),
    ("validate_audio", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
