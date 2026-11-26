"""Optional component profiling changes execution cost; reset preserves it."""
import numpy as np
from force_toolkit import StreamingMADForceEstimator

X = np.random.default_rng(7).normal(size=(400, 3))
fast = StreamingMADForceEstimator(3).fit(X)
profiled = StreamingMADForceEstimator(3, collect_component_timing=True).fit(X)
np.testing.assert_array_equal(fast.finalize(), profiled.finalize())
assert fast.get_diagnostics()['details']['component_timing_s'] is None
assert profiled.get_diagnostics()['details']['component_timing_enabled']
print(profiled.get_diagnostics()['details']['component_timing_s'])
profiled.reset()
assert profiled.collect_component_timing
