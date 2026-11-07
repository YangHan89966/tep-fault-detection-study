"""Unchanged pure horizon/annotation helpers from legacy generator v1.2; new entrypoint elsewhere."""
from typing import Any
import math
def current_shutdown_reasons(simulator: Any) -> tuple[list[str], dict[str, float]]:
    """Recompute the exact pinned Python-backend safety predicates."""
    measurements = simulator.process.get_xmeas()
    process_state = simulator.process._teproc
    observed = {
        "reactor_pressure_kpa": float(measurements[6]),
        "reactor_level_ft3": float(process_state.vlr / 35.3145),
        "reactor_temperature_c": float(measurements[8]),
        "separator_level_ft3": float(process_state.vls / 35.3145),
        "stripper_level_ft3": float(process_state.vlc / 35.3145),
    }
    reasons: list[str] = []
    if observed["reactor_pressure_kpa"] > 3000.0:
        reasons.append("REACTOR_PRESSURE_GT_3000_KPA")
    if observed["reactor_level_ft3"] > 24.0:
        reasons.append("REACTOR_LEVEL_GT_24_FT3")
    if observed["reactor_level_ft3"] < 2.0:
        reasons.append("REACTOR_LEVEL_LT_2_FT3")
    if observed["reactor_temperature_c"] > 175.0:
        reasons.append("REACTOR_TEMPERATURE_GT_175_C")
    if observed["separator_level_ft3"] > 12.0:
        reasons.append("SEPARATOR_LEVEL_GT_12_FT3")
    if observed["separator_level_ft3"] < 1.0:
        reasons.append("SEPARATOR_LEVEL_LT_1_FT3")
    if observed["stripper_level_ft3"] > 8.0:
        reasons.append("STRIPPER_LEVEL_GT_8_FT3")
    if observed["stripper_level_ft3"] < 1.0:
        reasons.append("STRIPPER_LEVEL_LT_1_FT3")
    if not reasons:
        reasons.append("PINNED_BACKEND_SHUTDOWN_FLAG_WITHOUT_RECOMPUTED_LIMIT_MATCH")
    return reasons, observed


def simulate_registered_horizon(
    simulator: Any,
    np: Any,
    duration_hours: float,
    fault_id: int,
    fault_onset_hours: float | None,
    record_interval: int,
) -> dict[str, Any]:
    """Run the pinned Python construction through the full registered horizon.

    The public TEPSimulator.simulate wrapper stops as soon as the process safety
    flag is raised.  The underlying pinned Python process, like the pinned
    Fortran TEFUNC, instead sets all state derivatives to zero.  The original
    Fortran main loop does not terminate on ISD; it continues controller calls,
    TEFUNC evaluation, and scheduled output until NPTS.  This routine preserves
    the Python trajectory exactly before shutdown and continues the same
    controller/process measurement path after shutdown without interpolation,
    seed replacement, or synthetic padding.
    """
    if not simulator.initialized:
        simulator.initialize()

    dt_hours = float(simulator.dt)
    n_steps = int(np.ceil(duration_hours / dt_hours))
    n_records = (n_steps // record_interval) + 1

    initial_measurements = np.asarray(simulator.process.get_xmeas(), dtype=np.float64)
    initial_mvs = np.asarray(simulator.process.get_xmv(), dtype=np.float64)
    if initial_measurements.shape != (41,) or initial_mvs.shape != (12,):
        raise RuntimeError("pinned backend initial interface shape")

    times_hours = np.zeros(n_records, dtype=np.float64)
    measurements = np.zeros((n_records, 41), dtype=np.float64)
    manipulated_vars = np.zeros((n_records, 12), dtype=np.float64)
    times_hours[0] = float(simulator.time)
    measurements[0] = initial_measurements
    manipulated_vars[0] = initial_mvs
    record_index = 1

    disturbance_activated = False
    shutdown_time_hours: float | None = None
    shutdown_step: int | None = None
    shutdown_reasons: list[str] = []
    shutdown_observed_values: dict[str, float] = {}

    for step in range(1, n_steps + 1):
        if (
            fault_id > 0
            and not disturbance_activated
            and fault_onset_hours is not None
            and simulator.time >= fault_onset_hours
        ):
            simulator.set_disturbance(fault_id, 1)
            simulator.set_ground_truth(fault_id)
            disturbance_activated = True

        if simulator.control_mode.value != "open_loop":
            xmeas = simulator.process.get_xmeas()
            xmv = simulator.process.get_xmv()
            new_xmv = simulator.controller.calculate(xmeas, xmv, simulator.step_count)
            for index in range(12):
                simulator.process.set_xmv(index + 1, new_xmv[index])

        derivatives = np.asarray(
            simulator.process.evaluate(simulator.time, simulator.process.state.yy),
            dtype=np.float64,
        )
        if derivatives.shape != (50,) or not np.isfinite(derivatives).all():
            raise RuntimeError("nonfinite or malformed state derivative")

        simulator.time += dt_hours
        new_state = simulator.process.state.yy + derivatives * dt_hours
        if new_state.shape != (50,) or not np.isfinite(new_state).all():
            raise RuntimeError("nonfinite or malformed integrated state")
        simulator.process.state.yy = new_state
        simulator.step_count += 1

        if shutdown_time_hours is None and simulator.process.is_shutdown():
            shutdown_time_hours = float(simulator.time)
            shutdown_step = int(step)
            shutdown_reasons, shutdown_observed_values = current_shutdown_reasons(simulator)

        if step % record_interval == 0:
            if record_index >= n_records:
                raise RuntimeError("record index overflow")
            times_hours[record_index] = float(simulator.time)
            measurements[record_index] = simulator.process.get_xmeas()
            manipulated_vars[record_index] = simulator.process.get_xmv()
            record_index += 1

    if record_index != n_records:
        raise RuntimeError(f"record count mismatch:{record_index}!={n_records}")

    shutdown_time_seconds = (
        None if shutdown_time_hours is None else shutdown_time_hours * 3600.0
    )
    first_released_sample = (
        None
        if shutdown_time_seconds is None
        else max(1, int(math.ceil((shutdown_time_seconds - 1e-9) / 180.0)))
    )
    return {
        "time_seconds": times_hours * 3600.0,
        "measurements": measurements,
        "manipulated_vars": manipulated_vars,
        "shutdown_occurred": shutdown_time_hours is not None,
        "shutdown_time_hours": shutdown_time_hours,
        "shutdown_time_seconds": shutdown_time_seconds,
        "shutdown_step": shutdown_step,
        "shutdown_reasons": shutdown_reasons,
        "shutdown_observed_values": shutdown_observed_values,
        "shutdown_first_released_sample": first_released_sample,
    }


