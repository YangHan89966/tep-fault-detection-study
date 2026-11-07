/**
 * One of the twenty `IDV` disturbances, as a value object.
 *
 * Reading it does not touch a simulation. Everything on it comes from
 * `tepsim_core::FAULTS`, which `tepsim-oracle/tests/fault_table.rs` compares
 * against the original's header table and against the code that implements
 * each fault.
 */
export class Fault {
    static __wrap(ptr) {
        const obj = Object.create(Fault.prototype);
        obj.__wbg_ptr = ptr;
        FaultFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        FaultFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_fault_free(ptr, 0);
    }
    /**
     * Whether the fault reaches the plant model at all.
     *
     * False for the three sticking faults, which widen a valve dead band and
     * touch no equation. In an open-loop run they do nothing, and that is not
     * a bug in the scenario.
     * @returns {boolean}
     */
    get affectsThePlant() {
        const ret = wasm.fault_affectsThePlant(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * The one-based walk channels it drives, for a random or spike fault.
     * @returns {Uint32Array}
     */
    get channels() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.fault_channels(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v1 = getArrayU32FromWasm0(r0, r1).slice();
            wasm.__wbindgen_export(r0, r1 * 4, 4);
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * What the fault does, where the header is silent or vaguer than the code.
     * @returns {string}
     */
    get effect() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.fault_effect(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * The one-based `IDV` index, `1..=20`.
     * @returns {number}
     */
    get id() {
        const ret = wasm.fault_id(this.__wbg_ptr);
        return ret;
    }
    /**
     * How it enters: `step`, `random`, `spike` or `sticking`.
     *
     * `spike` is a random-variation fault whose channels are spike trains
     * rather than walks. The header table calls both "Unknown" or "Random
     * Variation" and the distinction is why three of them are reported in the
     * literature as the hardest to detect: they are intermittent.
     * @returns {string}
     */
    get kind() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.fault_kind(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * A non-empty string for a fault panel: the published description, or the
     * effect where the original says only "Unknown".
     * @returns {string}
     */
    get label() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.fault_label(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * The `teprob.f` line or lines it acts on.
     * @returns {string}
     */
    get line() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.fault_line(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * The description from `teprob.f:172-191`, verbatim.
     *
     * Five of them say only "Unknown"; [`Fault::effect`] is what the source
     * actually does in those cases.
     * @returns {string}
     */
    get published() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.fault_published(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Whether its channels are spike trains rather than walks.
     * @returns {boolean}
     */
    get spiking() {
        const ret = wasm.fault_spiking(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * The one-based valves it sticks, for a sticking fault.
     * @returns {Uint32Array}
     */
    get valves() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.fault_valves(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v1 = getArrayU32FromWasm0(r0, r1).slice();
            wasm.__wbindgen_export(r0, r1 * 4, 4);
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
}
if (Symbol.dispose) Fault.prototype[Symbol.dispose] = Fault.prototype.free;

/**
 * A complete, reproducible description of a run.
 *
 * Mutable, because a settings panel edits it in place. [`Sim`] takes a copy at
 * construction, so editing a scenario never disturbs a simulation already
 * running.
 *
 * Every field a run's output depends on is here, which is what makes a
 * recorded dataset reproducible from its scenario alone. [`Scenario::digest`]
 * is a 16-character summary of all of it.
 */
export class Scenario {
    static __wrap(ptr) {
        const obj = Object.create(Scenario.prototype);
        obj.__wbg_ptr = ptr;
        ScenarioFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ScenarioFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_scenario_free(ptr, 0);
    }
    /**
     * The one-based indices of every requested disturbance.
     *
     * A getter, to match [`Sim::active_faults`]. That one is ground truth and
     * this one is configuration, and they differ: the driver switches
     * `IDV(12)` on at hour eight whatever was asked for.
     * @returns {Uint8Array}
     */
    get activeFaults() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.scenario_activeFaults(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v1 = getArrayU8FromWasm0(r0, r1).slice();
            wasm.__wbindgen_export(r0, r1 * 1, 1);
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Clear every disturbance flag.
     */
    clearFaults() {
        wasm.scenario_clearFaults(this.__wbg_ptr);
    }
    /**
     * Whether the plant runs closed loop under the published control scheme.
     *
     * Open loop is a diagnostic mode, not a useful operating one: the plant
     * trips on reactor pressure after about three hours. It is here because
     * the difference between the two is the clearest single statement of what
     * the control layer does.
     * @returns {boolean}
     */
    get controlled() {
        const ret = wasm.scenario_controlled(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * A 16-character hex digest of the scenario.
     *
     * Identifies the run's description, not its output. [`Sim::checksum`] is
     * the digest of the samples. Two scenarios with the same digest produce
     * the same numbers; that is the whole claim, and it is what makes a run
     * shareable as a link rather than as a file.
     * @returns {string}
     */
    digest() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.scenario_digest(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Whether the driver forces `IDV(12)` on at eight hours regardless of what
     * was asked for, as `temain_mod.f:366-368` does.
     *
     * `false` by default: Tier 7 established that the published `dNN_te`
     * files were generated with `temain_mod.f:367` replaced rather than kept,
     * so forcing it is a departure from them rather than a match. Delta D-011,
     * signed off 2026-08-28.
     * @returns {boolean}
     */
    get driverForcesIdv12() {
        const ret = wasm.scenario_driverForcesIdv12(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Whether `IDV(id)` is on. `false` for an out-of-range index.
     * @param {number} id
     * @returns {boolean}
     */
    fault(id) {
        const ret = wasm.scenario_fault(this.__wbg_ptr, id);
        return ret !== 0;
    }
    /**
     * Parse a scenario back from [`Scenario::text`].
     *
     * # Errors
     *
     * Throws with a message naming what was wrong: a missing or unknown
     * field, a malformed number, a value out of range, or a format version
     * this build does not read. Nothing is defaulted quietly, so a link that
     * cannot be honoured says so instead of opening a different run.
     * @param {string} text
     * @returns {Scenario}
     */
    static fromText(text) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passStringToWasm0(text, wasm.__wbindgen_export2, wasm.__wbindgen_export3);
            const len0 = WASM_VECTOR_LEN;
            wasm.scenario_fromText(retptr, ptr0, len0);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return Scenario.__wrap(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Run length in simulated hours.
     * @returns {number}
     */
    get hours() {
        const ret = wasm.scenario_hours(this.__wbg_ptr);
        return ret;
    }
    /**
     * The integrator: `euler`, `rk4` or `dopri5`.
     * @returns {string}
     */
    get integrator() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.scenario_integrator(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Whether this scenario's integrator reproduces the original.
     *
     * Only Euler does. Everything the validation ladder claims is a claim
     * about Euler, so a run using anything else is a *better* integration of
     * the same equations and not a reproduction of the same numbers. A user
     * interface that offers the choice should say so.
     * @returns {boolean}
     */
    get isFaithful() {
        const ret = wasm.scenario_isFaithful(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * The fault-free plant for 48 hours, closed loop, from the seed
     * `teprob.f` ships with, integrated by the faithful Euler method.
     *
     * 48 hours is `NPTS = 172800` at a one-second step, the run
     * `temain_mod.f` was written to do.
     */
    constructor() {
        const ret = wasm.scenario_new();
        this.__wbg_ptr = ret;
        ScenarioFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * How many samples it will record, or `0` if it is invalid.
     *
     * Cheap enough to call on every keystroke in a settings panel, which is
     * what it is for.
     * @returns {number}
     */
    get sampleCount() {
        const ret = wasm.scenario_sampleCount(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Record one sample every this many integrator steps.
     *
     * 180 by default, which is the 180-second spacing `temain_mod.f:401`
     * writes at and the spacing of the published `d00` to `d21` files.
     * @returns {number}
     */
    get sampleEvery() {
        const ret = wasm.scenario_sampleEvery(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * The generator word. `teprob.f:1187` compiles in 4651207995.
     * @returns {number}
     */
    get seed() {
        const ret = wasm.scenario_seed(this.__wbg_ptr);
        return ret;
    }
    /**
     * Set or clear `IDV(id)`.
     *
     * # Errors
     *
     * Throws if `id` is outside `1..=20`. `Scenario::with_fault` in the
     * facade panics instead, which is right for a Rust caller and fatal in
     * wasm, where a panic aborts the module.
     * @param {number} id
     * @param {boolean} active
     */
    setFault(id, active) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.scenario_setFault(retptr, this.__wbg_ptr, id, active);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            if (r1) {
                throw takeObject(r0);
            }
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Choose the integrator by name.
     *
     * # Errors
     *
     * Throws if the name is not one of `euler`, `rk4`, `dopri5` or
     * `dormand-prince`.
     * @param {string} name
     */
    setIntegrator(name) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passStringToWasm0(name, wasm.__wbindgen_export2, wasm.__wbindgen_export3);
            const len0 = WASM_VECTOR_LEN;
            wasm.scenario_setIntegrator(retptr, this.__wbg_ptr, ptr0, len0);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            if (r1) {
                throw takeObject(r0);
            }
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Run closed loop, or open loop with the valves held.
     * @param {boolean} controlled
     */
    set controlled(controlled) {
        wasm.scenario_set_controlled(this.__wbg_ptr, controlled);
    }
    /**
     * Set whether the driver forces `IDV(12)` on at eight hours.
     * @param {boolean} forced
     */
    set driverForcesIdv12(forced) {
        wasm.scenario_set_driverForcesIdv12(this.__wbg_ptr, forced);
    }
    /**
     * Set the run length in simulated hours.
     * @param {number} hours
     */
    set hours(hours) {
        wasm.scenario_set_hours(this.__wbg_ptr, hours);
    }
    /**
     * Set the sampling cadence in integrator steps. Must be at least one.
     * @param {number} steps
     */
    set sampleEvery(steps) {
        wasm.scenario_set_sampleEvery(this.__wbg_ptr, steps);
    }
    /**
     * Set the generator word. Must be finite and positive.
     * @param {number} seed
     */
    set seed(seed) {
        wasm.scenario_set_seed(this.__wbg_ptr, seed);
    }
    /**
     * Set the integrator step in hours.
     *
     * Changing it changes every number in the run: the original's one-second
     * fixed Euler step is part of what is being reproduced, not an
     * implementation detail. Delta D-005.
     * @param {number} hours
     */
    set stepHours(hours) {
        wasm.scenario_set_stepHours(this.__wbg_ptr, hours);
    }
    /**
     * Set whether a trip ends the run.
     * @param {boolean} ends
     */
    set tripEndsTheRun(ends) {
        wasm.scenario_set_tripEndsTheRun(this.__wbg_ptr, ends);
    }
    /**
     * Integrator step in hours. One second by default, as `INTGTR` uses.
     * @returns {number}
     */
    get stepHours() {
        const ret = wasm.scenario_stepHours(this.__wbg_ptr);
        return ret;
    }
    /**
     * How many integrator steps this scenario is, or `0` if it is invalid.
     * @returns {number}
     */
    get steps() {
        const ret = wasm.scenario_steps(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * This scenario as one line of canonical text.
     *
     * Every field, written out, tagged with the format version, in characters
     * a URL fragment carries without percent-encoding. This is the *only*
     * thing the page and the worker exchange about a scenario: before it
     * existed, `app.js`, `worker.js` and `share.js` each enumerated the fields
     * by hand, and a field added here reached none of them.
     *
     * Round-trips through [`Scenario::from_text`] bit for bit, so
     * [`Scenario::digest`] survives it. See `tepsim::text`.
     * @returns {string}
     */
    get text() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.scenario_text(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Whether a trip ends the run rather than freezing the plant.
     *
     * `true` by default. Setting it `false` gives what `teprob.f:807-811`
     * does: the plant freezes and keeps reporting, which is three quarters of
     * `d06.dat`. Delta D-007, signed off 2026-08-28.
     * @returns {boolean}
     */
    get tripEndsTheRun() {
        const ret = wasm.scenario_tripEndsTheRun(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Why this scenario cannot be run, or `undefined` if it can.
     * @returns {string | undefined}
     */
    validationError() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.scenario_validationError(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            let v1;
            if (r0 !== 0) {
                v1 = getStringFromWasm0(r0, r1);
                wasm.__wbindgen_export(r0, r1 * 1, 1);
            }
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * The baseline with one disturbance switched on.
     *
     * # Errors
     *
     * Throws if `id` is outside `1..=20`.
     * @param {number} id
     * @returns {Scenario}
     */
    static withFault(id) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.scenario_withFault(retptr, id);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return Scenario.__wrap(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
}
if (Symbol.dispose) Scenario.prototype[Symbol.dispose] = Scenario.prototype.free;

/**
 * A run in progress, advanced in chunks.
 *
 * The shape this is built for is one `Sim` per Web Worker, driven by a loop
 * that calls [`Sim::step_chunk`], posts the result with its buffer in the
 * transfer list, and yields to the event loop so the worker can answer control
 * messages between chunks. `crates/tepsim-wasm/www/worker.js` is that loop, in
 * about forty lines.
 */
export class Sim {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        SimFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_sim_free(ptr, 0);
    }
    /**
     * Ground truth: the disturbances active as of the most recent sample.
     *
     * Not the same as the scenario's list. The driver switches `IDV(12)` on at
     * eight hours whatever was asked for, so this list can grow mid-run
     * without the caller doing anything. Delta D-011.
     * @returns {Uint8Array}
     */
    get activeFaults() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.sim_activeFaults(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v1 = getArrayU8FromWasm0(r0, r1).slice();
            wasm.__wbindgen_export(r0, r1 * 1, 1);
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * A 16-character hex digest of every value emitted so far.
     *
     * FNV-1a over IEEE 754 bit patterns, integer arithmetic throughout, so the
     * digest cannot itself become a source of the cross-platform disagreement
     * it exists to detect. Print it beside a native run of the same scenario:
     * equal digests mean the wasm build reproduced the native one bit for bit.
     * @returns {string}
     */
    checksum() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.sim_checksum(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Short stable identifiers for the columns, in order.
     * @returns {string[]}
     */
    get columnIds() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.sim_columnIds(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v1 = getArrayJsValueFromWasm0(r0, r1);
            wasm.__wbindgen_export(r0, r1 * 4, 4);
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Human-readable labels for the columns, in order.
     * @returns {string[]}
     */
    get columnLabels() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.sim_columnLabels(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v1 = getArrayJsValueFromWasm0(r0, r1);
            wasm.__wbindgen_export(r0, r1 * 4, 4);
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * How many samples have been emitted.
     * @returns {number}
     */
    get emittedSamples() {
        const ret = wasm.sim_emittedSamples(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Simulated time so far, in hours.
     * @returns {number}
     */
    get hours() {
        const ret = wasm.sim_hours(this.__wbg_ptr);
        return ret;
    }
    /**
     * Hours since `IDV(id)` came on, or `undefined` if it never did.
     *
     * The number a detection-delay figure should be measured against. The
     * original records nothing of the sort, which is why delays in the
     * literature are computed against whatever onset the author assumed.
     * @param {number} id
     * @returns {number | undefined}
     */
    hoursSinceOnset(id) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.sim_hoursSinceOnset(retptr, this.__wbg_ptr, id);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r2 = getDataViewMemory0().getFloat64(retptr + 8 * 1, true);
            return r0 === 0 ? undefined : r2;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Whether the run has stopped producing samples.
     * @returns {boolean}
     */
    get isFinished() {
        const ret = wasm.sim_isFinished(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Validate a scenario and start a run at step zero.
     *
     * # Errors
     *
     * Throws if the scenario cannot produce a well-defined run: a seed that is
     * not finite and positive, a non-positive duration or step, a zero
     * sampling cadence, or a run past the step or sample limit.
     * @param {Scenario} scenario
     */
    constructor(scenario) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            _assertClass(scenario, Scenario);
            wasm.sim_new(retptr, scenario.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            this.__wbg_ptr = r0;
            SimFinalization.register(this, this.__wbg_ptr, this);
            return this;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * How the run ended: `completed`, `tripped`, `solve_failed`, or
     * `undefined` while it is still going.
     *
     * A trip appears here as soon as it happens even though the run continues
     * past it: `teprob.f:807-811` freezes the plant rather than stopping it,
     * and the frozen samples are part of the behaviour under test.
     * @returns {string | undefined}
     */
    get outcome() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.sim_outcome(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            let v1;
            if (r0 !== 0) {
                v1 = getStringFromWasm0(r0, r1);
                wasm.__wbindgen_export(r0, r1 * 1, 1);
            }
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Whether a requested change needs the run rebuilt to take effect.
     * @returns {boolean}
     */
    get pendingRestart() {
        const ret = wasm.sim_pendingRestart(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Fraction of the planned samples emitted, in `0.0..=1.0`.
     * @returns {number}
     */
    get progress() {
        const ret = wasm.sim_progress(this.__wbg_ptr);
        return ret;
    }
    /**
     * The scenario as requested, including changes not yet in effect.
     *
     * Pass it to the [`Sim`] constructor to apply them.
     * @returns {Scenario}
     */
    get requestedScenario() {
        const ret = wasm.sim_requestedScenario(this.__wbg_ptr);
        return Scenario.__wrap(ret);
    }
    /**
     * Values per packed row: the time, then the 53 channels.
     *
     * The stride into the `Float64Array` [`Sim::step_chunk`] returns.
     * @returns {number}
     */
    get rowWidth() {
        const ret = wasm.sim_rowWidth(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Run to the end and return every remaining row in one array.
     *
     * The same numbers `tepsim::Simulation::run` produces, in the same order.
     * For tests and short runs: a worker that calls this cannot answer a
     * message until the run is over, which is what [`Sim::step_chunk`] exists
     * to avoid.
     * @returns {Float64Array}
     */
    runToEnd() {
        const ret = wasm.sim_runToEnd(this.__wbg_ptr);
        return takeObject(ret);
    }
    /**
     * The scenario this run is executing.
     * @returns {Scenario}
     */
    get scenario() {
        const ret = wasm.sim_scenario(this.__wbg_ptr);
        return Scenario.__wrap(ret);
    }
    /**
     * Ask for `IDV(id)` to be set or cleared.
     *
     * # Errors
     *
     * Throws if `id` is outside `1..=20`.
     *
     * # This does not change a run already under way
     *
     * `tepsim::Simulation` takes its scenario at construction and hands the
     * disturbance vector to the driver there. There is no seam in the facade
     * for changing it mid-run, and these bindings must not invent one: a
     * disturbance that switched on through a path the native API does not have
     * would produce a run no native caller could reproduce, and reproducibility
     * is the entire point of this project.
     *
     * So the request is recorded and [`Sim::pending_restart`] goes `true`. The
     * browser's honest move is to rebuild: `new Sim(sim.requestedScenario)`.
     * At a hundred times real time that is quick enough to feel immediate, and
     * the run it produces is one a native caller can reproduce exactly.
     * @param {number} id
     * @param {boolean} active
     */
    setFault(id, active) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.sim_setFault(retptr, this.__wbg_ptr, id, active);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            if (r1) {
                throw takeObject(r0);
            }
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Advance the run by at most `max_samples` samples.
     *
     * Returns a fresh `Float64Array`, row-major, `rowWidth` values per row,
     * holding **whatever completed**: fewer rows than asked for at the end of
     * a run, an empty array once it is over. The caller never has to work out
     * how many are left.
     *
     * One sample costs `sampleEvery` integrator steps, so the chunk size is a
     * latency control: it is how long this thread is unavailable. A worker
     * that wants to answer a "stop" within a frame should pick a chunk it can
     * finish in a frame.
     *
     * The returned array owns its `ArrayBuffer`, so it goes straight into a
     * `postMessage` transfer list. After transfer it is detached on this side,
     * which is correct: this side is done with it.
     * @param {number} max_samples
     * @returns {Float64Array}
     */
    stepChunk(max_samples) {
        const ret = wasm.sim_stepChunk(this.__wbg_ptr, max_samples);
        return takeObject(ret);
    }
    /**
     * How many integrator steps have run.
     * @returns {number}
     */
    get stepsTaken() {
        const ret = wasm.sim_stepsTaken(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * How many samples the run will record if nothing ends it early.
     * @returns {number}
     */
    get totalSamples() {
        const ret = wasm.sim_totalSamples(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * How many integrator steps the run will take.
     * @returns {number}
     */
    get totalSteps() {
        const ret = wasm.sim_totalSteps(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * The first shutdown condition that fired, or `undefined`.
     *
     * One of the eight `teprob.f:703-710` tests, as prose.
     * @returns {string | undefined}
     */
    get tripCause() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.sim_tripCause(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            let v1;
            if (r0 !== 0) {
                v1 = getStringFromWasm0(r0, r1);
                wasm.__wbindgen_export(r0, r1 * 1, 1);
            }
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Simulated hours at which the plant tripped, or `undefined`.
     * @returns {number | undefined}
     */
    get tripHours() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.sim_tripHours(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r2 = getDataViewMemory0().getFloat64(retptr + 8 * 1, true);
            return r0 === 0 ? undefined : r2;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
}
if (Symbol.dispose) Sim.prototype[Symbol.dispose] = Sim.prototype.free;

/**
 * The algorithm behind [`Sim::checksum`] and [`Scenario::digest`].
 * @returns {string}
 */
export function checksumAlgorithm() {
    let deferred1_0;
    let deferred1_1;
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        wasm.checksumAlgorithm(retptr);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        deferred1_0 = r0;
        deferred1_1 = r1;
        return getStringFromWasm0(r0, r1);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
        wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
    }
}

/**
 * Short stable identifiers for the columns of a packed row.
 * @returns {string[]}
 */
export function columnIds() {
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        wasm.columnIds(retptr);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        var v1 = getArrayJsValueFromWasm0(r0, r1);
        wasm.__wbindgen_export(r0, r1 * 4, 4);
        return v1;
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
    }
}

/**
 * Human-readable labels for the columns of a packed row.
 * @returns {string[]}
 */
export function columnLabels() {
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        wasm.columnLabels(retptr);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        var v1 = getArrayJsValueFromWasm0(r0, r1);
        wasm.__wbindgen_export(r0, r1 * 4, 4);
        return v1;
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
    }
}

/**
 * Units for the columns of a packed row.
 * @returns {string[]}
 */
export function columnUnits() {
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        wasm.columnUnits(retptr);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        var v1 = getArrayJsValueFromWasm0(r0, r1);
        wasm.__wbindgen_export(r0, r1 * 4, 4);
        return v1;
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
    }
}

/**
 * How many disturbances this model has.
 *
 * Twenty, not the twenty-one of the later literature: `teprob.f:340` loops
 * `DO 500 I=1,20`.
 * @returns {number}
 */
export function faultCount() {
    const ret = wasm.faultCount();
    return ret >>> 0;
}

/**
 * One disturbance by its one-based index, or `undefined` if out of range.
 * @param {number} id
 * @returns {Fault | undefined}
 */
export function faultInfo(id) {
    const ret = wasm.faultInfo(id);
    return ret === 0 ? undefined : Fault.__wrap(ret);
}

/**
 * The twenty `IDV` disturbances, from `tepsim_core::FAULTS`.
 * @returns {Fault[]}
 */
export function faults() {
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        wasm.faults(retptr);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        var v1 = getArrayJsValueFromWasm0(r0, r1);
        wasm.__wbindgen_export(r0, r1 * 4, 4);
        return v1;
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
    }
}

/**
 * The integrators a scenario can choose: `euler`, `rk4`, `dopri5`.
 * @returns {string[]}
 */
export function integrators() {
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        wasm.integrators(retptr);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        var v1 = getArrayJsValueFromWasm0(r0, r1);
        wasm.__wbindgen_export(r0, r1 * 4, 4);
        return v1;
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
    }
}

/**
 * How many manipulated variables the plant accepts. `XMV(12)`.
 * @returns {number}
 */
export function manipulatedCount() {
    const ret = wasm.manipulatedCount();
    return ret >>> 0;
}

/**
 * The largest number of samples one scenario may record.
 * @returns {number}
 */
export function maxSamples() {
    const ret = wasm.maxSamples();
    return ret >>> 0;
}

/**
 * The largest number of integrator steps one scenario may ask for.
 * @returns {number}
 */
export function maxSteps() {
    const ret = wasm.maxSteps();
    return ret >>> 0;
}

/**
 * How many measurements the plant reports. `XMEAS(41)`.
 * @returns {number}
 */
export function measurementCount() {
    const ret = wasm.measurementCount();
    return ret >>> 0;
}

/**
 * Values per packed row: the time, then the 53 channels.
 * @returns {number}
 */
export function rowWidth() {
    const ret = wasm.rowWidth();
    return ret >>> 0;
}

/**
 * Zero-based offsets of the analyser-sampled measurements, `XMEAS(23..=41)`.
 *
 * These hold their value between analyser reports, so a chart should draw them
 * as steps rather than interpolating.
 * @returns {Uint32Array}
 */
export function sampledColumns() {
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        wasm.sampledColumns(retptr);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        var v1 = getArrayU32FromWasm0(r0, r1).slice();
        wasm.__wbindgen_export(r0, r1 * 4, 4);
        return v1;
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
    }
}

/**
 * Run a fixed one-hour baseline scenario and return the digest of its output.
 *
 * The cross-platform determinism check, reduced to one string. A browser that
 * prints a different value from a native run of the same commit has broken the
 * invariant the whole validation ladder rests on, and the page should say so
 * loudly rather than plot the numbers.
 *
 * 3,600 integrator steps and 20 samples: quick enough to run on page load.
 * @returns {string}
 */
export function selfCheckDigest() {
    let deferred1_0;
    let deferred1_1;
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        wasm.selfCheckDigest(retptr);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        deferred1_0 = r0;
        deferred1_1 = r1;
        return getStringFromWasm0(r0, r1);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
        wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
    }
}

/**
 * The crate version, from `Cargo.toml`.
 * @returns {string}
 */
export function version() {
    let deferred1_0;
    let deferred1_1;
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        wasm.version(retptr);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        deferred1_0 = r0;
        deferred1_1 = r1;
        return getStringFromWasm0(r0, r1);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
        wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
    }
}
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg_Error_408e67f47ca7b58b: function(arg0, arg1) {
            const ret = Error(getStringFromWasm0(arg0, arg1));
            return addHeapObject(ret);
        },
        __wbg___wbindgen_throw_bb96b2010945f0bc: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg_fault_new: function(arg0) {
            const ret = Fault.__wrap(arg0);
            return addHeapObject(ret);
        },
        __wbg_new_from_slice_f6e95bc2809a2b07: function(arg0, arg1) {
            const ret = new Float64Array(getArrayF64FromWasm0(arg0, arg1));
            return addHeapObject(ret);
        },
        __wbindgen_cast_0000000000000001: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return addHeapObject(ret);
        },
    };
    return {
        __proto__: null,
        "./tepsim_wasm_bg.js": import0,
    };
}

const FaultFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_fault_free(ptr, 1));
const ScenarioFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_scenario_free(ptr, 1));
const SimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_sim_free(ptr, 1));

function addHeapObject(obj) {
    if (heap_next === heap.length) heap.push(heap.length + 1);
    const idx = heap_next;
    heap_next = heap[idx];

    heap[idx] = obj;
    return idx;
}

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
}

function dropObject(idx) {
    if (idx < 1028) return;
    heap[idx] = heap_next;
    heap_next = idx;
}

function getArrayF64FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat64ArrayMemory0().subarray(ptr / 8, ptr / 8 + len);
}

function getArrayJsValueFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    const mem = getDataViewMemory0();
    const result = [];
    for (let i = ptr; i < ptr + 4 * len; i += 4) {
        result.push(takeObject(mem.getUint32(i, true)));
    }
    return result;
}

function getArrayU32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

let cachedFloat64ArrayMemory0 = null;
function getFloat64ArrayMemory0() {
    if (cachedFloat64ArrayMemory0 === null || cachedFloat64ArrayMemory0.byteLength === 0) {
        cachedFloat64ArrayMemory0 = new Float64Array(wasm.memory.buffer);
    }
    return cachedFloat64ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    return decodeText(ptr >>> 0, len);
}

let cachedUint32ArrayMemory0 = null;
function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function getObject(idx) { return heap[idx]; }

let heap = new Array(1024).fill(undefined);
heap.push(undefined, null, true, false);

let heap_next = heap.length;

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeObject(idx) {
    const ret = getObject(idx);
    dropObject(idx);
    return ret;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedDataViewMemory0 = null;
    cachedFloat64ArrayMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (!module.ok) {
            throw new Error(`failed to fetch Wasm: ${module.status} ${module.statusText} fetching '${module.url}'`);
        }

        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('tepsim_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
