"""Daniell cell: conservative finite volumes + an implicitly coupled load.

SI internally. Cu concentration in a planar diffusion film and a FINITE mixed
reservoir; Zn half-cell is well mixed. Ideal activities, reversible interfaces,
excess supporting electrolyte, constant user-selected T, no convection inside film.
E_standard, D and resistance are intentionally held fixed when T is varied.
This is a deliberately simplified model, not a calibrated battery predictor.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from typing import Literal
import math
import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import brentq

F = 96485.33212
RGAS = 8.31446261815324
T_K = 298.15
N_E = 2
NF = N_E * F
E_STANDARD = 1.10
ALPHA = RGAS * T_K / NF
MOLAR_CU = 63.546  # g/mol; not used in the electrochemical solver


@dataclass(frozen=True)
class Settings:
    cu_M: float = 0.10
    temperature_C: float = 25.0
    zn_M: float = 0.10
    volume_cu_ml: float = 0.50
    volume_zn_ml: float = 0.50
    area_cm2: float = 1.0
    film_mm: float = 0.50
    diffusion_m2_s: float = 0.70e-9
    resistance_ohm: float = 1000.0
    internal_ohm: float = 50.0
    duration_s: float = 3600.0
    dt_s: float = 2.0
    cells: int = 48
    protocol: str = "continuous"  # continuous, rest, open
    rest_start_frac: float = 0.35
    rest_end_frac: float = 0.65
    surface_floor_M: float = 1.0e-8

    @property
    def alpha(self) -> float:
        return RGAS * (self.temperature_C + 273.15) / NF

    @property
    def area(self) -> float:
        return self.area_cm2 * 1e-4

    @property
    def length(self) -> float:
        return self.film_mm * 1e-3

    @property
    def vcu(self) -> float:
        return self.volume_cu_ml * 1e-6

    @property
    def vzn(self) -> float:
        return self.volume_zn_ml * 1e-6

    @property
    def total_r(self) -> float:
        return self.resistance_ohm + self.internal_ohm

    def validate(self) -> None:
        for key, value in asdict(self).items():
            if isinstance(value, (float, int)) and not math.isfinite(value):
                raise ValueError(f"Non-finite parameter: {key}")
        positive = ("cu_M", "zn_M", "volume_cu_ml", "volume_zn_ml",
                    "area_cm2", "film_mm", "diffusion_m2_s", "resistance_ohm",
                    "duration_s", "dt_s", "surface_floor_M")
        for key in positive:
            if getattr(self, key) <= 0:
                raise ValueError(f"Parameter must be positive: {key}")
        if not 0 <= self.temperature_C <= 60:
            raise ValueError("Use a temperature from 0 to 60 degrees Celsius.")
        if self.internal_ohm < 0:
            raise ValueError("Internal resistance must be nonnegative.")
        if isinstance(self.cells, bool) or int(self.cells) != self.cells or not 8 <= self.cells <= 240:
            raise ValueError("Use an integer from 8 to 240 for cells.")
        if self.area * self.length >= self.vcu * 0.95:
            raise ValueError("The diffusion film must occupy <95% of the Cu solution volume.")
        if min(self.cu_M, self.zn_M) <= self.surface_floor_M * 10:
            raise ValueError("Initial concentrations must exceed the numerical cutoff.")
        if self.duration_s / self.dt_s > 24000:
            raise ValueError("Too many time steps; increase dt or shorten duration.")
        if self.protocol not in {"continuous", "rest", "open"}:
            raise ValueError("Unknown load protocol.")
        if not 0 < self.rest_start_frac < self.rest_end_frac < 1:
            raise ValueError("Rest fractions must satisfy 0 < start < end < 1.")

    def is_closed(self, time_s: float) -> bool:
        if self.protocol == "open":
            return False
        if self.protocol == "rest":
            start = self.duration_s * self.rest_start_frac
            end = self.duration_s * self.rest_end_frac
            return not (start <= time_s < end)
        return True


def nernst(cu: float | np.ndarray, zn: float | np.ndarray,
           temperature_C: float = 25.0):
    """Ideal activities; E_standard is fixed at the educational value 1.10 V.

    Temperature changes ONLY RT/(2F); this is not a complete temperature model.
    The ratio of equal-unit concentrations equals the activity ratio only under
    the ideal-solution assumption. No logarithm is evaluated at c <= 0.
    """
    cu, zn = np.asarray(cu, dtype=float), np.asarray(zn, dtype=float)
    if (not np.all(np.isfinite(cu)) or not np.all(np.isfinite(zn))
            or np.any(cu <= 0) or np.any(zn <= 0)):
        raise ValueError("Nernst requires positive, finite concentrations.")
    if not math.isfinite(temperature_C) or temperature_C <= -273.15:
        raise ValueError("Absolute temperature must be positive and finite.")
    alpha = RGAS * (temperature_C + 273.15) / NF
    answer = E_STANDARD + alpha * (np.log(cu) - np.log(zn))
    return float(answer) if answer.ndim == 0 else answer


def nernst_terms(cu_M: float, zn_M: float, temperature_C: float = 25.0) -> dict:
    """Return every live term shown in the equation panel."""
    e = nernst(cu_M, zn_M, temperature_C)
    k = RGAS * (temperature_C + 273.15) / NF * math.log(10)
    logq = math.log10(zn_M) - math.log10(cu_M)
    return {"Q": zn_M / cu_M, "log10_Q": logq,
            "temperature_K": temperature_C + 273.15,
            "k_V_per_decade": k, "concentration_shift_V": -k*logq,
            "E_V": e, "E_standard_V": E_STANDARD,
            "log10_K_ideal": E_STANDARD / k}


def capacity_mAh(cu_M: float, volume_cu_ml: float) -> float:
    """Stoichiometric upper bound, not usable capacity; Zn is assumed in excess."""
    if cu_M <= 0 or volume_cu_ml <= 0:
        raise ValueError("Concentration and volume must be positive.")
    return cu_M * volume_cu_ml * 1e-3 * NF / 3.6


def solve_surface(a: float, beta: float, zn: float, zn_slope: float,
                  resistance: float, temperature_C: float = 25.0) -> tuple[float, float]:
    """Solve c_s=a-beta*I and E0+alpha*ln(c_s/(zn+zn_slope*I))=R*I.

    A log-concentration root avoids subtractive cancellation near transport
    limitation. Very small concentrations are used ONLY to locate a cutoff;
    states below Settings.surface_floor_M are never accepted in a trajectory.
    """
    if a <= 0 or zn <= 0 or beta < 0 or resistance <= 0:
        raise ValueError("Invalid nonlinear circuit coefficients.")
    alpha = RGAS * (temperature_C + 273.15) / NF
    if nernst(a, zn, temperature_C) <= 0:
        return 0.0, a  # this app implements spontaneous discharge only
    if beta == 0:
        return nernst(a, zn, temperature_C) / resistance, a
    loga = math.log(a)

    def residual(log_ratio: float) -> float:
        # y = log(c_s/a), NOT log(c_s). This avoids cancellation of two
        # nearly equal logarithms at high concentrations / small currents.
        current = -a * math.expm1(log_ratio) / beta
        return (E_STANDARD + alpha *
                (loga + log_ratio - math.log(zn + zn_slope * current))
                - resistance * current)

    y = brentq(residual, -200.0, 0.0, xtol=1e-300, rtol=1e-14, maxiter=200)
    current = -a * math.expm1(y) / beta
    return current, a * math.exp(y)



class FilmOperator:
    """Conservative tridiagonal backward-Euler diffusion operator.

    First N cells have volume A*dx. The last unknown is a finite mixed bulk.
    At x=L, film/bulk exchange uses the distance dx/2, not a fixed reservoir.
    At x=0, Cu consumption is I/(2F). There is no Cu transfer to the Zn cell.
    """
    def __init__(self, settings: Settings):
        self.s = settings
        self.n = settings.cells
        self.dx = settings.length / self.n
        self.volumes = np.r_[np.full(self.n, settings.area * self.dx),
                             settings.vcu - settings.area * settings.length]
        conductance = settings.area * settings.diffusion_m2_s / self.dx
        self.g = np.r_[np.full(self.n - 1, conductance), 2.0 * conductance]
        self.gamma = self.dx / (2 * NF * settings.area * settings.diffusion_m2_s)
        self._cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    def matrix_response(self, dt: float):
        # Normal time steps share the matrix and response; event refinements do not.
        key = round(dt, 10)
        if key not in self._cache:
            diag = self.volumes.copy()
            diag[:-1] += dt * self.g
            diag[1:] += dt * self.g
            ab = np.zeros((3, self.n + 1))
            ab[1] = diag
            ab[0, 1:] = -dt * self.g
            ab[2, :-1] = -dt * self.g
            rhs = np.zeros(self.n + 1)
            rhs[0] = dt / NF
            response = solve_banded((1, 1), ab, rhs, check_finite=False)
            if len(self._cache) > 40:
                self._cache.clear()
            self._cache[key] = ab, response
        return self._cache[key]

    def diffuse(self, concentration: np.ndarray, dt: float):
        ab, response = self.matrix_response(dt)
        base = solve_banded((1, 1), ab, self.volumes * concentration,
                            check_finite=False)
        return base, response


@dataclass
class State:
    cu: np.ndarray
    zn: float
    charge: float = 0.0
    energy: float = 0.0
    internal_heat: float = 0.0


def _advance(state: State, s: Settings, op: FilmOperator | None,
             dt: float, closed: bool) -> tuple[State, float, float]:
    if op is None:
        base = state.cu.copy()
        response = np.array([dt / (NF * s.vcu)])
        beta = response[0]
    else:
        base, response = op.diffuse(state.cu, dt)
        beta = response[0] + op.gamma
    if closed:
        current, surface = solve_surface(float(base[0]), float(beta), state.zn,
                                         dt / (NF * s.vzn), s.total_r, s.temperature_C)
    else:
        current, surface = 0.0, float(base[0])
    cu = base - response * current
    # No clipping or silent inventory repair: conservation follows the FV scheme.
    if np.min(cu) < -1e-10 or not np.all(np.isfinite(cu)):
        raise ArithmeticError("The concentration solver produced an invalid state.")
    new = State(cu=cu, zn=state.zn + current * dt / (NF * s.vzn),
                charge=state.charge + current * dt,
                energy=state.energy + current * current * s.resistance_ohm * dt,
                internal_heat=state.internal_heat + current * current * s.internal_ohm * dt)
    return new, current, surface


def _observe(state: State, s: Settings, op: FilmOperator | None,
             closed: bool, initial: bool = False) -> tuple[float, float]:
    if not closed:
        return 0.0, float(state.cu[0])
    if op is None or initial:
        return max(nernst(state.cu[0], state.zn, s.temperature_C), 0.0) / s.total_r, float(state.cu[0])
    return solve_surface(float(state.cu[0]), op.gamma, state.zn, 0.0, s.total_r, s.temperature_C)


SERIES_FIELDS = ("time_s", "current_A", "step_current_A", "terminal_V", "bulk_nernst_V",
                 "surface_nernst_V", "cu_average_M", "cu_bulk_M", "cu_surface_M",
                 "zn_M", "charge_C", "energy_J", "internal_heat_J", "closed",
                 "cu_balance_error_mol", "zn_balance_error_mol")


def simulate(settings: Settings) -> dict:
    """Advance A/B together, stopping BOTH at the first validity boundary.

    Equal comparison duration is guaranteed. Charge and delivered energy use
    interval currents, not trapezoidal integration across ideal switches.
    """
    s = settings
    s.validate()
    op = FilmOperator(s)
    mixed = State(np.array([s.cu_M * 1000.0]), s.zn_M * 1000.0)
    film = State(np.full(s.cells + 1, s.cu_M * 1000.0), s.zn_M * 1000.0)
    data = {key: {f: [] for f in SERIES_FIELDS} for key in ("mixed", "diffusion")}
    profiles = []
    initial_ncu = s.cu_M * 1000 * s.vcu
    initial_nzn = s.zn_M * 1000 * s.vzn
    floor = s.surface_floor_M * 1000
    status = "duration"
    limiting_model = None
    t = 0.0

    def record(time: float, a: State, b: State, step_a: float, step_b: float):
        closed = s.is_closed(time + 1e-8)
        for name, state, operator, step_i in (("mixed", a, None, step_a),
                                              ("diffusion", b, op, step_b)):
            current, cs = _observe(state, s, operator, closed, initial=(time == 0))
            ncu = (float(state.cu[0]) * s.vcu if operator is None
                   else float(np.dot(op.volumes, state.cu)))
            cavg = ncu / s.vcu
            es = nernst(cs, state.zn, s.temperature_C)
            terminal = current * s.resistance_ohm if closed else es
            values = (time, current, step_i, terminal, nernst(cavg, state.zn, s.temperature_C), es,
                      cavg / 1000, float(state.cu[-1]) / 1000, cs / 1000,
                      state.zn / 1000, state.charge, state.energy, state.internal_heat,
                      int(closed), initial_ncu - ncu - state.charge / NF,
                      state.zn * s.vzn - initial_nzn - state.charge / NF)
            for field, value in zip(SERIES_FIELDS, values):
                data[name][field].append(value)
        profiles.append(b.cu.copy() / 1000)

    record(0.0, mixed, film, 0.0, 0.0)
    events = [s.duration_s]
    if s.protocol == "rest":
        events += [s.duration_s * s.rest_start_frac, s.duration_s * s.rest_end_frac]
    events = sorted(events)
    for _ in range(26000):
        if t >= s.duration_s - 1e-9:
            break
        next_event = min(e for e in events if e > t + 1e-8)
        dt = min(s.dt_s, next_event - t, s.duration_s - t)
        closed = s.is_closed(t + dt * 0.5)

        def proposal(h):
            a, ia, ca = _advance(mixed, s, None, h, closed)
            b, ib, cb = _advance(film, s, op, h, closed)
            return a, ia, ca, b, ib, cb

        candidate = proposal(dt)
        if min(candidate[2], candidate[5]) < floor:
            limiting_model = "mixed" if candidate[2] < candidate[5] else "diffusion"
            # Locate the numerical model-domain event without accepting negatives.
            lo, hi = 0.0, dt
            accepted = None
            for _ in range(35):
                mid = (lo + hi) / 2
                trial = proposal(mid)
                if min(trial[2], trial[5]) >= floor:
                    lo, accepted = mid, trial
                else:
                    hi = mid
            if accepted is not None and lo > 1e-7:
                mixed, ia, _, film, ib, _ = accepted
                t += lo
                record(t, mixed, film, ia, ib)
            status = "surface_cutoff"
            break
        mixed, ia, _, film, ib, _ = candidate
        t += dt
        record(t, mixed, film, ia, ib)
        # A load reconnection can cross the surface cutoff instantaneously.
        last_cs = min(data["mixed"]["cu_surface_M"][-1],
                      data["diffusion"]["cu_surface_M"][-1])
        if last_cs < s.surface_floor_M:
            status, limiting_model = "reconnect_cutoff", "diffusion"
            break
    else:
        raise RuntimeError("Time step limit exceeded.")

    for name in data:
        for field in data[name]:
            data[name][field] = np.asarray(data[name][field], dtype=float)
    result = {
        "settings": asdict(s), "mixed": data["mixed"], "diffusion": data["diffusion"],
        "profiles_M": np.asarray(profiles),
        "x_mm": (np.arange(s.cells) + 0.5) * op.dx * 1000,
        "status": status, "limiting_model": limiting_model,
        "actual_duration_s": t, "initial_cu_mol": initial_ncu,
        "theoretical_capacity_mAh": initial_ncu * NF / 3.6,
        "film_volume_ml": s.area * s.length * 1e6,
    }
    result["checks"] = diagnostics(result)
    return result


def diagnostics(result: dict) -> dict:
    s = Settings(**result["settings"])
    answer = {}
    for model in ("mixed", "diffusion"):
        d = result[model]
        closed = d["closed"] > 0.5
        voltage_residual = np.abs(d["terminal_V"] - (d["surface_nernst_V"] - d["current_A"] * s.internal_ohm))
        circuit = np.max(voltage_residual)
        faraday = max(float(np.max(np.abs(d["cu_balance_error_mol"]))),
                      float(np.max(np.abs(d["zn_balance_error_mol"]))))
        # Explicitly test interval-integrated charge, including open-circuit intervals.
        q_from_current = np.sum(np.diff(d["time_s"]) * d["step_current_A"][1:])
        answer[model] = {
            "mass_error_mol": faraday,
            "relative_mass_error": faraday / result["initial_cu_mol"],
            "charge_error_C": abs(float(d["charge_C"][-1]) - q_from_current),
            "circuit_error_V": float(circuit),
            "minimum_surface_M": float(np.min(d["cu_surface_M"])),
            "valid": bool(faraday / result["initial_cu_mol"] < 1e-8 and circuit < 1e-8
                          and np.all(np.isfinite(d["terminal_V"]))
                          and np.all(d["cu_surface_M"] > 0)),
        }
    return answer


def brownian_experiment(diffusion: float, duration: float = 60.0,
                        particles: int = 2400, steps: int = 300, seed: int = 42) -> dict:
    """Independent free 1-D diffusion experiment; NOT a cell reaction solver.

    Each increment is Normal(0, sqrt(2*D*dt)). Initial x=0, no walls and no
    electric field. Mean-square displacement tends to 2Dt for many particles.
    """
    if diffusion <= 0 or duration <= 0 or steps < 2 or particles < 2:
        raise ValueError("Invalid Brownian experiment settings.")
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration, steps + 1)
    inc = rng.normal(0.0, math.sqrt(2 * diffusion * duration / steps),
                     size=(particles, steps))
    x = np.c_[np.zeros(particles), np.cumsum(inc, axis=1)]
    return {"time_s": t, "trajectories_m": x[:12], "msd_m2": np.mean(x*x, axis=0),
            "theory_m2": 2 * diffusion * t, "seed": seed, "particles": particles}


def convergence_experiment(settings: Settings) -> list[dict]:
    """Compare two dt values AND two meshes at one common requested horizon."""
    base = replace(settings, duration_s=min(settings.duration_s, 600.0),
                   protocol="continuous", dt_s=min(settings.dt_s, 2.0), cells=min(settings.cells, 80))
    rows = []
    for factor_dt, factor_x in ((1, 1), (0.5, 1), (0.5, 2)):
        s = replace(base, dt_s=base.dt_s * factor_dt, cells=base.cells * factor_x)
        r = simulate(s)
        d = r["diffusion"]
        rows.append({"dt_s": s.dt_s, "cells": s.cells, "duration_s": r["actual_duration_s"],
                     "status": r["status"], "charge_C": float(d["charge_C"][-1]),
                     "terminal_V": float(d["terminal_V"][-1]),
                     "surface_M": float(d["cu_surface_M"][-1])})
    return rows
