from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physics import (Settings, nernst, NF, ALPHA, FilmOperator, simulate,
                     brownian_experiment, convergence_experiment)
from visuals import replay_html as player_html, result_payload as payload, timeseries_frame, profile_frame


@pytest.fixture(scope='module')
def default():
    return simulate(Settings())


def test_equal_concentrations_give_standard_voltage():
    assert nernst(.1, .1) == pytest.approx(1.10)


def test_nernst_ratio_scale_invariance():
    assert nernst(.01, .1) == pytest.approx(nernst(.1, 1.0), abs=1e-14)


def test_tenfold_ratio_changes_voltage_by_rt_log10_over_2f():
    assert nernst(.01, .1)-nernst(.1, .1) == pytest.approx(-ALPHA*np.log(10))


@pytest.mark.parametrize('cu,zn', [(0,1), (1,0), (-1,1)])
def test_nonpositive_log_inputs_are_rejected(cu, zn):
    with pytest.raises(ValueError):
        nernst(cu,zn)


@pytest.mark.parametrize('name,value', [
    ('diffusion_m2_s',0),('resistance_ohm',0),('internal_ohm',-1),
    ('cu_M',-1),('dt_s',0),('area_cm2',float('nan')),('cells',2),
    ('duration_s',float('inf')),('cells',25.5)])
def test_invalid_settings_are_rejected(name,value):
    with pytest.raises(ValueError):
        replace(Settings(),**{name:value}).validate()


def test_film_and_bulk_volumes_sum_to_total():
    s=Settings();op=FilmOperator(s)
    assert op.volumes.sum()==pytest.approx(s.vcu,rel=1e-14)
    assert op.volumes[-1]>0


def test_impossible_geometry_rejected():
    with pytest.raises(ValueError):
        Settings(area_cm2=5,film_mm=2,volume_cu_ml=.2).validate()


def test_open_circuit_is_stationary():
    r=simulate(Settings(protocol='open',duration_s=600))
    for model in ['mixed','diffusion']:
        assert np.max(np.abs(r[model]['current_A']))==0
        assert r[model]['charge_C'][-1]==0
        assert np.max(np.abs(r[model]['terminal_V']-1.10))<1e-11
    np.testing.assert_allclose(r['profiles_M'],.1,atol=1e-12)


def test_moles_and_charge_conserved(default):
    for model in ['mixed','diffusion']:
        c=default['checks'][model]
        assert c['relative_mass_error']<1e-10
        assert c['charge_error_C']<1e-10
        assert c['valid']


def test_current_voltage_consistency(default):
    s=Settings()
    for m in ['mixed','diffusion']:
        d=default[m]
        np.testing.assert_allclose(d['terminal_V'],s.resistance_ohm*d['current_A'],atol=1e-10)
        np.testing.assert_allclose(d['surface_nernst_V'],s.total_r*d['current_A'],atol=1e-9)


def test_uniform_model_faraday_law(default):
    s=Settings();d=default['mixed']
    n_used=(s.cu_M-d['cu_average_M'])*1000*s.vcu
    np.testing.assert_allclose(n_used*NF,d['charge_C'],atol=1e-11)


def test_diffusion_reduces_current_for_default(default):
    assert default['diffusion']['current_A'][-1]<default['mixed']['current_A'][-1]
    assert default['diffusion']['charge_C'][-1]<default['mixed']['charge_C'][-1]
    assert default['diffusion']['cu_surface_M'][-1]<default['diffusion']['cu_average_M'][-1]


def test_surface_cutoff_not_zero_voltage():
    s=Settings(resistance_ohm=100,duration_s=600,dt_s=.5)
    r=simulate(s)
    assert r['status']=='surface_cutoff'
    assert 0<r['actual_duration_s']<s.duration_s
    assert r['diffusion']['terminal_V'][-1]>.1
    assert r['diffusion']['cu_average_M'][-1]>.09
    assert r['diffusion']['cu_surface_M'][-1]>=s.surface_floor_M*(1-1e-5)
    assert np.min(r['profiles_M'])>0
    assert r['mixed']['time_s'][-1]==r['diffusion']['time_s'][-1]


def test_rest_conserves_charge_and_recovers_surface():
    s=Settings(protocol='rest',duration_s=600,dt_s=1.3)
    r=simulate(s);d=r['diffusion'];t=d['time_s']
    start=s.duration_s*s.rest_start_frac;end=s.duration_s*s.rest_end_frac
    assert np.any(np.isclose(t,start)) and np.any(np.isclose(t,end))
    ix=np.flatnonzero((t>=start)&(t<end))
    assert np.max(np.abs(d['current_A'][ix]))==0
    np.testing.assert_allclose(d['charge_C'][ix],d['charge_C'][ix[0]],atol=1e-12)
    assert d['cu_surface_M'][ix[-1]]>d['cu_surface_M'][ix[0]]
    assert r['checks']['diffusion']['relative_mass_error']<1e-10


def test_dt_refinement_reduces_error():
    common=Settings(duration_s=300,cells=48)
    values=[]
    for dt in [4,2,1,.5]:
        r=simulate(replace(common,dt_s=dt));values.append(r['diffusion']['charge_C'][-1])
    coarse=abs(values[0]-values[-1]);fine=abs(values[2]-values[-1])
    assert fine<coarse
    assert fine/values[-1]<1e-4


def test_mesh_refinement_reduces_error():
    values=[]
    for cells in [12,24,48,96]:
        r=simulate(Settings(duration_s=300,dt_s=1,cells=cells))
        values.append(r['diffusion']['charge_C'][-1])
    assert abs(values[2]-values[3])<abs(values[0]-values[3])


def test_fast_diffusion_approaches_mixed_model():
    r=simulate(Settings(diffusion_m2_s=1e-6,duration_s=300,dt_s=1))
    a=r['mixed']['charge_C'][-1];b=r['diffusion']['charge_C'][-1]
    assert abs(a-b)/a<1e-5


def test_random_walk_msd_agrees_statistically():
    d=brownian_experiment(.7e-9,seed=42)
    assert abs(d['msd_m2'][-1]/d['theory_m2'][-1]-1)<.06
    assert d['trajectories_m'].shape==(12,301)
    np.testing.assert_array_equal(d['trajectories_m'][:,0],0)


def test_random_walk_reproducible():
    a=brownian_experiment(.7e-9,seed=7,particles=100,steps=30)
    b=brownian_experiment(.7e-9,seed=7,particles=100,steps=30)
    np.testing.assert_array_equal(a['trajectories_m'],b['trajectories_m'])


def test_serialized_replay_and_exports(default):
    p=payload(default,max_frames=80)
    assert len(p['diffusion']['time_s'])==80
    assert p['diffusion']['time_s'][-1]==default['actual_duration_s']
    html=player_html(default)
    assert '__PAYLOAD__' not in html and 'requestAnimationFrame' in html
    df=timeseries_frame(default)
    assert len(df)==2*len(default['mixed']['time_s'])
    pf=profile_frame(default)
    assert pf.shape[1]==Settings().cells+2


def test_payload_preserves_switch_boundaries():
    r=simulate(Settings(protocol='rest',duration_s=600))
    p=payload(r,max_frames=12)
    t=np.asarray(p['diffusion']['time_s'])
    assert 210 in t and 390 in t


@pytest.mark.parametrize("temp", [5.0, 25.0, 45.0])
def test_nernst_temperature_coefficient(temp):
    from physics import RGAS, F, nernst_terms
    actual = nernst(.01, .1, temp)-nernst(.1, .1, temp)
    expected = -RGAS*(temp+273.15)/(2*F)*np.log(10)
    assert actual == pytest.approx(expected, abs=1e-13)
    terms = nernst_terms(.01,.1,temp)
    assert terms["concentration_shift_V"] == pytest.approx(expected)
    assert terms["E_V"] == pytest.approx(nernst(.01,.1,temp))


@pytest.mark.parametrize("temp", [5.0, 45.0])
def test_temperature_in_all_circuit_paths(temp):
    s = Settings(cu_M=.2,zn_M=.1,temperature_C=temp,duration_s=120)
    r = simulate(s)
    for model in ["mixed","diffusion"]:
        d = r[model]
        np.testing.assert_allclose(d["surface_nernst_V"],
            nernst(d["cu_surface_M"],d["zn_M"],temp),atol=1e-12)
        np.testing.assert_allclose(d["surface_nernst_V"],d["current_A"]*s.total_r,
                                  atol=1e-9)
        assert r["checks"][model]["valid"]


@pytest.mark.parametrize("cu,zn,temp", [
    (np.nan,.1,25),(.1,np.inf,25),(.1,.1,np.nan),(.1,.1,-273.15)])
def test_nonfinite_nernst_arguments_rejected(cu,zn,temp):
    with pytest.raises(ValueError):
        nernst(cu,zn,temp)


def test_capacity_and_ratio_are_distinct():
    from physics import capacity_mAh
    assert nernst(.2,.4)==pytest.approx(nernst(.1,.2))
    assert capacity_mAh(.2,.5)==pytest.approx(2*capacity_mAh(.1,.5))


def test_external_resistance_does_not_change_initial_equilibrium():
    a=simulate(Settings(resistance_ohm=1000,duration_s=30))
    b=simulate(Settings(resistance_ohm=2000,duration_s=30))
    assert a["mixed"]["bulk_nernst_V"][0]==b["mixed"]["bulk_nernst_V"][0]
    assert a["mixed"]["current_A"][0]>b["mixed"]["current_A"][0]


def test_voltage_decomposition(default):
    d=default["diffusion"]
    concentration=d["bulk_nernst_V"]-d["surface_nernst_V"]
    ohmic=d["current_A"]*default["settings"]["internal_ohm"]
    np.testing.assert_allclose(d["bulk_nernst_V"]-concentration-ohmic,
                               d["terminal_V"],atol=1e-9)


def test_snapshot_contains_live_parameters_without_external_dependencies():
    from visuals import snapshot_html
    h=snapshot_html(Settings(cu_M=.01,zn_M=.5,temperature_C=40,protocol="open"))
    assert '"temperature_C":40' in h and '"cu_M":0.01' in h
    assert "__PAYLOAD__" not in h
    assert "https://" not in h and "<script src" not in h


def test_stoichiometry_99_percent_is_not_zero_volts():
    assert nernst(.001,.199)==pytest.approx(1.031999,abs=1e-5)


def test_extreme_valid_ui_inputs_are_finite():
    for s in [
        Settings(cu_M=.001,zn_M=1,resistance_ohm=10,duration_s=60,dt_s=.5),
        Settings(cu_M=1,zn_M=.001,resistance_ohm=100000,temperature_C=45,duration_s=60),
        Settings(protocol="open",temperature_C=5,duration_s=60),
    ]:
        r=simulate(s)
        for model in ("mixed","diffusion"):
            assert np.all(np.isfinite(r[model]["surface_nernst_V"]))
            assert np.all(r[model]["cu_surface_M"]>0)
            assert r["checks"][model]["relative_mass_error"]<1e-8


def test_randomized_valid_ui_parameter_combinations():
    rng=np.random.default_rng(2468)
    for i in range(80):
        area=rng.uniform(.2,5); film=rng.uniform(.1,2)
        s=Settings(cu_M=10**rng.uniform(-3,0),zn_M=10**rng.uniform(-3,0),
            temperature_C=rng.uniform(5,45),resistance_ohm=10**rng.uniform(1,5),
            internal_ohm=rng.uniform(0,500),diffusion_m2_s=rng.uniform(.1,3)*1e-9,
            area_cm2=area,film_mm=film,volume_cu_ml=max(.5,2*area*film*.1),
            duration_s=60,dt_s=1,cells=24,
            protocol=["continuous","rest","open"][i%3])
        r=simulate(s)
        for model in ("mixed","diffusion"):
            assert r["checks"][model]["valid"], (i,model,r["checks"][model])
            assert np.all(np.isfinite(r[model]["surface_nernst_V"]))
