"""Statistical-correctness acceptance tests for the direction-guiding mixture.

The expected values here are derived independently (closed-form cap-light
integrals); mean confidence intervals come from independent trials, and the
negative examples (zero-support direction, injected wrong densities) must
fail loudly rather than be reported as convergence.
"""

import json
import math
import os

import numpy as np
import pytest

from dirguide import acceptance as acc
from dirguide.acceptance import (
    ALBEDO,
    LIGHT_A,
    LIGHT_B,
    NORMAL_UP,
    DirectionalFixture,
    ci_covers,
    mean_ci,
    run_mid_light,
)

# Shared quick budgets so the suite stays fast but statistically meaningful.
MID_KW = dict(train_rounds=2, train_n=10000, eval_intervals=5,
              eval_n=10000, continued_train_n=2000)
BV_KW = dict(n_trials=6, train_n=4000, eval_n=8000)


# ------------------------------------------------------- analytic reference
def test_analytic_reference_is_independently_derived():
    """Closed form: L_o = rho * L_e * sin^2(alpha) * (n.w0) per channel."""
    fix = DirectionalFixture()
    fix.add_light(LIGHT_A)
    fix.add_light(LIGHT_B)
    expected = (
        ALBEDO * LIGHT_A.le * (LIGHT_A.sin2_alpha) * (NORMAL_UP @ LIGHT_A.w0)
        + ALBEDO * LIGHT_B.le * (LIGHT_B.sin2_alpha) * (NORMAL_UP @ LIGHT_B.w0)
    )
    assert np.allclose(fix.analytic_mean(), expected)
    # Caps must be fully above the horizon for the closed form to hold.
    assert LIGHT_A.fully_above_horizon(NORMAL_UP)
    assert LIGHT_B.fully_above_horizon(NORMAL_UP)


def test_mixture_pdf_normalizes_on_analytic_scene():
    res = acc.mixture_normalization_check(quadrature_n=100000)
    assert res["ok"], res
    ci = res["mean_ci"]
    assert ci["lo"] <= 1.0 <= ci["hi"]
    # BSDF component alone integrates to 1 over the hemisphere.
    assert abs(res["bsdf_integral"] - 1.0) < 0.02


def test_sampling_distribution_matches_exact_mixture():
    res = acc.sampling_distribution_check(n=150000)
    assert abs(res["expected_mass_total"] - 1.0) < 1e-9
    lo, hi = res["window"]
    assert lo <= res["chi2"] <= hi, res
    assert res["ok"]


def test_image_means_match_analytic_image():
    res = acc.image_sweep_check(width=20, height=12, n_per_pixel=1500,
                                train_n=8000)
    assert res["frac_pixels_within_5sigma"] >= 0.99, res["max_abs_z"]
    ci = res["mean_image_ci"]
    assert ci["lo"] <= ci["reference"] <= ci["hi"]
    assert res["ok"]


# ------------------------------------------------------- mid-sampling light
def test_mid_light_energy_checked_over_independent_intervals():
    """With a small exploration floor the new light's energy is recovered."""
    res = run_mid_light(**MID_KW)
    assert res["support"]["pdf_min"] > 0.0
    assert not res["zero_support"]
    assert res["verdict"] == "accept"
    assert res["converged"]
    # Multiple independent intervals back the grand mean CI.
    assert len(res["intervals"]) == MID_KW["eval_intervals"]
    assert ci_covers(res["grand_ci"], res["reference_luminance"])
    assert abs(res["missing_energy_fraction"]) < 0.05
    # Old table and its generating samples were accepted by the sampler.
    assert res["provenance_ok"]
    assert res["train_record_continues"]
    assert res["eval_records_separate"]


def test_zero_support_requires_rebuild_not_convergence():
    """Exploration off (eta=0, lam=1): the new light is completely missed.

    The interval means are stable, yet the fixture must demand a proposal
    rebuild instead of accepting the stable underestimate as convergence.
    """
    res = run_mid_light(eta=0.0, lam=1.0, **MID_KW)
    assert res["zero_support"]
    assert res["support"]["pdf_min"] == 0.0
    assert res["verdict"] == "rebuild_required"
    assert not res["converged"]
    assert res["stable_across_intervals"]
    assert res["stable_but_underestimated"]
    assert res["missing_energy_fraction"] > 0.15
    # The missed energy matches the independently derived new-light share.
    expected_missing = (
        res["new_light_reference"] / res["reference_luminance"])
    assert abs(res["missing_energy_fraction"] - expected_missing) < 0.05


def test_first_anomalous_component_saved_on_density_misuse(tmp_path):
    res = run_mid_light(eta=0.0, lam=1.0, artifact_dir=str(tmp_path), **MID_KW)
    comp = res["first_anomalous_component"]
    assert comp is not None
    assert comp["light"] == "B_new"
    assert comp["interval"] == 0
    assert comp["observed_luminance"] == pytest.approx(0.0, abs=1e-12)
    assert comp["expected_luminance"] > 0.0
    path = res["anomaly_artifact"]
    assert os.path.exists(path)
    payload = json.load(open(path))
    assert payload["schema"] == acc.ANOMALY_SCHEMA
    assert payload["first_anomalous_component"]["light"] == "B_new"
    assert payload["verdict"] == "rebuild_required"


def test_old_table_regenerated_by_its_samples():
    """The samples that generated the old table rebuild it exactly."""
    fix = DirectionalFixture(seed=11)
    fix.add_light(LIGHT_A)
    fix.add_light(LIGHT_B, active=False)
    fix.train_round(4000, rounds=2)
    exported = fix.export_table_with_samples()
    other = DirectionalFixture(seed=12)
    assert other.continue_with(exported)
    assert np.allclose(other.guide.W, fix.guide.W)
    # The learning record carries the generating samples forward.
    assert len(other.train_deposits) == len(fix.train_deposits)


# -------------------------------------------------- bias / variance trials
def test_bias_and_variance_asserted_separately():
    res = acc.bias_variance_study(**BV_KW)
    assert res["bias_ok"], res["mean_ci"]
    assert res["variance_ok"], res["variance_ratio"]
    assert ci_covers(res["mean_ci"], res["reference_luminance"])
    assert res["ok"]


def test_disabling_exploration_exposes_missing_energy():
    """Deliberately closing the exploration branch must show the shortfall."""
    res = run_mid_light(eta=0.0, lam=1.0, **MID_KW)
    assert res["missing_energy_fraction"] > 0.15
    with_floor = run_mid_light(**MID_KW)
    assert abs(with_floor["missing_energy_fraction"]) < 0.05


def test_training_size_does_not_systematically_darken():
    res = acc.training_size_invariance(
        train_sizes=(0, 1500, 12000), n_trials=5, eval_n=6000)
    assert res["all_cis_cover"]
    assert res["max_relative_deviation"] < 0.05
    assert res["ok"]
    # Sparse training is the positive example: even the tiniest training
    # set keeps the estimator unbiased.
    sparse = res["studies"][1]
    assert sparse["train_n"] == 1500
    assert ci_covers(sparse["mean_ci"], res["reference_luminance"])


def test_injected_wrong_density_is_caught():
    clean = acc.bias_variance_study(**BV_KW)
    assert clean["bias_ok"]
    stale = acc.bias_variance_study(corrupt="stale_pdf", **BV_KW)
    assert not stale["bias_ok"], "stale old density must fail the mean CI"
    scaled = acc.bias_variance_study(corrupt="scale_pdf", **BV_KW)
    assert not scaled["bias_ok"], "scaled density must fail the mean CI"
    # The scaled-down estimate is systematically darker, as expected.
    assert scaled["mean_ci"]["mean"] < clean["reference_luminance"]


def test_eval_data_separate_from_learning_records():
    fix = DirectionalFixture(seed=23)
    fix.add_light(LIGHT_A)
    fix.add_light(LIGHT_B)
    fix.train_round(3000)
    n_dep_after_train = fix.guide.n_deposits
    n_records = len(fix.train_deposits)
    rec = fix.eval_interval(5000, label="eval0")
    # Evaluation draws never enter the learning record.
    assert fix.guide.n_deposits == n_dep_after_train
    assert len(fix.train_deposits) == n_records
    assert len(fix.eval_records) == 1
    assert fix.eval_records[0] is rec
    assert not rec["train"]
    # The eval used the table version frozen at eval time.
    assert rec["used_version"] == fix.guide.version


def test_unguided_estimator_cross_check():
    res = acc.unguided_cross_check(n_rounds=5, train_n=4000, eval_n=8000)
    assert res["agrees_with_unguided"]
    assert ci_covers(res["difference_ci"], 0.0)
    # Guiding must actually reduce variance on this spiky light field.
    assert res["guided_over_unguided_var"] < 1.0
    assert res["ok"]


# ------------------------------------------------------- evidence integrity
def test_mean_ci_recomputed_from_execution_evidence():
    """Reported CIs must equal a recomputation from the recorded trials."""
    res = acc.bias_variance_study(**BV_KW)
    ci = res["mean_ci"]
    evidence = np.asarray(ci["evidence"])
    assert evidence.size == ci["n"] == BV_KW["n_trials"]
    recomputed = mean_ci(evidence)
    assert recomputed["mean"] == pytest.approx(ci["mean"], rel=1e-12)
    assert recomputed["lo"] == pytest.approx(ci["lo"], rel=1e-12)
    assert recomputed["hi"] == pytest.approx(ci["hi"], rel=1e-12)
    # Mid-light intervals also carry their raw means as evidence.
    mid = run_mid_light(**MID_KW)
    interval_means = [it["lum_mean"] for it in mid["intervals"]]
    assert np.allclose(mid["grand_ci"]["evidence"], interval_means)


def test_acceptance_suite_runs_and_has_no_false_reports(tmp_path):
    report = acc.run_quick_suite(str(tmp_path), quick=True)
    assert report["all_ok"]
    names = {c["name"] for c in report["checks"]}
    assert "zero_support_requires_rebuild" in names
    assert "injected_stale_density_caught" in names
    # Every mean-CI entry carries the evidence it was computed from.
    for check in report["checks"]:
        ci = check.get("mean_ci") or check.get("grand_ci")
        if ci and "evidence" in check.get("grand_ci", {}):
            assert len(ci["evidence"]) == ci["n"]
    # The saved anomaly artifact documents the misused old density.
    artifact = os.path.join(str(tmp_path), "first_anomalous_component.json")
    assert os.path.exists(artifact)
    payload = json.load(open(artifact))
    assert payload["misuse"] == "stale_density_after_mid_sampling_light_on"


# ------------------------------------------------------- kernel integration
def test_kernel_regression_case_drives_live_session():
    """The regression-example button must run the real guiding session."""
    from dirguide.kernel import Kernel

    k = Kernel()
    out = k.command({"cmd": "load_regression_case",
                     "case": "stale", "train_spp": 2})
    st = out["result"] if "result" in out else out
    # Moving the light restarts the image estimate (old pixels describe the
    # old lighting) while the stale policy keeps the trained table.
    assert st["spp"] == 0
    assert st["light_generation"] == 1
    assert st["strategy"] == "stale"
    # Real training happened: the table holds learned weight (no canned
    # convergence data), and the guide version advanced with each spp.
    assert st["n_deposits"] > 0
    assert st["guide_version"] > 0
    # Stepping further produces real estimates after the case was loaded.
    out2 = k.command({"cmd": "step", "spp": 1})
    st2 = out2["result"] if "result" in out2 else out2
    assert st2["spp"] == 1
    assert len(st2["history"]) == 1
    assert st2["history"][0]["mean_luma"] > 0


def test_kernel_run_acceptance_persists_report(tmp_path, monkeypatch):
    from dirguide import kernel as kernel_mod

    monkeypatch.setattr(kernel_mod, "ACCEPTANCE_DIR", str(tmp_path))
    monkeypatch.setattr(kernel_mod, "METRICS_DIR", str(tmp_path))
    k = kernel_mod.Kernel()
    out = k.command({"cmd": "run_acceptance"})
    st = out["result"] if "result" in out else out
    summary = st["acceptance"]
    assert summary["n_ok"] == summary["n_checks"]
    saved = os.path.join(str(tmp_path), "acceptance.json")
    assert os.path.exists(saved)
    report = json.load(open(saved))
    assert report["all_ok"]
    assert k.acceptance_report()["available"]
