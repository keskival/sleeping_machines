"""Numerical inclusion and local-teacher experiment for THEORY §219."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from sleeping_machines.temporal_modes import WinningTemporalModes, mode_flow


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    a = p.parse_args()
    out = Path("experiments/results/e139")/(a.tag+".json")
    out.parent.mkdir(exist_ok=True)
    if Path(a.tag).name != a.tag or out.exists():
        raise ValueError("Unique output required")
    torch.set_num_threads(1)
    torch.manual_seed(139)
    model = WinningTemporalModes(7, 5, 4, 3).double()
    with torch.no_grad():
        # All options implement the same selected SSM; delays remain a race.
        model.raw_rate.copy_(torch.linspace(-3., 1., 5).repeat(3, 1))
        model.frequency.copy_(torch.linspace(-7., 11., 5).repeat(3, 1))
        model.input.copy_(model.input[:1].clone().expand_as(model.input))
        model.output.copy_(model.output[:1].clone().expand_as(model.output))
        model.direct.copy_(model.direct[:1].clone().expand_as(model.direct))
        model.route[:, 0] = 1.
    times = torch.sort(torch.rand(19, dtype=torch.float64))[0]
    sources = torch.randint(0, 7, (len(times),))
    state = torch.zeros(5, 2, dtype=torch.float64)
    reference = torch.zeros(5, dtype=torch.complex128)
    last = times.new_tensor(0.)
    state_errors, output_errors, emission_offsets = [], [], []
    rates, frequency = model.rates()[0], model.frequency[0]
    for source, t in zip(sources, times):
        state, value, emitted, details = model.event(source, t, last, state, counterfactuals=True)
        reference = torch.exp(torch.complex(-rates, frequency)*(t-last))*reference+torch.view_as_complex(model.input[0, source])
        expected = model.output[0]@torch.view_as_real(reference).flatten()+model.direct[0, source]
        state_errors.append(float((state-torch.view_as_real(reference)).detach().abs().max()))
        output_errors.append(float((value-expected).detach().abs().max()))
        emission_offsets.append(float((emitted-t).detach()))
        assert int(details["winner"]) == 0 and details["emitted_vectors"] == 1
        assert torch.equal(details["candidate_states"][0], details["candidate_states"][1])
        last = t
    assert max(state_errors+output_errors) < 1e-12
    # Semigroup and exact decay/frequency teachers of one selected mode step.
    s = torch.randn(5, 2, dtype=torch.float64)
    rate = model.rates()[0].detach().requires_grad_()
    omega = model.frequency[0].detach().requires_grad_()
    dt = torch.tensor(.137, dtype=torch.float64)
    z = mode_flow(s, dt, rate, omega)
    teacher = torch.randn_like(z)
    loss = (z*teacher).sum()
    dr, dw = torch.autograd.grad(loss, (rate, omega))
    expected_r = -dt*(teacher*z).sum(-1)
    generator_z = torch.stack((-z[:, 1], z[:, 0]), -1)
    expected_w = dt*(teacher*generator_z).sum(-1)
    rate_error = float((dr-expected_r).abs().max())
    frequency_error = float((dw-expected_w).abs().max())
    combined = mode_flow(s, dt*2, rate, omega)
    split = mode_flow(z, dt, rate, omega)
    semigroup_error = float((combined-split).detach().abs().max())
    assert max(rate_error, frequency_error, semigroup_error) < 1e-12
    result = {"status": "completed", "events": len(times), "modes": model.modes,
        "max_complex_ssm_state_error": max(state_errors),
        "max_complex_ssm_output_error": max(output_errors),
        "constant_winning_delay_s": emission_offsets[0],
        "delay_offset_spread_s": max(emission_offsets)-min(emission_offsets),
        "semigroup_error": semigroup_error, "decay_teacher_error": rate_error,
        "frequency_teacher_error": frequency_error,
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path("sleeping_machines/temporal_modes.py"))},
        "scope": "Exact selected affine SSM operator and local teachers only; not a six-layer nonlinear reference classifier, learned route policy or SHD accuracy result"}
    out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
