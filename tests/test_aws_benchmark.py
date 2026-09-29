"""Output isolation and failure provenance for the AWS execution wrapper."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SPEC = importlib.util.spec_from_file_location('aws_benchmark', Path(__file__).parents[1] / 'experiments/aws_benchmark.py')
AWS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AWS)


@pytest.fixture
def invocation(tmp_path, monkeypatch):
    directory = tmp_path / 'experiments'
    directory.mkdir()
    script = directory / 'fixture.py'
    monkeypatch.setattr(AWS, '__file__', str(directory / 'aws_benchmark.py'))
    monkeypatch.setattr(AWS.subprocess, 'check_output', lambda *a, **kw: 'fixture-sha\n')
    monkeypatch.setattr(sys, 'path', list(sys.path))
    argv = ['aws_benchmark.py', '--run-tag', 'aws_fixture', '--script', 'experiments/fixture.py']
    monkeypatch.setattr(sys, 'argv', argv)
    return script, directory / 'results/aws_20260929/aws_fixture', argv


def test_preserves_existing_output_and_refuses_reuse(invocation, monkeypatch):
    script, out, argv = invocation
    original = script.parent / 'original'
    original.mkdir()
    (original / 'result.json').write_text('{"sentinel": true}')
    script.write_text('import json\nfrom pathlib import Path\n'
                      f'OUT = {str(original)!r}\n'
                      'Path(OUT, "result.json").write_text(json.dumps({"test_bpc": 2.5}))\n')
    AWS.main()
    assert json.loads((original / 'result.json').read_text()) == {'sentinel': True}
    assert json.loads((out / 'result.json').read_text()) == {'test_bpc': 2.5}
    assert json.loads((out / 'provenance.json').read_text())['status'] == 'completed'
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(FileExistsError):
        AWS.main()
    assert json.loads((out / 'result.json').read_text()) == {'test_bpc': 2.5}


@pytest.mark.parametrize('body,exception', [
    ('raise RuntimeError("fixture failure")', RuntimeError),
    ('Path(OUT, "result.json").write_text(json.dumps({"test_bpc": float("nan")}))', ValueError),
])
def test_failures_are_not_reported_as_completed(invocation, body, exception):
    script, out, _ = invocation
    script.write_text('import json\nfrom pathlib import Path\nOUT = "unused"\n' + body + '\n')
    with pytest.raises(exception):
        AWS.main()
    provenance = json.loads((out / 'provenance.json').read_text())
    assert provenance['status'] == 'failed'
    assert provenance['error']
    assert provenance['peak_rss_kb'] > 0
