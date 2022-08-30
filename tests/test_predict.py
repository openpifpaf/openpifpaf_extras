import json
import os
import subprocess
import sys


PYTHON = 'python3' if sys.platform != 'win32' else 'python'


def test_predict(tmpdir):
    """Test predict cli with a swin_s backbone."""

    cmd = [
        PYTHON, '-m', 'openpifpaf.predict',
        '--checkpoint=swin_s',
        '--batch-size=1',
        '--loader-workers=0',
        '--json-output', str(tmpdir),
        '--long-edge=321',
        'docs/coco/000000081988.jpg',
    ]
    subprocess.run(cmd, check=True)

    out_file = os.path.join(tmpdir, '000000081988.jpg.predictions.json')
    assert os.path.exists(out_file)

    with open(out_file, 'r') as f:
        predictions = json.load(f)

    assert len(predictions) == 5
