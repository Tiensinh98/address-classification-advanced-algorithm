import os

import pytest
import sys
import pathlib

project_dir = pathlib.Path(__file__).resolve().resolve().parent.parent
sys.path.append(os.path.join(project_dir, 'scripts'))


@pytest.fixture()
def basic_test_case():
    addresses = []
    with open('./test_cases/basic_wrong_case.txt', 'r') as f:
        for line in f.readlines():
            addresses.append(line.strip('\n'))
    return addresses
