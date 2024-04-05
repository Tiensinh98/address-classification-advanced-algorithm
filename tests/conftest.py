import os
import json
import pytest
import sys
import pathlib

project_dir = pathlib.Path(__file__).resolve().resolve().parent.parent
sys.path.append(os.path.join(project_dir, 'scripts'))
import classifier


@pytest.fixture(autouse=True)
def solution():
    return classifier.Solution()


@pytest.fixture()
def public_test_cases():
    f = open('test_cases/public_test.json')
    data = json.load(f)
    return data
