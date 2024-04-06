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
    test_file_path = str(project_dir) + '/tests/test_cases/public_test.json'
    f = open(test_file_path)
    data = json.load(f)
    return data
