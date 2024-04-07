import json
import pytest
import scripts.classifier as clf


@pytest.fixture(autouse=True)
def solution():
    return clf.Solution()


@pytest.fixture()
def full_address_cases():
    f = open('test_cases/full_address_test_cases.json')
    data = json.load(f)
    return data
