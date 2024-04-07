import pytest
import time
import json
import numpy as np

import classifier as clf


@pytest.fixture(autouse=True)
def solution():
    return clf.Solution()


@pytest.fixture()
def full_address_cases():
    f = open('cases/full_address_test_cases.json')
    data = json.load(f)
    return data


def test_full_address_cases(solution, full_address_cases):
    timer = []
    for test_case in full_address_cases:
        start = time.time()
        output = solution.process(test_case['text'])
        time_elapsed = time.time() - start
        timer.append(time_elapsed)
        assert time_elapsed < 0.01
        expected_result = test_case['result']
        assert expected_result['province'] == output['province']
        assert expected_result['district'] == output['district']
        assert expected_result['ward'] == output['ward']
    print(timer)
    print('Average: ', np.average(timer))
    assert np.average(timer) < 4. / 100
