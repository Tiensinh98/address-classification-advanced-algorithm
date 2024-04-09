import pytest
import time
import json
import numpy as np

import classifier as clf


@pytest.fixture(autouse=True)
def solution():
    return clf.Solution()


@pytest.mark.parametrize('case_file_path', [
    'test_cases/basic_cases.json',
    'test_cases/confusing_number_hcm_cases.json',
    'test_cases/inconsistent_information.json',
    'test_cases/other_cases.json'
])
def test_full_address_cases(solution, case_file_path):
    test_cases = json.load(open(case_file_path, encoding='utf-8'))
    count = 0
    timer = []
    # print('Reading test case: ', case_file_path, '\n')
    for test_case in test_cases:
        start = time.time()
        output = solution.process(test_case['text'])
        time_elapsed = time.time() - start
        timer.append(time_elapsed)
        try:
            assert time_elapsed < 0.2
            expected_result = test_case['result']
            assert expected_result['province'] == output['province']
            assert expected_result['district'] == output['district']
            assert expected_result['ward'] == output['ward']
            count += 1
        except AssertionError:
            # print('Failed!')
            # print('Input Address', test_case['text'])
            # print('Expected result', test_case['result'])
            # print('Actual', output)
            # print('Time Elapsed', time_elapsed)
            continue

    print(f'Passed: {count} cases / {len(test_cases)}')
    assert np.average(timer) < 4. / 100, 'Average time exceeds the allowed amount of time.'
