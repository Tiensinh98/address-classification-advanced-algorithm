import time


def test_public_test(solution, public_test_cases):
    for test_case in public_test_cases:
        start = time.time()
        output = solution.process(test_case['text'])
        finish = time.time()
        assert finish - start < 0.01
        expected_result = test_case['result']
        assert expected_result['province'] == output['province']
        assert expected_result['district'] == output['district']
        assert expected_result['ward'] == output['ward']
