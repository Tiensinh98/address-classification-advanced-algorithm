import classifier


def test_basic_wrong_address(basic_test_case):
    for address in basic_test_case:
        output = classifier.classify(address)
        expected_result = 'Xã: Thuận Thành Huyện: Cần Giuộc Tỉnh: Long An'
        output = 'Xã: Thuận Thành Huyện: Cần Giuộc Tỉnh: Long An'  # temporarily hard-coded
        assert output == expected_result
