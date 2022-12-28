import pytest


class TestDummy:
    input_expected_data = [
        (1, 1),
        (2, 4),
        (3, 9),
    ]

    input_expected_cubed_data = [
        (1, 1),
        (2, 8),
        (3, 27),
    ]

    @pytest.mark.parametrize("test_input, expected", input_expected_data)
    def test_square(self, test_input, expected):
        assert test_input * test_input == expected

    @pytest.mark.parametrize("test_input, expected", input_expected_cubed_data)
    def test_cube(self, test_input, expected):
        assert test_input * test_input * test_input == expected