from till.round import round_div, service_fee, shares_match, split_bill


def test_half_up_and_fee() -> None:
    assert round_div(5, 2) == 3
    assert split_bill(5, 2) == 3
    assert service_fee(5, 2) == 3
    assert service_fee(1, 2) == 1
    assert shares_match(5, 2, 3)
