from till.offers import half_off_offer, offer_label, offer_price


def test_offers() -> None:
    assert offer_price(8) == 8
    assert offer_label("  mug ") == "mug"
    assert half_off_offer(10) == 5
    assert half_off_offer(11) == 5
