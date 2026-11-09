"""Integer cent parsing for dollar strings such as '1.25'."""


def parse_cents(text: str) -> int:
    raw = text.strip()
    sign = -1 if raw.startswith("-") else 1
    body = raw[1:] if raw.startswith("-") else raw
    dollars, sep, frac = body.partition(".")
    if sep and not frac.isdigit():
        raise ValueError(f"bad money: {text}")
    if not dollars.isdigit() and dollars != "":
        raise ValueError(f"bad money: {text}")
    frac = (frac + "00")[:2]
    return sign * (int(dollars or "0") * 100 + int(frac))
