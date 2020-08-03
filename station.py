SUMMER = [11, 12, 1, 2]
WINTER = [5, 6, 7, 8]
EQUINOX = [3, 4, 9, 10]


def is_summer(aux):
    a, b = aux
    if a.month in SUMMER:
        return True
    elif b.month in SUMMER:
        return True
    else:
        return False


def is_winter(aux):
    a, b = aux
    if a.month in WINTER:
        return True
    elif b.month in WINTER:
        return True
    else:
        return False


def is_equinox(aux):
    a, b = aux
    if a.month in EQUINOX:
        return True
    elif b.month in EQUINOX:
        return True
    else:
        return False
