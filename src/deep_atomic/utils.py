def detect_broadcast_dim(from_, to_):
    detected = []
    for i in range(-1, -len(to_) - 1, -1):
        if i < -len(from_):
            detected.append((i + len(to_), False))
        elif from_[i] == 1 and to_[i] != 1:
            detected.append((i + len(to_), True))  # True tp keepdims
        elif from_[i] != to_[i]:
            raise ValueError(f"{from_} cannot be broadcasted to target shape {to_}")
    return detected


def is_broadcastable(from_, to_):
    res = True
    for i in range(-1, -len(to_) - 1, -1):
        res = res and (from_[i] == to_[i] or from_[i] == 1 and to_[i] != 1)
    return res
