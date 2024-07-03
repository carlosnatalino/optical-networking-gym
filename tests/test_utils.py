import random

import optical_networking_gym.utils as ongu
import optical_networking_gym.topology as ongt

def test_span_ongu():
    s = ongu.Span(length=80, attenuation=0.2, noise_figure=4.5)
    assert s.attenuation_normalized != 0  # TODO: improve test
    assert s.noise_figure_normalized != 0  # TODO: improve test

    prelim = s.attenuation_normalized
    s.set_attenuation(attenuation=0.3)
    assert s.attenuation_normalized != prelim

    try:
        s.attenuation_normalized = 10
        raise ValueError("This attribute should not be writable.")
    except AttributeError:
        pass

def test_span_ongt():
    s = ongt.Span(length=80, attenuation=0.2, noise_figure=4.5)
    assert s.attenuation_normalized != 0  # TODO: improve test
    assert s.noise_figure_normalized != 0  # TODO: improve test

    prelim = s.attenuation_normalized
    s.set_attenuation(attenuation=0.3)
    assert s.attenuation_normalized != prelim

    try:
        s.attenuation_normalized = 10
        raise ValueError("This attribute should not be writable.")
    except AttributeError:
        pass

def test_link():
    spans = []
    total_length = 0
    for _ in range(5):
        t = random.random() * 80
        total_length += t
        s = ongt.Span(
            length=random.random() * 80,
            attenuation=random.random() * 0.2,
            noise_figure=random.random() * 5.5,
        )
        spans.append(s)
    l = ongt.Link(id=0, node1="t", node2="d", length=total_length, spans=tuple(spans))
    assert len(l.spans) == 5