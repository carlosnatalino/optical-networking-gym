import optical_networking_gym.utils as ongu

def test_span():
    s = ongu.Span(length=80, attenuation=0.2, noise_figure=4.5)
    assert s.attenuation_normalized != 0  # TODO: improve test
    assert s.noise_figure_normalized != 0  # TODO: improve test

    prelim = s.attenuation_normalized
    s.set_attenuation(attenuation=0.3)
    assert s.attenuation_normalized != prelim

    s.attenuation_normalized = 10
