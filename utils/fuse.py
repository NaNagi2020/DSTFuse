import torch


def fusion(en1, en2, alpha=0.5, strategy_type='addition'):
    f_0 = en1*alpha + en2*(1-alpha)
    return f_0