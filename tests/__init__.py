"""
Unit test package for drillcore_transformations.
"""
from pathlib import Path

import numpy as np
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import dictionaries, floats, functions, text

alpha_strategy = floats(min_value=-90, max_value=90)
beta_strategy = floats(min_value=-360, max_value=360)
trend_strategy = floats(min_value=0, max_value=90)
plunge_strategy = floats(min_value=-90, max_value=90)
gamma_strategy = floats(min_value=-360, max_value=360)
vector_strategy = arrays(np.float64, shape=3, elements=floats(-1, 1))
amount_strategy = floats(0, np.pi * 2)
dip_strategy = floats(min_value=0, max_value=90)
dir_strategy = floats(min_value=0, max_value=360)
function_strategy = functions()
text_strategy = text()
dict_strategy = dictionaries(text_strategy, text_strategy)

sample_csv = (
    Path(__file__).parent.parent / Path("sample_data/Logging_sheet.csv")
).absolute()
sample_csv_result = (
    Path(__file__).parent.parent / Path("sample_data/Logging_sheet_transformed.csv")
).absolute()
