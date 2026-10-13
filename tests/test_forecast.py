"""A forecast never uses a model trained after its time, or data its inputs could not have had."""

import pandas as pd
import pytest

from airalert.forecast import check_forecast_time

CUTOFF, END = pd.Timestamp("2026-10-12 12:00", tz="UTC"), pd.Timestamp("2026-10-12 21:50", tz="UTC")


def test_forecast_at_or_after_the_cutoff_within_the_data_is_allowed():
    check_forecast_time(CUTOFF, CUTOFF, END)
    check_forecast_time(pd.Timestamp("2026-10-12 21:00", tz="UTC"), CUTOFF, END)


@pytest.mark.parametrize("at", ["2026-10-12 11:00",  # before the model's cutoff
                                "2026-10-12 22:00",  # after the data ends
                                "2026-10-12 13:30"])  # not a whole hour
def test_other_forecast_times_are_refused(at):
    with pytest.raises(SystemExit):
        check_forecast_time(pd.Timestamp(at, tz="UTC"), CUTOFF, END)
