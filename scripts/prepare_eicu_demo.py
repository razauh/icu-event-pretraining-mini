"""Prepare eICU demo tables into patient-level event streams."""

from icu_pretrain.data.eicu_event_builder import build_event_streams


if __name__ == "__main__":
    build_event_streams()

