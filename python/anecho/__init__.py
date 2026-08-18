"""anecho — the Anecho voice-focus runtime.

    from anecho import Model, Processor, UsageReporter

    model = Model.from_file("your-model.anecho")     # from your dashboard
    proc = Processor(model, license_key)
    enhanced = proc.process(block)                   # float32 @ 16 kHz

Audio never leaves your process; the UsageReporter sends durations only.
"""
from .processor import Model, Processor, ProcessorConfig, ProcessorParameter, Vad, PRODUCT
from .telemetry import UsageReporter
from .license import verify_license
from . import errors

__all__ = ["Model", "Processor", "ProcessorConfig", "ProcessorParameter", "Vad",
           "PRODUCT", "UsageReporter", "verify_license", "errors"]
