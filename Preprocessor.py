from abc import ABC
from typing import List
import numpy as np
from custom_types import Signal
from scipy.signal import butter, lfilter, iirnotch


class Preprocessor(ABC, FSBase):
    """
    Abstract base class for signal preprocessors.
    Provides a callable interface to apply preprocessing on a list of signals.
    """

    def __call__(self, signals: List[Signal]) -> List[Signal]:
        # Apply preprocessing to each signal in the list
        return [self.preprocess(signal) for signal in signals]

    def preprocess(self, signal: Signal) -> Signal:
        # Abstract method for preprocessing a single signal
        pass


class SimplePreprocessor(Preprocessor):
    """
    Simple preprocessor that returns the signal unchanged.
    """

    def preprocess(self, signal: Signal) -> Signal:
        return signal


class BasicPreprocessor(Preprocessor):
    """
    Basic preprocessor that applies a bandpass filter followed by a notch filter.
    Useful for removing baseline wander and power line interference.
    """

    def preprocess(self, signal: Signal) -> Signal:
        # Apply bandpass filter between 0.5 and 30 Hz
        signal = self.bandpass_filter(signal, 0.5, 30.0)
        # Apply notch filter to remove 50 Hz interference
        signal = self.notch_filter(signal, 50)
        return signal

    def bandpass_filter(self, signal: Signal, lower_fr: float, higher_fr: float) -> Signal:
        # Design and apply a 4th-order Butterworth bandpass filter
        w_low = lower_fr * 2 / self.fs
        w_high = higher_fr * 2 / self.fs
        b, a = butter(N=4, Wn=[w_low, w_high], btype='band')
        return lfilter(b, a, signal)

    def notch_filter(self, signal: Signal, freq: float, quality_factor: float = 30.0) -> Signal:
        # Design and apply a notch filter to remove specified frequency
        w0 = freq * 2 / self.fs
        b, a = iirnotch(w0, quality_factor)
        return lfilter(b, a, signal)
    

class SARModelPreprocessor(Preprocessor):
    """
    SAR model-specific preprocessor that applies a highpass filter and a notch filter.
    """

    def preprocess(self, signal: Signal) -> Signal:
        # Apply highpass filter with a cutoff at 2 Hz
        signal = self.highpass_filter(signal, 2.0)
        # Apply notch filter to remove 50 Hz interference
        signal = self.notch_filter(signal, 50)
        return signal

    def highpass_filter(self, signal: Signal, cutoff: float) -> Signal:
        # Design and apply a 1st-order Butterworth highpass filter
        w_cut = cutoff * 2 / self.fs
        b, a = butter(N=1, Wn=w_cut, btype='high', analog=False)
        return lfilter(b, a, signal)

    def notch_filter(self, signal: Signal, freq: float, quality_factor: float = 30.0) -> Signal:
        # Design and apply a notch filter to remove specified frequency
        w0 = freq * 2 / self.fs
        b, a = iirnotch(w0, quality_factor)
        return lfilter(b, a, signal)
