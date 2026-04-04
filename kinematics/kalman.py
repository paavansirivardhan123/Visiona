from typing import Optional
from core.config import Config

class KalmanFilter1D:
    def __init__(self, process_noise: float = None, measurement_noise: float = None):
        self._q = process_noise if process_noise is not None else Config.KF_PROCESS_NOISE
        self._r = measurement_noise if measurement_noise is not None else Config.KF_MEASUREMENT_NOISE
        self._x: Optional[float] = None
        self._v: float = 0.0
        self._p00 = 1.0
        self._p01 = 0.0
        self._p10 = 0.0
        self._p11 = 1.0
        self._initialized = False

    def update(self, measurement: Optional[float], dt: float = 1.0) -> float:
        if not self._initialized:
            if measurement is None:
                return 0.0
            self._x = measurement
            self._initialized = True
            return measurement
        x_pred = self._x + self._v * dt
        v_pred = self._v
        p00 = self._p00 + dt * (self._p10 + self._p01) + dt * dt * self._p11 + self._q
        p01 = self._p01 + dt * self._p11
        p10 = self._p10 + dt * self._p11
        p11 = self._p11 + self._q
        if measurement is None:
            self._x, self._v = x_pred, v_pred
            self._p00, self._p01, self._p10, self._p11 = p00, p01, p10, p11
            return self._x
        if abs(measurement - x_pred) > Config.DEPTH_SPIKE_THRESHOLD:
            self._x, self._v = x_pred, v_pred
            self._p00, self._p01, self._p10, self._p11 = p00, p01, p10, p11
            return self._x
        y  = measurement - x_pred
        s  = p00 + self._r
        k0 = p00 / s
        k1 = p10 / s
        self._x = x_pred + k0 * y
        self._v = v_pred + k1 * y
        self._p00 = (1 - k0) * p00
        self._p01 = (1 - k0) * p01
        self._p10 = p10 - k1 * p00
        self._p11 = p11 - k1 * p01
        return self._x

    @property
    def estimate(self) -> Optional[float]:
        return self._x if self._initialized else None
