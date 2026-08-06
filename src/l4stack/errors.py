class L4StackError(RuntimeError):
    """Base stack exception."""


class ConfigurationError(L4StackError):
    """Raised for invalid configuration."""


class CarlaConnectionError(L4StackError):
    """Raised when CARLA cannot be reached or is incompatible."""


class SensorTimeoutError(L4StackError):
    """Raised when required sensor data does not arrive for a frame."""
