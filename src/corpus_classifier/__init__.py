__version__ = "0.1.0"

def __getattr__(name):
    if name == "Classifier":
        from .model import Classifier
        return Classifier
    raise AttributeError(name)
