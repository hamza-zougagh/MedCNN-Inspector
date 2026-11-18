"""Trusted local model factories.

Researchers may edit this installed file to register their own nn.Module factory:

    from .workbench import register_factory
    register_factory("my_cnn", lambda classes: MyCNN(num_classes=classes))

Only code already present on the local filesystem is imported. The dashboard
never imports Python code from an upload or a model specification.
"""
