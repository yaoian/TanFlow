"""Registry mapping paper model variants to their implementations."""
from dataclasses import dataclass
from scripts import execution_flow, backbone_flow

ARMS = (
    "TanFlow-Full",
    "Terminal-Only",
    "Unconstrained-Flow",
    "Chord-Source",
    "Point-Source",
    "Direct-TCN",
    "Transformer",
)

# Compatibility aliases are accepted by the CLI, but are never public variants.
ALIASES = {
    "TanFlow": "TanFlow-Full",
    "TanFlow-Base": "TanFlow-Full",
    "Main": "TanFlow-Full",
    "Final-Only": "Terminal-Only",
    "No-Proj": "Unconstrained-Flow",
    "NoProjection": "Unconstrained-Flow",
    "Bi-Linear": "Chord-Source",
    "Bi-Noprior": "Chord-Source",
    "S0": "Point-Source",
    "Bi-DR": "Direct-TCN",
    "1-Step-Direct-TCN": "Direct-TCN",
}

# The implementation keys are private dispatch details, not CLI variant names.
_IMPLEMENTATIONS = {
    "TanFlow-Full": (execution_flow, "Main"),
    "Terminal-Only": (execution_flow, "Final-Only"),
    "Unconstrained-Flow": (execution_flow, "Unconstrained-Flow"),
    "Chord-Source": (execution_flow, "Bi-Noprior"),
    "Point-Source": (execution_flow, "Point-Source"),
    "Direct-TCN": (execution_flow, "Bi-DR"),
    "Transformer": (backbone_flow, "Transformer"),
}


@dataclass
class Arm:
    name: str
    module: object
    implementation: str

    @classmethod
    def resolve(cls, name):
        canonical = ALIASES.get(name, name)
        try:
            module, implementation = _IMPLEMENTATIONS[canonical]
        except KeyError:
            raise ValueError(f"Unknown arm: {name}; choose from {', '.join(ARMS)}") from None
        return cls(canonical, module, implementation)

    @property
    def family(self):
        return self.module.__name__.split(".")[-1]

    def model(self, seed, device="cpu"):
        if self.module == execution_flow:
            return self.module.new_model(
                seed, mode="direct_residual" if self.name == "Direct-TCN" else "flow", device=device)
        return self.module.new_model(seed, arm=self.implementation, device=device)

    def conditions(self, data, gap, device):
        return self.module.conditions(data, gap, device, self.implementation)

    def bind(self, model, conditions):
        return self.module.bind_field(model, conditions)

    def source(self, conditions, data, rows, step, calibrator):
        if self.name == "Direct-TCN":
            return conditions.v0
        return self.module.source(conditions, data, rows, step, calibrator, self.implementation)

    def target(self, conditions, data, rows, device):
        return data["v_gt"][rows].to(device)

    def objective(self, field, conditions, target, source, tau):
        return self.module.objective(
            field, conditions, target, source, tau, self.implementation)

    def predict(self, model, conditions, gap):
        return self.module.predict(
            self.bind(model, conditions), conditions, gap, self.implementation)


def gaps_for(arm, gap):
    return [gap]
