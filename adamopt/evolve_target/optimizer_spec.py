"""
Evolvable optimizer spec for NanoChat — AlphaEvolve target program.

alphaevolve mutates the values inside EVOLVE-BLOCK markers.
eval.sh runs this file, validates the spec, and (optionally) tests it on GPU.
"""

import json


def build_spec() -> str:
    """Return a MatrixOptimizerSpec as a JSON string."""
    # EVOLVE-BLOCK-START
    # Core optimizer parameters — tune these for better training dynamics.
    # momentum: [0, 1) — higher = more momentum, 0.95 is standard Muon.
    # nesterov: True = look-ahead momentum (usually better).
    # ns_steps: Newton-Schulz iterations for polar decomposition (5 is safe).
    # update_multiplier: scales the final update (1.0 = no change).
    # weight_decay: L2 regularization strength (0.0-0.5 typical).
    #
    # Available modes for trust_ratio: "none", "layerwise"
    # Available modes for clip: "none", "update_rms", "global_norm"
    # Available modes for decay: "none", "decoupled", "cautious"
    # Available modes for second_moment: "none", "factored_rms"
    #
    # stateful_control.enabled: enables adaptive behavior conditioned on
    # training signals (loss EMA, grad norms, etc.) via a gating network.
    # Gate coefficients control which signals the gate listens to.
    # Actuators define how aggressively the optimizer adapts when the gate
    # outputs "aggressive" vs "conservative" signals.
    spec = {
        "name": "evolved_muon_optimizer",
        "momentum": 0.95,
        "nesterov": True,
        "momentum_placement": "pre_orthogonal",
        "orthogonalization": "polar_express",
        "ns_steps": 5,
        "trust_ratio": {
            "mode": "none",
            "clamp_min": 0.25,
            "clamp_max": 4.0,
            "eps": 1e-8
        },
        "clip": {
            "mode": "none",
            "threshold": 1.0
        },
        "decay": {
            "mode": "cautious",
            "weight_decay": 0.2
        },
        "second_moment": {
            "mode": "factored_rms",
            "beta2": 0.95,
            "eps": 1e-10
        },
        "stateful_control": {
            "enabled": False,
            "ema_beta": 0.9,
            "loss_normalizer": 2.0,
            "improvement_normalizer": 0.05,
            "grad_norm_normalizer": 1.0,
            "update_ratio_normalizer": 0.05,
            "gate": {
                "coefficients": {
                    "loss_ema": 0.0,
                    "loss_improvement_ema": 0.0,
                    "grad_norm_ema": 0.0,
                    "update_ratio_ema": 0.0,
                    "grad_alignment_ema": 0.0,
                    "step_fraction": 0.0
                },
                "bias": 0.0,
                "sharpness": 1.0
            },
            "actuators": {
                "update_multiplier": {"aggressive": 1.15, "conservative": 0.85},
                "trust_ratio_mix": {"aggressive": 1.0, "conservative": 0.0},
                "clip_threshold": {"aggressive": 2.0, "conservative": 0.5},
                "beta2": {"aggressive": 0.90, "conservative": 0.99},
                "orthogonal_mix": {"aggressive": 1.0, "conservative": 0.35}
            }
        },
        "update_multiplier": 1.0,
        "lr_aspect_scale": True,
        "matrix_only": True,
        "metadata": {}
    }
    # EVOLVE-BLOCK-END
    return json.dumps(spec)


if __name__ == "__main__":
    # When run directly, print the spec JSON (used by eval.sh)
    print(build_spec())
