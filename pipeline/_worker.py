#!/usr/bin/env python3
"""
_worker() function implementation from Table 2.1
Worker function for Marabou verification in isolated processes
"""

import os
import sys
import io
import numpy as np
from maraboupy import Marabou


def _worker(model, flat_x, c_true, c_adv, epsilon, timeout, return_dict):
    """Worker function for Marabou verification.
    
    Runs verification in an isolated process to reclaim memory.
    
    Args:
        model: ONNX model path
        flat_x: Flattened input vector
        c_true: True class
        c_adv: Adversarial class
        epsilon: Perturbation bound
        timeout: Timeout in seconds
        return_dict: Dictionary to store results
    """
    sys.stdout = io.StringIO()  # silence marabou prints
    try:
        net = Marabou.read_onnx(model)
        ivars = net.inputVars[0].flatten()
        ovars = net.outputVars[0][0]

        for v, pv in zip(ivars, flat_x):
            net.setLowerBound(v, max(0.0, pv - epsilon))
            net.setUpperBound(v, min(1.0, pv + epsilon))

        # want: target > true class ==> ovars[c_true] - ovars[c_adv] <= 0
        net.addInequality([ovars[c_adv], ovars[c_true]], [-1, 1], 0)

        opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=timeout)
        result = net.solve(options=opts)
        return_dict["status"] = result[0]

        if result[0] == "sat":
            vals = result[1]
            return_dict["adv_flat"] = np.array([vals[v] for v in ivars])

    except MemoryError:
        return_dict["status"] = "oom"
    except Exception as e:
        msg = str(e)
        if "bad_alloc" in msg or "MemoryError" in msg:
            return_dict["status"] = "oom"
        else:
            return_dict["status"] = f"error: {e}"


if __name__ == "__main__":
    print("_worker() function - Marabou verification worker")
    print("Usage: Import and call _worker() in multiprocessing context")
