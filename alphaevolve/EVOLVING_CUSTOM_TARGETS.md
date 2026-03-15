# Evolving Arbitrary Targets with AlphaEvolve

AlphaEvolve has been refactored to act as a generic evolution engine. You can now use it to optimize arbitrary code targets—like CUDA kernels, custom loss functions, or data pipelines—without modifying the `alphaevolve` source code itself.

Here is the step-by-step guide to evolving a new target.

---

## Step 1: Create a Target Folder

Create a dedicated directory for your optimization task. Treat this folder as a self-contained environment. AlphaEvolve will clone this directory into an isolated workspace for each candidate.

```bash
mkdir my_kernel_target
cd my_kernel_target
```

## Step 2: Write the Seed Code (with Markers)

Create the base program that you want AlphaEvolve to mutate. The LLM mutator looks for specific markers to know where it is allowed to make changes.

Create `my_kernel.cu` (or whatever language you are using):

```cpp
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void my_custom_kernel(float* out, const float* in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // EVOLVE-BLOCK-START
        // Optimize this naive implementation to use shared memory
        // or a different memory access pattern.
        out[idx] = in[idx] * in[idx] + 2.0f * in[idx];
        // EVOLVE-BLOCK-END
    }
}
```

*Note: The LLM will only rewrite the code between `EVOLVE-BLOCK-START` and `EVOLVE-BLOCK-END`.*

## Step 3: Write the Evaluator Script (`eval.sh`)

AlphaEvolve relies on a standard CLI contract: your evaluation script must take the candidate file as an argument, run whatever compilation and testing is necessary, and output a specific JSON payload to `stdout`.

Create `eval.sh`:

```bash
#!/bin/bash
# Usage: bash eval.sh <candidate_file>

CANDIDATE_FILE="$1"
if [ -z "$CANDIDATE_FILE" ]; then
    echo '{"valid": false, "aggregate_score": -1.0, "failure_reasons": ["No file provided"]}'
    exit 1
fi

# Step 1: Compile the code
# (Assuming you have a main.cpp that includes or links the candidate)
nvcc -O3 -o /tmp/kernel_test "$CANDIDATE_FILE" main.cpp > /tmp/compile_log.txt 2>&1

if [ $? -ne 0 ]; then
    # Compilation failed
    ERR=$(head -n 3 /tmp/compile_log.txt | tr '\n' ' ')
    echo "{\"valid\": false, \"aggregate_score\": -1.0, \"failure_reasons\": [\"Compilation failed: $ERR\"]}"
    exit 0
fi

# Step 2: Run the benchmark and capture output
# Assume your program prints statistics to stdout, e.g., "throughput_gbps: 450.5"
OUTPUT=$(/tmp/kernel_test)

if [ $? -ne 0 ]; then
    echo "{\"valid\": false, \"aggregate_score\": -1.0, \"failure_reasons\": [\"Runtime crash\"]}"
    exit 0
fi

# Step 3: Parse metrics and format the JSON response
# (Here using inline python to parse the output and format the JSON constraint)
echo "$OUTPUT" | python3 -c "
import sys, json, re
text = sys.stdin.read()

# Extract your target metrics
match = re.search(r'throughput_gbps:\s*([0-9.]+)', text)
if match:
    throughput = float(match.group(1))
    
    # The JSON must match this structure exactly
    result = {
        'valid': True,
        'aggregate_score': throughput,  # What AlphaEvolve ranks by
        'metrics': {
            'throughput_gbps': throughput
        },
        'failure_reasons': []
    }
    print(json.dumps(result))
else:
    print(json.dumps({'valid': False, 'aggregate_score': -1.0, 'failure_reasons': ['Missing metric']}))
"
```

Make sure the script is executable: `chmod +x eval.sh`

## Step 4: Add Context Documents (Optional but Recommended)

Write a short Markdown file explaining the API, hardware constraints, or specific strategies you want the LLM to try. AlphaEvolve will pass this to the LLM as background knowledge.

Create `CONTEXT.md`:
```md
# Optimization Strategies
- We are running on an NVIDIA A100.
- Try to utilize `__shared__` memory for intermediate accumulators.
- Avoid thread divergence in the main warp loop.
```

## Step 5: Define the Task Configuration (`task.json`)

This is the file that binds everything together. It tells AlphaEvolve where the files are and how to interpret the evaluation script.

Create `task.json` in the same directory:

```json
{
    "name": "optimize_my_kernel",
    "description": "Optimize a custom CUDA kernel for maximum throughput.",
    "seed_file": "my_kernel.cu",
    "mutable_files": ["my_kernel.cu"],
    "context_files": ["CONTEXT.md"],
    "eval_command": "bash eval.sh {candidate_file}",
    "eval_mode": "command",
    "metric_keys": ["throughput_gbps"],
    "primary_metric": "throughput_gbps",
    "eval_timeout": 60
}
```

## Step 6: Run AlphaEvolve

Your folder is now completely decoupled from AlphaEvolve. You run the tournament by pointing AlphaEvolve's CLI at your target directory.

```bash
uv run python alphaevolve/mvp/cli.py run --task-dir /path/to/my_kernel_target --mode gemini
```

### What happens inside the engine?
1. AlphaEvolve creates a workspace copy for `cand_0001`.
2. The LLM reads `my_kernel.cu` and `CONTEXT.md` and generates a diff for the `EVOLVE-BLOCK`.
3. The diff is applied to create the candidate.
4. AlphaEvolve executes `bash eval.sh /path/to/workspace/my_kernel.cu`.
5. `eval.sh` compiles, runs the program, and writes the JSON metric payload to stdout.
6. AlphaEvolve parses the JSON. If `"valid": true`, it logs the candidate's `aggregate_score` and advances to the tournament ranking phase. 
7. The cycle repeats, breeding new candidates from the ones with the highest `aggregate_score`.
