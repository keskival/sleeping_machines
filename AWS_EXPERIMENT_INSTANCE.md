# AWS instance recommendation for the current experiments

**Recommendation as of 2026-09-28: use a Linux `c7i.4xlarge` On-Demand instance for the next CPU experiment tranche.** It provides 16 vCPUs and 32 GiB RAM. Check the selected region's live estimate in the [AWS Pricing Calculator](https://calculator.aws/#/). AWS lists C7i as compute-optimized Intel Xeon instances with AMX support for CPU matrix work ([C7i specifications](https://aws.amazon.com/ec2/instance-types/c7i/)).

## Why this is the right first machine

The queued character-LM work is currently CPU-only. The E64 and E77 scripts construct CPU tensors and models, and both set PyTorch intra-op threads to one; they do not move the model or data to CUDA. A GPU instance would therefore charge for an accelerator the present runs cannot use. A compute-optimized C7i is a sensible isolated host for the current baselines and E77 pilots while retaining enough RAM for deeper/longer runs.

The recommendation is for an isolated host with RAM headroom, not a promise of 16× speedup: current scripts restrict Torch to one CPU thread. Actual speed depends on single-thread performance and Python-side event simulation. Keep the experiment queue serialized and retain the safe runner's limits.

## Safe launch profile

- Use Linux On-Demand for the first runs. The current long jobs do not save frequent resumable checkpoints, so Spot interruption could discard substantial work.
- Run one job at a time through `experiments/queue/run_safe.sh` on the host. It already enforces a 6,000,000 KB virtual-memory limit, 3,500,000 KB process-group RSS limit, one math thread, and a 6,000 MB minimum available-memory floor. Keep those checks enabled.
- Use a 100 GiB gp3 EBS volume for the OS, environment, data, and results. Copy results off-instance, then stop or terminate the instance when idle; check EBS retention separately when stopping it.
- Add an AWS budget alert before launching. EC2 On-Demand is billed for running time, while attached EBS storage can continue to incur charges after stopping ([AWS On-Demand billing](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-on-demand-instances.html), [Pricing Calculator](https://docs.aws.amazon.com/pricing-calculator/latest/userguide/ec2-estimates.html)).

Do not buy a long-term commitment for this exploratory phase. Include EBS, transfer, and any retained snapshots in the estimate.

## When to switch to a GPU instance

The current code does not use CUDA. Once the matched baselines and candidate have explicit CUDA support, **`g7e.2xlarge`** is the high-memory single-GPU pilot: one NVIDIA RTX PRO Server 6000 Blackwell GPU with 96 GB GPU memory, 8 vCPUs, and 64 GiB host RAM. A lower-memory comparison is **`g6e.2xlarge`**, with one L40S (48 GB GPU memory), 8 vCPUs, and 64 GiB host RAM. AWS lists G7e on its [instance page](https://aws.amazon.com/ec2/instance-types/g7e/) and gives [accelerated-computing specifications](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html). These machines become useful after device transfer is implemented; current E64/E77 constructors keep models and inputs on CPU, and E77's Python event/candidate-search work will need profiling before GPU benefit can be expected. Use the AWS GPU AMI with a current NVIDIA driver and first run a short matched pilot that records tokens per second, peak host/GPU memory, and full-run cost.

### Sources

- [AWS C7i instance details](https://aws.amazon.com/ec2/instance-types/c7i/)
- [AWS G7e instance details](https://aws.amazon.com/ec2/instance-types/g7e/)
- [AWS accelerated instance specifications](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html)
- [AWS Deep Learning Base GPU AMI, Ubuntu 24.04](https://docs.aws.amazon.com/dlami/latest/devguide/aws-deep-learning-x86-base-gpu-ami-ubuntu-24-04.html)
- [AWS Pricing Calculator: EC2 estimates](https://docs.aws.amazon.com/pricing-calculator/latest/userguide/ec2-estimates.html)
