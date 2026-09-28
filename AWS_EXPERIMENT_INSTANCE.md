# AWS instance recommendation for the current experiments

**Recommendation as of 2026-09-28: use a Linux `c7i.4xlarge` On-Demand instance for the next CPU experiment tranche.** It provides 16 vCPUs and 32 GiB RAM. Check the selected region's live estimate in the [AWS Pricing Calculator](https://calculator.aws/#/). AWS lists C7i as compute-optimized Intel Xeon instances with AMX support for CPU matrix work ([C7i specifications](https://aws.amazon.com/ec2/instance-types/c7i/)).

## Why this is the right first machine

The E77 candidate and current queued work remain CPU-only; E64 now has an explicit `--device cuda` option for matched LSTM/Transformer baselines. Keep a compute-optimized C7i as the lower-cost host for E77 and the current queue. Use a GPU instance only for the E64 baseline pilot until E77's Python-side event generation and candidate scoring are ported and profiled.

The recommendation is for an isolated host with RAM headroom, not a promise of 16× speedup: current scripts restrict Torch to one CPU thread. Actual speed depends on single-thread performance and Python-side event simulation. Keep the experiment queue serialized and retain the safe runner's limits.

## Safe launch profile

- Use Linux On-Demand for the first runs. The current long jobs do not save frequent resumable checkpoints, so Spot interruption could discard substantial work.
- Run one job at a time through `experiments/queue/run_safe.sh` on the host. It already enforces a 6,000,000 KB virtual-memory limit, 3,500,000 KB process-group RSS limit, one math thread, and a 6,000 MB minimum available-memory floor. Keep those checks enabled.
- Use a 100 GiB gp3 EBS volume for the OS, environment, data, and results. Copy results off-instance, then stop or terminate the instance when idle; check EBS retention separately when stopping it.
- Add an AWS budget alert before launching. EC2 On-Demand is billed for running time, while attached EBS storage can continue to incur charges after stopping ([AWS On-Demand billing](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-on-demand-instances.html), [Pricing Calculator](https://docs.aws.amazon.com/pricing-calculator/latest/userguide/ec2-estimates.html)).

Do not buy a long-term commitment for this exploratory phase. Include EBS, transfer, and any retained snapshots in the estimate.

## When to switch to a GPU instance

For an E64 GPU baseline pilot, **`g6e.2xlarge`** is the lower-memory comparison with one L40S (48 GB GPU memory), 8 vCPUs, and 64 GiB host RAM; **`g7e.2xlarge`** provides one RTX PRO Server 6000 Blackwell GPU with 96 GB GPU memory. AWS lists G7e on its [instance page](https://aws.amazon.com/ec2/instance-types/g7e/) and gives [accelerated-computing specifications](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html). E77 still needs a separate CUDA port and performance study. Use the AWS GPU AMI with a current NVIDIA driver and first run a short matched E64 pilot that records BPC, tokens per second, host RSS, peak allocated GPU memory, and full-run cost.

### Sources

- [AWS C7i instance details](https://aws.amazon.com/ec2/instance-types/c7i/)
- [AWS G7e instance details](https://aws.amazon.com/ec2/instance-types/g7e/)
- [AWS accelerated instance specifications](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html)
- [AWS Deep Learning Base GPU AMI, Ubuntu 24.04](https://docs.aws.amazon.com/dlami/latest/devguide/aws-deep-learning-x86-base-gpu-ami-ubuntu-24-04.html)
- [AWS Pricing Calculator: EC2 estimates](https://docs.aws.amazon.com/pricing-calculator/latest/userguide/ec2-estimates.html)
