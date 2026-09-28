# AWS instance recommendation for the current experiments

**Recommendation as of 2026-09-28: use a Linux `c7i.4xlarge` On-Demand instance for the next experiment tranche.** It provides 16 vCPUs and 32 GiB RAM. In us-east-1, the current listed On-Demand rate is about $0.714/hour before EBS, data transfer, and tax; confirm the price and region in the [AWS Pricing Calculator](https://calculator.aws/#/). AWS lists C7i as 4th-generation Intel Xeon compute-optimized instances, with up to 3.8 GHz max-core turbo and AMX support for CPU matrix work ([C7i specifications](https://aws.amazon.com/ec2/instance-types/c7i/)). The listed us-east-1 rate comes from [DoiT's EC2 price table](https://www.doit.com/compute/compute/aws/us-east-1/c7i.4xlarge), so treat it as a current estimate, not a quote.

## Why this is the right first machine

The queued character-LM work is currently CPU-only. The E64 and E77 scripts construct CPU tensors and models, and both set PyTorch intra-op threads to one; they do not move the model or data to CUDA. A GPU instance would therefore charge for an accelerator the present runs cannot use. A compute-optimized C7i is a sensible isolated host for the current baselines and E77 pilots while retaining enough RAM for deeper/longer runs.

This recommendation does **not** imply that a 16-vCPU instance makes a single run 16 times faster: the scripts currently restrict Torch to one CPU thread. It gives the experiment host capacity and memory headroom; actual speedup depends on single-thread performance and the event simulator's Python overhead. Keep the experiment queue serialized until it has reliable checkpoints and the memory monitor has been exercised on that host.

## Safe launch profile

- Use Linux On-Demand for the first runs. The current long jobs do not save frequent resumable checkpoints, so Spot interruption could discard substantial work.
- Start one safe-runner job at a time. In its training container, cap memory at 24 GiB with swap equal to the same limit, and cap CPU at 8 vCPUs; that leaves host capacity for the OS, Docker, and monitoring. Keep the runner's RSS watchdog enabled.
- Use a 100 GiB gp3 EBS volume for the OS, environment, data, and results. Copy results off-instance, then stop or terminate the instance when idle; check EBS retention separately when stopping it.
- Add an AWS budget alert before launching. EC2 On-Demand is billed for running time, while attached EBS storage can continue to incur charges after stopping ([AWS On-Demand billing](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-on-demand-instances.html), [Pricing Calculator](https://docs.aws.amazon.com/pricing-calculator/latest/userguide/ec2-estimates.html)).

The $0.714/hour estimate is about $0.71 for each running hour, or about $5.71 for an eight-hour session, excluding storage and transfer. Do not buy a long-term commitment for this exploratory phase.

## When to switch to a GPU instance

After a CUDA-capable path is implemented and checked against the CPU reference, try **`g6e.2xlarge`**: one NVIDIA L40S (48 GB GPU memory), 8 vCPUs, and 64 GiB host RAM. AWS lists those specifications on its [accelerated-computing instance page](https://aws.amazon.com/ec2/instance-types/accelerated-computing/). This is a useful single-GPU development box for matched Transformer controls and GPU kernels. It is not the right purchase for today's CPU-only scripts. Reassess after measuring tokens/second, peak host/GPU memory, and full-run cost on a short matched pilot.

### Sources

- [AWS C7i instance details](https://aws.amazon.com/ec2/instance-types/c7i/)
- [AWS accelerated instance specifications (G6e)](https://aws.amazon.com/ec2/instance-types/accelerated-computing/)
- [AWS Pricing Calculator: EC2 estimates](https://docs.aws.amazon.com/pricing-calculator/latest/userguide/ec2-estimates.html)
- [Current us-east-1 C7i.4xlarge price table](https://www.doit.com/compute/compute/aws/us-east-1/c7i.4xlarge)
