# AWS experiment host runbook

This provisions a single-purpose EC2 host for the Sleeping Machines queues. The CPU setup matches the code that is runnable today. A GPU setup is available for later CUDA-enabled comparisons; installing a GPU wheel does not move the current experiments onto the GPU.

## Choose the instance

- **Current CPU queue:** `c7i.4xlarge`, On-Demand, x86_64 Ubuntu 24.04 LTS. It has 16 vCPUs and 32 GiB RAM. The scripts currently build CPU tensors, so a GPU would sit idle. The safe runner still caps each job at 6,000,000 KB virtual memory and 3,500,000 KB RSS, and stops below 6,000 MB host `MemAvailable`.
- **GPU pilot after CUDA support is added:** `g7e.2xlarge` for one RTX PRO Server 6000 Blackwell GPU with 96 GB GPU memory and 64 GiB host RAM. `g6e.2xlarge` is the lower-memory comparison: one L40S with 48 GB GPU memory and the same host RAM. Confirm regional availability and compare current rates in the [AWS Pricing Calculator](https://calculator.aws/#/). AWS publishes the current [G7e specifications](https://aws.amazon.com/ec2/instance-types/g7e/) and [accelerated instance table](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html).

The GPU is useful for matched Transformer/LSTM baselines and a CUDA-port of the candidate model. The current scripts do not move their models and data to CUDA. E77 also includes Python-side event generation and dense candidate scoring, so a GPU alone does not make it sparse or fast. First make device use explicit, preserve CPU/GPU numerical agreement, and run a short throughput and memory pilot. The safe runner does not monitor GPU VRAM; keep `nvidia-smi` visible during any GPU pilot.

Use On-Demand for the first runs because the current training scripts do not save frequent resumable model checkpoints. Create a 100 GiB gp3 EBS volume and an AWS budget alert. Check storage retention and charges before stopping or terminating the instance; [EC2 On-Demand billing](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-on-demand-instances.html) covers running-instance charges, and the [calculator guide](https://docs.aws.amazon.com/pricing-calculator/latest/userguide/ec2-estimates.html) includes EBS and transfer estimates.

## Launch and secure the host

1. In EC2, choose the instance above and Ubuntu 24.04 x86_64. For the GPU route, choose the current **Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)**; its [AWS documentation](https://docs.aws.amazon.com/dlami/latest/devguide/aws-deep-learning-x86-base-gpu-ami-ubuntu-24-04.html) publishes the latest AMI lookup and driver releases. Confirm `nvidia-smi` works after boot.
2. Set the root EBS volume to 100 GiB gp3. Add an inbound SSH rule restricted to your current IP. Use an EC2 instance profile if the host needs S3; do not put AWS keys or Git tokens in this script or shell history.
3. Create a budget alert before leaving the instance running. Stop it when experiments are idle; export results first if the EBS volume may be deleted.

## Put the checkout at `/workspace`

`experiments/queue/run_safe.sh` intentionally uses `/workspace` for the checkout, venv and working directory. The bootstrap script checks this requirement rather than silently producing a host that cannot run the queue.

```bash
sudo install -d -o "$USER" -g "$(id -gn)" /workspace
git clone https://github.com/keskival/sleeping_machines.git /workspace
cd /workspace
```

Use your approved Git authentication method if the repository is private. Do not start the same queue on AWS while the existing workstation runner is active: its `flock` lock is local to one machine and cannot coordinate across hosts. The current checkout has an E64b 10M LSTM run in progress, and that script only saves its model at the end; let the existing chain finish or deliberately stop it before replaying that work on AWS.

## Install the experiment environment

The installer is idempotent and does not start any experiment. It installs OS utilities used by the safe runner, creates `.venv-docker`, installs repository requirements plus `h5py` and the selected PyTorch wheel, and downloads the 100 MB text8 file expected at `data/text8/text8`.

For the CPU host:

```bash
./scripts/bootstrap_aws_experiments.sh --cpu
```

For a GPU AMI after choosing a GPU instance:

```bash
./scripts/bootstrap_aws_experiments.sh --gpu
```

The GPU mode requires `nvidia-smi` and a driver new enough for the current Blackwell-capable PyTorch wheel. It checks that PyTorch can access CUDA. The script installs no CUDA driver itself; use the AWS GPU AMI. To record the resolved packages for a run:

```bash
.venv-docker/bin/python -m pip freeze > /tmp/sm-experiment-packages.txt
```

The report PDF can be rebuilt with:

```bash
MPLCONFIGDIR=/tmp/mpl_sm_report .venv-docker/bin/python report/make_pdf.py
```

## Run a queue without hanging the host

Use `tmux` so the runner remains attached to a terminal after SSH disconnects:

```bash
cd /workspace
tmux new -s sm-experiments
```

Inside `tmux`, inspect the queue file, confirm no other queue is running on this host, then start exactly one queue:

```bash
sed -n '1,160p' experiments/queue/e71a.txt
WAIT=1 ./experiments/queue/run_safe.sh experiments/queue/e71a.txt
```

Replace the filename with the reviewed queue you intend to run. `WAIT=1` waits for the host-local lock; it does not coordinate with other machines. The runner serializes jobs, pins CPU math libraries to one thread, enforces its address-space and process-RSS ceilings, and stops if host available memory drops below its floor. Do not bypass it for training jobs. The safe runner logs each job under `experiments/queue/logs/` and the queue runner summary under `experiments/queue/runner_<queue>.out`.

Detach from tmux with `Ctrl-B`, then `D`; reattach with `tmux attach -t sm-experiments`. To monitor host pressure from another shell:

```bash
free -h
tail -n 20 experiments/queue/runner_e71a.out
tail -n 10 experiments/queue/logs/<active-job>.log
```

Do not raise `MEM_CAP_KB`, `MEM_CAP_RSS_KB`, or lower `MIN_AVAIL_MB` until a measured run justifies it. On a GPU, also monitor `watch -n 2 nvidia-smi`; the safe runner does not cap GPU memory.

## Preserve results and shut down

Copy reviewed results and logs off the instance before termination, using your own S3 bucket or `scp` to your workstation. Keep the data and checkpoints needed to reproduce any reported metric. Training state is not resumable for the present E64 scripts until they finish.

When a queue finishes, review `experiments/FINDINGS.md` and the relevant result JSON before editing report claims. Rebuild the PDF after report changes. Stop or terminate the EC2 instance when idle, and check the EBS volume's delete-on-termination setting so the results are retained or removed as intended.

## References

- [AWS C7i](https://aws.amazon.com/ec2/instance-types/c7i/)
- [AWS G7e](https://aws.amazon.com/ec2/instance-types/g7e/)
- [AWS accelerated instance specifications](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html)
- [AWS Deep Learning Base GPU AMI, Ubuntu 24.04](https://docs.aws.amazon.com/dlami/latest/devguide/aws-deep-learning-x86-base-gpu-ami-ubuntu-24-04.html)
- [PyTorch Linux installation guidance](https://pytorch.org/get-started/locally/)
- [text8 source archive](https://mattmahoney.net/dc/text8.zip)
