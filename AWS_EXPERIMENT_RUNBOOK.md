# AWS experiment host runbook

This provisions a single-purpose EC2 host for the Sleeping Machines queues. The CPU setup remains the default. E64 LSTM/Transformer baselines now have an explicit CUDA option; E77 remains CPU-only because its Python-side event generation and candidate scoring have not been ported or profiled on a GPU.

## Choose the instance

- **CPU queue and E77:** `c7i.4xlarge`, On-Demand, x86_64 Ubuntu 24.04 or 26.04 LTS. It has 16 vCPUs and 32 GiB RAM. E77 still runs on CPU.
- **E64 CUDA baseline pilot:** `g7e.2xlarge` for one RTX PRO Server 6000 Blackwell GPU with 96 GB GPU memory and 64 GiB host RAM. `g6e.2xlarge` is the lower-memory comparison: one L40S with 48 GB GPU memory and the same host RAM. Confirm regional availability and compare current rates in the [AWS Pricing Calculator](https://calculator.aws/#/). AWS publishes the current [G7e specifications](https://aws.amazon.com/ec2/instance-types/g7e/) and [accelerated instance table](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html).

The E64 baseline source accepts `--device cuda`; CPU remains the default. The CUDA path limits PyTorch's caching allocator to half of total visible VRAM by default and saves CPU-portable checkpoints for E76. This cap covers PyTorch allocator use, not memory already used by other GPU processes; inspect `nvidia-smi` first. CUDA execution has not yet been validated on this CPU-only workspace, so begin with the short pilot below. E77 is not CUDA-enabled. The safe runner does not monitor GPU VRAM, so keep `nvidia-smi` visible during any GPU pilot. PyTorch documents the allocator limit [here](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.set_per_process_memory_fraction.html).

Use On-Demand for the first runs because the current training scripts do not save frequent resumable model checkpoints. Create a 100 GiB gp3 EBS volume and an AWS budget alert. Check storage retention and charges before stopping or terminating the instance; [EC2 On-Demand billing](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-on-demand-instances.html) covers running-instance charges, and the [calculator guide](https://docs.aws.amazon.com/pricing-calculator/latest/userguide/ec2-estimates.html) includes EBS and transfer estimates.

## Launch and secure the host

1. In EC2, choose the instance above and Ubuntu 24.04 or 26.04 x86_64. For the GPU route, choose the current **Deep Learning Base OSS Nvidia Driver GPU AMI** for the same Ubuntu version; AWS publishes the [Ubuntu 24.04](https://docs.aws.amazon.com/dlami/latest/devguide/aws-deep-learning-x86-base-gpu-ami-ubuntu-24-04.html) and [Ubuntu 26.04](https://docs.aws.amazon.com/dlami/latest/devguide/aws-deep-learning-x86-base-gpu-ami-ubuntu-26-04.html) AMI lookups and driver releases. Confirm `nvidia-smi` works after boot.
2. Set the root EBS volume to 100 GiB gp3. Add an inbound SSH rule restricted to your current IP. Use an EC2 instance profile if the host needs S3; do not put AWS keys or Git tokens in this script or shell history.
3. Create a budget alert before leaving the instance running. Stop it when experiments are idle; export results first if the EBS volume may be deleted.

## Put the checkout at `/workspace`

`experiments/queue/run_safe.sh` intentionally uses `/workspace` for the checkout, venv and working directory. The bootstrap script checks this requirement rather than silently producing a host that cannot run the queue.

```bash
sudo install -d -o "$USER" -g "$(id -gn)" /workspace
git clone https://github.com/keskival/sleeping_machines.git /workspace
cd /workspace
```

Use your approved Git authentication method if the repository is private. Do not start the same queue on AWS while the workstation runner is active: its `flock` lock is local to one machine and cannot coordinate across hosts. Confirm the workstation queue is complete before replaying those jobs on AWS; the current training scripts save models only at the end and cannot resume an interrupted run.

## Install the experiment environment

The installer supports Ubuntu 24.04 and 26.04 LTS on x86_64 and does not start any experiment. It installs OS utilities used by the safe runner, creates `.venv-docker`, installs repository requirements plus `h5py` and the selected PyTorch wheel, and downloads the 100 MB text8 file expected at `data/text8/text8`.

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

Before a CUDA pilot, confirm the GPU is idle with `nvidia-smi`. CUDA uses a unified virtual address space for host and GPU allocations, so the runner's default 6,000,000 KB `ulimit -v` can constrain its mappings. For the GPU queue only, remove that virtual-address-space limit while keeping the physical RSS ceiling and host-memory floor unchanged:

```bash
MEM_CAP_KB=unlimited MEM_CAP_RSS_KB=3500000 MIN_AVAIL_MB=6000 WAIT=1 ./experiments/queue/run_safe.sh /tmp/e64_cuda_pilot.txt
```

`RLIMIT_AS` limits virtual address space, not physical RAM. The runner's 3,500,000 KB process-group RSS ceiling and 6,000 MB `MemAvailable` floor remain enabled as host-RAM safeguards ([Linux `getrlimit`](https://www.man7.org/linux/man-pages/man2/prlimit.2.html)); NVIDIA documents CUDA's unified host/GPU virtual address space in its [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html). Keep the queue serial and the physical-memory safeguards unchanged.

Create a short, separate pilot queue (the normal 10M jobs are not a first CUDA smoke run):

```bash
cat > /tmp/e64_cuda_pilot.txt <<'EOF'
e64_tf_cuda_pilot experiments/e64_lm_baselines.py --model tf --D 200000 --passes 2 --size 128 --layers 2 --dropout 0.1 --valid 20000 --test 100000 --device cuda --gpu_memory_fraction 0.5
EOF
```

Record the pilot's BPC, wall time, `gpu_peak_allocated_gib`, host RSS, and `nvidia-smi` utilization. Repeat the same command with `--device cpu` on the same host before comparing speed or numerical results. This pilot does not validate CUDA support for E77.

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

## Use Codex on AWS for heavier benchmarks

Run Codex from the AWS checkout and use it for one benchmark at a time. The
queue lock at `/tmp/experiments-runner.lock` serializes jobs on that EC2 host;
it does not coordinate with the workstation or another EC2 instance. Keep AWS
run tags distinct (for example, include `aws`) so results from two machines do
not overwrite one another when copied or merged. Use a separate Git branch on
AWS while the workstation checkout is also changing.

Codex CLI automatically reads repository `AGENTS.md` instructions, so the
checkout's [host rules](AGENTS.md) apply at session start. The prompt below
sets the current AWS task and the larger benchmark priority.

Give Codex this instruction at the start of its AWS session:

```text
This EC2 host is for completing the larger, previously deferred Transformer
benchmarks. Start with the existing benchmark configurations and results; do
not invent a new comparison before checking REPORT.md, experiments/FINDINGS.md,
and experiments/queue. Prioritize the most informative deferred large run.

Before each run, inspect free -h, MemAvailable, running experiment processes,
the GPU with nvidia-smi if present, and the intended queue/log/result paths. Run
only one training job at a time. Never launch a benchmark directly with Python:
put it in a uniquely named one-job queue and use
experiments/queue/run_safe.sh, which owns the host-local lock. If the lock is
busy, wait or report that; never bypass or delete it. The lock is local to this
EC2 host and does not coordinate with the workstation.

Choose MEM_CAP_KB, MEM_CAP_RSS_KB, MIN_AVAIL_MB, and JOB_TIMEOUT_S from the
actual instance capacity and the model's measured needs. Keep at least 8 GiB
MemAvailable for the OS and other services, retain the RSS watchdog, and use
tmux for long runs. For CUDA, check that the GPU is otherwise idle, monitor
nvidia-smi, and set the PyTorch GPU-memory fraction explicitly. Do not run two
GPU jobs concurrently.

Give every run a unique AWS-specific run_tag and output name. Never overwrite
an earlier result or reuse a successful queue job name for changed settings.
Record the exact command, instance type, CPU/GPU, memory peak, wall time, and
benchmark metrics. Update findings and the report only from completed result
files. Work on a dedicated AWS Git branch; commit the benchmark and report
changes there with `ALLOW_NON_MAIN_BRANCH=1 ./commit_done.sh 'AWS benchmark results'`.
Do not push to main while another host may be changing it.
```

The checkout's `run_safe.sh` defaults protect a small host and may be too
restrictive for a large Transformer. Raise the per-job caps only after reading
the instance memory and confirming that the chosen RSS cap still leaves the
8 GiB floor. For a CUDA process, the runbook's `MEM_CAP_KB=unlimited` example
removes only the virtual-address-space cap; the process RSS ceiling and
available-memory floor must remain active.

## Preserve results and shut down

Copy reviewed results and logs off the instance before termination, using your own S3 bucket or `scp` to your workstation. Keep the data and checkpoints needed to reproduce any reported metric. Training state is not resumable for the present E64 scripts until they finish.

When a queue finishes, review `experiments/FINDINGS.md` and the relevant result JSON before editing report claims. Rebuild the PDF after report changes. Stop or terminate the EC2 instance when idle, and check the EBS volume's delete-on-termination setting so the results are retained or removed as intended.

## References

- [AWS C7i](https://aws.amazon.com/ec2/instance-types/c7i/)
- [AWS G7e](https://aws.amazon.com/ec2/instance-types/g7e/)
- [AWS accelerated instance specifications](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html)
- [AWS Deep Learning Base GPU AMI, Ubuntu 24.04](https://docs.aws.amazon.com/dlami/latest/devguide/aws-deep-learning-x86-base-gpu-ami-ubuntu-24-04.html)
- [AWS Deep Learning Base GPU AMI, Ubuntu 26.04](https://docs.aws.amazon.com/dlami/latest/devguide/aws-deep-learning-x86-base-gpu-ami-ubuntu-26-04.html)
- [PyTorch Linux installation guidance](https://pytorch.org/get-started/locally/)
- [text8 source archive](https://mattmahoney.net/dc/text8.zip)
