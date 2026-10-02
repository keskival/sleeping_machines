# AWS: trained E64 Transformer as a race-attention stream (Theory §397), inference only

Run on the host that holds the saved E64 Transformer checkpoints ({"args","state"} from experiments/e64_lm_baselines.py):

    python experiments/race_transformer_eval.py --tag aws_race_transformer_D10M_<ts> --checkpoint <path to D=10M transformer .pt> --chars 20000
    python experiments/race_transformer_eval.py --tag aws_race_transformer_D90M_<ts> --checkpoint <path to D=90M transformer .pt> --chars 20000

One-job queues through run_safe. Inference only: no training, no official-test selection. The text is text8[95M:95M+20000],
scored with the E64 window protocol. Gates: expected delivery must reproduce the reference bpc (|Δ| < 1e-4); then report
the sparsity frontier (bpc vs value reads and key scores per token). Contracts: tests/test_race_transformer.py.
