"""Shape-based training estimates for E64; multiply-add counts as two FLOPs."""


def estimate_training_flops(args, params, steps, alphabet=27):
    """Estimate forward, backward, clipping and Adam; excludes validation/test.

    Backward is approximated as twice forward. Dense attention is counted even
    with a causal mask. Transcendentals, division and sqrt count as one operation;
    elementary overhead is approximate, not an operator trace or hardware count.
    """
    width = args['size']
    context = args.get('ctx', 256)
    tokens = steps * args.get('batch_size', 32) * context
    if args['model'] == 'tf':
        layers = args.get('layers', 2)
        # QKV/output projections, two FFN matrices, QK and attention-value.
        matrix_per_token = layers * (24 * width ** 2 + 4 * context * width) + 2 * width * alphabet
        elementwise_per_token = layers * (30 * width + 5 * 4 * context) + 5 * alphabet
    else:
        # Four input/recurrent gates, embedding width 64, output projection.
        matrix_per_token = 8 * width * (64 + width) + 2 * width * alphabet
        elementwise_per_token = 30 * width + 5 * alphabet
    forward = tokens * (matrix_per_token + elementwise_per_token)
    backward = 2 * forward
    adam = steps * params * 14
    clipping = steps * params * 5
    return dict(method='shape_based_v1', multiply_add_flops=2,
                forward_flops=forward, backward_flops=backward,
                adam_flops=adam, gradient_clipping_flops=clipping,
                total_training_flops=forward + backward + adam + clipping,
                training_token_positions=tokens,
                assumptions=['Backward approximated as 2x forward',
                             'Dense attention includes masked positions',
                             'Elementwise overhead is approximate; nonlinear/divide/sqrt count as 1',
                             'Adam 14 and gradient clipping 5 operations per parameter per step'],
                excludes=['Validation and test inference', 'Data loading and indexing',
                          'Memory movement', 'RNG generation', 'Learning-rate scheduler'])
