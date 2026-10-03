"""Static tensor accounting and finite race certificates; standard library only."""
import math


def cache_accounting(payload, depth, heads, pool, lanes, scalar_bytes=4):
    """Final tensor payloads of current _stacked/sparse_logits, not RSS/DRAM.

    Unit parameters are copied even for losing units. Rates are transformed;
    frequencies are stored in float64. Queries are stacked once per layer.
    Excludes raw model storage, temporary tensors, context and input/output.
    """
    dimensions = (payload, depth, heads, pool, lanes)
    if any(not isinstance(x, int) or x < 1 for x in dimensions) or payload % 2 or scalar_bytes not in (4, 8):
        raise ValueError('Positive dimensions, even payload and float32/64 required')
    slots = depth*heads*pool
    # Four P×P maps; key, gate bias and control map total4P; rate P/2;
    # clock/control bias total3; frequency P/2, always float64.
    unit_scalar_elements = 4*payload**2+4*payload+payload//2+3
    frequency_elements = payload//2
    query_elements = depth*heads*payload*(heads*payload)
    stacked = slots*(scalar_bytes*unit_scalar_elements+8*frequency_elements)+scalar_bytes*query_elements
    source_parameter_bytes = scalar_bytes*(slots*(unit_scalar_elements+frequency_elements)+query_elements)
    memory = lanes*slots*payload*scalar_bytes
    reads = memory
    timestamps = lanes*slots*8
    readiness = lanes*slots
    return dict(available_slots=slots, selected_writes_per_position=depth*heads,
                scored_keys_per_position=slots, sparse_proposals_per_position=depth*heads,
                final_stacked_tensor_bytes_per_call=stacked,
                parameter_payload_bytes_read_for_stacking=source_parameter_bytes,
                minimum_stacking_payload_read_plus_final_write_bytes=source_parameter_bytes+stacked,
                unit_memory_bytes=memory, cached_key_read_bytes=reads,
                timestamp_bytes=timestamps, readiness_bytes=readiness,
                allocated_unit_state_and_cache_bytes=memory+reads+timestamps+readiness,
                scope='Shape-derived tensor payloads, not measured allocator/RSS/DRAM traffic or energy; final stack plus unit cache excludes raw model, temporaries, context and input/output')


def race_certificate(reference_times, candidate_times, reference_winner, candidate_winner):
    """Certify a finite candidate perturbation using log-race-time margins.

    A sufficient condition: for every losing j, l_j-l_w > |delta_j|+|delta_w|.
    Times/decisions must refer to the same entering race; this is not a
    global floating-point guarantee or a proof from similar output logits.
    """
    if not reference_times or len(reference_times) != len(candidate_times):
        raise ValueError('Matching nonempty candidate times required')
    if not all(math.isfinite(t) and t > 0 for t in list(reference_times)+list(candidate_times)):
        raise ValueError('Finite positive race times required')
    for times, winner in ((reference_times, reference_winner), (candidate_times, candidate_winner)):
        if not isinstance(winner, int) or winner != min(range(len(times)), key=lambda i: times[i]):
            raise ValueError('Decision disagrees with the supplied race times/tie rule')
    logs = [math.log(t) for t in reference_times]
    other = [math.log(t) for t in candidate_times]
    error = [abs(a-b) for a, b in zip(logs, other)]
    losers = [i for i in range(len(logs)) if i != reference_winner]
    margin = min((logs[j]-logs[reference_winner] for j in losers), default=None)
    residual = min((logs[j]-logs[reference_winner]-error[j]-error[reference_winner]
                    for j in losers), default=None)
    certified = not losers or residual > 0
    agrees = reference_winner == candidate_winner
    if certified and not agrees:
        raise AssertionError('Margin certificate contradicted the decision')
    return dict(winner_agrees=agrees, certified=certified, reference_log_margin=margin,
                conservative_residual_margin=residual, max_abs_log_time_difference=max(error))
