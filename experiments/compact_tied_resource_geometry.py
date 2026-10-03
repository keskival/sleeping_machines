"""Compact tied-bank payloads; stdlib, not measured traffic or runtime savings."""
from sparse_inference_accounting import cache_accounting


def compact_accounting(payload, depth, heads, pool, lanes, scalar_bytes=4):
    ordinary = cache_accounting(payload, depth, heads, pool, lanes, scalar_bytes)
    if any(type(x) is not int for x in (payload, depth, heads, pool, lanes, scalar_bytes)):
        raise ValueError('Integer dimensions required')
    groups = depth * heads
    slots = groups * pool
    shared = 4 * payload ** 2 + 3 * payload + 2
    private = payload + payload // 2 + 1
    query = depth * heads ** 2 * payload ** 2
    compact = scalar_bytes * (groups * shared + slots * private + query) + 8 * slots * (payload // 2)
    assembly_read = scalar_bytes * (groups * shared + slots * (2 * payload + 1) + query)
    difference = ordinary['final_stacked_tensor_bytes_per_call'] - compact
    assert difference == scalar_bytes * groups * (pool - 1) * shared
    return dict(ordinary, compact_final_packed_tensor_bytes=compact,
                compact_parameter_payload_bytes_read_for_stacking=assembly_read,
                compact_heavy_map_banks=groups,
                ordinary_heavy_map_banks=slots,
                avoided_duplicate_packed_tensor_bytes=difference,
                heavy_shared_scalar_elements_per_bank=shared,
                private_parameter_scalar_elements=slots * (2 * payload + 1),
                explicit_heavy_winner_map_gather_payload_bytes_per_position=0,
                scope='Shape-derived tensor payloads for tied maps; state/cache and all key scoring unchanged. Not measured RSS, internal kernel temporaries, DRAM, wall time or energy.')
