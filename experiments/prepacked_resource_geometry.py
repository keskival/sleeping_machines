"""Weight-preparation payload amortization; stdlib, not measured DRAM or TCO."""


def preparation_payload(assembly_read_bytes, assembly_write_bytes, snapshot_bytes, requests):
    """Same assembly policy, with one additional private weight snapshot.

    Counts declared tensor payload reads/writes, not allocator/copy temporary
    traffic, metadata, state/cache setup, device transfers or cache residency.
    The private snapshot isolates the caller's model and adds one full read
    and write. Every weight version must pay its preparation again.
    """
    values = (assembly_read_bytes, assembly_write_bytes, snapshot_bytes, requests)
    if any(not isinstance(x, int) or isinstance(x, bool) for x in values):
        raise ValueError('Integer byte payloads and request count required')
    if min(values[:3]) < 0 or requests < 1 or assembly_read_bytes+assembly_write_bytes < 1:
        raise ValueError('Nonnegative payloads, nonzero assembly and positive requests required')
    assembly = assembly_read_bytes+assembly_write_bytes
    ordinary = requests*assembly
    prepared = assembly+2*snapshot_bytes
    return dict(requests_per_weight_version=requests,
                ordinary_repeated_assembly_payload_rw_bytes=ordinary,
                prepared_snapshot_and_once_assembly_payload_rw_bytes=prepared,
                payload_rw_difference_bytes=ordinary-prepared,
                first_request_count_with_positive_payload_difference=2+(2*snapshot_bytes)//assembly,
                retained_private_snapshot_plus_stack_payload_bytes=snapshot_bytes+assembly_write_bytes,
                payload_per_request_prepared_bytes=prepared/requests,
                scope='Declared tensor payload ledger; no measured physical traffic, wall-time, cost or energy saving')
