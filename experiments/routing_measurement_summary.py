"""Streaming routing summaries and mixture accounting; standard library only."""
import math


def mixture_summary(seed_bits, mixture_bits, targets):
    if targets <= 0 or not seed_bits:
        raise ValueError('Positive scored targets and at least one seed required')
    values = [x/targets for x in seed_bits]
    mixed = mixture_bits/targets
    if not all(math.isfinite(x) for x in values+[mixed]):
        raise ValueError('Nonfinite mixture score')
    mean = math.fsum(values)/len(values)
    gap = mean-mixed
    if gap < -1e-9:
        raise ValueError('Probability mixture violates the component-mean Jensen bound')
    return dict(seed_bpc=values, mean_seed_bpc=mean, mixture_bpc=mixed,
                jensen_gap_bpc=gap, mixture_minus_first_seed_bpc=mixed-values[0],
                scope='Jensen compares the mixture with mean constituent NLL; first-seed difference can have either sign')


class RoutingSummary:
    """Merge sufficient statistics without retaining scores or trajectories.

    Buckets distinguish each (depth, head). Counts include warming and scored
    input positions. Shared race noise prevents an IID interpretation.
    """
    def __init__(self, depth, heads, pool):
        if min(depth, heads, pool) < 1:
            raise ValueError('Positive routing dimensions required')
        self.depth, self.heads, self.pool = depth, heads, pool
        self.buckets = {}

    def add(self, depth, head, winner_counts, probability_sums, max_sum,
            entropy_sum, above_09, examples, boundary_keys=None,
            clock_sensitivity_sum=None, weak_clock_races=None):
        if not (0 <= depth < self.depth and 0 <= head < self.heads):
            raise ValueError('Invalid addressed bucket')
        if examples <= 0 or len(winner_counts) != self.pool or len(probability_sums) != self.pool:
            raise ValueError('Invalid routing statistic dimensions')
        if any(not isinstance(x, int) or x < 0 for x in winner_counts) or sum(winner_counts) != examples:
            raise ValueError('Observed winner counts must total the actual race count')
        if not all(math.isfinite(x) and x >= 0 for x in probability_sums):
            raise ValueError('Invalid probability totals')
        if abs(math.fsum(probability_sums)-examples) > 1e-9*examples:
            raise ValueError('Expected probability mass has the wrong denominator')
        if not math.isfinite(max_sum) or not examples/self.pool-1e-9 <= max_sum <= examples+1e-9:
            raise ValueError('Invalid maximum-probability sum')
        if not math.isfinite(entropy_sum) or not -1e-9 <= entropy_sum <= examples*math.log2(self.pool)+1e-9:
            raise ValueError('Invalid entropy sum')
        if not isinstance(above_09, int) or not 0 <= above_09 <= examples:
            raise ValueError('Invalid sharp-race count')
        if boundary_keys is not None and (not isinstance(boundary_keys, int) or not 0 <= boundary_keys <= examples*self.pool):
            raise ValueError('Invalid clamped-boundary key count')
        if clock_sensitivity_sum is not None:
            if not math.isfinite(clock_sensitivity_sum) or not 0 <= clock_sensitivity_sum <= .0025*examples+1e-12:
                raise ValueError('Invalid post-clamp common-clock sensitivity')
            if not isinstance(weak_clock_races, int) or not 0 <= weak_clock_races <= examples:
                raise ValueError('Invalid weak-clock race count')
        key = (depth, head)
        row = self.buckets.setdefault(key, dict(races=0, winner_counts=[0]*self.pool,
                                               probability_sums=[0.]*self.pool,
                                               max_sum=0., entropy_sum=0., above_09=0,
                                               measured_key_races=0, boundary_keys=0,
                                               measured_clock_races=0, sensitivity_sum=0., weak_clock_races=0))
        row['races'] += examples
        for i in range(self.pool):
            row['winner_counts'][i] += winner_counts[i]
            row['probability_sums'][i] += probability_sums[i]
        row['max_sum'] += max_sum
        row['entropy_sum'] += entropy_sum
        row['above_09'] += above_09
        if boundary_keys is not None:
            row['measured_key_races'] += examples
            row['boundary_keys'] += boundary_keys
        if clock_sensitivity_sum is not None:
            row['measured_clock_races'] += examples
            row['sensitivity_sum'] += clock_sensitivity_sum
            row['weak_clock_races'] += weak_clock_races

    def report(self):
        output = []
        for (depth, head), row in sorted(self.buckets.items()):
            count = row['races']
            output.append(dict(depth=depth, head=head, races=count,
                               winner_counts=list(row['winner_counts']),
                               observed_winner_fraction=[x/count for x in row['winner_counts']],
                               expected_probability_mass=list(row['probability_sums']),
                               mean_pi_per_unit=[x/count for x in row['probability_sums']],
                               mean_max_pi=row['max_sum']/count,
                               mean_entropy_bits=row['entropy_sum']/count,
                               max_entropy_bits=math.log2(self.pool),
                               fraction_max_pi_above_09=row['above_09']/count,
                               key_races_measured=row['measured_key_races'],
                               clock_races_measured=row['measured_clock_races'],
                               boundary_key_count=row['boundary_keys'] if row['measured_key_races'] else None,
                               fraction_keys_at_clamp_boundary=row['boundary_keys']/(self.pool*row['measured_key_races']) if row['measured_key_races'] else None,
                               mean_postclamp_common_clock_sensitivity=row['sensitivity_sum']/row['measured_clock_races'] if row['measured_clock_races'] else None,
                               fraction_clock_sensitivity_below_one_percent_max=row['weak_clock_races']/row['measured_clock_races'] if row['measured_clock_races'] else None))
        return output
