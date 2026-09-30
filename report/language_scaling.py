"""Declared architectural work scenarios, not fitted accuracy scaling laws.

Two FLOPs/MAC. Linear elementwise/state terms are illustrative unit-weight
allowances. Physical races replace numeric clock simulation only; candidate
discovery, traffic and joules remain separate. See theory note 48 for derivation.
"""
import math


def full_bank_comparison(width=256,heads=4,depth=8,context=4096,expansion=4,
                         update_targets=128,ours_update=None,alphabet=27,state_per_width=128):
    """Matched common projection/content block; every available key is scored.

    This isolates the attention substitution. It is not the fitted prototype's
    capacity/quality claim. Lower-order special functions are kept separately.
    """
    d,H,L,N,r,U,A=width,heads,depth,context,expansion,update_targets,alphabet
    if min(d,H,L,N,r,U,A)<1 or d%H:raise ValueError('Valid matched dimensions required')
    ours_update=U if ours_update is None else ours_update
    common=L*(8+4*r)*d*d+2*A*d
    state=L*state_per_width*d
    params=L*(4+2*r)*d*d+2*A*d
    return dict(transformer_inference=common+4*L*N*d,
        ours_inference=common+2*L*N*d+state,
        transformer_training=3*common+12*L*N*d+19*params/U,
        ours_training=3*common+10*L*N*d+3*state+20*params/ours_update,
        transformer_value_bytes=4*L*N*d,ours_value_bytes=4*L*d,
        transformer_key_value_bytes=8*L*N*d,ours_key_value_bytes=4*L*(N+1)*d,
        normalizing_exponentials=H*L*N,
        scope='Shared content/projections/parameter count, full-bank scoring, illustrative state allowance. No receiver-alternative extra work; add it for that architecture. Logical accesses, not DRAM or joules.')


def estimates(width=256,heads=4,depth=8,context=4096,candidates=12,pool=2,
              expansion=4,alphabet=27,transformer_update=8192,ours_update=128,
              state_inference=128,state_training=192):
    d,H,L,N,C,p,r,A=width,heads,depth,context,candidates,pool,expansion,alphabet
    if min(d,H,L,N,C,p,A,transformer_update,ours_update)<1 or C>N or d%H:
        raise ValueError('Positive dimensions; divisible head width and C <= N required')
    transformer_forward=L*((8+4*r)*d*d+4*N*d+30*d+5*H*N)+2*A*d
    ours_forward=L*((4+(2*p+14)/H)*d*d+2*(C+p)*d+state_inference*d+4*H*C)+2*d*d+2*A*d
    ours_teaching_forward=L*((4+(8*p+8)/H)*d*d+2*(C+p)*d+state_training*d+4*H*C)+2*d*d+2*A*d
    transformer_parameters=L*(4+2*r)*d*d+2*A*d
    ours_parameters=L*(2+(4*A*p+4)/H)*d*d+d*d+2*A*d
    # Illustrative occupancy under independent uniform symbols. Real text is
    # nonuniform; measured operator traces supersede this optimizer estimate.
    distinct=A*(1-(1-1/A)**ours_update)
    ours_updated_parameters=L*(2+(4*distinct*p+4)/H)*d*d+d*d+2*A*d
    return dict(transformer_inference=transformer_forward,ours_inference=ours_forward,
        transformer_training=3*transformer_forward+19*transformer_parameters/transformer_update,
        ours_training=3*ours_teaching_forward+4*L*(C+p)*d+20*ours_updated_parameters/ours_update,
        transformer_parameters=transformer_parameters,ours_parameters=ours_parameters,
        ours_updated_parameters=ours_updated_parameters,expected_distinct_symbols=distinct)


def winner_average_error_bound(variance,samples,omitted_probability=0.,value_norm_bound=0.):
    """RMS sampling error plus the retained-mass truncation bias bound."""
    if variance<0 or samples<1 or not 0<=omitted_probability<1 or value_norm_bound<0:
        raise ValueError('Valid variance, sample count, omitted mass and value bound required')
    return math.sqrt(variance/samples)+2*omitted_probability*value_norm_bound
