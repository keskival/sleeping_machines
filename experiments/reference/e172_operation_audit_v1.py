"""Executed ATen arithmetic by stage, including backward and Adam.

Shape-based mathematical work, not CPU instructions, physical memory traffic,
or joules. A multiply-add is two FLOPs. Transcendentals and comparisons have
separate ledgers. Fused reductions use explicit conventional formulas. Unknown
floating operators are visible and prevent a claim of complete coverage.
"""
from collections import defaultdict
from math import prod
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
from torch.utils.flop_counter import flop_registry


def tensors(value):
    return [v for v in tree_flatten(value)[0] if isinstance(v, torch.Tensor)]


def floating_size(value):
    return sum(v.numel() for v in tensors(value) if v.is_floating_point())


class OperationAudit(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.rows = defaultdict(lambda: dict(calls=0, arithmetic_flops=0,
            special_function_evaluations=0, comparisons=0, floating_input_elements=0,
            floating_output_elements=0, classification='unclassified'))

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        row = self.rows[str(func)]
        row['calls'] += 1
        row['floating_input_elements'] += floating_size((args, kwargs))
        row['floating_output_elements'] += floating_size(out)
        flops, special, comparisons, classification = self.formula(func,args,kwargs,out)
        row['arithmetic_flops'] += int(flops)
        row['special_function_evaluations'] += int(special)
        row['comparisons'] += int(comparisons)
        row['classification'] = classification
        return out

    def formula(self, func, args, kwargs, out):
        name = str(func).split('.')[1]
        base = name.rstrip('_')
        n = floating_size(out)
        x = tensors(args)[0] if tensors(args) else None
        inp = floating_size(x) if x is not None else 0
        if func._overloadpacket in flop_registry:
            f = flop_registry[func._overloadpacket](*args, **kwargs, out_val=out)
            if base in ('addmm','baddbmm') and kwargs.get('beta',1) != 0:
                f += n
            return f,0,0,'registered contraction (2 FLOPs/MAC)'
        if base in ('add','sub','mul','div'):
            extra = n if base in ('add','sub') and kwargs.get('alpha',1)!=1 else 0
            return n+extra,0,0,'elementwise arithmetic'
        if base in ('addcmul','addcdiv','lerp'):
            return 3*n,0,0,'three scalar arithmetic operations'
        if base in ('neg',): return n,0,0,'scalar sign arithmetic'
        if base in ('pow', 'square'):
            exponent=args[1] if len(args)>1 else 2
            return (n,0,0,'square') if exponent==2 else (0,n,0,'power special function')
        if base in ('exp','expm1','log','log1p','sqrt','rsqrt','sin','cos','tanh','erf'):
            return 0,n,0,'special function'
        if base=='sigmoid':return 2*n,n,0,'sigmoid: exp + add + reciprocal'
        if base=='sigmoid_backward':return 3*n,0,0,'sigmoid derivative'
        if base=='tanh_backward':return 3*n,0,0,'tanh derivative'
        if base=='threshold_backward':return 0,0,n,'threshold comparison/mask'
        if base in ('relu','clamp','clamp_min','clamp_max','maximum','minimum','abs','sign'):
            return 0,0,n,'comparison/sign selection'
        if base in ('sum','mean'):
            f=max(inp-n,0)+(n if base=='mean' else 0)
            return f,0,0,'reduction additions and optional division'
        if base in ('index_add','scatter_add'):
            return floating_size(args[3] if base=='index_add' else args[3]),0,0,'source scatter additions'
        if base=='embedding_dense_backward':
            return floating_size(args[0]),0,0,'embedding gradient scatter-add upper count'
        if base=='_softmax':
            dim=args[1]; r=inp//x.shape[dim]
            return 3*inp-r,inp,max(inp-r,0),'stable softmax arithmetic + exp + max'
        if base=='_softmax_backward_data':
            dim=args[2];r=inp//x.shape[dim]
            return 4*inp-r,0,0,'softmax backward reduction'
        if base=='_log_softmax':
            dim=args[1];r=inp//x.shape[dim]
            return 3*inp-r,inp+r,max(inp-r,0),'stable log-softmax arithmetic + exp/log + max'
        if base=='_log_softmax_backward_data':
            dim=args[2];r=inp//x.shape[dim]
            return 3*inp-r,inp,0,'log-softmax backward'
        if base in ('native_layer_norm','native_layer_norm_backward'):
            normalized=args[1] if base=='native_layer_norm' else args[2]
            features=prod(normalized);rows=inp//features
            # Backward's first argument is the gradient with the same shape as input.
            return (7*inp+rows,rows,0,'fused layer norm conventional formula') if base=='native_layer_norm' else (
                11*inp-2*rows,0,0,'fused layer norm backward conventional formula')
        if base in ('gelu','gelu_backward'):
            return (5*inp,inp,0,'GELU erf formula') if base=='gelu' else (
                11*inp,2*inp,0,'GELU backward erf/exp formula')
        if base in ('softplus','softplus_backward'):
            return (2*inp,2*inp,inp,'softplus exp/log formula') if base=='softplus' else (
                3*inp,inp,inp,'softplus derivative')
        if base.startswith('_foreach'):
            op=base[len('_foreach_'):]
            elements=floating_size(args[0])
            if op in ('add','sub','mul','div'):return elements,0,0,'foreach elementwise arithmetic'
            if op in ('lerp','addcmul','addcdiv'):return 3*elements,0,0,'foreach compound arithmetic'
            if op in ('sqrt',):return 0,elements,0,'foreach special function'
            if op=='norm':
                ts=tensors(args[0]);return 2*elements-len(ts),len(ts),0,'foreach L2 norm squares/reductions'
        if base=='linalg_vector_norm':
            norm=args[1] if len(args)>1 else 2
            if norm==2:return 2*inp-n,n,0,'L2 norm squares/reduction/sqrt'
            if norm==1:return inp-n,0,inp,'L1 norm abs and sum'
        if base in ('nll_loss_forward','nll_loss_backward'):
            # Selection, negation and reduction; no hidden dense class contraction.
            labels=args[1] if base=='nll_loss_forward' else args[2]
            items=labels.numel()
            return 2*items,0,0,'NLL selection/reduction upper scalar count'
        if base.startswith('_scaled_dot_product_flash_attention_for_cpu'):
            q=args[1] if base.endswith('backward') else args[0]
            k=args[2] if base.endswith('backward') else args[1]
            v=args[3] if base.endswith('backward') else args[2]
            B,H,L,D=q.shape;S=k.shape[-2];dv=v.shape[-1];pairs=B*H*L*S
            backward=base.endswith('backward')
            f=pairs*(6*D+4*dv) if backward else 2*pairs*(D+dv)
            return (f+(5*pairs if backward else 4*pairs),pairs,pairs,
                    'fused CPU attention contraction + scalar formula')
        # Views/copies/lookups are zero arithmetic; their memory cost is NOT zero.
        zero={'alias','detach','view','view_as','reshape','_reshape_alias','unsqueeze','squeeze',
              'expand','permute','transpose','t','slice','select','as_strided','clone','copy',
              '_to_copy','to','contiguous','empty','empty_like','empty_strided','zeros','zeros_like',
              'ones','ones_like','full','full_like','new_empty','new_zeros','new_ones','new_full',
              'fill','zero','cat','stack','split','split_with_sizes','unbind','index','index_select',
              'index_copy','gather','scatter','embedding','masked_fill','where','_local_scalar_dense',
              'isfinite','isnan','isinf','any','all','eq','ne','gt','ge','lt','le','nonzero','argmin',
              'argmax','sort','argsort','bincount','arange','repeat_interleave','lift_fresh',
              'lift_fresh_copy','set','resize','_unsafe_view','_unsafe_index','scalar_tensor'}
        if base in zero:return 0,0,0,'data movement, indexing or control (excluded from FLOPs)'
        if not inp and not n:return 0,0,0,'integer/control (excluded from FLOPs)'
        return 0,0,0,'UNSUPPORTED floating operator'

    def result(self):
        rows=dict(self.rows)
        unknown={name:row for name,row in rows.items() if row['classification'].startswith('UNSUPPORTED')}
        return dict(arithmetic_flops=sum(r['arithmetic_flops'] for r in rows.values()),
            special_function_evaluations=sum(r['special_function_evaluations'] for r in rows.values()),
            formula_coverage_complete=not unknown,unsupported_floating_operators=unknown,
            operators=rows)
