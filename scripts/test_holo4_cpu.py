"""CPU proofs for INT4 packing, draft activation and dtype boundaries."""
import importlib.util
import ast
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import torch

repo = Path('/repo')

def module(name):
    p = repo / 'patches' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, p)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

head = module('vllm_xpu_draft_lmhead_int4')
mtp = module('vllm_xpu_draft_mtp_int4')

class DraftTests(unittest.TestCase):
    def test_native_autoround_boundary_and_one_time_scale_conversion(self):
        source = Path('/source/vllm/model_executor/layers/quantization/inc/schemes/inc_wna16_linear.py')
        if not source.exists():
            self.skipTest('Patched source is not mounted')
        node = next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='INCXPULinearMethod')
        namespace = {'torch':torch,'Parameter':torch.nn.Parameter,'INCXPULinearBase':object}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
        method = namespace['INCXPULinearMethod']()
        method.group_size = 128
        method.is_awq_packed = False
        layer = torch.nn.Module()
        layer.qweight = torch.nn.Parameter(torch.zeros(16,16,dtype=torch.int32),requires_grad=False)
        layer.scales = torch.nn.Parameter(torch.ones(1,16,dtype=torch.bfloat16),requires_grad=False)
        method.process_weights_after_loading(layer)
        self.assertEqual(layer.scales.dtype,torch.float16)
        scales_id = layer.scales.data_ptr()
        seen=[]
        def gemm(x,qweight,bias,scales,*args):
            seen.append((x.dtype,bias.dtype,scales.dtype,scales.data_ptr()))
            return torch.ones(x.shape[0],16,dtype=torch.float16)+bias
        with patch.object(torch.ops._xpu_C,'int4_gemm_w4a16',gemm,create=True):
            out=method.apply_weights(layer,torch.ones(2,128,dtype=torch.bfloat16),torch.ones(16,dtype=torch.bfloat16))
        self.assertEqual(out.dtype,torch.bfloat16)
        self.assertTrue((out==2).all())
        self.assertEqual(seen,[(torch.float16,torch.float16,torch.float16,scales_id)])

    def test_zero_and_nonzero_packing(self):
        torch.manual_seed(17)
        w = torch.randn(9, 256, dtype=torch.bfloat16)
        w[0] = 0
        for quantize in [head.quantize_lmhead_to_int4, mtp.quantize_to_int4]:
            q, scales, zeros, group = quantize(w)
            shifts = torch.arange(0,32,4,dtype=torch.int64)
            decoded = (((q.t().to(torch.int64).unsqueeze(-1) >> shifts) & 15)-8).reshape(9,2,128)
            restored = (decoded.float()*scales.t().float().unsqueeze(-1)).reshape(9,256)
            self.assertTrue(torch.isfinite(scales).all())
            self.assertTrue((scales > 0).all())
            self.assertTrue((restored[0] == 0).all())
            bound = scales.t().float().unsqueeze(-1)/2 + .004
            self.assertTrue(((restored-w.float()).reshape(9,2,128).abs() <= bound).all())
            self.assertEqual(zeros.item(),8)
            self.assertEqual(group,128)

    def test_fork_env_and_one_time_activation(self):
        class Model:
            lm_head = torch.nn.Linear(128,16,bias=False,dtype=torch.bfloat16)
        model = Model()
        with patch.dict(os.environ, {'VLLM_XPU_DRAFT_LMHEAD_INT4':'1','B70_DRAFT_LMHEAD_INT4':'0'}):
            head.build_draft_lmhead_int4(model)
            first = model._vllm_xpu_lmhead_int4
            head.build_draft_lmhead_int4(model)
            self.assertIs(model._vllm_xpu_lmhead_int4,first)

    def test_kernel_private_fp16_and_public_bf16_boundary(self):
        weight = torch.randn(16,128,dtype=torch.bfloat16)
        q, scales, zeros, group = mtp.quantize_to_int4(weight)
        x = torch.randn(2,3,128,dtype=torch.bfloat16)
        seen = []
        def gemm(kernel_x, *args):
            seen.append(kernel_x.dtype)
            return torch.ones(kernel_x.shape[0],16,dtype=torch.float16)
        with patch.object(torch.ops._xpu_C, 'int4_gemm_w4a16', gemm, create=True):
            linear = mtp._VllmXpuMTPInt4LinearMethod(q,scales,zeros,group)
            layer = torch.nn.Linear(128,16,bias=False,dtype=torch.bfloat16)
            output = linear.apply(layer,x,None)
            self.assertEqual(output.dtype,torch.bfloat16)
            self.assertEqual(output.shape,(2,3,16))
            model = type('Model',(),{'lm_head':layer})()
            logits = head.int4_lmhead_logits(model,x,q,scales,zeros,group)
            self.assertEqual(logits.dtype,torch.bfloat16)
            self.assertEqual(logits.shape,(2,3,16))
        self.assertEqual(seen,[torch.float16,torch.float16])

if __name__ == '__main__':
    assert not torch.xpu.is_initialized()
    unittest.main()
