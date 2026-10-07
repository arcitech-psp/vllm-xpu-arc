"""CPU-only compilation of an isolated W4A16 prototype; no model lifecycle."""
import argparse,os,pathlib
p=argparse.ArgumentParser();p.add_argument('--onednn',choices=['installed','3132'],default='installed');args=p.parse_args()
os.environ['MAX_JOBS']='1'
compiler='/opt/intel/oneapi/compiler/'+('2026.0'if args.onednn=='3132'else'2026.0')+'/bin/'
os.environ['CXX']=compiler+'icpx'
os.environ['PATH']=compiler+'/opt/venv/bin:'+os.environ['PATH']
os.environ['TORCH_XPU_ARCH_LIST']=''
from torch.utils.cpp_extension import load
import torch
assert not torch.xpu.is_initialized()
root=pathlib.Path(__file__).resolve().parent;build=root/('mxfp4-w4a16-build-3132'if args.onednn=='3132'else'mxfp4-w4a16-build');build.mkdir(exist_ok=True)
dnnl='/work/onednn-3.13.2-install'if args.onednn=='3132'else'/opt/intel/oneapi/dnnl/2026.0'
libname='vllm_xpu_dnnl_3132'if args.onednn=='3132'else'dnnl'
load(name='vllm_xpu_mxfp4_woq',sources=[str(root/'mxfp4_w4a16.cpp')],
 extra_include_paths=[dnnl+'/include'],extra_cflags=['-O2','-fsycl','-Wno-deprecated-declarations'],
 extra_ldflags=['-fsycl','-ltorch_xpu','-lc10_xpu','-L'+dnnl+'/lib','-Wl,-rpath,'+dnnl+'/lib','-l'+libname],
 with_sycl=False,is_python_module=False,build_directory=str(build),verbose=True)
assert not torch.xpu.is_initialized(),'CPU-only build must not create a competing XPU context'
print('MXFP4_W4A16_COMPILED_NO_XPU_CONTEXT',flush=True)
