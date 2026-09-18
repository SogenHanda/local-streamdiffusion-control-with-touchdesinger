import unittest

import torch

from streamdiffusion_bridge.cuda_graph import CapturedVAECall
from streamdiffusion_bridge.tensorrt_unet import BufferedTensorRTUNet


class FakeContext:
    def __init__(self, engine):
        self.engine = engine
    def set_tensor_address(self, name, address):
        return address == self.engine.tensors[name].data_ptr()
    def execute_async_v3(self, _stream):
        t = self.engine.tensors
        t['latent'].copy_(t['sample'] + t['timestep'][:,None,None,None] + t['encoder_hidden_states'].mean((1,2))[:,None,None,None])
        return True


class FakeEngine:
    def __init__(self):
        self.tensors = {}
        self.allocations = 0
        self.context = FakeContext(self)
    def allocate_buffers(self, shape_dict, device):
        self.allocations += 1
        self.tensors = {k:torch.empty(v,device=device) for k,v in shape_dict.items()}


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for stream/graph regression')
class BufferLifetimeTests(unittest.TestCase):
    def test_unet_refreshes_all_inputs_and_recaptures_changed_shape(self):
        for graph in (False, True):
            with self.subTest(graph=graph):
                engine = FakeEngine()
                adapter = BufferedTensorRTUNet(engine, use_cuda_graph=graph)
                try:
                    for i,batch in enumerate((2,2,2,1,1)):
                        image = torch.full((batch,4,8,8), float(i),device='cuda')
                        steps = torch.full((batch,),float(i+1),device='cuda')
                        prompt = torch.full((batch,2,3),float(i+2),device='cuda')
                        result = adapter(image,steps,prompt).sample
                        torch.testing.assert_close(result, torch.full_like(image, 3*i+3))
                        self.assertEqual(engine.allocations, 1 if i < 3 else 2)
                finally:
                    adapter.close()

    def test_vae_graph_reads_new_input_and_options(self):
        def callback(value, scale=2):
            return (value * scale + 1,)
        adapter = CapturedVAECall(callback)
        try:
            for n,scale in ((8,2),(8,2),(4,3),(4,3)):
                value = torch.randn((1,3,n,n),device='cuda')
                actual = adapter(value,scale=scale)[0]
                torch.testing.assert_close(actual,callback(value,scale)[0])
        finally:
            adapter.close()
