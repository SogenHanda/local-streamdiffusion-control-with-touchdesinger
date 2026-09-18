import unittest

import numpy as np
import torch

from streamdiffusion_bridge.image_processing import decoded_tensor_to_pil


class ImageConversionTests(unittest.TestCase):
    def test_matches_original_rounding_for_all_finite_half_values(self):
        # Every finite half value, including clipping boundaries and rounding
        # ties, must produce the same 8-bit value as the original NumPy path.
        halves = np.arange(65536, dtype=np.uint16).view(np.float16)
        halves = halves[np.isfinite(halves)].copy()
        tensor = torch.from_numpy(halves).reshape(1, 1, 1, -1)
        for device in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
            with self.subTest(device=device):
                value = tensor.to(device)
                normalized = (value / 2 + 0.5).clamp(0, 1)
                expected = (normalized.cpu().permute(0,2,3,1).float().numpy()*255).round().astype(np.uint8)
                actual = np.asarray(decoded_tensor_to_pil(value)[0])
                np.testing.assert_array_equal(actual, expected[0,:,:,0])

    def test_rgb_batch_preserves_channel_and_frame_order(self):
        values = torch.tensor([[-1., 0., 1.], [1., -1., 0.]]).reshape(2,3,1,1)
        actual = decoded_tensor_to_pil(values)
        self.assertEqual(actual[0].getpixel((0,0)), (0,128,255))
        self.assertEqual(actual[1].getpixel((0,0)), (255,0,128))
