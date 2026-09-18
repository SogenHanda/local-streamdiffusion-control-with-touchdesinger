import unittest

import torch
from PIL import Image

from streamdiffusion_bridge.latent_morph import LatentMorph, camera_guide
from streamdiffusion_bridge.config import AppConfig
from streamdiffusion_bridge.engine import StreamDiffusionEngine


class LatentMorphTests(unittest.TestCase):
    def apply(self, morph, value, now, **kw):
        options = dict(strength=1.0, frames=8, motion_threshold=1.0)
        options.update(kw)
        return morph.apply(value, now=now, **options)

    def test_structural_step_is_gradual_monotonic_and_fully_settles(self):
        morph = LatentMorph()
        zero = torch.zeros(1, 4, 8, 8)
        for n in range(8):
            self.apply(morph, zero, n / 30)
        values = [self.apply(morph, zero + 1, n / 30).mean().item() for n in range(8, 18)]
        self.assertLess(values[0], 0.4)
        self.assertEqual(values, sorted(values))
        self.assertAlmostEqual(values[-1], 1.0, places=6)

    def test_changed_camera_region_rejects_trails_while_static_region_morphs(self):
        morph = LatentMorph()
        latent = torch.zeros(1, 4, 16, 16)
        guide = torch.zeros(1, 3, 16, 16)
        for n in range(8):
            self.apply(morph, latent, n / 30, guide=guide)
        changed = guide.clone()
        changed[:, :, :, 8:] = 1
        result = self.apply(morph, latent + 1, 8 / 30, guide=changed)
        self.assertLess(result[0, :, 8, 2].mean().item(), 0.4)
        self.assertGreater(result[0, :, 8, 12].mean().item(), 0.98)

    def test_duration_is_similar_at_different_fps(self):
        responses = []
        for fps in (30, 60, 120):
            morph = LatentMorph()
            for n in range(fps // 2):
                self.apply(morph, torch.zeros(1, 1, 1, 1), n / fps)
            for n in range(fps // 2, round(fps * .6) + 1):
                result = self.apply(morph, torch.ones(1, 1, 1, 1), n / fps)
            responses.append(result.item())
        self.assertLess(max(responses) - min(responses), .08)

    def test_identical_frames_preserve_detail_and_history_owns_storage(self):
        morph = LatentMorph()
        torch.manual_seed(7)
        detail = torch.randn(1, 4, 16, 16)
        for n in range(40):
            result = self.apply(morph, detail, n / 30)
            torch.testing.assert_close(result, detail)
        snapshot = morph.history[-1][1].clone()
        detail.zero_()
        torch.testing.assert_close(morph.history[-1][1], snapshot)

    def test_reset_shape_change_stall_and_disabled_filter_have_no_stale_output(self):
        morph = LatentMorph()
        old = torch.ones(1, 4, 8, 8)
        self.apply(morph, old, 0)
        torch.testing.assert_close(self.apply(morph, old * 0, 1), old * 0)
        changed = torch.ones(1, 4, 4, 4)
        torch.testing.assert_close(self.apply(morph, changed, 1.01), changed)
        torch.testing.assert_close(self.apply(morph, changed * 0, 1.02, strength=0), changed * 0)
        morph.reset()
        torch.testing.assert_close(self.apply(morph, changed, 1.03), changed)
        for n in range(100):
            self.apply(morph, changed, 2 + n / 10000)
        self.assertLessEqual(len(morph.history), 64)

    def test_camera_guide_matches_latent_size_and_constant_edges(self):
        latent = torch.zeros(1, 4, 8, 8)
        guide = camera_guide(Image.new('RGB', (64, 64), 'white'), latent)
        self.assertEqual(tuple(guide.shape), (1, 3, 8, 8))
        torch.testing.assert_close(guide, torch.ones_like(guide))

    def test_maximum_rgb_smoothing_keeps_the_current_frame(self):
        engine = StreamDiffusionEngine(AppConfig(temporal_smoothing=1), lambda _: None)
        engine._previous_raw_output = Image.new('RGB', (8, 8), (110, 110, 110))
        result = engine._smooth_output(Image.new('RGB', (8, 8), (100, 100, 100)))
        self.assertGreater(result.getpixel((0, 0))[0], 100)
        self.assertLessEqual(result.getpixel((0, 0))[0], 105)

    def test_camera_guidance_matches_the_one_frame_delay_of_two_step_inference(self):
        for steps in (1, 2):
            with self.subTest(steps=steps):
                engine = StreamDiffusionEngine(AppConfig(lcm_steps=steps, temporal_feedback=0), lambda _: None)
                observed = []
                def stream(_image):
                    guide = engine._morph_camera_input
                    observed.append(guide.getpixel((0, 0)) if guide is not None else None)
                    return torch.zeros(1, 3, 8, 8)
                engine.stream = stream
                engine._postprocess_image = object()
                engine._warmed_up = True
                engine.process(Image.new('RGB', (64, 64), 'red'))
                engine.process(Image.new('RGB', (64, 64), 'blue'))
                self.assertEqual(observed, [(255, 0, 0), (0, 0, 255)] if steps == 1 else [None, (255, 0, 0)])

    def test_fast_motion_keeps_guide_for_the_next_settling_frame(self):
        engine = StreamDiffusionEngine(AppConfig(latent_morph_strength=1, scene_cut_threshold=.1), lambda _: None)
        engine.last_motion_score = 1
        engine._morph_camera_input = Image.new('RGB', (64, 64), 'white')
        latent = torch.ones(1, 4, 8, 8)
        torch.testing.assert_close(engine._stabilize_generated_latent(latent), latent)
        self.assertIsNotNone(engine._latent_history[-1][2])
