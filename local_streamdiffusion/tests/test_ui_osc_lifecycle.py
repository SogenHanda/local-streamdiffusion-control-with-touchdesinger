from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from streamdiffusion_bridge.ui import BACKEND_LABELS, DiffusionApp


class OSCStartStopLifecycleTests(unittest.TestCase):
    def test_monitor_keeps_python_owned_graph_setting_when_collecting_config(self) -> None:
        def variable(value):
            return SimpleNamespace(get=lambda *args: value)
        values = {
            "resolution": "512 × 512", "model_path": "models/realistic-vision-v5.1",
            "lcm_lora_path": "models/lcm-lora-sdv1-5", "spout_input": "TD_Camera",
            "spout_output": "AI_Output", "denoise": 22, "seed": 1,
            "target_fps": 120, "spout_sample_fps": 60,
            "acceleration_backend": next(k for k,v in BACKEND_LABELS.items() if v == "tensorrt"),
            "use_lcm_lora": True, "lcm_steps": "2", "tiny_vae": True,
            "temporal_feedback": .03, "temporal_smoothing": 1,
            "latent_morph_strength": 1, "latent_history_frames": 8,
            "scene_cut_threshold": 1, "flip_input": False, "flip_output": False,
            "offline": True,
        }
        app = SimpleNamespace(**{k+"_var": variable(v) for k,v in values.items()},
                              _selected_model_profile=lambda: None,
                              prompt_text=variable("a test prompt"))
        for enabled in (False, True):
            app._tensorrt_cuda_graph = enabled
            self.assertEqual(DiffusionApp._collect_config(app).tensorrt_cuda_graph, enabled)

    def test_osc_stop_schedules_complete_application_close(self) -> None:
        app = SimpleNamespace(
            _generation_desired=True,
            _append_log=Mock(),
            after_idle=Mock(),
            _on_close=Mock(),
        )

        DiffusionApp._stop_from_osc(app)

        self.assertFalse(app._generation_desired)
        app.after_idle.assert_called_once_with(app._on_close)

    def test_live_state_is_resent_and_dedupe_cache_is_cleared(self) -> None:
        worker = SimpleNamespace(
            running=True,
            update_prompt=Mock(),
            update_live_settings=Mock(),
        )
        prompt_text = Mock()
        prompt_text.get.return_value = "organic chair\n"
        settings = {"denoise_index": 12, "temporal_smoothing": 0.7}
        app = SimpleNamespace(
            worker=worker,
            prompt_text=prompt_text,
            _current_live_settings=Mock(return_value=settings),
            _osc_last_live_values={"denoise_index": 12},
            _append_log=Mock(),
        )

        DiffusionApp._sync_live_state_to_worker(app)

        worker.update_prompt.assert_called_once_with("organic chair")
        worker.update_live_settings.assert_called_once_with(settings)
        self.assertEqual(app._osc_last_live_values, {})

if __name__ == "__main__":
    unittest.main()
