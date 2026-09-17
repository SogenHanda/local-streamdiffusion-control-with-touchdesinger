from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from streamdiffusion_bridge.ui import DiffusionApp


class OSCStartStopLifecycleTests(unittest.TestCase):
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
