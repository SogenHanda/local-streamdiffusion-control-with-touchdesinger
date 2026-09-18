import threading
import unittest
from unittest.mock import Mock, patch

from streamdiffusion_bridge.timing import FrameWaiter


class FrameWaiterTests(unittest.TestCase):
    def test_fallback_obeys_stop_and_bounds_wait(self):
        with patch('streamdiffusion_bridge.timing.sys.platform', 'test'):
            waiter = FrameWaiter()
        stop = Mock()
        stop.is_set.return_value = False
        waiter.wait(1.0,stop)
        stop.wait.assert_called_once_with(.05)
        stop.is_set.return_value = True
        waiter.wait(1.0,stop)
        self.assertEqual(stop.wait.call_count,1)

    def test_exception_releases_timer_once(self):
        with patch('streamdiffusion_bridge.timing.sys.platform', 'test'):
            waiter = FrameWaiter()
        kernel = Mock()
        waiter._kernel, waiter._timer = kernel, 123
        with self.assertRaises(ValueError):
            with waiter:
                raise ValueError('receiver failed')
        waiter.close()
        kernel.CloseHandle.assert_called_once_with(123)

    def test_failed_timer_uses_interruptible_fallback(self):
        with patch('streamdiffusion_bridge.timing.sys.platform','test'):
            waiter=FrameWaiter()
        kernel=Mock()
        kernel.SetWaitableTimer.return_value=False
        waiter._kernel,waiter._timer=kernel,123
        stop=threading.Event()
        stop.wait=Mock()
        waiter.wait(.01,stop)
        kernel.CloseHandle.assert_called_once_with(123)
        stop.wait.assert_called_once_with(.01)
