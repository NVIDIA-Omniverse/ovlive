# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immediate public-API failure witnesses; real workflows live in reference tests."""
from dataclasses import dataclass
from enum import IntEnum
import importlib.util
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class Result(IntEnum):
    OK = 0
    TIMEOUT = 9
    BUSY = 14


class LiveError(Exception):
    def __init__(self, result):
        self.result = result


@dataclass
class Created:
    session_id: str = 'confirmed-session'
    uri: str = 'live://example.test/confirmed-session'


@dataclass
class Error:
    result: int = 0


@dataclass
class Status:
    revision: int
    state: int = 2


class HeadlessSampleTests(unittest.TestCase):
    def setUp(self):
        self.api = SimpleNamespace(Result=Result, LiveError=LiveError)
        source = Path(__file__).resolve().parents[2] / 'source/bindings/python/omni/usd/live/sample.py'
        spec = importlib.util.spec_from_file_location('sample_contract', source)
        self.sample = importlib.util.module_from_spec(spec)
        tf = SimpleNamespace(ErrorException=type('ErrorException', (Exception,), {}))
        with patch.dict(sys.modules, {'pxr': SimpleNamespace(Plug=None, Sdf=None, Tf=tf),
                                    'omni': SimpleNamespace(), 'omni.usd': SimpleNamespace(live=self.api)}):
            spec.loader.exec_module(self.sample)

    def test_timed_out_session_retains_same_request_and_forbids_second_create(self):
        request = Mock(wait=Mock(return_value=Result.TIMEOUT))
        self.api.create_session_with_callback = Mock(return_value=request)
        owner = self.sample.Participant.__new__(self.sample.Participant)
        owner.session_request = owner.created = None
        owner.endpoint = object()
        owner.record = Mock()
        with self.assertRaises(LiveError) as caught:
            owner.create('base', 'name')
        self.assertEqual(caught.exception.result, Result.TIMEOUT)
        self.assertIs(owner.session_request, request)
        with self.assertRaises(ValueError):
            owner.create('base', 'name')
        self.api.create_session_with_callback.assert_called_once()
        request.wait.return_value = Result.OK
        request.outcome = (Result.OK, Created())
        request.error = Error()
        result = owner.execute({'operation': 'session-outcome'})
        self.assertEqual(result['session']['session_id'], 'confirmed-session')
        self.assertEqual(owner.created, Created())
        self.api.create_session_with_callback.assert_called_once()

    def test_close_cannot_retire_owners_while_session_request_is_unsettled(self):
        owner = self.sample.Participant.__new__(self.sample.Participant)
        owner.closed = False
        owner.host = Mock(call=lambda action: action())
        owner.session_request = Mock(wait=Mock(return_value=Result.TIMEOUT))
        owner.context = Mock()
        with self.assertRaises(LiveError):
            owner.close()
        owner.context.close.assert_not_called()

    def test_log_retirement_timeout_retries_without_releasing_subscriptions_twice(self):
        owner = self.sample.Participant.__new__(self.sample.Participant)
        owner.closed = owner.dirty = owner.export_required = owner.exported = False
        owner.session_request = owner.layer = owner.creation = owner.editor = owner.joined = None
        owner.host = Mock(call=lambda action: action())
        subscriptions = [Mock(close=Mock(return_value=Result.OK)) for _ in range(3)]
        owner.status_observer, owner.auth_observer, owner.diagnostics = subscriptions
        report = dict(written_bytes=1, dropped=0, error=None)
        owner.log = Mock(close=Mock(side_effect=[TimeoutError(), report]))
        owner.context = Mock(close=Mock(return_value=Result.OK))
        owner.endpoint = Mock(close=Mock(return_value=Result.OK))
        owner.runtime = Mock(close=Mock(return_value=Result.OK))
        with self.assertRaises(TimeoutError):
            owner.close()
        owner.context.close.assert_not_called()
        self.assertEqual(owner.close(), dict(result=0, closed=True, logging=report))
        self.assertEqual(owner.close(), owner.final_reply)
        for subscription in subscriptions:
            subscription.close.assert_called_once()
        owner.host.close.assert_called_once()
        owner.runtime.close.assert_called_once()

    def test_endpoint_and_runtime_close_retry_only_while_busy(self):
        close = Mock(side_effect=[Result.BUSY, Result.BUSY, Result.OK])
        self.sample.Participant.closed_when_released(close)
        self.assertEqual(close.call_count, 3)
        close = Mock(side_effect=[Result.BUSY, Result.TIMEOUT, Result.OK])
        with self.assertRaises(LiveError) as raised:
            self.sample.Participant.closed_when_released(close)
        self.assertEqual((raised.exception.result, close.call_count), (Result.TIMEOUT, 2))

    def test_broken_reply_channel_returns_control_to_preservation(self):
        stream = Mock()
        stream.write.side_effect = BrokenPipeError()
        with patch.object(sys, 'stdout', stream):
            self.assertFalse(self.sample.reply({'type': 'reply', 'result': 19}))
            replacement = sys.stdout
        replacement.close()

    def test_status_revisions_progress_while_log_writer_is_blocked_and_full(self):
        entered, release = threading.Event(), threading.Event()
        def write(text):
            entered.set()
            if not release.wait(5):
                raise AssertionError('test writer release watchdog')
            return len(text)
        stream = Mock(write=write)
        log = self.sample.EventLog(stream, max_records=1)
        owner = self.sample.Participant.__new__(self.sample.Participant)
        owner.log, owner.changed, owner.statuses = log, threading.Condition(), {}
        try:
            log.record('diagnostic')
            self.assertTrue(entered.wait(5), 'writer did not enter instrumented boundary')
            for revision in (1, 2):
                owner.status(SimpleNamespace(endpoint=1, session=0, uri='live://test/s/l', status=Status(revision)))
                self.assertEqual(owner.statuses['live://test/s/l'].revision, revision)
            self.assertFalse(release.is_set())
            self.assertEqual(log.dropped, 2)
        finally:
            release.set()
            report = log.close()
        self.assertIsNone(report['error'])
        self.assertEqual(report['dropped'], 2)

    def test_log_io_failure_and_file_budget_are_explicit(self):
        broken = Mock()
        broken.write.side_effect = OSError('secret exception text is not retained')
        log = self.sample.EventLog(broken)
        log.record('diagnostic')
        self.assertTrue(log.done.wait(5))
        log.record('status')
        report = log.close()
        self.assertEqual((report['error'], report['dropped']), ('OSError', 2))
        stream = Mock()
        bounded = self.sample.EventLog(stream, max_file_bytes=1)
        bounded.record('diagnostic')
        report = bounded.close()
        self.assertEqual((report['error'], report['dropped'], report['written_bytes']), (None, 1, 0))
        stream.write.assert_not_called()


if __name__ == '__main__':
    unittest.main()
