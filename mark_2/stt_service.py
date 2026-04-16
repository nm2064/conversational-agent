"""
Apple SFSpeechRecognizer STT Service (via pyobjc).
Streams partial transcription results in real-time using on-device recognition.
Runs audio capture + recognition on a dedicated thread with its own NSRunLoop.
"""

import threading
import queue

import AppKit  # Must be imported first to initialize Cocoa runtime
import Speech
import AVFoundation
import Foundation


class AppleSTTService:
    """
    Wraps Apple's SFSpeechRecognizer for real-time streaming transcription.

    All Speech/AVFoundation work runs on a dedicated thread with an NSRunLoop,
    so callbacks fire correctly. The public API is thread-safe via a command queue.
    """

    def __init__(self, locale="en-US"):
        self._locale_id = locale
        self._result_queue = queue.Queue()  # (tag, text) tuples for the consumer
        self._cmd_queue = queue.Queue()     # commands for the worker thread
        self._ready_event = threading.Event()
        self._stop_event = threading.Event()
        self._is_listening = False

        # Objects created on the worker thread
        self._recognizer = None
        self._engine = None
        self._request = None
        self._task = None

        # Start worker thread
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._ready_event.wait(timeout=10)

    def _run_loop(self):
        """Worker thread: create objects and run NSRunLoop to process callbacks."""
        locale = Foundation.NSLocale.alloc().initWithLocaleIdentifier_(self._locale_id)
        self._recognizer = Speech.SFSpeechRecognizer.alloc().initWithLocale_(locale)
        self._engine = AVFoundation.AVAudioEngine.alloc().init()

        on_device = self._recognizer.supportsOnDeviceRecognition()
        print(f"  SFSpeechRecognizer: available={self._recognizer.isAvailable()}, on-device={on_device}", flush=True)
        self._ready_event.set()

        # Run loop: process callbacks and commands
        run_loop = Foundation.NSRunLoop.currentRunLoop()
        while not self._stop_event.is_set():
            # Process any pending commands
            while not self._cmd_queue.empty():
                try:
                    cmd = self._cmd_queue.get_nowait()
                    if cmd == "START":
                        self._do_start()
                    elif cmd == "STOP":
                        self._do_stop()
                except queue.Empty:
                    break

            # Tick the run loop to process audio/recognition callbacks
            run_loop.runUntilDate_(Foundation.NSDate.dateWithTimeIntervalSinceNow_(0.02))

    def _do_start(self):
        if self._is_listening:
            return

        self._request = Speech.SFSpeechAudioBufferRecognitionRequest.alloc().init()
        self._request.setShouldReportPartialResults_(True)
        self._request.setTaskHint_(Speech.SFSpeechRecognitionTaskHintDictation)

        if self._recognizer.supportsOnDeviceRecognition():
            self._request.setRequiresOnDeviceRecognition_(True)

        input_node = self._engine.inputNode()
        fmt = input_node.outputFormatForBus_(0)

        req = self._request  # capture for block

        def tap_handler(buffer, when):
            if req is not None:
                req.appendAudioPCMBuffer_(buffer)

        input_node.installTapOnBus_bufferSize_format_block_(0, 1024, fmt, tap_handler)

        self._engine.prepare()
        ok, err = self._engine.startAndReturnError_(None)
        if not ok:
            self._result_queue.put(("ERROR", f"Audio engine failed: {err}"))
            return

        self._is_listening = True

        result_queue = self._result_queue  # capture for block

        def result_handler(result, error):
            if result is not None:
                text = result.bestTranscription().formattedString()
                if result.isFinal():
                    result_queue.put(("FINAL", text))
                else:
                    result_queue.put(("PARTIAL", text))
            if error is not None:
                err_code = error.code()
                # 216=cancelled, 1110=no speech — expected
                if err_code not in (216, 1110):
                    result_queue.put(("ERROR", str(error)))

        self._task = self._recognizer.recognitionTaskWithRequest_resultHandler_(
            self._request, result_handler
        )

    def _do_stop(self):
        if not self._is_listening:
            return
        self._is_listening = False

        try:
            self._engine.stop()
            self._engine.inputNode().removeTapOnBus_(0)
        except Exception:
            pass

        if self._request is not None:
            self._request.endAudio()
            self._request = None

        if self._task is not None:
            self._task.cancel()
            self._task = None

    # ── Public API (called from main thread) ──────────────────────────

    def start_listening(self):
        # Clear stale results
        while not self._result_queue.empty():
            try:
                self._result_queue.get_nowait()
            except queue.Empty:
                break
        self._cmd_queue.put("START")

    def stop_listening(self):
        self._cmd_queue.put("STOP")

    def get_latest(self, timeout=0.03):
        """
        Drain the queue and return the latest result.
        Returns: (tag, text) where tag is "PARTIAL", "FINAL", or "ERROR"
                 or None if nothing new.
        """
        latest = None
        try:
            msg = self._result_queue.get(timeout=timeout)
            latest = msg
        except queue.Empty:
            pass

        # Drain remaining without blocking
        while True:
            try:
                msg = self._result_queue.get_nowait()
                # Prefer FINAL over PARTIAL
                if latest is None or msg[0] == "FINAL":
                    latest = msg
                elif msg[0] == "PARTIAL":
                    latest = msg
            except queue.Empty:
                break

        return latest

    @property
    def is_listening(self):
        return self._is_listening

    def shutdown(self):
        self._do_stop()
        self._stop_event.set()
