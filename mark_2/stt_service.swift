import Foundation
import Speech
import AVFoundation

// Disable stdout buffering so Python gets output immediately
setbuf(stdout, nil)
setbuf(stderr, nil)

class STTService: NSObject {
    let speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))!
    let audioEngine = AVAudioEngine()
    var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    var recognitionTask: SFSpeechRecognitionTask?
    var isListening = false

    func requestAuthorization(completion: @escaping (Bool) -> Void) {
        SFSpeechRecognizer.requestAuthorization { status in
            switch status {
            case .authorized:
                completion(true)
            case .denied:
                fputs("Speech recognition permission denied. Go to System Settings > Privacy & Security > Speech Recognition.\n", stderr)
                completion(false)
            case .restricted:
                fputs("Speech recognition is restricted on this device.\n", stderr)
                completion(false)
            case .notDetermined:
                fputs("Speech recognition permission not determined.\n", stderr)
                completion(false)
            @unknown default:
                completion(false)
            }
        }
    }

    func startListening() {
        guard !isListening else { return }

        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let request = recognitionRequest else {
            print("ERROR:Failed to create recognition request")
            return
        }
        request.shouldReportPartialResults = true
        request.taskHint = .dictation

        // Use on-device recognition if available (lower latency)
        if #available(macOS 13, *) {
            if speechRecognizer.supportsOnDeviceRecognition {
                request.requiresOnDeviceRecognition = true
            }
        }

        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
            request.append(buffer)
        }

        do {
            audioEngine.prepare()
            try audioEngine.start()
        } catch {
            print("ERROR:Audio engine failed: \(error.localizedDescription)")
            return
        }

        isListening = true

        recognitionTask = speechRecognizer.recognitionTask(with: request) { [weak self] result, error in
            guard let self = self else { return }

            if let result = result {
                let text = result.bestTranscription.formattedString
                print("PARTIAL:\(text)")

                if result.isFinal {
                    print("FINAL:\(text)")
                    self.doStop()
                }
            }

            if let error = error {
                let nsError = error as NSError
                // Code 216 = cancelled, 1110 = no speech detected — both are expected
                if nsError.code != 216 && nsError.code != 1110 {
                    print("ERROR:\(error.localizedDescription)")
                }
                if self.isListening {
                    self.doStop()
                }
            }
        }

        print("LISTENING")
    }

    func doStop() {
        guard isListening else { return }
        isListening = false

        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        recognitionRequest = nil
        recognitionTask = nil

        print("STOPPED")
    }
}

// ── Main ──────────────────────────────────────────────────────────────

let service = STTService()

let semaphore = DispatchSemaphore(value: 0)
var authorized = false

service.requestAuthorization { ok in
    authorized = ok
    semaphore.signal()
}
semaphore.wait()

if !authorized {
    print("ERROR:Not authorized")
    exit(1)
}

print("READY")

// Read commands from stdin on a background thread
DispatchQueue.global(qos: .userInteractive).async {
    while let line = readLine() {
        let cmd = line.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
        DispatchQueue.main.async {
            switch cmd {
            case "START":
                service.startListening()
            case "STOP":
                service.doStop()
            case "QUIT":
                service.doStop()
                exit(0)
            default:
                break
            }
        }
    }
    // stdin closed — exit
    DispatchQueue.main.async { exit(0) }
}

// Run the main loop (required for AVAudioEngine and Speech callbacks)
RunLoop.main.run()
