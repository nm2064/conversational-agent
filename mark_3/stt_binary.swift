import Foundation
import Speech
import AVFoundation
import Darwin

let sampleRate: Double = 16000
let logPath = "/tmp/stt_binary.log"
let socketPath = "/tmp/stt.sock"

func logLine(_ msg: String) {
    let line = "[\(Date())] \(msg)\n"
    if let data = line.data(using: .utf8) {
        if let fh = FileHandle(forWritingAtPath: logPath) {
            fh.seekToEndOfFile()
            fh.write(data)
            try? fh.close()
        } else {
            try? data.write(to: URL(fileURLWithPath: logPath))
        }
    }
}

try? "".write(to: URL(fileURLWithPath: logPath), atomically: true, encoding: .utf8)
logLine("daemon start pid=\(getpid())")

// --- Auth ---
let authSem = DispatchSemaphore(value: 0)
var authStatus: SFSpeechRecognizerAuthorizationStatus = .notDetermined
SFSpeechRecognizer.requestAuthorization { status in
    authStatus = status
    authSem.signal()
}
authSem.wait()
logLine("auth status=\(authStatus.rawValue)")
guard authStatus == .authorized else {
    logLine("auth denied — daemon exiting")
    exit(1)
}

guard let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US")), recognizer.isAvailable else {
    logLine("SFSpeechRecognizer unavailable")
    exit(1)
}
let callbackQueue = OperationQueue()
callbackQueue.name = "stt.callback"
recognizer.queue = callbackQueue
logLine("recognizer ready onDevice=\(recognizer.supportsOnDeviceRecognition)")

guard let pcmFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                    sampleRate: sampleRate,
                                    channels: 1,
                                    interleaved: false) else {
    logLine("failed to create AVAudioFormat")
    exit(1)
}

// --- Unix socket server ---
unlink(socketPath)
let serverFd = socket(AF_UNIX, SOCK_STREAM, 0)
guard serverFd >= 0 else { logLine("socket() failed"); exit(1) }

var addr = sockaddr_un()
addr.sun_family = sa_family_t(AF_UNIX)
let pathBytes = Array(socketPath.utf8)
withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
    ptr.withMemoryRebound(to: CChar.self, capacity: 104) { cptr in
        for i in 0..<pathBytes.count where i < 103 { cptr[i] = CChar(pathBytes[i]) }
        cptr[pathBytes.count] = 0
    }
}
let addrLen = socklen_t(MemoryLayout<sockaddr_un>.size)
let bindResult = withUnsafePointer(to: &addr) { p in
    p.withMemoryRebound(to: sockaddr.self, capacity: 1) { sp in
        bind(serverFd, sp, addrLen)
    }
}
guard bindResult == 0 else { logLine("bind failed errno=\(errno)"); exit(1) }
guard listen(serverFd, 4) == 0 else { logLine("listen failed"); exit(1) }
logLine("listening on \(socketPath)")

func readExact(_ fd: Int32, _ n: Int) -> Data? {
    var data = Data(count: n)
    var got = 0
    while got < n {
        let r = data.withUnsafeMutableBytes { ptr -> Int in
            read(fd, ptr.baseAddress!.advanced(by: got), n - got)
        }
        if r <= 0 { return nil }
        got += r
    }
    return data
}

func writeAll(_ fd: Int32, _ data: Data) {
    var sent = 0
    let n = data.count
    data.withUnsafeBytes { ptr in
        while sent < n {
            let r = write(fd, ptr.baseAddress!.advanced(by: sent), n - sent)
            if r <= 0 { break }
            sent += r
        }
    }
}

// Streaming protocol per connection:
//   repeat:
//     read uint32 LE N
//     if N == 0 -> end of audio, wait for final result
//     else read N bytes float32 mono 16kHz PCM, append to recognizer
//   send uint32 LE textLen, then textLen UTF-8 bytes; close.
func handleClient(_ clientFd: Int32) {
    defer { close(clientFd) }

    let request = SFSpeechAudioBufferRecognitionRequest()
    request.shouldReportPartialResults = true
    if recognizer.supportsOnDeviceRecognition {
        request.requiresOnDeviceRecognition = true
    }

    var finalText = ""
    var lastPartial = ""
    var done = false

    let task = recognizer.recognitionTask(with: request) { result, error in
        if let result = result {
            let text = result.bestTranscription.formattedString
            if !text.isEmpty { lastPartial = text }
            if result.isFinal {
                finalText = text.isEmpty ? lastPartial : text
                done = true
            }
        }
        if error != nil { done = true }
    }
    _ = task

    var framesAppended = 0
    streamLoop: while true {
        guard let lenData = readExact(clientFd, 4) else { break }
        let chunkLen = Int(lenData.withUnsafeBytes { $0.load(as: UInt32.self) })
        if chunkLen == 0 { break streamLoop }  // client signalled end-of-audio
        guard chunkLen > 0, chunkLen % 4 == 0, chunkLen < 10_000_000 else { break }
        guard let audioData = readExact(clientFd, chunkLen) else { break }

        let frameCount = AVAudioFrameCount(chunkLen / 4)
        guard let buf = AVAudioPCMBuffer(pcmFormat: pcmFormat, frameCapacity: frameCount) else { continue }
        buf.frameLength = frameCount
        audioData.withUnsafeBytes { raw in
            guard let src = raw.baseAddress?.assumingMemoryBound(to: Float.self),
                  let dst = buf.floatChannelData?[0] else { return }
            dst.update(from: src, count: Int(frameCount))
        }
        request.append(buf)
        framesAppended += Int(frameCount)
    }

    request.endAudio()
    let deadline = Date(timeIntervalSinceNow: 10)
    while !done && Date() < deadline {
        RunLoop.current.run(mode: .default, before: Date(timeIntervalSinceNow: 0.02))
    }
    let text = done ? finalText : lastPartial
    logLine("served frames=\(framesAppended) text=\"\(text)\"")

    let textData = text.data(using: .utf8) ?? Data()
    var outLen = UInt32(textData.count).littleEndian
    let lenBytes = withUnsafeBytes(of: &outLen) { Data($0) }
    writeAll(clientFd, lenBytes)
    writeAll(clientFd, textData)
}

let serverQueue = DispatchQueue(label: "stt.server")
serverQueue.async {
    while true {
        let clientFd = accept(serverFd, nil, nil)
        if clientFd < 0 { continue }
        handleClient(clientFd)
    }
}

signal(SIGTERM) { _ in
    logLine("received SIGTERM, exiting")
    exit(0)
}
signal(SIGINT) { _ in
    logLine("received SIGINT, exiting")
    exit(0)
}
signal(SIGPIPE, SIG_IGN)

logLine("entering RunLoop.main.run()")
RunLoop.main.run()
logLine("RunLoop.main.run() returned — should never happen")
