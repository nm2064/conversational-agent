import Foundation
import Speech

setbuf(stdout, nil)
print("Requesting Speech Recognition permission...")
print("A dialog should appear. Please click 'Allow'.")

let semaphore = DispatchSemaphore(value: 0)

SFSpeechRecognizer.requestAuthorization { status in
    switch status {
    case .authorized:
        print("SUCCESS: Speech recognition authorized!")
    case .denied:
        print("DENIED: Go to System Settings > Privacy & Security > Speech Recognition")
    case .restricted:
        print("RESTRICTED: Speech recognition is restricted on this device")
    case .notDetermined:
        print("NOT_DETERMINED: Permission was not determined")
    @unknown default:
        print("UNKNOWN: status = \(status.rawValue)")
    }
    semaphore.signal()
}

semaphore.wait()
