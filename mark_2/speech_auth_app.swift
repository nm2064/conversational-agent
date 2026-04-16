import Cocoa
import Speech

class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        SFSpeechRecognizer.requestAuthorization { status in
            DispatchQueue.main.async {
                switch status {
                case .authorized:
                    let alert = NSAlert()
                    alert.messageText = "Speech Recognition Authorized!"
                    alert.informativeText = "You can now close this and run the voice assistant."
                    alert.runModal()
                case .denied:
                    let alert = NSAlert()
                    alert.messageText = "Speech Recognition Denied"
                    alert.informativeText = "Please go to System Settings > Privacy & Security > Speech Recognition and enable it."
                    alert.runModal()
                default:
                    break
                }
                NSApp.terminate(nil)
            }
        }
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.run()
