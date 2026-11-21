import AppKit
import CryptoKit
import WebKit

@main
@MainActor
final class TwitchShorts: NSObject, NSApplicationDelegate, WKNavigationDelegate, WKUIDelegate {
    private struct Health: Decodable {
        let workspace: String
        let busy: Bool
    }

    private let studio = URL(string: "http://127.0.0.1:8765/")!
    private let fileManager = FileManager.default
    private var process: Process?
    private var startup: Task<Void, Never>?
    private var window: NSWindow!
    private var webView: WKWebView!
    private var startupView: NSStackView!
    private var status: NSTextField!

    private var workspace: URL {
        let path = Bundle.main.object(forInfoDictionaryKey: "TwitchWorkspace") as? String ?? ""
        return URL(fileURLWithPath: path, isDirectory: true).resolvingSymlinksInPath().standardizedFileURL
    }

    private var logFile: URL {
        fileManager.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Logs/Twitch Shorts/server.log")
    }

    static func main() {
        let app = NSApplication.shared
        let delegate = TwitchShorts()
        app.delegate = delegate
        app.setActivationPolicy(.regular)
        withExtendedLifetime(delegate) { app.run() }
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        installMenu()
        webView = WKWebView()
        webView.navigationDelegate = self
        webView.uiDelegate = self
        webView.autoresizingMask = [.width, .height]
        webView.isHidden = true

        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1240, height: 820),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable],
                          backing: .buffered, defer: false)
        window.title = "Twitch Shorts"
        window.minSize = NSSize(width: 700, height: 530)
        window.isReleasedWhenClosed = false
        window.appearance = NSAppearance(named: .darkAqua)
        window.center()
        window.setFrameAutosaveName("Twitch Shorts")
        let root = window.contentView!
        webView.frame = root.bounds
        root.addSubview(webView)

        let icon = NSImageView(image: NSApp.applicationIconImage)
        icon.widthAnchor.constraint(equalToConstant: 92).isActive = true
        icon.heightAnchor.constraint(equalToConstant: 92).isActive = true
        status = NSTextField(wrappingLabelWithString: "Запускаю студию…")
        status.alignment = .center
        status.maximumNumberOfLines = 0
        status.widthAnchor.constraint(equalToConstant: 420).isActive = true
        let retry = NSButton(title: "Повторить подключение", target: self, action: #selector(connect))
        startupView = NSStackView(views: [icon, status, retry])
        startupView.orientation = .vertical
        startupView.spacing = 16
        startupView.translatesAutoresizingMaskIntoConstraints = false
        root.addSubview(startupView)
        NSLayoutConstraint.activate([
            startupView.centerXAnchor.constraint(equalTo: root.centerXAnchor),
            startupView.centerYAnchor.constraint(equalTo: root.centerYAnchor),
        ])
        showWindow()
        connect()
    }

    private func showWindow() {
        if window.isMiniaturized { window.deminiaturize(nil) }
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        showWindow()
        return true
    }

    private func installMenu() {
        let main = NSMenu()
        let app = NSMenuItem()
        app.submenu = NSMenu()
        app.submenu?.addItem(withTitle: "Скрыть Twitch Shorts", action: #selector(NSApplication.hide(_:)), keyEquivalent: "h")
        app.submenu?.addItem(.separator())
        app.submenu?.addItem(withTitle: "Выйти из Twitch Shorts", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        main.addItem(app)
        let edit = NSMenuItem()
        edit.submenu = NSMenu(title: "Правка")
        for (title, action, shortcut) in [
            ("Отменить", "undo:", "z"), ("Вырезать", "cut:", "x"),
            ("Копировать", "copy:", "c"), ("Вставить", "paste:", "v"), ("Выбрать всё", "selectAll:", "a"),
        ] {
            edit.submenu?.addItem(withTitle: title, action: Selector(action), keyEquivalent: shortcut)
        }
        main.addItem(edit)
        let view = NSMenuItem()
        view.submenu = NSMenu(title: "Вид")
        view.submenu?.addItem(withTitle: "Обновить студию", action: #selector(connect), keyEquivalent: "r").target = self
        main.addItem(view)
        NSApp.mainMenu = main
    }

    private func checkHealth() async -> (responded: Bool, health: Health?) {
        let url = studio.appendingPathComponent("api/health")
        let request = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 1.5)
        guard let (data, response) = try? await URLSession.shared.data(for: request),
              let http = response as? HTTPURLResponse else { return (false, nil) }
        guard http.url == url, http.statusCode == 200,
              let health = try? JSONDecoder().decode(Health.self, from: data) else { return (true, nil) }
        let expected = SHA256.hash(data: Data(workspace.path.utf8)).map { String(format: "%02x", $0) }.joined()
        return (true, health.workspace == expected ? health : nil)
    }

    @objc private func connect() {
        guard startup == nil else { return }
        webView.isHidden = true
        startupView.isHidden = false
        status.stringValue = "Запускаю студию…"
        startup = Task {
            defer { startup = nil }
            let server = await checkHealth()
            if server.health != nil {
                loadStudio()
                return
            }
            if server.responded {
                status.stringValue = "Порт 8765 занят. Закройте другой процесс, использующий этот порт, и повторите подключение."
                return
            }
            do {
                if process?.isRunning != true { try startServer() }
            } catch {
                status.stringValue = "Не удалось запустить студию.\n" + error.localizedDescription
                return
            }
            for _ in 0..<150 {
                if await checkHealth().health != nil {
                    loadStudio()
                    return
                }
                if process?.isRunning != true { break }
                try? await Task.sleep(nanoseconds: 400_000_000)
            }
            status.stringValue = "Студия не ответила. Журнал запуска находится в:\n" + logFile.path
        }
    }

    private func startServer() throws {
        let launcher = workspace.appendingPathComponent("twitch.sh")
        guard fileManager.isExecutableFile(atPath: launcher.path) else {
            throw CocoaError(.fileNoSuchFile, userInfo: [
                NSLocalizedDescriptionKey: "Проверьте папку проекта: " + workspace.path,
            ])
        }
        try fileManager.createDirectory(at: logFile.deletingLastPathComponent(), withIntermediateDirectories: true)
        if !fileManager.fileExists(atPath: logFile.path) {
            fileManager.createFile(atPath: logFile.path, contents: nil, attributes: [.posixPermissions: 0o600])
        }
        let log = try FileHandle(forWritingTo: logFile)
        defer { try? log.close() }
        try log.seekToEnd()
        let child = Process()
        child.executableURL = launcher
        child.currentDirectoryURL = workspace
        var environment = ProcessInfo.processInfo.environment
        environment["PATH"] = "/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
        child.environment = environment
        child.standardOutput = log
        child.standardError = log
        try child.run()
        process = child
    }

    private func loadStudio() {
        webView.load(URLRequest(url: studio, cachePolicy: .reloadIgnoringLocalCacheData))
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        startupView.isHidden = true
        webView.isHidden = false
    }

    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        status.stringValue = "Не удалось открыть студию. Повторите подключение."
    }

    func webView(_ webView: WKWebView, decidePolicyFor navigation: WKNavigationAction,
                 decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        let url = navigation.request.url
        if url?.host == studio.host, url?.port == studio.port, url?.scheme == studio.scheme {
            decisionHandler(.allow)
        } else {
            decisionHandler(.cancel)
            if navigation.navigationType == .linkActivated, let url, ["https", "http"].contains(url.scheme ?? "") {
                NSWorkspace.shared.open(url)
            }
        }
    }

    func webView(_ webView: WKWebView, runOpenPanelWith parameters: WKOpenPanelParameters,
                 initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping ([URL]?) -> Void) {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = parameters.allowsMultipleSelection
        panel.beginSheetModal(for: window) { response in
            completionHandler(response == .OK ? panel.urls : nil)
        }
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard process?.isRunning == true else { return .terminateNow }
        Task {
            if await checkHealth().health?.busy == true {
                let alert = NSAlert()
                alert.messageText = "Студия обрабатывает видео"
                alert.informativeText = "Можно закрыть окно. Обработка продолжится, а студию можно снова открыть из Dock."
                alert.addButton(withTitle: "Закрыть окно")
                alert.addButton(withTitle: "Продолжить работу")
                if alert.runModal() == .alertFirstButtonReturn { window.close() }
                sender.reply(toApplicationShouldTerminate: false)
                return
            }
            process?.terminate()
            sender.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }
}
