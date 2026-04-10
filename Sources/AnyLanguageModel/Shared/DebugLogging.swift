import Foundation

enum DebugLogging {
    private static let environment = ProcessInfo.processInfo.environment

    static var isAnthropicIODebugEnabled: Bool {
        guard let rawValue = environment["DEBUG_ANTHROPIC_IO"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        else {
            return false
        }

        switch rawValue {
        case "1", "true", "yes", "on":
            return true
        default:
            return false
        }
    }

    static func log(_ message: @autoclosure () -> String) {
        guard isAnthropicIODebugEnabled else { return }
        writeToStderr("[AnyLanguageModel DEBUG] \(message())\n")
    }

    static func writeArtifact(
        prefix: String,
        fileExtension: String,
        data: Data
    ) {
        guard isAnthropicIODebugEnabled else { return }

        let directoryURL = debugDirectoryURL()
        do {
            try FileManager.default.createDirectory(
                at: directoryURL,
                withIntermediateDirectories: true
            )
            let fileURL = directoryURL.appendingPathComponent("\(prefix).\(fileExtension)")
            try data.write(to: fileURL, options: .atomic)
            log("Wrote debug artifact: \(fileURL.path)")
        } catch {
            writeToStderr("[AnyLanguageModel DEBUG] Failed to write artifact \(prefix).\(fileExtension): \(error)\n")
        }
    }

    static func writeArtifact(
        prefix: String,
        fileExtension: String,
        string: String
    ) {
        writeArtifact(
            prefix: prefix,
            fileExtension: fileExtension,
            data: Data(string.utf8)
        )
    }

    private static func debugDirectoryURL() -> URL {
        if let customDirectory = environment["DEBUG_ANTHROPIC_IO_DIR"]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !customDirectory.isEmpty {
            return URL(fileURLWithPath: customDirectory, isDirectory: true)
        }

        return FileManager.default.temporaryDirectory
            .appendingPathComponent("swiftcodeagent-anthropic-debug", isDirectory: true)
    }

    private static func writeToStderr(_ string: String) {
        guard let data = string.data(using: .utf8) else { return }
        FileHandle.standardError.write(data)
    }
}
