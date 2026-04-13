import Foundation
import Testing

@testable import AnyLanguageModel

@Suite("AnthropicLanguageModelV2", .serialized)
struct AnthropicLanguageModelV2Tests {
    @Test func returnsTextOnlyStep() async throws {
        let transport = MockAnthropicV2Transport()
        transport.enqueueJSON(
            """
            {
              "id": "msg_1",
              "type": "message",
              "role": "assistant",
              "content": [{ "type": "text", "text": "ok" }],
              "model": "claude-test",
              "stop_reason": "end_turn"
            }
            """
        )

        let model = AnthropicLanguageModelV2(
            apiKey: "test-api-key",
            model: "claude-test",
            session: transport.session
        )

        let step = try await model.step(
            transcript: promptOnlyTranscript("say hi"),
            tools: [],
            instructions: nil,
            generating: String.self,
            includeSchemaInPrompt: true,
            options: GenerationOptions()
        )

        #expect(step.finalContent == "ok")
        #expect(step.toolCalls.isEmpty)
        #expect(step.finishReason == .completed)
    }

    @Test func returnsAssistantTextAndToolCallsInSameRound() async throws {
        let transport = MockAnthropicV2Transport()
        transport.enqueueJSON(
            """
            {
              "id": "msg_1",
              "type": "message",
              "role": "assistant",
              "content": [
                { "type": "text", "text": "Let me check." },
                {
                  "type": "tool_use",
                  "id": "toolu_1",
                  "name": "getWeather",
                  "input": { "city": "San Francisco" }
                }
              ],
              "model": "claude-test",
              "stop_reason": "tool_use"
            }
            """
        )

        let model = AnthropicLanguageModelV2(
            apiKey: "test-api-key",
            model: "claude-test",
            session: transport.session
        )

        let step = try await model.step(
            transcript: promptOnlyTranscript("weather?"),
            tools: [WeatherTool()],
            instructions: nil,
            generating: String.self,
            includeSchemaInPrompt: true,
            options: GenerationOptions()
        )

        #expect(step.finalContent == nil)
        #expect(step.toolCalls.count == 1)
        #expect(step.toolCalls.first?.toolName == "getWeather")
        #expect(step.finishReason == .toolUse)

        let responseText = step.assistantResponse?.segments.compactMap { segment -> String? in
            guard case .text(let text) = segment else { return nil }
            return text.content
        }.joined()
        #expect(responseText == "Let me check.")
    }

    @Test func supportsMultipleToolCallsInOneStep() async throws {
        let transport = MockAnthropicV2Transport()
        transport.enqueueJSON(
            """
            {
              "id": "msg_1",
              "type": "message",
              "role": "assistant",
              "content": [
                {
                  "type": "tool_use",
                  "id": "toolu_weather_1",
                  "name": "getWeather",
                  "input": { "city": "San Francisco" }
                },
                {
                  "type": "tool_use",
                  "id": "toolu_weather_2",
                  "name": "getWeather",
                  "input": { "city": "New York" }
                }
              ],
              "model": "claude-test",
              "stop_reason": "tool_use"
            }
            """
        )

        let model = AnthropicLanguageModelV2(
            apiKey: "test-api-key",
            model: "claude-test",
            session: transport.session
        )

        let step = try await model.step(
            transcript: promptOnlyTranscript("compare weather"),
            tools: [WeatherTool()],
            instructions: nil,
            generating: String.self,
            includeSchemaInPrompt: true,
            options: GenerationOptions()
        )

        #expect(step.toolCalls.map(\.id) == ["toolu_weather_1", "toolu_weather_2"])
        #expect(step.finishReason == .toolUse)
    }

    @Test func mapsProviderFinishReasons() async throws {
        let transport = MockAnthropicV2Transport()
        transport.enqueueJSON(
            """
            {
              "id": "msg_1",
              "type": "message",
              "role": "assistant",
              "content": [{ "type": "text", "text": "trimmed" }],
              "model": "claude-test",
              "stop_reason": "max_tokens"
            }
            """
        )

        let model = AnthropicLanguageModelV2(
            apiKey: "test-api-key",
            model: "claude-test",
            session: transport.session
        )

        let step = try await model.step(
            transcript: promptOnlyTranscript("hi"),
            tools: [],
            instructions: nil,
            generating: String.self,
            includeSchemaInPrompt: true,
            options: GenerationOptions()
        )

        #expect(step.finishReason == .maxTokens)
    }

    @Test func preservesPauseTurnAndRefusalFinishReasons() async throws {
        let transport = MockAnthropicV2Transport()
        transport.enqueueJSON(
            """
            {
              "id": "msg_1",
              "type": "message",
              "role": "assistant",
              "content": [{ "type": "text", "text": "paused" }],
              "model": "claude-test",
              "stop_reason": "pause_turn"
            }
            """
        )
        transport.enqueueJSON(
            """
            {
              "id": "msg_2",
              "type": "message",
              "role": "assistant",
              "content": [{ "type": "text", "text": "refused" }],
              "model": "claude-test",
              "stop_reason": "refusal"
            }
            """
        )

        let model = AnthropicLanguageModelV2(
            apiKey: "test-api-key",
            model: "claude-test",
            session: transport.session
        )

        let pauseStep = try await model.step(
            transcript: promptOnlyTranscript("hi"),
            tools: [],
            instructions: nil,
            generating: String.self,
            includeSchemaInPrompt: true,
            options: GenerationOptions()
        )
        #expect(pauseStep.finishReason == .pauseTurn)

        let refusalStep = try await model.step(
            transcript: promptOnlyTranscript("hi"),
            tools: [],
            instructions: nil,
            generating: String.self,
            includeSchemaInPrompt: true,
            options: GenerationOptions()
        )
        #expect(refusalStep.finishReason == .refusal)
    }
}

private func promptOnlyTranscript(_ prompt: String) -> Transcript {
    Transcript(entries: [
        .prompt(
            Transcript.Prompt(
                segments: [.text(.init(content: prompt))]
            )
        )
    ])
}

private final class MockAnthropicV2Transport {
    let session: URLSession

    private var responseQueue: [Data] = []
    private(set) var requests: [URLRequest] = []

    init() {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [MockAnthropicV2URLProtocol.self]
        self.session = URLSession(configuration: config)
        MockAnthropicV2URLProtocol.store = self
    }

    func enqueueJSON(_ json: String) {
        responseQueue.append(Data(json.utf8))
    }

    func dequeueResponse() throws -> Data {
        guard !responseQueue.isEmpty else {
            throw NSError(domain: "MockAnthropicV2Transport", code: 1, userInfo: [NSLocalizedDescriptionKey: "No queued response available."])
        }
        return responseQueue.removeFirst()
    }

    func record(_ request: URLRequest) {
        requests.append(request)
    }
}

private final class MockAnthropicV2URLProtocol: URLProtocol {
    nonisolated(unsafe) static weak var store: MockAnthropicV2Transport?

    override class func canInit(with request: URLRequest) -> Bool {
        request.url?.path.contains("/v1/messages") == true
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

    override func startLoading() {
        do {
            guard let store = Self.store else {
                throw NSError(domain: "MockAnthropicV2URLProtocol", code: 1, userInfo: [NSLocalizedDescriptionKey: "Missing transport store."])
            }

            var capturedRequest = request
            capturedRequest.httpBody = request.httpBody ?? Self.readBody(from: request.httpBodyStream)
            store.record(capturedRequest)
            let data = try store.dequeueResponse()

            let response = HTTPURLResponse(
                url: request.url!,
                statusCode: 200,
                httpVersion: "HTTP/1.1",
                headerFields: ["Content-Type": "application/json"]
            )!

            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: data)
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }

    override func stopLoading() {}

    private static func readBody(from stream: InputStream?) -> Data? {
        guard let stream else { return nil }
        stream.open()
        defer { stream.close() }

        var data = Data()
        let bufferSize = 4096
        let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: bufferSize)
        defer { buffer.deallocate() }

        while stream.hasBytesAvailable {
            let read = stream.read(buffer, maxLength: bufferSize)
            if read > 0 {
                data.append(buffer, count: read)
            } else {
                break
            }
        }

        return data.isEmpty ? nil : data
    }
}
