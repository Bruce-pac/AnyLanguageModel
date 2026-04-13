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

    @Test func sendsInstructionsAsTopLevelSystemPrompt() async throws {
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

        _ = try await model.step(
            transcript: promptOnlyTranscript("say hi"),
            tools: [],
            instructions: .init(
                segments: [.text(.init(content: "You are a careful coding agent."))],
                toolDefinitions: []
            ),
            generating: String.self,
            includeSchemaInPrompt: true,
            options: GenerationOptions()
        )

        let payload = try requestPayload(from: transport)
        #expect(payload["system"] as? String == "You are a careful coding agent.")

        let messages = try #require(payload["messages"] as? [[String: Any]])
        #expect(messages.count == 1)
        #expect(messages[0]["role"] as? String == "user")

        let content = try #require(messages[0]["content"] as? [[String: Any]])
        #expect(content.count == 1)
        #expect(content[0]["type"] as? String == "text")
        #expect(content[0]["text"] as? String == "say hi")
    }

    @Test func separatesMultipleInstructionSegmentsInSystemPrompt() async throws {
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

        _ = try await model.step(
            transcript: promptOnlyTranscript("say hi"),
            tools: [],
            instructions: .init(
                segments: [
                    .text(.init(content: "First instruction.")),
                    .text(.init(content: "Second instruction.")),
                ],
                toolDefinitions: []
            ),
            generating: String.self,
            includeSchemaInPrompt: true,
            options: GenerationOptions()
        )

        let payload = try requestPayload(from: transport)
        #expect(payload["system"] as? String == "First instruction.\n\nSecond instruction.")
    }

    @Test func preservesAssistantTextAndToolUseInSingleAssistantMessageWhenEncodingTranscript() async throws {
        let transport = MockAnthropicV2Transport()
        transport.enqueueJSON(
            """
            {
              "id": "msg_2",
              "type": "message",
              "role": "assistant",
              "content": [{ "type": "text", "text": "done" }],
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

        let transcript = Transcript(entries: [
            .response(
                Transcript.Response(
                    assetIDs: [],
                    segments: [.text(.init(content: "Let me check."))]
                )
            ),
            .toolCalls(
                Transcript.ToolCalls([
                    Transcript.ToolCall(
                        id: "toolu_1",
                        toolName: "getWeather",
                        arguments: GeneratedContent(properties: ["city": "San Francisco"])
                    )
                ])
            ),
            .toolOutput(
                Transcript.ToolOutput(
                    id: "toolu_1",
                    toolName: "getWeather",
                    segments: [.text(.init(content: "Sunny"))]
                )
            ),
            .prompt(
                Transcript.Prompt(
                    segments: [.text(.init(content: "Now summarize."))]
                )
            ),
        ])

        _ = try await model.step(
            transcript: transcript,
            tools: [WeatherTool()],
            instructions: .init(
                segments: [.text(.init(content: "You are a careful coding agent."))],
                toolDefinitions: []
            ),
            generating: String.self,
            includeSchemaInPrompt: true,
            options: GenerationOptions()
        )

        let payload = try requestPayload(from: transport)
        #expect(payload["system"] as? String == "You are a careful coding agent.")

        let messages = try #require(payload["messages"] as? [[String: Any]])
        #expect(messages.count == 3)

        #expect(messages[0]["role"] as? String == "assistant")
        let assistantContent = try #require(messages[0]["content"] as? [[String: Any]])
        #expect(assistantContent.count == 2)
        #expect(assistantContent[0]["type"] as? String == "text")
        #expect(assistantContent[0]["text"] as? String == "Let me check.")
        #expect(assistantContent[1]["type"] as? String == "tool_use")
        #expect(assistantContent[1]["id"] as? String == "toolu_1")
        #expect(assistantContent[1]["name"] as? String == "getWeather")

        #expect(messages[1]["role"] as? String == "user")
        let toolResultContent = try #require(messages[1]["content"] as? [[String: Any]])
        #expect(toolResultContent.count == 1)
        #expect(toolResultContent[0]["type"] as? String == "tool_result")
        #expect(toolResultContent[0]["tool_use_id"] as? String == "toolu_1")

        #expect(messages[2]["role"] as? String == "user")
        let promptContent = try #require(messages[2]["content"] as? [[String: Any]])
        #expect(promptContent.count == 1)
        #expect(promptContent[0]["type"] as? String == "text")
        #expect(promptContent[0]["text"] as? String == "Now summarize.")
    }

    @Test func coalescesMultipleToolResultsIntoSingleUserMessage() async throws {
        let transport = MockAnthropicV2Transport()
        transport.enqueueJSON(
            """
            {
              "id": "msg_2",
              "type": "message",
              "role": "assistant",
              "content": [{ "type": "text", "text": "done" }],
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

        let transcript = Transcript(entries: [
            .response(
                Transcript.Response(
                    assetIDs: [],
                    segments: [.text(.init(content: "Checking both files."))]
                )
            ),
            .toolCalls(
                Transcript.ToolCalls([
                    Transcript.ToolCall(
                        id: "toolu_1",
                        toolName: "read_file",
                        arguments: GeneratedContent(properties: ["path": "a.txt"])
                    ),
                    Transcript.ToolCall(
                        id: "toolu_2",
                        toolName: "read_file",
                        arguments: GeneratedContent(properties: ["path": "b.txt"])
                    ),
                ])
            ),
            .toolOutput(
                Transcript.ToolOutput(
                    id: "toolu_1",
                    toolName: "read_file",
                    segments: [.text(.init(content: "A"))]
                )
            ),
            .toolOutput(
                Transcript.ToolOutput(
                    id: "toolu_2",
                    toolName: "read_file",
                    segments: [.text(.init(content: "B"))]
                )
            ),
        ])

        _ = try await model.step(
            transcript: transcript,
            tools: [],
            instructions: nil,
            generating: String.self,
            includeSchemaInPrompt: true,
            options: GenerationOptions()
        )

        let payload = try requestPayload(from: transport)
        let messages = try #require(payload["messages"] as? [[String: Any]])
        #expect(messages.count == 2)

        #expect(messages[0]["role"] as? String == "assistant")
        #expect(messages[1]["role"] as? String == "user")

        let userContent = try #require(messages[1]["content"] as? [[String: Any]])
        #expect(userContent.count == 2)
        #expect(userContent[0]["type"] as? String == "tool_result")
        #expect(userContent[0]["tool_use_id"] as? String == "toolu_1")
        #expect(userContent[1]["type"] as? String == "tool_result")
        #expect(userContent[1]["tool_use_id"] as? String == "toolu_2")
    }

    @Test func flushesToolResultsBeforeNextAssistantToolUse() async throws {
        let transport = MockAnthropicV2Transport()
        transport.enqueueJSON(
            """
            {
              "id": "msg_3",
              "type": "message",
              "role": "assistant",
              "content": [{ "type": "text", "text": "done" }],
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

        let transcript = Transcript(entries: [
            .prompt(
                Transcript.Prompt(
                    segments: [.text(.init(content: "Read the file requirements.txt"))]
                )
            ),
            .toolCalls(
                Transcript.ToolCalls([
                    Transcript.ToolCall(
                        id: "functions.read_file:0",
                        toolName: "read_file",
                        arguments: GeneratedContent(properties: ["path": "requirements.txt"])
                    )
                ])
            ),
            .toolOutput(
                Transcript.ToolOutput(
                    id: "functions.read_file:0",
                    toolName: "read_file",
                    segments: [.text(.init(content: "\"Error: no such file\""))]
                )
            ),
            .toolCalls(
                Transcript.ToolCalls([
                    Transcript.ToolCall(
                        id: "functions.bash:1",
                        toolName: "bash",
                        arguments: GeneratedContent(properties: ["command": "ls -la"])
                    )
                ])
            ),
            .toolOutput(
                Transcript.ToolOutput(
                    id: "functions.bash:1",
                    toolName: "bash",
                    segments: [.text(.init(content: "\"total 64\""))]
                )
            ),
        ])

        _ = try await model.step(
            transcript: transcript,
            tools: [],
            instructions: nil,
            generating: String.self,
            includeSchemaInPrompt: true,
            options: GenerationOptions()
        )

        let payload = try requestPayload(from: transport)
        let messages = try #require(payload["messages"] as? [[String: Any]])
        #expect(messages.count == 5)

        #expect(messages[0]["role"] as? String == "user")
        let firstUserContent = try #require(messages[0]["content"] as? [[String: Any]])
        #expect(firstUserContent.count == 1)
        #expect(firstUserContent[0]["type"] as? String == "text")
        #expect(firstUserContent[0]["text"] as? String == "Read the file requirements.txt")

        #expect(messages[1]["role"] as? String == "assistant")
        let firstAssistantContent = try #require(messages[1]["content"] as? [[String: Any]])
        #expect(firstAssistantContent.count == 1)
        #expect(firstAssistantContent[0]["type"] as? String == "tool_use")
        #expect(firstAssistantContent[0]["id"] as? String == "functions.read_file:0")

        #expect(messages[2]["role"] as? String == "user")
        let secondUserContent = try #require(messages[2]["content"] as? [[String: Any]])
        #expect(secondUserContent.count == 1)
        #expect(secondUserContent[0]["type"] as? String == "tool_result")
        #expect(secondUserContent[0]["tool_use_id"] as? String == "functions.read_file:0")

        #expect(messages[3]["role"] as? String == "assistant")
        let secondAssistantContent = try #require(messages[3]["content"] as? [[String: Any]])
        #expect(secondAssistantContent.count == 1)
        #expect(secondAssistantContent[0]["type"] as? String == "tool_use")
        #expect(secondAssistantContent[0]["id"] as? String == "functions.bash:1")

        #expect(messages[4]["role"] as? String == "user")
        let finalUserContent = try #require(messages[4]["content"] as? [[String: Any]])
        #expect(finalUserContent.count == 1)
        #expect(finalUserContent[0]["type"] as? String == "tool_result")
        #expect(finalUserContent[0]["tool_use_id"] as? String == "functions.bash:1")
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

private func requestPayload(from transport: MockAnthropicV2Transport) throws -> [String: Any] {
    let request = try #require(transport.requests.last)
    let body = try #require(request.httpBody)
    let payload = try JSONSerialization.jsonObject(with: body) as? [String: Any]
    return try #require(payload)
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
