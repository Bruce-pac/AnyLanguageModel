import Foundation
import Testing

@testable import AnyLanguageModel

@Suite("LoopingAnthropicLanguageModel", .serialized)
struct LoopingAnthropicLanguageModelTests {
    @Test func appliesCustomOptionsFromLoopingModelType() async throws {
        let transport = MockAnthropicTransport()
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

        let model = LoopingAnthropicLanguageModel(
            apiKey: "test-api-key",
            model: "claude-test",
            session: transport.session
        )

        let lmSession = LanguageModelSession(model: model)

        var options = GenerationOptions(temperature: 0.2, maximumResponseTokens: 111)
        options[custom: LoopingAnthropicLanguageModel.self] = .init(
            topP: 0.9,
            topK: 12,
            stopSequences: ["STOP"],
            toolChoice: .any
        )

        _ = try await lmSession.respond(to: "say hi", options: options)

        #expect(transport.requests.count == 1)
        let firstRequestBody = try transport.requestBodyJSONObject(at: 0)
        let body = try #require(firstRequestBody)

        #expect(body["top_p"] as? Double == 0.9)
        #expect(body["top_k"] as? Int == 12)
        #expect((body["stop_sequences"] as? [String]) == ["STOP"])

        let toolChoice = body["tool_choice"] as? [String: Any]
        #expect(toolChoice?["type"] as? String == "any")
    }

    @Test func decodesThinkingBlockBeforeToolUse() async throws {
        let transport = MockAnthropicTransport()
        transport.enqueueJSON(
            """
            {
              "id": "msg_1",
              "type": "message",
              "role": "assistant",
              "content": [
                {
                  "type": "thinking",
                  "thinking": "I should inspect the workspace first.",
                  "signature": "sig_1"
                },
                {
                  "type": "text",
                  "text": "Let me check the files."
                },
                {
                  "type": "tool_use",
                  "id": "toolu_weather_1",
                  "name": "getWeather",
                  "input": { "city": "San Francisco" }
                }
              ],
              "model": "claude-test",
              "stop_reason": "tool_use"
            }
            """
        )
        transport.enqueueJSON(
            """
            {
              "id": "msg_2",
              "type": "message",
              "role": "assistant",
              "content": [{ "type": "text", "text": "It is sunny in San Francisco." }],
              "model": "claude-test",
              "stop_reason": "end_turn"
            }
            """
        )

        let model = LoopingAnthropicLanguageModel(
            apiKey: "test-api-key",
            model: "claude-test",
            session: transport.session
        )

        let lmSession = LanguageModelSession(model: model, tools: [WeatherTool()])
        let response = try await lmSession.respond(to: "What's the weather?")

        #expect(response.content.contains("sunny"))
        #expect(transport.requests.count == 2)
    }

    @Test func loopsToolUseUntilFinalAssistantResponse() async throws {
        let transport = MockAnthropicTransport()
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
                }
              ],
              "model": "claude-test",
              "stop_reason": "tool_use"
            }
            """
        )
        transport.enqueueJSON(
            """
            {
              "id": "msg_2",
              "type": "message",
              "role": "assistant",
              "content": [{ "type": "text", "text": "It is sunny in San Francisco." }],
              "model": "claude-test",
              "stop_reason": "end_turn"
            }
            """
        )

        let model = LoopingAnthropicLanguageModel(
            apiKey: "test-api-key",
            model: "claude-test",
            session: transport.session
        )

        let lmSession = LanguageModelSession(model: model, tools: [WeatherTool()])
        let response = try await lmSession.respond(to: "What's the weather?")

        #expect(response.content.contains("sunny"))
        #expect(transport.requests.count == 2)

        let secondRequest = try #require(transport.requests.count > 1 ? transport.requests[1] : nil)
        let secondBodyJSONData = try #require(secondRequest.httpBody)
        let secondBody = String(decoding: secondBodyJSONData, as: UTF8.self)
        let secondBodyObject = try #require(try transport.requestBodyJSONObject(at: 1))
        let messages = try #require(secondBodyObject["messages"] as? [[String: Any]])
        let assistantToolUseMessage = try #require(
            messages.first {
                ($0["role"] as? String) == "assistant"
                    && (($0["content"] as? [[String: Any]])?.contains { ($0["type"] as? String) == "tool_use" } == true)
            }
        )
        let contentBlocks = try #require(assistantToolUseMessage["content"] as? [[String: Any]])
        let toolUseBlock = try #require(contentBlocks.first { ($0["type"] as? String) == "tool_use" })
        let input = try #require(toolUseBlock["input"] as? [String: Any])

        #expect(secondBody.contains("tool_result"))
        #expect(secondBody.contains("toolu_weather_1"))
        #expect(secondBody.contains("sunny"))
        #expect(input["city"] as? String == "San Francisco")
        #expect(input["kind"] == nil)
        #expect(input["properties"] == nil)
        #expect(input["orderedKeys"] == nil)
    }

    @Test func supportsMultipleToolCallsInOneRound() async throws {
        let transport = MockAnthropicTransport()
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
        transport.enqueueJSON(
            """
            {
              "id": "msg_2",
              "type": "message",
              "role": "assistant",
              "content": [{ "type": "text", "text": "Both cities are sunny." }],
              "model": "claude-test",
              "stop_reason": "end_turn"
            }
            """
        )

        let model = LoopingAnthropicLanguageModel(
            apiKey: "test-api-key",
            model: "claude-test",
            session: transport.session
        )

        let lmSession = LanguageModelSession(model: model, tools: [WeatherTool()])
        _ = try await lmSession.respond(to: "Compare weather")

        #expect(transport.requests.count == 2)

        let secondRequest = try #require(transport.requests.count > 1 ? transport.requests[1] : nil)
        let secondBodyJSONData = try #require(secondRequest.httpBody)
        let secondBody = String(decoding: secondBodyJSONData, as: UTF8.self)

        #expect(secondBody.contains("toolu_weather_1"))
        #expect(secondBody.contains("toolu_weather_2"))

        let toolResultCount = secondBody.components(separatedBy: "\"tool_result\"").count - 1
        #expect(toolResultCount == 2)
    }
}

private final class MockAnthropicTransport {
    let session: URLSession

    private var responseQueue: [Data] = []
    private(set) var requests: [URLRequest] = []

    init() {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [MockAnthropicURLProtocol.self]
        self.session = URLSession(configuration: config)
        MockAnthropicURLProtocol.store = self
    }

    func enqueueJSON(_ json: String) {
        responseQueue.append(Data(json.utf8))
    }

    func dequeueResponse() throws -> Data {
        guard !responseQueue.isEmpty else {
            throw NSError(domain: "MockAnthropicTransport", code: 1, userInfo: [NSLocalizedDescriptionKey: "No queued response available."])
        }
        return responseQueue.removeFirst()
    }

    func record(_ request: URLRequest) {
        requests.append(request)
    }

    func requestBodyJSONObject(at index: Int) throws -> [String: Any]? {
        guard requests.indices.contains(index), let body = requests[index].httpBody else { return nil }
        let object = try JSONSerialization.jsonObject(with: body)
        return object as? [String: Any]
    }
}

private final class MockAnthropicURLProtocol: URLProtocol {
    nonisolated(unsafe) static weak var store: MockAnthropicTransport?

    override class func canInit(with request: URLRequest) -> Bool {
        request.url?.path.contains("/v1/messages") == true
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

    override func startLoading() {
        do {
            guard let store = Self.store else {
                throw NSError(domain: "MockAnthropicURLProtocol", code: 1, userInfo: [NSLocalizedDescriptionKey: "Missing transport store."])
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
