import Foundation
import JSONSchema

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

/// A V2 Anthropic adapter that performs one provider step per call.
public struct AnthropicLanguageModelV2: LanguageModelV2 {
    public typealias UnavailableReason = Never
    public typealias CustomGenerationOptions = AnthropicLanguageModel.CustomGenerationOptions

    public static let defaultBaseURL = URL(string: "https://api.anthropic.com")!
    public static let defaultAPIVersion = "2023-06-01"

    public let baseURL: URL
    private let tokenProvider: @Sendable () -> String
    public let apiVersion: String
    public let betas: [String]?
    public let model: String

    private let httpSession: HTTPSession

    public init(
        baseURL: URL = defaultBaseURL,
        apiKey tokenProvider: @escaping @autoclosure @Sendable () -> String,
        apiVersion: String = defaultAPIVersion,
        betas: [String]? = nil,
        model: String,
        session: HTTPSession = makeDefaultSession()
    ) {
        var baseURL = baseURL
        if !baseURL.path.hasSuffix("/") {
            baseURL = baseURL.appendingPathComponent("")
        }

        self.baseURL = baseURL
        self.tokenProvider = tokenProvider
        self.apiVersion = apiVersion
        self.betas = betas
        self.model = model
        self.httpSession = session
    }

    public func step<Content>(
        transcript: Transcript,
        tools: [any Tool],
        instructions: Transcript.Instructions?,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> ModelStepResult<Content> where Content: Generable {
        _ = includeSchemaInPrompt

        let url = baseURL.appendingPathComponent("v1/messages")
        let headers = buildHeaders()

        let anthropicTools: [AnthropicTool] = try tools.map { tool in
            try convertToolToAnthropicFormat(tool)
        }

        let responseSchema = type == String.self ? nil : try convertSchemaToAnthropicFormat(Content.generationSchema)
        let params = try createMessageParams(
            model: model,
            system: anthropicSystemPrompt(from: instructions),
            messages: anthropicMessages(from: transcript),
            tools: anthropicTools.isEmpty ? nil : anthropicTools,
            responseSchema: responseSchema,
            options: options
        )

        let body = try JSONEncoder().encode(params)

        let message: AnthropicMessageResponse = try await httpSession.fetch(
            .post,
            url: url,
            headers: headers,
            body: body
        )

        let assistantContent = extractAssistantResponse(from: message.content)
        let toolCalls = try convertToolUsesToCalls(from: message.content)
        let finishReason = mapFinishReason(message.stopReason)

        if !toolCalls.isEmpty {
            return ModelStepResult(
                assistantResponse: assistantContent.response,
                toolCalls: toolCalls,
                finalContent: nil,
                rawContent: nil,
                finishReason: finishReason
            )
        }

        let text = assistantContent.text
        if type == String.self {
            let raw = GeneratedContent(text)
            return ModelStepResult(
                assistantResponse: assistantContent.response,
                toolCalls: [],
                finalContent: text as? Content,
                rawContent: raw,
                finishReason: finishReason
            )
        }

        let raw = try GeneratedContent(json: text)
        let content = try Content(raw)
        return ModelStepResult(
            assistantResponse: assistantContent.response,
            toolCalls: [],
            finalContent: content,
            rawContent: raw,
            finishReason: finishReason
        )
    }

    private func buildHeaders() -> [String: String] {
        var headers: [String: String] = [
            "x-api-key": tokenProvider(),
            "anthropic-version": apiVersion,
        ]

        if let betas = betas, !betas.isEmpty {
            headers["anthropic-beta"] = betas.joined(separator: ",")
        }

        return headers
    }
}

private func anthropicSystemPrompt(from instructions: Transcript.Instructions?) -> String? {
    guard let instructions else {
        return nil
    }

    let parts = instructions.segments.compactMap { segment -> String? in
        switch segment {
        case .text(let text):
            return text.content
        case .structure(let structure):
            return structure.content.jsonString
        case .image:
            return nil
        }
    }

    let content = parts.joined(separator: "\n\n")

    return content.isEmpty ? nil : content
}

private func mapFinishReason(_ reason: AnthropicMessageResponse.StopReason?) -> ModelFinishReason {
    switch reason {
    case .toolUse:
        return .toolUse
    case .endTurn:
        return .completed
    case .pauseTurn:
        return .pauseTurn
    case .refusal:
        return .refusal
    case .maxTokens, .modelContextWindowExceeded:
        return .maxTokens
    case .stopSequence:
        return .stopSequence
    case .none:
        return .unknown(nil)
    }
}

private func createMessageParams(
    model: String,
    system: String?,
    messages: [AnthropicMessage],
    tools: [AnthropicTool]?,
    responseSchema: JSONSchema?,
    options: GenerationOptions
) throws -> [String: JSONValue] {
    var params: [String: JSONValue] = [
        "model": .string(model),
        "messages": try JSONValue(messages),
        "max_tokens": .int(options.maximumResponseTokens ?? 1024),
    ]

    if let system {
        params["system"] = .string(system)
    }
    if let tools, !tools.isEmpty {
        params["tools"] = try JSONValue(tools)
    }
    if let responseSchema {
        let schemaValue = try JSONValue(responseSchema)
        if case .object(let schemaObject) = schemaValue, schemaObject.isEmpty {
            // Anthropic rejects empty schemas; omit output_config in this case.
        } else {
            params["output_config"] = .object(
                [
                    "format": .object(
                        [
                            "type": .string("json_schema"),
                            "schema": schemaValue,
                        ]
                    )
                ]
            )
        }
    }
    if let temperature = options.temperature {
        params["temperature"] = .double(temperature)
    }

    if let customOptions = options[custom: AnthropicLanguageModelV2.self] {
        if let topP = customOptions.topP {
            params["top_p"] = .double(topP)
        }
        if let topK = customOptions.topK {
            params["top_k"] = .int(topK)
        }
        if let stopSequences = customOptions.stopSequences, !stopSequences.isEmpty {
            params["stop_sequences"] = .array(stopSequences.map { JSONValue.string($0) })
        }
        if let metadata = customOptions.metadata {
            var metadataObject: [String: JSONValue] = [:]
            if let userID = metadata.userID {
                metadataObject["user_id"] = .string(userID)
            }
            if !metadataObject.isEmpty {
                params["metadata"] = .object(metadataObject)
            }
        }
        if let toolChoice = customOptions.toolChoice {
            switch toolChoice {
            case .auto:
                params["tool_choice"] = .object(["type": .string("auto")])
            case .any:
                params["tool_choice"] = .object(["type": .string("any")])
            case .tool(let name):
                params["tool_choice"] = .object([
                    "type": .string("tool"),
                    "name": .string(name),
                ])
            case .disabled:
                params["tool_choice"] = .object(["type": .string("none")])
            }
        }
        if let thinking = customOptions.thinking {
            params["thinking"] = .object([
                "type": .string(thinking.type.rawValue),
                "budget_tokens": .int(thinking.budgetTokens),
            ])
        }
        if let serviceTier = customOptions.serviceTier {
            params["service_tier"] = .string(serviceTier.rawValue)
        }
        if let extraBody = customOptions.extraBody {
            for (key, value) in extraBody {
                params[key] = value
            }
        }
    }

    return params
}

private func convertSchemaToAnthropicFormat(_ schema: GenerationSchema) throws -> JSONSchema {
    let resolvedSchema = try schema.inlined()
    let data = try JSONEncoder().encode(resolvedSchema)
    return try JSONDecoder().decode(JSONSchema.self, from: data)
}

private func convertToolToAnthropicFormat(_ tool: any Tool) throws -> AnthropicTool {
    let schema = try convertSchemaToAnthropicFormat(tool.parameters)
    return AnthropicTool(name: tool.name, description: tool.description, inputSchema: schema)
}

private func toGeneratedContent(_ value: [String: JSONValue]?) throws -> GeneratedContent {
    guard let value else { return GeneratedContent(properties: [:]) }
    let data = try JSONEncoder().encode(JSONValue.object(value))
    let json = String(data: data, encoding: .utf8) ?? "{}"
    return try GeneratedContent(json: json)
}

private func fromGeneratedContent(_ content: GeneratedContent) throws -> [String: JSONValue] {
    let jsonObject = try content.toJSONValue()
    let data = try JSONSerialization.data(withJSONObject: jsonObject, options: [.fragmentsAllowed])
    let jsonValue = try JSONDecoder().decode(JSONValue.self, from: data)

    guard case .object(let dict) = jsonValue else {
        return [:]
    }
    return dict
}

private func extractAssistantResponse(
    from content: [AnthropicContent]
) -> (response: Transcript.Response?, text: String) {
    let textSegments: [Transcript.Segment] = content.compactMap { block in
        guard case .text(let text) = block else { return nil }
        return .text(.init(content: text.text))
    }
    let response = textSegments.isEmpty ? nil : Transcript.Response(assetIDs: [], segments: textSegments)
    let text = textSegments.compactMap { segment -> String? in
        guard case .text(let text) = segment else { return nil }
        return text.content
    }.joined()
    return (response, text)
}

private func convertToolUsesToCalls(
    from content: [AnthropicContent]
) throws -> [Transcript.ToolCall] {
    let toolUses: [AnthropicToolUse] = content.compactMap { block in
        if case .toolUse(let use) = block { return use }
        return nil
    }

    return try toolUses.map { use in
        Transcript.ToolCall(
            id: use.id,
            toolName: use.name,
            arguments: try toGeneratedContent(use.input)
        )
    }
}

private func anthropicMessages(from transcript: Transcript) -> [AnthropicMessage] {
    var messages = [AnthropicMessage]()
    var pendingAssistantContent: [AnthropicContent] = []
    var pendingUserToolResults: [AnthropicContent] = []

    func flushAssistantContent() {
        guard !pendingAssistantContent.isEmpty else { return }
        messages.append(.init(role: .assistant, content: pendingAssistantContent))
        pendingAssistantContent.removeAll(keepingCapacity: true)
    }

    func flushUserToolResults() {
        guard !pendingUserToolResults.isEmpty else { return }
        messages.append(.init(role: .user, content: pendingUserToolResults))
        pendingUserToolResults.removeAll(keepingCapacity: true)
    }

    for item in transcript {
        switch item {
        case .instructions:
            continue
        case .prompt(let prompt):
            flushAssistantContent()
            flushUserToolResults()
            messages.append(
                .init(
                    role: .user,
                    content: convertSegmentsToAnthropicContent(prompt.segments)
                )
            )
        case .response(let response):
            pendingAssistantContent.append(contentsOf: convertSegmentsToAnthropicContent(response.segments))
        case .toolCalls(let toolCalls):
            let toolUseBlocks: [AnthropicContent] = toolCalls.map { call in
                let input = try? fromGeneratedContent(call.arguments)
                return .toolUse(
                    AnthropicToolUse(
                        id: call.id,
                        name: call.toolName,
                        input: input
                    )
                )
            }
            pendingAssistantContent.append(contentsOf: toolUseBlocks)
        case .toolOutput(let toolOutput):
            flushAssistantContent()
            pendingUserToolResults.append(
                .toolResult(
                    AnthropicToolResult(
                        toolUseId: toolOutput.id,
                        content: convertSegmentsToAnthropicContent(toolOutput.segments)
                    )
                )
            )
        }
    }
    flushAssistantContent()
    flushUserToolResults()

    return messages
}

private struct AnthropicTool: Codable, Sendable {
    let name: String
    let description: String
    let inputSchema: JSONSchema

    enum CodingKeys: String, CodingKey {
        case name
        case description
        case inputSchema = "input_schema"
    }
}

private struct AnthropicMessage: Codable, Sendable {
    enum Role: String, Codable, Sendable { case user, assistant }

    let role: Role
    let content: [AnthropicContent]
}

private enum AnthropicContent: Codable, Sendable {
    case text(AnthropicText)
    case thinking(AnthropicThinking)
    case image(AnthropicImage)
    case toolUse(AnthropicToolUse)
    case toolResult(AnthropicToolResult)

    enum CodingKeys: String, CodingKey { case type }

    enum ContentType: String, Codable {
        case text = "text"
        case thinking = "thinking"
        case image = "image"
        case toolUse = "tool_use"
        case toolResult = "tool_result"
    }

    init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(ContentType.self, forKey: .type)
        switch type {
        case .text:
            self = .text(try AnthropicText(from: decoder))
        case .thinking:
            self = .thinking(try AnthropicThinking(from: decoder))
        case .image:
            self = .image(try AnthropicImage(from: decoder))
        case .toolUse:
            self = .toolUse(try AnthropicToolUse(from: decoder))
        case .toolResult:
            self = .toolResult(try AnthropicToolResult(from: decoder))
        }
    }

    func encode(to encoder: any Encoder) throws {
        switch self {
        case .text(let t): try t.encode(to: encoder)
        case .thinking(let t): try t.encode(to: encoder)
        case .image(let i): try i.encode(to: encoder)
        case .toolUse(let u): try u.encode(to: encoder)
        case .toolResult(let r): try r.encode(to: encoder)
        }
    }
}

private struct AnthropicText: Codable, Sendable {
    let type: String
    let text: String

    init(text: String) {
        self.type = "text"
        self.text = text
    }
}

private struct AnthropicThinking: Codable, Sendable {
    let type: String
    let thinking: String
    let signature: String?
}

private struct AnthropicImage: Codable, Sendable {
    struct Source: Codable, Sendable {
        let type: String
        let mediaType: String?
        let data: String?
        let url: String?

        enum CodingKeys: String, CodingKey {
            case type
            case mediaType = "media_type"
            case data
            case url
        }
    }

    let type: String
    let source: Source

    init(base64Data: String, mimeType: String) {
        self.type = "image"
        self.source = Source(type: "base64", mediaType: mimeType, data: base64Data, url: nil)
    }

    init(url: String) {
        self.type = "image"
        self.source = Source(type: "url", mediaType: nil, data: nil, url: url)
    }
}

private func convertSegmentsToAnthropicContent(_ segments: [Transcript.Segment]) -> [AnthropicContent] {
    var blocks: [AnthropicContent] = []
    blocks.reserveCapacity(segments.count)
    for segment in segments {
        switch segment {
        case .text(let t):
            blocks.append(.text(AnthropicText(text: t.content)))
        case .structure(let s):
            blocks.append(.text(AnthropicText(text: s.content.jsonString)))
        case .image(let img):
            switch img.source {
            case .url(let url):
                blocks.append(.image(AnthropicImage(url: url.absoluteString)))
            case .data(let data, let mimeType):
                blocks.append(.image(AnthropicImage(base64Data: data.base64EncodedString(), mimeType: mimeType)))
            }
        }
    }
    return blocks
}

private struct AnthropicToolUse: Codable, Sendable {
    let type: String
    let id: String
    let name: String
    let input: [String: JSONValue]?

    init(id: String, name: String, input: [String: JSONValue]?) {
        self.type = "tool_use"
        self.id = id
        self.name = name
        self.input = input
    }
}

private struct AnthropicToolResult: Codable, Sendable {
    let type: String
    let toolUseId: String
    let content: [AnthropicContent]

    enum CodingKeys: String, CodingKey {
        case type
        case toolUseId = "tool_use_id"
        case content
    }

    init(toolUseId: String, content: [AnthropicContent]) {
        self.type = "tool_result"
        self.toolUseId = toolUseId
        self.content = content
    }
}

private struct AnthropicMessageResponse: Codable, Sendable {
    let id: String
    let type: String
    let role: String
    let content: [AnthropicContent]
    let model: String
    let stopReason: StopReason?

    enum CodingKeys: String, CodingKey {
        case id, type, role, content, model
        case stopReason = "stop_reason"
    }

    enum StopReason: String, Codable {
        case endTurn = "end_turn"
        case maxTokens = "max_tokens"
        case stopSequence = "stop_sequence"
        case toolUse = "tool_use"
        case pauseTurn = "pause_turn"
        case refusal = "refusal"
        case modelContextWindowExceeded = "model_context_window_exceeded"
    }
}
