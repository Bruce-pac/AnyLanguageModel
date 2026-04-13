import Foundation
import Testing

@testable import AnyLanguageModel

@Suite("ToolCallResolver", .serialized)
struct ToolCallResolverTests {
    @Test func executesToolsAndNotifiesDelegate() async throws {
        let delegate = RecordingAgentDelegate()
        let agent = MockAgent(
            tools: [EchoTool()],
            delegate: delegate
        )
        let call = Transcript.ToolCall(
            id: "call_1",
            toolName: "echo",
            arguments: GeneratedContent(properties: ["text": "hi"])
        )

        let resolver = ToolCallResolver()
        let outcome = try await resolver.resolve([call], session: agent)

        guard case .outputs(let outputs) = outcome else {
            Issue.record("Expected tool outputs")
            return
        }

        #expect(outputs.count == 1)
        let text = outputs[0].segments.compactMap { segment -> String? in
            switch segment {
            case .text(let text):
                return text.content
            case .structure(let structure):
                return structure.content.jsonString
            case .image:
                return nil
            }
        }.joined()
        #expect(text.contains("echo:hi"))

        let recorded = await delegate.recordedEvents()
        #expect(recorded == [
            "generated:1",
            "decision:echo",
            "executed:echo",
        ])
    }

    @Test func returnsStopWhenDelegateRequestsStop() async throws {
        let delegate = RecordingAgentDelegate(decision: .stop)
        let agent = MockAgent(
            tools: [EchoTool()],
            delegate: delegate
        )
        let call = Transcript.ToolCall(
            id: "call_1",
            toolName: "echo",
            arguments: GeneratedContent(properties: ["text": "hi"])
        )

        let resolver = ToolCallResolver()
        let outcome = try await resolver.resolve([call], session: agent)

        guard case .stop = outcome else {
            Issue.record("Expected stop outcome")
            return
        }

        let recorded = await delegate.recordedEvents()
        #expect(recorded == [
            "generated:1",
            "decision:echo",
        ])
    }

    @Test func usesProvideOutputDecisionWithoutExecutingTool() async throws {
        let manualSegments: [Transcript.Segment] = [.text(.init(content: "manual-output"))]
        let delegate = RecordingAgentDelegate(decision: .provideOutput(manualSegments))
        let agent = MockAgent(
            tools: [EchoTool()],
            delegate: delegate
        )
        let call = Transcript.ToolCall(
            id: "call_1",
            toolName: "echo",
            arguments: GeneratedContent(properties: ["text": "hi"])
        )

        let resolver = ToolCallResolver()
        let outcome = try await resolver.resolve([call], session: agent)

        guard case .outputs(let outputs) = outcome else {
            Issue.record("Expected tool outputs")
            return
        }

        #expect(outputs.count == 1)
        let text = outputs[0].segments.compactMap { segment -> String? in
            guard case .text(let text) = segment else { return nil }
            return text.content
        }.joined()
        #expect(text == "manual-output")

        let recorded = await delegate.recordedEvents()
        #expect(recorded == [
            "generated:1",
            "decision:echo",
            "executed:echo",
        ])
    }

    @Test func propagatesExecutionFailureAndNotifiesDelegate() async throws {
        let delegate = RecordingAgentDelegate()
        let agent = MockAgent(
            tools: [FailingTool()],
            delegate: delegate
        )
        let call = Transcript.ToolCall(
            id: "call_1",
            toolName: "failing_tool",
            arguments: GeneratedContent(properties: ["text": "boom"])
        )

        let resolver = ToolCallResolver()

        await #expect(throws: ToolCallResolutionError.self) {
            _ = try await resolver.resolve([call], session: agent)
        }

        let recorded = await delegate.recordedEvents()
        #expect(recorded == [
            "generated:1",
            "decision:failing_tool",
            "failed:failing_tool",
        ])
    }
}

private actor MockAgent: AgentProtocol {
    var transcript: Transcript
    let tools: [any Tool]
    let instructions: Transcript.Instructions?
    let toolExecutionDelegate: (any ToolExecutionDelegateV2)?

    init(
        transcript: Transcript = Transcript(),
        tools: [any Tool],
        instructions: Transcript.Instructions? = nil,
        delegate: (any ToolExecutionDelegateV2)? = nil
    ) {
        self.transcript = transcript
        self.tools = tools
        self.instructions = instructions
        self.toolExecutionDelegate = delegate
    }

}

private actor RecordingAgentDelegate: ToolExecutionDelegateV2 {
    private let decision: ToolExecutionDecision
    private var events: [String] = []

    init(decision: ToolExecutionDecision = .execute) {
        self.decision = decision
    }

    func recordedEvents() -> [String] {
        events
    }

    func didGenerateToolCalls(_ toolCalls: [Transcript.ToolCall], in session: any AgentProtocol) async {
        events.append("generated:\(toolCalls.count)")
    }

    func toolCallDecision(
        for toolCall: Transcript.ToolCall,
        in session: any AgentProtocol
    ) async -> ToolExecutionDecision {
        events.append("decision:\(toolCall.toolName)")
        return decision
    }

    func didExecuteToolCall(
        _ toolCall: Transcript.ToolCall,
        output: Transcript.ToolOutput,
        in session: any AgentProtocol
    ) async {
        events.append("executed:\(toolCall.toolName)")
    }

    func didFailToolCall(
        _ toolCall: Transcript.ToolCall,
        error: any Error,
        in session: any AgentProtocol
    ) async {
        events.append("failed:\(toolCall.toolName)")
    }
}

private struct EchoTool: Tool {
    let name = "echo"
    let description = "echoes input"

    @Generable
    struct Arguments {
        var text: String
    }

    func call(arguments: Arguments) async throws -> String {
        "echo:\(arguments.text)"
    }
}

private struct FailingTool: Tool {
    let name = "failing_tool"
    let description = "always throws"

    @Generable
    struct Arguments {
        var text: String
    }

    func call(arguments: Arguments) async throws -> String {
        struct StubError: Error {}
        throw StubError()
    }
}
