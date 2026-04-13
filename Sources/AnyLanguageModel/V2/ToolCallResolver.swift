import Foundation

public enum AgentToolResolutionOutcome: Sendable {
    case stop
    case outputs([Transcript.ToolOutput])
}

public struct ToolCallResolutionError: Error, LocalizedError {
    public let toolName: String
    public let underlyingError: any Error

    public init(toolName: String, underlyingError: any Error) {
        self.toolName = toolName
        self.underlyingError = underlyingError
    }

    public var errorDescription: String? {
        "Tool execution failed for \(toolName): \(underlyingError.localizedDescription)"
    }
}

/// Shared tool execution resolver for agent-driven loops.
public struct ToolCallResolver: Sendable {
    public init() {}

    public func resolve(
        _ calls: [Transcript.ToolCall],
        session: any AgentProtocol
    ) async throws -> AgentToolResolutionOutcome {
        guard !calls.isEmpty else {
            return .outputs([])
        }

        let tools = await session.tools
        var toolsByName: [String: any Tool] = [:]
        for tool in tools where toolsByName[tool.name] == nil {
            toolsByName[tool.name] = tool
        }

        let delegate = await session.toolExecutionDelegate
        if let delegate {
            await delegate.didGenerateToolCalls(calls, in: session)
        }

        var decisions: [ToolExecutionDecision] = []
        decisions.reserveCapacity(calls.count)

        if let delegate {
            for call in calls {
                let decision = await delegate.toolCallDecision(for: call, in: session)
                if case .stop = decision {
                    return .stop
                }

                decisions.append(decision)
            }
        } else {
            decisions = Array(repeating: .execute, count: calls.count)
        }

        var outputs: [Transcript.ToolOutput] = []
        outputs.reserveCapacity(calls.count)

        for (index, call) in calls.enumerated() {
            switch decisions[index] {
            case .stop:
                return .stop
            case .provideOutput(let segments):
                let output = Transcript.ToolOutput(
                    id: call.id,
                    toolName: call.toolName,
                    segments: segments
                )
                if let delegate {
                    await delegate.didExecuteToolCall(call, output: output, in: session)
                }
                outputs.append(output)
            case .execute:
                guard let tool = toolsByName[call.toolName] else {
                    let output = Transcript.ToolOutput(
                        id: call.id,
                        toolName: call.toolName,
                        segments: [.text(.init(content: "Tool not found: \(call.toolName)"))]
                    )
                    if let delegate {
                        await delegate.didExecuteToolCall(call, output: output, in: session)
                    }
                    outputs.append(output)
                    continue
                }

                do {
                    let segments = try await tool.makeOutputSegments(from: call.arguments)
                    let output = Transcript.ToolOutput(
                        id: call.id,
                        toolName: tool.name,
                        segments: segments
                    )
                    if let delegate {
                        await delegate.didExecuteToolCall(call, output: output, in: session)
                    }
                    outputs.append(output)
                } catch {
                    if let delegate {
                        await delegate.didFailToolCall(call, error: error, in: session)
                    }
                    throw ToolCallResolutionError(toolName: call.toolName, underlyingError: error)
                }
            }
        }

        return .outputs(outputs)
    }
}
