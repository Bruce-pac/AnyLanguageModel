import Foundation

/// A runtime agent that owns loop state and transcript mutation.
public protocol AgentProtocol: Sendable {
    var transcript: Transcript { get async }
    var tools: [any Tool] { get async }
    var instructions: Transcript.Instructions? { get async }
    var toolExecutionDelegate: (any ToolExecutionDelegateV2)? { get async }
}

/// A delegate that observes and controls tool execution for agent-driven loops.
public protocol ToolExecutionDelegateV2: Sendable {
    /// Notifies the delegate when the model generated tool calls.
    func didGenerateToolCalls(_ toolCalls: [Transcript.ToolCall], in session: any AgentProtocol) async

    /// Asks the delegate how to handle a tool call.
    func toolCallDecision(
        for toolCall: Transcript.ToolCall,
        in session: any AgentProtocol
    ) async -> ToolExecutionDecision

    /// Notifies the delegate after a tool call produced output.
    func didExecuteToolCall(
        _ toolCall: Transcript.ToolCall,
        output: Transcript.ToolOutput,
        in session: any AgentProtocol
    ) async

    /// Notifies the delegate when tool execution failed.
    func didFailToolCall(
        _ toolCall: Transcript.ToolCall,
        error: any Error,
        in session: any AgentProtocol
    ) async
}

extension ToolExecutionDelegateV2 {
    public func didGenerateToolCalls(_ toolCalls: [Transcript.ToolCall], in session: any AgentProtocol) async {}

    public func toolCallDecision(
        for toolCall: Transcript.ToolCall,
        in session: any AgentProtocol
    ) async -> ToolExecutionDecision {
        .execute
    }

    public func didExecuteToolCall(
        _ toolCall: Transcript.ToolCall,
        output: Transcript.ToolOutput,
        in session: any AgentProtocol
    ) async {}

    public func didFailToolCall(
        _ toolCall: Transcript.ToolCall,
        error: any Error,
        in session: any AgentProtocol
    ) async {}
}
