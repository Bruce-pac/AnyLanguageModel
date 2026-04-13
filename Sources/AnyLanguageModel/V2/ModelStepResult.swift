import Foundation

/// A unified single-step result from a provider adapter.
public struct ModelStepResult<Content: Generable>: Sendable {
    public let assistantResponse: Transcript.Response?
    public let toolCalls: [Transcript.ToolCall]
    public let finalContent: Content?
    public let rawContent: GeneratedContent?
    public let finishReason: ModelFinishReason

    public init(
        assistantResponse: Transcript.Response?,
        toolCalls: [Transcript.ToolCall],
        finalContent: Content?,
        rawContent: GeneratedContent?,
        finishReason: ModelFinishReason
    ) {
        self.assistantResponse = assistantResponse
        self.toolCalls = toolCalls
        self.finalContent = finalContent
        self.rawContent = rawContent
        self.finishReason = finishReason
    }
}
