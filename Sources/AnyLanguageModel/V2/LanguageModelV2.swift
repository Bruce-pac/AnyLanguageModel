import Foundation

/// A single-step provider adapter for agent-first loops.
public protocol LanguageModelV2: Sendable {
    associatedtype UnavailableReason

    /// The type of custom generation options this model accepts.
    associatedtype CustomGenerationOptions: AnyLanguageModel.CustomGenerationOptions = Never

    var availability: Availability<UnavailableReason> { get }

    func step<Content>(
        transcript: Transcript,
        tools: [any Tool],
        instructions: Transcript.Instructions?,
        generating type: Content.Type,
        includeSchemaInPrompt: Bool,
        options: GenerationOptions
    ) async throws -> ModelStepResult<Content> where Content: Generable
}

extension LanguageModelV2 {
    public var isAvailable: Bool {
        if case .available = availability {
            return true
        }

        return false
    }
}

extension LanguageModelV2 where UnavailableReason == Never {
    public var availability: Availability<UnavailableReason> {
        .available
    }
}
