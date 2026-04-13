/// A normalized model stop reason across providers.
public enum ModelFinishReason: Sendable, Equatable {
    case toolUse
    case completed
    case pauseTurn
    case refusal
    case maxTokens
    case stopSequence
    case unknown(String?)
}
