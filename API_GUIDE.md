# AnyLanguageModel API 使用指南

> AnyLanguageModel 是 Apple FoundationModels 框架的跨平台替代品，提供统一的 Swift API 来对接多种语言模型提供商。只需更改 import 语句即可从 FoundationModels 迁移：
>
> ```diff
> - import FoundationModels
> + import AnyLanguageModel
> ```

---

## 目录

- [核心概念](#核心概念)
- [快速开始](#快速开始)
- [模型提供商](#模型提供商)
- [Session（会话）管理](#session会话管理)
- [Prompt 与 Instructions](#prompt-与-instructions)
- [结构化生成（Guided Generation）](#结构化生成guided-generation)
- [流式响应（Streaming）](#流式响应streaming)
- [工具调用（Tool Calling）](#工具调用tool-calling)
- [工具执行控制（ToolExecutionDelegate）](#工具执行控制toolexecutiondelegate)
- [图片输入（Multimodal）](#图片输入multimodal)
- [GenerationOptions（生成选项）](#generationoptions生成选项)
- [自定义生成选项（Custom Generation Options）](#自定义生成选项custom-generation-options)
- [Transcript（对话记录）](#transcript对话记录)
- [Observation（响应式观察）](#observation响应式观察)
- [反馈机制（Feedback）](#反馈机制feedback)
- [LanguageModel 协议实现类对比](#languagemodel-协议实现类对比)
- [坑点与注意事项](#坑点与注意事项)

---

## 核心概念

AnyLanguageModel 的架构围绕以下核心类型展开：

| 类型 | 角色 |
|---|---|
| `LanguageModel` | 协议，所有模型提供商的抽象接口 |
| `LanguageModelSession` | 会话管理器，持有模型、工具、指令和对话记录 |
| `Prompt` | 用户输入 |
| `Instructions` | 系统指令（System Prompt） |
| `GeneratedContent` | 模型返回的原始内容（类 JSON 的树形结构） |
| `Generable` | 协议 + 宏，让自定义类型可作为结构化输出目标 |
| `GenerationOptions` | 生成参数控制（温度、采样、最大 token 等） |
| `Tool` | 工具协议，让模型调用你的代码 |
| `Transcript` | 完整对话历史记录 |

**调用流程**：创建 Model → 创建 Session → 调用 `respond` 或 `streamResponse` → 获取结果。

---

## 快速开始

```swift
import AnyLanguageModel

// 1. 选择模型
let model = OpenAILanguageModel(
    apiKey: ProcessInfo.processInfo.environment["OPENAI_API_KEY"]!,
    model: "gpt-4o-mini"
)

// 2. 创建会话
let session = LanguageModelSession(model: model)

// 3. 获取响应
let response = try await session.respond(to: "Hello!")
print(response.content) // String
```

---

## 模型提供商

### 创建各提供商模型实例

```swift
// Apple Foundation Models（macOS 26+ / iOS 26+）
let system = SystemLanguageModel.default

// OpenAI（支持 Chat Completions 和 Responses 两种 API）
let openai = OpenAILanguageModel(
    apiKey: "sk-...",
    model: "gpt-4o-mini"
)

// OpenAI 兼容端点使用 Chat Completions API
let openaiCompat = OpenAILanguageModel(
    baseURL: URL(string: "https://api.example.com")!,
    apiKey: "sk-...",
    model: "gpt-4o-mini",
    apiVariant: .chatCompletions
)

// Open Responses（多提供商 Responses API 兼容端点）
let openResponses = OpenResponsesLanguageModel(
    baseURL: URL(string: "https://openrouter.ai/api/v1/")!,
    apiKey: "...",
    model: "openai/gpt-4o-mini"
)

// Anthropic Claude
let anthropic = AnthropicLanguageModel(
    apiKey: "...",
    model: "claude-sonnet-4-5-20250929"
)

// Google Gemini
let gemini = GeminiLanguageModel(
    apiKey: "...",
    model: "gemini-2.5-flash"
)

// Ollama（本地，默认连接 localhost:11434）
let ollama = OllamaLanguageModel(model: "qwen3")

// Ollama（自定义端点）
let ollamaRemote = OllamaLanguageModel(
    endpoint: URL(string: "http://remote:11434")!,
    model: "llama3.2"
)

// MLX（需启用 MLX trait，仅 Apple Silicon）
let mlx = MLXLanguageModel(modelId: "mlx-community/Qwen3-0.6B-4bit")

// CoreML（需启用 CoreML trait）
let coreml = CoreMLLanguageModel(url: URL(fileURLWithPath: "path/to/model.mlmodelc"))

// llama.cpp（需启用 Llama trait）
let llama = LlamaLanguageModel(modelPath: "/path/to/model.gguf")
```

---

## Session（会话）管理

`LanguageModelSession` 是所有交互的入口，它管理模型、工具、系统指令和对话历史。

### 创建 Session

```swift
// 基础用法
let session = LanguageModelSession(model: model)

// 带系统指令
let session = LanguageModelSession(model: model, instructions: "你是一个有帮助的助手")

// 使用 Instructions builder
let session = LanguageModelSession(model: model) {
    "你是一个专业翻译"
    "请用简洁的语言回复"
}

// 带工具
let session = LanguageModelSession(model: model, tools: [WeatherTool()])

// 带工具和指令
let session = LanguageModelSession(
    model: model,
    tools: [WeatherTool()],
    instructions: "你是天气助手"
)

// 从已有 Transcript 恢复会话
let session = LanguageModelSession(model: model, transcript: savedTranscript)
```

### 发送请求

```swift
// 返回 String
let response = try await session.respond(to: "Hello")
print(response.content) // String

// 使用 Prompt builder
let response = try await session.respond {
    Prompt("Translate this to French:")
    Prompt("Hello, world!")
}

// 生成结构化类型
let response = try await session.respond(
    to: "Generate a cat profile",
    generating: CatProfile.self
)
print(response.content) // CatProfile

// 使用 schema 生成 GeneratedContent
let response = try await session.respond(
    to: "Generate data",
    schema: MyType.generationSchema
)
print(response.rawContent) // GeneratedContent
```

### Response 结构

```swift
let response = try await session.respond(to: "Hello")
response.content           // 解码后的内容（String、自定义 Generable 类型等）
response.rawContent        // GeneratedContent（原始生成内容）
response.transcriptEntries // 本次交互产生的 Transcript 条目
```

---

## Prompt 与 Instructions

### Prompt

`Prompt` 是用户输入的封装，支持字符串字面量和 Result Builder。

```swift
// 字符串
let prompt = Prompt("What's the weather?")

// Result Builder（支持条件语句）
let prompt = Prompt {
    "Base context"
    if includeDetails {
        "Include detailed analysis"
    }
    if let customInstruction {
        customInstruction
    }
}
```

**注意**：Builder 会自动修剪首尾空白和换行，多行通过换行连接。

### Instructions

`Instructions` 用于系统指令，与 `Prompt` 的 API 风格相同。

```swift
let instructions = Instructions {
    "You are a helpful assistant"
    if isVerbose {
        "Provide detailed explanations"
    }
}
```

**自定义类型作为 Prompt/Instructions**：任何 `Generable` 类型自动遵守 `PromptRepresentable` 和 `InstructionsRepresentable`，它们的 JSON 表示将被嵌入 prompt 中。

---

## 结构化生成（Guided Generation）

用 `@Generable` 宏标记 struct 或 enum，配合 `@Guide` 约束属性值，让模型直接输出强类型数据。

### 基本用法

```swift
@Generable(description: "一只猫的基本信息")
struct CatProfile {
    var name: String       // 无需 @Guide 的简单字段
    
    @Guide(description: "猫的年龄", .range(0...20))
    var age: Int
    
    @Guide(description: "一句话描述猫的性格")
    var profile: String
}

let session = LanguageModelSession(model: model)
let response = try await session.respond(
    to: "Generate a cute rescue cat",
    generating: CatProfile.self
)
let cat = response.content // CatProfile 实例
print(cat.name, cat.age, cat.profile)
```

### @Guide 约束类型

```swift
// 字符串约束
@Guide(description: "颜色", .anyOf(["red", "blue", "green"]))
var color: String

@Guide(description: "固定值", .constant("hello"))
var greeting: String

@Guide(description: "邮箱", .pattern(#"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"#))
var email: String

// 数值约束
@Guide(description: "年龄", .range(0...150))
var age: Int

@Guide(description: "评分", .minimum(0), .maximum(10))
var score: Double

// 数组约束
@Guide(description: "标签", .count(1...5))
var tags: [String]

@Guide(description: "至少2项", .minimumCount(2))
var items: [Item]

@Guide(description: "元素约束", .element(.range(0...100)))
var scores: [Int]
```

### 支持的类型

`@Generable` 宏可应用于 struct 和 enum。以下基础类型内建支持 `Generable`：

- `String`、`Bool`、`Int`、`Float`、`Double`、`Decimal`
- `Optional<T>` where T: Generable
- `Array<T>` where T: Generable

### 嵌套类型与枚举

```swift
@Generable(description: "RGB 颜色值")
struct RGBValues {
    @Guide(description: "Red", .range(0...255))
    var r: Int
    @Guide(description: "Green", .range(0...255))
    var g: Int
    @Guide(description: "Blue", .range(0...255))
    var b: Int
}

@Generable(description: "颜色信息")
struct ColorInfo {
    var name: String
    var rgb: RGBValues  // 嵌套 Generable 类型
}

@Generable(description: "情感分析结果")
enum Sentiment {
    case positive
    case negative
    case neutral
}
```

### PartiallyGenerated

`@Generable` 宏会为 struct 自动生成 `PartiallyGenerated` 嵌套类型（所有属性变为 Optional），用于流式场景中表示部分生成的内容：

```swift
// 编译器自动生成：
// CatProfile.PartiallyGenerated {
//     var name: String?
//     var age: Int?
//     var profile: String?
// }
```

### DynamicGenerationSchema

运行时动态构建 schema（无需编译时宏）：

```swift
let schema = DynamicGenerationSchema(
    name: "Person",
    description: "A person",
    properties: [
        .init(name: "name", description: "Full name", schema: .init(scalar: .string)),
        .init(name: "age", description: "Age", schema: .init(scalar: .integer), isOptional: true)
    ]
)
```

---

## 流式响应（Streaming）

### 基本流式

```swift
let stream = session.streamResponse(to: "Write a story")
for try await snapshot in stream {
    print(snapshot.content) // 逐步输出的部分内容
}
```

### 流式 + 结构化类型

```swift
let stream = session.streamResponse(
    to: "Generate a cat",
    generating: CatProfile.self
)
for try await snapshot in stream {
    // snapshot.content 是 CatProfile.PartiallyGenerated
    // 属性可能为 nil（尚未生成）
    if let name = snapshot.content.name {
        print("Name: \(name)")
    }
}
```

### 收集完整结果

```swift
let stream = session.streamResponse(to: "Hello")
let response = try await stream.collect()
print(response.content) // 完整的 String
```

### 使用 Builder 语法

```swift
let stream = session.streamResponse {
    Prompt("Write a poem about swift programming")
}
```

---

## 工具调用（Tool Calling）

### 定义工具

```swift
struct WeatherTool: Tool {
    let name = "getWeather"
    let description = "获取指定城市的天气信息"
    
    @Generable
    struct Arguments {
        @Guide(description: "城市名称")
        var city: String
    }
    
    func call(arguments: Arguments) async throws -> String {
        "The weather in \(arguments.city) is sunny and 72°F"
    }
}
```

### 使用工具

```swift
let session = LanguageModelSession(model: model, tools: [WeatherTool()])

let response = try await session.respond {
    Prompt("How's the weather in Cupertino?")
}
// 模型自动决定是否调用工具，工具结果会自动注入对话
print(response.content)
```

### Tool 协议详解

| 属性/方法 | 类型 | 说明 |
|---|---|---|
| `name` | `String` | 工具唯一名称，默认为类型名 |
| `description` | `String` | 自然语言描述，告诉模型何时使用 |
| `parameters` | `GenerationSchema` | 参数 schema，Arguments 遵守 Generable 时自动推导 |
| `includesSchemaInInstructions` | `Bool` | 是否将 schema 注入系统指令，默认 `true` |
| `call(arguments:)` | `async throws -> Output` | 执行逻辑，Output 需遵守 `PromptRepresentable` |

### Tool 的 Output 类型

Tool 的 `Output` 必须遵守 `PromptRepresentable`。常见选择：

- `String` — 最简单，直接返回文本
- `[String]` — 数组类型（Array 遵守 PromptRepresentable）
- 任意 `Generable` 类型 — 结构化输出会被序列化为 JSON 注入上下文

---

## 工具执行控制（ToolExecutionDelegate）

> ⚠️ `ToolExecutionDelegate` 是 AnyLanguageModel 独有的 API，使用后代码不再与 FoundationModels 框架兼容。

```swift
actor MyToolDelegate: ToolExecutionDelegate {
    // 模型生成了工具调用时触发
    func didGenerateToolCalls(
        _ toolCalls: [Transcript.ToolCall],
        in session: LanguageModelSession
    ) async {
        print("Tool calls generated: \(toolCalls.map(\.toolName))")
    }
    
    // 决定如何处理每个工具调用
    func toolCallDecision(
        for toolCall: Transcript.ToolCall,
        in session: LanguageModelSession
    ) async -> ToolExecutionDecision {
        // .execute     — 正常执行工具（默认）
        // .stop        — 生成工具调用后停止，不执行
        // .provideOutput([...]) — 提供自定义输出，跳过真实执行
        .execute
    }
    
    // 工具执行成功后触发
    func didExecuteToolCall(
        _ toolCall: Transcript.ToolCall,
        output: Transcript.ToolOutput,
        in session: LanguageModelSession
    ) async {
        print("Tool \(toolCall.toolName) executed")
    }
    
    // 工具执行失败时触发
    func didFailToolCall(
        _ toolCall: Transcript.ToolCall,
        error: any Error,
        in session: LanguageModelSession
    ) async {
        print("Tool \(toolCall.toolName) failed: \(error)")
    }
}

session.toolExecutionDelegate = MyToolDelegate()
```

典型场景：在执行前弹窗让用户确认、注入缓存结果、监控工具使用情况。

---

## 图片输入（Multimodal）

```swift
// 单张图片（URL）
let response = try await session.respond(
    to: "Describe this image",
    image: .init(url: URL(string: "https://example.com/photo.jpg")!)
)

// 多张图片
let response = try await session.respond(
    to: "Compare these images",
    images: [
        .init(url: URL(string: "https://example.com/a.jpg")!),
        .init(url: URL(fileURLWithPath: "/path/to/b.png"))
    ]
)

// Base64 数据
let response = try await session.respond(
    to: "What's in this image?",
    image: .init(
        data: imageData,
        mimeType: "image/png"
    )
)

// 图片 + 结构化输出
let response = try await session.respond(
    to: "Extract text from receipt",
    images: [.init(url: receiptURL)],
    generating: ReceiptData.self
)
```

### 图片支持矩阵

| 提供商 | 支持图片 |
|---|:---:|
| OpenAI | ✅ |
| Open Responses | ✅ |
| Anthropic | ✅ |
| Google Gemini | ✅ |
| Ollama | 视模型而定（需 VLM 模型） |
| MLX | 视模型而定（需 VLM 模型） |
| Apple Foundation Models | ❌ |
| CoreML | ❌ |
| llama.cpp | ❌ |

---

## GenerationOptions（生成选项）

```swift
var options = GenerationOptions(
    sampling: .greedy,                    // 采样策略
    temperature: 0.7,                     // 温度（0~1）
    maximumResponseTokens: 1024           // 最大输出 token 数
)

let response = try await session.respond(to: "Hello", options: options)
```

### 采样策略（SamplingMode）

```swift
// 贪心采样：始终选最大概率 token，输出确定性最高
.greedy

// Top-K 采样：从概率最高的 K 个 token 中随机选
.random(top: 40, seed: 42)

// Nucleus（Top-P）采样：累积概率达到阈值的 token 池中随机选
.random(probabilityThreshold: 0.9, seed: 42)
```

> **注意**：`seed` 不保证完全确定性输出，仅尽力而为。

---

## 自定义生成选项（Custom Generation Options）

每个模型提供商都可定义自己的 `CustomGenerationOptions` 类型，通过下标访问：

```swift
var options = GenerationOptions(temperature: 0.7)

// OpenAI 专属
options[custom: OpenAILanguageModel.self] = .init(
    topP: 0.9,
    frequencyPenalty: 0.5,
    presencePenalty: 0.3,
    stopSequences: ["END"],
    reasoningEffort: .high,          // o-series 模型推理力度
    extraBody: [                     // 供应商特定参数
        "custom_param": .string("value")
    ]
)

// Anthropic 专属
options[custom: AnthropicLanguageModel.self] = .init(
    topP: 0.9,
    topK: 40,
    stopSequences: ["END", "STOP"],
    thinking: .init(budgetTokens: 4096),  // Extended Thinking
    toolChoice: .auto,
    serviceTier: .priority
)

// Gemini 专属
options[custom: GeminiLanguageModel.self] = .init(
    thinking: .dynamic,              // .disabled / .dynamic / .budget(1024)
    serverTools: [.googleSearch],    // 服务端工具
    jsonMode: true                   // JSON 输出模式
)

// Ollama 专属（动态字典）
options[custom: OllamaLanguageModel.self] = [
    "seed": .int(42),
    "repeat_penalty": .double(1.2),
    "num_ctx": .int(4096),
    "stop": .array([.string("###")])
]

// llama.cpp 专属
options[custom: LlamaLanguageModel.self] = .init(
    contextSize: 4096,
    batchSize: 512,
    threads: 8,
    seed: 42,
    topK: 40,
    topP: 0.95,
    repeatPenalty: 1.2,
    mirostat: .v2(tau: 5.0, eta: 0.1)
)

// MLX 专属
var mlxOptions = MLXLanguageModel.CustomGenerationOptions.default
mlxOptions.kvCache = .init(maxSize: 4096, bits: 4, groupSize: 64, quantizedStart: 128)
mlxOptions.userInputProcessing = .resize(to: CGSize(width: 512, height: 512))
mlxOptions.additionalContext = ["user_name": .string("Alice")]
options[custom: MLXLanguageModel.self] = mlxOptions
```

---

## Transcript（对话记录）

`Transcript` 是完整的对话历史，Session 的每次交互都会自动追加条目。

### 条目类型

```swift
session.transcript // Transcript

// 遍历条目
for entry in session.transcript {
    switch entry {
    case .instructions(let instr):
        // 系统指令
    case .prompt(let prompt):
        // 用户输入
    case .response(let response):
        // 模型回复
    case .toolCalls(let calls):
        // 模型发起的工具调用
    case .toolOutput(let output):
        // 工具执行结果
    }
}
```

### Segment 类型

每个条目内部由 `Segment` 组成：

```swift
// 文本段
Transcript.Segment.text(TextSegment(content: "Hello"))

// 结构化段（来自 Generable 类型）
Transcript.Segment.structure(StructuredSegment(source: "toolName", content: generatedContent))

// 图片段
Transcript.Segment.image(ImageSegment(url: imageURL))
Transcript.Segment.image(ImageSegment(data: imageData, mimeType: "image/png"))
```

### 从 Transcript 恢复会话

```swift
// 保存对话（Transcript 遵守 Codable）
let data = try JSONEncoder().encode(session.transcript)

// 恢复对话
let transcript = try JSONDecoder().decode(Transcript.self, from: data)
let newSession = LanguageModelSession(model: model, transcript: transcript)
```

---

## Observation（响应式观察）

`LanguageModelSession` 标记为 `@Observable`，可与 SwiftUI 无缝集成。

### 可观察属性

- `session.transcript` — 对话记录变更时触发
- `session.isResponding` — 正在生成响应时为 `true`

```swift
// SwiftUI 用法
struct ChatView: View {
    let session: LanguageModelSession
    
    var body: some View {
        VStack {
            // 自动响应 transcript 变化
            ForEach(session.transcript.filter { ... }) { entry in
                MessageView(entry: entry)
            }
            
            if session.isResponding {
                ProgressView()
            }
        }
    }
}
```

```swift
// 手动观察
withObservationTracking {
    _ = session.transcript
} onChange: {
    print("Transcript changed!")
}
```

> **注意**：必须在 `withObservationTracking` 的闭包中 **读取** 属性才能触发跟踪。不读取则不会收到通知。

---

## 反馈机制（Feedback）

```swift
let issue = LanguageModelFeedback.Issue(
    category: .tooVerbose,
    explanation: "回复包含不必要的冗余内容"
)

let feedback = LanguageModelFeedback(
    sentiment: .negative,
    issues: [issue]
)

// 提交反馈附件
let data = session.logFeedbackAttachment(
    sentiment: feedback.sentiment,
    issues: feedback.issues
)
```

### 反馈类别

| 类别 | 说明 |
|---|---|
| `.unhelpful` | 没有帮助 |
| `.tooVerbose` | 过于冗长 |
| `.didNotFollowInstructions` | 未遵循指令 |
| `.incorrect` | 内容不正确 |
| `.stereotypeOrBias` | 刻板印象或偏见 |
| `.suggestiveOrSexual` | 暗示性或色情内容 |
| `.vulgarOrOffensive` | 粗俗或冒犯性内容 |
| `.triggeredGuardrailUnexpectedly` | 意外触发安全护栏 |

---

## LanguageModel 协议实现类对比

### 总览表

| 实现类 | 运行位置 | 需要 Trait | 工具调用 | 图片输入 | 结构化生成 | 流式 |
|---|---|:---:|:---:|:---:|:---:|:---:|
| `SystemLanguageModel` | 设备端 | — | ✅ | ❌ | ✅ | ✅ |
| `OpenAILanguageModel` | 云端 | — | ✅ | ✅ | ✅ | ✅ |
| `OpenResponsesLanguageModel` | 云端 | — | ✅ | ✅ | ✅ | ✅ |
| `AnthropicLanguageModel` | 云端 | — | ✅ | ✅ | ✅ | ✅ |
| `GeminiLanguageModel` | 云端 | — | ✅ | ✅ | ✅ | ✅ |
| `OllamaLanguageModel` | 本地 HTTP | — | ✅ | 视模型 | ✅ | ✅ |
| `MLXLanguageModel` | 设备端 | `MLX` | ✅ | 视模型 | ✅ | ✅ |
| `CoreMLLanguageModel` | 设备端 | `CoreML` | ❌ | ❌ | ✅ | ✅ |
| `LlamaLanguageModel` | 设备端 | `Llama` | ❌ | ❌ | ✅ | ✅ |

### 各实现详解

#### `SystemLanguageModel`

- **类型**：`actor`
- **运行要求**：macOS 26+ / iOS 26+ / visionOS 26+
- **包装对象**：Apple FoundationModels 框架的 `SystemLanguageModel`
- **Availability**：有 `UnavailableReason`（设备不支持时返回具体原因）
- **特点**：唯一有 `UnavailableReason` 关联类型的实现；API 与 FoundationModels 完全兼容

#### `OpenAILanguageModel`

- **类型**：`struct`
- **UnavailableReason**：`Never`（始终 available）
- **API 变体**：`.chatCompletions`（旧版）或 `.responses`（默认，新版）
- **自定义选项**：`topP`、`frequencyPenalty`、`presencePenalty`、`stopSequences`、`logitBias`、`seed`、`topK`、`minP`、`reasoningEffort`、`serviceTier`、`extraBody`
- **特点**：支持自定义 `baseURL` 对接 OpenAI 兼容服务

#### `OpenResponsesLanguageModel`

- **类型**：`struct`
- **UnavailableReason**：`Never`
- **必须参数**：`baseURL`（无默认端点）
- **自定义选项**：`toolChoice`、`allowedTools`、`topP`、`presencePenalty`、`frequencyPenalty`、`parallelToolCalls`、`maxToolCalls`、`reasoningEffort`、`reasoning`、`verbosity`、`maxOutputTokens`、`store`、`metadata`、`truncationStrategy`、`extraBody`
- **特点**：选项最丰富，可对接 OpenRouter 等多种服务

#### `AnthropicLanguageModel`

- **类型**：`struct`
- **UnavailableReason**：`Never`
- **自定义选项**：`topP`、`topK`、`stopSequences`、`metadata`、`toolChoice`、`thinking`（Extended Thinking）、`serviceTier`、`extraBody`
- **特点**：支持 `betas` 参数启用实验性功能；支持 Extended Thinking

#### `GeminiLanguageModel`

- **类型**：`struct`
- **UnavailableReason**：`Never`
- **自定义选项**：`thinking`（`.disabled`/`.dynamic`/`.budget(Int)`）、`serverTools`、`jsonMode`
- **独有功能**：**服务端工具**（`.googleSearch`、`.googleMaps`、`.codeExecution`、`.urlContext`），这些工具在 Google 基础设施上运行，不可作为其他模型的 client tool
- **特点**：支持 `apiVersion` 参数

#### `OllamaLanguageModel`

- **类型**：`struct`
- **UnavailableReason**：`Never`
- **自定义选项**：`Dictionary<String, JSONValue>`（动态键值对）
- **特点**：本地运行无需 API Key；自定义选项是动态字典而非强类型 struct，可传任意模型参数

#### `MLXLanguageModel`

- **类型**：`struct`
- **需要 Trait**：`MLX`（仅 Apple Silicon）
- **UnavailableReason**：`Never`
- **自定义选项**：`kvCache`、`userInputProcessing`、`additionalContext`
- **特点**：支持 GPU 内存策略配置（`.automatic`）；Vision 取决于加载的具体模型

#### `CoreMLLanguageModel`

- **类型**：`struct`
- **需要 Trait**：`CoreML`
- **最低版本**：macOS 15.0+
- **特点**：无工具调用；加载本地 `.mlmodelc` 模型；支持 compute unit 选择

#### `LlamaLanguageModel`

- **类型**：`class`（注意：不是 struct）
- **需要 Trait**：`Llama`
- **自定义选项**：参数极其丰富（`contextSize`、`batchSize`、`threads`、`seed`、`temperature`、`topK`、`topP`、`minP`、`frequencyPenalty`、`presencePenalty`、`repeatPenalty`、`repeatLastN`、`mirostat` 等）
- **特点**：唯一使用 `class` 的实现；不支持工具调用；支持 token 级别的约束生成；支持日志回调（`LogLevel`）

---

## 坑点与注意事项

### 1. Xcode 26 编译错误

在 Xcode 26 中 targeting macOS 15 / iOS 18 或更早版本时，可能出现类似 `Conformance of 'String' to 'Generable' is only available in macOS 26.0 or newer` 的错误。**解决方案**：使用 Xcode 16 编译。

### 2. Package Traits 依赖解析失败

启用 traits 时 SPM 可能报 "exhausted attempts to resolve the dependencies graph"。**解决方案**：将对应 trait 的底层依赖也直接加入你的 `Package.swift`：

```swift
dependencies: [
    .package(url: "https://github.com/huggingface/AnyLanguageModel.git", from: "0.8.0", traits: ["MLX"]),
    .package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "2.25.5"),  // MLX trait 对应依赖
]
```

### 3. CustomGenerationOptions 不可 Codable 往返

`GenerationOptions` 虽然遵守 `Codable`，但 `customOptionsStorage` 在解码时**会丢失**——因为没有类型注册表来还原类型擦除的自定义选项。序列化/反序列化 `GenerationOptions` 后 custom options 会变为空。

### 4. toolExecutionDelegate 破坏 FoundationModels 兼容性

使用 `session.toolExecutionDelegate` 后，代码不再是 FoundationModels 的 drop-in 替换。如果需要保持兼容性，避免使用此 API。

### 5. @Generable 宏不适用于 class

`@Generable` 仅支持 `struct` 和 `enum`，不支持 `class`。

### 6. Optional 属性在 @Generable 中的行为

在 `@Generable` struct 中，`Optional` 属性在 schema 中标记为非 required。模型可能生成 `null` 或直接省略该字段。`PartiallyGenerated` 中**所有属性**都是 Optional（包括原本非 Optional 的），因为流式场景下任何字段都可能尚未生成。

### 7. 流式 collect() 返回的 transcriptEntries 为空

`ResponseStream.collect()` 返回的 `Response` 的 `transcriptEntries` 是空的 `[]`。完整的 transcript 条目由 Session 管理并自动追加到 `session.transcript`。

### 8. respond 返回值是 @discardableResult

`respond(to:)` 等方法标记为 `@discardableResult`，不消费返回值不会产生编译警告。但 transcript 仍会被更新。

### 9. Session 不支持并发请求

同一 Session 不应同时发起多个 `respond` 或 `streamResponse`。会话内部通过 `isResponding` 状态跟踪当前是否正在生成，并发请求可能抛出 `GenerationError.concurrentRequests`。

### 10. Observation 需要显式读取属性

使用 `withObservationTracking` 时，必须在闭包体内**实际读取**被观察属性（如 `_ = session.transcript`），否则不会注册观察。仅引用 session 对象本身不够。

### 11. Tool 的 name 默认值

`Tool` 的 `name` 属性默认返回类型名（`String(describing: Self.self)`），例如 `WeatherTool` 的默认 name 是 `"WeatherTool"`。如果需要更语义化的名称（如 `"getWeather"`），需显式声明 `let name = "getWeather"`。

### 12. includesSchemaInInstructions 的影响

`Tool` 的 `includesSchemaInInstructions` 默认为 `true`，意味着工具的 name、description 和 parameters schema 会被注入到系统指令中。如果模型已预训练知道该工具，设为 `false` 可减少 prompt 长度，但对 zero-shot 场景应始终保持 `true`。

### 13. Gemini 服务端工具的局限

Gemini 的 `serverTools`（如 `.googleSearch`、`.googleMaps`）在 Google 服务端执行，**不能**作为其他模型的客户端 `Tool` 使用，也不能与 `ToolExecutionDelegate` 交互。

### 14. LlamaLanguageModel 是 class

与其他模型实现（struct）不同，`LlamaLanguageModel` 是引用类型（class），因为它持有 llama.cpp 的内部状态。赋值和传递时请注意引用语义。

### 15. OpenAI apiVariant 默认值

`OpenAILanguageModel` 默认使用 `.responses` API（新版）。如果你的兼容端点只支持旧版 Chat Completions API，需要显式指定 `apiVariant: .chatCompletions`。

### 16. 平台要求

最低平台要求：iOS 17 / macOS 14 / visionOS 1 / Linux。但 `SystemLanguageModel` 额外要求 macOS 26+ / iOS 26+。其他云端模型无额外平台限制。

### 17. HTTP 传输层

在 Linux 或启用 `AsyncHTTPClient` trait 时使用 `AsyncHTTPClient`，其他平台默认使用 `URLSession`。Linux 上有专门的线程安全网关（`LinuxURLSessionRequestGate`）。

### 18. GeneratedContent 的 value 取值

从 `GeneratedContent` 提取值时，使用类型安全的 `value(_:)` 方法。类型不匹配时会抛出 `GeneratedContentError.typeMismatch`。不存在的属性访问会抛出 `propertyNotFound`。

```swift
let name: String = try content.value(String.self, forProperty: "name")
let age: Int = try content.value(Int.self, forProperty: "age")
```

### 19. GenerationError 类型

了解可能的错误类型以便正确处理：

| 错误 | 说明 |
|---|---|
| `exceededContextWindowSize` | 上下文窗口超限 |
| `assetsUnavailable` | 资源不可用 |
| `guardrailViolation` | 触发安全护栏 |
| `unsupportedGuide` | 不支持的 Guide 约束 |
| `unsupportedLanguageOrLocale` | 不支持的语言/地区 |
| `decodingFailure` | 解码失败 |
| `rateLimited` | 速率限制 |
| `concurrentRequests` | 并发请求冲突 |
| `refusal` | 模型拒绝回答（含 `Refusal` 对象可获取解释） |
