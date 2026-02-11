import Foundation
import JSONSchema
import Testing

@testable import AnyLanguageModel

private let openaiAPIKey: String? = ProcessInfo.processInfo.environment["OPENAI_API_KEY"]

@Suite("OpenAILanguageModel")
struct OpenAILanguageModelTests {
    @Test func customHost() throws {
        let customURL = URL(string: "https://example.com")!
        let model = OpenAILanguageModel(baseURL: customURL, apiKey: "test", model: "test-model")
        #expect(model.baseURL.absoluteString.hasSuffix("/"))
    }

    @Test func apiVariantParameterization() throws {
        for apiVariant in [OpenAILanguageModel.APIVariant.chatCompletions, .responses] {
            let model = OpenAILanguageModel(apiKey: "test-key", model: "test-model", apiVariant: apiVariant)
            #expect(model.apiVariant == apiVariant)
            #expect(model.model == "test-model")
        }
    }

    @Suite("OpenAILanguageModel Chat Completions API", .enabled(if: openaiAPIKey?.isEmpty == false))
    struct ChatCompletionsTests {
        private let apiKey = openaiAPIKey!

        private var model: OpenAILanguageModel {
            OpenAILanguageModel(apiKey: apiKey, model: "gpt-4o-mini", apiVariant: .chatCompletions)
        }

        @Test func basicResponse() async throws {
            let session = LanguageModelSession(model: model)

            let response = try await session.respond(to: "Say hello")
            #expect(!response.content.isEmpty)
        }

        @Test func withInstructions() async throws {
            let session = LanguageModelSession(
                model: model,
                instructions: "You are a helpful assistant. Be concise."
            )

            let response = try await session.respond(to: "What is 2+2?")
            #expect(!response.content.isEmpty)
        }

        @Test func streaming() async throws {
            let session = LanguageModelSession(model: model)

            let stream = session.streamResponse(to: "Count to 5")
            var chunks: [String] = []

            for try await response in stream {
                chunks.append(response.content)
            }

            #expect(!chunks.isEmpty)
        }

        @Test func streamingString() async throws {
            let session = LanguageModelSession(model: model)

            let stream = session.streamResponse(to: "Say 'Hello' slowly")

            var snapshots: [LanguageModelSession.ResponseStream<String>.Snapshot] = []
            for try await snapshot in stream {
                snapshots.append(snapshot)
            }

            #expect(!snapshots.isEmpty)
            #expect(!snapshots.last!.rawContent.jsonString.isEmpty)
        }

        @Test func withGenerationOptions() async throws {
            let session = LanguageModelSession(model: model)

            let options = GenerationOptions(
                temperature: 0.7,
                maximumResponseTokens: 50
            )

            let response = try await session.respond(
                to: "Tell me a fact",
                options: options
            )
            #expect(!response.content.isEmpty)
        }

        @Test func withCustomGenerationOptions() async throws {
            let session = LanguageModelSession(model: model)

            var options = GenerationOptions(
                temperature: 0.7,
                maximumResponseTokens: 50
            )

            options[custom: OpenAILanguageModel.self] = .init(
                extraBody: ["user": .string("test-user-id")]
            )

            let response = try await session.respond(
                to: "Say hello",
                options: options
            )
            #expect(!response.content.isEmpty)
        }

        @Test func multimodalWithImageURL() async throws {
            let session = LanguageModelSession(model: model)
            let response = try await session.respond(
                to: "Describe this image",
                image: .init(url: testImageURL)
            )
            #expect(!response.content.isEmpty)
        }

        @Test func multimodalWithImageData() async throws {
            let session = LanguageModelSession(model: model)
            let response = try await session.respond(
                to: "Describe this image",
                image: .init(data: testImageData, mimeType: "image/png")
            )
            #expect(!response.content.isEmpty)
        }

        @Test func conversationContext() async throws {
            let session = LanguageModelSession(model: model)

            let firstResponse = try await session.respond(to: "My favorite color is blue")
            #expect(!firstResponse.content.isEmpty)

            let secondResponse = try await session.respond(to: "What did I just tell you?")
            #expect(secondResponse.content.contains("color"))
        }

        @Test func withTools() async throws {
            let weatherTool = spy(on: WeatherTool())
            let session = LanguageModelSession(model: model, tools: [weatherTool])

            var options = GenerationOptions()
            options[custom: OpenAILanguageModel.self] = .init(
                maxToolCalls: 1
            )

            let response = try await withOpenAIRateLimitRetry {
                try await session.respond(
                    to: "Call getWeather for San Francisco exactly once, then summarize in one sentence.",
                    options: options
                )
            }

            #expect(!response.content.isEmpty)
            let calls = await weatherTool.calls
            #expect(!calls.isEmpty)
            if let firstCall = calls.first {
                #expect(firstCall.arguments.city.localizedCaseInsensitiveContains("san"))
            }

            var foundToolCall = false
            var foundToolOutput = false
            for entry in response.transcriptEntries {
                switch entry {
                case .toolCalls(let toolCalls):
                    #expect(!toolCalls.isEmpty)
                    if let firstToolCall = toolCalls.first {
                        #expect(firstToolCall.toolName == "getWeather")
                    }
                    foundToolCall = true
                case .toolOutput(let toolOutput):
                    #expect(toolOutput.toolName == "getWeather")
                    foundToolOutput = true
                default:
                    break
                }
            }
            #expect(foundToolCall)
            #expect(foundToolOutput)
        }

        @Test func withToolsConversationContinuesAcrossTurns() async throws {
            let weatherTool = spy(on: WeatherTool())
            let session = LanguageModelSession(model: model, tools: [weatherTool])

            var options = GenerationOptions()
            options[custom: OpenAILanguageModel.self] = .init(
                maxToolCalls: 1
            )

            _ = try await withOpenAIRateLimitRetry {
                try await session.respond(
                    to: "Call getWeather for San Francisco exactly once, then reply with only: done",
                    options: options
                )
            }

            let secondResponse = try await withOpenAIRateLimitRetry {
                try await session.respond(
                    to: "Which city did the tool call use? Reply with city only."
                )
            }
            #expect(!secondResponse.content.isEmpty)
            #expect(secondResponse.content.localizedCaseInsensitiveContains("san"))

            let calls = await weatherTool.calls
            #expect(calls.count >= 1)
        }

        @Suite("Structured Output")
        struct StructuredOutputTests {
            @Generable
            struct Person {
                @Guide(description: "The person's full name")
                var name: String

                @Guide(description: "The person's age in years")
                var age: Int

                @Guide(description: "The person's email address")
                var email: String?
            }

            @Generable
            struct Book {
                @Guide(description: "The book's title")
                var title: String

                @Guide(description: "The book's author")
                var author: String

                @Guide(description: "The publication year")
                var year: Int
            }

            private var model: OpenAILanguageModel {
                OpenAILanguageModel(apiKey: openaiAPIKey!, model: "gpt-4o-mini", apiVariant: .chatCompletions)
            }

            @Test func basicStructuredOutput() async throws {
                let session = LanguageModelSession(model: model)
                let response = try await session.respond(
                    to: "Generate a person named John Doe, age 30, email john@example.com",
                    generating: Person.self
                )

                #expect(!response.content.name.isEmpty)
                #expect(response.content.name.contains("John") || response.content.name.contains("Doe"))
                #expect(response.content.age > 0)
                #expect(response.content.age <= 100)
                #expect(response.content.email != nil)
            }

            @Test func structuredOutputWithOptionalField() async throws {
                let session = LanguageModelSession(model: model)
                let response = try await session.respond(
                    to: "Generate a person named Jane Smith, age 25, with no email",
                    generating: Person.self
                )

                #expect(!response.content.name.isEmpty)
                #expect(response.content.name.contains("Jane") || response.content.name.contains("Smith"))
                #expect(response.content.age > 0)
                #expect(response.content.age <= 100)
                #expect(response.content.email == nil || response.content.email?.isEmpty == true)
            }

            @Test func structuredOutputWithNestedTypes() async throws {
                let session = LanguageModelSession(model: model)
                let response = try await session.respond(
                    to: "Generate a book titled 'The Swift Programming Language' by 'Apple Inc.' published in 2024",
                    generating: Book.self
                )

                #expect(!response.content.title.isEmpty)
                #expect(!response.content.author.isEmpty)
                #expect(response.content.year >= 2020)
            }

            @Test func streamingStructuredOutput() async throws {
                let session = LanguageModelSession(model: model)
                let stream = session.streamResponse(
                    to: "Generate a person named Alice, age 28, email alice@example.com",
                    generating: Person.self
                )

                var snapshots: [LanguageModelSession.ResponseStream<Person>.Snapshot] = []
                for try await snapshot in stream {
                    snapshots.append(snapshot)
                }

                #expect(!snapshots.isEmpty)
                let finalSnapshot = snapshots.last!
                #expect((finalSnapshot.content.name?.isEmpty ?? true) == false)
                #expect((finalSnapshot.content.age ?? 0) > 0)
            }
        }
    }

    @Suite("OpenAILanguageModel Responses API", .enabled(if: openaiAPIKey?.isEmpty == false))
    struct ResponsesTests {
        private let apiKey = openaiAPIKey!

        private var model: OpenAILanguageModel {
            OpenAILanguageModel(apiKey: apiKey, model: "gpt-4o-mini", apiVariant: .responses)
        }

        @Test func basicResponse() async throws {
            let session = LanguageModelSession(model: model)

            let response = try await session.respond(to: "Say hello")
            #expect(!response.content.isEmpty)
        }

        @Test func withInstructions() async throws {
            let session = LanguageModelSession(
                model: model,
                instructions: "You are a helpful assistant. Be concise."
            )

            let response = try await session.respond(to: "What is 2+2?")
            #expect(!response.content.isEmpty)
        }

        @Test func streaming() async throws {
            let session = LanguageModelSession(model: model)

            let stream = session.streamResponse(to: "Count to 5")
            var chunks: [String] = []

            for try await response in stream {
                chunks.append(response.content)
            }

            #expect(!chunks.isEmpty)
        }

        @Test func streamingString() async throws {
            let session = LanguageModelSession(model: model)

            let stream = session.streamResponse(to: "Say 'Hello' slowly")

            var snapshots: [LanguageModelSession.ResponseStream<String>.Snapshot] = []
            for try await snapshot in stream {
                snapshots.append(snapshot)
            }

            #expect(!snapshots.isEmpty)
            #expect(!snapshots.last!.rawContent.jsonString.isEmpty)
        }

        @Test func withGenerationOptions() async throws {
            let session = LanguageModelSession(model: model)

            let options = GenerationOptions(
                temperature: 0.7,
                maximumResponseTokens: 50
            )

            let response = try await session.respond(
                to: "Tell me a fact",
                options: options
            )
            #expect(!response.content.isEmpty)
        }

        @Test func withCustomGenerationOptions() async throws {
            let session = LanguageModelSession(model: model)

            var options = GenerationOptions(
                temperature: 0.7,
                maximumResponseTokens: 50
            )

            options[custom: OpenAILanguageModel.self] = .init(
                extraBody: ["user": "test-user-id"]
            )

            let response = try await session.respond(
                to: "Say hello",
                options: options
            )
            #expect(!response.content.isEmpty)
        }

        @Test func multimodalWithImageURL() async throws {
            let session = LanguageModelSession(model: model)
            let response = try await session.respond(
                to: "Describe this image",
                image: .init(url: testImageURL)
            )
            #expect(!response.content.isEmpty)
        }

        @Test func multimodalWithImageData() async throws {
            let session = LanguageModelSession(model: model)
            let response = try await session.respond(
                to: "Describe this image",
                image: .init(data: testImageData, mimeType: "image/png")
            )
            #expect(!response.content.isEmpty)
        }

        @Test func conversationContext() async throws {
            let session = LanguageModelSession(model: model)

            let firstResponse = try await session.respond(to: "My favorite color is blue")
            #expect(!firstResponse.content.isEmpty)

            let secondResponse = try await session.respond(to: "What did I just tell you?")
            #expect(secondResponse.content.contains("color"))
        }

        @Test func withTools() async throws {
            let weatherTool = WeatherTool()
            let session = LanguageModelSession(model: model, tools: [weatherTool])

            let response = try await session.respond(to: "How's the weather in San Francisco?")

            var foundToolOutput = false
            for case let .toolOutput(toolOutput) in response.transcriptEntries {
                #expect(toolOutput.toolName == "getWeather")
                foundToolOutput = true
            }
            #expect(foundToolOutput)
        }

        @Suite("Structured Output")
        struct StructuredOutputTests {
            @Generable
            struct Person {
                @Guide(description: "The person's full name")
                var name: String

                @Guide(description: "The person's age in years")
                var age: Int

                @Guide(description: "The person's email address")
                var email: String?
            }

            @Generable
            struct Book {
                @Guide(description: "The book's title")
                var title: String

                @Guide(description: "The book's author")
                var author: String

                @Guide(description: "The publication year")
                var year: Int
            }

            private var model: OpenAILanguageModel {
                OpenAILanguageModel(apiKey: openaiAPIKey!, model: "gpt-4o-mini", apiVariant: .responses)
            }

            @Test func basicStructuredOutput() async throws {
                let session = LanguageModelSession(model: model)
                let response = try await session.respond(
                    to: "Generate a person named John Doe, age 30, email john@example.com",
                    generating: Person.self
                )

                #expect(!response.content.name.isEmpty)
                #expect(response.content.name.contains("John") || response.content.name.contains("Doe"))
                #expect(response.content.age > 0)
                #expect(response.content.age <= 100)
                #expect(response.content.email != nil)
            }

            @Test func structuredOutputWithOptionalField() async throws {
                let session = LanguageModelSession(model: model)
                let response = try await session.respond(
                    to: "Generate a person named Jane Smith, age 25, with no email",
                    generating: Person.self
                )

                #expect(!response.content.name.isEmpty)
                #expect(response.content.name.contains("Jane") || response.content.name.contains("Smith"))
                #expect(response.content.age > 0)
                #expect(response.content.age <= 100)
                #expect(response.content.email == nil || response.content.email?.isEmpty == true)
            }

            @Test func structuredOutputWithNestedTypes() async throws {
                let session = LanguageModelSession(model: model)
                let response = try await session.respond(
                    to: "Generate a book titled 'The Swift Programming Language' by 'Apple Inc.' published in 2024",
                    generating: Book.self
                )

                #expect(!response.content.title.isEmpty)
                #expect(!response.content.author.isEmpty)
                #expect(response.content.year >= 2020)
            }

            @Test func streamingStructuredOutput() async throws {
                let session = LanguageModelSession(model: model)
                let stream = session.streamResponse(
                    to: "Generate a person named Alice, age 28, email alice@example.com",
                    generating: Person.self
                )

                var snapshots: [LanguageModelSession.ResponseStream<Person>.Snapshot] = []
                for try await snapshot in stream {
                    snapshots.append(snapshot)
                }

                #expect(!snapshots.isEmpty)
                let finalSnapshot = snapshots.last!
                #expect((finalSnapshot.content.name?.isEmpty ?? true) == false)
                #expect((finalSnapshot.content.age ?? 0) > 0)
            }
        }
    }

    @Suite("OpenAILanguageModel Responses Request Body")
    struct ResponsesRequestBodyTests {
        private let model = "test-model"

        private func inputArray(from body: JSONValue) -> [JSONValue]? {
            guard case let .object(obj) = body else { return nil }
            guard case let .array(input)? = obj["input"] else { return nil }
            return input
        }

        private func stringValue(_ value: JSONValue?) -> String? {
            guard case let .string(text)? = value else { return nil }
            return text
        }

        private func firstObject(withType type: String, in input: [JSONValue]) -> [String: JSONValue]? {
            for value in input {
                guard case let .object(obj) = value else { continue }
                guard case let .string(foundType)? = obj["type"], foundType == type else { continue }
                return obj
            }
            return nil
        }

        private func containsKey(_ value: JSONValue, key: String) -> Bool {
            guard case let .object(obj) = value else { return false }
            return obj[key] != nil
        }

        private func makePrompt(_ text: String = "Continue.") -> Transcript.Prompt {
            Transcript.Prompt(segments: [.text(.init(content: text))])
        }

        private func makeTranscriptWithToolCalls() throws -> Transcript {
            let arguments = try GeneratedContent(json: #"{"city":"Paris"}"#)
            let call = Transcript.ToolCall(id: "call-1", toolName: "getWeather", arguments: arguments)
            let toolCalls = Transcript.ToolCalls([call])
            return Transcript(entries: [
                .toolCalls(toolCalls),
                .prompt(makePrompt()),
            ])
        }
    }
}

private func withOpenAIRateLimitRetry<T>(
    maxAttempts: Int = 4,
    operation: @escaping () async throws -> T
) async throws -> T {
    var attempt = 1
    while true {
        do {
            return try await operation()
        } catch let error as URLSessionError {
            if case .httpError(_, let detail) = error,
                detail.contains("rate_limit_exceeded"),
                attempt < maxAttempts
            {
                let delaySeconds = UInt64(attempt)
                try await Task.sleep(nanoseconds: delaySeconds * 1_000_000_000)
                attempt += 1
                continue
            }
            throw error
        } catch {
            throw error
        }
    }
}

// MARK: - Streaming Tool Call Tests (mocked)

@Suite("OpenAI streaming tool calls (mocked)", .serialized)
struct OpenAIStreamingToolCallTests {
    private let baseURL = URL(string: "https://mock-openai-streaming.local")!

    @Test func responsesStreamToolCallExecution() async throws {
        var requestCount = 0
        MockSSEURLProtocol.handler = { request in
            defer { requestCount += 1 }
            if requestCount == 0 {
                // First request: model returns a tool call
                return [
                    sseEvent(
                        #"{"type":"response.tool_call.created","tool_call":{"id":"call_1","type":"function","function":{"name":"getWeather","arguments":""}}}"#
                    ),
                    sseEvent(
                        #"{"type":"response.tool_call.delta","tool_call":{"id":"call_1","function":{"arguments":"{\"city\":\"San Francisco\"}"}}}"#
                    ),
                    sseEvent(#"{"type":"response.completed","finish_reason":"tool_calls"}"#),
                ]
            } else {
                // Second request: model returns text after tool results
                return [
                    sseEvent(#"{"type":"response.output_text.delta","delta":"Sunny and 72°F."}"#),
                    sseEvent(#"{"type":"response.completed","finish_reason":"stop"}"#),
                ]
            }
        }
        defer { MockSSEURLProtocol.handler = nil }

        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [MockSSEURLProtocol.self]

        let model = OpenAILanguageModel(
            baseURL: baseURL,
            apiKey: "test-key",
            model: "gpt-test",
            apiVariant: .responses,
            session: URLSession(configuration: config)
        )
        let session = LanguageModelSession(model: model, tools: [WeatherTool()])

        var snapshots: [LanguageModelSession.ResponseStream<String>.Snapshot] = []
        for try await snapshot in session.streamResponse(to: "What's the weather?") {
            snapshots.append(snapshot)
        }

        // Should have made 2 requests (tool call + continuation)
        #expect(requestCount == 2)
        #expect(!snapshots.isEmpty)

        // Final snapshot should have the text response
        let final = snapshots.last!
        #expect(final.rawContent.jsonString.contains("Sunny"))

        // The last snapshot's transcript entries should include tool call and output
        var hasToolCalls = false
        var hasToolOutput = false
        for entry in final.transcriptEntries {
            if case .toolCalls = entry { hasToolCalls = true }
            if case .toolOutput = entry { hasToolOutput = true }
        }
        #expect(hasToolCalls)
        #expect(hasToolOutput)
    }

    @Test func chatCompletionsStreamToolCallExecution() async throws {
        var requestCount = 0
        MockSSEURLProtocol.handler = { request in
            defer { requestCount += 1 }
            if requestCount == 0 {
                // First request: model returns tool call deltas
                return [
                    sseEvent(
                        #"{"id":"evt_1","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"getWeather","arguments":""}}]},"finish_reason":null}]}"#
                    ),
                    sseEvent(
                        #"{"id":"evt_1","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]},"finish_reason":null}]}"#
                    ),
                    sseEvent(
                        #"{"id":"evt_1","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Paris\"}"}}]},"finish_reason":null}]}"#
                    ),
                    sseEvent(#"{"id":"evt_1","choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#),
                ]
            } else {
                // Second request: model returns text after tool results
                return [
                    sseEvent(
                        #"{"id":"evt_2","choices":[{"delta":{"content":"Paris is sunny."},"finish_reason":null}]}"#
                    ),
                    sseEvent(#"{"id":"evt_2","choices":[{"delta":{},"finish_reason":"stop"}]}"#),
                ]
            }
        }
        defer { MockSSEURLProtocol.handler = nil }

        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [MockSSEURLProtocol.self]

        let model = OpenAILanguageModel(
            baseURL: baseURL,
            apiKey: "test-key",
            model: "gpt-test",
            apiVariant: .chatCompletions,
            session: URLSession(configuration: config)
        )
        let session = LanguageModelSession(model: model, tools: [WeatherTool()])

        var snapshots: [LanguageModelSession.ResponseStream<String>.Snapshot] = []
        for try await snapshot in session.streamResponse(to: "What's the weather in Paris?") {
            snapshots.append(snapshot)
        }

        // Should have made 2 requests (tool call + continuation)
        #expect(requestCount == 2)
        #expect(!snapshots.isEmpty)

        // Final snapshot should have the text response
        let final = snapshots.last!
        #expect(final.rawContent.jsonString.contains("Paris"))
    }

    @Test func streamingToolCallsAppearInTranscriptEntries() async throws {
        var requestCount = 0
        MockSSEURLProtocol.handler = { request in
            defer { requestCount += 1 }
            if requestCount == 0 {
                return [
                    sseEvent(
                        #"{"id":"evt_1","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"getWeather","arguments":"{\"city\":\"London\"}"}}]},"finish_reason":null}]}"#
                    ),
                    sseEvent(#"{"id":"evt_1","choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#),
                ]
            } else {
                return [
                    sseEvent(
                        #"{"id":"evt_2","choices":[{"delta":{"content":"London weather: sunny."},"finish_reason":null}]}"#
                    ),
                    sseEvent(#"{"id":"evt_2","choices":[{"delta":{},"finish_reason":"stop"}]}"#),
                ]
            }
        }
        defer { MockSSEURLProtocol.handler = nil }

        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [MockSSEURLProtocol.self]

        let model = OpenAILanguageModel(
            baseURL: baseURL,
            apiKey: "test-key",
            model: "gpt-test",
            apiVariant: .chatCompletions,
            session: URLSession(configuration: config)
        )
        let session = LanguageModelSession(model: model, tools: [WeatherTool()])

        let response = try await session.streamResponse(
            to: "What's the weather in London?"
        ).collect()

        // The response should include transcript entries for tool calls and outputs
        var hasToolCalls = false
        var hasToolOutput = false
        for entry in response.transcriptEntries {
            if case .toolCalls(let calls) = entry {
                hasToolCalls = true
                #expect(calls.first?.toolName == "getWeather")
            }
            if case .toolOutput(let output) = entry {
                hasToolOutput = true
                #expect(output.toolName == "getWeather")
            }
        }
        #expect(hasToolCalls)
        #expect(hasToolOutput)
        #expect(response.content.contains("London"))
    }

    @Test func streamingWithToolExecutionDelegateStop() async throws {
        var requestCount = 0
        MockSSEURLProtocol.handler = { request in
            defer { requestCount += 1 }
            return [
                sseEvent(
                    #"{"id":"evt_1","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"getWeather","arguments":"{\"city\":\"Berlin\"}"}}]},"finish_reason":null}]}"#
                ),
                sseEvent(#"{"id":"evt_1","choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#),
            ]
        }
        defer { MockSSEURLProtocol.handler = nil }

        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [MockSSEURLProtocol.self]

        let model = OpenAILanguageModel(
            baseURL: baseURL,
            apiKey: "test-key",
            model: "gpt-test",
            apiVariant: .chatCompletions,
            session: URLSession(configuration: config)
        )
        let session = LanguageModelSession(model: model, tools: [WeatherTool()])
        session.toolExecutionDelegate = StopDelegate()

        let response = try await session.streamResponse(
            to: "What's the weather?"
        ).collect()

        // With stop delegate, only 1 request should be made (no continuation)
        #expect(requestCount == 1)

        // Tool calls should be in transcript entries, but no tool output
        var hasToolCalls = false
        var hasToolOutput = false
        for entry in response.transcriptEntries {
            if case .toolCalls = entry { hasToolCalls = true }
            if case .toolOutput = entry { hasToolOutput = true }
        }
        #expect(hasToolCalls)
        #expect(!hasToolOutput)
    }

    // MARK: - Multiple parallel tool calls

    @Test func multipleParallelToolCalls() async throws {
        var requestCount = 0
        MockSSEURLProtocol.handler = { request in
            defer { requestCount += 1 }
            if requestCount == 0 {
                // Model calls both getWeather and calculate in parallel
                return [
                    sseEvent(
                        #"{"id":"evt_1","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"getWeather","arguments":"{\"city\":\"Tokyo\"}"}}]},"finish_reason":null}]}"#
                    ),
                    sseEvent(
                        #"{"id":"evt_1","choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_2","type":"function","function":{"name":"calculate","arguments":"{\"expression\":\"2+2\"}"}}]},"finish_reason":null}]}"#
                    ),
                    sseEvent(#"{"id":"evt_1","choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#),
                ]
            } else {
                return [
                    sseEvent(
                        #"{"id":"evt_2","choices":[{"delta":{"content":"Tokyo is sunny. 2+2=4."},"finish_reason":null}]}"#
                    ),
                    sseEvent(#"{"id":"evt_2","choices":[{"delta":{},"finish_reason":"stop"}]}"#),
                ]
            }
        }
        defer { MockSSEURLProtocol.handler = nil }

        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [MockSSEURLProtocol.self]

        let model = OpenAILanguageModel(
            baseURL: baseURL,
            apiKey: "test-key",
            model: "gpt-test",
            apiVariant: .chatCompletions,
            session: URLSession(configuration: config)
        )
        let session = LanguageModelSession(
            model: model,
            tools: [WeatherTool(), CalculateTool()]
        )

        let response = try await session.streamResponse(
            to: "What's the weather in Tokyo and what's 2+2?"
        ).collect()

        #expect(requestCount == 2)
        #expect(response.content.contains("Tokyo"))

        // Should have one toolCalls entry with both calls, plus two toolOutput entries
        var toolCallEntries: [Transcript.ToolCalls] = []
        var toolOutputEntries: [Transcript.ToolOutput] = []
        for entry in response.transcriptEntries {
            if case .toolCalls(let calls) = entry { toolCallEntries.append(calls) }
            if case .toolOutput(let output) = entry { toolOutputEntries.append(output) }
        }
        #expect(toolCallEntries.count == 1)
        #expect(toolCallEntries.first!.count == 2)
        #expect(toolOutputEntries.count == 2)

        let toolNames = Set(toolOutputEntries.map(\.toolName))
        #expect(toolNames.contains("getWeather"))
        #expect(toolNames.contains("calculate"))
    }

    // MARK: - Multiple rounds of tool calling

    @Test func multipleRoundsOfToolCalling() async throws {
        var requestCount = 0
        MockSSEURLProtocol.handler = { request in
            defer { requestCount += 1 }
            if requestCount == 0 {
                // First round: model calls getWeather for Tokyo
                return [
                    sseEvent(
                        #"{"id":"evt_1","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"getWeather","arguments":"{\"city\":\"Tokyo\"}"}}]},"finish_reason":null}]}"#
                    ),
                    sseEvent(#"{"id":"evt_1","choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#),
                ]
            } else if requestCount == 1 {
                // Second round: model calls getWeather for London
                return [
                    sseEvent(
                        #"{"id":"evt_2","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_2","type":"function","function":{"name":"getWeather","arguments":"{\"city\":\"London\"}"}}]},"finish_reason":null}]}"#
                    ),
                    sseEvent(#"{"id":"evt_2","choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#),
                ]
            } else {
                // Third request: model returns final text
                return [
                    sseEvent(
                        #"{"id":"evt_3","choices":[{"delta":{"content":"Tokyo and London are both sunny."},"finish_reason":null}]}"#
                    ),
                    sseEvent(#"{"id":"evt_3","choices":[{"delta":{},"finish_reason":"stop"}]}"#),
                ]
            }
        }
        defer { MockSSEURLProtocol.handler = nil }

        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [MockSSEURLProtocol.self]

        let model = OpenAILanguageModel(
            baseURL: baseURL,
            apiKey: "test-key",
            model: "gpt-test",
            apiVariant: .chatCompletions,
            session: URLSession(configuration: config)
        )
        let session = LanguageModelSession(model: model, tools: [WeatherTool()])

        let response = try await session.streamResponse(
            to: "Compare weather in Tokyo and London"
        ).collect()

        #expect(requestCount == 3)
        #expect(response.content.contains("Tokyo"))
        #expect(response.content.contains("London"))

        // Should have two toolCalls entries and two toolOutput entries
        var toolCallEntries: [Transcript.ToolCalls] = []
        var toolOutputEntries: [Transcript.ToolOutput] = []
        for entry in response.transcriptEntries {
            if case .toolCalls(let calls) = entry { toolCallEntries.append(calls) }
            if case .toolOutput(let output) = entry { toolOutputEntries.append(output) }
        }
        #expect(toolCallEntries.count == 2)
        #expect(toolOutputEntries.count == 2)
    }

    // MARK: - provideOutput delegate

    @Test func streamingWithProvideOutputDelegate() async throws {
        var requestCount = 0
        MockSSEURLProtocol.handler = { request in
            defer { requestCount += 1 }
            if requestCount == 0 {
                return [
                    sseEvent(
                        #"{"id":"evt_1","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"getWeather","arguments":"{\"city\":\"Rome\"}"}}]},"finish_reason":null}]}"#
                    ),
                    sseEvent(#"{"id":"evt_1","choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#),
                ]
            } else {
                return [
                    sseEvent(
                        #"{"id":"evt_2","choices":[{"delta":{"content":"Custom output was used."},"finish_reason":null}]}"#
                    ),
                    sseEvent(#"{"id":"evt_2","choices":[{"delta":{},"finish_reason":"stop"}]}"#),
                ]
            }
        }
        defer { MockSSEURLProtocol.handler = nil }

        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [MockSSEURLProtocol.self]

        let model = OpenAILanguageModel(
            baseURL: baseURL,
            apiKey: "test-key",
            model: "gpt-test",
            apiVariant: .chatCompletions,
            session: URLSession(configuration: config)
        )
        let weatherSpy = spy(on: WeatherTool())
        let session = LanguageModelSession(model: model, tools: [weatherSpy])
        session.toolExecutionDelegate = ProvideOutputDelegate(
            output: [.text(.init(content: "Stubbed: rainy in Rome"))]
        )

        let response = try await session.streamResponse(
            to: "What's the weather in Rome?"
        ).collect()

        // Should have made 2 requests (tool call with provided output + continuation)
        #expect(requestCount == 2)

        // The actual WeatherTool should NOT have been called
        let calls = await weatherSpy.calls
        #expect(calls.isEmpty)

        // Transcript entries should contain the provided output
        var foundProvidedOutput = false
        for entry in response.transcriptEntries {
            if case .toolOutput(let output) = entry {
                let text = output.segments.compactMap { seg -> String? in
                    if case .text(let t) = seg { return t.content }
                    return nil
                }.joined()
                if text.contains("Stubbed: rainy in Rome") {
                    foundProvidedOutput = true
                }
            }
        }
        #expect(foundProvidedOutput)
    }

    // MARK: - Tool not found

    @Test func streamingWithUnknownToolContinues() async throws {
        var requestCount = 0
        MockSSEURLProtocol.handler = { request in
            defer { requestCount += 1 }
            if requestCount == 0 {
                // Model calls a tool that doesn't exist
                return [
                    sseEvent(
                        #"{"id":"evt_1","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"nonExistentTool","arguments":"{}"}}]},"finish_reason":null}]}"#
                    ),
                    sseEvent(#"{"id":"evt_1","choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#),
                ]
            } else {
                // Model still responds after getting "tool not found" output
                return [
                    sseEvent(
                        #"{"id":"evt_2","choices":[{"delta":{"content":"I couldn't find that tool."},"finish_reason":null}]}"#
                    ),
                    sseEvent(#"{"id":"evt_2","choices":[{"delta":{},"finish_reason":"stop"}]}"#),
                ]
            }
        }
        defer { MockSSEURLProtocol.handler = nil }

        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [MockSSEURLProtocol.self]

        let model = OpenAILanguageModel(
            baseURL: baseURL,
            apiKey: "test-key",
            model: "gpt-test",
            apiVariant: .chatCompletions,
            session: URLSession(configuration: config)
        )
        let session = LanguageModelSession(model: model, tools: [WeatherTool()])

        let response = try await session.streamResponse(
            to: "Use the nonExistentTool"
        ).collect()

        // Should still complete: 2 requests (tool not found → continuation)
        #expect(requestCount == 2)
        #expect(response.content.contains("couldn't find"))

        // Tool output should contain the "not found" message
        var foundNotFound = false
        for entry in response.transcriptEntries {
            if case .toolOutput(let output) = entry {
                let text = output.segments.compactMap { seg -> String? in
                    if case .text(let t) = seg { return t.content }
                    return nil
                }.joined()
                if text.contains("Tool not found") {
                    foundNotFound = true
                }
            }
        }
        #expect(foundNotFound)
    }

    // MARK: - Tool execution failure

    @Test func streamingToolExecutionFailureThrows() async throws {
        MockSSEURLProtocol.handler = { request in
            [
                sseEvent(
                    #"{"id":"evt_1","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"failingTool","arguments":"{\"input\":\"fail\"}"}}]},"finish_reason":null}]}"#
                ),
                sseEvent(#"{"id":"evt_1","choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#),
            ]
        }
        defer { MockSSEURLProtocol.handler = nil }

        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [MockSSEURLProtocol.self]

        let model = OpenAILanguageModel(
            baseURL: baseURL,
            apiKey: "test-key",
            model: "gpt-test",
            apiVariant: .chatCompletions,
            session: URLSession(configuration: config)
        )
        let session = LanguageModelSession(model: model, tools: [FailingTool()])

        do {
            _ = try await session.streamResponse(
                to: "Call the failing tool"
            ).collect()
            Issue.record("Expected ToolCallError to be thrown")
        } catch let error as LanguageModelSession.ToolCallError {
            #expect(error.tool.name == "failingTool")
            #expect(error.underlyingError is FailingToolError)
        }
    }

    // MARK: - Structured output with tool calls

    @Test func structuredOutputWithToolCallsResponses() async throws {
        var requestCount = 0
        MockSSEURLProtocol.handler = { request in
            defer { requestCount += 1 }
            if requestCount == 0 {
                return [
                    sseEvent(
                        #"{"type":"response.tool_call.created","tool_call":{"id":"call_1","type":"function","function":{"name":"getWeather","arguments":""}}}"#
                    ),
                    sseEvent(
                        #"{"type":"response.tool_call.delta","tool_call":{"id":"call_1","function":{"arguments":"{\"city\":\"Denver\"}"}}}"#
                    ),
                    sseEvent(#"{"type":"response.completed","finish_reason":"tool_calls"}"#),
                ]
            } else {
                // Return JSON for structured output
                return [
                    sseEvent(
                        #"{"type":"response.output_text.delta","delta":"{\"city\":"}"#
                    ),
                    sseEvent(
                        #"{"type":"response.output_text.delta","delta":"\"Denver\",\"temperature\":72}"}"#
                    ),
                    sseEvent(#"{"type":"response.completed","finish_reason":"stop"}"#),
                ]
            }
        }
        defer { MockSSEURLProtocol.handler = nil }

        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [MockSSEURLProtocol.self]

        let model = OpenAILanguageModel(
            baseURL: baseURL,
            apiKey: "test-key",
            model: "gpt-test",
            apiVariant: .responses,
            session: URLSession(configuration: config)
        )
        let session = LanguageModelSession(model: model, tools: [WeatherTool()])

        var snapshots: [LanguageModelSession.ResponseStream<CityWeather>.Snapshot] = []
        for try await snapshot in session.streamResponse(
            to: "Get the weather in Denver",
            generating: CityWeather.self
        ) {
            snapshots.append(snapshot)
        }

        #expect(requestCount == 2)
        #expect(!snapshots.isEmpty)

        // The last snapshot should have parseable structured content
        let last = snapshots.last!
        #expect(last.content.city == "Denver")
        #expect(last.content.temperature == 72)

        // Should have tool entries
        var hasToolCalls = false
        for entry in last.transcriptEntries {
            if case .toolCalls = entry { hasToolCalls = true }
        }
        #expect(hasToolCalls)
    }

    // MARK: - Empty accumulator with tool_calls finish reason

    @Test func finishReasonToolCallsWithEmptyAccumulator() async throws {
        var requestCount = 0
        MockSSEURLProtocol.handler = { request in
            defer { requestCount += 1 }
            // Model says finish_reason is "tool_calls" but never sent any tool call deltas
            return [
                sseEvent(
                    #"{"id":"evt_1","choices":[{"delta":{"content":"Some text"},"finish_reason":null}]}"#
                ),
                sseEvent(#"{"id":"evt_1","choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#),
            ]
        }
        defer { MockSSEURLProtocol.handler = nil }

        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [MockSSEURLProtocol.self]

        let model = OpenAILanguageModel(
            baseURL: baseURL,
            apiKey: "test-key",
            model: "gpt-test",
            apiVariant: .chatCompletions,
            session: URLSession(configuration: config)
        )
        let session = LanguageModelSession(model: model, tools: [WeatherTool()])

        let response = try await session.streamResponse(
            to: "Test edge case"
        ).collect()

        // Should only make 1 request — the guard breaks out of the loop
        #expect(requestCount == 1)
        // Should still have the text content
        #expect(response.content.contains("Some text"))
    }

    // MARK: - Text mixed with tool calls in same response

    @Test func textContentMixedWithToolCalls() async throws {
        var requestCount = 0
        MockSSEURLProtocol.handler = { request in
            defer { requestCount += 1 }
            if requestCount == 0 {
                // Model sends some text before deciding to call a tool
                return [
                    sseEvent(
                        #"{"id":"evt_1","choices":[{"delta":{"content":"Let me check"},"finish_reason":null}]}"#
                    ),
                    sseEvent(
                        #"{"id":"evt_1","choices":[{"delta":{"content":" the weather."},"finish_reason":null}]}"#
                    ),
                    sseEvent(
                        #"{"id":"evt_1","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"getWeather","arguments":"{\"city\":\"Miami\"}"}}]},"finish_reason":null}]}"#
                    ),
                    sseEvent(#"{"id":"evt_1","choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#),
                ]
            } else {
                return [
                    sseEvent(
                        #"{"id":"evt_2","choices":[{"delta":{"content":"Miami is sunny and 85°F."},"finish_reason":null}]}"#
                    ),
                    sseEvent(#"{"id":"evt_2","choices":[{"delta":{},"finish_reason":"stop"}]}"#),
                ]
            }
        }
        defer { MockSSEURLProtocol.handler = nil }

        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [MockSSEURLProtocol.self]

        let model = OpenAILanguageModel(
            baseURL: baseURL,
            apiKey: "test-key",
            model: "gpt-test",
            apiVariant: .chatCompletions,
            session: URLSession(configuration: config)
        )
        let session = LanguageModelSession(model: model, tools: [WeatherTool()])

        var snapshots: [LanguageModelSession.ResponseStream<String>.Snapshot] = []
        for try await snapshot in session.streamResponse(to: "How's Miami?") {
            snapshots.append(snapshot)
        }

        #expect(requestCount == 2)

        // Text snapshots from the first request should have been yielded
        let textBeforeTools = snapshots.first { snap in
            snap.rawContent.jsonString.contains("Let me check")
        }
        #expect(textBeforeTools != nil)

        // Final response should contain the continuation text
        let final = snapshots.last!
        #expect(final.rawContent.jsonString.contains("Miami is sunny"))
    }

    // MARK: - Session transcript state after wrapStream completes

    @Test func sessionTranscriptContainsToolEntriesAfterStreaming() async throws {
        var requestCount = 0
        MockSSEURLProtocol.handler = { request in
            defer { requestCount += 1 }
            if requestCount == 0 {
                return [
                    sseEvent(
                        #"{"type":"response.tool_call.created","tool_call":{"id":"call_1","type":"function","function":{"name":"getWeather","arguments":""}}}"#
                    ),
                    sseEvent(
                        #"{"type":"response.tool_call.delta","tool_call":{"id":"call_1","function":{"arguments":"{\"city\":\"Seattle\"}"}}}"#
                    ),
                    sseEvent(#"{"type":"response.completed","finish_reason":"tool_calls"}"#),
                ]
            } else {
                return [
                    sseEvent(#"{"type":"response.output_text.delta","delta":"Seattle is rainy."}"#),
                    sseEvent(#"{"type":"response.completed","finish_reason":"stop"}"#),
                ]
            }
        }
        defer { MockSSEURLProtocol.handler = nil }

        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [MockSSEURLProtocol.self]

        let model = OpenAILanguageModel(
            baseURL: baseURL,
            apiKey: "test-key",
            model: "gpt-test",
            apiVariant: .responses,
            session: URLSession(configuration: config)
        )
        let session = LanguageModelSession(model: model, tools: [WeatherTool()])

        // Consume the stream fully so wrapStream appends to transcript
        for try await _ in session.streamResponse(to: "Seattle weather?") {}

        // Wait briefly for MainActor transcript updates
        try await Task.sleep(for: .milliseconds(50))

        let transcript = session.transcript
        var entryTypes: [String] = []
        for entry in transcript {
            switch entry {
            case .prompt: entryTypes.append("prompt")
            case .response: entryTypes.append("response")
            case .toolCalls: entryTypes.append("toolCalls")
            case .toolOutput: entryTypes.append("toolOutput")
            default: break
            }
        }

        // Transcript should contain: prompt, toolCalls, toolOutput, response (in that order)
        #expect(entryTypes.contains("prompt"))
        #expect(entryTypes.contains("toolCalls"))
        #expect(entryTypes.contains("toolOutput"))
        #expect(entryTypes.contains("response"))

        // Verify ordering: toolCalls and toolOutput come before response
        if let toolCallsIndex = entryTypes.firstIndex(of: "toolCalls"),
            let toolOutputIndex = entryTypes.firstIndex(of: "toolOutput"),
            let responseIndex = entryTypes.firstIndex(of: "response")
        {
            #expect(toolCallsIndex < responseIndex)
            #expect(toolOutputIndex < responseIndex)
        } else {
            Issue.record("Expected toolCalls, toolOutput, and response in transcript")
        }
    }

    // MARK: - Intermediate streaming snapshots during tool loop

    @Test func intermediateSnapshotsYieldedDuringToolLoop() async throws {
        var requestCount = 0
        MockSSEURLProtocol.handler = { request in
            defer { requestCount += 1 }
            if requestCount == 0 {
                return [
                    sseEvent(
                        #"{"id":"evt_1","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"getWeather","arguments":"{\"city\":\"Chicago\"}"}}]},"finish_reason":null}]}"#
                    ),
                    sseEvent(#"{"id":"evt_1","choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#),
                ]
            } else {
                return [
                    sseEvent(
                        #"{"id":"evt_2","choices":[{"delta":{"content":"Chicago"},"finish_reason":null}]}"#
                    ),
                    sseEvent(
                        #"{"id":"evt_2","choices":[{"delta":{"content":" is windy."},"finish_reason":null}]}"#
                    ),
                    sseEvent(#"{"id":"evt_2","choices":[{"delta":{},"finish_reason":"stop"}]}"#),
                ]
            }
        }
        defer { MockSSEURLProtocol.handler = nil }

        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [MockSSEURLProtocol.self]

        let model = OpenAILanguageModel(
            baseURL: baseURL,
            apiKey: "test-key",
            model: "gpt-test",
            apiVariant: .chatCompletions,
            session: URLSession(configuration: config)
        )
        let session = LanguageModelSession(model: model, tools: [WeatherTool()])

        var snapshots: [LanguageModelSession.ResponseStream<String>.Snapshot] = []
        for try await snapshot in session.streamResponse(to: "Chicago weather?") {
            snapshots.append(snapshot)
        }

        // We should see snapshots:
        // 1. After tool resolution (yielded with empty text + tool entries) — this is the intermediate snapshot
        // 2. "Chicago" text from continuation
        // 3. "Chicago is windy." text from continuation
        // 4. Final snapshot
        #expect(snapshots.count >= 3)

        // The intermediate snapshot (yielded after tool resolution, before continuation text) should have tool entries
        let snapshotsWithToolEntries = snapshots.filter { !$0.transcriptEntries.isEmpty }
        #expect(!snapshotsWithToolEntries.isEmpty)

        // The first snapshot with tool entries should have been yielded before the final text
        if let firstToolSnapshot = snapshotsWithToolEntries.first {
            var hasToolCalls = false
            var hasToolOutput = false
            for entry in firstToolSnapshot.transcriptEntries {
                if case .toolCalls = entry { hasToolCalls = true }
                if case .toolOutput = entry { hasToolOutput = true }
            }
            #expect(hasToolCalls)
            #expect(hasToolOutput)
        }

        // Final snapshot should have the complete text
        let final = snapshots.last!
        #expect(final.rawContent.jsonString.contains("Chicago is windy."))
    }

    // MARK: - Responses API: multiple parallel tool calls

    @Test func responsesMultipleParallelToolCalls() async throws {
        var requestCount = 0
        MockSSEURLProtocol.handler = { request in
            defer { requestCount += 1 }
            if requestCount == 0 {
                return [
                    sseEvent(
                        #"{"type":"response.tool_call.created","tool_call":{"id":"call_1","type":"function","function":{"name":"getWeather","arguments":""}}}"#
                    ),
                    sseEvent(
                        #"{"type":"response.tool_call.delta","tool_call":{"id":"call_1","function":{"arguments":"{\"city\":\"Paris\"}"}}}"#
                    ),
                    sseEvent(
                        #"{"type":"response.tool_call.created","tool_call":{"id":"call_2","type":"function","function":{"name":"getWeather","arguments":""}}}"#
                    ),
                    sseEvent(
                        #"{"type":"response.tool_call.delta","tool_call":{"id":"call_2","function":{"arguments":"{\"city\":\"Berlin\"}"}}}"#
                    ),
                    sseEvent(#"{"type":"response.completed","finish_reason":"tool_calls"}"#),
                ]
            } else {
                return [
                    sseEvent(
                        #"{"type":"response.output_text.delta","delta":"Paris and Berlin are both lovely."}"#
                    ),
                    sseEvent(#"{"type":"response.completed","finish_reason":"stop"}"#),
                ]
            }
        }
        defer { MockSSEURLProtocol.handler = nil }

        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [MockSSEURLProtocol.self]

        let model = OpenAILanguageModel(
            baseURL: baseURL,
            apiKey: "test-key",
            model: "gpt-test",
            apiVariant: .responses,
            session: URLSession(configuration: config)
        )
        let session = LanguageModelSession(model: model, tools: [WeatherTool()])

        let response = try await session.streamResponse(
            to: "Weather in Paris and Berlin?"
        ).collect()

        #expect(requestCount == 2)
        #expect(response.content.contains("Paris"))
        #expect(response.content.contains("Berlin"))

        var toolCallEntries: [Transcript.ToolCalls] = []
        var toolOutputEntries: [Transcript.ToolOutput] = []
        for entry in response.transcriptEntries {
            if case .toolCalls(let calls) = entry { toolCallEntries.append(calls) }
            if case .toolOutput(let output) = entry { toolOutputEntries.append(output) }
        }
        #expect(toolCallEntries.count == 1)
        #expect(toolCallEntries.first!.count == 2)
        #expect(toolOutputEntries.count == 2)
    }
}

// MARK: - Mock SSE URLProtocol

private func sseEvent(_ json: String) -> String {
    "data: \(json)\n\n"
}

private final class MockSSEURLProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var handler: ((URLRequest) -> [String])?

    override class func canInit(with request: URLRequest) -> Bool { true }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let handler = MockSSEURLProtocol.handler else {
            client?.urlProtocol(self, didFailWithError: URLError(.badServerResponse))
            return
        }

        let events = handler(request)
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 200,
            httpVersion: nil,
            headerFields: ["Content-Type": "text/event-stream"]
        )!

        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)

        let payload = events.joined()
        if let data = payload.data(using: .utf8) {
            client?.urlProtocol(self, didLoad: data)
        }
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

// MARK: - Test Delegates

private final class StopDelegate: ToolExecutionDelegate {
    func toolCallDecision(
        for toolCall: Transcript.ToolCall,
        in session: LanguageModelSession
    ) async -> ToolExecutionDecision {
        .stop
    }
}

private final class ProvideOutputDelegate: ToolExecutionDelegate {
    let outputSegments: [Transcript.Segment]

    init(output: [Transcript.Segment]) {
        self.outputSegments = output
    }

    func toolCallDecision(
        for toolCall: Transcript.ToolCall,
        in session: LanguageModelSession
    ) async -> ToolExecutionDecision {
        .provideOutput(outputSegments)
    }
}
