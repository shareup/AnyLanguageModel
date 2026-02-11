import AnyLanguageModel

@Generable
struct CityWeather {
    @Guide(description: "The city name")
    var city: String
    @Guide(description: "The temperature in Fahrenheit")
    var temperature: Int
}

struct CalculateTool: Tool {
    let name = "calculate"
    let description = "Evaluate a mathematical expression"

    @Generable
    struct Arguments {
        @Guide(description: "The mathematical expression to evaluate")
        var expression: String
    }

    func call(arguments: Arguments) async throws -> String {
        "Result: 4"
    }
}

struct FailingTool: Tool {
    let name = "failingTool"
    let description = "A tool that always fails"

    @Generable
    struct Arguments {
        @Guide(description: "Input")
        var input: String
    }

    func call(arguments: Arguments) async throws -> String {
        throw FailingToolError.intentionalFailure
    }
}

enum FailingToolError: Error {
    case intentionalFailure
}
