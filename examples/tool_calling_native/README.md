# Tool Calling Native Example

This example demonstrates how to use llama.cpp's native tool calling functionality in Rust, providing the same capabilities as the C++ `tool_calling_native.cpp` example.

## 🎯 Features

- **Real inference** with actual model execution (not simulated)
- **Native tool calling** using llama.cpp's built-in chat templates
- **Tool execution** with result integration back into the conversation
- **Template detection** automatically detects if the model supports tool calling
- **Complete workflow** from prompt to tool execution to final response
- **Command-line interface** similar to the C++ example

## 📋 Usage

```bash
cargo run --release -- --model /path/to/your/model.gguf [OPTIONS]
```

### Options

- `-m, --model <MODEL_PATH>` - Model file (GGUF format) **[Required]**
- `-p, --prompt <PROMPT>` - User prompt (default: "Calculate 15 + 25")
- `-c, --ctx-size <SIZE>` - Context size (default: 4096)
- `-v, --verbose` - Enable verbose output (shows generated prompts)
- `-h, --help` - Show help information

### Examples

```bash
# Basic usage with default calculator prompt
cargo run --release -- --model ./models/llama-3.1-8b-instruct.gguf

# Custom prompt with verbose output
cargo run --release -- --model ./models/hermes-2-pro-7b.gguf \
  --prompt "What is 42 multiplied by 13?" \
  --verbose

# Larger context size for complex conversations
cargo run --release -- --model ./models/functionary-v3.2.gguf \
  --ctx-size 8192 \
  --prompt "Help me calculate some math problems"
```

## 🤖 Supported Models

This example works best with models that support native tool calling:

- **Llama 3.1+ Instruct** (8B, 70B, 405B)
- **Hermes 2 Pro** (7B, 8B)
- **Functionary v3.1/v3.2**
- **Qwen 2.5 Instruct**
- **Mistral Nemo Instruct**
- **Command R/R+**

For models without native tool calling support, the example will fall back to a generic format.

## 🛠️ What it demonstrates

1. **Model loading and initialization**

   - Backend initialization
   - Model loading with GGUF format
   - Chat template initialization

2. **Tool definition and setup**

   - Creating tool schemas with JSON parameters
   - Tool choice configuration (auto, required, none)
   - Multiple tool support

3. **Template application with tools**

   - Automatic template format detection
   - Tool integration into conversation templates
   - Prompt generation for inference

4. **Real inference execution**

   - Context creation and configuration
   - Sampling setup (temperature, top-p, etc.)
   - Token generation and streaming output
   - End-of-generation detection

5. **Tool call parsing and execution**

   - Response parsing to extract tool calls
   - JSON argument parsing
   - Tool execution with error handling
   - Result integration back into conversation

6. **Complete conversation flow**
   - Multi-turn conversation support
   - Tool result integration
   - Continuation prompt generation

## 🧮 Built-in Calculator Tool

The example includes a simple calculator tool that can:

- Add numbers: "15 + 25" → 40
- Multiply numbers: "42 \* 13" → 546
- Handle unknown expressions with graceful fallback

This matches the behavior of the C++ example exactly.

## 📊 Output Example

```
🦙 Native Tool Calling Example
================================

🔧 Initializing llama backend...
✅ Backend initialized
📂 Loading model: ./models/llama-3.1-8b-instruct.gguf
✅ Model loaded successfully

🎭 Initializing chat templates...
✅ Templates initialized
Template was explicit: true

🛠️  Setting up tools...
✓ Calculator tool added

💬 Creating conversation...
👤 User: Calculate 15 + 25

🎨 Applying chat template with tools...
✅ Template applied successfully
🎭 Format: Llama3XWithBuiltinTools
📏 Prompt length: 892 characters

🚀 Starting inference...
✅ Context created (size: 4096)
📊 Prompt tokens: 178

🤖 Assistant: I'll help you calculate 15 + 25.

<tool_call>
{"name": "calculator", "arguments": {"expression": "15 + 25"}, "id": "call_123"}
</tool_call>

✅ Inference completed!

🔍 Parsing response for tool calls...
✅ Response parsed successfully
Role: assistant
Content: I'll help you calculate 15 + 25.

🛠️ Tool calls found:
  - Name: calculator
    Arguments: {"expression": "15 + 25"}
    ID: call_123
🔧 Executing calculator with: {"expression": "15 + 25"}
    Result: {"result": 40}

🔄 Tool execution result added to conversation.
📋 Final prompt for continuation ready (1157 chars)

🎯 Tool calling template applied successfully!
📋 Summary:
   • Model: ./models/llama-3.1-8b-instruct.gguf
   • Tools: 1 defined
   • Template format: Llama3XWithBuiltinTools
   • Prompt processed and inference completed

✅ Model supports native tool calling!
🚀 Tool calling workflow completed successfully!

🧹 Cleaning up...
✅ Tool calling example completed successfully!
```

## 🔍 Technical Details

- **C Wrapper Integration**: Uses the C wrapper layer to access `common/chat.h` functions
- **Memory Management**: Proper cleanup of all resources (model, context, sampler)
- **Error Handling**: Comprehensive error handling throughout the workflow
- **Performance**: Release build optimizations for fast inference
- **Safety**: All unsafe operations are properly wrapped in safe Rust APIs

## 🚀 Performance

The example is optimized for performance:

- Uses release builds by default
- Efficient memory management
- Streaming token generation
- Minimal allocations during inference

## 🔧 Requirements

- Rust 1.70+
- A GGUF model file (preferably tool-calling capable)
- Sufficient memory for the model and context

## 📝 Notes

- This implementation provides the same functionality as the C++ `tool_calling_native.cpp` example
- The calculator tool uses hardcoded results for demonstration (matching the C++ version)
- Tool calling support depends on the model's built-in capabilities
- Use `--verbose` to see the generated prompts and debug information

For more advanced tool calling scenarios, extend the tool definitions and execution logic as needed.
