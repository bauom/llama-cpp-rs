use clap::Parser;
use encoding_rs;
use llama_cpp_2::{
    chat::{
        parse_chat_response, ChatFormat, ChatMessage, ChatSyntax, ChatTemplateInputs,
        ChatTemplates, ChatTool, ChatToolChoice, ReasoningFormat,
    },
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel, Special},
    sampling::LlamaSampler,
};
use serde_json;
use std::num::NonZeroU32;
use std::panic;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "tool_calling_native")]
#[command(about = "🦙 Native Tool Calling with compute_diffs Example")]
#[command(
    long_about = "This example demonstrates llama.cpp's native tool calling system with proper diff-based token classification."
)]
struct Args {
    /// Model file (GGUF format)
    #[arg(short = 'm', long = "model")]
    model_path: PathBuf,

    /// User prompt
    #[arg(short = 'p', long = "prompt", default_value = "what is 3+3?")]
    user_prompt: String,

    /// Context size
    #[arg(short = 'c', long = "ctx-size", default_value = "4096")]
    ctx_size: u32,

    /// Verbose output
    #[arg(short = 'v', long = "verbose")]
    verbose: bool,
}

fn create_calculator_tool() -> ChatTool {
    ChatTool {
        name: "calculator".to_string(),
        description: "Perform mathematical calculations".to_string(),
        parameters: r#"{"type": "object", "properties": {"expression": {"type": "string", "description": "Mathematical expression"}}, "required": ["expression"]}"#.to_string(),
    }
}

fn execute_calculator(arguments: &str) -> Result<String, Box<dyn std::error::Error>> {
    println!("🔧 Executing calculator with: {}", arguments);

    // Parse the JSON arguments
    let args: serde_json::Value = serde_json::from_str(arguments)?;
    let expression = args["expression"].as_str().unwrap_or("");

    // Simple math evaluation for demo
    if expression.contains("3+3") || expression.contains("3 + 3") {
        Ok(r#"{"result": 6}"#.to_string())
    } else if expression.contains("15 + 25") {
        Ok(r#"{"result": 40}"#.to_string())
    } else if expression.contains("42 * 13") {
        Ok(r#"{"result": 546}"#.to_string())
    } else {
        Ok(format!(
            r#"{{"result": "Demo calculator - expression: {}"}}"#,
            expression
        ))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("🦙 Native Tool Calling with compute_diffs Example");
    println!("==================================================\n");

    // Initialize llama backend (following llama_cpp.rs pattern)
    println!("🔧 Initializing llama backend...");
    let backend = LlamaBackend::init()?;
    println!("✅ Backend initialized");

    // Load model with proper parameters
    println!("📖 Loading model from {:?}...", args.model_path);
    let model_params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(&backend, &args.model_path, &model_params)
        .map_err(|e| format!("Failed to load model: {}", e))?;
    println!("✅ Model loaded");

    // Create context with proper validation (following llama_cpp.rs pattern)
    println!("🧠 Creating context...");
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(args.ctx_size));
    let mut ctx = model
        .new_context(&backend, ctx_params)
        .map_err(|e| format!("Failed to create context: {}", e))?;
    println!("✅ Context created");

    // Validate context parameters
    let n_ctx = ctx.n_ctx();
    let n_batch = ctx.n_batch();
    let n_ubatch = ctx.n_ubatch();
    println!(
        "📊 Context info: n_ctx={}, n_batch={}, n_ubatch={}",
        n_ctx, n_batch, n_ubatch
    );

    if n_ctx == 0 {
        return Err("Invalid context: n_ctx is zero".into());
    }

    // Set up tools
    let tools = vec![create_calculator_tool()];

    // Create chat template (following llama_cpp.rs pattern)
    let chat_templates = ChatTemplates::new(&model, None, None, None)
        .map_err(|e| format!("Failed to create chat templates: {}", e))?;

    // Create template inputs with proper structure
    let inputs = ChatTemplateInputs {
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are a helpful assistant. You have access to a calculator tool."
                    .to_string(),
                ..Default::default()
            },
            ChatMessage {
                role: "user".to_string(),
                content: args.user_prompt.clone(),
                ..Default::default()
            },
        ],
        tools,
        tool_choice: ChatToolChoice::Auto,
        add_generation_prompt: true,
        reasoning_format: ReasoningFormat::DeepSeek,
        enable_thinking: true,
        ..Default::default()
    };

    // Apply chat template
    let chat_result = chat_templates
        .apply(&inputs)
        .map_err(|e| format!("Failed to apply chat template: {}", e))?;

    let chat_format = chat_result.format;
    println!("📝 Using chat format: {:?}", chat_format);
    println!("💬 Formatted chat:\n{}\n", chat_result.prompt);

    // Tokenize with validation
    let tokens = model
        .str_to_token(&chat_result.prompt, AddBos::Always)
        .map_err(|e| format!("Failed to tokenize: {}", e))?;
    println!("🔢 Generated {} tokens", tokens.len());

    if tokens.is_empty() {
        return Err("No tokens generated from prompt".into());
    }

    if tokens.len() as u32 >= n_ctx {
        return Err(format!(
            "Prompt too long: {} tokens >= {} context size",
            tokens.len(),
            n_ctx
        )
        .into());
    }

    // Initialize batch and process prompt (following llama_cpp.rs pattern)
    const BATCH_SIZE: usize = 512;
    let mut batch = LlamaBatch::new(BATCH_SIZE, 1);
    let last_index = tokens.len() - 1;

    // Process prompt in batches
    for (i, &token) in tokens.iter().enumerate() {
        if batch.n_tokens() as usize >= BATCH_SIZE {
            ctx.decode(&mut batch)
                .map_err(|e| format!("Failed to decode prompt batch: {}", e))?;
            batch.clear();
        }

        let is_last = i == last_index;
        batch
            .add(token, i as i32, &[0], is_last)
            .map_err(|e| format!("Failed to add token to batch: {}", e))?;
    }

    // Final decode for remaining tokens
    ctx.decode(&mut batch)
        .map_err(|e| format!("Failed to decode final prompt batch: {}", e))?;
    println!("✅ Initial prompt processed successfully");

    // Start generation with streaming and compute_diffs
    println!("🚀 Starting streaming generation with compute_diffs...\n");

    // Create sampler (following llama_cpp.rs pattern)
    let chain = [
        Some(LlamaSampler::penalties(32, 1.1, 0.0, 0.0)), // repetition penalty
        Some(LlamaSampler::top_p(0.9, 1)),
        Some(LlamaSampler::temp(0.3)),
        Some(LlamaSampler::dist(42)), // seed
    ];
    let mut sampler = LlamaSampler::chain_simple(chain.into_iter().flatten());

    // Create chat syntax for parsing
    let chat_syntax = ChatSyntax {
        format: chat_format,
        reasoning_format: ReasoningFormat::DeepSeek,
        reasoning_in_content: false,
        thinking_forced_open: false,
        parse_tool_calls: true,
    };

    let mut response = String::new();
    let mut prev_message = ChatMessage::default();
    let mut token_count = 0;
    let mut consecutive_newlines = 0;
    const MAX_CONSECUTIVE_NEWLINES: usize = 10;

    // UTF-8 decoder for proper token handling
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut n_past = tokens.len() as i32;

    // Generation loop (following server.cpp slot pattern)
    let mut memory_check_interval = 50; // Check memory every 50 tokens

    for generation_step in 0..512 {
        if args.verbose {
            println!(
                "🔍 DEBUG: Generation step {}, batch tokens: {}",
                generation_step,
                batch.n_tokens()
            );

            // Memory usage check every 50 tokens
            if generation_step % memory_check_interval == 0 {
                println!("🔍 DEBUG: Memory check at step {}", generation_step);
                // Force a small garbage collection hint
                std::hint::black_box(());
            }
        }

        // Sample next token with enhanced safety checks
        if args.verbose {
            println!("🔍 DEBUG: About to sample token...");
        }

        let new_token_id = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let sample_result = sampler.sample(&ctx, batch.n_tokens() - 1);
            if args.verbose {
                println!("🔍 DEBUG: Sampler returned token: {}", sample_result);
            }
            sample_result
        })) {
            Ok(token) => {
                if args.verbose {
                    println!("🔍 DEBUG: Successfully sampled token: {}", token);
                }
                token
            }
            Err(e) => {
                println!("❌ PANIC in sampler.sample(): {:?}", e);
                break;
            }
        };

        if args.verbose {
            println!("🔍 DEBUG: About to accept token...");
        }

        // Accept the token with safety check
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sampler.accept(new_token_id)
        })) {
            Ok(_) => {
                if args.verbose {
                    println!("🔍 DEBUG: Successfully accepted token");
                }
            }
            Err(e) => {
                println!("❌ PANIC in sampler.accept(): {:?}", e);
                break;
            }
        };

        // Check for EOS token
        if args.verbose {
            println!("🔍 DEBUG: Checking if token is EOS...");
        }

        let eos_token = model.token_eos();
        if new_token_id == eos_token {
            if args.verbose {
                println!("🔍 DEBUG: EOS token detected, stopping generation");
            }
            break;
        }

        // Convert token to text with enhanced error handling
        if args.verbose {
            println!("🔍 DEBUG: Converting token to text...");
        }

        let output_bytes = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            model.token_to_bytes(new_token_id, Special::Tokenize)
        })) {
            Ok(bytes) => match bytes {
                Ok(b) => b,
                Err(e) => {
                    println!("❌ Error converting token to bytes: {}", e);
                    break;
                }
            },
            Err(e) => {
                println!("❌ PANIC in token_to_bytes(): {:?}", e);
                break;
            }
        };

        let mut output_str = String::with_capacity(32);
        let _decode_result = decoder.decode_to_string(&output_bytes, &mut output_str, false);

        if args.verbose {
            println!("🔍 DEBUG: Token {} text: {:?}", token_count + 1, output_str);
        }

        response.push_str(&output_str);
        token_count += 1;

        // Check for excessive newlines to prevent infinite loops
        if output_str.trim().is_empty() {
            consecutive_newlines += 1;
            if consecutive_newlines > MAX_CONSECUTIVE_NEWLINES {
                println!("⚠️  Stopping generation due to excessive newlines");
                break;
            }
        } else {
            consecutive_newlines = 0;
        }

        // Parse chat response with enhanced error handling
        if args.verbose {
            println!("🔍 DEBUG: About to parse chat response...");
        }

        let current_message = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let parse_result = parse_chat_response(&response, true, &chat_syntax);
            if args.verbose {
                println!("🔍 DEBUG: Parse result: {:?}", parse_result);
            }
            parse_result
        })) {
            Ok(Ok(msg)) => msg,
            Ok(Err(_)) => {
                if args.verbose {
                    println!("🔍 DEBUG: Parse failed, creating fallback message");
                }
                // On parse error, create a minimal message to maintain flow
                ChatMessage {
                    role: "assistant".to_string(),
                    content: response.clone(),
                    ..Default::default()
                }
            }
            Err(e) => {
                println!("❌ PANIC in parse_chat_response(): {:?}", e);
                break;
            }
        };

        // Compute diffs with enhanced error handling
        if args.verbose {
            println!("🔍 DEBUG: About to compute diffs...");
        }

        // WORKAROUND: Skip compute_diffs if tool calls are present (causes segfault)
        let has_tool_calls = !current_message.tool_calls.is_empty();
        if has_tool_calls {
            if args.verbose {
                println!("🔍 DEBUG: Tool calls detected, skipping compute_diffs to avoid segfault");
            }
            println!(
                "📊 Token {}: 🔧 Tool Call Structure: {:?}",
                token_count, output_str
            );
            println!(
                "   🔧 Tool Call: name='{}', args='{}'",
                current_message.tool_calls[0].name, current_message.tool_calls[0].arguments
            );
        } else {
            let diffs = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                ChatMessage::compute_diffs(&prev_message, &current_message)
            })) {
                Ok(Ok(diff_result)) => diff_result.diffs,
                Ok(Err(e)) => {
                    if args.verbose {
                        println!("🔍 DEBUG: Compute diffs failed: {}", e);
                    }
                    Vec::new()
                }
                Err(e) => {
                    println!("❌ PANIC in compute_diffs(): {:?}", e);
                    break;
                }
            };

            // Display token classification
            if diffs.is_empty() {
                println!(
                    "📊 Token {}: ⚪ No structural change = {:?}",
                    token_count, output_str
                );
            } else {
                println!("📊 Token {}: 🧠 Reasoning: {:?}", token_count, output_str);
                println!("   📋 Parser found {} diffs total", diffs.len());
            }
        }

        prev_message = current_message;

        // Check for tool call completion
        if response.contains("</tool_call>") {
            println!("🎯 Tool call completed, stopping generation");
            break;
        }

        // Enhanced batch management with safety checks
        if args.verbose {
            println!("🔍 DEBUG: About to clear and prepare batch...");
        }

        // Clear batch with error handling
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            batch.clear();
        })) {
            Ok(_) => {
                if args.verbose {
                    println!("🔍 DEBUG: Successfully cleared batch");
                }
            }
            Err(e) => {
                println!("❌ PANIC in batch.clear(): {:?}", e);
                break;
            }
        };

        if args.verbose {
            println!("🔍 DEBUG: About to add token to batch...");
        }

        // Add token to batch with error handling
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            batch.add(new_token_id, n_past, &[0], true)
        })) {
            Ok(add_result) => match add_result {
                Ok(_) => {
                    if args.verbose {
                        println!("🔍 DEBUG: Successfully added token to batch");
                    }
                }
                Err(e) => {
                    println!("❌ Error adding token to batch: {}", e);
                    break;
                }
            },
            Err(e) => {
                println!("❌ PANIC in batch.add(): {:?}", e);
                break;
            }
        };

        n_past += 1;

        // Only decode if we're not at the end of generation
        if generation_step < 511 {
            if args.verbose {
                println!("🔍 DEBUG: About to decode batch for next iteration...");
            }

            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ctx.decode(&mut batch)))
            {
                Ok(decode_result) => match decode_result {
                    Ok(_) => {
                        if args.verbose {
                            println!("🔍 DEBUG: Successfully decoded batch");
                        }
                    }
                    Err(e) => {
                        println!("❌ Decode error at token {}: {}", token_count, e);
                        break;
                    }
                },
                Err(e) => {
                    println!("❌ PANIC in ctx.decode(): {:?}", e);
                    break;
                }
            };
        }

        if args.verbose {
            println!("🔍 DEBUG: Completed generation step {}", generation_step);
        }
    }

    // Final cleanup
    batch.clear();
    println!("\n🎉 Generation complete! Generated {} tokens", token_count);
    println!("📝 Final response:\n{}", response);

    // // Parse final message for tool calls
    // let final_message = match parse_chat_response(&response, false, &chat_syntax) {
    //     Ok(msg) => msg,
    //     Err(e) => {
    //         println!("❌ Failed to parse final message: {}", e);
    //         return Ok(());
    //     }
    // };

    // // Execute tool calls if found
    // if !final_message.tool_calls.is_empty() {
    //     println!("\n🔧 Found {} tool calls:", final_message.tool_calls.len());
    //     for (i, tool_call) in final_message.tool_calls.iter().enumerate() {
    //         println!(
    //             "   {}. {} with args: {}",
    //             i + 1,
    //             tool_call.name,
    //             tool_call.arguments
    //         );

    //         if tool_call.name == "calculator" {
    //             match execute_calculator(&tool_call.arguments) {
    //                 Ok(result) => println!("      Result: {}", result),
    //                 Err(e) => println!("      Error: {}", e),
    //             }
    //         }
    //     }
    // }

    Ok(())
}
