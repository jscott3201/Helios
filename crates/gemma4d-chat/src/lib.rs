#![doc = "Gemma4D chat prompt compilation and prompt fixture parsing."]

use gemma4d_tokenizer::sha256_hex;
use serde::{Deserialize, Serialize};
use std::{
    fmt, fs,
    path::{Path, PathBuf},
};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },
    EmptyMessages,
    InvalidFixture(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "{}: {}", path.display(), source),
            Self::Json { path, source } => write!(f, "{}: {}", path.display(), source),
            Self::EmptyMessages => write!(f, "chat prompt requires at least one message"),
            Self::InvalidFixture(message) => write!(f, "invalid prompt fixture: {message}"),
        }
    }
}

impl std::error::Error for Error {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    fn template_name(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingMode {
    Disabled,
    Enabled,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ChatTemplateConfig {
    #[serde(default = "default_template_version")]
    pub template_version: String,
    #[serde(default)]
    pub add_generation_prompt: bool,
    #[serde(default = "default_thinking_mode")]
    pub thinking_mode: ThinkingMode,
}

impl Default for ChatTemplateConfig {
    fn default() -> Self {
        Self {
            template_version: default_template_version(),
            add_generation_prompt: true,
            thinking_mode: ThinkingMode::Disabled,
        }
    }
}

fn default_template_version() -> String {
    "gemma4d-fixture-chat-v1".to_owned()
}

fn default_thinking_mode() -> ThinkingMode {
    ThinkingMode::Disabled
}

pub fn compile_prompt(messages: &[ChatMessage], config: &ChatTemplateConfig) -> Result<String> {
    if messages.is_empty() {
        return Err(Error::EmptyMessages);
    }

    let mut rendered = String::from("<bos>");
    for message in messages {
        rendered.push_str("<start_of_turn>");
        rendered.push_str(message.role.template_name());
        rendered.push('\n');
        rendered.push_str(&message.content);
        rendered.push_str("<end_of_turn>\n");
    }

    if config.add_generation_prompt {
        rendered.push_str("<start_of_turn>assistant\n");
        if config.thinking_mode == ThinkingMode::Enabled {
            rendered.push_str("<thinking>enabled</thinking>\n");
        }
    }

    Ok(rendered)
}

pub fn template_hash(config: &ChatTemplateConfig) -> Result<String> {
    let bytes = serde_json::to_vec(config).map_err(|source| Error::Json {
        path: PathBuf::from("<chat-template-config>"),
        source,
    })?;
    Ok(sha256_hex(&bytes))
}

#[derive(Debug, Clone, Deserialize)]
pub struct PromptFixture {
    pub name: String,
    #[serde(default)]
    pub reference_source: String,
    pub template: ChatTemplateConfig,
    pub messages: Vec<FixtureMessage>,
    pub expected_prompt: Option<String>,
    pub expected_token_ids: ExpectedTokenIds,
    #[serde(default)]
    pub expected_stop_tokens: Vec<String>,
}

impl PromptFixture {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(|source| Error::Io {
            path: path.to_owned(),
            source,
        })?;
        serde_json::from_slice(&bytes).map_err(|source| Error::Json {
            path: path.to_owned(),
            source,
        })
    }

    pub fn chat_messages(&self) -> Vec<ChatMessage> {
        self.messages
            .iter()
            .map(|message| ChatMessage {
                role: message.role,
                content: message.content(),
            })
            .collect()
    }

    pub fn expected_token_ids(&self) -> Vec<u32> {
        self.expected_token_ids.expand()
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct FixtureMessage {
    pub role: Role,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub repeat: Option<RepeatText>,
}

impl FixtureMessage {
    fn content(&self) -> String {
        match (&self.content, &self.repeat) {
            (Some(content), None) => content.clone(),
            (None, Some(repeat)) => repeat.text.repeat(repeat.times),
            (Some(content), Some(repeat)) => {
                let mut expanded = content.clone();
                expanded.push_str(&repeat.text.repeat(repeat.times));
                expanded
            }
            (None, None) => String::new(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct RepeatText {
    pub text: String,
    pub times: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ExpectedTokenIds {
    Exact(Vec<u32>),
    Pattern(ExpectedTokenPattern),
}

impl ExpectedTokenIds {
    fn expand(&self) -> Vec<u32> {
        match self {
            Self::Exact(ids) => ids.clone(),
            Self::Pattern(pattern) => {
                let mut ids = pattern.prefix.clone();
                for _ in 0..pattern.repeat_times {
                    ids.extend_from_slice(&pattern.repeat);
                }
                ids.extend_from_slice(&pattern.suffix);
                ids
            }
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ExpectedTokenPattern {
    pub prefix: Vec<u32>,
    pub repeat: Vec<u32>,
    pub repeat_times: usize,
    pub suffix: Vec<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use gemma4d_tokenizer::FixtureTokenizer;

    fn fixture(path: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures")
            .join(path)
    }

    #[test]
    fn compiles_system_user_and_generation_prompt() {
        let messages = vec![
            ChatMessage {
                role: Role::System,
                content: "You are Gemma4D.".to_owned(),
            },
            ChatMessage {
                role: Role::User,
                content: "Hello".to_owned(),
            },
        ];
        let rendered = compile_prompt(&messages, &ChatTemplateConfig::default()).expect("prompt");
        assert_eq!(
            rendered,
            "<bos><start_of_turn>system\nYou are Gemma4D.<end_of_turn>\n<start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>assistant\n"
        );
    }

    #[test]
    fn thinking_mode_changes_rendered_prompt_and_hash() {
        let disabled = ChatTemplateConfig::default();
        let enabled = ChatTemplateConfig {
            thinking_mode: ThinkingMode::Enabled,
            ..ChatTemplateConfig::default()
        };
        let messages = [ChatMessage {
            role: Role::User,
            content: "Hello".to_owned(),
        }];

        let rendered = compile_prompt(&messages, &enabled).expect("prompt");
        assert!(rendered.contains("<thinking>enabled</thinking>"));
        assert_ne!(
            template_hash(&disabled).expect("hash"),
            template_hash(&enabled).expect("hash")
        );
    }

    #[test]
    fn loads_prompt_fixture() {
        let prompt = PromptFixture::load(fixture("prompts/simple_chat.json")).expect("fixture");
        let rendered = compile_prompt(&prompt.chat_messages(), &prompt.template).expect("prompt");
        assert_eq!(Some(rendered), prompt.expected_prompt);
        assert_eq!(
            prompt.expected_token_ids(),
            vec![1, 3, 6, 9, 20, 4, 9, 3, 7, 9]
        );
    }

    #[test]
    fn all_prompt_fixture_token_ids_match_reference() {
        let tokenizer =
            FixtureTokenizer::load(fixture("tokenizer/tokenizer.json")).expect("tokenizer");
        let fixture_names = [
            "simple_chat",
            "system_prompt",
            "code_rust",
            "code_python",
            "long_prefix_4k",
            "tool_call_shape",
        ];

        for name in fixture_names {
            let fixture_path = fixture(&format!("prompts/{name}.json"));
            let prompt = PromptFixture::load(&fixture_path).expect("fixture");
            assert_eq!(prompt.name, name);

            let rendered =
                compile_prompt(&prompt.chat_messages(), &prompt.template).expect("prompt");
            if let Some(expected_prompt) = &prompt.expected_prompt {
                assert_eq!(&rendered, expected_prompt, "{name} rendered prompt");
            }

            let actual_ids = tokenizer.tokenize(&rendered);
            assert_eq!(actual_ids, prompt.expected_token_ids(), "{name} token ids");

            let stop_ids: Vec<u32> = prompt
                .expected_stop_tokens
                .iter()
                .map(|token| tokenizer.token_id(token).expect("known stop token"))
                .collect();
            assert_eq!(stop_ids, tokenizer.stop_token_ids(), "{name} stop tokens");
        }
    }
}
