use std::collections::HashMap;
use std::path::Path;
use anyhow::Result;

pub struct StubTagger {
    name: String,
}

impl StubTagger {
    pub fn new(name: &str) -> Self { Self { name: name.to_string() } }
}

impl super::Model for StubTagger {
    fn infer(&self, _image_path: &Path) -> Result<super::ModelOutput> {
        let mut tags = HashMap::new();
        tags.insert("stub_tag".to_string(), 0.99f32);
        Ok(super::ModelOutput::Tags(tags))
    }
}
