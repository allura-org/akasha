use std::path::Path;
use anyhow::Result;
use candle_core::{Device, Tensor};
use image::imageops::FilterType;

pub fn image_to_tensor(path: &Path, size: usize, device: &Device) -> Result<Tensor> {
    let img = image::open(path)?;
    let img = img.resize_to_fill(size as u32, size as u32, FilterType::Lanczos3);
    let rgb = img.to_rgb8();
    let pixels: Vec<f32> = rgb.pixels().flat_map(|p| {
        let r = p[0] as f32 / 255.0;
        let g = p[1] as f32 / 255.0;
        let b = p[2] as f32 / 255.0;
        // Normalize with standard ImageNet mean/std.
        [(r - 0.485) / 0.229, (g - 0.456) / 0.224, (b - 0.406) / 0.225]
    }).collect();

    let tensor = Tensor::from_vec(pixels, (size, size, 3), device)?
        .permute((2, 0, 1))? // (3, H, W)
        .unsqueeze(0)?;      // (1, 3, H, W)
    Ok(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "candle")]
    fn preprocess_test_image() {
        let device = Device::Cpu;
        let tensor = image_to_tensor(Path::new("test_imgs/dagnpats.png"), 448, &device).unwrap();
        assert_eq!(tensor.dims(), &[1, 3, 448, 448]);
    }
}
