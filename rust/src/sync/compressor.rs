use anyhow::{Context, Result};
use std::io::Write;

const COMPRESSION_LEVEL: i32 = 11;

pub fn compress(bytes: &[u8]) -> Result<Vec<u8>> {
    let mut encoder =
        zstd::stream::Encoder::new(Vec::new(), COMPRESSION_LEVEL).context("zstd encoder")?;
    encoder.write_all(bytes).context("zstd write")?;
    encoder.finish().context("zstd finish")
}

pub fn decompress(bytes: &[u8]) -> Result<Vec<u8>> {
    zstd::stream::decode_all(bytes).context("zstd decode")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let data = b"hello world".repeat(50);
        let c = compress(&data).unwrap();
        assert!(c.len() < data.len(), "zstd should compress repetitive data");
        let d = decompress(&c).unwrap();
        assert_eq!(d, data);
    }
}
