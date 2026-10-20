// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Decode an OMPF-wrapped OODF delta and print its logical signature.
//!
//! This is the schema-aware decode that an internal acceptance harness's deep
//! parity check needs but cannot do itself (Python's `flatc` can't compile the
//! schema's `[union]` vectors). It parses the OMPF container inline (no deps)
//! and runs Part 0 through `oodf_ffi::parse_delta` (the C++ verifier+parser),
//! then prints `is_diff` / `base_version` / `source_format` / struct-command
//! types / field summaries — the fields that determine whether the Kit client
//! will accept and apply the delta.
//!
//! Two output modes:
//!   (default) human-readable summary + a Kit-client-compatibility VERDICT.
//!   --json    a canonical, key-sorted JSON logical signature whose shape
//!             matches the acceptance harness's normalized form; the harness
//!             shells out to this binary to compute deep-parity signatures for
//!             both the Rust and the C++ endpoint. The signature drops
//!             OMPF-position-only fields (ext_value_index) and sorts every
//!             array by its intrinsic ordering keys so two byte-different but
//!             logically identical encodings compare equal.
//!
//! Input is OMPF (4-byte "OMPF" magic) — or, in `--raw` mode only, a bare OODF
//! delta with no OMPF wrapper (magic absent → the whole file is treated as
//! Part 0, no external values). This lets the harness's captured `.oodf`
//! fixtures be decoded directly.
//!
//! Usage:
//!   base64-of-OMPF on stdin:
//!     pbpaste | cargo run -q -p oodf-ffi --example decode_ompf
//!     pbpaste | cargo run -q -p oodf-ffi --example decode_ompf -- --json
//!   raw bytes from a file (OMPF or bare OODF):
//!     cargo run -q -p oodf-ffi --example decode_ompf -- --raw /path/to.ompf
//!     cargo run -q -p oodf-ffi --example decode_ompf -- --json --raw /path/to.oodf

use std::io::Read;

use sha2::{Digest, Sha256};

fn b64_decode(s: &str) -> Result<Vec<u8>, String> {
    fn val(c: u8) -> Option<u8> {
        match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a' + 26),
            b'0'..=b'9' => Some(c - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let mut out = Vec::new();
    let mut buf = 0u32;
    let mut bits = 0u32;
    for &c in s.as_bytes() {
        if c == b'=' || c.is_ascii_whitespace() {
            continue;
        }
        let v = val(c).ok_or_else(|| format!("invalid base64 byte: {c:#x}"))?;
        buf = (buf << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
        }
    }
    Ok(out)
}

const OMPF_MAGIC: u32 = 0x4650_4d4f; // "OMPF" little-endian

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// JSON-escape `s` and wrap it in double quotes — enough for the value set this
/// decoder emits (section names, key names, source format, hex strings).
fn jstr(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn jnum_f64(x: f64) -> String {
    if x.is_finite() {
        format!("{x}")
    } else {
        // NaN / Inf are not representable in strict JSON; null keeps the
        // signature parseable. Both endpoints decode through this same path,
        // so a non-finite time (which the schema should never carry) degrades
        // identically on each side.
        "null".to_string()
    }
}

/// Canonical JSON object for one struct command — keys emitted in alphabetical
/// order so the array can be sorted by the rendered string (matching the
/// Python normalizer's `sort(key=json.dumps(..., sort_keys=True))`).
fn cmd_to_json(c: &oodf_ffi::StructCommand) -> String {
    use oodf_ffi::StructCommand::*;
    match c {
        CreateSection(x) => format!(
            "{{\"parentId\":{},\"sectionId\":{},\"sectionName\":{},\"sectionType\":{},\"type\":{}}}",
            x.parent_id,
            x.section_id,
            jstr(&x.section_name),
            x.section_type,
            jstr("CommandCreateSection"),
        ),
        DeleteSection(x) => format!(
            "{{\"parentId\":{},\"sectionId\":{},\"type\":{}}}",
            x.parent_id,
            x.section_id,
            jstr("CommandDeleteSection"),
        ),
        MoveSection(x) => format!(
            "{{\"newName\":{},\"newParentId\":{},\"oldParentId\":{},\"sectionId\":{},\"type\":{}}}",
            jstr(&x.new_name),
            x.new_parent_id,
            x.old_parent_id,
            x.section_id,
            jstr("CommandMoveSection"),
        ),
        ReorderChildren(x) => {
            let children = x
                .children_list
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(",");
            format!(
                "{{\"childrenList\":[{}],\"childrenListId\":{},\"sectionId\":{},\"type\":{}}}",
                children,
                x.children_list_id,
                x.section_id,
                jstr("CommandReorderChildren"),
            )
        }
        DiffChangeSection(x) => format!(
            "{{\"parentId\":{},\"sectionId\":{},\"sectionName\":{},\"sectionOrder\":{},\"sectionType\":{},\"type\":{}}}",
            x.parent_id,
            x.section_id,
            jstr(&x.section_name),
            x.section_order,
            x.section_type,
            jstr("CommandDiffChangeSection"),
        ),
        DiffDeleteSection(x) => format!(
            "{{\"sectionId\":{},\"type\":{}}}",
            x.section_id,
            jstr("CommandDiffDeleteSection"),
        ),
    }
}

/// Build the canonical JSON logical signature, matching `normalize_ompf()`'s
/// dict shape. `externals` are the OMPF Part 1..N blobs (empty for raw OODF).
fn signature_json(d: &oodf_ffi::OwnedDelta, externals: &[&[u8]]) -> String {
    // setFields: drop ext_value_index; sort by (sectionId, keyName, setOrder).
    let mut set_fields: Vec<(u64, String, u64, String)> = d
        .set_fields
        .iter()
        .map(|f| {
            let obj = format!(
                "{{\"extValueSize\":{},\"keyName\":{},\"sectionId\":{},\"setOrder\":{},\"valueOrExtHash\":{}}}",
                f.ext_value_size,
                jstr(&f.key_name),
                f.section_id,
                f.set_order,
                jstr(&hex(&f.value_or_ext_hash)),
            );
            (f.section_id, f.key_name.clone(), f.set_order, obj)
        })
        .collect();
    set_fields.sort_by(|a, b| (a.0, &a.1, a.2).cmp(&(b.0, &b.1, b.2)));
    let set_fields_json = set_fields
        .iter()
        .map(|t| t.3.clone())
        .collect::<Vec<_>>()
        .join(",");

    // timeSamples: drop ext_value_index; sort by (sectionId, time).
    let mut time_samples: Vec<(u64, f64, String)> = d
        .time_samples
        .iter()
        .map(|t| {
            let obj = format!(
                "{{\"extValueSize\":{},\"sectionId\":{},\"time\":{},\"valueOrExtHash\":{}}}",
                t.ext_value_size,
                t.section_id,
                jnum_f64(t.time),
                jstr(&hex(&t.value_or_ext_hash)),
            );
            (t.section_id, t.time, obj)
        })
        .collect();
    time_samples.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });
    let time_samples_json = time_samples
        .iter()
        .map(|t| t.2.clone())
        .collect::<Vec<_>>()
        .join(",");

    // structCommands: sort by canonical rendered JSON (order-insensitive wire).
    let mut struct_cmds: Vec<String> = d.struct_commands.iter().map(cmd_to_json).collect();
    struct_cmds.sort();
    let struct_cmds_json = struct_cmds.join(",");

    // externalValueHashes: plain SHA-256 of each external blob, hex, sorted — a
    // fingerprint of the blob bytes, computed the same way for both endpoints,
    // that the Python normalizer carried. It is NOT the wire claim: a
    // setField/timeSample valueOrExtHash carries the content hash
    // (`content_hash`, the 1 MiB block hash), so the two values differ. It stays
    // plain SHA-256 so new signatures remain comparable with recorded runs.
    let mut ext_hashes: Vec<String> = externals
        .iter()
        .map(|b| {
            let mut h = Sha256::new();
            h.update(b);
            hex(&h.finalize())
        })
        .collect();
    ext_hashes.sort();
    let ext_hashes_json = ext_hashes
        .iter()
        .map(|h| jstr(h))
        .collect::<Vec<_>>()
        .join(",");

    format!(
        "{{\"baseVersion\":{},\"externalValueHashes\":[{}],\"isDiff\":{},\"setFields\":[{}],\"sourceFormat\":{},\"structCommands\":[{}],\"timeSamples\":[{}]}}",
        d.base_version,
        ext_hashes_json,
        d.is_diff,
        set_fields_json,
        jstr(&d.source_format),
        struct_cmds_json,
        time_samples_json,
    )
}

/// Block size of the external-value content hash: a fixed protocol constant.
const CONTENT_HASH_BLOCK_BYTES: usize = 1 << 20;

/// The content hash a `valueOrExtHash` must equal: SHA-256 over the
/// concatenated SHA-256 digests of the value's consecutive 1 MiB blocks (an
/// empty value hashes to SHA-256 of the empty string). The clients' definition,
/// not plain SHA-256 of the bytes. Written out here rather than imported, so
/// this decoder stays independent of the service code it checks.
fn content_hash(bytes: &[u8]) -> [u8; 32] {
    let mut outer = Sha256::new();
    for block in bytes.chunks(CONTENT_HASH_BLOCK_BYTES) {
        outer.update(Sha256::digest(block));
    }
    outer.finalize().into()
}

/// Validate wire references before normalization discards position-only
/// indices: every external reference must name an OMPF part that is exactly
/// `extValueSize` bytes long and whose content hash is its `valueOrExtHash`.
fn validate_externals(d: &oodf_ffi::OwnedDelta, externals: &[&[u8]]) -> Result<(), String> {
    let references = d
        .set_fields
        .iter()
        .map(|v| (v.ext_value_index, v.ext_value_size, &v.value_or_ext_hash))
        .chain(
            d.time_samples
                .iter()
                .map(|v| (v.ext_value_index, v.ext_value_size, &v.value_or_ext_hash)),
        );
    for (index, size, hash) in references {
        if size == 0 {
            continue;
        }
        let blob = index
            .checked_sub(1)
            .and_then(|i| usize::try_from(i).ok())
            .and_then(|i| externals.get(i))
            .ok_or_else(|| "external value index is outside OMPF parts".to_string())?;
        if blob.len() as u64 != size || content_hash(blob).as_slice() != hash.as_slice() {
            return Err("external value size or hash differs from referenced OMPF part".into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod conformance_tests {
    use super::*;

    /// A delta with one external reference (a field or a time sample) claiming
    /// `claim` for a `size`-byte value at `index`.
    fn one_reference(
        time_sample: bool,
        index: u32,
        size: u64,
        claim: Vec<u8>,
    ) -> oodf_ffi::OwnedDelta {
        let mut delta = oodf_ffi::OwnedDelta::default();
        if time_sample {
            delta.time_samples.push(oodf_ffi::TimeSample {
                ext_value_index: index,
                ext_value_size: size,
                value_or_ext_hash: claim,
                ..Default::default()
            });
        } else {
            delta.set_fields.push(oodf_ffi::SetField {
                ext_value_index: index,
                ext_value_size: size,
                value_or_ext_hash: claim,
                ..Default::default()
            });
        }
        delta
    }

    #[test]
    fn external_reference_negative_controls() {
        let data = b"external content";
        for time_sample in [false, true] {
            for index in [0, 1, 2, u32::MAX] {
                let delta = one_reference(
                    time_sample,
                    index,
                    data.len() as u64,
                    content_hash(data).to_vec(),
                );
                assert_eq!(validate_externals(&delta, &[data]).is_ok(), index == 1);
                assert!(
                    validate_externals(&delta, &[b"wrong content", data]).is_err() || index == 2
                );
                assert!(validate_externals(&delta, &[b"short"]).is_err());
            }
        }
    }

    /// The claim is the content hash, not plain SHA-256: a plain-SHA-256 claim
    /// for the right bytes is refused.
    #[test]
    fn a_plain_sha256_claim_is_refused() {
        let data = b"external content";
        for time_sample in [false, true] {
            let delta = one_reference(
                time_sample,
                1,
                data.len() as u64,
                Sha256::digest(data).to_vec(),
            );
            assert!(validate_externals(&delta, &[data]).is_err());
        }
    }

    /// The local content hash agrees with a client's own vector.
    #[test]
    fn content_hash_matches_a_client_vector() {
        // The C++ client library's test vector: one byte of 0x00.
        assert_eq!(
            hex(&content_hash(&[0x00])),
            "1406e05881e299367766d313e26c05564ec91bf721d31726bd6e46e60689539a"
        );
    }
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let mut json_mode = false;
    let mut validate_external_values = false;
    let mut raw_path: Option<String> = None;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--json" => json_mode = true,
            "--validate-externals" => validate_external_values = true,
            "--raw" => {
                i += 1;
                raw_path = Some(argv.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("--raw requires a file path");
                    std::process::exit(2);
                }));
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }

    let input: Vec<u8> = if let Some(path) = raw_path {
        std::fs::read(&path).expect("read raw file")
    } else {
        let mut s = String::new();
        std::io::stdin().read_to_string(&mut s).expect("read stdin");
        b64_decode(&s).expect("base64 decode stdin")
    };

    // Accept either an OMPF container or a bare OODF delta (no wrapper). The
    // normalize_ompf path always passes OMPF; the `.oodf` capture fixtures are
    // bare OODF, so treat a missing magic as "the whole input is Part 0".
    let magic = if input.len() >= 4 {
        u32::from_le_bytes(input[0..4].try_into().unwrap())
    } else {
        0
    };
    let is_ompf = magic == OMPF_MAGIC;

    let (delta_bytes, externals): (&[u8], Vec<&[u8]>) = if is_ompf {
        if input.len() < 8 {
            eprintln!("ERROR: OMPF input too short ({} bytes)", input.len());
            std::process::exit(2);
        }
        let num_parts = u32::from_le_bytes(input[4..8].try_into().unwrap()) as usize;
        if num_parts == 0 {
            (&[], Vec::new())
        } else {
            let mut offsets = Vec::with_capacity(num_parts);
            for p in 0..num_parts {
                let base = 8 + p * 8;
                offsets
                    .push(u64::from_le_bytes(input[base..base + 8].try_into().unwrap()) as usize);
            }
            let part0_end = if num_parts > 1 {
                offsets[1]
            } else {
                input.len()
            };
            let d = &input[offsets[0]..part0_end];
            let mut ext = Vec::new();
            for p in 1..num_parts {
                let end = if p + 1 < num_parts {
                    offsets[p + 1]
                } else {
                    input.len()
                };
                ext.push(&input[offsets[p]..end]);
            }
            (d, ext)
        }
    } else {
        // Bare OODF: the whole input is the delta, no external values.
        (&input[..], Vec::new())
    };

    if json_mode {
        if delta_bytes.is_empty() {
            println!("{{\"empty\":true}}");
            return;
        }
        match oodf_ffi::parse_delta(delta_bytes) {
            Ok(d) => {
                if validate_external_values {
                    if let Err(error) = validate_externals(&d, &externals) {
                        eprintln!("{error}");
                        std::process::exit(1);
                    }
                }
                println!("{}", signature_json(&d, &externals));
            }
            Err(e) => {
                eprintln!("parse_delta FAILED: {e:?}");
                std::process::exit(1);
            }
        }
        return;
    }

    // ── Human-readable mode ────────────────────────────────────────────────
    println!(
        "input: total={} bytes, {}",
        input.len(),
        if is_ompf {
            "OMPF container"
        } else {
            "bare OODF (no OMPF wrapper)"
        }
    );
    println!("  part[0] (OODF delta): {} bytes", delta_bytes.len());
    for (idx, e) in externals.iter().enumerate() {
        println!("  part[{}] (ext value): {} bytes", idx + 1, e.len());
    }

    if delta_bytes.is_empty() {
        println!("\n(empty delta — no OODF payload)");
        return;
    }

    match oodf_ffi::parse_delta(delta_bytes) {
        Ok(d) => {
            println!("\n=== OODF logical signature ===");
            println!("is_diff       : {}", d.is_diff);
            println!("base_version  : {}", d.base_version);
            println!("source_format : {:?}", d.source_format);
            println!("struct_commands ({}):", d.struct_commands.len());
            for c in &d.struct_commands {
                use oodf_ffi::StructCommand::*;
                let line = match c {
                    CreateSection(x) => format!(
                        "  CreateSection      sid={} parent={} name={:?} type={}",
                        x.section_id, x.parent_id, x.section_name, x.section_type
                    ),
                    DeleteSection(x) => {
                        format!(
                            "  DeleteSection      sid={} parent={}",
                            x.section_id, x.parent_id
                        )
                    }
                    MoveSection(x) => format!(
                        "  MoveSection        sid={} {}->{} name={:?}",
                        x.section_id, x.old_parent_id, x.new_parent_id, x.new_name
                    ),
                    ReorderChildren(x) => format!(
                        "  ReorderChildren    sid={} list_id={} children={:?}",
                        x.section_id, x.children_list_id, x.children_list
                    ),
                    DiffChangeSection(x) => format!(
                        "  DiffChangeSection  sid={} parent={} name={:?} type={} order={}",
                        x.section_id, x.parent_id, x.section_name, x.section_type, x.section_order
                    ),
                    DiffDeleteSection(x) => format!("  DiffDeleteSection  sid={}", x.section_id),
                };
                println!("{line}");
            }
            println!("set_fields ({}):", d.set_fields.len());
            for f in &d.set_fields {
                let v = if f.ext_value_size > 0 {
                    format!("ext(size={}, idx={})", f.ext_value_size, f.ext_value_index)
                } else {
                    format!("inline({} bytes)", f.value_or_ext_hash.len())
                };
                println!(
                    "  sid={} key={:?} set_order={} value={}",
                    f.section_id, f.key_name, f.set_order, v
                );
            }
            println!("time_samples ({}):", d.time_samples.len());
            for t in &d.time_samples {
                println!(
                    "  sid={} time={} value_len={}",
                    t.section_id,
                    t.time,
                    t.value_or_ext_hash.len()
                );
            }

            // The single line that decides Kit-client compatibility:
            let verdict = if d.is_diff && d.base_version == 0 {
                "  *** is_diff=true && base_version=0 → the Kit client REJECTS this as \"no longer supported\" (tears down the read) once it has applied a sequenced update ***"
            } else if d.is_diff {
                "  is_diff=true (routes client into ApplyDiffDelta path; ok only with base_version>0)"
            } else {
                "  is_diff=false (client structCommands apply path — the production/relay shape)"
            };
            println!("\nVERDICT:\n{verdict}");
        }
        Err(e) => {
            eprintln!("\nparse_delta FAILED: {e:?}");
            // Name the step of `parse_delta` that refused the bytes: the
            // entry-count probe runs before the verifier and the children-list
            // guard after it; the remaining kinds come from the size and magic
            // checks, the parse walk or the bridge conversion.
            let refused_by = match e {
                oodf_ffi::OodfError::VerifierFailed(_) => "The C++ verifier rejected these bytes",
                oodf_ffi::OodfError::TooManyTables { .. } => {
                    "The C++ parser's entry-count probe refused these bytes before the verifier \
                     ran: the delta carries more entries than one delta may hold"
                }
                oodf_ffi::OodfError::ChildrenListTooLong { .. } => {
                    "The C++ parser's children-list guard refused these bytes: a ReorderChildren \
                     children list is over the parser's defensive cap"
                }
                _ => "parse_delta rejected these bytes",
            };
            eprintln!("({refused_by} — that itself is a finding.)");
            std::process::exit(1);
        }
    }
}
