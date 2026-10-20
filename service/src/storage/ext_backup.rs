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

//! Backup confirmations on an external value's own meta, and the Valkey reads
//! the backup repair makes (SDD §22.17).
//!
//! **The confirmation.** `le:ext:meta:{hash_hex}` carries a third field,
//! [`BACKED_UP_IN_FIELD`], `<store id>:<ms>`: the backup store whose identity
//! is `<store id>` (`Config::checkpoint_store_identity`) held the value's
//! bytes at Valkey time `<ms>`. It is written only by
//! `scripts/ext_backup_confirm.lua` ([`confirm_ext_backup`]), only after the
//! backup store answered that it holds the bytes, and only while the meta
//! exists and still carries the `last_store_ms` stamp the caller read with the
//! bytes. Its absence is the "owed" state the backup repair
//! (`checkpoint::ext_backup_repair`) acts on, so every way a copy can be lost
//! — a dropped send, a failed upload, a discard at shutdown, a crash — is
//! covered by the copy simply never being confirmed: nothing is written ahead
//! of an upload and nothing on the store path waits for one.
//!
//! **Its teardown.** The GC's `DEL meta data` takes the field with the meta
//! when a value is collected, so a collected value carries no confirmation
//! that could outlive it, and the confirm script, which never creates a meta,
//! cannot revive one. The checkpoint collect clears it after it deletes the
//! value's backup copy ([`clear_ext_backup`]), so the next repair pass re-sends
//! a value Valkey still holds. The one exception, until b.g65's fence closes
//! it: a confirmation from an upload or existence check that answered just
//! before the delete, landing after the clear (the confirm script checks only
//! the meta and its stamp, and the collect changes neither). It stands for a
//! copy that no longer exists until the repair's daily audit re-checks it,
//! about 24 h later.
//!
//! **The repair's Valkey reads.** The meta walk's pipelined `HMGET`
//! ([`read_backup_marks`]), the size of a stored value ([`stored_ext_value_len`])
//! and its bytes ([`read_ext_value_bytes`], in [`EXT_VALUE_READ_CHUNK_BYTES`]
//! ranges into one buffer of the value's size, so it holds no more than the
//! re-send's charge), and Valkey's own clock ([`valkey_time_ms`]).
//! [`read_ext_value_bytes`] is a blob reader outside the GC miss protocol,
//! allowlisted by
//! `ext_blob_reader_guard_test::every_external_value_blob_read_site_under_src_is_allowlisted`
//! for the reason on its doc.
//!
//! Every helper counts a command timeout and notes a store write refused
//! `-OOM` where it receives the error ([`FredErrOrigin::observe_error`]);
//! callers only log the error they get back.

use std::sync::LazyLock;

use bytes::{Bytes, BytesMut};
use fred::interfaces::{ClientLike, HashesInterface, KeysInterface};
use fred::types::scripts::Script;
use fred::types::{ClusterHash, CustomCommand, Value};

use crate::observability::metrics::{SharedMetrics, EXT_BACKUP_MARK_OP_CONFIRM};
use crate::storage::client::{
    FredErrOrigin, ValkeyClient, POOL_LABEL_DATA, POOL_LABEL_MAINTENANCE,
};
use crate::storage::connection_buffers::GENERAL_LIMITS;
use crate::storage::ext_reconcile::LAST_STORE_MS_FIELD;
use crate::storage::keys::KeySchema;

/// The `ext:meta` field holding a value's backup confirmation,
/// `<store id>:<ms>` (module docs). The Lua writer names it literally, so the
/// two spellings must match.
pub const BACKED_UP_IN_FIELD: &str = "backed_up_in";

/// `scripts/ext_backup_confirm.lua`: write [`BACKED_UP_IN_FIELD`] only while
/// the meta exists and carries the caller's `last_store_ms`. The script header
/// carries the contract.
pub static EXT_BACKUP_CONFIRM_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/ext_backup_confirm.lua")));

/// A [`BACKED_UP_IN_FIELD`] value split into its store id and confirmation
/// time in Valkey milliseconds, or `None` when it is not of that shape
/// (`<id>:<decimal ms>`, the id non-empty). A field that does not parse is
/// treated as no confirmation by every reader.
pub fn parse_backed_up_in(raw: &str) -> Option<(&str, u64)> {
    let (store_id, ms) = raw.rsplit_once(':')?;
    if store_id.is_empty() {
        return None;
    }
    Some((store_id, ms.parse::<u64>().ok()?))
}

/// Record that the backup store `store_id` holds the bytes of `hash_hex`: one
/// `EVALSHA` of [`EXT_BACKUP_CONFIRM_SCRIPT`] on the data pool. `Ok(true)` when
/// the field was written; `Ok(false)` when the meta is gone or its
/// `last_store_ms` is no longer `last_store_ms` (pass `""` for a meta read
/// with no stamp), which is not an error: the value is checked again by the
/// next repair pass.
///
/// Call it only after the backup store answered that it holds the bytes.
pub async fn confirm_ext_backup(
    client: &ValkeyClient,
    keys: &KeySchema,
    hash_hex: &str,
    store_id: &str,
    last_store_ms: &str,
) -> Result<bool, fred::error::Error> {
    let origin = FredErrOrigin::client(client, POOL_LABEL_DATA);
    let written: i64 = EXT_BACKUP_CONFIRM_SCRIPT
        .evalsha_with_reload::<i64, _, _>(
            client.pool.next(),
            vec![keys.ext_meta(hash_hex)],
            vec![store_id.to_owned(), last_store_ms.to_owned()],
        )
        .await
        .inspect_err(|e| origin.observe_error(e))?;
    Ok(written == 1)
}

/// Remove `hash_hex`'s backup confirmation: `HDEL` of [`BACKED_UP_IN_FIELD`]
/// on the data pool. The checkpoint collect runs it after deleting the value's
/// backup copy, so a value Valkey still holds reads as owed and the repair
/// re-sends it, unless a confirmation that raced the delete lands after the
/// clear (module docs, "Its teardown"). On an absent meta it creates nothing.
pub async fn clear_ext_backup(
    client: &ValkeyClient,
    keys: &KeySchema,
    hash_hex: &str,
) -> Result<(), fred::error::Error> {
    let origin = FredErrOrigin::client(client, POOL_LABEL_DATA);
    let _: i64 = client
        .pool
        .hdel(keys.ext_meta(hash_hex), BACKED_UP_IN_FIELD)
        .await
        .inspect_err(|e| origin.observe_error(e))?;
    Ok(())
}

/// What one meta key's [`read_backup_marks`] reply carried, each field as
/// stored (`None` when absent).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackupMarks {
    /// [`BACKED_UP_IN_FIELD`].
    pub backed_up_in: Option<String>,
    /// `last_store_ms`, the store script's stamp.
    pub last_store_ms: Option<String>,
}

/// `HMGET backed_up_in last_store_ms` over `meta_keys`, in one pipelined round
/// trip on the maintenance pool; one [`BackupMarks`] per key, in order. A key
/// that no longer exists answers both fields absent.
pub async fn read_backup_marks(
    client: &ValkeyClient,
    meta_keys: &[String],
) -> Result<Vec<BackupMarks>, fred::error::Error> {
    if meta_keys.is_empty() {
        return Ok(Vec::new());
    }
    let origin = FredErrOrigin::client(client, POOL_LABEL_MAINTENANCE);
    let (maint_pool, _maint_token) = client.acquire_maintenance();
    let pipeline = maint_pool.next().pipeline();
    for key in meta_keys {
        let _: () = pipeline
            .hmget(key.as_str(), vec![BACKED_UP_IN_FIELD, LAST_STORE_MS_FIELD])
            .await
            .inspect_err(|e| origin.observe_error(e))?;
    }
    let replies: Vec<Value> = pipeline
        .all()
        .await
        .inspect_err(|e| origin.observe_error(e))?;
    if replies.len() != meta_keys.len() {
        return Err(fred::error::Error::new(
            fred::error::ErrorKind::Parse,
            format!(
                "an ext:meta HMGET pipeline answered {} replies for {} keys",
                replies.len(),
                meta_keys.len()
            ),
        ));
    }
    replies
        .into_iter()
        .map(|reply| {
            let mut fields: Vec<Option<String>> = reply.convert()?;
            if fields.len() != 2 {
                return Err(fred::error::Error::new(
                    fred::error::ErrorKind::Parse,
                    format!("an ext:meta HMGET of two fields answered {}", fields.len()),
                ));
            }
            let last_store_ms = fields.pop().flatten();
            let backed_up_in = fields.pop().flatten();
            Ok(BackupMarks {
                backed_up_in,
                last_store_ms,
            })
        })
        .collect()
}

/// The length in bytes of the value Valkey stores under `hash_hex`: `STRLEN`
/// of its data key on the data pool. `0` when the key is absent, which the
/// caller cannot tell from an empty value; its read of the bytes
/// ([`read_ext_value_bytes`]) does. Reads no bytes.
pub async fn stored_ext_value_len(
    client: &ValkeyClient,
    keys: &KeySchema,
    hash_hex: &str,
) -> Result<u64, fred::error::Error> {
    let origin = FredErrOrigin::client(client, POOL_LABEL_DATA);
    client
        .pool
        .strlen(keys.ext_data(hash_hex))
        .await
        .inspect_err(|e| origin.observe_error(e))
}

/// The bytes one ranged read of [`read_ext_value_bytes`] asks for: 64 B under
/// the general connections' read limit (`GENERAL_LIMITS.read_buffer_bytes`,
/// 64 KiB), the 64 B covering the reply's bulk-string header and trailer
/// (10 B at this size). Every reply frame then stays within that limit, so the
/// data-pool connection's read buffer stays inside what a general connection
/// keeps, which the connection-buffer bound already counts (SDD §15.3), and
/// the read adds no memory beyond the value's own buffer. A larger chunk would
/// regrow that read buffer to up to about four times the chunk for each reply,
/// memory nothing charges. Each range is its own round trip (about 1,600 for a
/// 100 MiB value), so no connection carries one value-sized reply ahead of
/// other callers'. Fixed, not a setting.
pub const EXT_VALUE_READ_CHUNK_BYTES: usize = GENERAL_LIMITS.read_buffer_bytes - 64;

const _: () = assert!(EXT_VALUE_READ_CHUNK_BYTES > 0);

/// What [`read_ext_value_bytes`] found under a hash.
#[derive(Debug)]
pub enum ExtValueRead {
    /// The value's bytes: exactly the expected length, in one buffer of that
    /// size.
    Bytes(Bytes),
    /// Valkey no longer holds the value: the GC collected it before or during
    /// the read.
    Gone,
    /// Valkey holds the value, but it changed under the read: it is now
    /// `now_len` bytes, not the expected length, or one range came back short.
    /// A verified store's heal (`ext_store_value.lua`) is what changes a
    /// stored value in place; the next pass reads it again at its new size.
    Changed { now_len: u64 },
}

/// Read the `len` bytes Valkey stores under `hash_hex`, to copy them to the
/// backup store: `GETRANGE`s of its data key on the data pool, each of at most
/// [`EXT_VALUE_READ_CHUNK_BYTES`], copied one at a time into a single buffer
/// allocated at `len` bytes and returned frozen, then a `STRLEN` that the
/// length is still `len` (and, for a zero length, an `EXISTS`). `len` is the
/// length the caller charged for, so the memory this read holds is that one
/// buffer, and its peak is that buffer plus one range: no reply is kept, and
/// no range outgrows the connection's counted read buffer. A range that comes
/// back short, or a final length other than `len`, ends the read as
/// [`ExtValueRead::Gone`] when the key is absent and
/// [`ExtValueRead::Changed`] otherwise.
///
/// The ranges are not one atomic read. A stored value changes in place only by
/// a verified store's heal, which replaces a string only when its length
/// differs (`ext_store_value.lua`), so the final `STRLEN` sees it. A value
/// collected between two ranges answers a short range, unless it is stored
/// again before the next one: then the later ranges come from the new string,
/// the final length matches, and the bytes join two stores. That is safe for
/// two reasons. A re-store of a content-addressed hash writes the same bytes,
/// unless an unverified store wrote wrong ones, the `observe` posture SDD
/// §22.17.1 already accepts. And this pass's confirmation of the copy is
/// refused, because every store re-stamps `last_store_ms`
/// (`ext_store_value.lua`) and `ext_backup_confirm.lua` writes only while the
/// stamp is the one the caller read with the value.
///
/// **Not the GC miss protocol, and safe without it.** The read serves no
/// snapshot: no layer read and no client is answered from these bytes, which
/// only become a copy in the backup store. A miss means the GC collected the
/// value, so no live object references it and there is nothing to back up; the
/// caller drops the hash. Bytes read just before a collection are a copy of a
/// value nothing references any more, a leak in the backup store at worst,
/// never a wrong answer to a reader.
///
/// The bytes are not re-hashed: they are what the store path stored under the
/// hash, copied without a second content check, the same posture fetch-on-miss
/// and the re-base backfill take toward the bytes they copy (SDD §11.1).
pub async fn read_ext_value_bytes(
    client: &ValkeyClient,
    keys: &KeySchema,
    hash_hex: &str,
    len: u64,
) -> Result<ExtValueRead, fred::error::Error> {
    let origin = FredErrOrigin::client(client, POOL_LABEL_DATA);
    let key = keys.ext_data(hash_hex);
    let capacity = usize::try_from(len).map_err(|_| {
        fred::error::Error::new(
            fred::error::ErrorKind::InvalidArgument,
            format!("a value of {len} bytes is larger than this platform can address"),
        )
    })?;
    let mut buf = BytesMut::with_capacity(capacity);
    while buf.len() < capacity {
        let start = buf.len();
        let want = (capacity - start).min(EXT_VALUE_READ_CHUNK_BYTES);
        let range: Option<Bytes> = client
            .pool
            .getrange(key.as_str(), start, start + want - 1)
            .await
            .inspect_err(|e| origin.observe_error(e))?;
        let range = range.unwrap_or_default();
        if range.len() < want {
            let (now_len, present) = stored_len_and_presence(client, &origin, &key).await?;
            return Ok(if present {
                ExtValueRead::Changed { now_len }
            } else {
                ExtValueRead::Gone
            });
        }
        // `GETRANGE` never answers more than it was asked for; the slice
        // keeps the buffer at `len` even if it did.
        buf.extend_from_slice(&range[..want]);
    }
    let (now_len, present) = stored_len_and_presence(client, &origin, &key).await?;
    Ok(match (present, now_len == len) {
        (true, true) => ExtValueRead::Bytes(buf.freeze()),
        (true, false) => ExtValueRead::Changed { now_len },
        (false, _) => ExtValueRead::Gone,
    })
}

/// The stored length of the data key `key` and whether it exists: a
/// `STRLEN`, then an `EXISTS` only when the length is 0 (an absent key and an
/// empty value both read 0). Reads no bytes.
async fn stored_len_and_presence(
    client: &ValkeyClient,
    origin: &FredErrOrigin<'_>,
    key: &str,
) -> Result<(u64, bool), fred::error::Error> {
    let now_len: u64 = client
        .pool
        .strlen(key)
        .await
        .inspect_err(|e| origin.observe_error(e))?;
    if now_len > 0 {
        return Ok((now_len, true));
    }
    let present: i64 = client
        .pool
        .exists(key)
        .await
        .inspect_err(|e| origin.observe_error(e))?;
    Ok((0, present > 0))
}

/// Valkey's own clock in milliseconds: a keyless `TIME` on the maintenance
/// pool.
///
/// Keyless, so under Cluster it lands on an arbitrary primary, not the one
/// that stamped a given meta: it removes this process's clock from a
/// comparison against a Valkey-written stamp, not inter-node skew, which the
/// deployment requirement that cluster nodes be time-synchronised covers
/// (`docs/service/docs/backing-store.md` § Requirements).
pub async fn valkey_time_ms(client: &ValkeyClient) -> Result<u64, fred::error::Error> {
    let origin = FredErrOrigin::client(client, POOL_LABEL_MAINTENANCE);
    let (maint_pool, _maint_token) = client.acquire_maintenance();
    let reply: Vec<String> = maint_pool
        .next()
        .custom(
            CustomCommand::new_static("TIME", ClusterHash::FirstKey, false),
            Vec::<Value>::new(),
        )
        .await
        .inspect_err(|e| origin.observe_error(e))?;
    let secs = reply.first().and_then(|s| s.parse::<u64>().ok());
    let usecs = reply.get(1).and_then(|s| s.parse::<u64>().ok());
    match (secs, usecs) {
        (Some(s), Some(us)) => Ok(s.saturating_mul(1000).saturating_add(us / 1000)),
        _ => Err(fred::error::Error::new(
            fred::error::ErrorKind::Parse,
            format!("an undecodable TIME reply: {}", reply.join(" ")),
        )),
    }
}

/// What the ext-mirror workers (`checkpoint::ext_mirror::spawn_ext_mirror_workers`)
/// and the backup repair need to confirm the copies the backup store holds:
/// the store, its keys,
/// the backup store's identity and the metrics the failures are counted on.
#[derive(Clone)]
pub struct ExtBackupMarks {
    client: ValkeyClient,
    keys: KeySchema,
    store_id: String,
    metrics: Option<SharedMetrics>,
}

impl ExtBackupMarks {
    /// Marks written under `store_id`, the backup store's identity
    /// (`Config::checkpoint_store_identity`).
    pub fn new(
        client: ValkeyClient,
        keys: KeySchema,
        store_id: String,
        metrics: Option<SharedMetrics>,
    ) -> Self {
        Self {
            client,
            keys,
            store_id,
            metrics,
        }
    }

    /// The backup store's identity the marks are written under.
    pub fn store_id(&self) -> &str {
        &self.store_id
    }

    /// Confirm `hash_hex` once the backup store answered that it holds the
    /// bytes — a successful upload, or the repair's existence check that found
    /// them ([`confirm_ext_backup`]) — best effort: a failure is counted as
    /// `checkpoint_ext_backup_mark_failed_total{op="confirm"}`, logged at
    /// DEBUG and otherwise ignored, because the value then keeps whatever
    /// confirmation it had (none, for a new copy) and the repair checks it
    /// again. A declined confirmation
    /// (`Ok(false)`) is not counted.
    pub async fn confirm(&self, hash_hex: &str, last_store_ms: &str) {
        if let Err(e) = confirm_ext_backup(
            &self.client,
            &self.keys,
            hash_hex,
            &self.store_id,
            last_store_ms,
        )
        .await
        {
            if let Some(m) = self.metrics.as_ref() {
                m.inc_checkpoint_ext_backup_mark_failed(EXT_BACKUP_MARK_OP_CONFIRM);
            }
            let prefix_end = hash_hex.len().min(16);
            tracing::debug!(
                hash_prefix = &hash_hex[..prefix_end],
                error = %e,
                "could not record an external value's backup confirmation; the backup repair \
                 checks the value again"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    //! The field's parse. The script, the helpers' Valkey round trips and the
    //! marks run against Valkey in
    //! `tests/integration/checkpoint/ext_backup_repair_test.rs`.
    use super::*;

    /// The confirm script writes `<store id>:<ms>`; the parse returns both
    /// halves, splitting at the last `:`.
    #[test]
    fn parse_backed_up_in_splits_the_store_id_from_the_confirmation_ms() {
        assert_eq!(
            parse_backed_up_in("0123456789abcdef:1760000000123"),
            Some(("0123456789abcdef", 1_760_000_000_123))
        );
        assert_eq!(parse_backed_up_in("id:0"), Some(("id", 0)));
    }

    /// Anything else is no confirmation: no separator, an empty store id, an
    /// empty, negative, fractional or non-decimal time, or one past `u64`.
    #[test]
    fn parse_backed_up_in_answers_none_for_every_other_shape() {
        for raw in [
            "",
            "garbage",
            ":123",
            "id:",
            "id:-5",
            "id:1.5",
            "id:12x",
            "id: 12",
            "id:18446744073709551616",
        ] {
            assert_eq!(parse_backed_up_in(raw), None, "{raw:?}");
        }
    }
}
