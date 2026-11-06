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

//! The Jobs' precondition (SDD SR-9.1, SR-10): the store must be held under
//! the incident the Job names, and the hold marker must say when the hold
//! was detected. Read before the lease (writing nothing) and again under it.

use crate::storage::client::ValkeyClient;
use crate::storage::dr_hold::{
    read_hold_marker, HoldMarkerFields, MARKER_FIELD_DETECTED_AT_MS, MARKER_FIELD_INCIDENT,
};
use crate::storage::keys::KeySchema;

/// What the hold marker says about a Job for one incident.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Precondition {
    /// No marker: the store is not held, and there is nothing to do (exit 0).
    NotHeld,
    /// Held under another incident (exit 4).
    DifferentIncident {
        /// The marker's incident.
        held: String,
    },
    /// Held, but a field the Job needs is missing or unreadable: the Job
    /// fails closed (writes nothing, stays held, exit 3), and never defaults
    /// the field to a match, to zero or to now (t2.tv3.s9.ao defer-6).
    MarkerIncomplete {
        /// The marker field: `incident` or `detected_at_ms`.
        field: &'static str,
    },
    /// Held under this incident, detected at `detected_at_ms` on the store's
    /// clock.
    Held {
        /// The incident.
        incident: String,
        /// The marker's `detected_at_ms`.
        detected_at_ms: u64,
    },
}

/// The precondition `marker` (as the hold poll's read returns it, `None`
/// when absent) gives a Job for `incident`.
///
/// Checked in this order: the marker exists; it names an incident; that
/// incident is `incident`; it carries a `detected_at_ms` that is a whole
/// number of milliseconds (surrounding whitespace allowed, as the hold
/// verdict reads it).
pub fn precondition_from_marker(marker: Option<&HoldMarkerFields>, incident: &str) -> Precondition {
    let Some(marker) = marker else {
        return Precondition::NotHeld;
    };
    let held = match marker.incident.as_deref() {
        Some(held) if !held.is_empty() => held,
        _ => {
            return Precondition::MarkerIncomplete {
                field: MARKER_FIELD_INCIDENT,
            }
        }
    };
    if held != incident {
        return Precondition::DifferentIncident {
            held: held.to_owned(),
        };
    }
    match marker
        .detected_at_ms
        .as_deref()
        .and_then(|ms| ms.trim().parse::<u64>().ok())
    {
        Some(detected_at_ms) => Precondition::Held {
            incident: held.to_owned(),
            detected_at_ms,
        },
        None => Precondition::MarkerIncomplete {
            field: MARKER_FIELD_DETECTED_AT_MS,
        },
    }
}

/// Read the hold marker (one `EVALSHA` of the hold poll's read-only script)
/// and return the precondition for `incident`. Writes nothing.
pub async fn read_precondition(
    client: &ValkeyClient,
    keys: &KeySchema,
    incident: &str,
) -> Result<Precondition, fred::error::Error> {
    let marker = read_hold_marker(client, keys).await?;
    Ok(precondition_from_marker(marker.as_ref(), incident))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn marker(incident: Option<&str>, detected_at_ms: Option<&str>) -> HoldMarkerFields {
        HoldMarkerFields {
            incident: incident.map(str::to_owned),
            detected_at_ms: detected_at_ms.map(str::to_owned),
            ..HoldMarkerFields::default()
        }
    }

    #[test]
    fn no_marker_is_not_held() {
        assert_eq!(precondition_from_marker(None, "i"), Precondition::NotHeld);
    }

    #[test]
    fn matching_marker_is_held_with_its_detection_time() {
        assert_eq!(
            precondition_from_marker(Some(&marker(Some("i"), Some(" 42 "))), "i"),
            Precondition::Held {
                incident: "i".into(),
                detected_at_ms: 42
            }
        );
    }

    #[test]
    fn another_incident_is_different_even_without_a_detection_time() {
        assert_eq!(
            precondition_from_marker(Some(&marker(Some("j"), None)), "i"),
            Precondition::DifferentIncident { held: "j".into() }
        );
    }

    /// A missing incident never counts as a match, and a missing or
    /// unparseable detection time never defaults (defer-6).
    #[test]
    fn missing_or_unreadable_fields_fail_closed_naming_the_field() {
        for m in [marker(None, Some("1")), marker(Some(""), Some("1"))] {
            assert_eq!(
                precondition_from_marker(Some(&m), "i"),
                Precondition::MarkerIncomplete { field: "incident" }
            );
        }
        for raw in [None, Some(""), Some("soon"), Some("-1"), Some("1.5")] {
            assert_eq!(
                precondition_from_marker(Some(&marker(Some("i"), raw)), "i"),
                Precondition::MarkerIncomplete {
                    field: "detected_at_ms"
                },
                "{raw:?}"
            );
        }
    }
}
