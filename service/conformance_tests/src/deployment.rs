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

//! Optional Storage API provisioning of the run's base layer.
//!
//! When `CONFORMANCE_STORAGE_ENDPOINT` is set, the run creates its own base
//! layer, `<CONFORMANCE_BASE_PATH>/conformance-<uuid>.usda` holding
//! `#usda 1.0\n`, with a `PUT` to
//! `<endpoint>/v1alpha/fileobject/by-address/<percent-encoded path>?data_object_size=<n>`,
//! runs the suite against it, and then `DELETE`s it, whether or not the `PUT`
//! appeared to succeed (its response may have been lost) and whatever the
//! suite's outcome. Only that UUID-named file is ever removed; a `DELETE`
//! answered 404 is not a failure.
//!
//! Requests carry the bearer token ([`Settings::authorization`], after the
//! token file has appeared), each under [`Settings::timeout`]. A redirect is
//! never followed, so the credential cannot be sent elsewhere. Errors are
//! reported by kind ([`ProvisionError::kind`]): HTTP and transport error text
//! can contain credentials and is never printed. Only a
//! [`ProvisionError::Config`] also carries its [`ConfigError`] text, which is
//! fixed text or names a variable, and never holds a value.
//!
//! SIGTERM or SIGINT ([`Interrupt`]) during the wait for the token file ends
//! provisioning before any request is sent; during the `PUT`, it abandons the
//! `PUT` and the `DELETE` is still attempted. Either way the suite does not
//! run and the outcome is [`ProvisionError::Interrupted`]. An interruption
//! during the suite is the runner's to handle; the `DELETE` follows it as
//! usual. Once either signal has arrived, the `DELETE` waits at most
//! [`INTERRUPT_DELETE_BOUND`] for its answer, counted from the later of the
//! signal and the `DELETE`'s start, so the reports printed after it still
//! beat the kubelet's SIGKILL.
//!
//! The suite's cases call only LiveService, so a run against an
//! implementation with no Storage API simply leaves the variable unset.

use std::future::Future;
use std::time::Duration;

use percent_encoding::{utf8_percent_encode, AsciiSet, NON_ALPHANUMERIC};

use crate::config::{ConfigError, Lookup, Settings};
use crate::runner::Interrupt;

/// Why provisioning or its cleanup failed. Carries no HTTP or transport error
/// text; only `Config` carries text, its [`ConfigError`]'s.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProvisionError {
    /// The configuration was refused.
    Config(ConfigError),
    /// The token file never appeared within the readiness timeout.
    TokenFileMissing,
    /// The Storage API answered with a redirect, which is never followed.
    Redirect,
    /// The Storage API answered `method` with an unexpected HTTP `status`.
    HttpStatus {
        /// `PUT` or `DELETE`.
        method: &'static str,
        /// The HTTP status code.
        status: u16,
    },
    /// The request failed before an answer: connection, TLS or timeout,
    /// including a `DELETE` given up at [`INTERRUPT_DELETE_BOUND`] after an
    /// interruption.
    Transport,
    /// SIGTERM or SIGINT arrived before the suite started: while waiting for
    /// the token file, or during the `PUT`.
    Interrupted,
}

impl ProvisionError {
    /// A fixed name of the failure, safe to print and to put in the JUnit
    /// `deployment-fixture` error.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Config(_) => "Config",
            Self::TokenFileMissing => "TokenFileMissing",
            Self::Redirect => "Redirect",
            Self::HttpStatus { .. } => "HttpStatus",
            Self::Transport => "Transport",
            Self::Interrupted => "Interrupted",
        }
    }
}

impl From<ConfigError> for ProvisionError {
    fn from(error: ConfigError) -> Self {
        Self::Config(error)
    }
}

/// Whether this run provisions its base layer: `CONFORMANCE_STORAGE_ENDPOINT`
/// is set and not blank.
pub fn requested(lookup: Lookup<'_>) -> bool {
    lookup("CONFORMANCE_STORAGE_ENDPOINT").is_some_and(|value| !value.trim().is_empty())
}

/// Provisions the base layer under `settings.base_path` (a writable resource
/// directory), runs `run` with the settings pointed at it
/// ([`Settings::with_base_path`]), then deletes it. `Ok` with `run`'s exit
/// code when provisioning and cleanup both succeeded; otherwise the failure,
/// returned only after the cleanup has been attempted (`run` is not called
/// when the `PUT` failed or was interrupted). A failed `DELETE` is the
/// failure returned, whatever came before it. `interrupt` resolving during
/// the token-file wait or the `PUT` is [`ProvisionError::Interrupted`] (see
/// the module doc), and once it has resolved the `DELETE` is bounded by
/// [`INTERRUPT_DELETE_BOUND`]; `run` gets no interrupt from here, so it must
/// hold its own clone. `settings.storage_endpoint` must be set.
pub async fn provision<F, Fut>(
    settings: &Settings,
    interrupt: Interrupt,
    run: F,
) -> Result<i32, ProvisionError>
where
    F: FnOnce(Settings) -> Fut,
    Fut: Future<Output = i32>,
{
    let endpoint = settings
        .storage_endpoint
        .as_deref()
        .ok_or_else(|| invalid("CONFORMANCE_STORAGE_ENDPOINT must be set for provisioning"))?
        .trim_end_matches('/');
    let base = settings.base_path.trim_end_matches('/');
    if base.is_empty() {
        return Err(invalid(
            "CONFORMANCE_BASE_PATH must be a writable resource directory for provisioning",
        ));
    }
    let path = format!("{base}/conformance-{}.usda", uuid::Uuid::new_v4().simple());
    let provisioned = settings.clone().with_base_path(path.clone());

    // A sidecar-written token may arrive after the suite starts. Nothing has
    // been sent yet, so an interruption here needs no cleanup.
    tokio::select! {
        biased;
        _ = interrupt.clone() => return Err(ProvisionError::Interrupted),
        waited = provisioned.wait_for_token() => waited.map_err(from_token_error)?,
    }

    let url = format!(
        "{endpoint}/v1alpha/fileobject/by-address/{}",
        utf8_percent_encode(&path, PATH_SEGMENT)
    );
    let http = storage_client(&provisioned)?;

    // The DELETE is attempted even when the PUT failed or was interrupted: a
    // PUT may have taken effect with its response lost or not yet read.
    let put = tokio::select! {
        biased;
        _ = interrupt.clone() => Err(ProvisionError::Interrupted),
        sent = send(&http, &provisioned, Method::Put, &url) => sent,
    };
    let outcome = match put {
        Ok(()) => Ok(run(provisioned.clone()).await),
        Err(error) => Err(error),
    };
    let cleanup = delete(&http, &provisioned, &url, interrupt).await;
    match cleanup {
        Err(error) => Err(error),
        Ok(()) => outcome,
    }
}

/// What the path is encoded with: everything but ASCII letters, digits and
/// `_.-~` is percent-encoded, `/` included, so the whole path is one segment.
const PATH_SEGMENT: &AsciiSet = &NON_ALPHANUMERIC
    .remove(b'_')
    .remove(b'.')
    .remove(b'-')
    .remove(b'~');

/// The body of the provisioned base layer.
const BASE_LAYER: &[u8] = b"#usda 1.0\n";

/// The two requests provisioning makes.
#[derive(Clone, Copy)]
enum Method {
    /// Creates the base layer, `?data_object_size=<n>` with [`BASE_LAYER`].
    Put,
    /// Removes it; a 404 answer is no failure.
    Delete,
}

fn invalid(message: &str) -> ProvisionError {
    ProvisionError::Config(ConfigError::Invalid(message.to_string()))
}

/// A token-file failure as a provisioning failure.
fn from_token_error(error: ConfigError) -> ProvisionError {
    match error {
        ConfigError::TokenFileMissing => ProvisionError::TokenFileMissing,
        other => ProvisionError::Config(other),
    }
}

/// The Storage API client: no redirects, [`Settings::timeout`] per request,
/// and the CA bundle in place of reqwest's built-in roots when one is set.
fn storage_client(settings: &Settings) -> Result<reqwest::Client, ProvisionError> {
    let mut builder = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(settings.timeout);
    if let Some(pem) = settings.ca_pem()? {
        let certificates = reqwest::Certificate::from_pem_bundle(&pem)
            .map_err(|_| invalid("CONFORMANCE_CA_FILE must hold PEM certificates"))?;
        builder = builder.tls_built_in_root_certs(false);
        for certificate in certificates {
            builder = builder.add_root_certificate(certificate);
        }
    }
    builder.build().map_err(|_| ProvisionError::Transport)
}

/// Sends the `DELETE` of the provisioned base layer. Once SIGTERM or SIGINT
/// has arrived, before it was sent or while it waits for its answer, the wait
/// is cut to [`INTERRUPT_DELETE_BOUND`] from the later of the signal and the
/// `DELETE`'s start (a signal that came earlier is noticed only when the
/// `DELETE` starts, up to about `crate::runner::INTERRUPT_CLEANUP_BOUND`
/// after it); a `DELETE` that gets no
/// answer within it is [`ProvisionError::Transport`], as a timed-out request
/// is (the file may be left behind).
///
/// Why the cut: the reports are printed only after the `DELETE`, and the
/// kubelet's SIGKILL follows SIGTERM after the pod's 30 s termination grace.
/// The runner's cleanup of the interrupted case takes at most
/// `crate::runner::INTERRUPT_CLEANUP_BOUND`; bounding the `DELETE` too keeps
/// both, and so the printed reports, inside the grace whatever
/// `CONFORMANCE_TIMEOUT` is. (Printing the reports before the `DELETE`
/// instead would print them twice, or without a `DELETE` failure, whenever
/// the `DELETE` then failed.)
async fn delete(
    http: &reqwest::Client,
    settings: &Settings,
    url: &str,
    interrupt: Interrupt,
) -> Result<(), ProvisionError> {
    let mut request = std::pin::pin!(send(http, settings, Method::Delete, url));
    tokio::select! {
        biased;
        answered = &mut request => return answered,
        _ = interrupt => {}
    }
    tokio::time::timeout(INTERRUPT_DELETE_BOUND, request)
        .await
        .unwrap_or(Err(ProvisionError::Transport))
}

/// How long the provisioned base layer's `DELETE` may still wait for its
/// answer once SIGTERM or SIGINT has arrived (see `delete`).
pub const INTERRUPT_DELETE_BOUND: Duration = Duration::from_secs(10);

// After SIGTERM, the interrupted case's cleanup and then the `DELETE` must
// both fit the pod's default 30 s termination grace (the chart sets no
// other), with room left to write and print the reports.
const _: () = assert!(
    crate::runner::INTERRUPT_CLEANUP_BOUND.as_secs() + INTERRUPT_DELETE_BOUND.as_secs() < 30
);

/// Sends one provisioning request with a freshly read bearer token. Any error
/// is reduced to its kind: reqwest's error text can carry the URL and more.
async fn send(
    http: &reqwest::Client,
    settings: &Settings,
    method: Method,
    url: &str,
) -> Result<(), ProvisionError> {
    let authorization = settings.authorization().map_err(from_token_error)?;
    let (name, request) = match method {
        Method::Put => (
            "PUT",
            http.put(format!("{url}?data_object_size={}", BASE_LAYER.len()))
                .body(BASE_LAYER),
        ),
        Method::Delete => ("DELETE", http.delete(url)),
    };
    let response = request
        .header(reqwest::header::AUTHORIZATION, authorization)
        .header(reqwest::header::CONTENT_TYPE, "application/octet-stream")
        .send()
        .await
        .map_err(|_| ProvisionError::Transport)?;
    let status = response.status().as_u16();
    match (method, status) {
        (_, 200 | 201 | 204) | (Method::Delete, 404) => Ok(()),
        (_, 300..=399) => Err(ProvisionError::Redirect),
        _ => Err(ProvisionError::HttpStatus {
            method: name,
            status,
        }),
    }
}
