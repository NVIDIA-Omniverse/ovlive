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

// Shared auth helpers for gRPC handlers: bearer-token extraction and
// read/write permission checks against the permissions service.
//
// All handler modules should import from here to avoid duplicate
// implementations drifting in subtle ways (error messages, cache behavior).

use tonic::{Request, Status};

use super::service::LiveEditService;
use super::BearerToken;
use crate::auth::{AuthResult, Permission};
use crate::error::ServiceError;

/// Extract the bearer token from request extensions, returning a cheap
/// `Arc<str>` clone (b.ohw low). Per-RPC the original token string allocates
/// once at interceptor time; every subsequent forwarding call (auth +
/// userinfo + cache key build) clones the `Arc<str>` for ~16 bytes of
/// refcount work instead of duplicating the token bytes.
//
// `result_large_err`: tonic::Status is large but is the required return type
// for gRPC helper functions — we cannot box it.
#[allow(clippy::result_large_err)]
pub(crate) fn extract_token<T>(request: &Request<T>) -> Result<std::sync::Arc<str>, Status> {
    let bearer = request
        .extensions()
        .get::<BearerToken>()
        .ok_or(ServiceError::Unauthenticated)?;
    Ok(bearer.arc())
}

/// Where the base path handed to a permission check came from.
///
/// The permission checks need it for one outcome only:
/// [`AuthResult::InvalidRequest`], the storage service refusing the probe as
/// a malformed request. That verdict is about the base path, and who is at
/// fault depends on where the path came from:
///
/// - [`BasePathSource::Request`] — the client sent the path in this RPC
///   (`CreateSession`, `ListSessions`). The client can fix it, so the RPC
///   answers `INVALID_ARGUMENT`, a code clients do not retry, and the
///   completion logs at INFO.
/// - [`BasePathSource::Session`] — the path was read from a stored session.
///   live-edit accepted it when the session was created, and the client
///   never sent it in this RPC, so a refusal now needs an operator. The RPC
///   answers `INTERNAL` and the completion logs at ERROR.
///
/// Every permission-check call site states its source explicitly, so a new
/// handler cannot inherit the wrong mapping by default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BasePathSource {
    /// The base path is a field of the current request.
    Request,
    /// The base path was read from a stored session.
    Session,
}

/// Check read permission on a base path. Returns ServiceError on failure.
///
/// `source` says where `base_path` came from; it picks the answer when the
/// storage service refuses the probe as a malformed request (see
/// [`BasePathSource`]).
pub(crate) async fn check_read_permission(
    svc: &LiveEditService,
    token: &str,
    base_path: &str,
    source: BasePathSource,
) -> Result<(), Status> {
    check_permission(svc, token, base_path, Permission::Read, source).await
}

/// Check write permission on a base path. Returns ServiceError on failure.
///
/// `source` says where `base_path` came from; it picks the answer when the
/// storage service refuses the probe as a malformed request (see
/// [`BasePathSource`]).
pub(crate) async fn check_write_permission(
    svc: &LiveEditService,
    token: &str,
    base_path: &str,
    source: BasePathSource,
) -> Result<(), Status> {
    check_permission(svc, token, base_path, Permission::Write, source).await
}

/// Read-permission outcome for callers that need to distinguish
/// `PathNotFound` from a hard `NOT_FOUND` response (b.jr8 mediums).
///
/// `ListSessions` (SDD §7.13) returns an empty list when the base path is
/// unknown — the standard [`check_read_permission`] maps `PathNotFound` to
/// `NOT_FOUND`, which is correct for every other handler but wrong for list.
/// This helper preserves the `PathNotFound` signal so the caller can branch.
pub(crate) enum ReadDecision {
    Allowed,
    PathNotFound,
}

/// Check read permission, surfacing `PathNotFound` to the caller instead of
/// mapping it to `NOT_FOUND`. Used by `ListSessions` (SDD §7.13) so the
/// empty-list semantic stays co-located with the other auth flows.
///
/// Other results (`Denied`, `Unauthenticated`, `InvalidRequest`, transport
/// error) still map directly to `Status`, identical to
/// [`check_read_permission`]. In particular `InvalidRequest` answers an
/// error (`INVALID_ARGUMENT` for a [`BasePathSource::Request`] path), never
/// an empty list: a path the storage service refuses as malformed is not an
/// unknown path.
pub(crate) async fn check_read_permission_allowing_path_not_found(
    svc: &LiveEditService,
    token: &str,
    base_path: &str,
    source: BasePathSource,
) -> Result<ReadDecision, Status> {
    let auth_result = svc
        .permissions_client
        .check_permission(token, base_path, Permission::Read)
        .await
        .map_err(ServiceError::from)?;

    match auth_result {
        AuthResult::Allowed => Ok(ReadDecision::Allowed),
        AuthResult::Unauthenticated => Err(ServiceError::Unauthenticated.into()),
        AuthResult::Denied => Err(ServiceError::PermissionDenied.into()),
        AuthResult::PathNotFound => Ok(ReadDecision::PathNotFound),
        AuthResult::InvalidRequest { backend_status } => {
            Err(invalid_request_error(source, backend_status).into())
        }
    }
}

async fn check_permission(
    svc: &LiveEditService,
    token: &str,
    base_path: &str,
    permission: Permission,
    source: BasePathSource,
) -> Result<(), Status> {
    let auth_result = svc
        .permissions_client
        .check_permission(token, base_path, permission)
        .await
        .map_err(ServiceError::from)?;

    match auth_result {
        AuthResult::Allowed => Ok(()),
        AuthResult::Unauthenticated => Err(ServiceError::Unauthenticated.into()),
        AuthResult::Denied => Err(ServiceError::PermissionDenied.into()),
        AuthResult::PathNotFound => {
            Err(ServiceError::NotFound(format!("storage path not found: {base_path}")).into())
        }
        AuthResult::InvalidRequest { backend_status } => {
            Err(invalid_request_error(source, backend_status).into())
        }
    }
}

/// Map the storage service refusing the permission probe as a malformed
/// request ([`AuthResult::InvalidRequest`]) to the RPC's error, by where the
/// base path came from (see [`BasePathSource`]).
///
/// `backend_status` is a fixed string from the closed set
/// [`AuthResult::InvalidRequest`] documents, never backend text, so
/// rendering it here keeps the auth path's code-only, no-body posture.
///
/// - `Request`: `INVALID_ARGUMENT`. The message tells the client what to fix
///   and which status the storage service gave (`HTTP 414` means the path is
///   too long).
/// - `Session`: `INTERNAL`, which logs one ERROR line carrying this detail.
///   The detail leaves the path out because the completion line already
///   carries `base_path`.
fn invalid_request_error(source: BasePathSource, backend_status: &'static str) -> ServiceError {
    match source {
        BasePathSource::Request => ServiceError::InvalidArgument(format!(
            "base_path is not a resource address the storage service accepts \
             (storage service answered {backend_status})"
        )),
        BasePathSource::Session => ServiceError::Internal(format!(
            "auth backend: storage service refused the session's stored base_path \
             as a malformed request ({backend_status})"
        )),
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error as _;

    use tonic::{Code, Status};

    use super::{invalid_request_error, BasePathSource};
    use crate::error::{InternalFault, INTERNAL_ERROR_MSG};

    /// A base path the client sent in this RPC, refused by the storage
    /// service as malformed, answers `INVALID_ARGUMENT` with the approved
    /// wire message: it tells the client what to fix and names the storage
    /// service's status, and carries no server-fault source.
    #[test]
    fn invalid_request_from_request_path_is_invalid_argument_with_wire_message() {
        let status = Status::from(invalid_request_error(BasePathSource::Request, "HTTP 400"));

        assert_eq!(status.code(), Code::InvalidArgument);
        assert_eq!(
            status.message(),
            "base_path is not a resource address the storage service accepts \
             (storage service answered HTTP 400)"
        );
        assert!(
            status
                .source()
                .and_then(|s| s.downcast_ref::<InternalFault>())
                .is_none(),
            "a client error must not carry an InternalFault (it would log at ERROR)"
        );
    }

    /// A session's stored base path, refused by the storage service as
    /// malformed, is a server-side inconsistency: redacted `INTERNAL` on the
    /// wire, with the un-redacted detail — naming the stored-path refusal and
    /// the storage service's status — carried as the `InternalFault` source
    /// that `log_rpc_failure` turns into the one ERROR line.
    #[test]
    fn invalid_request_from_session_path_is_redacted_internal_with_fault_detail() {
        let status = Status::from(invalid_request_error(BasePathSource::Session, "HTTP 400"));

        assert_eq!(status.code(), Code::Internal);
        assert_eq!(status.message(), INTERNAL_ERROR_MSG);
        let fault = status
            .source()
            .and_then(|s| s.downcast_ref::<InternalFault>())
            .expect("a stored-path refusal must carry an InternalFault source");
        assert!(
            fault.detail().contains("stored base_path") && fault.detail().contains("HTTP 400"),
            "fault detail must name the stored-path refusal and the backend status; got {:?}",
            fault.detail()
        );
        // Nothing reports this fault in a unit test; mark it so its drop
        // fallback does not emit a stray ERROR line.
        fault.mark_reported();
    }
}
