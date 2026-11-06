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

//! How a case fails: [`CaseError`] and the [`check!`](crate::check!),
//! [`check_eq!`](crate::check_eq!) and [`expect_code!`](crate::expect_code!)
//! macros. The macros are exported at the crate root, so a case module
//! imports them with `use crate::{check, check_eq, expect_code};`.
//!
//! The macros return `Err(CaseError)` from the enclosing function or `async`
//! block, so a failed check ends the case there, in every build profile. They
//! are never `debug_assert!`-style: a release binary checks exactly what a debug
//! one does. Every failure message carries the source location of the check.
//!
//! Whether a [`CaseError`] is reported as a JUnit `<failure>` or `<error>` is
//! decided by the runner from the phase it happened in (the case body is a
//! failure; session setup and client cleanup are errors), as pytest did.

use std::fmt;

use crate::client::CallError;
use crate::config::ConfigError;

/// Why a case, or its setup or cleanup, did not pass: a message for the
/// report. It must never contain a credential.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaseError {
    message: String,
}

/// The result of a case and of the helpers a case calls.
pub type CaseResult<T = ()> = Result<T, CaseError>;

impl CaseError {
    /// A failure with `message`.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// The report message.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for CaseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for CaseError {}

/// An RPC that was expected to succeed did not: `?` on a client call.
impl From<CallError> for CaseError {
    fn from(error: CallError) -> Self {
        Self::new(error.to_string())
    }
}

impl From<ConfigError> for CaseError {
    fn from(error: ConfigError) -> Self {
        Self::new(error.to_string())
    }
}

/// The body of [`expect_code!`]: `Ok(status)` when `result` is a server or
/// transport status with `code`, otherwise the failure naming what happened
/// instead. Not called directly.
#[doc(hidden)]
pub fn expect_code_at<T>(
    result: Result<T, CallError>,
    code: tonic::Code,
    location: &str,
    context: Option<String>,
) -> CaseResult<tonic::Status> {
    let context = context.map(|c| format!("{c}: ")).unwrap_or_default();
    match result {
        Err(CallError::Status(status)) if status.code() == code => Ok(status),
        Err(CallError::Status(status)) => Err(CaseError::new(format!(
            "{context}expected {code:?}, got {:?}: {} ({location})",
            status.code(),
            status.message()
        ))),
        Err(CallError::Local(message)) => Err(CaseError::new(format!(
            "{context}expected {code:?}, got a local failure: {message} ({location})"
        ))),
        Ok(_) => Err(CaseError::new(format!(
            "{context}expected {code:?}, the call succeeded ({location})"
        ))),
    }
}

/// Fails the case unless `cond` holds: `check!(cond)` or
/// `check!(cond, "format", args…)`.
#[macro_export]
macro_rules! check {
    ($cond:expr $(,)?) => {
        if !$cond {
            return ::core::result::Result::Err(::core::convert::From::from(
                $crate::check::CaseError::new(::std::format!(
                    "check failed: `{}` ({}:{})",
                    ::core::stringify!($cond),
                    ::core::file!(),
                    ::core::line!()
                )),
            ));
        }
    };
    ($cond:expr, $($arg:tt)+) => {
        if !$cond {
            return ::core::result::Result::Err(::core::convert::From::from(
                $crate::check::CaseError::new(::std::format!(
                    "{} (`{}`, {}:{})",
                    ::std::format!($($arg)+),
                    ::core::stringify!($cond),
                    ::core::file!(),
                    ::core::line!()
                )),
            ));
        }
    };
}

/// Fails the case unless `left == right`, reporting both values' `Debug`
/// forms: `check_eq!(left, right)` or `check_eq!(left, right, "format", args…)`.
#[macro_export]
macro_rules! check_eq {
    ($left:expr, $right:expr $(,)?) => {
        match (&$left, &$right) {
            (left, right) => {
                if !(*left == *right) {
                    return ::core::result::Result::Err(::core::convert::From::from(
                        $crate::check::CaseError::new(::std::format!(
                            "check_eq failed: `{}` == `{}`\n  left: {:?}\n right: {:?} ({}:{})",
                            ::core::stringify!($left),
                            ::core::stringify!($right),
                            left,
                            right,
                            ::core::file!(),
                            ::core::line!()
                        )),
                    ));
                }
            }
        }
    };
    ($left:expr, $right:expr, $($arg:tt)+) => {
        match (&$left, &$right) {
            (left, right) => {
                if !(*left == *right) {
                    return ::core::result::Result::Err(::core::convert::From::from(
                        $crate::check::CaseError::new(::std::format!(
                            "{}: `{}` == `{}`\n  left: {:?}\n right: {:?} ({}:{})",
                            ::std::format!($($arg)+),
                            ::core::stringify!($left),
                            ::core::stringify!($right),
                            left,
                            right,
                            ::core::file!(),
                            ::core::line!()
                        )),
                    ));
                }
            }
        }
    };
}

/// Expects a client call to end with a gRPC status of `code`, and evaluates
/// to that [`tonic::Status`] for further checks (its message, its
/// `grpc-status-details-bin`). Fails the case when the call succeeded, ended
/// with another code, or failed locally
/// ([`CallError::Local`](crate::client::CallError::Local): a timeout
/// waiting for a stream message, an unreadable token file):
///
/// ```ignore
/// let status = expect_code!(ctx.client.create(&ctx.session, b"not-ompf").await, Code::InvalidArgument);
/// let status = expect_code!(stream.next().await, Code::NotFound, "deleting the session ends the stream");
/// ```
///
/// The first argument is any `Result<T, crate::client::CallError>`.
#[macro_export]
macro_rules! expect_code {
    ($result:expr, $code:expr $(,)?) => {
        match $crate::check::expect_code_at(
            $result,
            $code,
            ::core::concat!(::core::file!(), ":", ::core::line!()),
            ::core::option::Option::None,
        ) {
            ::core::result::Result::Ok(status) => status,
            ::core::result::Result::Err(error) => {
                return ::core::result::Result::Err(::core::convert::From::from(error))
            }
        }
    };
    ($result:expr, $code:expr, $($arg:tt)+) => {
        match $crate::check::expect_code_at(
            $result,
            $code,
            ::core::concat!(::core::file!(), ":", ::core::line!()),
            ::core::option::Option::Some(::std::format!($($arg)+)),
        ) {
            ::core::result::Result::Ok(status) => status,
            ::core::result::Result::Err(error) => {
                return ::core::result::Result::Err(::core::convert::From::from(error))
            }
        }
    };
}
