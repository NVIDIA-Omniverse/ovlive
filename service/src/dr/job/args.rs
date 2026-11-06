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

//! The Jobs' command line: `live-edit dr restore <incident>` and
//! `live-edit dr accept-loss <incident> --confirm discard-lost-data
//! [--confirm discard-unconfirmed-data]` (SDD SR-8.1). Pure parsing; a
//! refusal is decided here, before the process connects to anything.

use std::fmt;

/// The value of `accept-loss`'s required `--confirm`: discard what is proven
/// lost.
pub const CONFIRM_DISCARD_LOST_DATA: &str = "discard-lost-data";

/// The value of `accept-loss`'s optional second `--confirm`: also discard
/// what could not be confirmed lost because the backup store answered with
/// an error (PD-2 option E). Accepted only beside
/// [`CONFIRM_DISCARD_LOST_DATA`].
pub const CONFIRM_DISCARD_UNCONFIRMED_DATA: &str = "discard-unconfirmed-data";

/// The usage text a refused command line prints.
pub const USAGE: &str = "usage:\n  \
    live-edit dr restore <incident>\n  \
    live-edit dr accept-loss <incident> --confirm discard-lost-data \
    [--confirm discard-unconfirmed-data]";

/// Which Job runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobMode {
    /// `live-edit dr restore`: rebuild what the store lost from backups.
    Restore,
    /// `live-edit dr accept-loss`: restore what can be restored, then discard
    /// what cannot.
    AcceptLoss,
}

impl JobMode {
    /// The subcommand: `restore` or `accept-loss`. Also the `{mode}` of every
    /// operator-facing line.
    pub fn name(self) -> &'static str {
        match self {
            Self::Restore => "restore",
            Self::AcceptLoss => "accept-loss",
        }
    }
}

impl fmt::Display for JobMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A parsed, accepted command line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JobArgs {
    /// Restore or accept-loss.
    pub mode: JobMode,
    /// The incident the store must be held under.
    pub incident: String,
    /// Accept-loss only: the second `--confirm`
    /// ([`CONFIRM_DISCARD_UNCONFIRMED_DATA`]) was given, so the Job also
    /// discards what it could not confirm lost. Always `false` for restore.
    pub discard_unconfirmed: bool,
}

/// Why a command line was refused. The process prints it with [`USAGE`] and
/// exits [`super::EXIT_USAGE`] before connecting to anything.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ArgsError {
    /// No mode, or one other than `restore` and `accept-loss`.
    #[error("unknown dr mode {0:?}: expected restore or accept-loss")]
    UnknownMode(String),
    /// No incident, or an empty one, or a flag in its place.
    #[error("the dr {mode} command needs an incident id as its first argument")]
    MissingIncident {
        /// The mode.
        mode: JobMode,
    },
    /// A flag this mode does not take, or a stray argument.
    #[error("unexpected argument {0:?}")]
    Unexpected(String),
    /// A flag given more often than it may be: `--confirm` with the same
    /// value twice, or more than twice.
    #[error("{0} is given more than once")]
    Repeated(&'static str),
    /// A flag with no value.
    #[error("{0} needs a value")]
    MissingValue(&'static str),
    /// `accept-loss` without `--confirm discard-lost-data`.
    #[error(
        "dr accept-loss discards data that cannot be restored and runs only with --confirm \
         {CONFIRM_DISCARD_LOST_DATA}"
    )]
    MissingConfirm,
    /// `--confirm` with any other value.
    #[error(
        "dr accept-loss runs only with --confirm {CONFIRM_DISCARD_LOST_DATA}, not --confirm {0:?}"
    )]
    WrongConfirm(String),
    /// `--confirm discard-unconfirmed-data` without
    /// `--confirm discard-lost-data`.
    #[error(
        "--confirm {CONFIRM_DISCARD_UNCONFIRMED_DATA} is accepted only together with --confirm \
         {CONFIRM_DISCARD_LOST_DATA}"
    )]
    UnconfirmedWithoutLostData,
}

const CONFIRM: &str = "--confirm";

/// Parse the arguments after `dr` (`argv[2..]`). `--confirm`, the one flag,
/// is taken by `accept-loss` only, with its value as the next argument or
/// after `=`: once with [`CONFIRM_DISCARD_LOST_DATA`], and optionally a
/// second time, in either order, with [`CONFIRM_DISCARD_UNCONFIRMED_DATA`].
/// The second value alone is refused.
pub fn parse_job_args<I>(args: I) -> Result<JobArgs, ArgsError>
where
    I: IntoIterator<Item = String>,
{
    let mut args = args.into_iter();
    let mode = match args.next().as_deref() {
        Some("restore") => JobMode::Restore,
        Some("accept-loss") => JobMode::AcceptLoss,
        Some(other) => return Err(ArgsError::UnknownMode(other.to_owned())),
        None => return Err(ArgsError::UnknownMode(String::new())),
    };
    let incident = match args.next() {
        Some(incident) if !incident.trim().is_empty() && !incident.starts_with("--") => incident,
        _ => return Err(ArgsError::MissingIncident { mode }),
    };

    let mut confirms: Vec<String> = Vec::new();
    while let Some(arg) = args.next() {
        let (flag, inline) = match arg.split_once('=') {
            Some((flag, value)) if flag.starts_with("--") => (flag.to_owned(), Some(value)),
            _ => (arg.clone(), None),
        };
        if flag != CONFIRM || mode != JobMode::AcceptLoss {
            return Err(ArgsError::Unexpected(arg));
        }
        if confirms.len() == 2 {
            return Err(ArgsError::Repeated(CONFIRM));
        }
        let value = match inline {
            Some(value) => value.to_owned(),
            None => args.next().ok_or(ArgsError::MissingValue(CONFIRM))?,
        };
        confirms.push(value);
    }

    let mut discard_unconfirmed = false;
    if mode == JobMode::AcceptLoss {
        let mut lost_data = false;
        for confirm in &confirms {
            let seen = match confirm.as_str() {
                CONFIRM_DISCARD_LOST_DATA => &mut lost_data,
                CONFIRM_DISCARD_UNCONFIRMED_DATA => &mut discard_unconfirmed,
                other => return Err(ArgsError::WrongConfirm(other.to_owned())),
            };
            if *seen {
                return Err(ArgsError::Repeated(CONFIRM));
            }
            *seen = true;
        }
        match (lost_data, discard_unconfirmed) {
            (true, _) => {}
            (false, true) => return Err(ArgsError::UnconfirmedWithoutLostData),
            (false, false) => return Err(ArgsError::MissingConfirm),
        }
    }
    Ok(JobArgs {
        mode,
        incident,
        discard_unconfirmed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<JobArgs, ArgsError> {
        parse_job_args(args.iter().map(|a| (*a).to_owned()))
    }

    #[test]
    fn restore_with_an_incident_parses() {
        let args = parse(&["restore", "inc-1"]).unwrap();
        assert_eq!(args.mode, JobMode::Restore);
        assert_eq!(args.incident, "inc-1");
    }

    #[test]
    fn accept_loss_needs_the_exact_confirm() {
        let ok = parse(&["accept-loss", "inc-1", "--confirm", "discard-lost-data"]).unwrap();
        assert_eq!(ok.mode, JobMode::AcceptLoss);
        assert_eq!(
            parse(&["accept-loss", "inc-1"]),
            Err(ArgsError::MissingConfirm)
        );
        assert_eq!(
            parse(&["accept-loss", "inc-1", "--confirm", "yes"]),
            Err(ArgsError::WrongConfirm("yes".into()))
        );
        assert!(parse(&["accept-loss", "inc-1", "--confirm=discard-lost-data"]).is_ok());
    }

    #[test]
    fn restore_refuses_a_confirm() {
        assert_eq!(
            parse(&["restore", "inc-1", "--confirm", "discard-lost-data"]),
            Err(ArgsError::Unexpected("--confirm".into()))
        );
    }

    #[test]
    fn mode_and_incident_are_required() {
        assert_eq!(parse(&[]), Err(ArgsError::UnknownMode(String::new())));
        assert_eq!(
            parse(&["rebuild", "x"]),
            Err(ArgsError::UnknownMode("rebuild".into()))
        );
        assert_eq!(
            parse(&["restore"]),
            Err(ArgsError::MissingIncident {
                mode: JobMode::Restore
            })
        );
        assert_eq!(
            parse(&["restore", "--confirm"]),
            Err(ArgsError::MissingIncident {
                mode: JobMode::Restore
            })
        );
    }

    #[test]
    fn repeated_stray_and_valueless_flags_are_refused() {
        let confirm = ["accept-loss", "inc-1", "--confirm", "discard-lost-data"];
        let mut repeated = confirm.to_vec();
        repeated.extend(["--confirm", "discard-lost-data"]);
        assert_eq!(parse(&repeated), Err(ArgsError::Repeated(CONFIRM)));
        let mut stray = confirm.to_vec();
        stray.push("extra");
        assert_eq!(parse(&stray), Err(ArgsError::Unexpected("extra".into())));
        assert_eq!(
            parse(&["accept-loss", "inc-1", "--confirm"]),
            Err(ArgsError::MissingValue(CONFIRM))
        );
    }

    /// Option E: the second confirm is accepted beside the first, in either
    /// order and in either spelling, and sets `discard_unconfirmed`; the
    /// first alone leaves it unset.
    #[test]
    fn accept_loss_takes_the_second_confirm_beside_the_first_in_either_order() {
        for args in [
            &[
                "accept-loss",
                "inc-1",
                "--confirm",
                "discard-lost-data",
                "--confirm",
                "discard-unconfirmed-data",
            ][..],
            &[
                "accept-loss",
                "inc-1",
                "--confirm",
                "discard-unconfirmed-data",
                "--confirm",
                "discard-lost-data",
            ][..],
            &[
                "accept-loss",
                "inc-1",
                "--confirm=discard-unconfirmed-data",
                "--confirm=discard-lost-data",
            ][..],
        ] {
            assert_eq!(
                parse(args),
                Ok(JobArgs {
                    mode: JobMode::AcceptLoss,
                    incident: "inc-1".to_owned(),
                    discard_unconfirmed: true,
                }),
                "{args:?}"
            );
        }
        let first_alone = parse(&["accept-loss", "inc-1", "--confirm", "discard-lost-data"]);
        assert_eq!(first_alone.map(|a| a.discard_unconfirmed), Ok(false));
    }

    /// Option E's refusals (exit 2 in the binary): the second confirm alone,
    /// either confirm given twice, and a third `--confirm`.
    #[test]
    fn accept_loss_refuses_the_second_confirm_alone_or_any_confirm_repeated() {
        assert_eq!(
            parse(&[
                "accept-loss",
                "inc-1",
                "--confirm",
                "discard-unconfirmed-data"
            ]),
            Err(ArgsError::UnconfirmedWithoutLostData)
        );
        assert_eq!(
            parse(&[
                "accept-loss",
                "inc-1",
                "--confirm",
                "discard-unconfirmed-data",
                "--confirm",
                "discard-unconfirmed-data",
            ]),
            Err(ArgsError::Repeated(CONFIRM)),
            "the second confirm twice"
        );
        assert_eq!(
            parse(&[
                "accept-loss",
                "inc-1",
                "--confirm",
                "discard-lost-data",
                "--confirm",
                "discard-lost-data",
                "--confirm",
                "discard-unconfirmed-data",
            ]),
            Err(ArgsError::Repeated(CONFIRM)),
            "a third --confirm"
        );
        assert_eq!(
            parse(&[
                "accept-loss",
                "inc-1",
                "--confirm",
                "discard-unconfirmed-data",
                "--confirm",
                "yes",
            ]),
            Err(ArgsError::WrongConfirm("yes".into()))
        );
    }
}
