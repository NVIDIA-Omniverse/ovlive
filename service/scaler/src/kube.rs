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

//! A minimal Kubernetes REST client: the four calls the scaler makes, and
//! nothing else.
//!
//! | Call | Verb on | RBAC the chart grants |
//! |---|---|---|
//! | [`KubeClient::list_endpoints`] | `list` `discovery.k8s.io/endpointslices` | every EndpointSlice in the namespace (slice names are generated, so the rule cannot name the Service's) |
//! | [`KubeClient::get_scale`] | `get` `apps/deployments/scale` | the service Deployment only |
//! | [`KubeClient::patch_replicas`] | `patch` `apps/deployments/scale` | the service Deployment only |
//! | [`KubeClient::patch_pod_deletion_cost`] | `patch` `pods` | only with `podDeletionCost: true`; every pod in the namespace |
//!
//! Pods come from the Service's EndpointSlices and replica counts from
//! `/scale`, which carries the counts and the label selector but no pod
//! template. So with the default rules the scaler can read no pod template,
//! Pod, Secret or ConfigMap, and so none of the containers' inline
//! environment values. With `podDeletionCost: true` it can read every pod
//! in the namespace: a patch returns the whole pod. A general-purpose
//! client crate was not used: the four calls need none of its machinery,
//! and it would bring a large generated API type tree into the build.
//!
//! The service-account token is projected and rotated by the kubelet, so it is
//! re-read from its file on every request rather than cached.

use std::path::{Path, PathBuf};
use std::time::Duration;

use reqwest::{Method, StatusCode};
use serde::Deserialize;

/// The projected service-account files every pod gets when its token is
/// mounted.
pub const SERVICE_ACCOUNT_DIR: &str = "/var/run/secrets/kubernetes.io/serviceaccount";

/// Port name on the Service (and so on its EndpointSlices) that serves the
/// pods' `/metrics` and `/scaling/v1`. The chart's Service names it so.
pub const METRICS_PORT_NAME: &str = "metrics";

/// The pod annotation the ReplicaSet controller reads to choose which pods to
/// remove first on a scale-down: lower cost goes first.
pub const POD_DELETION_COST_ANNOTATION: &str = "controller.kubernetes.io/pod-deletion-cost";

/// Bound on one API request, connection included. Well under the default
/// 5 s poll interval, so a slow API server delays one poll rather than piling
/// polls up.
const API_REQUEST_TIMEOUT: Duration = Duration::from_secs(4);

/// How much of an error response body is kept in the error, for the log line.
const ERROR_BODY_EXCERPT: usize = 256;

#[derive(Debug, thiserror::Error)]
pub enum KubeError {
    /// The in-cluster environment is incomplete (not running in a pod, or the
    /// service-account token is not mounted).
    #[error("not running in a Kubernetes pod with a mounted service-account token: {0}")]
    NotInCluster(String),
    #[error("could not read the service-account token {path}: {source}")]
    Token {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("could not build the API client: {0}")]
    Client(String),
    #[error("API request {method} {path} failed: {source}")]
    Transport {
        method: Method,
        path: String,
        source: reqwest::Error,
    },
    #[error("API request {method} {path} answered {status}: {body}")]
    Status {
        method: Method,
        path: String,
        status: StatusCode,
        body: String,
    },
    #[error("API response to {path} did not parse: {source}")]
    Decode {
        path: String,
        source: serde_json::Error,
    },
}

/// One server pod as an EndpointSlice lists it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Endpoint {
    /// The pod's UID: the key of everything the scaler remembers about a pod,
    /// so a replacement pod with a recycled name or address starts fresh.
    pub uid: String,
    /// The pod's name, for the deletion-cost annotation and logs.
    pub name: String,
    /// The pod's address (IPv4, or IPv6 without brackets).
    pub address: String,
    /// The metrics port, from the slice's [`METRICS_PORT_NAME`] port. `None`
    /// when the slice carries no such port.
    pub metrics_port: Option<u16>,
    /// The endpoint is ready to serve (`conditions.ready`; an absent value
    /// counts as ready, as the API documents).
    pub ready: bool,
    /// The pod is terminating (`conditions.terminating`).
    pub terminating: bool,
}

impl Endpoint {
    /// `http://<address>:<port><path>`, bracketing an IPv6 address. `None`
    /// without a metrics port.
    pub fn url(&self, path: &str) -> Option<String> {
        let port = self.metrics_port?;
        Some(if self.address.contains(':') {
            format!("http://[{}]:{port}{path}", self.address)
        } else {
            format!("http://{}:{port}{path}", self.address)
        })
    }
}

/// The Deployment's replica counts, from its `/scale` subresource.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Scale {
    /// `spec.replicas`: the count the Deployment is asked to run.
    pub replicas: u32,
    /// `status.replicas`: the pods its ReplicaSets have now that are not
    /// terminal or being deleted, Pending ones included. 0 when absent.
    pub status_replicas: u32,
}

/// The scaler's API client. Cheap to clone.
#[derive(Debug, Clone)]
pub struct KubeClient {
    /// `https://host:port`, no trailing slash.
    base: String,
    token_path: PathBuf,
    http: reqwest::Client,
}

impl KubeClient {
    /// A client for `base_url` (`https://host:port` or, in tests, `http://…`)
    /// that authenticates with the bearer token in `token_path`, re-read per
    /// request, and trusts only `ca_pem` when one is given.
    pub fn new(
        base_url: &str,
        token_path: impl Into<PathBuf>,
        ca_pem: Option<&[u8]>,
    ) -> Result<Self, KubeError> {
        let mut builder = reqwest::Client::builder().timeout(API_REQUEST_TIMEOUT);
        if let Some(pem) = ca_pem {
            let ca = reqwest::Certificate::from_pem(pem)
                .map_err(|e| KubeError::Client(format!("cluster CA certificate: {e}")))?;
            // Only the cluster CA: the API server's certificate is never
            // signed by a public root, and trusting public roots here would
            // only widen who can impersonate it.
            builder = builder
                .tls_built_in_root_certs(false)
                .add_root_certificate(ca);
        }
        let http = builder
            .build()
            .map_err(|e| KubeError::Client(e.to_string()))?;
        Ok(Self {
            base: base_url.trim_end_matches('/').to_owned(),
            token_path: token_path.into(),
            http,
        })
    }

    /// The client a pod gets: the API server from `KUBERNETES_SERVICE_HOST`
    /// and `KUBERNETES_SERVICE_PORT`, the token and cluster CA from the
    /// mounted service account.
    pub fn in_cluster() -> Result<Self, KubeError> {
        let host = std::env::var("KUBERNETES_SERVICE_HOST")
            .map_err(|_| KubeError::NotInCluster("KUBERNETES_SERVICE_HOST is not set".into()))?;
        let port = std::env::var("KUBERNETES_SERVICE_PORT")
            .map_err(|_| KubeError::NotInCluster("KUBERNETES_SERVICE_PORT is not set".into()))?;
        let host = if host.contains(':') {
            format!("[{host}]")
        } else {
            host
        };
        let dir = Path::new(SERVICE_ACCOUNT_DIR);
        let ca_path = dir.join("ca.crt");
        let ca = std::fs::read(&ca_path).map_err(|e| {
            KubeError::NotInCluster(format!("cannot read {}: {e}", ca_path.display()))
        })?;
        let token_path = dir.join("token");
        if !token_path.exists() {
            return Err(KubeError::NotInCluster(format!(
                "{} does not exist (automountServiceAccountToken must be true on the scaler pod)",
                token_path.display()
            )));
        }
        Self::new(&format!("https://{host}:{port}"), token_path, Some(&ca))
    }

    /// Every endpoint of the Service's EndpointSlices, one per pod UID.
    /// Endpoints without a pod reference are skipped: the scaler can neither
    /// key nor annotate them. A pod listed in more than one slice (one per
    /// address family) is kept once, at its first listing.
    pub async fn list_endpoints(
        &self,
        namespace: &str,
        service: &str,
    ) -> Result<Vec<Endpoint>, KubeError> {
        let path = format!("/apis/discovery.k8s.io/v1/namespaces/{namespace}/endpointslices");
        let query = [(
            "labelSelector",
            format!("kubernetes.io/service-name={service}"),
        )];
        let list: wire::EndpointSliceList =
            self.json(Method::GET, &path, Some(&query), None).await?;
        let mut seen = std::collections::HashSet::new();
        let mut endpoints = Vec::new();
        for slice in list.items {
            if !matches!(slice.address_type.as_str(), "IPv4" | "IPv6") {
                continue;
            }
            let metrics_port = slice
                .ports
                .iter()
                .find(|p| p.name.as_deref() == Some(METRICS_PORT_NAME))
                .and_then(|p| p.port)
                .and_then(|p| u16::try_from(p).ok());
            for ep in slice.endpoints {
                let Some(target) = ep.target_ref.filter(|t| t.kind.as_deref() == Some("Pod"))
                else {
                    continue;
                };
                let (Some(uid), Some(name), Some(address)) =
                    (target.uid, target.name, ep.addresses.into_iter().next())
                else {
                    continue;
                };
                if !seen.insert(uid.clone()) {
                    continue;
                }
                endpoints.push(Endpoint {
                    uid,
                    name,
                    address,
                    metrics_port,
                    ready: ep.conditions.ready.unwrap_or(true),
                    terminating: ep.conditions.terminating.unwrap_or(false),
                });
            }
        }
        Ok(endpoints)
    }

    /// `spec.replicas` and `status.replicas` of the Deployment's `/scale`
    /// subresource, in one GET. The scaler never reads the Deployment object
    /// itself, which carries the pod template.
    pub async fn get_scale(&self, namespace: &str, deployment: &str) -> Result<Scale, KubeError> {
        let path = scale_path(namespace, deployment);
        let scale: wire::Scale = self.json(Method::GET, &path, None, None).await?;
        Ok(Scale {
            replicas: scale.spec.replicas.unwrap_or(0),
            status_replicas: scale.status.replicas.unwrap_or(0),
        })
    }

    /// Set `spec.replicas` on the Deployment's `/scale` subresource, as a JSON
    /// merge patch. The value is absolute, so a repeated or late patch never
    /// compounds.
    pub async fn patch_replicas(
        &self,
        namespace: &str,
        deployment: &str,
        replicas: u32,
    ) -> Result<(), KubeError> {
        let path = scale_path(namespace, deployment);
        let body = serde_json::json!({ "spec": { "replicas": replicas } });
        self.merge_patch(&path, &body).await
    }

    /// Set the pod's [`POD_DELETION_COST_ANNOTATION`] to `cost`.
    pub async fn patch_pod_deletion_cost(
        &self,
        namespace: &str,
        pod: &str,
        cost: i32,
    ) -> Result<(), KubeError> {
        let path = format!("/api/v1/namespaces/{namespace}/pods/{pod}");
        let body = serde_json::json!({
            "metadata": { "annotations": { POD_DELETION_COST_ANNOTATION: cost.to_string() } }
        });
        self.merge_patch(&path, &body).await
    }

    async fn merge_patch(&self, path: &str, body: &serde_json::Value) -> Result<(), KubeError> {
        let body = serde_json::to_vec(body).map_err(|source| KubeError::Decode {
            path: path.to_owned(),
            source,
        })?;
        self.send(Method::PATCH, path, None, Some(body)).await?;
        Ok(())
    }

    async fn json<T: for<'de> Deserialize<'de>>(
        &self,
        method: Method,
        path: &str,
        query: Option<&[(&str, String)]>,
        body: Option<Vec<u8>>,
    ) -> Result<T, KubeError> {
        let bytes = self.send(method, path, query, body).await?;
        serde_json::from_slice(&bytes).map_err(|source| KubeError::Decode {
            path: path.to_owned(),
            source,
        })
    }

    /// One request: the token re-read, a merge-patch content type on a body,
    /// and any non-2xx answer turned into [`KubeError::Status`].
    async fn send(
        &self,
        method: Method,
        path: &str,
        query: Option<&[(&str, String)]>,
        body: Option<Vec<u8>>,
    ) -> Result<Vec<u8>, KubeError> {
        let token =
            std::fs::read_to_string(&self.token_path).map_err(|source| KubeError::Token {
                path: self.token_path.clone(),
                source,
            })?;
        let mut request = self
            .http
            .request(method.clone(), format!("{}{path}", self.base))
            .bearer_auth(token.trim())
            .header(reqwest::header::ACCEPT, "application/json");
        if let Some(query) = query {
            request = request.query(query);
        }
        if let Some(body) = body {
            request = request
                .header(
                    reqwest::header::CONTENT_TYPE,
                    "application/merge-patch+json",
                )
                .body(body);
        }
        let transport = |source| KubeError::Transport {
            method: method.clone(),
            path: path.to_owned(),
            source,
        };
        let response = request.send().await.map_err(transport)?;
        let status = response.status();
        let bytes = response.bytes().await.map_err(transport)?;
        if !status.is_success() {
            let excerpt =
                String::from_utf8_lossy(&bytes[..bytes.len().min(ERROR_BODY_EXCERPT)]).into_owned();
            return Err(KubeError::Status {
                method,
                path: path.to_owned(),
                status,
                body: excerpt,
            });
        }
        Ok(bytes.to_vec())
    }
}

fn scale_path(namespace: &str, deployment: &str) -> String {
    format!("/apis/apps/v1/namespaces/{namespace}/deployments/{deployment}/scale")
}

/// The fields the scaler reads from API objects; everything else is ignored.
mod wire {
    use serde::Deserialize;

    #[derive(Deserialize)]
    pub(super) struct EndpointSliceList {
        #[serde(default)]
        pub items: Vec<EndpointSlice>,
    }

    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub(super) struct EndpointSlice {
        #[serde(default)]
        pub address_type: String,
        #[serde(default)]
        pub endpoints: Vec<Endpoint>,
        #[serde(default)]
        pub ports: Vec<Port>,
    }

    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub(super) struct Endpoint {
        #[serde(default)]
        pub addresses: Vec<String>,
        #[serde(default)]
        pub conditions: Conditions,
        pub target_ref: Option<TargetRef>,
    }

    #[derive(Deserialize, Default)]
    pub(super) struct Conditions {
        pub ready: Option<bool>,
        pub terminating: Option<bool>,
    }

    #[derive(Deserialize)]
    pub(super) struct TargetRef {
        pub kind: Option<String>,
        pub name: Option<String>,
        pub uid: Option<String>,
    }

    #[derive(Deserialize)]
    pub(super) struct Port {
        pub name: Option<String>,
        pub port: Option<i64>,
    }

    #[derive(Deserialize)]
    pub(super) struct Scale {
        #[serde(default)]
        pub spec: ScaleSpec,
        #[serde(default)]
        pub status: ScaleStatus,
    }

    #[derive(Deserialize, Default)]
    pub(super) struct ScaleSpec {
        pub replicas: Option<u32>,
    }

    #[derive(Deserialize, Default)]
    pub(super) struct ScaleStatus {
        pub replicas: Option<u32>,
    }
}
