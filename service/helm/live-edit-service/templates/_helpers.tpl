{{- /*
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/ -}}

{{/*
Expand the name of the chart.
*/}}
{{- define "live-edit-service.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Canonical Service name. Deliberately NOT release-prefixed: the Service is the
stable network identity peers, ingress backends, and in-cluster clients resolve
by DNS, so it must be predictable across releases. Overridable via
service.nameOverride; defaults to the chart name (`live-edit-service`). The
Deployment / PDB / ServiceMonitor / NetworkPolicy stay on `fullname` (they are
per-release resources), so only the Service and its backend references use this.
*/}}
{{- define "live-edit-service.servicename" -}}
{{- default .Chart.Name .Values.service.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "live-edit-service.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "live-edit-service.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "live-edit-service.labels" -}}
helm.sh/chart: {{ include "live-edit-service.chart" . }}
{{ include "live-edit-service.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "live-edit-service.selectorLabels" -}}
app.kubernetes.io/name: {{ include "live-edit-service.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{/*
ServiceAccount name for the main Deployment. Operator-supplied `name` wins and
means "use it, don't touch it": when `serviceAccount.name` is set the chart
references that pre-provisioned ServiceAccount and NEVER renders its own SA
object (production clusters commonly forbid Helm from creating or modifying
ServiceAccounts; a chart-rendered SA over a pre-existing one fails the upgrade
with Helm's invalid-ownership-metadata error). `serviceAccount.create` is only
consulted when `name` is empty: true (default) creates the generated-name SA;
false with no name would silently run the pod as the namespace's shared
`default` SA, so it fails loudly instead.
*/}}
{{- define "live-edit-service.serviceAccountName" -}}
{{- if .Values.serviceAccount.name -}}
{{- .Values.serviceAccount.name -}}
{{- else if .Values.serviceAccount.create -}}
{{- include "live-edit-service.fullname" . -}}
{{- else -}}
{{- fail "serviceAccount: name is empty and create is false — the pod would silently run as the namespace's shared `default` ServiceAccount. Set serviceAccount.name to your pre-provisioned ServiceAccount (the chart will use it without creating or modifying it), or set serviceAccount.create=true to let the chart create one." -}}
{{- end -}}
{{- end }}

{{/*
Service account name for the smoke-test Job. Separate from the main
service SA so the test runs under its own AAD identity — the smoke-test
is a client of live-edit-service, not the service itself.
*/}}
{{- define "live-edit-service.smokeTestServiceAccountName" -}}
{{- if .Values.tests.serviceAccount.name -}}
{{- .Values.tests.serviceAccount.name | trunc 63 | trimSuffix "-" -}}
{{- else if .Values.tests.serviceAccount.create -}}
{{- printf "%s-test-smoke" (include "live-edit-service.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- fail "tests.serviceAccount: name is empty and create is false — the smoke-test pod would silently run as the namespace's shared `default` ServiceAccount. Set tests.serviceAccount.name to a pre-provisioned ServiceAccount (used without creating or modifying it), or set tests.serviceAccount.create=true." -}}
{{- end -}}
{{- end }}

{{- define "live-edit-service.imagePullSecrets" -}}
  {{- $pullSecrets := list }}

  {{- range ((.global).imagePullSecrets) -}}
    {{- $pullSecrets = append $pullSecrets (get . "name") -}}
  {{- end -}}

  {{- range .images -}}
    {{- range .pullSecrets -}}
      {{- $pullSecrets = append $pullSecrets (get . "name") -}}
    {{- end -}}
  {{- end -}}

  {{- if (not (empty $pullSecrets)) -}}
imagePullSecrets:
    {{- range $pullSecrets | uniq }}
  - name: {{ . }}
    {{- end -}}
  {{- end -}}
{{- end -}}

{{/*
A non-negative integer setting, returned as plain decimal digits; anything
else fails the render naming the key. Call with
(dict "key" "<dotted values path>" "value" <the value> "hint" "<unit / meaning>").
Null and "" fail here too, naming the key; a caller adds `required` only to
give a fuller message.

Why not `| int64` alone: sprig's int64 swallows its cast error, so "15s",
"fast" or true render "0", and 1.5 truncates to "1", each a silent change of
meaning. Accepted: a whole YAML number (Helm decodes values-file numbers as
float64; the int64 round trip keeps a large one from rendering with an
exponent, e.g. "8.64e+07"), an integer from `--set`, and a string of digits.
A string is taken the way the binary's integer parse takes it: surrounding
whitespace is trimmed and one leading "+" is allowed, so " 100" and "+120"
render "100" and "120". Leading zeros are dropped, so "015000" renders
"15000".
*/}}
{{- define "live-edit-service.nonNegativeInteger" -}}
{{- $value := .value -}}
{{- $digits := trimPrefix "+" (trim (toString $value)) -}}
{{- if kindIs "float64" $value -}}
{{- if eq $value (float64 (int64 $value)) -}}
{{- $digits = toString (int64 $value) -}}
{{- end -}}
{{- end -}}
{{- if not (regexMatch "^[0-9]+$" $digits) -}}
{{- fail (printf "%s must be a non-negative whole number (%s), got %q" .key .hint (toString $value)) -}}
{{- end -}}
{{- regexReplaceAll "^0+([0-9])" $digits "${1}" -}}
{{- end }}

{{- define "live-edit-service.tls-secret-name" -}}
{{- $defaultname := print "tls-cert-" (include "live-edit-service.fullname" .) -}}
{{- $name := default $defaultname .Values.httpProxy.tls.secretName }}
{{- printf "%s" $name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Derive the Valkey auth mode from the enabled-boolean credential sections.
AT MOST ONE of valkey.credentials.{static,accessTokenFile,awsIam}.enabled may be
true. With NONE enabled (the default) the mode is "static" — password / ACL
credentials carried in the connection URL, the binary's own default. Enabling
more than one fails rendering. Returns the string the binary's
LIVE_EDIT_VALKEY_AUTH_MODE expects: "static" | "accessTokenFile" | "iam".
*/}}
{{- define "live-edit-service.valkeyAuthMode" -}}
{{- $c := .Values.valkey.credentials -}}
{{- $modes := list -}}
{{- if $c.static.enabled -}}{{- $modes = append $modes "static" -}}{{- end -}}
{{- if $c.accessTokenFile.enabled -}}{{- $modes = append $modes "accessTokenFile" -}}{{- end -}}
{{- if $c.awsIam.enabled -}}{{- $modes = append $modes "iam" -}}{{- end -}}
{{- if gt (len $modes) 1 -}}
{{- fail (printf "valkey.credentials: at most one of static / accessTokenFile / awsIam may have enabled: true (found %d enabled: %v). Leave all disabled for static (URL-embedded) credentials, or enable exactly one." (len $modes) $modes) -}}
{{- end -}}
{{- if eq (len $modes) 0 -}}static{{- else -}}{{- index $modes 0 -}}{{- end -}}
{{- end }}

{{/*
Shared token-exchange mount directory. The token provisioner writes every
file-based token here; the live-edit container reads them read-only. Derived
from the checkpoint service-identity token path's directory (the Valkey token
path must share this directory so both files live in the one mounted volume).
*/}}
{{- define "live-edit-service.tokenMountPath" -}}
{{- .Values.checkpoint.credentials.accessTokenFile.path | dir -}}
{{- end }}

{{/*
Whether the single credential-provisioner sidecar should render: the
access-token-file service-identity mode is enabled AND a container image was
supplied. Resource defaults alone must not create a sidecar: operators can
provide the token file through their own volumes without running this container.
*/}}
{{- define "live-edit-service.accessTokenSidecarEnabled" -}}
{{- $provisioner := .Values.serviceIdentity.accessTokenFile.provisionerSidecar | default dict -}}
{{- $container := $provisioner.container | default dict -}}
{{- if and .Values.serviceIdentity.accessTokenFile.enabled $container.image -}}
true
{{- end -}}
{{- end }}

{{/*
Read-only token-exchange volumeMount for the live-edit (main) container.
*/}}
{{- define "live-edit-service.accessTokenVolumeMount" -}}
- name: token-exchange
  readOnly: true
  mountPath: {{ include "live-edit-service.tokenMountPath" . | quote }}
{{- end }}

{{/*
The token-provisioner container. The chart supplies a predefined name, which
replaces any operator `name`, and appends the writable token-exchange
volumeMount to any operator `volumeMounts`; the rest of the container (image,
env, args) comes verbatim from values.
*/}}
{{- define "live-edit-service.accessTokenSidecar" -}}
- name: {{ .Chart.Name }}-token-provisioner
  {{- (omit .Values.serviceIdentity.accessTokenFile.provisionerSidecar.container "volumeMounts" "name") | toYaml | nindent 2 }}
  volumeMounts:
    {{- with .Values.serviceIdentity.accessTokenFile.provisionerSidecar.container.volumeMounts }}
    {{- toYaml . | nindent 4 }}
    {{- end }}
    - name: token-exchange
      mountPath: {{ include "live-edit-service.tokenMountPath" . | quote }}
{{- end }}

{{/*
The shared memory-backed (tmpfs) token-exchange volume. Token material stays in
RAM and never spills to node disk; sizeLimit bounds it against pod memory.
*/}}
{{- define "live-edit-service.accessTokenVolume" -}}
- name: token-exchange
  emptyDir:
    sizeLimit: 4Mi
    medium: Memory
{{- end }}

{{/*
The autoscaler that owns the server Deployment's replica count when
autoscaling.enabled is true, from autoscaling.mode: "scaler" (the chart's
scaler Deployment, the default) or "hpa" (a HorizontalPodAutoscaler). Fails
rendering on any other value, and, in scaler mode, on settings the scaler
would refuse or ignore: a memory utilization target (the scaler has no
memory term), a CPU target with no resources.requests.cpu to take it as a
percentage of, and minReplicas above maxReplicas (the scaler's own
configuration check refuses both at startup, so the render fails first).

A missing autoscaling.mode fails with a message naming the usual cause:
values that predate this chart, kept by `helm upgrade --reuse-values`,
which replaces the chart's defaults with the old release's values. In
scaler mode the render also fails on an unset autoscaling.scaler setting or
one of the wrong type; the rule is in the comment above that check.
Evaluated by hpa.yaml whenever autoscaling is enabled, so a bad value fails
every render; hpa mode and autoscaling.enabled: false never check the
scaler block.
*/}}
{{- define "live-edit-service.autoscalingMode" -}}
{{- $mode := .Values.autoscaling.mode | default "" | toString -}}
{{- if eq $mode "" -}}
{{- fail "autoscaling.mode must be \"scaler\" or \"hpa\" (got \"\"): it is missing, usually because the values predate the autoscaling.mode setting (a `helm upgrade --reuse-values` keeps the old release's values). Upgrade with `--reset-then-reuse-values` (Helm 3.14+) in place of `--reuse-values`, or with your values files" -}}
{{- end -}}
{{- if not (has $mode (list "scaler" "hpa")) -}}
{{- fail (printf "autoscaling.mode must be \"scaler\" or \"hpa\" (got %q)" $mode) -}}
{{- end -}}
{{- if eq $mode "scaler" -}}
{{- /*
  The autoscaling.scaler block. Every key must be set, where "unset" means
  absent, null or the empty string (what Helm's `required` refuses). A null
  reaches this check only on the `--reuse-values` path: that flag uses the
  old release's values in place of the chart's defaults, and values that
  predate the scaler do not define its keys, so nothing deletes the user's
  null; on a normal install Helm deletes a null for any key the
  chart's defaults define, and the key reads as absent. Why each
  key is checked, in values.yaml order, which is the order the first failure
  is reported in:
  - the block itself, and the scaleDown, serviceAccount and rbac maps: the
    templates dereference them, so unset is a nil-pointer error; each must
    also be a map;
  - pollIntervalSeconds, catchupTargetSeconds, occupancyHeadroomPercent,
    scaleDown.windowSeconds, scaleDown.intervalSeconds: unset renders an
    empty value, and the scaler would quietly run on its own built-in
    default instead of the chart's value;
  - podDeletionCost: unset renders "<nil>". It must also be an unquoted
    boolean: scaler-rbac.yaml tests it with `if`, where any non-empty
    string, "false" included, is true, while the scaler reads its text, so a
    string would leave the Role and the scaler disagreeing;
  - rbac.create: unset silently renders no Role or RoleBinding;
  - serviceAccount.create, only while serviceAccount.name is unset (with a
    name, create is never read): unset reverses its true default, and the
    ServiceAccount helper would then blame "create is false".
  Exempt, because every template that reads them renders a correct result
  when they are absent or null: resources (renders resources: null, so no
  requests or limits), serviceAccount.name (read only for truthiness, the
  same as "", the default), podAnnotations and affinity (each read through
  `with`: no annotations, no affinity). serviceAccount.name, podAnnotations
  and affinity also render correctly as "", but resources does not: ""
  renders resources: "", which Kubernetes refuses at apply.
  Only podDeletionCost gets a type check, because it is the only key read by
  two consumers that interpret a string differently: scaler-rbac.yaml tests
  it with `if`, while scaler-deployment.yaml passes its text to the scaler.
  A quoted rbac.create or serviceAccount.create is only ever tested with
  `if` (rbac.create in scaler-rbac.yaml; serviceAccount.create there and in
  the scalerServiceAccountName helper), where it reads as true, so its
  readers agree, the same way the server's own serviceAccount.create does
  (serviceaccount.yaml); those booleans get no type check. The integers get no type or range check here
  either: the scaler refuses a non-integer or out-of-range value at startup
  and names the variable.
  A key added later is exempt only if every template that reads it renders a
  correct result when it is absent or null and its @param says what null
  means; otherwise it goes on the list below.
  The list is a `list` of dotted paths, never a `dict`, whose keys `range`
  would sort; a child is checked only after its parent passed.
*/ -}}
{{- $scaler := .Values.autoscaling.scaler -}}
{{- $problem := "" -}}
{{- $unsetText := "autoscaling.scaler is missing or incomplete (%s is absent, null or empty). Either the values predate the scaler: a `helm upgrade --reuse-values` keeps the old release's values in place of this chart's defaults. Upgrade with `--reset-then-reuse-values` (Helm 3.14+) in place of `--reuse-values`, or with your values files (setting autoscaling.mode: hpa there to keep the HorizontalPodAutoscaler). Or %s was set to null or empty, which it does not accept; in autoscaling.scaler only resources, serviceAccount.name, podAnnotations and affinity accept null" -}}
{{- if or (kindIs "invalid" $scaler) (eq (toString $scaler) "") -}}
{{- $problem = printf $unsetText "autoscaling.scaler" "autoscaling.scaler" -}}
{{- else if not (kindIs "map" $scaler) -}}
{{- $problem = printf "%s must be %s (got %#v)" "autoscaling.scaler" "a map of settings" $scaler -}}
{{- else -}}
{{- range $path := list "pollIntervalSeconds" "catchupTargetSeconds" "occupancyHeadroomPercent" "scaleDown" "scaleDown.windowSeconds" "scaleDown.intervalSeconds" "podDeletionCost" "serviceAccount" "serviceAccount.create" "rbac" "rbac.create" -}}
{{- if not $problem -}}
{{- $parts := splitList "." $path -}}
{{- $parent := $scaler -}}
{{- if eq (len $parts) 2 -}}
{{- $parent = get $scaler (first $parts) -}}
{{- end -}}
{{- $value := get $parent (last $parts) -}}
{{- $key := printf "autoscaling.scaler.%s" $path -}}
{{- $checked := true -}}
{{- if eq $path "serviceAccount.create" -}}
{{- $name := get $parent "name" -}}
{{- $checked = or (kindIs "invalid" $name) (eq (toString $name) "") -}}
{{- end -}}
{{- if $checked -}}
{{- if or (kindIs "invalid" $value) (eq (toString $value) "") -}}
{{- $problem = printf $unsetText $key $key -}}
{{- else if and (has $path (list "scaleDown" "serviceAccount" "rbac")) (not (kindIs "map" $value)) -}}
{{- $problem = printf "%s must be %s (got %#v)" $key "a map of settings" $value -}}
{{- else if and (eq $path "podDeletionCost") (not (kindIs "bool" $value)) -}}
{{- $problem = printf "%s must be %s (got %#v)" $key "true or false, unquoted" $value -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- if $problem -}}
{{- fail $problem -}}
{{- end -}}
{{- if .Values.autoscaling.targetMemoryUtilizationPercentage -}}
{{- fail (printf "autoscaling.targetMemoryUtilizationPercentage (%v) applies only in autoscaling.mode=hpa: the scaler does not scale on memory. Unset it, or set autoscaling.mode=hpa" .Values.autoscaling.targetMemoryUtilizationPercentage) -}}
{{- end -}}
{{- $cpuRequest := dig "requests" "cpu" "" (.Values.resources | default dict) | toString -}}
{{- if and .Values.autoscaling.targetCPUUtilizationPercentage (empty $cpuRequest) -}}
{{- fail (printf "autoscaling.targetCPUUtilizationPercentage (%v) needs resources.requests.cpu, which is empty: the scaler's CPU target is a percentage of that request. Set resources.requests.cpu, or set autoscaling.targetCPUUtilizationPercentage to null to turn the CPU signal off" .Values.autoscaling.targetCPUUtilizationPercentage) -}}
{{- end -}}
{{- if gt (int .Values.autoscaling.minReplicas) (int .Values.autoscaling.maxReplicas) -}}
{{- fail (printf "autoscaling.minReplicas (%v) is greater than autoscaling.maxReplicas (%v); the scaler refuses that configuration" .Values.autoscaling.minReplicas .Values.autoscaling.maxReplicas) -}}
{{- end -}}
{{- end -}}
{{- $mode -}}
{{- end }}

{{/*
"true" when the chart renders the scaler (autoscaling enabled in scaler mode).
*/}}
{{- define "live-edit-service.scalerEnabled" -}}
{{- if and .Values.autoscaling.enabled (eq (include "live-edit-service.autoscalingMode" .) "scaler") -}}
true
{{- end -}}
{{- end }}

{{/*
Name of the scaler's Deployment, Role and RoleBinding (and of its
ServiceAccount when the chart creates one). The fullname is cut to 56
characters before the suffix, so the result never truncates back onto the
server Deployment's own name.
*/}}
{{- define "live-edit-service.scalerFullname" -}}
{{- printf "%s-scaler" (include "live-edit-service.fullname" . | trunc 56 | trimSuffix "-") -}}
{{- end }}

{{/*
Scaler selector labels. Never a superset of a server selector: it carries
component=scaler and no app.kubernetes.io/name, so the server Deployment,
Service, PodDisruptionBudget and NetworkPolicy never select a scaler pod.
*/}}
{{- define "live-edit-service.scalerSelectorLabels" -}}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: scaler
{{- end }}

{{/*
Scaler object and pod labels: the common labels without app.kubernetes.io/name
(a server selector key, omitted after the merge as the smoke-test pod does),
plus component=scaler.
*/}}
{{- define "live-edit-service.scalerLabels" -}}
{{- $labels := mergeOverwrite (dict) (include "live-edit-service.labels" . | fromYaml) (include "live-edit-service.scalerSelectorLabels" . | fromYaml) -}}
{{- omit $labels "app.kubernetes.io/name" | toYaml -}}
{{- end }}

{{/*
ServiceAccount the scaler runs as, with the same semantics as the server's
(see live-edit-service.serviceAccountName): a supplied
autoscaling.scaler.serviceAccount.name is used as-is and the chart renders no
ServiceAccount; otherwise create=true makes one named after the scaler, and
create=false fails rather than run the scaler as the namespace's `default`
ServiceAccount (which the chart's RoleBinding would then have to grant).
*/}}
{{- define "live-edit-service.scalerServiceAccountName" -}}
{{- $sa := .Values.autoscaling.scaler.serviceAccount -}}
{{- if $sa.name -}}
{{- $sa.name -}}
{{- else if $sa.create -}}
{{- include "live-edit-service.scalerFullname" . -}}
{{- else -}}
{{- fail "autoscaling.scaler.serviceAccount: name is empty and create is false — the scaler would run as the namespace's shared `default` ServiceAccount. Set autoscaling.scaler.serviceAccount.name to a pre-provisioned ServiceAccount, or set autoscaling.scaler.serviceAccount.create=true." -}}
{{- end -}}
{{- end }}

{{/*
The pod memory checks. The service derives every memory budget from the
container's memory limit at startup, so:
- resources.limits.memory must be set and must not be zero. Kubernetes gives
  a container whose memory limit is absent or zero no memory limit at all:
  the kubelet hands the container runtime a memory limit only when it is
  non-zero, and the runtime reads 0 as "not specified". The pod would then
  run unbounded while the service, finding no limit in its cgroup, sizes
  every budget as for a pod at the chart's default 5Gi limit. Kubernetes does
  not refuse a zero limit itself: it accepts one beside a zero request (the
  0Mi request this chart would derive from it) and refuses it only beside a
  non-zero request, as a request above its limit;
- it must be a memory quantity this chart can read
  (live-edit-service.memoryQuantityBytes): the server's default memory request
  is derived from it (live-edit-service.serverResources);
- the four values that set those budgets before are gone from this chart.
  A removed value still set to its old default (what `helm upgrade
  --reuse-values` keeps from an older release's values) renders nothing; one
  set to anything else fails the render, naming it, because the service would
  silently ignore it otherwise.
Renders nothing; evaluated by configmap.yaml and by
live-edit-service.serverResources, so it runs on every render and before the
server's resources are rendered.
*/}}
{{- define "live-edit-service.podMemoryChecks" -}}
{{- $limits := (.Values.resources | default dict).limits | default dict -}}
{{- if not $limits.memory -}}
{{- fail "resources.limits.memory must be set: the service sizes every memory budget (uploads, catch-ups, the structural cache, the subscription ceiling) from the container's memory limit at startup. Set a limit (the chart default is 5Gi); a limit too small for the configuration stops the service at startup with the smallest limit it needs" -}}
{{- end -}}
{{- if eq (include "live-edit-service.memoryQuantityBytes" (dict "key" "resources.limits.memory" "value" $limits.memory)) "0" -}}
{{- fail (printf "resources.limits.memory is %#v, which is zero bytes: the service sizes every memory budget from the container's memory limit, and the default memory request is 70%% of it. Set a non-zero limit (the chart default is 5Gi)" $limits.memory) -}}
{{- end -}}
{{- $removed := list
  (dict "section" "valkey" "key" "subscriber_max_subscriptions" "old" "4096" "what" "the subscription ceiling")
  (dict "section" "valkey" "key" "catchup_build_memory_budget_mib" "old" "744" "what" "the catch-up memory budget")
  (dict "section" "layer" "key" "upload_memory_budget_mib" "old" "384" "what" "the upload memory budget")
  (dict "section" "layer" "key" "structural_cache_max_mib" "old" "256" "what" "the structural cache's memory budget") -}}
{{- range $removed -}}
{{- $section := index $.Values .section -}}
{{- if and (kindIs "map" $section) (hasKey $section .key) -}}
{{- $value := index $section .key -}}
{{- if not (or (kindIs "invalid" $value) (eq (toString $value) "") (eq (toString $value) .old)) -}}
{{- fail (printf "%s.%s is set to %v, but this chart no longer sets it: %s is derived from the container's memory limit (resources.limits.memory) at startup. Remove %s.%s from your values; to change %s, change resources.limits.memory" .section .key $value .what .section .key .what) -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- end }}

{{/*
A Kubernetes memory quantity as whole bytes, for the server container's memory
limit and request (live-edit-service.podMemoryChecks,
live-edit-service.serverResources). Called with (dict "key" <dotted values
path> "value" <the value>); returns plain decimal digits, rounded up to a whole
byte as Kubernetes rounds a quantity's value (away from zero). With "nanos"
true it returns the value in nanobytes instead, rounded up: the precision
Kubernetes keeps when it parses a quantity, so comparing two results matches
Kubernetes' own check that a request is not above its limit.

Accepted: every form Kubernetes' quantity grammar accepts (the apimachinery
resource.ParseQuantity, after the whitespace trim its JSON decoding applies):
an optional sign, a decimal number with or without a fraction, and at most one
suffix: Ki, Mi, Gi, Ti, Pi or Ei (powers of 1024), k, M, G, T, P or E (powers
of 1000), or a decimal exponent e<n> or E<n>. The exponent is the suffix, so
5e3Mi is not a quantity. Kubernetes narrows the exponent to 32 bits (it reads
5e4294967305 as 5e9); so does this helper. A number with no digit (".", "Ki")
reads as zero, or is refused, exactly where Kubernetes does either. A
values-file number (float64) and a --set integer are read through their
decimal text, which carries the same digits Helm renders into the manifest.

Refused, failing the render with the key and the value, rather than passed
into a manifest the pod would run with:
- what Kubernetes refuses: other spellings (5GB, 5gi, 5K), an exponent outside
  64 bits, a negative value, anything not a string or a number;
- a decimal exponent below -1000, after the 32-bit narrowing. This is the
  chart's own bound, not a copy of a Kubernetes refusal: Kubernetes' parser
  reads such a value by rescaling it by ten to the exponent's size, so its
  time grows with that size (apimachinery v0.31.4, measured: about 0.05 s at
  e-1000000, 1.7 s at e-10000000), and it does not finish near -2^31, where
  negating the exponent to a scale (Scale(exponent).infScale() in
  quantity.go) also overflows 32 bits, as does subtracting a fraction's
  length from it. The chart refuses rather than render a quantity the API
  server stalls on. -1000 costs no memory value: at that exponent a quantity
  is under one nanobyte unless it has more than 991 digits;
- the m, u and n suffixes (thousandths, millionths and billionths of a byte):
  Kubernetes accepts them, but on a pod memory value they are almost certainly
  a typo for Mi or M;
- a value above 8Pi (2^53 bytes), far beyond any pod.

The arithmetic is exact. The digits are scaled as a decimal string (multiplied
by 1024 per binary unit step, then shifted by the decimal exponent), never
through a float, and only digit strings this helper built, with no leading
zero, are converted to integers, by atoi. Never pass a user string to int64:
it reads a leading 0 as octal (010 is 8).
*/}}
{{- define "live-edit-service.memoryQuantityBytes" -}}
{{- $key := .key -}}
{{- $value := .value -}}
{{- $unreadable := printf "%s is %#v, which is not a Kubernetes memory quantity. Write a number of bytes, optionally with a decimal fraction, and at most one unit: Ki, Mi, Gi, Ti, Pi or Ei (powers of 1024), k, M, G, T, P or E (powers of 1000), or a decimal exponent such as e9; for example 5Gi" $key $value -}}
{{- $tooLarge := printf "%s is %#v, above 8Pi (9007199254740992 bytes), the largest memory quantity this chart reads: no pod has that much memory. Check the number and its unit" $key $value -}}
{{- $text := "" -}}
{{- if kindIs "string" $value -}}
{{- $text = trim $value -}}
{{- else if has (kindOf $value) (list "int" "int8" "int16" "int32" "int64" "uint" "uint8" "uint16" "uint32" "uint64" "float32" "float64") -}}
{{- $text = toString $value -}}
{{- end -}}
{{- $grammar := `^([+-]?)([0-9]*)(?:\.([0-9]*))?(Ki|Mi|Gi|Ti|Pi|Ei|[eE][+-]?[0-9]+|[numkMGTPE])?$` -}}
{{- if or (eq $text "") (not (regexMatch $grammar $text)) -}}
{{- fail $unreadable -}}
{{- end -}}
{{- $sign := regexReplaceAll $grammar $text "${1}" -}}
{{- $fraction := regexReplaceAll $grammar $text "${3}" -}}
{{- $digits := regexReplaceAll "^0+" (printf "%s%s" (regexReplaceAll $grammar $text "${2}") $fraction) "" -}}
{{- $suffix := regexReplaceAll $grammar $text "${4}" -}}
{{- $subByte := dict "m" "millibytes (thousandths of a byte)" "u" "microbytes (millionths of a byte)" "n" "nanobytes (billionths of a byte)" -}}
{{- if hasKey $subByte $suffix -}}
{{- fail (printf "%s is %#v: the %s suffix means %s, which is almost certainly a typo for a pod memory value. Write Mi for mebibytes or M for megabytes, for example 512Mi" $key $value $suffix (get $subByte $suffix)) -}}
{{- end -}}
{{- $exponent := 0 -}}
{{- $binarySteps := 0 -}}
{{- $decimalSuffixes := dict "" 0 "k" 3 "M" 6 "G" 9 "T" 12 "P" 15 "E" 18 -}}
{{- $binarySuffixes := dict "Ki" 1 "Mi" 2 "Gi" 3 "Ti" 4 "Pi" 5 "Ei" 6 -}}
{{- if hasKey $decimalSuffixes $suffix -}}
{{- $exponent = get $decimalSuffixes $suffix -}}
{{- else if hasKey $binarySuffixes $suffix -}}
{{- $binarySteps = get $binarySuffixes $suffix -}}
{{- else -}}
{{- /*
  A decimal exponent. Kubernetes reads it as a 64-bit integer and keeps its
  low 32 bits as a signed number; this reproduces that from the digits, so no
  value that does not fit is ever converted.
*/ -}}
{{- $exponentNegative := eq (substr 1 2 $suffix) "-" -}}
{{- $exponentDigits := regexReplaceAll "^[eE][+-]?0*" $suffix "" -}}
{{- if or (gt (len $exponentDigits) 19) (and (eq (len $exponentDigits) 19) (gt $exponentDigits "9223372036854775807")) -}}
{{- /* Beyond 64 bits Kubernetes refuses the suffix; -2^63 itself narrows to 0. */ -}}
{{- if not (and $exponentNegative (eq $exponentDigits "9223372036854775808")) -}}
{{- fail $unreadable -}}
{{- end -}}
{{- else if $exponentDigits -}}
{{- $narrowed := mod (atoi $exponentDigits) 4294967296 -}}
{{- if and $exponentNegative (gt $narrowed 0) -}}
{{- $narrowed = sub 4294967296 $narrowed -}}
{{- end -}}
{{- if ge $narrowed 2147483648 -}}
{{- $narrowed = sub $narrowed 4294967296 -}}
{{- end -}}
{{- /*
  The chart's lower bound on the exponent (see the comment above): Kubernetes'
  parser takes time growing with a negative exponent's size, and does not
  finish near -2^31.
*/ -}}
{{- if lt $narrowed -1000 -}}
{{- fail (printf "%s is %#v, whose decimal exponent (%d, as Kubernetes reads it) is below -1000, the lowest this chart accepts: Kubernetes' parser slows with the exponent's size and does not finish near -2147483648, and no memory value needs one. Write the number with a unit, for example 5Gi" $key $value $narrowed) -}}
{{- end -}}
{{- $exponent = $narrowed -}}
{{- end -}}
{{- end -}}
{{- /*
  A number with no digit at all reads as zero where Kubernetes' parser takes
  its integer path, and is refused where it does not: with Pi or Ei, or with
  an exponent below -9.
*/ -}}
{{- if and (not (regexMatch "[0-9]" (regexReplaceAll $grammar $text "${2}${3}"))) (or (ge $binarySteps 5) (lt $exponent -9)) -}}
{{- fail $unreadable -}}
{{- end -}}
{{- $nanos := "0" -}}
{{- if $digits -}}
{{- if eq $sign "-" -}}
{{- fail (printf "%s is %#v, a negative quantity: Kubernetes refuses a negative memory value" $key $value) -}}
{{- end -}}
{{- /*
  The value in nanobytes is $digits x 1024^$binarySteps x 10^$shift. The
  number is at least 10^(len($digits) - 1 + $shift), so a $shift that puts it
  at 10^25 or more is above 8Pi (about 9.007 x 10^24 nanobytes): refused here,
  before a string that long is built.
*/ -}}
{{- $shift := add (sub $exponent (len $fraction)) 9 -}}
{{- if ge (add (len $digits) $shift) 26 -}}
{{- fail $tooLarge -}}
{{- end -}}
{{- $product := $digits -}}
{{- range until (int $binarySteps) -}}
{{- $next := "" -}}
{{- $carry := 0 -}}
{{- range $digit := reverse (splitList "" $product) -}}
{{- $step := add (mul (atoi $digit) 1024) $carry -}}
{{- $next = printf "%d%s" (mod $step 10) $next -}}
{{- $carry = div $step 10 -}}
{{- end -}}
{{- if gt $carry 0 -}}
{{- $next = printf "%d%s" $carry $next -}}
{{- end -}}
{{- $product = $next -}}
{{- end -}}
{{- if ge $shift 0 -}}
{{- $nanos = printf "%s%s" $product (repeat (int $shift) "0") -}}
{{- else if ge (sub 0 $shift) (len $product) -}}
{{- /* Under one nanobyte: Kubernetes rounds a non-zero value up to 1n. */ -}}
{{- $nanos = "1" -}}
{{- else -}}
{{- /* Cut the digits below a nanobyte, rounding up as Kubernetes does. */ -}}
{{- $cut := int (add (len $product) $shift) -}}
{{- $nanos = substr 0 $cut $product -}}
{{- if regexMatch "[1-9]" (substr $cut (len $product) $product) -}}
{{- $nines := regexFind "9*$" $nanos -}}
{{- $head := trimSuffix $nines $nanos -}}
{{- if $head -}}
{{- $last := int (sub (len $head) 1) -}}
{{- $nanos = printf "%s%d%s" (substr 0 $last $head) (add1 (atoi (substr $last (len $head) $head))) (repeat (len $nines) "0") -}}
{{- else -}}
{{- $nanos = printf "1%s" (repeat (len $nines) "0") -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- if or (gt (len $nanos) 25) (and (eq (len $nanos) 25) (gt $nanos "9007199254740992000000000")) -}}
{{- fail $tooLarge -}}
{{- end -}}
{{- end -}}
{{- if .nanos -}}
{{- $nanos -}}
{{- else if gt (len $nanos) 9 -}}
{{- /* Whole bytes, rounded up: at most 16 digits, under 8Pi. */ -}}
{{- $cut := int (sub (len $nanos) 9) -}}
{{- $bytes := atoi (substr 0 $cut $nanos) -}}
{{- if regexMatch "[1-9]" (substr $cut (len $nanos) $nanos) -}}
{{- $bytes = add1 $bytes -}}
{{- end -}}
{{- $bytes -}}
{{- else if eq $nanos "0" -}}
0
{{- else -}}
1
{{- end -}}
{{- end }}

{{/*
The server container's resources, as YAML. Rendered as .Values.resources sets
them, except an unset resources.requests.memory (absent, null or empty): that
renders 70% of resources.limits.memory, rounded down to whole Mi (3584Mi at
5Gi, 1896Mi at 2709Mi), so the request follows the limit the service sizes
every memory budget from. Rounding down keeps the request at or below 70% of
the limit, readable, and the same on every render.

A set request is rendered exactly as written. One above the limit fails the
render, naming both, before Helm changes anything: Kubernetes would refuse the
Deployment. `helm upgrade --reuse-values` keeps the previous release's values,
so an older chart's memory request (1Gi, in 0.2.0-rc.84 and every later
release before the 70% default; from an earlier release `--reuse-values` fails
first on the renamed values, see
live-edit-service.renamedValueChecks) arrives as a set request and is kept:
the chart cannot tell it from one the deployer chose.

Works on a copy and never writes into .Values: the scaler
(scaler-deployment.yaml) and live-edit-service.autoscalingMode read
resources.requests.cpu from .Values as the deployer set it. Only the server
container's resources go through this helper.
*/}}
{{- define "live-edit-service.serverResources" -}}
{{- include "live-edit-service.podMemoryChecks" . -}}
{{- $resources := deepCopy (.Values.resources | default dict) -}}
{{- $limit := $resources.limits.memory -}}
{{- $limitBytes := atoi (include "live-edit-service.memoryQuantityBytes" (dict "key" "resources.limits.memory" "value" $limit)) -}}
{{- $derived := printf "%dMi" (div (mul $limitBytes 7) 10485760) -}}
{{- $requests := get $resources "requests" | default dict -}}
{{- if not (kindIs "map" $requests) -}}
{{- fail (printf "resources.requests must be a map of resource quantities such as cpu and memory (got %#v)" $requests) -}}
{{- end -}}
{{- $request := get $requests "memory" -}}
{{- if or (kindIs "invalid" $request) (eq (toString $request) "") -}}
{{- $_ := set $requests "memory" $derived -}}
{{- $_ := set $resources "requests" $requests -}}
{{- else -}}
{{- $requestNanos := include "live-edit-service.memoryQuantityBytes" (dict "key" "resources.requests.memory" "value" $request "nanos" true) -}}
{{- $limitNanos := include "live-edit-service.memoryQuantityBytes" (dict "key" "resources.limits.memory" "value" $limit "nanos" true) -}}
{{- if or (gt (len $requestNanos) (len $limitNanos)) (and (eq (len $requestNanos) (len $limitNanos)) (gt $requestNanos $limitNanos)) -}}
{{- fail (printf "resources.requests.memory (%#v) is above resources.limits.memory (%#v): Kubernetes refuses a request above the limit. Set the request at or below the limit, or remove it to get the chart default, 70%% of the limit (%s here)" $request $limit $derived) -}}
{{- end -}}
{{- end -}}
{{- toYaml $resources -}}
{{- end }}

{{/*
The renamed-value check. These values were renamed from snake_case to
camelCase; the chart reads only the new names, so an old name anywhere in the
values fails the render, never silently ignored. Every old name present is
listed in one message, so a values file is fixed in one pass.

An old name is refused even at its old default: `helm upgrade
--reuse-values` replaces this chart's defaults with the previous release's
chart values, so it brings every old name and none of the new ones, and the
render cannot run without them. `--reset-then-reuse-values` (Helm 3.14+) keeps
this chart's defaults and the previous release's own values only, so it fails
only on old names the deployer set. Renders nothing; evaluated first by the
two templates that read the renamed values, deployment.yaml (which Helm
renders before configmap.yaml) and configmap.yaml, so no check of a renamed
value reports before it.
*/}}
{{- define "live-edit-service.renamedValueChecks" -}}
{{- $renamed := list
  "authorization timeout_ms timeoutMs"
  "authorization cache_ttl_secs cacheTtlSecs"
  "authorization cache_negative_ttl_secs cacheNegativeTtlSecs"
  "authorization cache_max_entries cacheMaxEntries"
  "authorization max_attempts maxAttempts"
  "authorization retry_initial_delay_ms retryInitialDelayMs"
  "valkey pool_size poolSize"
  "valkey connect_timeout_secs connectTimeoutSecs"
  "valkey command_timeout_ms commandTimeoutMs"
  "valkey reconnect_max_delay_ms reconnectMaxDelayMs"
  "valkey subscriber_pool_size subscriberPoolSize"
  "valkey subscriber_max_concurrent_catchups subscriberMaxConcurrentCatchups"
  "valkey subscriber_admission_wait_ms subscriberAdmissionWaitMs"
  "valkey subscriber_admission_max_waiters subscriberAdmissionMaxWaiters"
  "valkey layer_stream_maxlen layerStreamMaxlen"
  "valkey layer_stream_max_bytes layerStreamMaxBytes"
  "valkey memory_bytes memoryBytes"
  "valkey ext_release_max_bytes extReleaseMaxBytes"
  "valkey maintenance_pool_size maintenancePoolSize"
  "valkey catchup_snapshot_read_concurrency catchupSnapshotReadConcurrency"
  "valkey catchup_fenced_read_concurrency catchupFencedReadConcurrency"
  "valkey snapshot_pool_size snapshotPoolSize"
  "valkey snapshot_atomic_read_max_items snapshotAtomicReadMaxItems"
  "valkey snapshot_atomic_read_max_bytes snapshotAtomicReadMaxBytes"
  "rpc deadline_ms deadlineMs"
  "rpc session_list_timeout_ms sessionListTimeoutMs"
  "layer max_size maxSize"
  "layer upload_admission_wait_ms uploadAdmissionWaitMs"
  "layer upload_admission_max_waiters uploadAdmissionMaxWaiters"
  "layer update_retry_max_attempts updateRetryMaxAttempts"
  "layer update_per_object_concurrency updatePerObjectConcurrency"
  "layer update_layer_backend updateLayerBackend"
  "layer catchup_snapshot_enabled catchupSnapshotEnabled"
  "layer structural_cache_audit_rate structuralCacheAuditRate"
  "layer structural_cache_audit_above_ceiling structuralCacheAuditAboveCeiling"
  "layer ext_value_verify extValueVerify"
  "session retry_max_attempts retryMaxAttempts"
  "session retry_initial_delay_ms retryInitialDelayMs"
  "presence broadcast_capacity broadcastCapacity"
  "presence sweeper_interval_secs sweeperIntervalSecs"
  "presence keepalive_interval_ms keepaliveIntervalMs"
  "shutdown timeout_secs timeoutSecs"
  "shutdown overall_budget_secs overallBudgetSecs"
  "shutdown stream_end_rate streamEndRate"
  "shutdown stream_end_max_window_secs streamEndMaxWindowSecs"
  "gc ext_reconcile_interval_secs extReconcileIntervalSecs" -}}
{{- $found := list -}}
{{- range $renamed -}}
{{- $name := splitList " " . -}}
{{- $section := index $.Values (index $name 0) -}}
{{- if and (kindIs "map" $section) (hasKey $section (index $name 1)) -}}
{{- $found = append $found (printf "%s.%s is now %s.%s" (index $name 0) (index $name 1) (index $name 0) (index $name 2)) -}}
{{- end -}}
{{- end -}}
{{- if $found -}}
{{- fail (printf "these values were renamed to camelCase, and this chart no longer reads the old names: %s. Rename them in your values files and `--set` flags. If this is a `helm upgrade --reuse-values`, which carries the previous chart's values with every old name, upgrade with your renamed values files instead, or with `--reset-then-reuse-values` (Helm 3.14+) when the release's own earlier values used no old name" (join "; " $found)) -}}
{{- end -}}
{{- end }}
