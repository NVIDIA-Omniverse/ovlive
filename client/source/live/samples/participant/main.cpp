// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Public C ABI plus the SDK-specific public USD adapter. No private runtime APIs.
#include <omni/usd/live/LiveUsd.h>
#include "DiagnosticLog.h"
#include <pxr/base/js/json.h>
#include <pxr/base/plug/registry.h>
#include <pxr/usd/sdf/attributeSpec.h>
#include <pxr/usd/sdf/changeBlock.h>
#include <pxr/usd/sdf/primSpec.h>
#include <pxr/usd/sdf/types.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <deque>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#if defined(_WIN32)
#    include <windows.h>
#    include <io.h>
#    include <share.h>
#    include <sys/stat.h>
#    include <climits>
#    include <cstdlib>
#else
#    include <unistd.h>
#endif

PXR_NAMESPACE_USING_DIRECTIVE
namespace hostApi = omni::usd::live::host;
namespace fs = std::filesystem;
namespace {
// The file primitives of the sample's own output files: a new file opened for writing only, never inherited.
#if defined(_WIN32)
int openNew(const fs::path& path) {
    int fd = -1;
    return _wsopen_s(&fd, path.c_str(), _O_WRONLY | _O_CREAT | _O_EXCL | _O_BINARY | _O_NOINHERIT, _SH_DENYNO,
               _S_IREAD | _S_IWRITE) == 0 ?
        fd :
        -1;
}
long long writeSome(int fd, const char* data, size_t size) {
    return _write(fd, data, static_cast<unsigned>(std::min<size_t>(size, INT_MAX)));
}
int syncFile(int fd) { return _commit(fd); }
int closeFile(int fd) { return _close(fd); }
// The CRT has no directory handle to flush; NTFS journals the new directory entry itself.
int syncDirectory(const fs::path&) { return 0; }
void unsetEnvironment(const char* name) { _putenv_s(name, ""); }
int standardOutput() { return _fileno(stdout); }
// The operator's retry event: CTRL_BREAK_EVENT, which a console control handler turns into an event the main thread
// waits on; the handler runs on its own thread, not in signal context.
HANDLE retryEvent = nullptr;
BOOL WINAPI onConsoleControl(DWORD type) {
    if (type != CTRL_BREAK_EVENT) return FALSE;
    SetEvent(retryEvent);
    return TRUE;
}
const char* const retrySignal = "CTRL_BREAK_EVENT";
bool prepareRetry() {
    // JSON Lines on stdout without newline translation.
    if (_setmode(_fileno(stdout), _O_BINARY) == -1) return false;
    retryEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    return retryEvent && SetConsoleCtrlHandler(onConsoleControl, TRUE);
}
void awaitRetry() { WaitForSingleObject(retryEvent, INFINITE); }
#else
int openNew(const fs::path& path) { return ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600); }
long long writeSome(int fd, const char* data, size_t size) { return ::write(fd, data, size); }
int syncFile(int fd) { return ::fsync(fd); }
int closeFile(int fd) { return ::close(fd); }
int syncDirectory(const fs::path& directory) {
    const int fd = ::open(directory.empty() ? "." : directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) return -1;
    const int sync = ::fsync(fd), closed = ::close(fd);
    return sync || closed ? -1 : 0;
}
void unsetEnvironment(const char* name) { ::unsetenv(name); }
int standardOutput() { return STDOUT_FILENO; }
// The operator's retry event: SIGUSR1, received with sigwait so no C++ runs in a signal handler.
sigset_t retrySignals;
const char* const retrySignal = "SIGUSR1";
bool prepareRetry() {
    sigemptyset(&retrySignals);
    sigaddset(&retrySignals, SIGUSR1);
    if (pthread_sigmask(SIG_BLOCK, &retrySignals, nullptr)) return false;
    std::signal(SIGPIPE, SIG_IGN);
    return true;
}
void awaitRetry() {
    int signal = 0;
    while (sigwait(&retrySignals, &signal)) {} // Explicit operator event; no success-path sleep.
}
#endif
constexpr uint32_t timeoutMs = 30000;
struct Failure {
    OmniUsdLiveResult result;
};
void check(OmniUsdLiveResult r) {
    if (r != OMNI_USD_LIVE_OK) throw Failure{r};
}
// A retired layer actor or request can hold its endpoint until the runtime
// scheduler releases it; the creation retirement wait does not cover that.
// Close returns BUSY meanwhile and keeps ownership for retry; any other result
// fails at once. The deadline is a failure-only watchdog.
template<class Close> void closeWhenReleased(Close close) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    OmniUsdLiveResult r;
    while ((r = close()) == OMNI_USD_LIVE_BUSY && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    check(r);
}
void require(bool b) {
    if (!b) throw Failure{OMNI_USD_LIVE_INVALID_ARGUMENT};
}
template<class T> T record() {
    T r{};
    r.struct_size = sizeof(r);
    r.abi_version = OMNI_USD_LIVE_ABI_VERSION;
    return r;
}
JsObject result(OmniUsdLiveResult r = OMNI_USD_LIVE_OK) { return {{"result", JsValue(int(r))}}; }
std::string json(const JsObject& value) {
    std::ostringstream stream;
    JsWriter writer(stream, JsWriter::Style::Compact);
    JsWriteValue(&writer, JsValue(value));
    return stream.str() + "\n";
}
std::string text(const JsObject& command, const char* key) {
    auto it = command.find(key);
    require(it != command.end() && it->second.IsString());
    auto value = it->second.GetString();
    require(value.find('\0') == std::string::npos);
    return value;
}
void writeAll(int fd, const std::string& bytes) {
    size_t offset = 0;
    while (offset < bytes.size()) {
        auto n = writeSome(fd, bytes.data() + offset, bytes.size() - offset);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) throw Failure{OMNI_USD_LIVE_IO_ERROR};
        offset += size_t(n);
    }
}
void writeNew(const fs::path& path, const std::string& bytes) {
    const int fd = openNew(path);
    if (fd < 0) throw Failure{OMNI_USD_LIVE_IO_ERROR};
    try {
        writeAll(fd, bytes);
        if (syncFile(fd)) throw Failure{OMNI_USD_LIVE_IO_ERROR};
    } catch (...) {
        closeFile(fd);
        throw;
    }
    if (closeFile(fd)) throw Failure{OMNI_USD_LIVE_IO_ERROR};
    if (syncDirectory(path.parent_path())) throw Failure{OMNI_USD_LIVE_IO_ERROR};
}
class EventLog {
    std::mutex mutex_;
    std::condition_variable changed_, done_;
    std::deque<std::string> queue_;
    size_t pending_ = 0, records_ = 0, written_ = 0;
    uint64_t dropped_ = 0;
    bool stopping_ = false, finished_ = false, error_ = false;
    int fd_;
    std::thread worker_;
    const size_t maxRecords_, maxPending_, maxFile_;
public:
    using Writer = std::function<void(int, const std::string&)>;
    explicit EventLog(const fs::path& path, Writer write = writeAll, size_t maxRecords = 1024,
        size_t maxPending = 1024 * 1024, size_t maxFile = 8 * 1024 * 1024)
        : fd_(openNew(path)), maxRecords_(maxRecords), maxPending_(maxPending), maxFile_(maxFile) {
        if (fd_ < 0) throw Failure{OMNI_USD_LIVE_IO_ERROR};
        try {
            worker_ = std::thread([this, write = std::move(write)] {
                for (;;) {
                    std::string line;
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        changed_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
                        if (queue_.empty()) break;
                        line = std::move(queue_.front());
                        queue_.pop_front();
                    }
                    bool failed = false;
                    try {
                        write(fd_, line);
                    } catch (...) { failed = true; }
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        pending_ -= line.size();
                        --records_;
                        if (failed) {
                            error_ = true;
                            dropped_ += 1 + queue_.size();
                            queue_.clear();
                            pending_ = records_ = 0;
                            break;
                        }
                        written_ += line.size();
                    }
                }
                const bool failed = closeFile(fd_) != 0;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    error_ |= failed;
                    finished_ = true;
                }
                done_.notify_all();
            });
        } catch (...) {
            closeFile(fd_);
            throw;
        }
    }
    void push(std::string line) noexcept {
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_ || error_ || records_ >= maxRecords_ || line.size() > maxPending_ - pending_ ||
                line.size() > maxFile_ - written_ - pending_) {
                ++dropped_;
                return;
            }
            const auto bytes = line.size();
            queue_.push_back(std::move(line));
            pending_ += bytes;
            ++records_;
            changed_.notify_one();
        } catch (...) { droppedFailure(); }
    }
    void droppedFailure() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        ++dropped_;
    }
    JsObject close() {
        std::unique_lock<std::mutex> lock(mutex_);
        stopping_ = true;
        changed_.notify_all();
        if (!done_.wait_for(lock, std::chrono::milliseconds(timeoutMs), [this] { return finished_; }))
            throw Failure{OMNI_USD_LIVE_TIMEOUT};
        auto summary = JsObject{{"written_bytes", JsValue(uint64_t(written_))}, {"dropped", JsValue(dropped_)},
            {"error", error_ ? JsValue("IOError") : JsValue()}};
        lock.unlock();
        if (worker_.joinable()) worker_.join();
        return summary;
    }
    ~EventLog() {
        if (worker_.joinable()) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stopping_ = true;
            }
            changed_.notify_all();
            worker_.join();
        }
    }
};
class Host {
    std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<std::function<void()>> jobs_;
    bool stop_ = false;
    std::thread worker_;
public:
    Host()
        : worker_([this] {
              for (;;) {
                  std::function<void()> job;
                  {
                      std::unique_lock<std::mutex> lock(mutex_);
                      changed_.wait(lock, [this] { return stop_ || !jobs_.empty(); });
                      if (jobs_.empty()) break;
                      job = std::move(jobs_.front());
                      jobs_.pop_front();
                  }
                  job();
                  job = {}; // Retire USD captures on this host before the next job.
              }
          }) {}
    bool current() const { return std::this_thread::get_id() == worker_.get_id(); }
    void dispatch(std::function<void()> job) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stop_ || jobs_.size() >= 128) throw Failure{OMNI_USD_LIVE_RESOURCE_EXHAUSTED};
        jobs_.push_back(std::move(job));
        changed_.notify_one();
    }
    template<class F> auto call(F action) -> decltype(action()) {
        auto task = std::make_shared<std::packaged_task<decltype(action())()>>(std::move(action));
        auto future = task->get_future();
        dispatch([task] { (*task)(); });
        if (future.wait_for(std::chrono::seconds(35)) != std::future_status::ready)
            throw Failure{OMNI_USD_LIVE_TIMEOUT};
        return future.get();
    }
    ~Host() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        changed_.notify_all();
        worker_.join();
    }
};
JsObject statusJson(const OmniUsdLiveStatus& s) {
    return {{"state", JsValue(int(s.state))}, {"result", JsValue(int(s.result))}, {"subject_id", JsValue(s.subject_id)},
        {"revision", JsValue(s.revision)}, {"recovery_action", JsValue(int(s.recovery_action))},
        {"pending_write_count", JsValue(s.pending_write_count)}, {"pending_bytes", JsValue(s.pending_bytes)},
        {"incident_id", JsValue(s.incident_id)}, {"buffered_edit_count", JsValue(s.buffered_edit_count)},
        {"read_sequence", JsValue(s.read_sequence)}, {"applied_sequence", JsValue(s.applied_sequence)}};
}
struct Participant {
    // This object survives every failed command/cleanup attempt. Host-owned USD
    // members are explicitly retired on that host before its destruction.
    fs::path output;
    EventLog log;
    Host host;
    std::unique_ptr<hostApi::PublicationContext> context;
    std::unique_ptr<hostApi::Creation> creation;
    SdfLayerRefPtr layer, seed; // The seed is exported while the creation has no active layer.
    OmniUsdLiveEndpoint endpoint = 0;
    OmniUsdLiveSession joined = 0;
    OmniUsdLiveRequest sessionRequest = 0, listRequest = 0;
    OmniUsdLiveCallbackHandle statuses = 0, diagnostics = 0, auth = 0;
    bool initialized = false, sessionAttempted = false, sessionCopied = false, sessionCopyFailed = false;
    bool dirty = false, exportRequired = false, exported = false, closed = false;
    OmniUsdLiveResult sessionResult = OMNI_USD_LIVE_BUSY, listResult = OMNI_USD_LIVE_BUSY;
    std::string sessionId, sessionUri, uri;
    JsArray metadata;
    std::mutex statusMutex;
    std::condition_variable statusChanged;
    OmniUsdLiveStatus lastStatus{};
    JsObject logging;
    explicit Participant(fs::path directory, EventLog::Writer write = writeAll, size_t maxLogRecords = 1024)
        : output(std::move(directory)), log(output / "events.jsonl", std::move(write), maxLogRecords) {}
    void event(JsObject value) noexcept {
        try {
            log.push(json(value));
        } catch (...) { log.droppedFailure(); }
    }
    static void OMNI_USD_LIVE_CALL status(void* raw, const OmniUsdLiveStatusEvent* e) noexcept {
        auto& p = *static_cast<Participant*>(raw);
        if (!e || !e->status) return;
        {
            std::lock_guard<std::mutex> lock(p.statusMutex);
            if (e->status->subject_kind == OMNI_USD_LIVE_SUBJECT_LAYER &&
                (p.lastStatus.subject_id != e->status->subject_id || p.lastStatus.revision < e->status->revision))
                p.lastStatus = *e->status;
        }
        p.statusChanged.notify_all();
        try {
            auto value = statusJson(*e->status);
            value["type"] = JsValue("status");
            p.event(std::move(value));
        } catch (...) { p.log.droppedFailure(); }
    }
    static void OMNI_USD_LIVE_CALL diagnostic(void* raw, const OmniUsdLiveDiagnosticRecord* e) noexcept {
        auto& p = *static_cast<Participant*>(raw);
        std::string line;
        if (e &&
            omni::usd::live::sample::formatDiagnosticRecord(*e, omni::usd::live::sample::LogFormat::JsonLines, line) ==
                OMNI_USD_LIVE_OK) {
            // Formatter output is already bounded JSON; wrap without interpreting borrowed fields.
            try {
                p.log.push("{\"type\":\"diagnostic\",\"record\":" + line.substr(0, line.size() - 1) + "}\n");
            } catch (...) { p.log.droppedFailure(); }
        } else p.log.droppedFailure();
    }
    static void OMNI_USD_LIVE_CALL challenge(void* raw, const OmniUsdLiveAuthRequired* e) noexcept {
        auto& p = *static_cast<Participant*>(raw);
        try {
            if (e)
                p.event({{"type", JsValue("auth")}, {"reason", JsValue(int(e->reason))},
                    {"challenge", JsValue(e->challenge_id)}});
        } catch (...) { p.log.droppedFailure(); }
    }
    static void OMNI_USD_LIVE_CALL sessionEvent(void* raw, const OmniUsdLiveSessionEvent* e) noexcept {
        auto& p = *static_cast<Participant*>(raw);
        try {
            if (e)
                p.event({{"type", JsValue("session")}, {"kind", JsValue(int(e->kind))},
                    {"result", JsValue(int(e->result))}});
        } catch (...) { p.log.droppedFailure(); }
    }
    static void OMNI_USD_LIVE_CALL created(void* raw, OmniUsdLiveResult r,
        const OmniUsdLiveCreatedSession* value) noexcept {
        auto& p = *static_cast<Participant*>(raw);
        p.sessionResult = r;
        try {
            if (value) {
                p.sessionId = value->session_id;
                p.sessionUri = value->uri;
                p.sessionCopied = true;
            }
        } catch (...) { p.sessionCopyFailed = true; } // Never convert lost identity into permission to create again.
    }
    static void OMNI_USD_LIVE_CALL listed(void* raw, OmniUsdLiveResult r, uint64_t count,
        const OmniUsdLiveSessionInfo* const* values) noexcept {
        auto& p = *static_cast<Participant*>(raw);
        p.listResult = r;
        try {
            if (r != OMNI_USD_LIVE_OK) return;
            if (count > 1024) throw Failure{OMNI_USD_LIVE_RESOURCE_EXHAUSTED};
            for (uint64_t i = 0; i < count; ++i) {
                const auto& v = *values[i];
                p.metadata.push_back(JsValue(
                    JsObject{{"session_id", JsValue(v.session_id)}, {"uri", JsValue(v.uri)}, {"name", JsValue(v.name)},
                        {"created_by", JsValue(v.created_by)}, {"created_time_ms", JsValue(v.created_time_ms)}}));
            }
        } catch (...) { p.listResult = OMNI_USD_LIVE_RESOURCE_EXHAUSTED; }
    }
    void initialize(const std::string& authority, const std::string& token, const std::string& caPem,
        const std::string& tlsServerName, uint32_t transportSecurity) {
        auto options = record<OmniUsdLiveRuntimeOptions>();
        options.request_timeout_ms = timeoutMs;
        check(omniUsdLiveInitialize(&options));
        initialized = true;
        auto ep = record<OmniUsdLiveEndpointOptions>();
        ep.authority = authority.c_str();
        ep.access_token = token.c_str();
        ep.ca_pem = caPem.c_str();
        ep.tls_server_name = tlsServerName.c_str();
        ep.transport_security = transportSecurity;
        check(omniUsdLiveRegisterEndpoint(&ep, &endpoint));
        check(omniUsdLiveRegisterStatusCallback(endpoint, this, status, &statuses));
        check(omniUsdLiveRegisterAuthRequiredCallback(endpoint, this, challenge, &auth));
        auto logs = record<OmniUsdLiveDiagnosticOptions>();
        logs.min_severity = OMNI_USD_LIVE_SEVERITY_INFO;
        check(omniUsdLiveRegisterDiagnosticCallback(&logs, this, diagnostic, &diagnostics));
        hostApi::PublicationContext* raw = nullptr;
        check(hostApi::createPublicationContext([this](auto job) { host.dispatch(std::move(job)); },
            [this] { return host.current(); }, &raw));
        context.reset(raw);
    }
    JsObject sessionOutcome() {
        require(sessionRequest);
        check(omniUsdLiveWait(sessionRequest, timeoutMs));
        auto error = record<OmniUsdLiveRequestError>();
        check(omniUsdLiveGetRequestError(sessionRequest, &error));
        OmniUsdLiveResult actual;
        check(omniUsdLiveGetRequestStatus(sessionRequest, &actual));
        auto reply = result(actual);
        reply["session"] = sessionCopied ?
            JsValue(JsObject{{"session_id", JsValue(sessionId)}, {"uri", JsValue(sessionUri)}}) :
            JsValue();
        reply["copy_failed"] = JsValue(sessionCopyFailed);
        reply["request_id"] = JsValue(error.request_id);
        reply["incident_id"] = JsValue(error.incident_id);
        reply["error"] = JsValue(JsObject{{"result", JsValue(int(error.result))},
            {"operation", JsValue(int(error.operation))}, {"attempt_id", JsValue(error.attempt_id)},
            {"grpc_status_present", JsValue(bool(error.grpc_status_present))},
            {"grpc_status", JsValue(int(error.grpc_status))}});
        auto logged = reply;
        logged["type"] = JsValue("session_outcome");
        event(std::move(logged));
        return reply;
    }
    JsObject resume() {
        require(bool(creation));
        if (layer) return result();
        check(creation->wait(timeoutMs));
        hostApi::CreationOutcome outcome;
        check(creation->getOutcome(&outcome));
        if (outcome.result != OMNI_USD_LIVE_OK) {
            auto r = result(outcome.result);
            r["export_required"] = JsValue(true);
            return r;
        }
        check(host.call([this] { return creation->prepareLayer(timeoutMs); }));
        check(host.call([this] { return creation->activate(); }));
        host.call([this] {
            check(creation->getLayer(&layer));
            exportRequired = false;
        });
        return result();
    }
    JsObject create(const std::string& base, const std::string& name) {
        require(!sessionAttempted);
        sessionAttempted = true;
        check(omniUsdLiveCreateSession(endpoint, base.c_str(), name.c_str(), this, created, &sessionRequest));
        auto outcome = sessionOutcome();
        check(OmniUsdLiveResult(outcome.at("result").GetInt()));
        require(sessionCopied && !sessionCopyFailed);
        check(omniUsdLiveListSessions(endpoint, base.c_str(), this, listed, &listRequest));
        check(omniUsdLiveWait(listRequest, timeoutMs));
        check(listResult);
        auto join = record<OmniUsdLiveJoinOptions>();
        join.app_name = "live-native-sample";
        check(omniUsdLiveJoinSession(sessionUri.c_str(), &join, this, sessionEvent, &joined));
        uint64_t size = 0;
        check(omniUsdLiveMakeLayerUri(sessionUri.c_str(), "sample.live", nullptr, &size));
        uri.resize(size);
        check(omniUsdLiveMakeLayerUri(sessionUri.c_str(), "sample.live", uri.data(), &size));
        uri.pop_back();
        host.call([this] {
            auto local = SdfLayer::CreateAnonymous("sample.usda");
            require(bool(local));
            auto prim = SdfPrimSpec::New(local, "Example", SdfSpecifierDef, "Xform");
            require(bool(prim));
            auto attribute = SdfAttributeSpec::New(prim, "value", SdfValueTypeNames->Int);
            require(bool(attribute));
            require(attribute->SetDefaultValue(VtValue(0)));
            hostApi::Creation* raw = nullptr;
            check(context->prepareCreation(*local, uri.c_str(), &raw));
            creation.reset(raw);
            seed = local; // Held with the creation, so close retires both on this host.
            exportRequired = true;
            check(creation->start());
        });
        auto r = resume();
        r["session"] = outcome.at("session");
        r["metadata"] = JsValue(metadata);
        return r;
    }
    OmniUsdLiveStatus layerStatus() {
        auto s = record<OmniUsdLiveStatus>();
        check(omniUsdLiveGetLayerStatus(uri.c_str(), &s));
        return s;
    }
    // The typed cause of a failed layer; INTERNAL_ERROR when no layer reports one.
    OmniUsdLiveResult failedLayerResult() {
        auto s = record<OmniUsdLiveStatus>();
        if (!uri.empty() && omniUsdLiveGetLayerStatus(uri.c_str(), &s) == OMNI_USD_LIVE_OK &&
            s.state == OMNI_USD_LIVE_STATE_FAILED && s.result != OMNI_USD_LIVE_OK)
            return s.result;
        return OMNI_USD_LIVE_INTERNAL_ERROR;
    }
    // Revision of the newest layer status the callback delivered, 0 before any.
    uint64_t statusRevision(uint64_t subject) {
        std::lock_guard<std::mutex> lock(statusMutex);
        return lastStatus.subject_id == subject ? lastStatus.revision : 0;
    }
    // The BUSY save saw buffered host edits, so each layer status published
    // after their record reports them as buffered until a flush empties the
    // buffer, which publishes a newer revision even when it issues nothing. A
    // status newer than `seen` with none buffered outside BLOCKED, or a
    // terminal status, therefore follows their admission or the layer's failure.
    void waitAdmitted(uint64_t subject, uint64_t seen, std::chrono::steady_clock::time_point deadline) {
        std::unique_lock<std::mutex> lock(statusMutex);
        if (!statusChanged.wait_until(lock, deadline, [this, subject, seen] {
            const auto& s = lastStatus;
            return s.subject_id == subject && s.revision > seen &&
                (s.state == OMNI_USD_LIVE_STATE_FAILED || s.state == OMNI_USD_LIVE_STATE_CLOSING ||
                    s.state == OMNI_USD_LIVE_STATE_CLOSED ||
                    (s.state != OMNI_USD_LIVE_STATE_BLOCKED && s.buffered_edit_count == 0));
        }))
            throw Failure{OMNI_USD_LIVE_TIMEOUT};
    }
    JsObject save() {
        if (!layer || exportRequired) return result(failedLayerResult());
        // An off-host save returns BUSY while host edits are buffered and
        // schedules the flush that admits them on the host. The status
        // callback reports that admission; then the save retries with the
        // time left. Save off the host: an on-host save flushes inline and
        // only polls, so a bounded blocking drain needs a caller that
        // progresses independently of the host. The deadline is a
        // failure-only watchdog.
        hostApi::CreationOutcome owner;
        check(creation->getOutcome(&owner));
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
        OmniUsdLiveResult r;
        for (;;) {
            const auto seen = statusRevision(owner.subjectId);
            const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
                deadline - std::chrono::steady_clock::now())
                                  .count();
            r = hostApi::save(*layer, uint32_t(std::max<decltype(left)>(0, left)));
            if (r != OMNI_USD_LIVE_BUSY) break;
            waitAdmitted(owner.subjectId, seen, deadline);
        }
        const auto s = layerStatus();
        if (r == OMNI_USD_LIVE_OK) dirty = false;
        if (s.state == OMNI_USD_LIVE_STATE_FAILED) exportRequired = true;
        auto reply = result(r);
        reply["status"] = JsValue(statusJson(s));
        return reply;
    }
    JsObject exportLayer(const std::string& path) {
        require(bool(creation));
        // SdfLayer::Export of the local content (design section 13): the live layer
        // once activated, else the caller's own seed. An existing destination
        // is refused so a retry never replaces an earlier export. The refusal
        // is best-effort: a file created between the check and the export is
        // overwritten.
        if (fs::exists(path)) return result(OMNI_USD_LIVE_ALREADY_EXISTS);
        const bool written = host.call([&] { return (layer ? layer : seed)->Export(path); });
        if (!written) return result(OMNI_USD_LIVE_IO_ERROR);
        exported = true;
        return result();
    }
    JsObject close() {
        if (closed) {
            auto r = result();
            r["closed"] = JsValue(true);
            r["logging"] = JsValue(logging);
            return r;
        }
        host.call([] {}); // A timed-out host job must finish before inspecting its ownership.
        if (sessionRequest) {
            sessionOutcome();
            // Identity copy failure after a successful mutation requires operator intervention.
            if (sessionResult == OMNI_USD_LIVE_OK && (!sessionCopied || sessionCopyFailed))
                return result(OMNI_USD_LIVE_INTERNAL_ERROR);
        }
        if (creation) {
            auto s = layerStatus();
            exportRequired |= s.state == OMNI_USD_LIVE_STATE_FAILED;
        }
        if (exportRequired && !exported) return result(failedLayerResult());
        if (dirty && !exported) {
            auto r = save();
            if (r.at("result").GetInt() != 0) return r;
        }
        if (creation) {
            auto r = host.call([this] { return creation->close(); });
            if (r == OMNI_USD_LIVE_BUSY) {
                check(creation->waitForClose(timeoutMs));
                r = host.call([this] { return creation->close(); });
            }
            check(r);
            host.call([this] {
                layer.Reset();
                seed.Reset();
                creation.reset();
            });
        }
        if (joined) {
            check(omniUsdLiveLeaveSession(joined, timeoutMs));
            joined = 0;
        }
        for (auto* request : {&listRequest, &sessionRequest})
            if (*request) {
                check(omniUsdLiveWait(*request, timeoutMs));
                check(omniUsdLiveReleaseRequest(*request));
                *request = 0;
            }
        for (auto* observer : {&statuses, &auth, &diagnostics})
            if (*observer) {
                check(omniUsdLiveUnregisterCallback(*observer, timeoutMs));
                *observer = 0;
            }
        logging = log.close();
        if (context) {
            check(context->close());
            context.reset();
        }
        if (endpoint) {
            closeWhenReleased([this] { return omniUsdLiveUnregisterEndpoint(endpoint, timeoutMs); });
            endpoint = 0;
        }
        if (initialized) {
            closeWhenReleased([] { return omniUsdLiveShutdown(timeoutMs); });
            initialized = false;
        }
        closed = true;
        return close();
    }
    JsObject execute(const JsObject& command) {
        host.call([] {});
        const auto op = text(command, "operation");
        if (op == "create")
            return create(text(command, "base_layer_key"),
                command.count("name") ? text(command, "name") : "Headless sample");
        if (op == "session-outcome") return sessionOutcome();
        if (op == "resume") return resume();
        if (op == "close") return close();
        if (op == "save") return save();
        if (op == "export") return exportLayer(text(command, "path"));
        if (op == "edit") {
            require(layer && !exportRequired);
            const auto& v = command.at("value");
            require(v.IsInt());
            auto value = v.GetInt64();
            require(value >= INT32_MIN && value <= INT32_MAX);
            host.call([this, value] {
                dirty = true;
                exported = false;
                SdfChangeBlock block;
                auto attribute = layer->GetAttributeAtPath(SdfPath("/Example.value"));
                require(bool(attribute));
                require(attribute->SetDefaultValue(VtValue(int(value))));
            });
            return result();
        }
        if (op == "inspect") {
            require(bool(layer));
            const auto value = host.call([this] {
                const auto attribute = layer->GetAttributeAtPath(SdfPath("/Example.value"));
                require(bool(attribute));
                const auto current = attribute->GetDefaultValue();
                require(current.IsHolding<int>());
                return current.UncheckedGet<int>();
            });
            auto r = result();
            r["value"] = JsValue(value);
            return r;
        }
        if (op == "status") {
            auto r = result();
            r["status"] = JsValue(statusJson(layerStatus()));
            r["exported"] = JsValue(exported);
            return r;
        }
        if (op == "wait-status") {
            require(bool(creation));
            const auto& requested = command.at("state");
            require(requested.IsInt());
            // State 6 is reserved (design section 16); no status publishes it.
            auto state = requested.GetInt64();
            require(state >= 1 && state <= 9 && state != 6);
            hostApi::CreationOutcome owner;
            check(creation->getOutcome(&owner));
            std::unique_lock<std::mutex> lock(statusMutex);
            if (!statusChanged.wait_for(lock, std::chrono::milliseconds(timeoutMs), [this, state, owner] {
                return lastStatus.subject_id == owner.subjectId && lastStatus.revision &&
                    lastStatus.state == uint32_t(state);
            }))
                throw Failure{OMNI_USD_LIVE_TIMEOUT};
            auto r = result();
            r["status"] = JsValue(statusJson(lastStatus));
            return r;
        }
        if (op == "refresh") {
            std::ifstream input(text(command, "token_file"));
            std::string token;
            for (char c; input.get(c);) {
                if (token.size() >= 65536) throw Failure{OMNI_USD_LIVE_RESOURCE_EXHAUSTED};
                token += c;
            }
            require(input.eof());
            while (!token.empty() && (token.back() == '\n' || token.back() == '\r')) token.pop_back();
            require(!token.empty() && token.find('\0') == std::string::npos);
            auto state = record<OmniUsdLiveAuthState>();
            check(omniUsdLiveGetAuthState(endpoint, &state));
            uint64_t generation = 0;
            auto r = result(omniUsdLiveSetAccessToken(endpoint, state.generation, token.c_str(), 0, &generation));
            r["generation"] = JsValue(generation);
            return r;
        }
        if (op == "snapshot") {
            auto name = command.count("name") ? text(command, "name") : "diagnostics.json";
            require(fs::path(name).filename() == name && name != "." && name != "..");
            auto options = record<OmniUsdLiveDiagnosticExportOptions>();
            options.max_output_bytes = 1024 * 1024;
            uint64_t size = 0;
            check(omniUsdLiveCopyDiagnostics(endpoint, &options, nullptr, &size));
            for (unsigned attempt = 0; attempt < 3; ++attempt) {
                std::string bytes(size, '\0');
                auto r = omniUsdLiveCopyDiagnostics(endpoint, &options, bytes.data(), &size);
                if (r == OMNI_USD_LIVE_BUFFER_TOO_SMALL) continue;
                check(r);
                require(size && size <= bytes.size());
                bytes.resize(size - 1);
                writeNew(output / name, bytes);
                return result();
            }
            return result(OMNI_USD_LIVE_BUSY);
        }
        throw Failure{OMNI_USD_LIVE_INVALID_ARGUMENT};
    }
    JsObject preserve() {
        host.call([] {});
        auto r = close();
        if (!closed && creation) {
            if (!layer) {
                try {
                    resume();
                } catch (const Failure&) {}
                r = close();
            }
            if (!closed) {
                check(OmniUsdLiveResult(exportLayer((output / "eof-export.usda").string()).at("result").GetInt()));
                r = close();
            }
        }
        return r;
    }
};
bool reply(JsObject value, const char* type = "reply") noexcept {
    try {
        value["type"] = JsValue(type);
        writeAll(standardOutput(), json(value));
        return true;
    } catch (...) { return false; }
}
} // namespace
int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout
            << "live_participant_sample --authority HOST:PORT --output NEW_DIR --plug-info PATH [--ca-bundle PEM_FILE] [--tls-server-name DNS_NAME] [--transport-security default|tls|plaintext]\n"
               "Token: LIVE_SAMPLE_ACCESS_TOKEN. Commands: JSON Lines on stdin. See docs/ovlive-samples.md.\n"
            << std::flush; // Replies bypass stdio; usage is the one buffered write and never waits for the exit flush.
        return std::cout ? 0 : 2;
    }
    if (!prepareRetry()) return 2;
    std::unique_ptr<Participant> participant;
    int exitCode = 0;
    try {
        std::string authority, output, plugin, caBundle, tlsServerName, transport = "default";
        bool transportSpecified = false;
        for (int i = 1; i < argc; i += 2) {
            require(i + 1 < argc);
            std::string key(argv[i]);
            if (key == "--authority") {
                require(authority.empty());
                authority = argv[i + 1];
            } else if (key == "--output") {
                require(output.empty());
                output = argv[i + 1];
            } else if (key == "--plug-info") {
                require(plugin.empty());
                plugin = argv[i + 1];
            } else if (key == "--ca-bundle") {
                require(caBundle.empty());
                caBundle = argv[i + 1];
            } else if (key == "--tls-server-name") {
                require(tlsServerName.empty());
                tlsServerName = argv[i + 1];
            } else if (key == "--transport-security") {
                require(!transportSpecified);
                transportSpecified = true;
                transport = argv[i + 1];
            } else require(false);
        }
        const char* supplied = std::getenv("LIVE_SAMPLE_ACCESS_TOKEN");
        require(supplied && *supplied);
        std::string token(supplied);
        unsetEnvironment("LIVE_SAMPLE_ACCESS_TOKEN");
        require(!authority.empty() && !output.empty() && !plugin.empty());
        require(transport == "default" || transport == "tls" || transport == "plaintext");
        std::string caPem;
        if (!caBundle.empty()) {
            std::ifstream input(caBundle, std::ios::binary);
            require(bool(input));
            caPem.resize(1024 * 1024 + 1);
            input.read(caPem.data(), caPem.size());
            caPem.resize(input.gcount());
            require(!input.bad());
            require(caPem.size() <= 1024 * 1024);
        }
        PlugRegistry::GetInstance().RegisterPlugins(plugin);
        auto directory = fs::absolute(output);
        require(fs::create_directory(directory));
        participant = std::make_unique<Participant>(directory);
        participant->initialize(authority, token, caPem, tlsServerName,
            transport == "tls"           ? OMNI_USD_LIVE_TRANSPORT_TLS :
                transport == "plaintext" ? OMNI_USD_LIVE_TRANSPORT_PLAINTEXT :
                                           OMNI_USD_LIVE_TRANSPORT_DEFAULT);
        token.clear();
        bool connected = reply({}, "ready");
        while (connected) {
            std::string line;
            bool oversized = false;
            for (char c; std::cin.get(c) && c != '\n';) {
                if (line.size() < 65536) line += c;
                else oversized = true;
            }
            if (line.empty() && !std::cin) break;
            JsObject response;
            try {
                require(!oversized);
                auto value = JsParseString(line);
                require(value.IsObject());
                response = participant->execute(value.GetJsObject());
            } catch (const Failure& e) { response = result(e.result); } catch (...) {
                response = result(OMNI_USD_LIVE_INTERNAL_ERROR);
            }
            connected = reply(std::move(response));
            if (participant->closed) return 0;
        }
    } catch (const Failure& e) {
        reply(result(e.result));
        if (!participant) return 2;
        exitCode = 2;
    } catch (...) {
        reply(result(OMNI_USD_LIVE_INTERNAL_ERROR));
        if (!participant) return 2;
        exitCode = 2;
    }
    // No exception path destroys admitted work. On preservation failure retain
    // every owner and the independently progressing host until operator retry.
    for (;;) {
        try {
            auto r = participant->preserve();
            if (participant->closed) {
                reply(std::move(r));
                return exitCode;
            }
        } catch (...) {}
        reply({{"retry_signal", JsValue(retrySignal)}, {"output", JsValue(participant->output.string())}}, "retained");
        awaitRetry();
    }
}
