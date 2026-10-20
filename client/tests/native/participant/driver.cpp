// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Compile the shipped implementation with an injected writer. No alternate logger.
#define main participantMain
#include "../../../source/live/samples/participant/main.cpp"
#undef main
#include <cassert>
int main() {
#if defined(_WIN32)
    // create_directory reports whether it created the directory, so a taken name moves to the next one.
    std::string directory;
    for(unsigned attempt=0; directory.empty(); ++attempt) {
        const auto candidate=fs::temp_directory_path()/
            ("live-participant-tests-"+std::to_string(GetCurrentProcessId())+"-"+std::to_string(attempt));
        if(fs::create_directory(candidate)) directory=candidate.string();
    }
#else
    std::string directory=(fs::temp_directory_path()/"live-participant-tests-XXXXXX").string();
    assert(mkdtemp(directory.data()));
#endif
    fs::path root(directory);
    try {
        std::mutex mutex; std::condition_variable changed;
        bool entered=false,release=false;
        auto write=[&](int fd,const std::string& bytes) {
            { std::unique_lock<std::mutex> lock(mutex); entered=true; changed.notify_all();
              if(!changed.wait_for(lock,std::chrono::seconds(5),[&]{return release;}))
                  throw std::runtime_error("writer release watchdog"); }
            writeAll(fd,bytes);
        };
        fs::create_directory(root/"blocked");
        Participant p(root/"blocked",write,1);
        p.log.push("first\n");
        {std::unique_lock<std::mutex> lock(mutex); assert(changed.wait_for(lock,std::chrono::seconds(5),[&]{return entered;}));}
        auto s=record<OmniUsdLiveStatus>(); s.subject_kind=OMNI_USD_LIVE_SUBJECT_LAYER;
        s.subject_id=123; s.state=OMNI_USD_LIVE_STATE_READY;
        auto event=record<OmniUsdLiveStatusEvent>(); event.status=&s;
        for(uint64_t revision:{1,2}) {
            s.revision=revision; Participant::status(&p,&event);
            std::lock_guard<std::mutex> lock(p.statusMutex); assert(p.lastStatus.revision==revision);
        }
        {std::lock_guard<std::mutex> lock(mutex); assert(!release); release=true;} changed.notify_all();
        auto report=p.log.close(); assert(report.at("dropped").GetUInt64()==2);
        assert(report.at("written_bytes").GetUInt64()==6 && report.at("error").IsNull());
        assert(p.log.close()==report); // Resumable/idempotent logger retirement.
        try {p.execute({{"operation",JsValue("wait-status")},{"state",JsValue(0)}}); assert(false);}
        catch(const Failure& f) {assert(f.result==OMNI_USD_LIVE_INVALID_ARGUMENT);}
        EventLog failed(root/"failed",[](int,const std::string&){throw std::runtime_error("private error");});
        failed.push("record\n"); report=failed.close();
        assert(report.at("dropped").GetUInt64()==1 && report.at("error").GetString()=="IOError");
        assert(report.at("written_bytes").GetUInt64()==0);
        failed.push("after failure\n"); assert(failed.close().at("dropped").GetUInt64()==2);
        std::atomic<int> writes{0};
        EventLog limited(root/"limited",[&](int,const std::string&){++writes;},1,1024,1);
        limited.push("oversized"); report=limited.close();
        assert(writes==0 && report.at("dropped").GetUInt64()==1 && report.at("error").IsNull());
        fs::remove_all(root);
        std::cout << "native status progresses with blocked/full logger; IO/size/drop/retirement checks passed\n";
        return 0;
    } catch(...) {fs::remove_all(root); throw;}
}
