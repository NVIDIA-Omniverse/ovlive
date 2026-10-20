// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Native projection only; expected semantics live in the independent Python model.
#include "WireScene.h"
#include "graph_value_json.h"
#include <pxr/usd/sdf/listOp.h>
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace omni::usd::live;
namespace {
constexpr std::size_t FileLimit=16u*1024*1024, TotalLimit=64u*1024*1024;
// Deliberate reporting defect used only by the reducer's native self-test.
// Normal qualification never enables it; no product library contains this flag.
bool testDropMarker=false;
void check(bool value,const char* message) { if(!value) throw std::runtime_error(message); }
struct Json {
    std::string text;
    void add(std::string_view value) {
        check(value.size()<=FileLimit-text.size(),"graph probe JSON limit"); text.append(value);
    }
    template<class T> void number(T value) {
        char bytes[64]; const auto encoded=std::to_chars(bytes,bytes+sizeof(bytes),value);
        check(encoded.ec==std::errc{},"graph probe numeric encoding failed"); add({bytes,static_cast<std::size_t>(encoded.ptr-bytes)});
    }
    void quote(std::string_view value) {
        constexpr char hex[]="0123456789abcdef";
        const auto escaped = [&](uint32_t unit) {
            const char bytes[]={'\\','u',hex[(unit>>12)&15],hex[(unit>>8)&15],hex[(unit>>4)&15],hex[unit&15]};
            add({bytes,6});
        };
        add("\"");
        for(std::size_t i=0;i<value.size();) {
            const auto byte=static_cast<unsigned char>(value[i++]);
            if(byte=='"') add("\\\"");
            else if(byte=='\\') add("\\\\");
            else if(byte<32 || byte==127) escaped(byte);
            else if(byte<128) { const char character=static_cast<char>(byte); add({&character,1}); }
            else {
                unsigned count; uint32_t scalar, minimum;
                if(byte>=0xc2 && byte<=0xdf) { count=1; scalar=byte&31; minimum=0x80; }
                else if(byte>=0xe0 && byte<=0xef) { count=2; scalar=byte&15; minimum=0x800; }
                else if(byte>=0xf0 && byte<=0xf4) { count=3; scalar=byte&7; minimum=0x10000; }
                else throw std::runtime_error("invalid UTF-8 in graph fixture");
                check(count<=value.size()-i,"truncated UTF-8 in graph fixture");
                while(count--) {
                    const auto next=static_cast<unsigned char>(value[i++]);
                    check((next&0xc0)==0x80,"invalid UTF-8 continuation in graph fixture");
                    scalar=(scalar<<6)|(next&63);
                }
                check(scalar>=minimum && scalar<=0x10ffff && !(scalar>=0xd800 && scalar<=0xdfff),
                      "invalid UTF-8 scalar in graph fixture");
                if(scalar<=0xffff) escaped(scalar);
                else { scalar-=0x10000; escaped(0xd800+(scalar>>10)); escaped(0xdc00+(scalar&1023)); }
            }
        }
        add("\"");
    }
    void value(const VtValue& value) {
        if(value.IsHolding<SdfTimeSampleMap>()) {
            add("{\"type\":\"time_samples\",\"value\":["); bool first=true;
            for(const auto& [time,sample]:value.UncheckedGet<SdfTimeSampleMap>()) {
                if(!first) add(",");
                first=false;
                uint64_t bits; std::memcpy(&bits,&time,sizeof(bits));
                add("{\"time_bits\":"); number(bits); add(",\"value\":"); this->value(sample); add("}");
            }
            add("]}");
        } else if(value.IsHolding<int>()) {
            add("{\"type\":\"int32\",\"value\":"); number(value.UncheckedGet<int>()); add("}");
        }
        else if(value.IsHolding<double>()) {
            add("{\"type\":\"double\",\"value\":");
            const auto scalar=value.UncheckedGet<double>();
            check(std::isfinite(scalar),"graph fixture contains unsupported nonfinite double");
            const auto begin=text.size(); number(scalar);
            if(text.find_first_of(".eE",begin)==std::string::npos) add(".0");
            add("}");
        } else if(value.IsHolding<bool>()) {
            add("{\"type\":\"bool\",\"value\":"); add(value.UncheckedGet<bool>()?"true":"false"); add("}");
        } else if(value.IsHolding<TfToken>()) {
            add("{\"type\":\"token\",\"value\":"); quote(value.UncheckedGet<TfToken>().GetString()); add("}");
        } else if(value.IsHolding<SdfVariability>()) {
            add("{\"type\":\"variability\",\"value\":"); number(static_cast<uint32_t>(value.UncheckedGet<SdfVariability>())); add("}");
        } else if(value.IsHolding<SdfSpecifier>()) {
            add("{\"type\":\"specifier\",\"value\":"); number(static_cast<uint32_t>(value.UncheckedGet<SdfSpecifier>())); add("}");
        } else if(value.IsHolding<SdfPathVector>()) {
            add("{\"type\":\"paths\",\"value\":["); bool first=true;
            for(const auto& path:value.UncheckedGet<SdfPathVector>()) {
                if(!first) add(",");
                first=false; quote(path.GetString());
            }
            add("]}");
        } else if(value.IsHolding<SdfPathListOp>()) {
            const auto& op=value.UncheckedGet<SdfPathListOp>();
            add("{\"type\":\"path_op\",\"value\":{\"explicit\":"); add(op.IsExplicit()?"true":"false");
            const std::pair<const char*,const SdfPathVector*> buckets[] = {
                {"explicit_items",&op.GetExplicitItems()}, {"added",&op.GetAddedItems()},
                {"prepended",&op.GetPrependedItems()}, {"appended",&op.GetAppendedItems()},
                {"deleted",&op.GetDeletedItems()}, {"ordered",&op.GetOrderedItems()}};
            for(const auto& [name, paths]:buckets) {
                add(","); quote(name); add(":["); bool first=true;
                for(const auto& path:*paths) { if(!first) add(","); first=false; quote(path.GetString()); }
                add("]");
            }
            add("}}");
        } else if(value.IsHolding<std::vector<TfToken>>()) {
            add("{\"type\":\"tokens\",\"value\":["); bool first=true;
            for(const auto& token:value.UncheckedGet<std::vector<TfToken>>()) {
                if(!first) add(",");
                first=false; quote(token.GetString());
            }
            add("]}");
        } else if(!graph_values::project(*this,value)) throw std::runtime_error("graph fixture contains unsupported field type");
    }
};
std::string inspect(const WireScenePtr& scene) {
    Json out; out.add("{\"source_format\":");
    out.quote(scene ? scene->sourceFormat() : std::string_view{});
    out.add(",\"graph\":[");
    if(scene) {
        struct Section { uint64_t id, parent; SdfSpecType type; std::string_view name; uint32_t order; SdfPath path; };
        std::vector<Section> sections;
        sections.reserve(scene->sectionCount());
        // The visible projection only: sections outside it report an empty path.
        scene->visitSections(16384,[&](const auto& section) {
            if(!section.path.IsEmpty()) sections.push_back({section.id, section.parent, section.type, section.name, section.order, section.path});
            return true;
        });
        std::sort(sections.begin(),sections.end(),[](const auto& a,const auto& b) { return a.id<b.id; });
        bool first=true;
        for(const auto& section:sections) {
            if(!first) out.add(",");
            first=false;
            out.add("{\"id\":"); out.number(section.id);
            out.add(",\"parent\":"); out.number(section.parent);
            out.add(",\"type\":"); out.number(static_cast<uint32_t>(section.type));
            out.add(",\"name\":"); out.quote(section.name);
            out.add(",\"order\":"); out.number(section.order);
            out.add(",\"path\":"); out.quote(section.path.GetString());
            out.add(",\"fields\":{");
            auto fields=scene->data()->List(section.path);
            std::sort(fields.begin(),fields.end(),[](const auto& a,const auto& b) { return a.GetString()<b.GetString(); });
            bool firstField=true;
            for(const auto& field:fields) {
                if(testDropMarker && field==TfToken("reducerMarker")) continue;
                if(!firstField) out.add(",");
                firstField=false;
                out.quote(field.GetString()); out.add(":"); out.value(scene->data()->Get(section.path,field));
            }
            out.add("}}");
        }
    }
    out.add("]}"); return std::move(out.text);
}
// The projection plus every graph row, including rows outside it, and the
// last update's field changes: the state a later update builds on.
std::string complete(const WireScenePtr& scene) {
    auto text=inspect(scene);
    if(!scene) return text;
    scene->visitSections(16384,[&](const auto& section) {
        text+="|"+std::to_string(section.id)+","+std::to_string(section.parent)+","+
            std::to_string(static_cast<uint32_t>(section.type))+","+std::string(section.name)+","+
            std::to_string(section.order)+","+section.path.GetString();
        return true;
    });
    for(const auto& change:scene->fieldChanges())
        text+="#"+std::to_string(change.section)+","+std::to_string(change.setOrder)+","+std::to_string(change.record)+","+
            change.field.GetString();
    return text;
}
std::string read(const char* path) {
    std::ifstream input(path,std::ios::binary|std::ios::ate);
    check(bool(input),"cannot open graph fixture");
    const auto size=input.tellg();
    check(size>=0 && static_cast<uint64_t>(size)<=FileLimit,"graph fixture file limit");
    std::string bytes(static_cast<std::size_t>(size),'\0'); input.seekg(0);
    if(!bytes.empty()) check(bool(input.read(bytes.data(),static_cast<std::streamsize>(bytes.size()))),"cannot read graph fixture");
    check(input.peek()==std::char_traits<char>::eof(),"graph fixture changed during read");
    return bytes;
}
}
int main(int argc,char** argv) {
    try {
        const int first=argc>1 && std::string_view(argv[1])=="--test-drop-marker" ? 2 : 1;
        testDropMarker=first==2;
        check(argc>first && argc-first<=1024,"usage: live_scene_graph_probe [--test-drop-marker] OMPF_FILE... (at most 1024)");
        WireSceneLimits limits; limits.maxSections=16384; limits.maxFields=65536; limits.envelope.maxBytes=FileLimit;
        WireScenePtr current;
        // Per input: its bytes, the scene before it, whether it applied, and the rendered scene after it.
        std::vector<std::string> inputs;
        std::vector<WireScenePtr> before;
        std::vector<bool> applied;
        std::vector<std::string> after;
        std::vector<WireScenePtr> retained;
        std::vector<std::string> baselines;
        std::size_t inputBytes=0, renderedBytes=0;
        for(int i=first;i<argc;++i) {
            const auto bytes=read(argv[i]);
            check(bytes.size()<=TotalLimit-inputBytes,"graph probe cumulative input limit"); inputBytes+=bytes.size();
            const char* result="ok";
            WireScenePtr candidate;
            inputs.push_back(bytes); before.push_back(current);
            try { candidate=buildSceneCandidate(current,bytes,limits,nullptr,WireSceneBatchPolicy::Delivered); }
            catch(const wire::DecodeError& error) {
                switch(error.kind) {
                case wire::ErrorKind::Unsupported: result="unsupported"; break;
                case wire::ErrorKind::ResourceLimit: result="limit"; break;
                case wire::ErrorKind::Malformed: case wire::ErrorKind::Integrity: result="malformed"; break;
                }
            }
            // Check after rejection too: failed staging must not mutate inputs.
            for(std::size_t n=0;n<retained.size();++n)
                check(inspect(retained[n])==baselines[n],"retained immutable graph changed");
            if(candidate) {
                auto baseline=inspect(candidate);
                check(baseline.size()<=TotalLimit-renderedBytes,"graph probe retained baseline limit");
                renderedBytes+=baseline.size();
                retained.push_back(candidate); baselines.push_back(std::move(baseline)); current=std::move(candidate);
            }
            applied.push_back(std::string_view(result)=="ok");
            after.push_back(complete(current));
            std::cout<<"{\"result\":\""<<result<<"\",\"scene\":"<<inspect(current)<<"}\n";
            check(bool(std::cout),"graph probe output failed");
        }
        // Every window of up to 8 consecutive inputs, built once as a delivered
        // batch, equals building each input in turn; a window holding a
        // rejected input builds nothing.
        for(std::size_t start=0;start<inputs.size();++start)
            for(std::size_t end=start+1;end<=inputs.size() && end-start<=8;++end) {
                std::vector<WireSceneDelivery> batch;
                for(std::size_t i=start;i<end;++i) batch.push_back({inputs[i],nullptr});
                WireScenePtr built;
                try { built=buildDeliveredScene(before[start],batch,limits); } catch(const wire::DecodeError&) {}
                if(!applied[end-1] ? bool(built) : !built || complete(built)!=after[end-1])
                    std::cerr<<"graph probe batch of inputs "<<start<<".."<<end-1<<'\n';
                if(!applied[end-1]) { check(!built,"a batch holding a rejected update built a scene"); break; }
                check(complete(built)==after[end-1],"a delivered batch differs from building each update in turn");
            }
        // Every window of up to 8 consecutive inputs replayed as pending local
        // intents in one build equals replaying each in turn: PendingReplay
        // candidates, where a Create of a live identity is a predicted
        // rejection that leaves the scene unchanged and a Reorder skips the
        // non-children it names, and, with a strict tail, the last an
        // OrdinaryUpdate candidate. A
        // window whose replay fails in turn fails in one build.
        for(std::size_t start=0;start<inputs.size();++start) {
            WireScenePtr prefix=before[start];
            for(std::size_t end=start+1;end<=inputs.size() && end-start<=8;++end) {
                std::vector<std::string_view> intents;
                for(std::size_t i=start;i<end;++i) intents.push_back(inputs[i]);
                const WireScenePtr previousPrefix=prefix;
                for(const bool strict:{false,true}) {
                    WireScenePtr sequential, built;
                    bool sequentialFailed=false, builtFailed=false;
                    try {
                        sequential=buildSceneCandidate(previousPrefix,inputs[end-1],limits,nullptr,
                            strict?WireSceneBatchPolicy::OrdinaryUpdate:WireSceneBatchPolicy::PendingReplay);
                    } catch(const wire::DecodeError&) { sequentialFailed=true; }
                    try { built=buildReplayedScene(before[start],intents,limits,nullptr,strict); }
                    catch(const wire::DecodeError&) { builtFailed=true; }
                    if(sequentialFailed!=builtFailed || (!sequentialFailed && complete(built)!=complete(sequential)))
                        std::cerr<<"graph probe replay of inputs "<<start<<".."<<end-1<<(strict?" strict":"")<<'\n';
                    check(sequentialFailed==builtFailed,"one-pass replay failed differently from replaying each intent");
                    if(!sequentialFailed)
                        check(complete(built)==complete(sequential),"one-pass replay differs from replaying each intent");
                    if(!strict) {
                        if(sequentialFailed) { prefix.reset(); break; }
                        prefix=sequential;
                    }
                }
                if(!prefix) break;
            }
        }
        return 0;
    } catch(const std::exception& error) {
        std::cerr<<"graph probe failed: "<<error.what()<<'\n'; return 1;
    }
}
