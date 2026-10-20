// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "DiagnosticLog.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using namespace omni::usd::live::sample;
namespace {
void check(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
template<class T> T record() {
    T value{}; value.struct_size = sizeof(T); value.abi_version = OMNI_USD_LIVE_ABI_VERSION; return value;
}
struct Fixture {
    OmniUsdLiveDiagnosticRecord event = record<OmniUsdLiveDiagnosticRecord>();
    std::array<OmniUsdLiveDiagnosticField,16> fields{};
    std::array<const OmniUsdLiveDiagnosticField*,16> pointers{};
    std::array<std::string,16> names;
    Fixture() {
        event.schema_version = OMNI_USD_LIVE_DIAGNOSTIC_SCHEMA_VERSION;
        event.event_code = 9; event.event_name = "event\"\\\n\x80";
        event.severity = OMNI_USD_LIVE_SEVERITY_WARN;
        event.timestamp_ms = UINT64_MAX; event.elapsed_ms = 42; event.runtime_instance_id = 43;
        event.sequence = 44; event.endpoint = 45; event.subject_id = 46; event.incident_id = 47;
        event.field_count = 4; event.fields = pointers.data();
        for (std::size_t i=0; i<fields.size(); ++i) {
            fields[i] = record<OmniUsdLiveDiagnosticField>();
            names[i] = "key" + std::to_string(i); fields[i].name = names[i].c_str();
            fields[i].kind = OMNI_USD_LIVE_FIELD_UINT64; pointers[i] = &fields[i];
        }
        fields[0].name = "unsigned"; fields[0].unsigned_value = UINT64_MAX;
        fields[1].name = "signed"; fields[1].kind = OMNI_USD_LIVE_FIELD_INT64;
        fields[1].signed_value = INT64_MIN;
        fields[2].name = "string\"\n"; fields[2].kind = OMNI_USD_LIVE_FIELD_STRING;
        fields[2].string_value = "quote\"slash\\\n\r\t\x01\x7f\xff";
        fields[3].name = "boolean"; fields[3].kind = OMNI_USD_LIVE_FIELD_BOOLEAN; fields[3].unsigned_value = 1;
    }
};
std::string format(const Fixture& fixture, LogFormat kind, std::size_t limit=65536) {
    std::string output;
    check(formatDiagnosticRecord(fixture.event,kind,output,limit)==OMNI_USD_LIVE_OK,"valid record rejected");
    check(!output.empty() && output.back()=='\n' && std::count(output.begin(),output.end(),'\n')==1,"line injection or missing final newline");
    for (std::size_t i=0;i+1<output.size();++i)
        check(static_cast<unsigned char>(output[i])>=32 && static_cast<unsigned char>(output[i])<127,"unescaped output byte");
    return output;
}
void rejected(const Fixture& fixture, OmniUsdLiveResult expected,
              LogFormat kind=LogFormat::JsonLines, std::size_t limit=65536) {
    const std::string original("unchanged\0sentinel",18); auto output=original;
    check(formatDiagnosticRecord(fixture.event,kind,output,limit)==expected,"wrong rejection result");
    check(output==original,"failure changed caller output");
}
void tests() {
    Fixture fixture;
    const auto json=format(fixture,LogFormat::JsonLines);
    const auto text=format(fixture,LogFormat::Text);
    for (const auto* item : {"\"timestamp_ms\":18446744073709551615", "\"elapsed_ms\":42", "\"runtime_instance_id\":43",
        "\"sequence\":44", "\"endpoint\":45", "\"subject_id\":46", "\"incident_id\":47", "\"event_code\":9",
        "\"schema_version\":1", "\"severity\":3", "\"field_count\":4", "\"text_encoding\":\"escaped_bytes\"",
        "\"unsigned\":{\"kind\":1,\"value\":18446744073709551615}",
        "\"signed\":{\"kind\":2,\"value\":-9223372036854775808}",
        "\"boolean\":{\"kind\":4,\"value\":true}",
        "event\\\"\\\\\\u000a\\u0080", "quote\\\"slash\\\\\\u000a\\u000d\\u0009\\u0001\\u007f\\u00ff"})
        check(json.find(item)!=std::string::npos,"JSON lost scalar/type/escaped bytes");
    check(text.find("WARN \"event") == 0 && text.find("\"signed\"[2]=-9223372036854775808")!=std::string::npos,
          "text lost severity or signed field kind");
    for (const auto kind : {LogFormat::JsonLines,LogFormat::Text}) {
        const auto encoded=format(fixture,kind);
        check(format(fixture,kind,encoded.size())==encoded,"exact byte limit rejected or changed output");
        rejected(fixture,OMNI_USD_LIVE_RESOURCE_EXHAUSTED,kind,encoded.size()-1);
        rejected(fixture,OMNI_USD_LIVE_RESOURCE_EXHAUSTED,kind,0);
        rejected(fixture,OMNI_USD_LIVE_INVALID_ARGUMENT,kind,65537);
    }
    rejected(fixture,OMNI_USD_LIVE_INVALID_ARGUMENT,static_cast<LogFormat>(99));
    fixture.fields[1].signed_value=INT64_MAX; fixture.fields[3].unsigned_value=0;
    check(format(fixture,LogFormat::JsonLines).find("\"value\":9223372036854775807")!=std::string::npos,"positive signed value lost");
    check(format(fixture,LogFormat::JsonLines).find("\"kind\":4,\"value\":false")!=std::string::npos,"false boolean lost");
    fixture.event.field_count=16; format(fixture,LogFormat::JsonLines);
    fixture.event.field_count=17; rejected(fixture,OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    fixture.event.field_count=0; fixture.event.fields=nullptr;
    check(format(fixture,LogFormat::JsonLines).find("\"fields\":{}")!=std::string::npos,"empty fields malformed");
    for (int mutation=0;mutation<15;++mutation) {
        Fixture invalid;
        OmniUsdLiveResult expected=OMNI_USD_LIVE_INVALID_ARGUMENT;
        switch(mutation) {
        case 0: --invalid.event.struct_size; break;
        case 1: invalid.event.abi_version=2u<<16; expected=OMNI_USD_LIVE_UNSUPPORTED; break;
        case 2: ++invalid.event.schema_version; expected=OMNI_USD_LIVE_UNSUPPORTED; break;
        case 3: invalid.event.severity=5; break;
        case 4: invalid.event.event_name=nullptr; break;
        case 5: invalid.event.event_name=""; break;
        case 6: invalid.event.event_code=0; break;
        case 7: invalid.event.fields=nullptr; break;
        case 8: invalid.pointers[0]=nullptr; break;
        case 9: --invalid.fields[0].struct_size; break;
        case 10: invalid.fields[0].abi_version=2u<<16; expected=OMNI_USD_LIVE_UNSUPPORTED; break;
        case 11: invalid.fields[0].kind=0; break;
        case 12: invalid.fields[3].unsigned_value=2; break;
        case 13: invalid.fields[2].string_value=nullptr; break;
        case 14: invalid.fields[1].name=invalid.fields[0].name; break;
        }
        rejected(invalid,expected);
    }
    Fixture lengths;
    const std::string atName(64,'n'), longName(65,'n'), atString(512,'s'), longString(513,'s');
    lengths.event.event_name=atName.c_str(); lengths.fields[0].name=atName.c_str(); lengths.fields[2].string_value=atString.c_str();
    format(lengths,LogFormat::JsonLines);
    lengths.event.event_name=longName.c_str(); rejected(lengths,OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    lengths.event.event_name="event"; lengths.fields[0].name=longName.c_str(); rejected(lengths,OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    lengths.fields[0].name=nullptr; rejected(lengths,OMNI_USD_LIVE_INVALID_ARGUMENT);
    lengths.fields[0].name=""; rejected(lengths,OMNI_USD_LIVE_INVALID_ARGUMENT);
    lengths.fields[0].name="valid"; lengths.fields[2].string_value=longString.c_str(); rejected(lengths,OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    lengths.fields[2].string_value=""; format(lengths,LogFormat::JsonLines);
    lengths.event.abi_version |= 0xffff; lengths.fields[0].abi_version |= 0xffff;
    format(lengths,LogFormat::JsonLines); // Same-major extension is accepted.
}
}
int main(int argc,char** argv) {
    try {
        tests();
        if(argc==2 && std::string(argv[1])=="--emit") {
            Fixture fixture;
            std::cout<<format(fixture,LogFormat::JsonLines)<<format(fixture,LogFormat::Text);
        } else {
            check(argc==1,"usage: diagnostic-log-tests [--emit]");
            std::cout<<"diagnostic formatter bounds, typed fields and injection checks passed\n";
        }
        return 0;
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
