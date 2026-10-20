// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "DiagnosticLog.h"
#include <array>
#include <charconv>
#include <new>
#include <stdexcept>
#include <string_view>

namespace omni::usd::live::sample {
namespace {
using Result = OmniUsdLiveResult;
void require(bool condition, Result result = OMNI_USD_LIVE_INVALID_ARGUMENT) {
    if (!condition) throw result;
}
template<class T> void header(const T& value) {
    require(value.struct_size >= sizeof(T));
    require((value.abi_version >> 16) == (OMNI_USD_LIVE_ABI_VERSION >> 16), OMNI_USD_LIVE_UNSUPPORTED);
}
std::string_view bounded(const char* value, std::size_t limit, bool empty = false) {
    require(value != nullptr);
    for (std::size_t size = 0; size <= limit; ++size) {
        if (!value[size]) {
            require(empty || size);
            return {value, size};
        }
    }
    throw Result(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
}
struct Writer {
    std::string bytes;
    const std::size_t limit;
    explicit Writer(std::size_t maxBytes) : limit(maxBytes) { bytes.reserve(limit < 1024 ? limit : 1024); }
    void add(std::string_view value) {
        require(value.size() <= limit - bytes.size(), OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
        bytes.append(value.data(), value.size());
    }
    template<class T> void number(T value) {
        char buffer[32];
        const auto converted = std::to_chars(buffer, buffer + sizeof(buffer), value);
        require(converted.ec == std::errc{}, OMNI_USD_LIVE_INTERNAL_ERROR);
        add({buffer, static_cast<std::size_t>(converted.ptr - buffer)});
    }
    void quoted(std::string_view value) {
        static constexpr char hex[] = "0123456789abcdef";
        add("\"");
        for (const unsigned char byte : value) {
            if (byte == '"') add("\\\"");
            else if (byte == '\\') add("\\\\");
            else if (byte < 32 || byte >= 127) {
                const char escape[] = {'\\', 'u', '0', '0', hex[byte >> 4], hex[byte & 15]};
                add({escape, sizeof(escape)});
            } else {
                const char character = static_cast<char>(byte);
                add({&character, 1});
            }
        }
        add("\"");
    }
};
}
OmniUsdLiveResult formatDiagnosticRecord(const OmniUsdLiveDiagnosticRecord& record, LogFormat format,
    std::string& output, std::size_t maxBytes) noexcept {
    try {
        require(format == LogFormat::JsonLines || format == LogFormat::Text);
        require(maxBytes <= 65536);
        require(maxBytes != 0, OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
        header(record);
        require(record.schema_version == OMNI_USD_LIVE_DIAGNOSTIC_SCHEMA_VERSION, OMNI_USD_LIVE_UNSUPPORTED);
        require(record.event_code != 0 && record.severity <= OMNI_USD_LIVE_SEVERITY_ERROR);
        require(record.field_count <= 16, OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
        require(!record.field_count || record.fields);
        const auto event = bounded(record.event_name, 64);
        std::array<std::string_view, 16> names{}, strings{};
        for (std::size_t i = 0; i < record.field_count; ++i) {
            require(record.fields[i] != nullptr);
            const auto& field = *record.fields[i];
            header(field);
            names[i] = bounded(field.name, 64);
            for (std::size_t j = 0; j < i; ++j) require(names[i] != names[j]);
            require(field.kind >= OMNI_USD_LIVE_FIELD_UINT64 && field.kind <= OMNI_USD_LIVE_FIELD_BOOLEAN);
            if (field.kind == OMNI_USD_LIVE_FIELD_STRING) strings[i] = bounded(field.string_value, 512, true);
            if (field.kind == OMNI_USD_LIVE_FIELD_BOOLEAN) require(field.unsigned_value <= 1);
        }
        Writer writer(maxBytes);
        const bool json = format == LogFormat::JsonLines;
        if (json) writer.add("{\"text_encoding\":\"escaped_bytes\"");
        else {
            static constexpr const char* levels[] = {"TRACE", "DEBUG", "INFO", "WARN", "ERROR"};
            writer.add(levels[record.severity]);
            writer.add(" ");
            writer.quoted(event);
            writer.add(" text_encoding=escaped_bytes");
        }
        const auto number = [&](const char* key, auto value) {
            writer.add(json ? "," : " ");
            if (json) {
                writer.quoted(key);
                writer.add(":");
            } else {
                writer.add(key);
                writer.add("=");
            }
            writer.number(value);
        };
        number("struct_size", record.struct_size);
        number("abi_version", record.abi_version);
        number("schema_version", record.schema_version);
        number("event_code", record.event_code);
        if (json) {
            writer.add(",\"event_name\":");
            writer.quoted(event);
        }
        number("severity", record.severity);
        number("timestamp_ms", record.timestamp_ms);
        number("elapsed_ms", record.elapsed_ms);
        number("runtime_instance_id", record.runtime_instance_id);
        number("sequence", record.sequence);
        number("endpoint", record.endpoint);
        number("subject_id", record.subject_id);
        number("incident_id", record.incident_id);
        number("field_count", record.field_count);
        writer.add(json ? ",\"fields\":{" : " fields={");
        for (std::size_t i = 0; i < record.field_count; ++i) {
            if (i) writer.add(",");
            const auto& field = *record.fields[i];
            writer.quoted(names[i]);
            writer.add(json ? ":{\"kind\":" : "[");
            writer.number(field.kind);
            writer.add(json ? ",\"value\":" : "]=");
            switch (field.kind) {
            case OMNI_USD_LIVE_FIELD_UINT64: writer.number(field.unsigned_value); break;
            case OMNI_USD_LIVE_FIELD_INT64: writer.number(field.signed_value); break;
            case OMNI_USD_LIVE_FIELD_STRING: writer.quoted(strings[i]); break;
            case OMNI_USD_LIVE_FIELD_BOOLEAN: writer.add(field.unsigned_value ? "true" : "false"); break;
            default: throw Result(OMNI_USD_LIVE_INVALID_ARGUMENT);
            }
            if (json) writer.add("}");
        }
        writer.add(json ? "}}\n" : "}\n");
        output.swap(writer.bytes);
        return OMNI_USD_LIVE_OK;
    } catch (Result result) { return result; } catch (const std::bad_alloc&) {
        return OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
    } catch (const std::length_error&) { return OMNI_USD_LIVE_RESOURCE_EXHAUSTED; } catch (...) {
        return OMNI_USD_LIVE_INTERNAL_ERROR;
    }
}
} // namespace omni::usd::live::sample
