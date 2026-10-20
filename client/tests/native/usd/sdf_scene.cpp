// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "SdfSceneData.h"
#include <pxr/usd/sdf/data.h>
#include <pxr/usd/sdf/fileFormat.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/sdf/primSpec.h>
#include <pxr/usd/sdf/attributeSpec.h>
#include <pxr/usd/sdf/namespaceEdit.h>
#include <pxr/usd/sdf/schema.h>
#include <pxr/base/tf/errorMark.h>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <map>
#include <random>
#include <thread>
#include <future>
#include <system_error>

using namespace PXR_NS;
using omni::usd::live::SdfSceneData;
namespace {
class SceneFormat final : public SdfFileFormat {
public:
    SceneFormat() : SdfFileFormat(TfToken("scene_test"), TfToken("1"), TfToken("usd"), "scene_test") {}
    SdfAbstractDataRefPtr InitData(const FileFormatArguments&) const override {
        auto data = TfCreateRefPtr(new SdfSceneData);
        data->CreateSpec(SdfPath::AbsoluteRootPath(), SdfSpecTypePseudoRoot); return data;
    }
    bool CanRead(const std::string&) const override { return false; }
    bool Read(SdfLayer*, const std::string&, bool) const override { return false; }
    bool ReadFromString(SdfLayer* layer, const std::string& text) const override {
        auto temporary = SdfLayer::CreateAnonymous("parse.usda");
        if (!temporary->ImportFromString(text)) return false;
        SdfAbstractDataRefPtr data = TfCreateRefPtr(new SdfSceneData);
        data->CopyFrom(Inspect(*temporary)); _SetLayerData(layer, data); return true;
    }
    bool WriteToString(const SdfLayer& layer, std::string* output, const std::string& comment) const override {
        return SdfFileFormat::FindById(TfToken("usda"))->WriteToString(layer, output, comment);
    }
    static SdfAbstractDataConstPtr Inspect(const SdfLayer& layer) { return _GetLayerData(layer); }
protected:
    SdfAbstractDataRefPtr _InitDetachedData(const FileFormatArguments& args) const override { return InitData(args); }
};
struct Record {
    SdfSpecType type;
    std::map<TfToken, VtValue> fields;
    bool operator==(const Record& b) const { return type == b.type && fields == b.fields; }
};
using State = std::map<SdfPath, Record>;
State snapshot(const SdfAbstractData& data) {
    struct Visitor : SdfAbstractDataSpecVisitor {
        State state;
        bool VisitSpec(const SdfAbstractData& data, const SdfPath& path) override {
            Record record{data.GetSpecType(path), {}};
            for (const auto& field : data.List(path)) {
                VtValue value; assert(data.Has(path, field, &value));
                assert(value == data.Get(path, field)); record.fields.emplace(field, value);
            }
            state.emplace(path, std::move(record)); return true;
        }
        void Done(const SdfAbstractData&) override {}
    } visitor;
    data.VisitSpecs(&visitor); return visitor.state;
}
void equal(const SdfAbstractData& actual, const SdfAbstractData& expected) {
    const auto a = snapshot(actual), b = snapshot(expected);
    if (a != b) {
        std::cerr << "actual data:\n"; actual.WriteToStream(std::cerr);
        std::cerr << "expected data:\n"; expected.WriteToStream(std::cerr);
    }
    assert(a == b);
    assert(actual.ListAllTimeSamples() == expected.ListAllTimeSamples());
}
void equal(const SdfLayerRefPtr& actual, const SdfLayerRefPtr& expected) {
    assert(dynamic_cast<const SdfSceneData*>(SceneFormat::Inspect(*actual).operator->()));
    equal(*SceneFormat::Inspect(*actual), *SceneFormat::Inspect(*expected));
    std::string a, b; assert(actual->ExportToString(&a) && expected->ExportToString(&b));
    assert(a == b);
}
void rawContracts() {
    auto actual = TfCreateRefPtr(new SdfSceneData);
    auto expected = TfCreateRefPtr(new SdfData);
    auto both = [&](auto operation) { operation(*actual); operation(*expected); equal(*actual, *expected); };
    both([](auto& data) {
        data.CreateSpec(SdfPath("/P"), SdfSpecTypePrim);
        data.CreateSpec(SdfPath("/P/C"), SdfSpecTypePrim);
        data.CreateSpec(SdfPath("/P/C.x"), SdfSpecTypeAttribute);
        data.Set(SdfPath("/P/C.x"), TfToken("default"), VtValue(7));
    });
    both([](auto& data) { data.MoveSpec(SdfPath("/P"), SdfPath("/Q")); });
    assert(actual->HasSpec(SdfPath("/P/C")) && !actual->HasSpec(SdfPath("/Q/C")));
    both([](auto& data) { data.EraseSpec(SdfPath("/P/C")); });
    assert(actual->HasSpec(SdfPath("/P/C.x")));
    both([](auto& data) { data.CreateSpec(SdfPath("/P/C.x"), SdfSpecTypeRelationship); });
    struct Visitor : SdfAbstractDataSpecVisitor {
        unsigned visits = 0, done = 0;
        bool VisitSpec(const SdfAbstractData&, const SdfPath&) override { ++visits; return false; }
        void Done(const SdfAbstractData&) override { ++done; }
    } visitor;
    actual->VisitSpecs(&visitor); assert(visitor.visits == 1 && visitor.done == 1);
    const auto before = snapshot(*actual);
    {
        TfErrorMark mark;
        actual->CopyFrom(SdfAbstractDataConstPtr{});
        assert(!mark.IsClean()); mark.Clear();
        assert(snapshot(*actual) == before);
    }
    actual->CopyFrom(actual); assert(snapshot(*actual) == before);
    auto clone = TfCreateRefPtr(new SdfSceneData); clone->CopyFrom(actual); equal(*clone, *actual);
    struct FailingSource : SdfData {
        mutable unsigned reads = 0;
        VtValue Get(const SdfPath& path, const TfToken& field) const override {
            if (++reads == 2) throw std::bad_alloc();
            return SdfData::Get(path, field);
        }
    };
    auto failing = TfCreateRefPtr(new FailingSource);
    failing->CreateSpec(SdfPath("/Different"), SdfSpecTypePrim);
    failing->Set(SdfPath("/Different"), TfToken("first"), VtValue(9));
    failing->Set(SdfPath("/Different"), TfToken("second"), VtValue(10));
    bool failed = false;
    try { actual->CopyFrom(failing); } catch (const std::bad_alloc&) { failed = true; }
    assert(failed && snapshot(*actual) == before);
    auto malformed = TfCreateRefPtr(new SdfData);
    malformed->CreateSpec(SdfPath("/Different"), SdfSpecTypeAttribute);
    malformed->Set(SdfPath("/Different"), SdfFieldKeys->TimeSamples, VtValue(9));
    failed = false;
    try { actual->CopyFrom(malformed); } catch (const std::invalid_argument&) { failed = true; }
    assert(failed && snapshot(*actual) == before);
    both([](auto& data) { data.Set(SdfPath("/P/C.x"), SdfFieldKeys->TimeSamples, VtValue(SdfTimeSampleMap{})); });
    both([](auto& data) { data.SetTimeSample(SdfPath("/P/C.x"), 1, VtValue(2)); });
    int stored = 0;
    SdfAbstractDataTypedValue<int> typed(&stored);
    assert(actual->QueryTimeSample(SdfPath("/P/C.x"), 1, &typed) && stored == 2);
    both([](auto& data) { data.EraseTimeSample(SdfPath("/P/C.x"), 1); });
    both([](auto& data) { data.Set(SdfPath("/P/C.x"), TfToken("default"), VtValue(SdfValueBlock{})); });
    both([](auto& data) { data.Erase(SdfPath("/P/C.x"), TfToken("default")); });
    assert(!actual->Has(SdfPath("/P/C.x"), TfToken("default")));
    std::cout << "raw Sdf one-spec, type, visitor, copy and presence contracts passed" << std::endl;
}
void publicLayer(uint64_t seed) {
    SdfFileFormatRefPtr format = TfCreateRefPtr(new SceneFormat);
    auto actual = SdfLayer::CreateAnonymous("actual.scene_test", format);
    auto expected = SdfLayer::CreateAnonymous("oracle.usda");
    assert(actual && expected); equal(actual, expected);
    auto both = [&](auto operation) { operation(actual); operation(expected); equal(actual, expected); };
    both([](const auto& layer) {
        auto root = SdfPrimSpec::New(layer, "Root", SdfSpecifierDef);
        auto child = SdfPrimSpec::New(root, "Child", SdfSpecifierDef);
        SdfPrimSpec::New(root, "Other", SdfSpecifierOver);
        auto attr = SdfAttributeSpec::New(child, "value", SdfValueTypeNames->Double);
        attr->SetDefaultValue(VtValue(3.0));
    });
    both([](const auto& layer) {
        layer->SetFieldDictValueByKey(SdfPath("/Root"), SdfFieldKeys->CustomData, TfToken("nested:key"), VtValue(4));
    });
    std::mt19937_64 random(seed);
    for (unsigned step = 0; step < 128; ++step) {
        const double time = double(random() % 16), value = double(random() % 10000);
        const auto op = random() % 3;
        std::cout << "sdf_step=" << step << " op=" << op << " time=" << time << " value=" << value << std::endl;
        both([&](const auto& layer) {
            if (op == 0) layer->EraseTimeSample(SdfPath("/Root/Child.value"), time);
            else layer->SetTimeSample(SdfPath("/Root/Child.value"), time, op == 1 ? VtValue(value) : VtValue(SdfValueBlock{}));
        });
        const auto a = SceneFormat::Inspect(*actual), b = SceneFormat::Inspect(*expected);
        for (double query : {-1.0, 0.0, 7.5, 15.0, 20.0}) {
            double al = -99, ah = -99, bl = -99, bh = -99;
            assert(a->GetBracketingTimeSamples(query, &al, &ah) == b->GetBracketingTimeSamples(query, &bl, &bh));
            assert(al == bl && ah == bh);
            al = ah = bl = bh = -99;
            const SdfPath path("/Root/Child.value");
            assert(a->GetBracketingTimeSamplesForPath(path, query, &al, &ah) == b->GetBracketingTimeSamplesForPath(path, query, &bl, &bh));
            assert(al == bl && ah == bh);
            al = bl = -99;
            assert(a->GetPreviousTimeSampleForPath(path, query, &al) == b->GetPreviousTimeSampleForPath(path, query, &bl));
            assert(al == bl);
        }
    }
    std::cout << "public namespace rename" << std::endl;
    both([](const auto& layer) {
        SdfBatchNamespaceEdit edits; edits.Add(SdfPath("/Root/Child"), SdfPath("/Root/Renamed"));
        assert(layer->Apply(edits));
    });
    both([](const auto& layer) {
        auto root = layer->GetPrimAtPath(SdfPath("/Root"));
        root->SetNameChildren({layer->GetPrimAtPath(SdfPath("/Root/Other")), layer->GetPrimAtPath(SdfPath("/Root/Renamed"))});
        root->SetNameChildrenOrder({TfToken("Renamed"), TfToken("Other")});
    });
    both([](const auto& layer) {
        SdfBatchNamespaceEdit edits;
        edits.Add(SdfPath("/Root/Renamed.value"), SdfPath("/Root/Other.moved"));
        assert(layer->Apply(edits));
    });
    both([](const auto& layer) {
        SdfBatchNamespaceEdit edits;
        edits.Add(SdfPath("/Root/Other"), SdfPath("/Root/Renamed/Nested"));
        assert(layer->Apply(edits));
    });
    auto transferred = SdfLayer::CreateAnonymous("transfer.scene_test", format);
    std::cout << "public transfer" << std::endl;
    transferred->TransferContent(expected); equal(transferred, expected);
    std::string text; assert(actual->ExportToString(&text));
    auto imported = SdfLayer::CreateAnonymous("import.scene_test", format);
    std::cout << "public import" << std::endl;
    auto importedOracle = SdfLayer::CreateAnonymous("import-oracle.usda");
    assert(imported->ImportFromString(text) && importedOracle->ImportFromString(text));
    // Stock USDA round trips normalize implicit __AnyType__ authoring. Compare
    // identical parse operations, while still requiring byte-identical export.
    equal(imported, importedOracle);
    std::string roundTrip; assert(imported->ExportToString(&roundTrip) && roundTrip == text);
    both([](const auto& layer) {
        SdfBatchNamespaceEdit edits; edits.Add(SdfPath("/Root/Renamed"), SdfPath()); assert(layer->Apply(edits));
    });
    both([](const auto& layer) { layer->Clear(); });
    std::cout << "public Sdf layer oracle passed" << std::endl;
}
void concurrentData() {
    auto data = TfCreateRefPtr(new SdfSceneData);
    const SdfPath path("/P"); data->CreateSpec(path, SdfSpecTypePrim);
    std::promise<void> start; auto go = start.get_future().share();
    std::vector<std::thread> writers;
    for (int i = 0; i < 4; ++i) writers.emplace_back([&, i] {
        go.wait();
        for (int n = 0; n < 100; ++n) {
            data->SetDictValueByKey(path, SdfFieldKeys->CustomData, TfToken("k" + std::to_string(i)), VtValue(n));
            data->SetTimeSample(path, i * 100 + n, VtValue(n));
        }
    });
    start.set_value(); for (auto& thread : writers) thread.join();
    assert(data->GetNumTimeSamplesForPath(path) == 400);
    for (int i = 0; i < 4; ++i) for (int n = 0; n < 100; ++n) {
        VtValue value; assert(data->QueryTimeSample(path, i * 100 + n, &value) && value == VtValue(n));
    }
    for (int i = 0; i < 4; ++i) assert(data->GetDictValueByKey(path, SdfFieldKeys->CustomData, TfToken("k" + std::to_string(i))) == VtValue(99));
    auto other = TfCreateRefPtr(new SdfSceneData); other->CopyFrom(data);
    const auto expected = snapshot(*data);
    std::promise<void> copyStart; auto copyGo = copyStart.get_future().share();
    std::atomic<unsigned> ready{0};
    std::thread first([&] { ++ready; copyGo.wait(); for (unsigned i = 0; i < 100; ++i) data->CopyFrom(other); });
    std::thread second([&] { ++ready; copyGo.wait(); for (unsigned i = 0; i < 100; ++i) other->CopyFrom(data); });
    while (ready != 2) std::this_thread::yield();
    copyStart.set_value(); first.join(); second.join();
    assert(snapshot(*data) == expected && snapshot(*other) == expected);
    std::cout << "backing-store compound dictionary and opposing-copy contracts passed" << std::endl;
}
// SdfSceneData::_VisitSpecs visits a snapshot of the paths without holding the
// read lock, as stock SdfData does, so a visitor may use the data, and two datas
// visited into each other never nest their locks.
void visitReleasesReadLock() {
    auto data = TfCreateRefPtr(new SdfSceneData);
    data->CreateSpec(SdfPath("/A"), SdfSpecTypePrim);
    struct Author : SdfAbstractDataSpecVisitor {
        SdfSceneData* target;
        bool VisitSpec(const SdfAbstractData&, const SdfPath&) override {
            target->CreateSpec(SdfPath("/Visited"), SdfSpecTypePrim); // Throws while a read guard is held.
            return false;
        }
        void Done(const SdfAbstractData&) override {}
    } author;
    author.target = data.operator->();
    try {
        data->VisitSpecs(&author);
    } catch (const std::system_error&) {
        std::cerr << "visit held the read lock" << std::endl; std::abort();
    }
    assert(data->HasSpec(SdfPath("/Visited")));
    std::cout << "visit releases the read lock" << std::endl;
}
}
int main(int argc, char** argv) {
    const auto seed = argc == 2 ? std::stoull(argv[1]) : 42;
    std::cout << "master_seed=" << seed << " scenario=sdf-scene" << std::endl;
    rawContracts(); publicLayer(seed); concurrentData(); visitReleasesReadLock();
}
