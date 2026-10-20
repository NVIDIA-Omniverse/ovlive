// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// SDK diagnostic only. Deliberately broken no-op storage is confined to this process.
#include <pxr/pxr.h>
#include <pxr/base/tf/errorMark.h>
#include <pxr/usd/sdf/data.h>
#include <pxr/usd/sdf/fileFormat.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/sdf/layerStateDelegate.h>
#include <pxr/usd/sdf/namespaceEdit.h>
#include <pxr/usd/sdf/schema.h>
#include <pxr/usd/sdf/spec.h>
#include <iostream>
#include <iomanip>
#include <dlfcn.h>
#include <stdexcept>
PXR_NAMESPACE_USING_DIRECTIVE
class Data final : public SdfData {
public:
    bool reject=false;
    unsigned moves=0,sets=0;
    bool StreamsData() const override { return true; }
    bool IsDetached() const override { return false; }
    void MoveSpec(const SdfPath& a,const SdfPath& b) override {
        ++moves; if(!reject) SdfData::MoveSpec(a,b);
    }
    void Set(const SdfPath& p,const TfToken& f,const VtValue& v) override {
        ++sets; if(!reject) SdfData::Set(p,f,v);
    }
    void Set(const SdfPath& p,const TfToken& f,const SdfAbstractDataConstValue& v) override {
        ++sets; if(!reject) SdfData::Set(p,f,v);
    }
};
class Format final : public SdfFileFormat {
    SdfAbstractDataRefPtr data_;
public:
    explicit Format(SdfAbstractDataRefPtr data):SdfFileFormat(TfToken("boundary_probe"),TfToken("1"),TfToken("usd"),"boundary_probe"),data_(std::move(data)) {}
    SdfAbstractDataRefPtr InitData(const FileFormatArguments&) const override { return data_; }
    bool CanRead(const std::string&) const override { return false; }
    bool Read(SdfLayer*,const std::string&,bool) const override { return false; }
    static const SdfAbstractData* backing(const SdfLayer& layer) { return _GetLayerData(layer).operator->(); }
protected:
    SdfAbstractDataRefPtr _InitDetachedData(const FileFormatArguments& a) const override { return InitData(a); }
};
class Delegate final : public SdfSimpleLayerStateDelegate {
public:
    unsigned notifications=0;
protected:
    void _OnSetField(const SdfPath&,const TfToken&,const VtValue&) override {
        ++notifications; _GetLayer()->SetPermissionToEdit(false);
    }
    void _OnSetField(const SdfPath&,const TfToken&,const SdfAbstractDataConstValue&) override {
        ++notifications; _GetLayer()->SetPermissionToEdit(false);
    }
};
int main(int argc,char** argv) try {
    if(argc!=2) return 2;
    // Attribute observations to the loaded Sdf implementation, not just link inputs.
    const auto function=static_cast<SdfLayerRefPtr (*)(const std::string&,const SdfLayer::FileFormatArguments&)>(&SdfLayer::FindOrOpen);
    Dl_info loaded{};
    if(!dladdr(reinterpret_cast<const void*>(function),&loaded) || !loaded.dli_fname)
        throw std::runtime_error("loaded Sdf library identity unavailable");
    const std::string library=loaded.dli_fname;
    if(library.find_first_of("\r\n\t")!=std::string::npos)
        throw std::runtime_error("unsupported control character in SDK path");
    const auto data=TfCreateRefPtr(new Data);
    data->CreateSpec(SdfPath::AbsoluteRootPath(),SdfSpecTypePseudoRoot);
    data->CreateSpec(SdfPath("/P"),SdfSpecTypePrim);
    data->Set(SdfPath::AbsoluteRootPath(),SdfChildrenKeys->PrimChildren,VtValue(TfTokenVector{TfToken("P")}));
    data->Set(SdfPath("/P"),SdfFieldKeys->Specifier,VtValue(SdfSpecifierDef));
    const SdfFileFormatRefPtr format=TfCreateRefPtr(new Format(data));
    const auto layer=SdfLayer::CreateAnonymous("boundary.boundary_probe",format);
    if(!layer || Format::backing(*layer)!=data.operator->()) throw std::runtime_error("backing not adopted");
    data->sets=0;
    TfErrorMark mark;
    std::cout << std::boolalpha;
    std::cout << "{\"schema_version\":1,\"boundary\":\"usd_edit_boundary_diagnostic\",\"usd_version\":"
              << PXR_VERSION << ",\"sdf_library\":" << std::quoted(library) << "}\n";
    const std::string mode=argv[1];
    if(mode=="permission") {
        layer->SetPermissionToEdit(false);
        layer->SetField(SdfPath("/P"),TfToken("customData"),VtValue(VtDictionary{}));
        const auto readonlyCalls=data->sets;
        mark.Clear();
        layer->SetPermissionToEdit(true);
        layer->SetField(SdfPath("/P"),TfToken("customData"),VtValue(VtDictionary{}));
        std::cout << "{\"mode\":\"permission\",\"readonly_data_calls\":" << readonlyCalls
                  << ",\"override_permission\":" << layer->PermissionToEdit()
                  << ",\"override_data_calls\":" << data->sets << "}\n";
    } else if(mode=="noop_move" || mode=="move_control") {
        const SdfSpec retained=layer->GetObjectAtPath(SdfPath("/P")).GetSpec();
        if(retained.IsDormant() || retained.GetPath()!=SdfPath("/P"))
            throw std::runtime_error("initial retained spec is not valid at /P");
        data->reject=mode=="noop_move";
        SdfBatchNamespaceEdit edits; edits.Add(SdfPath("/P"),SdfPath("/Q"));
        const bool applied=layer->Apply(edits);
        std::cout << "{\"mode\":" << std::quoted(mode) << ",\"apply_returned_success\":" << applied
                  << ",\"data_move_calls\":" << data->moves
                  << ",\"old_data_present\":" << data->HasSpec(SdfPath("/P"))
                  << ",\"new_data_present\":" << data->HasSpec(SdfPath("/Q"))
                  << ",\"retained_path\":\"" << retained.GetPath().GetString()
                  << "\",\"retained_dormant\":" << retained.IsDormant() << "}\n";
    } else if(mode=="delegate") {
        const auto delegate=TfCreateRefPtr(new Delegate);
        layer->SetStateDelegate(delegate);
        layer->SetField(SdfPath("/P"),TfToken("documentation"),VtValue(std::string("probe")));
        std::cout << "{\"mode\":\"delegate\",\"notifications\":" << delegate->notifications
                  << ",\"permission_after_callback\":" << layer->PermissionToEdit()
                  << ",\"data_set_calls\":" << data->sets
                  << ",\"value_was_written\":" << (data->Get(SdfPath("/P"),TfToken("documentation"))==VtValue(std::string("probe"))) << "}\n";
    } else return 2;
    const bool unexpected=!mark.IsClean(); mark.Clear();
    return unexpected?3:0;
} catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 2; }
