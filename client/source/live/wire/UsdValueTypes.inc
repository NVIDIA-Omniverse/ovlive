// SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

// Note that changing any of the enum value numbers will break backward
// compatibility.  Adding additional types will not.
//
// Also note that the enumerant value 0 is reserved, and corresponds to the
// enumerant 'Invalid'.

// These types are packed into an opaque binary blobs
// USD_TYPE(<enumerant>, <enumerant-value>, <c++ type>, <version>)

// Array types.
USD_TYPE(Bool, 1, bool, 0)
USD_TYPE(UChar, 2, uint8_t, 0)
USD_TYPE(Int, 3, int, 0)
USD_TYPE(UInt, 4, unsigned int, 0)
USD_TYPE(Int64, 5, int64_t, 0)
USD_TYPE(UInt64, 6, uint64_t, 0)

USD_TYPE(Half, 7, GfHalf, 0)
USD_TYPE(Float, 8, float, 0)
USD_TYPE(Double, 9, double, 0)

USD_TYPE(String, 10, std::string, 0)

USD_TYPE(Token, 11, TfToken, 0)

USD_TYPE(AssetPath, 12, SdfAssetPath, 0)

USD_TYPE(Matrix2d, 13, GfMatrix2d, 0)
USD_TYPE(Matrix3d, 14, GfMatrix3d, 0)
USD_TYPE(Matrix4d, 15, GfMatrix4d, 0)

USD_TYPE(Quatd, 16, GfQuatd, 0)
USD_TYPE(Quatf, 17, GfQuatf, 0)
USD_TYPE(Quath, 18, GfQuath, 0)

USD_TYPE(Vec2d, 19, GfVec2d, 0)
USD_TYPE(Vec2f, 20, GfVec2f, 0)
USD_TYPE(Vec2h, 21, GfVec2h, 0)
USD_TYPE(Vec2i, 22, GfVec2i, 0)

USD_TYPE(Vec3d, 23, GfVec3d, 0)
USD_TYPE(Vec3f, 24, GfVec3f, 0)
USD_TYPE(Vec3h, 25, GfVec3h, 0)
USD_TYPE(Vec3i, 26, GfVec3i, 0)

USD_TYPE(Vec4d, 27, GfVec4d, 0)
USD_TYPE(Vec4f, 28, GfVec4f, 0)
USD_TYPE(Vec4h, 29, GfVec4h, 0)
USD_TYPE(Vec4i, 30, GfVec4i, 0)

// Non-array types.
USD_TYPE(Dictionary, 31, VtDictionary, 0)

USD_TYPE(TokenListOp, 32, SdfTokenListOp, 0)
USD_TYPE(StringListOp, 33, SdfStringListOp, 0)
USD_TYPE(PathListOp, 34, SdfPathListOp, 0)
USD_TYPE(ReferenceListOp, 35, SdfReferenceListOp, 0)
USD_TYPE(IntListOp, 36, SdfIntListOp, 0)
USD_TYPE(Int64ListOp, 37, SdfInt64ListOp, 0)
USD_TYPE(UIntListOp, 38, SdfUIntListOp, 0)
USD_TYPE(UInt64ListOp, 39, SdfUInt64ListOp, 0)

USD_TYPE(PathVector, 40, SdfPathVector, 0)
USD_TYPE(TokenVector, 41, std::vector<TfToken>, 0)

USD_TYPE(Specifier, 42, SdfSpecifier, 0)
USD_TYPE(Permission, 43, SdfPermission, 0)
USD_TYPE(Variability, 44, SdfVariability, 0)

USD_TYPE(VariantSelectionMap, 45, SdfVariantSelectionMap, 0)
USD_TYPE(TimeSamples, 46, SdfTimeSampleMap, 0)
USD_TYPE(Payload, 47, SdfPayload, 0)

USD_TYPE(DoubleVector, 48, std::vector<double>, 0)
USD_TYPE(LayerOffsetVector, 49, std::vector<SdfLayerOffset>, 0)
USD_TYPE(StringVector, 50, std::vector<std::string>, 0)

USD_TYPE(ValueBlock, 51, SdfValueBlock, 0)
USD_TYPE(Value, 52, VtValue, 0)

USD_TYPE(UnregisteredValue, 53, SdfUnregisteredValue, 0)
USD_TYPE(UnregisteredValueListOp, 54, SdfUnregisteredValueListOp, 0)

USD_TYPE(Path, 55, SdfPath, 0)
USD_TYPE(SpecType, 56, SdfSpecType, 0)

// Crate version 0.8.0
USD_TYPE(PayloadListOp, 57, SdfPayloadListOp, 0)

// Crate version 0.9.0
USD_TYPE(TimeCode, 58, SdfTimeCode, 0)

// USD Resolver 2.0
USD_TYPE(Range1d, 59, GfRange1d, 0)
USD_TYPE(Range2d, 60, GfRange2d, 0)
USD_TYPE(Range3d, 61, GfRange3d, 0)

USD_TYPE(Range1f, 62, GfRange1f, 0)
USD_TYPE(Range2f, 63, GfRange2f, 0)
USD_TYPE(Range3f, 64, GfRange3f, 0)

#undef USD_TYPE
