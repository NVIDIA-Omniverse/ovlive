// SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#pragma managed(push, off)
#pragma warning(push)
#pragma warning(disable : 4244 4305)
#include <pxr/base/gf/range1d.h>
#include <pxr/base/gf/range1f.h>
#include <pxr/base/gf/range2d.h>
#include <pxr/base/gf/range2f.h>
#include <pxr/base/gf/range3d.h>
#include <pxr/base/gf/range3f.h>
#include <pxr/usd/sdf/changeList.h>
#include <pxr/usd/sdf/childrenPolicies.h>
#include <pxr/usd/sdf/childrenUtils.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/sdf/layerOffset.h>
#include <pxr/usd/sdf/layerStateDelegate.h>
#include <pxr/usd/sdf/notice.h>
#include <pxr/usd/sdf/payload.h>
#include <pxr/usd/sdf/reference.h>
#include <pxr/usd/sdf/schema.h>
#include <pxr/usd/sdf/spec.h>
#include <tbb/concurrent_hash_map.h>
#pragma warning(pop)
#pragma managed(pop)

PXR_NAMESPACE_USING_DIRECTIVE;
