// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "SceneAttachment.h"
#include <chrono>
#include <cstdint>
#include <exception>
#include <future>
#include <memory>
#include <stdexcept>

namespace omni::usd::live::test {
// The layer's progress result and the ID of the one local intent one host
// authoring job's flush admitted, or ID 0 when that flush admitted none.
struct HostAdmission {
    OmniUsdLiveResult result = OMNI_USD_LIVE_OK;
    uint64_t id = 0;
};
// Run `write` in one host job, then read the admission in a fence job that
// job queues after the flush job its edits queued: the live layer's record
// sink queues at most one flush on each recorded edit, whether the edit
// arrives through SdfLayer or directly on the host backing. `enqueue`
// appends a job to the host queue that also runs the layer's flush and
// publication jobs, and is callable from the host thread. When journal
// application runs as host publication tasks, the fence precedes every
// publication job the flush's submission causes, so the admitted intent is
// still pending when the fence reads it. The preparing() path and
// applyDetached apply on the actor thread and are not used by callers of
// this helper. Any admission count other
// than zero or one fails the caller: its oracle binds one intent per edit.
template <class Enqueue, class Write>
HostAdmission authorAdmittedIntent(Enqueue enqueue, const std::shared_ptr<SceneAttachment>& attachment, Write write) {
    const auto outcome = std::make_shared<std::promise<HostAdmission>>();
    auto admitted = outcome->get_future();
    enqueue([enqueue, write, attachment, outcome]() mutable {
        try {
            const auto before = attachment->snapshot()->localAdmitted;
            write();
            enqueue([attachment, outcome, before] {
                try {
                    HostAdmission admission{attachment->progress().result, 0};
                    const auto snapshot = attachment->snapshot();
                    if (snapshot->localAdmitted != before) {
                        if (snapshot->localAdmitted != before + 1 || snapshot->pending.empty())
                            throw std::runtime_error("one host authoring job admitted other than one intent");
                        admission.id = snapshot->pending.back().id;
                    }
                    outcome->set_value(admission);
                } catch (...) { outcome->set_exception(std::current_exception()); }
            });
        } catch (...) { outcome->set_exception(std::current_exception()); }
    });
    if (admitted.wait_for(std::chrono::seconds(8)) != std::future_status::ready)
        throw std::runtime_error("host authoring job or its fence did not run");
    return admitted.get();
}
} // namespace omni::usd::live::test
