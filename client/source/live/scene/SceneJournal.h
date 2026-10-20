// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "WireScene.h"
#include "FieldEditIntent.h"
#include "LiveLayerProgress.h"
#include <functional>
#include <map>
#include <mutex>
#include <tuple>

namespace omni::usd::live {
// A captured-log admission rejection that no wait for pending work can
// relieve, classified where it is raised: a compile limit the log alone
// trips (envelope size, operations, records, work, section IDs), an update
// larger than any admission allows, replay work the intent exceeds with no
// other intent pending, or an exhausted journal ordinal. A flush
// fails the layer with `result`. Every other captured rejection keeps its
// plain result.
struct CapturedRejection {
    OmniUsdLiveResult result;
};
// Conservative graph/data ownership charge without constructing encoded scene
// bytes. Shares the caller's work budget; SDK caches are not an RSS bound.
size_t measureRetainedScene(const WireScene&, const WireSceneLimits&, size_t available, WireSceneWorkBudget& work);
class SceneJournal final {
public:
    struct Limits {
        WireSceneLimits scene;
        size_t maxPending = 4096;
        size_t maxPendingBytes = 256u * 1024 * 1024;
        size_t maxReplayBytes = 512u * 1024 * 1024;
        size_t maxReplayWork = 512u * 1024 * 1024;
        FieldEditLimits authoring; // Its scene limits inherit scene above at admission.
    };
    struct Intent {
        explicit Intent(std::string value, std::shared_ptr<const FieldEditIntent> source = {}, uint64_t ordinal = 0,
            uint64_t revision = 0, uint64_t dependencies = 0)
            : bytes(std::move(value)), authored(std::move(source)),
              retainedBytes(charge(bytes.size(), authored ? authored->retainedBytes() : 0)), localOrdinal(ordinal),
              baseRevision(revision), dependencyCutoff(dependencies) {}
        const std::string bytes; // Complete wire effect; authored retains the original ordered USD log.
        const std::shared_ptr<const FieldEditIntent> authored;
        // The wire bytes plus compiled authoring ownership: what this intent
        // alone retains. No intent pins a scene; replay builds from the
        // journal's authoritative scene and these bytes.
        const size_t retainedBytes;
        // Ordinals count accepted local edits only, independently of actor work
        // IDs and service sequences. The base revision is the journal revision
        // the edit was compiled against, which can include preceding optimistic
        // local edits; a cutoff is provenance, never permission to replay.
        const uint64_t localOrdinal, baseRevision, dependencyCutoff;
    private:
        static size_t charge(size_t bytes, size_t owned) {
            if (owned > SIZE_MAX - bytes) throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
            return bytes + owned;
        }
    };
    struct Pending {
        uint64_t id;
        std::shared_ptr<const Intent> intent;
        // Whether the visible scene's replay applies this intent; false when
        // it predicts the service's rejection of the whole intent.
        bool replayed = true;
        // The work of this intent's step in a replay of every pending intent,
        // measured when the visible scene last applied it: its decoding and
        // its application to the scene before it, continuing from the
        // previous intent's step, so it holds a derivation of paths and
        // child groups only where that replay does one.
        size_t replayWork = 0;
    };
    // How a transaction derived the visible scene (design section 8).
    enum class Derivation {
        None, // No transaction derived it: the journal's first snapshot or an empty commit.
        Authoritative, // No intent is pending, so the visible scene is the authoritative one.
        Replayed, // Every pending intent replayed onto the authoritative scene.
        Extended, // The admitted intent applied to the previous visible scene.
        Retained, // The batch confirmed leading intents only: the previous visible scene.
        Overlaid, // The batch's peer records the pending intents do not hide applied to the previous visible scene.
    };
    struct Snapshot {
        WireScenePtr authoritative, visible;
        uint64_t applied = 0;
        // Monotonic local publication identity, including local admissions
        // that do not advance the authoritative service frontier.
        uint64_t revision = 0;
        std::vector<Pending> pending;
        size_t pendingBytes = 0;
        // Defensive latch for a broken submitter that throws/returns a wrong ID
        // after publication. Admission remains accepted; never resubmit it.
        OmniUsdLiveResult submissionFault = OMNI_USD_LIVE_OK;
        uint64_t initialApplied = 0;
        // Journal retirement means authoritative application, before the
        // actor's separately recorded host-publication retirement boundary.
        uint64_t localAdmitted = 0, localRetired = 0;
        // Logical work the transaction that published this snapshot did, as
        // its work budgets count it: compilation and the new intent's replay
        // for an admission, each update's application and the visible
        // scene's derivation for ordered work. An admission's budget also
        // holds replayWork, which this excludes. Deterministic for a given
        // history; zero for an empty commit.
        size_t work = 0;
        Derivation derivation = Derivation::None;
        // The pending intents' wire bytes and replayWork, summed.
        size_t pendingWireBytes = 0, replayWork = 0;
        // Upper bounds of the field count and metadata bytes of every
        // intermediate scene the replay of the pending intents passes
        // through, after the authoritative scene; zero with none pending.
        size_t replayFields = 0, replayMetadataBytes = 0;
        // Whether the replay's step after the last pending intent derives
        // child groups when it reads them: none is derived after the last
        // pending intent, or it sets WireReplayStep::rederives.
        bool replayRederives = false;
    };
    using SnapshotPtr = std::shared_ptr<const Snapshot>;
    using Submit = std::function<uint64_t(std::string_view, std::shared_ptr<const Intent>, LayerProgress::Publication)>;
    SceneJournal(WireScenePtr initial, uint64_t applied, Limits limits, std::shared_ptr<SectionIdAllocator> ids = {});
    SnapshotPtr snapshot() const;
    // Submitter must invoke publication synchronously exactly once on success,
    // return its ID, and never throw afterward. Bind to LayerActor::submit.
    // Failure before publication preserves the old snapshot and accepts no edit.
    // The submitter runs under journal guards and must not reenter the journal.
    uint64_t submitLocal(std::string_view bytes, const Submit& submitter);
    // Compile a host-captured edit log against the explicit base it was authored
    // on and admit it as one local intent. There is no BUSY on
    // a changed base: the edits are already applied in the host layer, so the
    // intent is replayed onto the current scene like any pending intent.
    // Records addressing identities the log removes are dropped; scratch Create
    // commands and structural child-membership overrides reject before admission.
    // An empty log admits nothing and returns 0. A log whose compiled envelope
    // is empty, or whose every record is a field or sample record addressing a
    // section a committed peer delete removed from the visible scene (the
    // service would store it as invisible orphan data), admits and issues
    // nothing and returns 0 without checking the pending-intent limit; a
    // non-null `applied` receives the verified scene. On admission a non-null
    // `applied` receives base with this intent applied: the host's content.
    // A compiled update over the service's entry limits (authoring.entries)
    // throws DeltaEntryLimitExceeded, a definitive rejection, and admits nothing.
    // A log without a capture (null data) is a prefix of the host buffer,
    // verified against its replay only (compileFieldEdits).
    uint64_t submitCaptured(const WireScenePtr& base, const std::shared_ptr<const SdfOperationLog>& sealed,
        const Submit& submitter, WireScenePtr* applied = nullptr);
    // Compiles a host-captured log as submitCaptured does and checks every
    // limit an admission of it alone would fail definitively, without
    // admitting it, allocating section identities or changing the journal.
    // Returns the compiled update's size, zero when it would admit nothing;
    // throws as submitCaptured does for a limit.
    size_t probeCaptured(const WireScenePtr& base, const std::shared_ptr<const SdfOperationLog>& sealed) const;
    // Bind to the actor apply callback. Confirmed local work with the exact
    // retained ID/owner/bytes retires its intent, after the committed local
    // writes that retire with it (absorbedLocal), and without applying its
    // bytes when a net-empty catch-up passed it (netted); peer coverage that
    // absorbed local work retires exactly those leading intents; no other
    // peer work retires one. Peer work a net-empty catch-up passed (netted)
    // applies no bytes. Call the actor's
    // applied bookkeeping only after this returns successfully.
    // Return the exact committed owner, rather than sampling a potentially
    // newer local admission after releasing the transaction guard.
    SnapshotPtr apply(const LayerProgress::Work&, std::string_view bytes, std::shared_ptr<const void> localIntent = {});
    // One ordered work, its delivered or confirmed bytes and, for confirmed
    // local work, its retained intent.
    struct Application {
        const LayerProgress::Work& work;
        std::string_view bytes;
        std::shared_ptr<const void> localIntent;
    };
    // Applies consecutive ordered work as one transaction, with the same
    // checks and effects as applying each alone in order: each work after the
    // first starts above the previous one's last position, and the positions
    // between them are empty commits. Pending intents replay once, onto the
    // last authoritative scene, and one snapshot publishes. A failing work
    // publishes nothing.
    SnapshotPtr apply(const std::vector<Application>&);
    // Advances applied through empty commits the layer's progress applied
    // without work; every position at or below `through` holds applied work.
    // Callers advance through work.first - 1 before applying work, which the
    // actor hands over only at its applied position plus one.
    void applyEmpty(uint64_t through) noexcept;
    // The configured limits with the work budgets that scale with a scene
    // grown with its size: replay work and authoring work each gain room for
    // 32 retained charges of the scene. Byte quotas stay as configured.
    Limits effectiveLimits(const WireScene& scene) const;
    // The configured limits. Of the authoring limits, only the work budget
    // grows with the scene (effectiveLimits).
    const Limits& limits() const noexcept { return limits_; }
private:
    // The target of a field or sample record: a section's field, or the
    // section's time samples for a sample record, which names no field.
    struct Footprint {
        uint64_t section;
        std::string field;
        bool sample;
    };
    struct FootprintView {
        uint64_t section;
        std::string_view field;
        bool sample;
    };
    struct FootprintOrder {
        using is_transparent = void;
        template<class A, class B> bool operator()(const A& a, const B& b) const {
            return std::tie(a.section, a.sample, a.field) < std::tie(b.section, b.sample, b.field);
        }
    };
    using FootprintIndex = std::map<Footprint, size_t, FootprintOrder>;
    // The distinct record targets of one intent's wire bytes.
    std::vector<Footprint> footprintOf(std::string_view bytes) const;
    // The index of the pending intents a visible scene applies.
    FootprintIndex index(const std::vector<Pending>&) const;
    static void retire(FootprintIndex&, const Footprint* begin, const Footprint* end) noexcept;
    static void retire(FootprintIndex& index, const std::vector<Footprint>& keys) noexcept {
        retire(index, keys.data(), keys.data() + keys.size());
    }
    // Derives the prepared visible scene by applying the peer updates that
    // open the batch to the previous visible scene, when the result equals
    // replaying the pending intents onto the batch's authoritative scene.
    // False, with `prepared` unchanged, when it may not.
    bool overlay(const Snapshot& previous, Snapshot& prepared, const std::vector<std::string_view>& peers,
        const WireSceneLimits& limits, WireSceneWorkBudget& work) const;
    uint64_t submit(std::string_view, const Submit&, const WireScenePtr& explicitBase,
        const std::shared_ptr<const SdfOperationLog>& captured, WireScenePtr* applied = nullptr);
    const Limits limits_;
    const std::shared_ptr<SectionIdAllocator> ids_;
    mutable std::mutex mutex_;
    std::mutex transactionMutex_;
    SnapshotPtr state_;
    // For each record target, how many pending intents of the published
    // snapshot that its visible scene applies hold a record of it. Guarded
    // by transactionMutex_ and changed only with the publication it follows.
    FootprintIndex footprint_;
};
} // namespace omni::usd::live
