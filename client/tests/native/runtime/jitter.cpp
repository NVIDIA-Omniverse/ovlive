// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveJitter.h"
#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace omni::usd::live;
namespace {
using Clock = JitterEstimator::Clock;
using Time = Clock::time_point;
using Us = std::chrono::microseconds;
using Ms = std::chrono::milliseconds;
void checkJitter(bool condition,const char* message) { if (!condition) throw std::runtime_error(message); }
OmniUsdLiveJitterParams params(uint32_t constant = 10,uint32_t multiple = 2,uint32_t maximum = 1000) {
    return {sizeof(OmniUsdLiveJitterParams),OMNI_USD_LIVE_ABI_VERSION,constant,multiple,maximum};
}
Time time(int64_t us) { return Time(std::chrono::duration_cast<Clock::duration>(Us(us))); }
void fallbackAndConfig() {
    JitterEstimator estimator;
    const auto received = time(1000000);
    const auto defaults = params();
    checkJitter(estimator.due(900000,received,received,defaults) == received+Ms(10),"unsampled default fallback is not 10ms");
    checkJitter(estimator.due(0,received,received,defaults) == received+Ms(10),"zero server timestamp changed fallback");
    estimator.sample(99999,0,received);
    checkJitter(estimator.due(900000,received,received,defaults) == received+Ms(10),"zero-server sample initialized estimator");
    checkJitter(estimator.due(900000,received,received,params(90,UINT32_MAX,7)) == received+Ms(7),"fallback ignored maximum");
    checkJitter(estimator.due(UINT64_MAX,received,received+Ms(2),params(99,99,0)) == received+Ms(2),"zero maximum did not disable holdback");
    checkJitter(estimator.due(900000,received,received,params(0,0,1000)) == received,"zero constant fallback was delayed");
    // Changing settings recomputes from this update's original receipt, even
    // when configuration changes after its previous deadline has elapsed.
    checkJitter(estimator.due(0,received,received+Ms(2),params(50,0,20)) == received+Ms(20),"fallback configuration extended receipt-relative cap");
    checkJitter(estimator.due(0,received,received+Ms(19),params(50,0,20)) == received+Ms(20),"fallback reanchored delay to now");
    checkJitter(estimator.due(0,received,received+Ms(21),params(50,0,20)) == received+Ms(21),"expired update received a fresh fallback delay");
    const auto maximum = params(UINT32_MAX,UINT32_MAX,UINT32_MAX);
    const auto exact = received+Us(uint64_t(UINT32_MAX)*1000);
    checkJitter(estimator.due(0,received,received,maximum) == exact,"uint32 maximum milliseconds overflowed conversion");
}
void measuredMapping() {
    JitterEstimator estimator;
    const auto received = time(1000000);
    // A 5000us RTT supplies its 2500us one-way estimate. At receipt the
    // server is estimated at 1,002,500us; default holdback is 15,000us.
    estimator.sample(2500,1000000,received);
    checkJitter(estimator.due(1000000,received,received,params()) == received+Us(12500),"half-RTT server clock mapping differs");
    checkJitter(estimator.due(1000001,received,received,params()) == received+Us(12501),"wire microseconds were rounded to milliseconds");
    checkJitter(estimator.due(1002500,received,received,params()) == received+Us(15000),"adaptive holdback differs at estimated current server time");
    checkJitter(estimator.due(1000000,received,received+Us(5000),params()) == received+Us(12500),"stable measured clock extended an existing deadline");
    checkJitter(estimator.due(1000000,received,received+Us(12500),params()) == received+Us(12500),"measured update did not become due");
    checkJitter(estimator.due(1000000,received,received+Us(20000),params()) == received+Us(20000),"already-due measured update acquired a new delay");
    checkJitter(estimator.due(1000000,received,received,params(0,0,1000)) == received,"zero adaptive holdback delayed an old update");
    checkJitter(estimator.due(UINT64_MAX,received,received+Ms(2),params(10,2,3)) == received+Ms(3),"future server timestamp bypassed receipt-relative maximum");
    checkJitter(estimator.due(UINT64_MAX,received,received+Ms(3),params(10,2,3)) == received+Ms(3),"maximum did not expire");
    checkJitter(estimator.due(1000000,received,received,params(0,1,1000)) == received,"dynamic multiplier was not recomputed");
    checkJitter(estimator.due(0,received,received,params()) == received+Ms(10),"missing update timestamp did not use constant fallback");
}
void gradualClockCorrection() {
    const auto first = time(1000000), second = first+Ms(100), observed = second+Ms(100);
    for (bool ahead : {false,true}) {
        JitterEstimator estimator;
        estimator.sample(1000,1000000,first);
        // At the second sample the old clock predicts1,101,000. The new
        // estimate differs by5000us, but only1000us correction is allowed over
        // the following100ms, in addition to ordinary elapsed time.
        estimator.sample(1000,ahead ? 1105000 : 1095000,second);
        const uint64_t expected = ahead ? 1202000 : 1200000;
        checkJitter(estimator.due(expected,observed,observed,params(0,0,1000)) == observed,
                    "clock correction moved beyond its bounded slope");
        checkJitter(estimator.due(expected+1,observed,observed,params(0,0,1000)) == observed+Us(1),
                    "clock correction lost microsecond mapping or jumped to target");
    }
}
void meansAndRing() {
    const auto received = time(2000000);
    constexpr uint64_t epoch = 2000000, settleUs = 100000000;
    const auto settled = received+Us(settleUs);
    auto add = [&](JitterEstimator& estimator,uint64_t latency) {
        // Every sample implies the same server time at receipt. Clock mapping
        // cannot mask the latency statistic under test.
        estimator.sample(latency,epoch-latency,received);
    };
    auto inferred = [&](const JitterEstimator& estimator) {
        return estimator.due(epoch+settleUs,settled,settled,params(0,1,1000));
    };
    JitterEstimator small;
    add(small,1000); checkJitter(inferred(small) == settled+Us(1000),"one-sample latency differs");
    add(small,3000); checkJitter(inferred(small) == settled+Us(2000),"two-sample average differs");
    checkJitter(small.due(epoch+99,received+Us(99),received+Us(99),params(0,1,1000)) == received+Us(1099),
                "latency target moved before one correction microsecond accrued");
    checkJitter(small.due(epoch+100,received+Us(100),received+Us(100),params(0,1,1000)) == received+Us(1101),
                "latency correction did not preserve100us/1us slope");
    add(small,5000); checkJitter(inferred(small) == settled+Us(3000),"three-sample average differs");
    add(small,7000); checkJitter(inferred(small) == settled+Us(4000),"four-sample average differs");
    JitterEstimator rounded;
    add(rounded,1000); add(rounded,1001);
    checkJitter(inferred(rounded) == settled+Us(1000),"integer mean rounding differs");

    JitterEstimator outlier;
    for (unsigned i = 0; i < 29; ++i) add(outlier,1000);
    add(outlier,1000000);
    checkJitter(inferred(outlier) == settled+Us(1000),"isolated large latency outlier inflated settled holdback");
    JitterEstimator ring;
    add(ring,10000); // The oldest sample is deliberately not first after sorting.
    for (unsigned i = 0; i < 29; ++i) add(ring,1000);
    checkJitter(inferred(ring) == settled+Us(1000),"ring initial outlier was retained");
    for (unsigned i = 0; i < 7; ++i) add(ring,10000);
    // Seven high samples remain after evicting the original high sample. At
    //30 samples the upper quartile is still in the23 low observations.
    checkJitter(inferred(ring) == settled+Us(1000),"sorting corrupted FIFO sample eviction");
    add(ring,10000);
    checkJitter(inferred(ring) == settled+Us(10000),"eighth high sample did not change upper-quartile cluster");
    for (unsigned i = 0; i < 30; ++i) add(ring,2000);
    checkJitter(inferred(ring) == settled+Us(2000),"old observations survived complete 30-sample ring eviction");
}
void resetAndOldSamples() {
    JitterEstimator estimator;
    const auto received = time(2000000);
    estimator.sample(2500,1000000,received);
    const auto expected = estimator.due(1000001,received,received,params());
    estimator.sample(UINT64_MAX,UINT64_MAX,received-Us(1));
    estimator.sample(UINT64_MAX,0,received+Ms(100));
    checkJitter(estimator.due(1000001,received,received,params()) == expected,"backward/zero-server sample changed clock or latency history");
    estimator.reset();
    checkJitter(estimator.due(UINT64_MAX,received,received,params()) == received+Ms(10),"reset retained previous measured clock");
    // New generation may start at a smaller server and monotonic timestamp.
    const auto earlier = received-Ms(100);
    estimator.sample(3,100,earlier);
    checkJitter(estimator.due(100,earlier,earlier,params()) == earlier+Us(10003),"reset retained previous sample history or backward-receipt fence");
    estimator.reset(); estimator.reset();
    checkJitter(estimator.due(100,earlier,earlier,params(4,9,8)) == earlier+Ms(4),"repeated reset changed fallback");
}
void saturationAndClockExtremes() {
    const auto received = time(1000000);
    JitterEstimator addition;
    addition.sample(200,UINT64_MAX-100,received);
    checkJitter(addition.due(500,received,received,params(0,0,1000)) == received,"server-plus-one-way estimate wrapped uint64");
    JitterEstimator multiplication;
    multiplication.sample(UINT64_C(1)<<63,100,received);
    checkJitter(multiplication.due((UINT64_C(1)<<63)+100,received,received,params(0,2,1000)) == received+Ms(1000),
                "latency multiplier wrapped instead of saturating");
    JitterEstimator combined;
    const uint64_t constant = uint64_t(UINT32_MAX)*1000;
    const uint64_t latency = UINT64_MAX-constant+1;
    combined.sample(latency,1,received);
    checkJitter(combined.due(latency+1,received,received,params(UINT32_MAX,1,1000)) == received+Ms(1000),
                "constant-plus-adaptive delay wrapped uint64");
    JitterEstimator mean;
    // These three observations sum to2^64. Correct averaging has a very large
    // target; after100ms its allowed1% movement is1000us. Wrapped sum would0.
    mean.sample(0,UINT64_MAX,received);
    mean.sample(UINT64_MAX,1,received);
    mean.sample(1,UINT64_MAX-1,received);
    const auto later = received+Ms(100);
    checkJitter(mean.due(UINT64_MAX,later,later,params(0,1,1000)) == later+Us(1000),"mean summation wrapped uint64");
    JitterEstimator extreme;
    const auto nearMaximum = Clock::time_point::max()-Us(500);
    checkJitter(extreme.due(0,nearMaximum,nearMaximum,params()) == Clock::time_point::max(),"receipt deadline overflowed monotonic maximum");
    checkJitter(extreme.due(0,Clock::time_point::max(),Clock::time_point::max(),params()) == Clock::time_point::max(),"clock maximum was rescheduled");
    const auto nearMinimum = Clock::time_point::min()+Ms(1);
    checkJitter(extreme.due(0,nearMinimum,nearMinimum,params()) == nearMinimum+Ms(10),"negative clock epoch changed fallback arithmetic");
    extreme.sample(10,1000,Clock::time_point::min());
    const auto farFuture = Clock::time_point::max()-Ms(2000);
    // This query spans almost the entire signed native clock domain. UBSan
    // catches subtraction before conversion even if a wrapped result is capped.
    checkJitter(extreme.due(UINT64_MAX,farFuture,farFuture,params()) == farFuture+Ms(1000),"extreme elapsed clock span broke maximum cap");
    checkJitter(extreme.due(1,farFuture,farFuture,params()) == farFuture,"extreme elapsed clock span stranded an old update");
}
void decisionObservations() {
    JitterEstimator estimator;
    const auto received = time(1000000);
    auto decision = estimator.decide(1000000, received, received, params());
    checkJitter(!decision.sampled && decision.targetHoldUs == 10000 && !decision.sampleAgeUs,
                "cold-start decision invented a sample");
    estimator.sample(2500, 1000000, received);
    decision = estimator.decide(1000000, received, received + Ms(5), params());
    checkJitter(decision.sampled && decision.targetHoldUs == 15000 && decision.sampleAgeUs == 5000 &&
                decision.due == received + Us(12500), "sample age/target differs from scheduling decision");
    decision = estimator.decide(0, received, received + Ms(5), params(9, 99, 7));
    checkJitter(!decision.sampled && !decision.sampleAgeUs && decision.targetHoldUs == 7000,
                "missing timestamp did not report capped constant fallback");
    decision = estimator.decide(UINT64_MAX, received, received, params(99, 99, 0));
    checkJitter(!decision.sampled && !decision.targetHoldUs && decision.due == received,
                "disabled policy reported an active target");
    estimator.reset();
    decision = estimator.decide(1000000, received, received, params());
    checkJitter(!decision.sampled && !decision.sampleAgeUs, "reset retained diagnostic sample provenance");
}
void observedWaitIntervals() {
    JitterWait wait;
    const auto zero = time(0);
    checkJitter(!wait.finish(zero), "inactive finish manufactured an episode");
    wait.begin(7, zero, zero + Ms(10));
    checkJitter(wait.active() && wait.work() == 7, "head identity not retained");
    wait.observe(zero + Ms(2), zero + Ms(10));
    wait.observe(zero + Ms(4), zero + Ms(10));
    auto measured = wait.finish(zero + Ms(13));
    checkJitter(measured && measured->holdMs == 10 && measured->postDueMs == 3 && !wait.active(),
                "overdue observation was mislabeled as jitter holdback");
    checkJitter(!wait.finish(zero + Ms(20)), "repeated completion duplicated timing");

    // Configuration shortens/zeros at the next observation, never retroactively.
    wait.begin(8, zero, zero + Ms(10));
    wait.observe(zero + Ms(6), zero + Ms(2));
    measured = wait.finish(zero + Ms(6));
    checkJitter(measured->holdMs == 6 && !measured->postDueMs, "shortening rewrote past observed hold");
    wait.begin(8, zero, zero + Ms(10));
    wait.observe(zero + Ms(3), zero + Ms(3));
    measured = wait.finish(zero + Ms(3));
    checkJitter(measured->holdMs == 3 && !measured->postDueMs, "zero policy lost previous observed hold");

    wait.begin(9, zero, zero + Ms(5));
    wait.observe(zero + Ms(3), zero + Ms(10));
    measured = wait.finish(zero + Ms(12));
    checkJitter(measured->holdMs == 10 && measured->postDueMs == 2, "extension mixed hold with late observation");
    wait.begin(9, zero, zero + Ms(5));
    wait.observe(zero + Ms(8), zero + Ms(10));
    measured = wait.finish(zero + Ms(12));
    checkJitter(measured->holdMs == 7 && measured->postDueMs == 5, "late extension retroactively filled an overdue gap");

    // Same retained work may outlive an estimator/read generation. Tracking
    // consumes only the actual checked/due pair and never consults an estimator.
    wait.begin(10, zero, zero + Ms(10));
    wait.observe(zero + Ms(4), zero + Ms(6));
    checkJitter(wait.work() == 10, "recheck changed retained work identity");
    measured = wait.finish(zero + Ms(7));
    checkJitter(measured->holdMs == 6 && measured->postDueMs == 1, "frozen eligibility was not respected");
    // Host-slot preemption or observed failure/close ends this episode. A
    // later wait of the same head starts fresh, excluding intervening host work.
    wait.begin(10, zero, zero + Ms(10));
    measured = wait.finish(zero + Ms(2));
    checkJitter(measured->holdMs == 2 && !measured->postDueMs, "interrupted hold duration differs");
    wait.begin(10, zero + Ms(8), zero + Ms(10));
    measured = wait.finish(zero + Ms(11));
    checkJitter(measured->holdMs == 2 && measured->postDueMs == 1, "resumed episode included the host-slot gap");

    wait.begin(11, zero, zero + Ms(2));
    for (int i = 1; i <= 15; ++i) wait.observe(zero + Us(i * 100), zero + Ms(2));
    measured = wait.finish(zero + Us(2700));
    checkJitter(measured->holdMs == 2 && measured->postDueMs == 0, "per-check rounding erased fractional holds");
    wait.begin(12, Clock::time_point::min(), Clock::time_point::max() - Ms(3));
    measured = wait.finish(Clock::time_point::max());
    checkJitter(measured->holdMs > UINT64_C(1000000000) && measured->postDueMs == 3,
                "extreme native tick difference overflowed");
}
} // namespace

void jitterEstimatorTests() {
    fallbackAndConfig(); measuredMapping(); gradualClockCorrection(); meansAndRing(); resetAndOldSamples(); saturationAndClockExtremes();
    decisionObservations(); observedWaitIntervals();
    std::cout << "jitter estimator numerical mapping, ring eviction, receipt caps and saturation passed" << std::endl;
}
