"""
Tests for the abuse guard: a per-IP rate limit plus a global daily cap.

Uses an injected clock so tests run instantly and deterministically -
no real sleeping.
"""
import pytest
from app.guard import RateGuard, GuardConfig


class FakeClock:
    def __init__(self, start=0.0):
        self.t = start

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


class TestPerIpLimit:
    def test_allows_up_to_the_limit(self):
        clock = FakeClock()
        guard = RateGuard(GuardConfig(per_ip_limit=3, per_ip_window_s=60, daily_cap=1000), now=clock)
        for _ in range(3):
            assert guard.check("1.2.3.4").allowed

    def test_blocks_once_the_limit_is_exceeded(self):
        clock = FakeClock()
        guard = RateGuard(GuardConfig(per_ip_limit=3, per_ip_window_s=60, daily_cap=1000), now=clock)
        for _ in range(3):
            guard.check("1.2.3.4")
        result = guard.check("1.2.3.4")
        assert not result.allowed
        assert result.reason == "per_ip_limit"

    def test_different_ips_are_independent(self):
        clock = FakeClock()
        guard = RateGuard(GuardConfig(per_ip_limit=1, per_ip_window_s=60, daily_cap=1000), now=clock)
        assert guard.check("1.1.1.1").allowed
        assert guard.check("2.2.2.2").allowed

    def test_window_slides_and_recovers(self):
        clock = FakeClock()
        guard = RateGuard(GuardConfig(per_ip_limit=2, per_ip_window_s=60, daily_cap=1000), now=clock)
        assert guard.check("1.2.3.4").allowed
        assert guard.check("1.2.3.4").allowed
        assert not guard.check("1.2.3.4").allowed
        clock.advance(61)
        assert guard.check("1.2.3.4").allowed


class TestGlobalDailyCap:
    def test_blocks_once_the_global_cap_is_hit_even_across_many_ips(self):
        clock = FakeClock()
        guard = RateGuard(GuardConfig(per_ip_limit=1000, per_ip_window_s=60, daily_cap=2), now=clock)
        assert guard.check("1.1.1.1").allowed
        assert guard.check("2.2.2.2").allowed
        result = guard.check("3.3.3.3")
        assert not result.allowed
        assert result.reason == "daily_cap"

    def test_daily_cap_resets_after_24_hours(self):
        clock = FakeClock()
        guard = RateGuard(GuardConfig(per_ip_limit=1000, per_ip_window_s=60, daily_cap=1), now=clock)
        assert guard.check("1.1.1.1").allowed
        assert not guard.check("2.2.2.2").allowed
        clock.advance(24 * 3600 + 1)
        assert guard.check("3.3.3.3").allowed


class TestUntrustedForwardedHeaders:
    def test_client_supplied_forwarded_for_is_never_trusted_as_the_key(self):
        # The guard must be keyed on the real transport-layer IP the caller
        # passes in, not on request headers - those are trivially spoofable
        # and would let one attacker rotate through fake identities.
        clock = FakeClock()
        guard = RateGuard(GuardConfig(per_ip_limit=1, per_ip_window_s=60, daily_cap=1000), now=clock)
        assert guard.check("9.9.9.9").allowed
        # Same real IP, pretend header claims differ - guard.check only ever
        # takes one argument, so there is no header to spoof through it.
        assert not guard.check("9.9.9.9").allowed
