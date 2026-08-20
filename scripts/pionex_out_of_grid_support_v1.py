#!/usr/bin/env python3
"""Shared Phase 4D support for live Pionex states outside the configured grid."""

from __future__ import annotations


def corrected_initial_states(lines, start_price):
    """
    Return interval states for a mature arithmetic grid.

    Below the lower bound: every interval is holding ETH / waiting to sell.
    Above the upper bound: every interval has sold / is waiting to buy on re-entry.
    Inside the band: intervals below the current pivot wait to buy, and intervals
    at/above the pivot wait to sell.

    This corrects the legacy above-upper edge case, which left one synthetic
    sell interval active after the bot had already escaped above the grid.
    """
    if len(lines) < 2:
        return []

    if start_price <= lines[0]:
        return ["sell"] * (len(lines) - 1)

    if start_price >= lines[-1]:
        return ["buy"] * (len(lines) - 1)

    pivot = 0
    for idx, level in enumerate(lines):
        if level <= start_price:
            pivot = idx
        else:
            break

    return ["buy" if idx < pivot else "sell" for idx in range(len(lines) - 1)]
