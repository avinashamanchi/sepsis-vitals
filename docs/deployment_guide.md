# Deployment Guide

## Target

Raspberry Pi 4 or equivalent low-cost board running an offline local service.

## V1 Architecture

- Python Flask server.
- Local model artifact.
- CSV/tablet/manual vitals input.
- Browser-based nurse console.
- Local audit log.
- USB or low-bandwidth model update path.

## Offline Assumptions

- No internet required after setup.
- Power may be intermittent.
- Device must recover after reboot.
- All clinical data remains local unless explicitly exported under the data-use
  agreement.

## Hardware Fallbacks

If Raspberry Pi supply is constrained, test:

- Orange Pi.
- Used mini PC.
- Android tablet with local web runtime.

## Maintenance

- Review model drift every 6 months.
- Re-check calibration after site workflow changes.
- Record false positives and missed clinical concern in the audit log.
